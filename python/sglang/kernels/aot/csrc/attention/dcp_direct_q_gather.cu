// Direct symmetric-memory DCP query gather — sglang port of vLLM #50484
// (csrc/libtorch_stable/attention/dcp_utils/dcp_direct_q_gather.cu).
//
// Each rank multicast-writes its local query head-slice into every consumer's
// final query buffer via NVLS `multimem.st`, then publishes a release-scope
// epoch signal (also multicast) and spin-waits for all peers. After the gather
// every rank's local `final_query` holds the fully-gathered query
// [T, world_size * H_per_rank, D] — no NCCL AllGather needed.
//
// Requires SM90+ and an NVSwitch fabric (multicast). The Python workspace gates
// on `symm_mem.multicast_ptr != 0` and falls back to NCCL otherwise.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <cstdio>

#include "utils.h"

namespace sglang::direct_dcp_qgather {

constexpr uint64_t kSpinLimit = 100000000;

// Advance the invocation ID; its low bit selects one of two staging slots.
__global__ void increment_epoch_kernel(int64_t* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    epoch[0] += 1;
  }
}

// Replicate one 16-byte payload to every symmetric-buffer replica (NVLS
// multicast). SM90+ only; trap on older arch so the launch is obviously
// broken rather than silently degrading.
__device__ __forceinline__ void multimem_store_16(uint4* mc_ptr, uint4 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(mc_ptr), "r"(value.x), "r"(value.y), "r"(value.z),
                 "r"(value.w)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

// Publish prior system-scope writes and signal every replica.
__device__ __forceinline__ void multimem_store_release_system(uint32_t* mc_ptr,
                                                              uint32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.release.sys.global.u32 [%0], %1;"
               :
               : "l"(mc_ptr), "r"(value)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void store_release_system(uint32_t* ptr,
                                                     uint32_t value) {
  uint64_t address = reinterpret_cast<uint64_t>(ptr);
  asm volatile("st.global.release.sys.u32 [%0], %1;"
               :
               : "l"(address), "r"(value)
               : "memory");
}

__device__ __forceinline__ uint32_t load_acquire_system(const uint32_t* ptr) {
  uint32_t value;
  uint64_t address = reinterpret_cast<uint64_t>(ptr);
  asm volatile("ld.global.acquire.sys.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ bool wait_for_epoch(const uint32_t* ptr,
                                               uint32_t epoch) {
  for (uint64_t spins = 0; spins < kSpinLimit; ++spins) {
    if (load_acquire_system(ptr) == epoch) {
      return true;
    }
  }
  return false;
}

inline void check_launch(const char* operation) {
  cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              std::string(operation) + " kernel launch failed: " +
                  cudaGetErrorString(error));
}

// Multicast each rank's head slice directly into every consumer's final query
// buffer. Reuse is ordered by the downstream DCP output synchronization.
__global__ void direct_dcp_q_gather_multimem_kernel(
    const uint4* local_query, uint4* mc_final_query, uint32_t* mc_signal,
    const uint32_t* received_signal, int64_t* epoch_ptr, uint32_t* completion,
    int64_t world_size, int64_t rank, int64_t num_tokens,
    int64_t bytes_per_token, int64_t query_token_stride_bytes,
    int64_t destination_token_stride_bytes) {
  // The common one-token decode case uses one block. Fold the epoch update
  // into that publication kernel to avoid an otherwise separate tiny launch.
  if (gridDim.x == 1 && threadIdx.x == 0) {
    epoch_ptr[0] += 1;
  }
  __syncthreads();
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t signal_slot = static_cast<int64_t>(epoch & 1u);

  int64_t items_per_token = bytes_per_token / sizeof(uint4);
  int64_t source_token_stride = query_token_stride_bytes / sizeof(uint4);
  int64_t destination_token_stride =
      destination_token_stride_bytes / sizeof(uint4);
  int64_t destination_head_offset = rank * items_per_token;
  for (int64_t token_idx = blockIdx.x; token_idx < num_tokens;
       token_idx += gridDim.x) {
    for (int64_t token_item = threadIdx.x; token_item < items_per_token;
         token_item += blockDim.x) {
      multimem_store_16(
          mc_final_query + token_idx * destination_token_stride +
              destination_head_offset + token_item,
          local_query[token_idx * source_token_stride + token_item]);
    }
  }

  // Publish all multicast writes before incrementing completion.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }

  if (gridDim.x > 1) {
    uint32_t completed = atomicAdd(completion, 1u);
    if (completed + 1u != gridDim.x) {
      return;
    }
    atomicExch(completion, 0u);
  }

  multimem_store_release_system(mc_signal + signal_slot * world_size + rank,
                                epoch);

  for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
    int64_t signal_item = signal_slot * world_size + source_rank;
    if (!wait_for_epoch(received_signal + signal_item, epoch)) {
      printf("direct DCP q-gather multimem timeout source=%lld epoch=%u\n",
             static_cast<long long>(source_rank), epoch);
      asm volatile("trap;");
    }
  }
}

}  // namespace sglang::direct_dcp_qgather

void direct_dcp_q_gather(
    const at::Tensor& local_query, at::Tensor& final_query,
    at::Tensor& received_signal, at::Tensor& completion, at::Tensor& epoch,
    int64_t world_size, int64_t rank, int64_t max_num_tokens,
    int64_t padded_num_heads, int64_t query_mc_ptr, int64_t signal_mc_ptr) {
  using sglang::direct_dcp_qgather::check_launch;
  using sglang::direct_dcp_qgather::direct_dcp_q_gather_multimem_kernel;
  using sglang::direct_dcp_qgather::increment_epoch_kernel;

  TORCH_CHECK(local_query.is_cuda(), "local query must be a CUDA tensor");
  at::ScalarType dtype = local_query.scalar_type();
  TORCH_CHECK(local_query.dim() == 3, "local query must have shape [T,H,D]");
  TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");

  int64_t num_tokens = local_query.size(0);
  int64_t heads_per_rank = local_query.size(1);
  int64_t head_dim = local_query.size(2);
  int64_t gathered_num_heads = world_size * heads_per_rank;
  int64_t element_size = local_query.element_size();
  TORCH_CHECK(num_tokens > 0 && num_tokens <= max_num_tokens,
              "token count exceeds symmetric q-gather buffer capacity");
  TORCH_CHECK(heads_per_rank > 0 && head_dim > 0,
              "query head dimensions must be positive");
  TORCH_CHECK(padded_num_heads >= gathered_num_heads,
              "padded query heads must cover all gathered heads");
  TORCH_CHECK(local_query.stride(2) == 1 &&
                  local_query.stride(1) == head_dim &&
                  local_query.stride(0) >= heads_per_rank * head_dim,
              "local query must have packed heads");

  TORCH_CHECK(
      final_query.is_cuda() && final_query.scalar_type() == dtype &&
          final_query.is_contiguous() && final_query.dim() == 3 &&
          final_query.size(0) == num_tokens &&
          final_query.size(1) == gathered_num_heads &&
          final_query.size(2) == head_dim,
      "final query must be contiguous with shape [T,world_size*H,D]");
  TORCH_CHECK(
      received_signal.is_cuda() && received_signal.is_contiguous() &&
          received_signal.scalar_type() == at::ScalarType::Int &&
          received_signal.dim() == 2 && received_signal.size(0) == 2 &&
          received_signal.size(1) == world_size,
      "received signal has the wrong symmetric buffer layout");
  TORCH_CHECK(completion.is_cuda() && completion.is_contiguous() &&
                  completion.scalar_type() == at::ScalarType::Int &&
                  completion.numel() == 1,
              "completion counter must be one CUDA int32 tensor");
  TORCH_CHECK(epoch.is_cuda() && epoch.is_contiguous() &&
                  epoch.scalar_type() == at::ScalarType::Long &&
                  epoch.numel() == 1,
              "epoch must be a one-element CUDA int64 tensor");

  const at::cuda::OptionalCUDAGuard device_guard(local_query.device());
  int64_t device_index = local_query.device().index();
  TORCH_CHECK(
      final_query.device().index() == device_index &&
          received_signal.device().index() == device_index &&
          completion.device().index() == device_index &&
          epoch.device().index() == device_index,
      "direct DCP q-gather tensors must be on the same CUDA device");

  int64_t query_token_stride_bytes = local_query.stride(0) * element_size;
  int64_t bytes_per_token = heads_per_rank * head_dim * element_size;
  int64_t gathered_token_stride_bytes =
      gathered_num_heads * head_dim * element_size;
  bool vectorized =
      reinterpret_cast<uintptr_t>(local_query.data_ptr()) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(final_query.data_ptr()) % alignof(uint4) == 0 &&
      query_token_stride_bytes % sizeof(uint4) == 0 &&
      bytes_per_token % sizeof(uint4) == 0;
  TORCH_CHECK(vectorized,
              "direct DCP q-gather requires 16-byte-aligned pointers and strides");
  TORCH_CHECK(query_mc_ptr != 0 && signal_mc_ptr != 0,
              "direct DCP q-gather requires multicast pointers");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  int64_t blocks = num_tokens < world_size ? num_tokens : world_size;
  if (blocks > 1) {
    increment_epoch_kernel<<<1, 1, 0, stream>>>(
        epoch.data_ptr<int64_t>());
    check_launch("direct DCP q-gather epoch");
  }

  direct_dcp_q_gather_multimem_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(local_query.data_ptr()),
      reinterpret_cast<uint4*>(static_cast<uintptr_t>(query_mc_ptr)),
      reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(signal_mc_ptr)),
      reinterpret_cast<const uint32_t*>(received_signal.data_ptr<int32_t>()),
      epoch.data_ptr<int64_t>(),
      reinterpret_cast<uint32_t*>(completion.data_ptr<int32_t>()),
      world_size, rank, num_tokens, bytes_per_token, query_token_stride_bytes,
      gathered_token_stride_bytes);
  check_launch("direct DCP q-gather");
}
