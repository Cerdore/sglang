# NCCL EP — Optional MoE Dispatch/Combine Backend for sglang

**Status:** Design / investigation report — PR-split proposal
**Source requirement:** Google Doc "NCCL EP — Add NCCL EP as an optional MoE dispatch/combine backend"
**Reference upstream:** [NVIDIA/nccl `contrib/nccl_ep` @ v2.30.7-1](https://github.com/NVIDIA/nccl/tree/v2.30.7-1/contrib/nccl_ep)
**Date:** 2026-07-23

---

## 1. What NCCL EP is, and how it compares to DeepEP

NCCL EP is NVIDIA's first-party **Expert-Parallelism communication primitive** layered on NCCL's Device API (Load-Store Accessible + GPU-Initiated Networking). It does the same job as DeepEP — asymmetric all-to-all **dispatch** (tokens → assigned experts) and **combine** (expert outputs → weighted reduction back to source ranks) — but it is shipped inside NCCL itself and exposed to Python via **`nccl4py`** (`from nccl.ep import NCCLLibrary`).

Two algorithms mirror DeepEP's two modes almost exactly:

| NCCL EP algorithm | DeepEP analogue | Target workload |
|---|---|---|
| `NCCL_EP_ALGO_LOW_LATENCY` (LL) | `Buffer.low_latency_dispatch/combine` | Decode / latency-sensitive inference |
| `NCCL_EP_ALGO_HIGH_THROUGHPUT` (HT) | `Buffer.dispatch/combine` (get_dispatch_layout) | Prefill / training, large batches |

Key API surface (from `nccl_ep.h`):

- `ncclEpCreateGroup(ep_group, ncclComm_t comm, &group_config)` — collective; binds to an **existing NCCL communicator** (not a fresh one).
- `ncclEpCreateHandle / ncclEpUpdateHandle` — caches per-topk routing (`topk_idx`) + layout info (`expert_counters`, `src_rank_counters`).
- `ncclEpDispatch / ncclEpCombine` — the collective data movement. `send_only` (LL) splits into transfer + `ncclEpComplete()` for comp/comm overlap.
- `NCCL_EP_AUTO` sentinel auto-sizes `rdma_buffer_size`, `num_qp_per_rank`, `num_channels`, `max_num_sms`, `max_dispatch_tokens_per_rank`.
- Layouts: HT → `FLAT` / `EXPERT_MAJOR`; LL → `EXPERT_MAJOR` / `RANK_MAJOR`.
- LL fault-tolerance / zero-token-rank mask APIs: `ncclEpMaskQuery / ncclEpMaskUpdate / ncclEpMaskClean / ncclEpGetAsyncError / ncclEpErrorClear`.
- `NCCL_EP_AUTO` buffer sizing is **not** CUDA-graph-capturable across a reallocation; explicit `rdma_buffer_size > 0` is required for graph capture (matches DeepEP's static-buffer requirement).

**Prerequisites:** CUDA 13+, NCCL ≥ 2.29 with Device API/GIN, Hopper (H100) or Blackwell. `pip install nccl4py[cu13]` (only cu13 today).

---

## 2. sglang's existing MoE EP architecture (evidence-based)

The integration shape is already established by DeepEP/Mooncake/Nixl/MoriEP. A new backend is a known plugin pattern.

### 2.1 Backend interface contract

`python/sglang/srt/layers/moe/token_dispatcher/base.py:280` — `BaseDispatcher` ABC:
```python
class BaseDispatcher(ABC):
    def dispatch(self, hidden_states, topk_output: TopKOutput) -> DispatchOutput: ...
    def combine(self, combine_input: CombineInput) -> torch.Tensor: ...
    def set_quant_config(self, quant_config: dict): ...
    # optional two-phase + overlap hooks for CUDA-graph / SBO:
    def dispatch_a/dispatch_b(self, ...); combine_a/combine_b(self, ...); ...
    def set_overlap_args(...)/clear_overlap_args(): ...
    def register_pre/post_dispatch_hook(...); register_pre/post_combine_hook(...)
```

Output/input formats are tagged with `DispatchOutputFormat` / `CombineInputFormat` enums (`base.py:169-268`): `STANDARD, DEEPEP_NORMAL, DEEPEP_LL, FLASHINFER, ASCEND_TP`. The MoE runner branches on the format.

### 2.2 DeepEP reference implementation (the template to mirror)

`python/sglang/srt/layers/moe/token_dispatcher/deepep.py`:
- `DeepEPBuffer` (line 161) — process-wide buffer manager, state on `ctx.resources.buffers["deepep_ep_state"]`. `get_deepep_buffer(group, hidden_size, param_bytes, deepep_mode, num_max_dispatch_tokens_per_rank, num_experts)` computes NVL/RDMA sizes via `Buffer.get_dispatch_config / get_combine_config / get_low_latency_rdma_size_hint`, then `Buffer(group, num_nvl_bytes, num_rdma_bytes, low_latency_mode=, num_qps_per_rank=, allow_mnnvl=, use_fabric=)`.
  - **Critical:** `group` is a `dist.ProcessGroup`; the C++ side converts it to `ncclComm_t`. NCCL EP needs the same `ncclComm_t` — obtainable from `PyNcclCommunicator.comm` (`pynccl.py:60`, `ncclComm_t` exposed as `self.comm`).
- `_DeepEPDispatcherImplNormal` (HT, line 495) uses `buffer.get_dispatch_layout(...)` → `buffer.dispatch(...)` → `buffer.combine(...)`.
- `_DeepEPDispatcherImplLowLatency` (LL, line 655) uses `buffer.low_latency_dispatch(...)` → `buffer.low_latency_combine(...)`.
- `DeepEPDispatcher` (line 863) is the facade with `dispatch_a/dispatch_b/combine_a/combine_b` two-phase API + `_Stage` state machine (line 856).
- `DeepEPMode` (`utils.py:169`): `NORMAL / LOW_LATENCY / AUTO`; `AUTO.resolve(is_extend_in_batch)` → NORMAL for prefill, LL for decode.

### 2.3 Backend selection & wiring

- `MoeA2ABackend` enum (`utils.py:28`): `NONE, DEEPEP, MOONCAKE, NIXL, MORI, ASCEND_FUSEEP, ASCEND_TP, FLASHINFER, MEGAMOE, CUSTOMIZED`.
- `create_moe_dispatcher(...)` (`fused_moe_triton/layer.py:108`) constructs the dispatcher; wrapped in `MaybeTboDeepEPDispatcher` for TBO.
- `get_moe_impl_class(...)` (`ep_moe/layer.py:279`) returns `DeepEPMoE` for EP backends, else `FusedMoE`; `DeepEPMoE.forward_impl` (`ep_moe/layer.py:177`) = `dispatch → run_moe_core → combine`.
- Server args (`server_args.py`): `--moe-a2a-backend` (line 2191), `--moe-runner-backend` (2210), `--deepep-mode` (2224), `--deepep-config` (2280), `--deepep-dispatcher-output-dtype` (2234), `--enable-eplb` (2252), `--enable-waterfill` (2309), `--enable-two-batch-overlap` / `--enable-single-batch-overlap` (2752/2755).
- Runtime capability detection & fallback: `initialize_moe_config(server_args)` (`utils.py:288`); server_args post-processing forces `none` on NPU, disables CUDA graph for `deepep_mode=normal` (line 6296), resolves MORI `auto→normal` (6348).

### 2.4 CUDA Graph, EPLB, shared experts, zero-token ranks, SBO/TBO, deterministic

- **CUDA Graph:** `DeepEPCudaGraphRunnerAdapter` (`runner_utils/deepep_adapter.py:25`) records/replays dispatch mode; used at `decode_cuda_graph_runner.py:991/1070`. Two-phase dispatch (`dispatch_a/b`) lets the wait sit at a graph boundary. Static buffers required; `num_max_dispatch_tokens_per_rank` asserted ≤1024 (`deepep.py:381`). `SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE` inserts a `barrier()` before dispatch (`deepep.py:80-86`).
- **EPLB:** `eplb/expert_distribution.py:61` `ExpertDistributionRecorder` with `on_deepep_dispatch_normal/low_latency` hooks (lines 98-111). Backends call these from dispatch (`deepep.py:592, 706`). Enabled via `--enable-eplb`; `--ep-dispatch-algorithm static/dynamic/fake/lp`, `--init-expert-location`.
- **Shared experts:** per-rank fused shared slots for DeepEP-class backends (`utils.py:386-393`): `num_shared_slots = num_fused_shared_experts * moe_ep_size`. Standard path maps shared IDs into tail of `local_expert_mapping` (`standard.py:191-200`). Waterfill (`waterfill.py:69`, `--enable-waterfill`) dispatches fused shared expert as extra routed slot; supports DeepEP + MegaMOE.
- **Zero-token ranks:** DeepEP encodes via layout (`num_tokens_per_rank` may be 0, `is_token_in_rank` mask). MoriEP allocates empty `(0, ...)` tensors (`moriep.py:574-604`). FlashInfer uses `invalid_token_expert_id` sentinel (`flashinfer.py:102, 258-263`). Nixl/Mooncake use `active_ranks` mask from `ElasticEPStateManager` (`nixl.py:233`, `mooncake.py:239,274`).
- **SBO / TBO:** `batch_overlap/two_batch_overlap.py` (`MaybeTboDeepEPDispatcher` holds 2 inner dispatchers; `--enable-two-batch-overlap`). `batch_overlap/single_batch_overlap.py` (`--enable-single-batch-overlap`); DeepEP LL combine consumes `CombineOverlapArgs` (`deepep.py:784-826`); disabled on SM90 (`utils.py:306-310`).
- **Deterministic:** `--enable-deterministic-inference` (`server_args.py:3141`) selects deterministic Triton configs; forbids non-`none` A2A in some resolvers (`server_args.py:4132`). `DeepEPMoE.deprecate_flag` falls back to `FusedMoE` Triton path for unquantized BF16+deep_gemm+LL combos (`ep_moe/layer.py:86-127`).

---

## 3. Integration design

### 3.1 New backend enum + server args

- Add `NCCL_EP` to `MoeA2ABackend` (`utils.py:28`).
- Add `nccl_ep` choice to `--moe-a2a-backend` (`server_args.py:2191`).
- Add `--nccl-ep-mode {auto,low_latency,high_throughput}` (default `auto`) mirroring `--deepep-mode`.
- Add `--nccl-ep-config` (JSON) mirroring `--deepep-config` for group/handle tuning (`rdma_buffer_size`, `num_qp_per_rank`, `num_channels`, `max_num_sms`).
- Add `--nccl-ep-dispatcher-output-dtype {auto,bf16,fp8}` (NCCL EP does **not** support fp16→fp8 type conversion today; offer bf16/fp8 only).

### 3.2 Capability detection (runtime + packaging)

- Import guard mirroring `use_deepep` (`deepep.py:50-62`):
  ```python
  try:
      from nccl.ep import NCCLLibrary, NCCL_EP_ALGO_LOW_LATENCY, NCCL_EP_ALGO_HIGH_THROUGHPUT
      _nccl_ep_available = True
  except ImportError:
      _nccl_ep_available = False
  ```
- Runtime gating in `initialize_moe_config`: if `moe_a2a_backend == "nccl_ep"` and not `_nccl_ep_available` → fall back to `none` with a warning (and to `deepep` if available), matching the existing NPU/MORI fallback pattern. Require CUDA 13 + NCCL ≥ 2.29 + Hopper/Blackwell; gate out on HIP/NPU and on SM < 90.
- Packaging: document `nccl4py[cu13]` as an **optional extra** in `pyproject.toml` (do not hard-depend — same posture as `deep_ep`). Add an optional extra `[project.optional-dependencies] nccl-ep = ["nccl4py[cu13]>=..."]`.

### 3.3 Buffer manager — `NcclEpBuffer`

New `python/sglang/srt/layers/moe/token_dispatcher/nccl_ep.py`, structured exactly like `DeepEPBuffer`:
- State on `ctx.resources.buffers["nccl_ep_state"]` (per-process).
- `get_nccl_ep_buffer(group, hidden_size, num_experts, mode, num_max_dispatch_tokens_per_rank)`:
  1. Obtain `ncclComm_t` from the EP group's PyNccl communicator (resolve via the same path DeepEP uses to get `group`, then `PyNcclCommunicator.comm`).
  2. Build `ncclEpGroupConfig_t` with `algorithm`, `num_experts`, `max_dispatch_tokens_per_rank` (≤1024 for graph safety), `max_token_bytes = hidden_size * dtype_bytes`, `rdma_buffer_size` (explicit > 0 for graph capture; use `NCCL_EP_AUTO` only when graph is off), `num_qp_per_rank / num_channels / max_num_sms = NCCL_EP_AUTO`.
  3. `ncclEpCreateGroup(...)` once (collective, cached).
  4. Pre-allocate dispatch/combine recv buffers sized from `max_recv_tokens_per_rank` per the resolved layout.
- `set_dispatch_mode_*` / `clean_buffer` mirroring `DeepEPBuffer` for mode switching and LL buffer cleanup.
- Handle `SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE`-style `barrier()` parity if precompile stages are used.

### 3.4 Dispatcher — `NcclEpDispatcher`

New `BaseDispatcher` subclass (file above), following the DeepEP two-phase template:
- **LL impl** (`_NcclEpDispatcherImplLowLatency`):
  - `dispatch_a`: build handle via `ncclEpCreateHandle(layout=EXPERT_MAJOR or RANK_MAJOR, topk_idx, layout_info)`; `ncclEpDispatch(..., send_only=True)` to overlap; outputs preallocated in buffer manager.
  - `dispatch_b`: `ncclEpComplete(...)` to reap recv.
  - `combine_a`: `ncclEpCombine(..., send_only=True)`; `combine_b`: `ncclEpComplete(...)`. Consume `CombineOverlapArgs` for SBO parity with DeepEP LL (`deepep.py:784-826`).
- **HT impl** (`_NcclEpDispatcherImplNormal`):
  - `ncclEpCreateHandle(layout=FLAT/EXPERT_MAJOR)` per forward (or `ncclEpUpdateHandle` for cached routing); `ncclEpDispatch` → expert compute → `ncclEpCombine`. `pass_direction = forward` (decode/prefill); backward path for training deferred (HT prefill scope first; training backward is a follow-up).
- Output format: add `NCCL_EP_LL` and `NCCL_EP_HT` to `DispatchOutputFormat` / `CombineInputFormat` (`base.py:169-268`) + `DispatchOutputChecker`/`CombineInputChecker` guards. `run_moe_core` branches on these (the received layout differs from DeepEP's, so the MoE kernel launch must reshape accordingly — expert-major vs rank-major indexing).
- Wire into `create_moe_dispatcher` (`fused_moe_triton/layer.py:108`) and `get_moe_impl_class` (`ep_moe/layer.py:279`) → reuse `DeepEPMoE` if the recv layout is convertible, else add an `NcclEpMoE` subclass.
- Mode resolution: add `NcclEpMode.AUTO.resolve(is_extend_in_batch)` → HT for prefill, LL for decode, identical semantics to `DeepEPMode`.
- EPLB: call `get_global_expert_distribution_recorder().on_*` from dispatch (`deepep.py:592,706` pattern) — add `on_nccl_ep_dispatch_*` hooks or reuse DeepEP ones if the counters are equivalent.

### 3.5 CUDA Graph adapter

- Reuse the `DeepEPCudaGraphRunnerAdapter` pattern (`runner_utils/deepep_adapter.py`): add `NcclEpCudaGraphRunnerAdapter` (or generalize the existing one) recording/replaying the resolved mode; wire at `decode_cuda_graph_runner.py:991/1070`.
- **Hard requirement:** explicit `rdma_buffer_size > 0` (no `NCCL_EP_AUTO` reallocation inside capture); static recv buffers; `max_dispatch_tokens_per_rank` fixed and ≤1024. `ncclEpCreateGroup`/`ncclEpInitHandle` with auto-sizing must run **outside** capture (group creation at init, not in the graph).

### 3.6 EPLB, shared experts, zero-token ranks, SBO/TBO, deterministic

- **EPLB:** recorder hooks in dispatch (above); compatible with `--enable-eplb` / `--ep-dispatch-algorithm`. Verify `expert_counters`/`src_rank_counters` from `ncclEpLayoutInfo_t` feed the same stats as DeepEP's `num_tokens_per_expert`.
- **Shared experts:** adopt the per-rank fused-shared-slot convention (`num_fused_shared_experts * moe_ep_size`, `utils.py:386-393`); ensure handle `topk_idx` includes shared-expert IDs mapped to the local-expert tail (as `standard.py:191-200` does). Waterfill support: add NCCL EP to the `WaterfillBalancer` allowlist (`waterfill.py`) alongside DeepEP/MegaMOE.
- **Zero-token ranks:** use LL `enable_mask` + `ncclEpMaskQuery/Update/Clean` APIs (this is NCCL EP's first-class equivalent of DeepEP's zero-rank layout / Nixl's `active_ranks`). Integrate with `ElasticEPStateManager` masks if elastic EP is active. Allocate empty `(0, ...)` recv tensors as MoriEP does (`moriep.py:574-604`) to keep kernel ABI happy. Add `ncclEpGetAsyncError`/`ErrorClear` to the timeout/health path (mirrors `SGLANG_DEEPEP_PRECOMPILE` barrier pattern).
- **SBO/TBO:** `MaybeTboDeepEPDispatcher` (`two_batch_overlap.py`) already keys off `is_tbo_enabled()` and DeepEP-class backends — add NCCL EP to its backend allowlist so 2 inner NCCL-EP dispatchers/buffers overlap. For SBO, implement `set_overlap_args`/`CombineOverlapArgs` consumption in the LL combine (DeepEP LL template at `deepep.py:784-826`); gate out on SM90 (`utils.py:306-310`).
- **Deterministic fallback:** gate NCCL EP out under `--enable-deterministic-inference` (extend the `server_args.py:4132` check); on quant/config incompatibility fall back to `FusedMoE` Triton path via a `deprecate_flag`-style guard (`ep_moe/layer.py:86-127` pattern). Keep `MoeA2ABackend.NONE` + standard all-gather/reduce-scatter as the ultimate fallback.

### 3.7 Correctness parity

- Add a parity test harness (mirror `test/...` DeepEP parity tests): for the same `(hidden, topk_ids, topk_weights, num_experts, ep_size)`, assert `NcclEpDispatcher.combine(dispatch(...))` ≈ `StandardDispatcher` and ≈ `DeepEPDispatcher` within fp tolerance across `ep_size ∈ {2,4,8}`, decode + prefill shapes, zero-token-rank cases, and bf16/fp8 dispatch dtypes.
- NCCL EP fp16→fp8 type-conversion is unsupported today → restrict fp8 dispatch to models that quantize before dispatch (DeepEP parity) and document the gap.

---

## 4. PR breakdown (matches the source doc's 4 PRs)

> Order is intentional: each PR is independently reviewable and mergeable; later PRs build on earlier ones.

### PR 1 — Packaging + runtime capability detection *(owner: TBD)*
**Goal:** Make `nccl4py` an optional dependency and add the `nccl_ep` backend selector + runtime gating — no functional dispatch yet.

Scope:
1. Add `NCCL_EP` to `MoeA2ABackend` (`utils.py:28`) + `nccl_ep` choice to `--moe-a2a-backend` (`server_args.py:2191`).
2. Add import guard `_nccl_ep_available` (mirror `use_deepep`) + CUDA/NCCL/SM capability check.
3. `initialize_moe_config`: if `nccl_ep` selected but unavailable → warn + fall back to `none` (then `deepep`).
4. `pyproject.toml`: optional `[nccl-ep]` extra (`nccl4py[cu13]`); do not add a hard dep.
5. Skeleton `NcclEpDispatcher(BaseDispatcher)` raising `NotImplementedError` for `dispatch/combine`, wired into `create_moe_dispatcher` and `get_moe_impl_class`.
6. `create_moe_dispatcher` raises `NotImplementedError` for `NCCL_EP` until PR2.
7. Tests: capability-detection unit tests (import-present/absent, SM gating, fallback path).

Exit criteria: `--moe-a2a-backend nccl_ep` is accepted, capability is detected, and missing `nccl4py` degrades gracefully. No data movement.

### PR 2 — LL path: decode correctness parity *(owner: TBD)*
**Goal:** Functional low-latency dispatch/combine for decode, numerically equal to standard + DeepEP.

Scope:
1. `NcclEpBuffer.get_nccl_ep_buffer(...)` with explicit `rdma_buffer_size > 0`, `ncclEpCreateGroup` from EP group's `ncclComm_t`.
2. `_NcclEpDispatcherImplLowLatency` with two-phase `dispatch_a/b`, `combine_a/b` + `ncclEpComplete` (send_only overlap).
3. Add `NCCL_EP_LL` to `DispatchOutputFormat`/`CombineInputFormat` + checkers; `run_moe_core` reshape for LL expert-major / rank-major layout.
4. `NcclEpMode.AUTO` → LL for decode.
5. Zero-token-rank handling via `enable_mask` + empty-tensor allocation; `ncclEpGetAsyncError` health path.
6. EPLB recorder hooks in LL dispatch.
7. Shared-expert per-rank slot mapping.
8. Tests: correctness parity vs Standard + DeepEP across `ep_size ∈ {2,4,8}`, decode shapes, zero-token-rank, bf16 + fp8 dispatch; `SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE`-style barrier.

Exit criteria: decode E2E matches DeepEP within tolerance on H100/Blackwell, `ep_size ≥ 2`.

### PR 3 — HT path: prefill (and training-style) workloads *(owner: TBD)*
**Goal:** High-throughput dispatch/combine for prefill; forward pass only first, backward as follow-up.

Scope:
1. `_NcclEpDispatcherImplNormal` using `ncclEpCreateHandle/UpdateHandle` (FLAT / EXPERT_MAJOR), `ncclEpDispatch` → compute → `ncclEpCombine`, `pass_direction=forward`.
2. `NCCL_EP_HT` format tags + `run_moe_core` reshape for HT flat layout (matches DeepEP normal recv shape).
3. `NcclEpMode.AUTO` → HT for prefill (`is_extend_in_batch`).
4. `--nccl-ep-config` group/handle tuning plumbing.
5. fp8 dispatch via `scales` tensor (`ncclEpDispatchInputs_t.scales`); document fp16→fp8 conversion-unsupported gap.
6. Backward pass (HT combine with `topk_weights`, `pass_direction=backward`) — implement if training support is in scope, else stub + follow-up issue.
7. Tests: prefill correctness parity vs DeepEP normal; backward parity if implemented.

Exit criteria: prefill E2E matches DeepEP normal; `--nccl-ep-mode high_throughput` works standalone.

### PR 4 — CUDA Graph + EPLB + shared experts + zero-token ranks + SBO/TBO + deterministic fallback *(owner: TBD)*
**Goal:** Production hardening — graph capture, load balancing, overlap, fault tolerance, deterministic fallback.

Scope:
1. `NcclEpCudaGraphRunnerAdapter` (or generalize `DeepEPCudaGraphRunnerAdapter`); wire at `decode_cuda_graph_runner.py:991/1070`. Enforce explicit `rdma_buffer_size` + static buffers + ≤1024 tokens; group/handle creation outside capture.
2. EPLB: full `ExpertDistributionRecorder` integration (`on_nccl_ep_dispatch_*`); verify `ncclEpLayoutInfo_t.expert_counters` parity with DeepEP stats; `--enable-eplb` + `--ep-dispatch-algorithm` end-to-end.
3. Shared experts + Waterfill: per-rank fused slots; add NCCL EP to `WaterfillBalancer` allowlist; `--enable-waterfill` E2E.
4. Zero-token ranks: `enable_mask` + `ncclEpMaskQuery/Update/Clean` integrated with `ElasticEPStateManager`; empty-tensor ABI; `ncclEpGetAsyncError`/`ErrorClear` health/timeout path.
5. SBO/TBO: add NCCL EP to `MaybeTboDeepEPDispatcher` + `single_batch_overlap` allowlists; implement `CombineOverlapArgs` consumption in LL combine (DeepEP LL template); SM90 gate.
6. Deterministic fallback: gate out under `--enable-deterministic-inference` (extend `server_args.py:4132`); `deprecate_flag`-style fallback to `FusedMoE` Triton path on config incompatibility; keep `MoeA2ABackend.NONE` as ultimate fallback.
7. Tests: CUDA-graph capture/replay correctness; EPLB rebalance E2E; waterfill E2E; SBO + TBO overlap correctness; zero-token-rank fault injection; deterministic-mode fallback.

Exit criteria: full feature parity with DeepEP backend's production feature set; NCCL EP usable as a drop-in `--moe-a2a-backend nccl_ep` replacement for DeepEP on Hopper/Blackwell.

---

## 5. Open questions / risks

1. **`ncclComm_t` acquisition:** DeepEP's `Buffer(group, ...)` wraps the torch `ProcessGroup`. NCCL EP's `ncclEpCreateGroup` needs a raw `ncclComm_t`. Confirm whether `nccl.ep`'s `NCCLLibrary` accepts a `ncclComm_t` obtained from sglang's `PyNcclCommunicator.comm`, or whether it requires a comm created via `nccl4py` itself. This is the single biggest integration unknown — **probe first** in PR1 (spike: create a group, run a no-op handle, tear down).
2. **nccl4py Python API surface:** The C API is documented; the Python `nccl.ep` bindings' exact signatures (tensor structs, handle lifecycle, async/stream args) need to be confirmed against the installed `nccl4py` package. The design assumes the Python API mirrors the C API 1:1 — verify before PR2.
3. **CUDA 13 only:** `nccl4py[cu13]` only; sglang's cu12 DeepEP build (`is_cu12` path, `deepep.py:271`) cannot use NCCL EP. Gate to cu13; on cu12 fall back to DeepEP.
4. **fp16→fp8 type conversion unsupported** by NCCL EP today → fp8 dispatch parity with DeepEP requires pre-quantized dispatch input (as sglang already does for DeepEP). Document and constrain.
5. **Training backward:** NCCL EP HT backward needs `topk_weights` through both dispatch and combine. If training is a stated goal, PR3 must include it; if inference-only, defer to a follow-up.
6. **Layout → MoE kernel reshape:** NCCL EP's EXPERT_MAJOR/RANK_MAJOR (LL) and FLAT/EXPERT_MAJOR (HT) receive layouts differ from DeepEP's. `run_moe_core` must reshape/index correctly; risk of subtle indexing bugs → the parity harness (PR2) is the guard.
7. **MNNVL / fabric:** DeepEP has `use_fabric` for GB200/GB300. NCCL EP's inter-node path uses `NCCL_GIN_TYPE=3` (GDAKI) — confirm whether `nccl4py` exposes the equivalent, and whether multi-node needs MPI launch.
