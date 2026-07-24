"""NCCL EP MoE dispatch/combine backend (low-latency path).

Wraps the ``nccl4py`` / ``nccl.ep`` library (NVIDIA NCCL EP ``contrib/nccl_ep``)
as an optional MoE A2A backend selected by ``--moe-a2a-backend nccl_ep``.

Design (see ``.omc/plans/nccl-ep-pr1-ll-merged-plan.md`` and the spike summary at
``docs/specs/nccl-ep-spikes/summary.md``):

* **LL only** (decode). The HT (prefill) path is a follow-up.
* **bf16 dispatch + Python-side post-quant** is the *only* viable fp8 route.
  NCCL EP LL dispatch forces bf16 input (``nccl_ep.cc`` asserts
  ``x->datatype == ncclBfloat16``). The fp8 wire path exists but is officially
  NOT SUPPORTED (``RELEASE.md`` "No FP8 support") and un-QA'd; we do not rely on
  it. DeepEP fuses bf16->fp8 into its dispatch kernel; NCCL EP does not, so we
  call ``sglang_per_token_group_quant_fp8(group_size=128)`` on the bf16 recv to
  produce the ``(fp8, scale)`` that ``apply_deepep_ll`` consumes. This is a pure
  pointwise function — mathematically identical to DeepEP's in-kernel quant, so
  downstream GEMM sees the same input (only network bandwidth differs: NCCL EP
  carries bf16, ~2x token bytes).
* Emits ``DEEPEP_LL``-format dispatch output, reusing ``DeepEPMoE.run_moe_core`` +
  ``W4AFp8MoEMethod.apply_deepep_ll`` unchanged (Q5 spike PASS).
"""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    get_nccl_ep_mode,
    get_nccl_ep_num_max_dispatch_tokens_per_rank,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.layers.moe.fused_moe_triton.layer import MoeRunnerConfig

logger = logging.getLogger(__name__)

# NCCL EP LL dispatch only accepts hidden sizes in this hardcoded allowlist
# (device/macros.cuh:161-171 SWITCH_HIDDEN) and requires hidden % 128 == 0
# (nccl_ep.cc:2450). The %512 check is on the (unused) fp8-wire path only.
_NCCL_EP_LL_SUPPORTED_HIDDEN = {2048, 2560, 4096, 5120, 6144, 7168, 8192}

# LL top-k upper bound (low_latency.cu:1022, kNumMaxTopK). Models with topk>9
# (e.g. Nemotron Super) fall back to DeepEP LL — upstream issue #2103.
_NCCL_EP_LL_MAX_TOPK = 9

# Mirror DeepEP's per-rank dispatch budget cap (deepep.py:381).
_NCCL_EP_MAX_DISPATCH_TOKENS_PER_RANK_CAP = 1024
_NCCL_EP_DEFAULT_MAX_DISPATCH_TOKENS_PER_RANK = 1024

# fp8 per-token-group quant block size for the LL GEMM path
# (cutlass_w4a8_moe_deepep_ll / deep_gemm.py:768-769 hardcode 128).
_NCCL_EP_FP8_GROUP_SIZE = 128


def is_nccl_ep_available() -> bool:
    """Whether nccl4py + NCCL >= 2.29 + Hopper/Blackwell are usable here.

    Conservative: importability is the gating signal (the nccl4py wheel already
    requires CUDA 13 and bundles NCCL). We additionally gate on arch below.
    """
    if importlib.util.find_spec("nccl.ep") is None:
        return False
    if not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_capability(0)[0]
    if cc < 9:  # Hopper (sm_90) or Blackwell.
        return False
    return True


def nccl_ep_capability_reason() -> str:
    """Human-readable reason why NCCL EP is unavailable (for fallback logs)."""
    if importlib.util.find_spec("nccl.ep") is None:
        return "nccl4py (nccl.ep) not importable"
    if not torch.cuda.is_available():
        return "no CUDA device"
    cc = torch.cuda.get_device_capability(0)[0]
    if cc < 9:
        return f"GPU arch sm_{cc}x not supported (need Hopper/Blackwell sm_90+)"
    return "unknown"


# ----------------------------- Buffer (Group/Handle lifecycle) -----------------------------

_nccl_ep_runtime = None  # cached import of nccl.ep / nccl.core


def _load_nccl_ep():
    global _nccl_ep_runtime
    if _nccl_ep_runtime is not None:
        return _nccl_ep_runtime
    import nccl.core as nccl_core
    import nccl.ep as nccl_ep

    _nccl_ep_runtime = (nccl_core, nccl_ep)
    return _nccl_ep_runtime


class NcclEpBuffer:
    """Process-wide NCCL EP group + static recv buffers (mirrors DeepEPBuffer).

    State lives on ``ctx.resources.buffers["nccl_ep_state"]`` (one per process).
    The NCCL EP group is created once from the sglang EP group's ``ncclComm_t``
    (Q1 spike PASS: ``ep.Group.create`` accepts an external communicator).
    """

    @classmethod
    def _state(cls):
        from types import SimpleNamespace

        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        state = buffers.get("nccl_ep_state")
        if state is None:
            state = SimpleNamespace(
                group=None,
                num_experts=None,
                num_local_experts=None,
                hidden_size=None,
                max_dispatch_tokens_per_rank=None,
                max_recv_tokens_per_rank=None,
            )
            buffers["nccl_ep_state"] = state
        return state

    @classmethod
    def get_buffer(
        cls,
        ep_group: "GroupCoordinator",
        hidden_size: int,
        num_experts: int,
        num_local_experts: int,
        max_dispatch_tokens_per_rank: int,
    ) -> "NcclEpBuffer":
        state = cls._state()
        if state.group is None:
            state.num_experts = num_experts
            state.num_local_experts = num_local_experts
            state.hidden_size = hidden_size
            state.max_dispatch_tokens_per_rank = max_dispatch_tokens_per_rank
            # LL auto = nRanks * max_dispatch_tokens_per_rank (nccl_ep.h:200).
            world_size = ep_group.world_size
            state.max_recv_tokens_per_rank = world_size * max_dispatch_tokens_per_rank
            cls._create_group(state, ep_group)
        return state

    @classmethod
    def _create_group(cls, state, ep_group: "GroupCoordinator"):
        nccl_core, nccl_ep = _load_nccl_ep()

        pynccl = ep_group.pynccl_comm
        if pynccl is None or not getattr(pynccl, "available", False):
            raise RuntimeError(
                "NCCL EP requires a live PyNccl communicator on the EP group "
                "(pynccl_comm unavailable)."
            )
        comm_ptr = pynccl.comm.value  # ncclComm_t (c_void_p) -> int
        wrapped_comm = nccl_core.Communicator(ptr=comm_ptr)

        # Explicit rdma_buffer_size keeps create_handle local (no collective, no
        # realloc) — required for CUDA graph later and cheap per-step handles.
        rdma_buffer_size = cls._low_latency_rdma_size_hint(
            nccl_ep, state, ep_group.world_size
        )
        cfg = nccl_ep.GroupConfig(
            algorithm=nccl_ep.Algorithm.LOW_LATENCY,
            num_experts=state.num_experts,
            max_dispatch_tokens_per_rank=state.max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank=0,  # 0 = auto = nRanks * max_dispatch
            max_token_bytes=state.hidden_size * 2,  # bf16 = 2 bytes/elem
            rdma_buffer_size=rdma_buffer_size,
        )
        state.group = nccl_ep.Group.create(wrapped_comm, cfg)
        logger.info(
            "NCCL EP group created: num_experts=%d world_size=%d hidden=%d "
            "max_dispatch_tokens_per_rank=%d rdma_buffer_size=%d",
            state.num_experts,
            ep_group.world_size,
            state.hidden_size,
            state.max_dispatch_tokens_per_rank,
            rdma_buffer_size,
        )

    @staticmethod
    def _low_latency_rdma_size_hint(nccl_ep, state, world_size: int) -> int:
        """Worst-case RDMA buffer size for LL (explicit so handle init is local).

        nccl4py may expose a native hint; fall back to a conservative upper bound
        based on the max recv footprint (E_local * max_recv * H * 2 bytes bf16)
        doubled for send+recv slots.
        """
        hint_fn = getattr(nccl_ep, "get_low_latency_rdma_size_hint", None)
        if callable(hint_fn):
            try:
                return int(
                    hint_fn(
                        state.max_dispatch_tokens_per_rank,
                        state.hidden_size,
                        world_size,
                        state.num_experts,
                    )
                )
            except Exception:
                pass  # signature mismatch; fall through to the bound below
        e_local = state.num_local_experts
        max_recv = state.max_recv_tokens_per_rank
        per_slot = e_local * max_recv * state.hidden_size * 2  # bf16 bytes
        # send buffer + recv buffer + counters; round up generously.
        return int(per_slot * 4)

    @classmethod
    def destroy(cls):
        state = cls._state()
        g = state.group
        if g is not None:
            try:
                g.destroy()
            except Exception:
                pass
            state.group = None


# ----------------------------- Dispatcher (LL path) -----------------------------


class NcclEpDispatcher(BaseDispatcher):
    """NCCL EP low-latency token dispatcher.

    Mirrors the DeepEP LL contract: ``dispatch`` returns a
    ``DeepEPLLDispatchOutput`` (DEEPEP_LL format), reusing the DeepEP compute
    path. NCCL EP carries bf16 on the wire; we quantize bf16->fp8 (group 128)
    after dispatch to feed ``apply_deepep_ll``.
    """

    def __init__(self, moe_runner_config: "MoeRunnerConfig", ep_group: "GroupCoordinator"):
        super().__init__()
        nccl_core, nccl_ep = _load_nccl_ep()
        self._nccl_ep = nccl_ep

        self.ep_group = ep_group
        self.router_topk = moe_runner_config.top_k
        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.hidden_size = moe_runner_config.hidden_size
        self.params_dtype = moe_runner_config.params_dtype
        self.world_size = ep_group.world_size

        # Resolve dispatch mode. This PR implements LOW_LATENCY only; HT raises
        # NotImplementedError at resolve() time.
        mode = get_nccl_ep_mode().resolve(is_extend_in_batch=False)
        assert mode.is_low_latency(), "NCCL EP: only low_latency mode is implemented"
        self.mode = mode

        # Per-rank dispatch budget (Group-level fixed, review #4).
        budget = get_nccl_ep_num_max_dispatch_tokens_per_rank()
        if budget <= 0:
            budget = _NCCL_EP_DEFAULT_MAX_DISPATCH_TOKENS_PER_RANK
        assert (
            budget <= _NCCL_EP_MAX_DISPATCH_TOKENS_PER_RANK_CAP
        ), f"NCCL EP max_dispatch_tokens_per_rank {budget} exceeds cap {_NCCL_EP_MAX_DISPATCH_TOKENS_PER_RANK_CAP}"
        self.num_max_dispatch_tokens_per_rank = budget

        # Validate LL hard constraints early (fail fast at init, not at first dispatch).
        if self.hidden_size not in _NCCL_EP_LL_SUPPORTED_HIDDEN:
            raise ValueError(
                f"NCCL EP LL only supports hidden in {sorted(_NCCL_EP_LL_SUPPORTED_HIDDEN)} "
                f"(got {self.hidden_size}); use --moe-a2a-backend deepep for this model."
            )
        if self.router_topk > _NCCL_EP_LL_MAX_TOPK:
            raise ValueError(
                f"NCCL EP LL supports topk <= {_NCCL_EP_LL_MAX_TOPK} (got {self.router_topk}); "
                f"fall back to DeepEP LL (upstream nccl_ep issue #2103)."
            )

        self.buffer = NcclEpBuffer.get_buffer(
            ep_group,
            self.hidden_size,
            self.num_experts,
            self.num_local_experts,
            self.num_max_dispatch_tokens_per_rank,
        )

        self.handle = None
        self.stream = torch.cuda.Stream()

        # Lazily imported; fp8 quant is only needed when the downstream GEMM
        # consumes fp8 (w4afp8). For bf16-only models there is no LL compute
        # path (run_moe_core asserts False), so w4afp8 is the supported scope.
        self._quant_fp8 = None

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config

    # ---- two-phase hooks (not used in this PR; SBO/TBO is out of scope) ----
    def dispatch_a(self, hidden_states, topk_output):
        raise NotImplementedError("NCCL EP two-phase dispatch is not implemented yet")

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError("NCCL EP two-phase dispatch is not implemented yet")

    def combine_a(self, *args, **kwargs):
        raise NotImplementedError("NCCL EP two-phase combine is not implemented yet")

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError("NCCL EP two-phase combine is not implemented yet")

    # ---- single-shot dispatch/combine (PR1 path) ----
    def dispatch(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        nccl_ep = self._nccl_ep
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids.to(torch.int64)

        t = hidden_states.shape[0]
        if t > self.num_max_dispatch_tokens_per_rank:
            raise ValueError(
                f"NCCL EP: decode batch ({t}) exceeds per-rank dispatch budget "
                f"{self.num_max_dispatch_tokens_per_rank} (nccl_ep.cc:3088 assert); "
                f"increase SGLANG_NCCL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK-equivalent "
                f"server arg or reduce batch."
            )

        state = self.buffer
        E_local = self.num_local_experts
        H = self.hidden_size
        max_recv = state.max_recv_tokens_per_rank

        with torch.cuda.stream(self.stream):
            handle = state.group.create_handle(
                layout=nccl_ep.Layout.EXPERT_MAJOR,
                topk_idx=nccl_ep.Tensor(topk_ids),
                config=nccl_ep.HandleConfig(),
                stream=self.stream.cuda_stream,
            )
            self.handle = handle

            recv_tokens = torch.zeros(
                (E_local, max_recv, H),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            expert_counters = torch.zeros(
                (E_local,), dtype=torch.int32, device=hidden_states.device
            )
            expert_offsets = torch.zeros(
                (E_local + 1,), dtype=torch.int32, device=hidden_states.device
            )
            recv_total = torch.zeros(
                (1,), dtype=torch.int32, device=hidden_states.device
            )

            inputs = nccl_ep.DispatchInputs(tokens=nccl_ep.Tensor(hidden_states))
            outputs = nccl_ep.DispatchOutputs(tokens=nccl_ep.Tensor(recv_tokens))
            layout_info = nccl_ep.LayoutInfo(
                expert_counters=nccl_ep.Tensor(expert_counters),
                expert_offsets=nccl_ep.Tensor(expert_offsets),
                recv_total_counter=nccl_ep.Tensor(recv_total),
            )
            handle.dispatch(
                inputs,
                outputs,
                layout_info=layout_info,
                config=nccl_ep.DispatchConfig(),
                stream=self.stream.cuda_stream,
            )
        self.stream.synchronize()

        # bf16 -> fp8_e4m3 + per-token-group scale (block=128). This is the
        # *only* viable fp8 route: NCCL EP LL carries bf16 (fp8 wire is
        # NOT SUPPORTED), and apply_deepep_ll consumes fp8+scale. Pure pointwise
        # quant => numerically identical to DeepEP's in-kernel quant.
        hs_fp8, hs_scale = self._quantize_fp8(recv_tokens, expert_counters)

        # expected_m mirrors DeepEP LL (deepep.py:675-678): the upper bound on
        # recv tokens per local expert. group_size here is the EP "group" (world
        # size), matching DeepEP's buffer.group_size usage.
        expected_m = (
            hidden_states.shape[0] * self.world_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts

        # Stash for combine (combine output is [T, H] in original token order).
        self._dispatched_topk_weights = topk_weights
        self._dispatched_t = t

        return DeepEPLLDispatchOutput(
            hs_fp8,
            hs_scale,
            topk_ids,
            topk_weights,
            expert_counters,
            expected_m,
        )

    def _quantize_fp8(self, recv_tokens_bf16: torch.Tensor, masked_m: torch.Tensor):
        if self._quant_fp8 is None:
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            self._quant_fp8 = sglang_per_token_group_quant_fp8
        # recv_tokens_bf16 is 3D [E_local, max_recv, H]; the quant helper natively
        # supports 3D + masked_m (fp8_kernel.py:599-651). masked_m=expert_counters
        # closes the loop on DeepEPLLDispatchOutput field 4 (Q5 spike).
        return self._quant_fp8(
            recv_tokens_bf16,
            group_size=_NCCL_EP_FP8_GROUP_SIZE,
            masked_m=masked_m,
        )

    def combine(self, combine_input) -> torch.Tensor:
        # DeepEPMoE wraps combine_input into DeepEPLLCombineInput(hidden_states, topk_ids, topk_weights).
        nccl_ep = self._nccl_ep
        if isinstance(combine_input, DeepEPLLCombineInput):
            expert_outputs = combine_input.hidden_states
            topk_weights = combine_input.topk_weights
        else:
            # Fallback: treat as a raw tensor (expert outputs).
            expert_outputs = combine_input
            topk_weights = self._dispatched_topk_weights

        # Combine output is [T, H] in original token order; T is the dispatched
        # token count (cached in dispatch). combine assert (nccl_ep.cc:3088)
        # requires topk_weights->sizes[0] <= num_max_dispatch_tokens_per_rank.
        t = self._dispatched_t
        if t > self.num_max_dispatch_tokens_per_rank:
            raise ValueError(
                f"NCCL EP: token count ({t}) exceeds per-rank dispatch budget "
                f"{self.num_max_dispatch_tokens_per_rank} (nccl_ep.cc:3088)."
            )

        H = self.hidden_size
        combined = torch.zeros(
            (t, H), dtype=torch.bfloat16, device=expert_outputs.device
        )

        with torch.cuda.stream(self.stream):
            inputs = nccl_ep.CombineInputs(tokens=nccl_ep.Tensor(expert_outputs))
            outputs = nccl_ep.CombineOutputs(
                tokens=nccl_ep.Tensor(combined),
                topk_weights=nccl_ep.Tensor(topk_weights),
            )
            self.handle.combine(
                inputs,
                outputs,
                config=nccl_ep.CombineConfig(),
                stream=self.stream.cuda_stream,
            )
        self.stream.synchronize()

        if self.handle is not None:
            try:
                self.handle.destroy()
            except Exception:
                pass
            self.handle = None
        return combined
