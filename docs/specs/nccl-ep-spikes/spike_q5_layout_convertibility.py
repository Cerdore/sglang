"""
NCCL EP Spike Q5 — run_moe_core layout convertibility probe.

QUESTION (the REAL binary go/no-go for PR2, per Architect+Critic):
    Can NCCL EP's LL EXPERT_MAJOR / RANK_MAJOR receive layout be shaped into
    DeepEPLLDispatchOutput (the exact NamedTuple apply_deepep_ll consumes) WITHOUT
    adding a new quant-method entry point?

WHY IT MATTERS (verified against the tree):
    DeepEPMoE.run_moe_core (ep_moe/layer.py:210-243) routes ONLY on
    format_is_deepep_normal / format_is_deepep_ll. The deepgemm branches assert False.
    The single live LL compute route is:
        forward_cutlass_w4afp8_masked -> W4AFp8MoEMethod.apply_deepep_ll  (w4afp8.py:329)
    which unpacks:
        hidden_states, hidden_scales, topk_ids, _, masked_m, _ = dispatch_output
    i.e. it needs the DeepEPLLDispatchOutput shape:
        DeepEPLLDispatchOutput(
            hidden_states,        # [num_local_experts, max_recv_per_expert, hidden]  (masked expert-major)
            hidden_states_scale,  # fp8 scales or None
            topk_ids,             # [num_tokens, topk]
            topk_weights,         # [num_tokens, topk]
            masked_m,             # [num_local_experts]  (valid-token count per expert)
            expected_m,           # int
        )
    (see deepep.py:109-122 for the NamedTuple, deepep.py:715 for how LL dispatch fills it.)

    If NCCL EP's ncclEpDispatch(EXPERT_MAJOR) recv tensor + ncclEpLayoutInfo_t.expert_counters
    can be reshaped/renamed into exactly this tuple, then:
        PREFERRED PATH: NcclEpDispatcher emits DEEPEP_LL-format outputs directly.
        run_moe_core + apply_deepep_ll are reused UNCHANGED. This is the true #29402 mirror.
        => PR2 stays size L.
    If NOT (e.g. the recv layout is rank-major-only, or scales are packed differently,
    or there is no per-expert masked_m equivalent), then:
        FALLBACK PATH: PR2 must add NCCL_EP_LL format tag + an elif branch in run_moe_core
        + a new W4AFp8MoEMethod.apply_nccl_ep_ll.
        => PR2 balloons to size XL.

HOW TO RUN (GPU box with nccl4py[cu13]):
    torchrun --nproc_per_node=2 docs/specs/nccl-ep-spikes/spike_q5_layout_convertibility.py

    Static-only mode (no nccl4py, no GPU — just prints the target contract to match):
    python3 docs/specs/nccl-ep-spikes/spike_q5_layout_convertibility.py --contract-only

RESULT:
    Q5=PASS     recv layout coercible to DeepEPLLDispatchOutput  -> layout-reuse path, PR2=L
    Q5=FAIL     not coercible                                    -> new-quant-entry path, PR2=XL
    Record the verdict + the exact reshape (or the blocking reason) in PR1's description.

NOTE: investigation spike, not production code (expected to be deleted / folded into a
      gated parity test once PR1 lands).
"""

import argparse
import os
import sys


TARGET_CONTRACT = """
apply_deepep_ll unpacks exactly (w4afp8.py:337):
    hidden_states, hidden_scales, topk_ids, _, masked_m, _ = dispatch_output

DeepEPLLDispatchOutput NamedTuple (deepep.py:109):
    field 0  hidden_states        torch.Tensor   [E_local, M_max, H]  masked expert-major, dtype fp8_e4m3 or bf16
    field 1  hidden_states_scale  Tensor|None    fp8 per-(token,group) scale, or None for bf16
    field 2  topk_ids             torch.Tensor   [T, topk]  int
    field 3  topk_weights         torch.Tensor   [T, topk]  float32
    field 4  masked_m             torch.Tensor   [E_local]  int32  valid tokens per local expert
    field 5  expected_m           int            hint = ceil(T*topk / E_local) upper bound

To emit DEEPEP_LL directly, an NCCL EP LL dispatch MUST be able to produce:
  - a masked expert-major hidden buffer (NCCL EP EXPERT_MAJOR layout should map here)
  - per-local-expert valid counts (NCCL EP ncclEpLayoutInfo_t.expert_counters -> masked_m)
  - fp8 scales in the SAME packing cutlass_w4a8_moe_deepep_ll expects (or None for bf16)
  - the original topk_ids/topk_weights passed through (NCCL EP handle caches topk_idx)

BLOCKERS that would force the fallback path:
  - NCCL EP LL only supports RANK_MAJOR (not EXPERT_MAJOR) for the dtype/config we need
    -> would require a scatter to expert-major before the GEMM (extra kernel, new path)
  - expert_counters are global/int64 while masked_m must be per-local-expert int32
    -> a cast+slice may suffice (still PASS) OR a semantic mismatch (FAIL)
  - fp8 scale packing differs from cutlass_w4a8_moe_deepep_ll's expectation
    -> would require re-quant/repack (new path, FAIL)
"""


def contract_only():
    print(TARGET_CONTRACT, flush=True)
    print("Static contract printed. Map NCCL EP LL recv layout onto these 6 fields.", flush=True)
    print("If every field is a rename/reshape/cast -> Q5=PASS. If any needs a kernel -> Q5=FAIL.",
          flush=True)


def run_probe(rank):
    import torch

    try:
        import nccl.ep as ep
    except ImportError as e:
        print(f"[rank{rank}] nccl4py not importable: {e}", flush=True)
        print(f"[rank{rank}] Q5=SKIP  install nccl4py[cu13] on CUDA13 + H100/Blackwell", flush=True)
        return

    # Reference the sglang target tuple so the probe fails loudly if the contract drifts.
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput

    fields = DeepEPLLDispatchOutput._fields
    print(f"[rank{rank}] DeepEPLLDispatchOutput fields (must match): {fields}", flush=True)
    expected = ("hidden_states", "hidden_states_scale", "topk_ids",
                "topk_weights", "masked_m", "expected_m")
    if fields != expected:
        print(f"[rank{rank}] CONTRACT DRIFT: expected {expected}, got {fields}", flush=True)
        print(f"[rank{rank}] Re-read w4afp8.py apply_deepep_ll before trusting this spike.", flush=True)

    # --- The actual probe (fill in once nccl4py signature is known from Q1/Q2) ---
    # 1. build a small NCCL EP LL group + handle (topk_ids for a toy [T, topk] routing)
    # 2. run ncclEpDispatch with layout=EXPERT_MAJOR, capture the recv tensor + expert_counters
    # 3. attempt: DeepEPLLDispatchOutput(
    #        hidden_states = recv.view(E_local, M_max, H),
    #        hidden_states_scale = recv_scale_or_None,
    #        topk_ids = topk_ids,
    #        topk_weights = topk_weights,
    #        masked_m = expert_counters.to(torch.int32),
    #        expected_m = int(math.ceil(T * topk / E_local)),
    #    )
    # 4. feed it through a REAL DeepEPMoE.run_moe_core path (w4afp8) on a tiny model,
    #    OR at minimum assert cutlass_w4a8_moe_deepep_ll accepts the shapes without error.
    # 5. Q5=PASS if the tuple constructs AND the GEMM consumes it; Q5=FAIL otherwise.
    print(f"[rank{rank}] TODO: wire steps 1-5 using the nccl4py signature resolved by Q1/Q2.",
          flush=True)
    print(f"[rank{rank}] The target reshape is spelled out above; this probe confirms it round-trips.",
          flush=True)
    print(f"[rank{rank}] Q5=PENDING  (needs nccl4py signature from Q1 spike to complete)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract-only", action="store_true",
                    help="print the DeepEPLLDispatchOutput contract to match; no GPU/nccl4py needed")
    args = ap.parse_args()

    if args.contract_only:
        contract_only()
        return

    rank = int(os.environ.get("RANK", "0"))
    run_probe(rank)


if __name__ == "__main__":
    sys.exit(main())
