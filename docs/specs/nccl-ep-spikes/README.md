# NCCL EP — PR1 De-Risking Spikes

Two **investigation spikes** that answer the binary go/no-go questions gating PR2 of the
[NCCL EP execution plan](../../../.omc/plans/nccl-ep-execution-plan.md). Run these on a GPU box
**before** writing any functional dispatch code. Record the verdicts in PR1's description; PR2's
scope (size L vs XL) depends on them.

> These are throwaway probes, not production code. They use `hasattr`/`getattr` to introspect the
> **unknown** nccl4py Python surface (that discovery is their entire job, cf. open question Q2) —
> this is deliberate and does not conflict with the repo's `no-getattr-defensive` rule, which
> targets defensive access to *known* fields in production code. Expect to delete these or fold
> them into gated unit tests once PR1 lands.

## Prerequisites

- Hopper (H20 / H100 / H200, sm_90) or Blackwell GPU, ≥ 2 visible (the probes need `world_size ≥ 2`).
  **H20 is fine** — it is same-generation Hopper (only down-clocked); #25231 validated nccl4py on H20.
- CUDA 13, NCCL ≥ 2.29 with Device API / GIN.
- `pip install nccl4py[cu13]`.
- An editable sglang install from this tree (`pip install -e python/`).

## Quick start (one command on the H20 box)

```bash
cd /path/to/sglang-fea_ncclep
bash docs/specs/nccl-ep-spikes/run_all.sh          # default 2 GPUs
# or: NPROC=2 bash docs/specs/nccl-ep-spikes/run_all.sh
```

`run_all.sh` runs the env probe + both spikes in order and tees everything to a
timestamped `spike-run-*.log`. **Paste that whole log back.** If a step blocks
(e.g. nccl4py missing on a cu12 box) later steps still run and report SKIP — the
log is complete regardless.

The key lines to look for:
- `Q1=PASS|PARTIAL|FAIL` (comm binding)
- `Q5=PASS|FAIL|PENDING` (layout convertibility)
- the `nccl.ep` exports + `create_group` signature (from the `--raw` step)

## The two questions

| Spike | Question | If PASS | If FAIL |
|-------|----------|---------|---------|
| **Q1** — `spike_q1_comm_binding.py` | Does `nccl.ep` accept an **external** `ncclComm_t` from `get_moe_ep_group().pynccl_comm.comm`? | PR2 reuses sglang's existing EP comm directly. | PR2 must add a new nccl4py-owned comm lifecycle (coordinate with #25231). |
| **Q5** — `spike_q5_layout_convertibility.py` | Can NCCL EP's LL `EXPERT_MAJOR` recv layout be reshaped into `DeepEPLLDispatchOutput` **without** a new quant-method entry point? | **Layout-reuse path**: emit `DEEPEP_LL` tags, reuse `run_moe_core` + `apply_deepep_ll` unchanged (true #29402 mirror). PR2 = **L**. | **New-entry path**: add `NCCL_EP_LL` tag + `run_moe_core` elif branch + `apply_nccl_ep_ll`. PR2 = **XL**. |

Q5 is the **higher-probability** scope risk (per the Architect/Critic consensus) — Q1 is likely
positive because `PyNcclCommunicator.comm` already exposes a raw `ncclComm_t` today
(`pynccl.py:111`, `ncclComm_t = ctypes.c_void_p` at `pynccl_wrapper.py:73`).

## Running

```bash
cd /path/to/sglang-fea_ncclep

# Q1 — comm binding (2 ranks)
torchrun --nproc_per_node=2 docs/specs/nccl-ep-spikes/spike_q1_comm_binding.py
# API-introspection only (no sglang init), to first learn the nccl.ep create_group signature:
torchrun --nproc_per_node=2 docs/specs/nccl-ep-spikes/spike_q1_comm_binding.py --raw

# Q5 — layout convertibility (2 ranks)
torchrun --nproc_per_node=2 docs/specs/nccl-ep-spikes/spike_q5_layout_convertibility.py
# Print the target contract to match, no GPU/nccl4py needed (works anywhere, incl. this mac):
python3 docs/specs/nccl-ep-spikes/spike_q5_layout_convertibility.py --contract-only
```

## Recommended order

1. `spike_q1_comm_binding.py --raw` first — it prints the `nccl.ep` exports and the
   `create_group` signature. This resolves open question **Q2** (does the Python API mirror the C
   API 1:1?) and tells you exactly how to finish both probes.
2. `spike_q1_comm_binding.py` (full) — get the Q1 verdict.
3. `spike_q5_layout_convertibility.py` — using the signature from step 1, complete steps 1–5 in
   `run_probe()` and get the Q5 verdict.

## Filling in the verdicts

Each script prints `Q1=<verdict>` / `Q5=<verdict>`. Copy the verdict **and the evidence** (the
accepted `create_group` signature for Q1; the exact reshape or the blocking reason for Q5) into
PR1's description under a "Spike results" heading. PR2's go/no-go and size estimate follow
directly:

- Q1=PASS **and** Q5=PASS → PR2 is size **L**, layout-reuse path, plugs into the existing comm.
- Q1=PASS, Q5=FAIL → PR2 is size **XL** (adds `apply_nccl_ep_ll`).
- Q1=FAIL → PR2 blocked on comm-ownership design; PR1 packaging still lands standalone.

## What else to verify on the same box (opportunistic, while you're there)

These are cheap to check on H20 during the same session and each de-risks a later PR.
`run_all.sh` covers Q1/Q2/Q3/Q5; the rest are manual one-liners.

| Check | Why | How |
|-------|-----|-----|
| **Q2** — Python API mirrors C API 1:1 | PR2 LL impl shape | Covered by `spike_q1 --raw` (step 2 of `run_all.sh`); read the printed `create_group` signature + `dir(nccl.ep)`. |
| **Q3** — CUDA 13 gate | PR1 capability detection | Covered by `probe_env.py`. cu12 → confirms the DeepEP fallback path. |
| **NCCL EP self-test works on H20** | rules out a library-vs-H20 bug confusing the spikes | If NCCL was built from source with `contrib/nccl_ep`, run its `ep_test` / `ep_bench` binaries (LL, BF16, 8 experts) once. If using only the `nccl4py` wheel, skip — the spikes exercise the same path. |
| **LL `enable_mask` / zero-token APIs exist in the wheel** | PR2 zero-token ranks + PR4b fault path | `python3 -c "import nccl.ep as e; print([n for n in dir(e) if 'mask' in n.lower() or 'error' in n.lower()])"` — expect mask/query/update/clean/async-error surface. |
| **Explicit `rdma_buffer_size` accepted (not just `NCCL_EP_AUTO`)** | PR4a CUDA-graph capture needs static buffers | When completing the Q1 probe, pass an explicit `rdma_buffer_size>0` to `create_group` and confirm it is accepted; note whether `NCCL_EP_AUTO` and explicit sizing are both available. |
| **`topk>9` / `hidden>1024` behavior** | PR2 #2103 fallback scope | Optional: try creating an LL group with `num_topk=10` or `hidden=1024`; confirm it errors/rejects (validates the fallback-to-DeepEP gate is needed). |

Report each result as a short line in PR1's "Spike results" section. Anything that
errors unexpectedly on H20 (vs. an H100 expectation) is worth flagging — H20's
reduced NVLink/BW should not affect *functional* correctness, only throughput.

## Verified interface points (as of this tree)

- `get_moe_ep_group()` — `python/sglang/srt/distributed/parallel_state.py:1849`
- `GroupCoordinator.pynccl_comm` — `parallel_state.py:427-429`
- `PyNcclCommunicator.comm : ncclComm_t` — `pynccl.py:111`
- `ncclComm_t = ctypes.c_void_p` — `pynccl_wrapper.py:73`
- `DeepEPLLDispatchOutput` NamedTuple — `token_dispatcher/deepep.py:109-122`
- LL dispatch fills it — `deepep.py:715`
- `run_moe_core` routing (deepgemm asserts False) — `ep_moe/layer.py:210-243`
- `W4AFp8MoEMethod.apply_deepep_ll` unpack — `quantization/w4afp8.py:329-337`
