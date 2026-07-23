#!/usr/bin/env bash
#
# NCCL EP spike — one-shot driver for the remote H20 box.
#
# Runs, in order, and tees everything to a timestamped log you can paste back:
#   0. probe_env.py            — verify CUDA13 / Hopper / >=2 GPU / NCCL>=2.29 / nccl4py / sglang
#   1. spike_q5 --contract-only — print the layout contract (works even with no GPU)
#   2. spike_q1 --raw          — introspect the nccl.ep API surface (resolves Q2)
#   3. spike_q1 (full)         — Q1 verdict: external ncclComm_t binding
#   4. spike_q5 (full)         — Q5 verdict: LL recv layout convertibility
#
# Usage (from the repo root):
#   bash docs/specs/nccl-ep-spikes/run_all.sh
#   NPROC=2 bash docs/specs/nccl-ep-spikes/run_all.sh     # override GPU count (default 2)
#
# If a step BLOCKs (e.g. nccl4py missing on a cu12 box), later steps still run and
# report SKIP — the log is complete regardless. Paste the whole log back.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
NPROC="${NPROC:-2}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="$HERE/spike-run-${TS}.log"

cd "$REPO_ROOT"

{
  echo "========================================================================"
  echo "NCCL EP spike run @ ${TS}"
  echo "repo: $REPO_ROOT"
  echo "nproc_per_node: $NPROC"
  echo "python: $(command -v python3)"
  echo "torchrun: $(command -v torchrun || echo 'NOT FOUND — pip install torch')"
  echo "========================================================================"

  echo
  echo "############ STEP 0: environment probe ############"
  python3 docs/specs/nccl-ep-spikes/probe_env.py

  echo
  echo "############ STEP 1: Q5 contract (static, no GPU needed) ############"
  python3 docs/specs/nccl-ep-spikes/spike_q5_layout_convertibility.py --contract-only

  echo
  echo "############ STEP 2: Q1 raw API introspection ############"
  echo "# learns the nccl.ep create_group signature (resolves open question Q2)"
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node="$NPROC" \
      docs/specs/nccl-ep-spikes/spike_q1_comm_binding.py --raw
  else
    echo "SKIP: torchrun not found"
  fi

  echo
  echo "############ STEP 3: Q1 full — external ncclComm_t binding ############"
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node="$NPROC" \
      docs/specs/nccl-ep-spikes/spike_q1_comm_binding.py
  else
    echo "SKIP: torchrun not found"
  fi

  echo
  echo "############ STEP 4: Q5 full — layout convertibility ############"
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node="$NPROC" \
      docs/specs/nccl-ep-spikes/spike_q5_layout_convertibility.py
  else
    echo "SKIP: torchrun not found"
  fi

  echo
  echo "========================================================================"
  echo "DONE. Paste this whole log back. Key lines to look for:"
  echo "  Q1=PASS|PARTIAL|FAIL   (step 3)"
  echo "  Q5=PASS|FAIL|PENDING   (step 4)"
  echo "  and the nccl.ep exports + create_group signature (step 2)"
  echo "========================================================================"
} 2>&1 | tee "$LOG"

echo
echo "Log saved to: $LOG"
