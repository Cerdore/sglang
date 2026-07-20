# Multi-node NCCL RAS dead-rank reproduction

End-to-end repro of the **cross-node** RAS dead-rank path: a rank on one node
dies, the survivor node's RAS reports it as `missing_ranks` → `unresponsive`
(immediate) → `considered_dead` (~60s). This path is not reproducible
single-node (a single-node hard-crash tears the whole job down before RAS's
~60s `considered_dead`).

It is **not** a CI test (CI is single-node). It's a manual cluster repro for
extra confidence on the cross-node semantics — optional per the PR1 plan
(`§E`), and the only place multi-node adds value beyond the single-node
"2-rank simulation" already in
`test/registered/unit/distributed/fixtures/nccl_ras_status_dead.json`.

## What it checks

- RAS bootstrap meshes across 2 nodes (node0's `localhost:28028` sees rank1
  on node1).
- After SIGKILL of rank1 on node1, node0's RAS (still alive, idle — no pending
  collective) reports rank1 in `missing_ranks[]` with `unresponsive=True`
  within ~30s and `considered_dead=True` within ~90s.
- Soft assertions in `ras_multinode_timeline.py` fail the run otherwise.

## Prereqs

- 2 nodes, each with ≥1 GPU, reachable over the network, passwordless SSH
  from node0 to node1 (`NODE1_SSH`).
- Same NCCL (≥ 2.28.7) + torch on both.
- `NCCL_RAS_ENABLE=1` is set by the worker; no SGLang server needed (this is a
  raw NCCL idle-comm repro).

## Run

On **node0** (the world-rank-0 / RAS collector host):

```bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=2 --nproc_per_node=1 \
  --rdzv_id=ras-mn --rdzv_backend=c10d \
  --rdzv_endpoint=$NODE0_IP:29500 \
  test/manual/distributed/ras_multinode_worker.py
```

On **node1** (same command, simultaneously):

```bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=2 --nproc_per_node=1 \
  --rdzv_id=ras-mn --rdzv_backend=c10d \
  --rdzv_endpoint=$NODE0_IP:29500 \
  test/manual/distributed/ras_multinode_worker.py
```

Both ranks print `RAS up at localhost:28028, idling` once the communicator is
built. Then, **on node0** in a second shell:

```bash
NODE1_SSH=user@$NODE1_IP python \
  test/manual/distributed/ras_multinode_timeline.py
```

The timeline script polls `localhost:28028`, picks rank1's pid + host from
STATUS, SSH-kills it on node1, and asserts the transition windows. A `PASS`
line means the cross-node dead-rank path reproduced.

## Tuning

- `POLL_SECONDS` (default 120) — extend if NCCL's `considered_dead` is scaled
  by `NCCL_RAS_TIMEOUT_FACTOR` on your cluster.
- `RAS_HOST`/`RAS_PORT` — if running the timeline from a third host that
  reaches node0 via a non-loopback address (port-forward 28028).
- To also verify a live **SGLang server** path (production scenario, where a
  pending collective may trigger NCCL abort + the forward watchdog rather
  than a clean RAS dead-rank report), launch SGLang `--tp 2` across the 2
  nodes with `SGLANG_NCCL_RAS_ENABLE=1 --enable-metrics` and `curl
  /metrics | grep nccl_ras` after killing node1's scheduler; expect
  `nccl_ras_poll_success` to drop and/or `nccl_ras_missing_ranks` to rise
  before the job tears down.
