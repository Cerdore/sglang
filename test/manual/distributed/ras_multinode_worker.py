"""Idle NCCL worker for the multi-node RAS dead-rank repro.

Runs on each node via ``torchrun --nnodes=2 --nproc_per_node=1`` (one rank per
node). Each rank enables NCCL RAS and leaves ``NCCL_RAS_ADDR`` at its default
``localhost:28028`` — each node binds its own loopback socket, and the RAS
bootstrap meshes them, so querying EITHER node's 28028 returns the full
job-wide view (including the other node's rank).

After ``init_process_group`` + a single ``barrier`` (forces communicator
creation), the rank idles forever. Idle = no pending collective, so NCCL does
NOT abort the survivor when a peer dies — the survivor's RAS keep-alive stays
up and reports the dead peer as ``unresponsive`` (~immediate) then
``considered_dead`` (~60s). This is the clean path that a live serving server
(whose pending collectives trigger NCCL abort on peer death) cannot reproduce
single-node.

Companion: ``ras_multinode_timeline.py`` (run on node0) polls 28028, SIGKILLs
rank1 on the other node via SSH, and records/asserts the transition timeline.
"""
import os
import time

import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank = int(os.environ["RANK"])
# Enable RAS; keep the default socket (localhost:28028) — each node binds its
# own loopback, and the RAS bootstrap meshes them across nodes.
os.environ.setdefault("NCCL_RAS_ENABLE", "1")

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
dist.barrier()  # force a communicator so RAS has something to report on

print(
    f"[rank{rank} pid={os.getpid()} host={os.uname().nodename} dev={local_rank}] "
    f"RAS up at localhost:28028, idling",
    flush=True,
)
while True:
    time.sleep(1)
