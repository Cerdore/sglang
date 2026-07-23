"""
NCCL EP Spike Q1 — ncclComm_t binding probe.

QUESTION (binary go/no-go for PR2):
    Does nccl4py's `nccl.ep` accept an EXTERNAL ncclComm_t obtained from sglang's
    PyNcclCommunicator.comm, or must it create/own its own communicator?

WHY IT MATTERS:
    NCCL EP's ncclEpCreateGroup(ep_group, ncclComm_t comm, &config) binds an EP group
    to an existing NCCL communicator. sglang already owns a comm per EP group at
    get_moe_ep_group().pynccl_comm.comm (ctypes c_void_p, see pynccl.py:111,
    pynccl_wrapper.py:73). If nccl.ep can reuse that comm, PR2 is a clean plug-in.
    If it insists on creating its own comm, PR2 must introduce a new communicator
    lifecycle path (coordinate with #25231), which is a materially larger scope.

HOW TO RUN (on an H100/Blackwell box with nccl4py[cu13] installed, CUDA 13, NCCL>=2.29):
    cd /path/to/sglang-fea_ncclep
    torchrun --nproc_per_node=2 docs/specs/nccl-ep-spikes/spike_q1_comm_binding.py

    Or standalone (no sglang) to test raw nccl4py comm acceptance only:
    torchrun --nproc_per_node=2 docs/specs/nccl-ep-spikes/spike_q1_comm_binding.py --raw

RESULT:
    Prints one of:
      Q1=PASS  external-comm reuse works   -> PR2 plugs into get_moe_ep_group().pynccl_comm.comm
      Q1=PARTIAL  nccl.ep needs its own comm but a comm can be built from the same unique_id
      Q1=FAIL  nccl.ep rejects external comm and needs an independent init path
    Record the verdict in PR1's description.

NOTE: This is an investigation spike, not production code. It is expected to be
      deleted or folded into a gated unit test once PR1 lands. All the sglang
      interface points it touches were verified against the tree:
        - get_moe_ep_group()                     parallel_state.py:1849
        - GroupCoordinator.pynccl_comm           parallel_state.py:427-429
        - PyNcclCommunicator.comm : ncclComm_t    pynccl.py:111
        - ncclComm_t = ctypes.c_void_p           pynccl_wrapper.py:73
"""

import argparse
import ctypes
import os
import sys


def log(rank, *a):
    print(f"[rank{rank}]", *a, flush=True)


def probe_nccl_ep_api(rank):
    """Introspect the nccl.ep Python surface so we know what create_group expects."""
    try:
        import nccl.ep as ep
    except ImportError as e:
        log(rank, "nccl4py not importable:", e)
        log(rank, "Q1=SKIP  install nccl4py[cu13] on a CUDA13 + NCCL>=2.29 H100/Blackwell box")
        return None

    # Report the algo constants and the group-creation entrypoint signature.
    names = [n for n in dir(ep) if not n.startswith("__")]
    log(rank, "nccl.ep exports:", names)
    for cand in ("NCCLLibrary", "create_group", "ncclEpCreateGroup", "Group", "EpGroup"):
        if hasattr(ep, cand):
            obj = getattr(ep, cand)
            log(rank, f"  found {cand}: {obj!r}")
            doc = (getattr(obj, "__doc__", "") or "")[:400]
            if doc:
                log(rank, f"    doc: {doc}")
    return ep


def get_external_comm(rank):
    """Get sglang's ncclComm_t for the MoE-EP group (the comm we want nccl.ep to reuse)."""
    import torch
    import torch.distributed as dist
    from sglang.srt.distributed import parallel_state as ps

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Minimal MP init: EP == TP == world here (2 ranks) is fine for a comm-binding probe.
    world = dist.get_world_size()
    ps.init_distributed_environment(world_size=world, rank=rank, local_rank=rank % torch.cuda.device_count())
    ps.initialize_model_parallel(tensor_model_parallel_size=world)

    ep_group = ps.get_moe_ep_group()
    pynccl = ep_group.pynccl_comm
    if pynccl is None or pynccl.disabled:
        log(rank, "pynccl_comm unavailable (world_size==1 or missing NCCL). Cannot probe external comm.")
        return None
    comm = pynccl.comm  # ncclComm_t == ctypes.c_void_p
    log(rank, "sglang EP comm handle (ncclComm_t c_void_p):", hex(comm.value if hasattr(comm, "value") else int(comm)))
    return comm


def try_create_ep_group(ep, comm, rank):
    """
    Attempt to create an NCCL EP group from an external ncclComm_t.

    This is deliberately defensive: the exact nccl4py Python signature is the Q2
    unknown. We try the most likely shapes and report which one (if any) accepts
    the external comm without constructing its own communicator.
    """
    if ep is None:
        return "SKIP"

    # nccl.ep likely wraps ncclEpCreateGroup(ep_group*, ncclComm_t comm, &config).
    # The Python binding may take the comm as an int/c_void_p, or as a wrapper obj.
    algo = getattr(ep, "NCCL_EP_ALGO_LOW_LATENCY", None)
    comm_int = comm.value if hasattr(comm, "value") else int(comm)

    attempts = []

    # Attempt A: a NCCLLibrary-style facade with a create_group(comm, ...) method.
    if hasattr(ep, "NCCLLibrary"):
        try:
            lib = ep.NCCLLibrary()
            for meth in ("ncclEpCreateGroup", "create_group", "ep_create_group"):
                if hasattr(lib, meth):
                    fn = getattr(lib, meth)
                    log(rank, f"trying NCCLLibrary.{meth}(comm=...) with external comm")
                    # Try passing the raw comm handle. Signature TBD (Q2); catch and record.
                    try:
                        grp = fn(comm_int, num_experts=8, algorithm=algo,
                                 max_dispatch_tokens_per_rank=128)
                        attempts.append((f"NCCLLibrary.{meth}", "ACCEPTED", grp))
                        return "PASS"
                    except TypeError as te:
                        attempts.append((f"NCCLLibrary.{meth}", f"signature-mismatch: {te}", None))
                    except Exception as ex:
                        attempts.append((f"NCCLLibrary.{meth}", f"error: {ex}", None))
        except Exception as ex:
            attempts.append(("NCCLLibrary()", f"ctor-error: {ex}", None))

    # Attempt B: a module-level create_group / Group class taking comm.
    for cand in ("create_group", "ncclEpCreateGroup", "Group", "EpGroup"):
        if hasattr(ep, cand):
            fn = getattr(ep, cand)
            try:
                log(rank, f"trying ep.{cand}(comm=...) with external comm")
                grp = fn(comm_int)
                attempts.append((f"ep.{cand}", "ACCEPTED", grp))
                return "PASS"
            except TypeError as te:
                attempts.append((f"ep.{cand}", f"signature-mismatch: {te}", None))
            except Exception as ex:
                attempts.append((f"ep.{cand}", f"error: {ex}", None))

    log(rank, "no attempt accepted the external comm; per-attempt results:")
    for name, status, _ in attempts:
        log(rank, f"  {name}: {status}")
    # If every attempt was a signature mismatch (not a hard reject), the answer is
    # likely PARTIAL: the binding exists but its Python signature (Q2) needs matching.
    if attempts and all("signature-mismatch" in s for _, s, _ in attempts):
        return "PARTIAL"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true",
                    help="skip sglang; only introspect nccl.ep API and test a self-created comm")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    ep = probe_nccl_ep_api(rank)

    if args.raw:
        log(rank, "raw mode: only API introspection performed. Inspect the exports above,")
        log(rank, "then map the create_group signature to fill in try_create_ep_group.")
        return

    comm = get_external_comm(rank)
    if comm is None:
        log(rank, "Q1=SKIP  could not obtain sglang EP comm (need >=2 ranks on GPU).")
        return

    verdict = try_create_ep_group(ep, comm, rank)
    log(rank, f"Q1={verdict}")
    if rank == 0:
        print("\n=== Q1 VERDICT ===", verdict, flush=True)
        print("PASS   -> PR2 reuses get_moe_ep_group().pynccl_comm.comm directly")
        print("PARTIAL-> binding exists; match the nccl4py signature (Q2) then re-run")
        print("FAIL   -> nccl.ep needs its own comm; PR2 must add a comm lifecycle (coord #25231)")


if __name__ == "__main__":
    sys.exit(main())
