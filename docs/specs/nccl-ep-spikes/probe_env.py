"""
NCCL EP Spike — environment probe.

Checks every prerequisite the two go/no-go spikes (Q1, Q5) depend on, and prints a
single PASS/BLOCK verdict per requirement. Run this FIRST on the remote H20 box.

Usage:
    python3 docs/specs/nccl-ep-spikes/probe_env.py

No GPU is strictly required to run it (it degrades gracefully), but a meaningful
verdict needs the target box. Requirements checked:
    - CUDA 13            (nccl4py[cu13] is CUDA-13 only)  -> gates nccl4py install
    - GPU arch           (Hopper sm_90 e.g. H20/H100, or Blackwell) -> NCCL EP requirement
    - GPU count >= 2     (spikes need world_size >= 2)
    - NCCL >= 2.29       (Device API / GIN)               -> NCCL EP requirement
    - nccl4py importable (nccl.ep)                         -> the library under test
    - sglang importable  (editable install from this tree)
"""

import importlib
import re
import sys


def line(tag, ok, detail):
    mark = "PASS " if ok else ("WARN " if ok is None else "BLOCK")
    print(f"[{mark}] {tag}: {detail}", flush=True)
    return ok


def check_cuda():
    try:
        import torch
    except ImportError as e:
        return line("CUDA13", False, f"torch not importable: {e}")
    cuda = torch.version.cuda
    if cuda is None:
        return line("CUDA13", False, "torch has no CUDA (CPU build)")
    major = int(cuda.split(".")[0])
    ok = major >= 13
    return line("CUDA13", ok if major >= 13 else False,
                f"torch.version.cuda={cuda} (need major>=13 for nccl4py[cu13]; "
                f"{'ok' if ok else 'cu12 -> NCCL EP unavailable, falls back to DeepEP (plan Q3)'})")


def check_gpu_arch():
    try:
        import torch
        if not torch.cuda.is_available():
            return line("GPU_ARCH", False, "torch.cuda.is_available() == False")
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)  # (major, minor)
        major = cc[0]
        # Hopper = 9.x (H20/H100/H200), Blackwell = 10.x/12.x
        ok = major >= 9
        return line("GPU_ARCH", ok,
                    f"{name} cc={cc[0]}.{cc[1]} "
                    f"({'Hopper/Blackwell OK' if ok else 'pre-Hopper -> NCCL EP unsupported'})")
    except Exception as e:
        return line("GPU_ARCH", False, f"error: {e}")


def check_gpu_count():
    try:
        import torch
        n = torch.cuda.device_count()
        return line("GPU_COUNT>=2", n >= 2, f"{n} visible (spikes need >=2 for world_size>=2)")
    except Exception as e:
        return line("GPU_COUNT>=2", False, f"error: {e}")


def check_nccl_version():
    try:
        import torch
        if not torch.cuda.is_available():
            return line("NCCL>=2.29", None, "no CUDA; cannot read NCCL version")
        ver = torch.cuda.nccl.version()  # tuple like (2, 29, 3) on recent torch
        if isinstance(ver, int):
            # older torch encodes as int major*1000+minor*100+patch or similar
            s = str(ver)
            return line("NCCL>=2.29", None, f"torch.cuda.nccl.version()={ver} (decode manually)")
        major, minor = ver[0], ver[1]
        ok = (major, minor) >= (2, 29)
        return line("NCCL>=2.29", ok,
                    f"{'.'.join(map(str, ver))} "
                    f"({'ok' if ok else 'need >=2.29 for Device API/GIN'})")
    except Exception as e:
        return line("NCCL>=2.29", None, f"could not read NCCL version: {e}")


def check_nccl4py():
    try:
        import nccl.ep as ep  # noqa: F401
    except ImportError as e:
        return line("nccl4py", False,
                    f"nccl.ep not importable: {e} -> pip install nccl4py[cu13]")
    names = [n for n in dir(ep) if not n.startswith("__")]
    return line("nccl4py", True, f"nccl.ep imported; exports: {names}")


def check_sglang():
    try:
        import sglang  # noqa: F401
        from sglang.srt.distributed.parallel_state import get_moe_ep_group  # noqa: F401
        from sglang.srt.layers.moe.token_dispatcher.deepep import (  # noqa: F401
            DeepEPLLDispatchOutput,
        )
        return line("sglang", True, "sglang + EP entry points importable")
    except Exception as e:
        return line("sglang", False, f"import failed: {e} -> pip install -e python/")


def main():
    print("=== NCCL EP spike environment probe ===", flush=True)
    results = [
        check_cuda(),
        check_gpu_arch(),
        check_gpu_count(),
        check_nccl_version(),
        check_nccl4py(),
        check_sglang(),
    ]
    hard = [r for r in results if r is False]
    print("\n=== SUMMARY ===", flush=True)
    if not hard:
        print("All hard requirements PASS (WARN items are informational). "
              "You can run both spikes.", flush=True)
    else:
        print(f"{len(hard)} requirement(s) BLOCK. Resolve them before running the spikes.", flush=True)
        print("If only CUDA13/nccl4py BLOCK on a cu12 box, that itself confirms plan Q3: "
              "NCCL EP is cu13-only and falls back to DeepEP on cu12.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
