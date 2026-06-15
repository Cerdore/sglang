"""Diagnose load_extension failure in sglang worker context."""
import os, sys

# Set up LD_LIBRARY_PATH like the worker would
os.environ.setdefault("LD_LIBRARY_PATH", "")
cudnn = "/home/chen/.python/sglang/lib/python3.12/site-packages/nvidia/cudnn/lib"
if cudnn not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = cudnn + ":" + os.environ["LD_LIBRARY_PATH"]

print("Python:", sys.executable)
print("CWD:", os.getcwd())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

from sglang.multimodal_gen.native.singleview_loader import (
    load_extension, extension_load_error, _ROOT, _load_prebuilt_extension
)
print("_ROOT:", _ROOT)
print("_ROOT exists:", os.path.isdir(str(_ROOT)))
build_dir = _ROOT / "build" / "torch_extensions"
print("build_dir:", build_dir)
print("build_dir exists:", os.path.isdir(str(build_dir)))
if os.path.isdir(str(build_dir)):
    import glob
    sos = glob.glob(str(build_dir / "**/*.so"), recursive=True)
    print("Found .so files:", len(sos))
    for s in sos[:3]: print("  ", s)

# Try prebuilt load
prebuilt = _load_prebuilt_extension()
print("_load_prebuilt_extension():", prebuilt is not None)
if prebuilt is None:
    err = extension_load_error()
    print("extension_load_error:", err)

# Try full load
ext = load_extension()
print("load_extension() returned None:", ext is None)
if ext is None:
    err = extension_load_error()
    print("extension_load_error:", err)
