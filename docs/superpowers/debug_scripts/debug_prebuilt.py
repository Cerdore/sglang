import importlib.util, os, sys, glob
from pathlib import Path

ROOT = Path("/mnt/e/gitRepo/sglang/python/sglang/multimodal_gen/native/omnidreams_singleview")
build_dir = ROOT / "build" / "torch_extensions"

# Add lib paths like _load_prebuilt_extension does
cudnn_lib = "/home/chen/.python/sglang/lib/python3.12/site-packages/nvidia/cudnn/lib"
import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
ld = os.environ.get("LD_LIBRARY_PATH", "")
for lib in [cudnn_lib, torch_lib]:
    if lib not in ld:
        ld = lib + ":" + ld
os.environ["LD_LIBRARY_PATH"] = ld

for pattern in ("omnidreams_singleview_native_*/omnidreams*.so", "*/omnidreams*.so"):
    candidates = sorted(glob.glob(str(build_dir / pattern)))
    for so_path in candidates:
        try:
            spec = importlib.util.spec_from_file_location("test_prebuilt", so_path)
            if spec is None or spec.loader is None:
                print(f"spec/loader None: {so_path}")
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"OK:  {so_path}")
            print(f"     has optimized_dit_forward={hasattr(mod, optimized_dit_forward)}")
            print(f"     has omnidreams_vae_encode_wan_fp8={hasattr(mod, omnidreams_vae_encode_wan_fp8)}")
        except Exception as e:
            print(f"FAIL: {so_path}: {e}")
