from sglang.multimodal_gen.native.singleview_loader import _load_prebuilt_extension
import time
t0 = time.time()
mod = _load_prebuilt_extension()
print("time: %.2fs" % (time.time()-t0))
print("mod is None:", mod is None)
if mod is not None:
    print("has optimized_dit_forward:", hasattr(mod, "optimized_dit_forward"))
    print("has omnidreams_vae_encode_wan_fp8:", hasattr(mod, "omnidreams_vae_encode_wan_fp8"))
    print("has sage3_is_built:", hasattr(mod, "sage3_is_built"))
