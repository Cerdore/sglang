import multiprocessing as mp
from sglang.multimodal_gen.native import load_extension

def worker():
    ext = load_extension()
    print(f"ext is None: {ext is None}")
    if ext is not None:
        print("has vae symbols:", hasattr(ext, "omnidreams_vae_encode_wan_fp8"))
        status = ext.omnidreams_vae_backend_status("vae_encoder", "fp8")
        print("status:", status)
    else:
        from sglang.multimodal_gen.native.singleview_loader import extension_load_error
        err = extension_load_error()
        print("load error:", err)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=worker)
    p.start()
    p.join(timeout=120)
    print("worker exit:", p.exitcode)
