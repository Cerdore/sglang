import torch, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
from sglang.multimodal_gen.native import load_extension
from sglang.multimodal_gen.native.singleview_loader import load_python_module
from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import _SGLTransformerAdapter

device = torch.device("cuda"); dtype = torch.bfloat16
cfg = OmniDreamsDiTConfig()
dit = OmniDreamsDiT(config=cfg, hf_config={})
dit.post_load_weights(); dit = dit.to(device, dtype=dtype).eval()
ext = load_extension(); opt = load_python_module("optimized_dit")
arch = cfg.arch_config
len_t, hp, wp = 2, 44, 80
B,T,HW,D = 1,2,hp*wp,2048

# Direct test of _make_cosmos_hdmap_cache with raw state_dict
from sglang.multimodal_gen.native.omnidreams_singleview.python.optimized_dit import _make_cosmos_hdmap_cache

sd = dit.state_dict()
w = {k: v.to(device) for k, v in sd.items()}
hdmap = torch.randn(1, 1, 2, HW, 64, device=device, dtype=dtype)
w_key = "additional_patch_embedding.proj.1.weight"
print("w_hd", w_key, list(w[w_key].shape))
r = _make_cosmos_hdmap_cache(hdmap_patched=hdmap, weights=w, model_channels=2048, dtype=dtype)
print("result:", list(r.shape))

# Now test executor with ar_idx=0
adapter = _SGLTransformerAdapter(dit, arch, len_t, dtype, hp*8, wp*8)
ex = opt.OptimizedDiTExecutor(adapter, ext, dit_backend="bf16", attention_backend="auto")
timestep = torch.tensor([0.5], device=device).expand(B).contiguous()
ex._ensure_invariant_tensors(ar_idx=0, timesteps=timestep)
ex._ensure_weights_snapshot()

# Directly read the weight from optimized dict
ow = ex._optimized_weights.get(w_key)
print("optimized", w_key, list(ow.shape) if ow is not None else "MISSING")

emb0 = ex._ensure_hdmap_tensor(ar_idx=0, input_for_ext=hdmap)
print("ar_idx=0 hdmap:", "cached" if emb0 is not None else "None")
emb1 = ex._ensure_hdmap_tensor(ar_idx=1, input_for_ext=hdmap)
print("ar_idx=1 hdmap:", "cached" if emb1 is not None else "None")
