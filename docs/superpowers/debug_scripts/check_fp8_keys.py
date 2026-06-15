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
arch = cfg.arch_config; len_t, hp, wp = 2, 44, 80
adapter = _SGLTransformerAdapter(dit, arch, len_t, dtype, hp*8, wp*8)
ex = opt.OptimizedDiTExecutor(adapter, ext, dit_backend="fp8_kvcache_cudnn", attention_backend="auto")
print("OK"); B,T,HW = 1,2,hp*wp
timestep = torch.tensor([0.5], device=device).expand(B).contiguous()
ex._ensure_invariant_tensors(ar_idx=0, timesteps=timestep)
ex._ensure_weights_snapshot()
w = ex._optimized_weights
top = [k for k in sorted(w.keys()) if not k.startswith("blocks.")]
print(f"top keys: {len(top)}")
for k in top:
    v = w[k]
    if isinstance(v, torch.Tensor): print(f"  {k}: {list(v.shape)}")
    else: print(f"  {k}: {v}")
