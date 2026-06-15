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
B,T,HW = 1,2,hp*wp
num_blocks = 28
shape = (B, 6*HW, 16, 128)

adapter = _SGLTransformerAdapter(dit, arch, len_t, dtype, hp*8, wp*8)
ex = opt.OptimizedDiTExecutor(adapter, ext, dit_backend="fp8_kvcache_cudnn", attention_backend="auto")

timestep = torch.tensor([0.5], device=device).expand(B).contiguous()
ex._ensure_invariant_tensors(ar_idx=0, timesteps=timestep)
ex._ensure_weights_snapshot()

w_before = ex._optimized_weights.get("additional_patch_embedding.proj.1.weight")
print("Before release:", "present" if w_before is not None else "MISSING")
if w_before is not None:
    print("  shape:", list(w_before.shape))

# Now trigger release
from sglang.multimodal_gen.runtime.models.dits.omnidreams_kvcache import BlockKVCache
caches = []
for _ in range(num_blocks):
    c = BlockKVCache(k_shape=shape, v_shape=shape, seq_dim=1, chunk_size=T*HW, window_size=6*HW, sink_size=0, device=device, dtype=dtype)
    c.before_update(0); c.update(torch.randn(B, T*HW, 16, 128, device=device, dtype=dtype), torch.randn(B, T*HW, 16, 128, device=device, dtype=dtype)); c.after_update(0); c.before_update(1)
    caches.append(c)

kc = [torch.randn(B,512,2048,device=device,dtype=dtype) for _ in range(num_blocks)]
vc = [torch.randn(B,512,2048,device=device,dtype=dtype) for _ in range(num_blocks)]
ks = [c._k for c in caches]; vs = [c._v for c in caches]

rt = ex._ensure_fp8_runtime(k_cross=kc, v_cross=vc, k_self=ks, v_self=vs, tokens=T*HW, cache=None)
ex._apply_runtime_config(rt)

w_after = ex._optimized_weights.get("additional_patch_embedding.proj.1.weight")
print("After release:", "present" if w_after is not None else "MISSING")
if w_after is not None:
    print("  shape:", list(w_after.shape))

# Also check all non-block keys
all_keys = sorted([k for k in ex._optimized_weights.keys() if not k.startswith("blocks.")])
print("\nNon-block keys after release:", len(all_keys))
for k in all_keys:
    v = ex._optimized_weights[k]
    sh = list(v.shape) if isinstance(v, torch.Tensor) else "non-tensor"
    print(f"  {k}: {sh}")
