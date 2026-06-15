"""Isolate OptimizedDiTExecutor setup and verify each step."""
import torch, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
from sglang.multimodal_gen.native import load_extension
from sglang.multimodal_gen.native.singleview_loader import load_python_module
from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import _SGLTransformerAdapter, _NetCfg

device = torch.device("cuda")
dtype = torch.bfloat16

# 1. Load DiT
cfg = OmniDreamsDiTConfig()
dit = OmniDreamsDiT(config=cfg, hf_config={})
dit.post_load_weights()
dit = dit.to(device, dtype=dtype).eval()
print("1. DiT loaded OK, params:", sum(p.numel() for p in dit.parameters()))

# 2. Load native ext
ext = load_extension()
print("2. Ext loaded OK:", ext is not None)
print("   has optimized_dit_forward:", hasattr(ext, "optimized_dit_forward"))

# 3. Build adapter config
arch = cfg.arch_config
len_t, hp, wp = 2, 44, 80
net_cfg = _NetCfg(arch)
print("3. _NetCfg:", {k: v for k, v in net_cfg.__dict__.items() if not k.startswith("_")})

# 4. Build adapter
adapter = _SGLTransformerAdapter(dit, arch, len_t, dtype, hp * 8, wp * 8)
print("4. Adapter OK, config.use_cuda_graph:", adapter.config.use_cuda_graph)

# 5. Build executor (this is where v1/v2 crashed)
opt = load_python_module("optimized_dit")
ex = opt.OptimizedDiTExecutor(
    adapter,
    ext,
    dit_backend="fp8_kvcache_cudnn",
    attention_backend="auto",
    sparge_topk=None,
)
print("5. Executor created OK")

# 6. Prepare B=1, T=2 dummy inputs
B, T, HW, D = 1, 2, hp * wp, 2048
noisy = torch.randn(B, 1, T, HW, D, device=device, dtype=dtype)
mask = torch.zeros(B, 1, T, HW, 4, device=device, dtype=dtype)
hdmap = torch.randn(B, 1, T, HW, 16, device=device, dtype=dtype)
timestep = torch.tensor([0.5], device=device).expand(B).contiguous()
rope_freqs = torch.randn(T * HW, 1, 1, 128, device=device, dtype=torch.float32)

# 7. Build invariants
print("7. Building invariants...")
inv = ex._ensure_invariant_tensors(ar_idx=0, timesteps=timestep)
print("   inv keys:", [k for k in dir(inv) if not k.startswith("_")] if hasattr(inv, "__dict__") else type(inv))

# 8. Build rope
rope_cos, rope_sin = ex._ensure_rope_tensors(ar_idx=0, rope_freqs=rope_freqs)
print("8. Rope OK: cos", rope_cos.shape, "sin", rope_sin.shape)

# 9. Weights snapshot -- CRITICAL
print("9. Taking weights snapshot...")
weights = ex._ensure_weights_snapshot()
print("   weight keys:", len(weights))
# Check a few
for k in ["blocks.0.self_attn.q_proj.weight", "blocks.0.self_attn.q_proj.weight_fp8_prepared",
          "blocks.0.self_attn.q_proj.weight_fp8_prepared_scale"]:
    if k in weights:
        w = weights[k]
        print(f"   {k}: shape={list(w.shape)}, dtype={w.dtype}, device={w.device}")
    else:
        print(f"   {k}: MISSING")

# 10. Build dummy KV caches
k_self = [torch.randn(B, 12, HW, 128, device=device, dtype=dtype) for _ in range(28)]
v_self = [torch.randn(B, 12, HW, 128, device=device, dtype=dtype) for _ in range(28)]
k_cross = [torch.randn(B, 512, 2048, device=device, dtype=dtype) for _ in range(28)]
v_cross = [torch.randn(B, 512, 2048, device=device, dtype=dtype) for _ in range(28)]

# 11. Build runtime config
print("11. Building runtime config...")
tokens = T * HW
rt = ex._ensure_fp8_runtime(
    k_cross=k_cross, v_cross=v_cross, k_self=k_self, v_self=v_self,
    tokens=tokens, cache=None
)
print("   runtime config keys:", len(rt) if rt else 0)
if rt:
    # Print first few values  
    for k, v in list(rt.items())[:5]:
        if isinstance(v, torch.Tensor):
            print(f"   {k}: shape={list(v.shape)}")
        else:
            print(f"   {k}: {v}")

# 12. Apply runtime
print("12. Applying runtime config...")
ex._apply_runtime_config(rt)

# 13. Try forward
print("13. Running optimized_dit_forward with dummy inputs...")
try:
    compute_write_start = opt.compute_self_attn_write_start
    write_start = compute_write_start(k_self[0]) if callable(compute_write_start) else 0
    print(f"    write_start: {write_start}")
    
    out = ex._predict_flow_ext_impl(
        noisy, mask, hdmap, None, timestep, rope_freqs,
        inv.t_emb, inv.t_emb_silu, inv.adaln_lora,
        inv.final_shift, inv.final_scale,
        rope_cos, rope_sin,
        inv.block_mods_sa, inv.block_mods_ca, inv.block_mods_mlp,
        k_cross, v_cross, k_self, v_self, write_start,
    )
    print("    SUCCESS! output shape:", list(out.shape))
except Exception as e:
    print(f"    FAILED: {e}")
    import traceback
    traceback.print_exc()
