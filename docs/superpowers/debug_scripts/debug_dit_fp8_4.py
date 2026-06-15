"""Isolate native optimized_dit_forward with initialized caches."""
import torch, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
from sglang.multimodal_gen.native import load_extension
from sglang.multimodal_gen.native.singleview_loader import load_python_module
from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import _SGLTransformerAdapter
from sglang.multimodal_gen.runtime.models.dits.omnidreams_kvcache import BlockKVCache

device = torch.device("cuda")
dtype = torch.bfloat16

cfg = OmniDreamsDiTConfig()
dit = OmniDreamsDiT(config=cfg, hf_config={})
dit.post_load_weights()
dit = dit.to(device, dtype=dtype).eval()

ext = load_extension()
opt = load_python_module("optimized_dit")
arch = cfg.arch_config

len_t, hp, wp = 2, 44, 80
B, T, HW, D = 1, 2, hp * wp, 2048
num_blocks = 28
head_dim = 128
local_heads = 16
chunk_tokens = T * HW
window_tokens = 6 * HW
sink_tokens = 0
shape = (B, sink_tokens + window_tokens, local_heads, head_dim)

adapter = _SGLTransformerAdapter(dit, arch, len_t, dtype, hp * 8, wp * 8)
ex = opt.OptimizedDiTExecutor(adapter, ext, dit_backend="fp8_kvcache_cudnn",
                              attention_backend="auto", sparge_topk=None)
print("1. Executor ready")

# Init caches for chunk 0
caches = []
for _ in range(num_blocks):
    c = BlockKVCache(k_shape=shape, v_shape=shape, seq_dim=1,
                      chunk_size=chunk_tokens, window_size=window_tokens,
                      sink_size=sink_tokens, device=device, dtype=dtype)
    c.before_update(0)
    # Fill with dummy K/V for chunk 0
    c.update(
        torch.randn(B, chunk_tokens, local_heads, head_dim, device=device, dtype=dtype),
        torch.randn(B, chunk_tokens, local_heads, head_dim, device=device, dtype=dtype)
    )
    c.after_update(0)
    caches.append(c)
print("2. Caches initialized, chunk_idx=0")

cross_kv = [(torch.randn(B, 512, D, device=device, dtype=dtype),
             torch.randn(B, 512, D, device=device, dtype=dtype))
            for _ in range(num_blocks)]

noisy = torch.randn(B, 1, T, HW, D, device=device, dtype=dtype)
mask = torch.zeros(B, 1, T, HW, 4, device=device, dtype=dtype)
hdmap_in = torch.randn(B, 1, T, HW, 16, device=device, dtype=dtype)
timestep = torch.tensor([0.5], device=device).expand(B).contiguous()
rope_freqs = torch.randn(T * HW, 1, 1, 128, device=device, dtype=torch.float32)

inv = ex._ensure_invariant_tensors(ar_idx=0, timesteps=timestep)
rope_cos, rope_sin = ex._ensure_rope_tensors(ar_idx=0, rope_freqs=rope_freqs)
ex._ensure_weights_snapshot()
print("3. Weights & invariants ready")

k_self = [c._k for c in caches]
v_self = [c._v for c in caches]
kc = [kv[0] for kv in cross_kv]
vc = [kv[1] for kv in cross_kv]

rt = ex._ensure_fp8_runtime(k_cross=kc, v_cross=vc, k_self=k_self, v_self=v_self,
                            tokens=T*HW, cache=None)
ex._apply_runtime_config(rt)
print("4. Runtime applied")

ws = opt.compute_self_attn_write_start(caches[0])
print("5. write_start:", ws)

hdmap_emb = ex._ensure_hdmap_tensor(ar_idx=0, input_for_ext=hdmap_in)
hdmap_ext = hdmap_in if hdmap_emb is None else ex._empty_hdmap_tensor(device=device, dtype=dtype)

print("6. Running native kernel...")
try:
    out = ex._predict_flow_ext_impl(
        noisy, mask, hdmap_ext, hdmap_emb, timestep, rope_freqs,
        inv.t_emb, inv.t_emb_silu, inv.adaln_lora,
        inv.final_shift, inv.final_scale,
        rope_cos, rope_sin,
        inv.block_mods_sa, inv.block_mods_ca, inv.block_mods_mlp,
        kc, vc, k_self, v_self, ws,
    )
    print("SUCCESS! shape:", list(out.shape))
except RuntimeError as e:
    msg = str(e)
    print("FAILED:", msg[:200])
