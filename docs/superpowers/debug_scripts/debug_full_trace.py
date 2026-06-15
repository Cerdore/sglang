import torch, os, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
from sglang.multimodal_gen.native import load_extension
from sglang.multimodal_gen.native.singleview_loader import load_python_module
from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import _SGLTransformerAdapter
from sglang.multimodal_gen.runtime.models.dits.omnidreams_kvcache import BlockKVCache

device = torch.device("cuda"); dtype = torch.bfloat16
cfg = OmniDreamsDiTConfig(); dit = OmniDreamsDiT(config=cfg, hf_config={})
dit.post_load_weights(); dit = dit.to(device, dtype=dtype).eval()
ext = load_extension(); opt = load_python_module("optimized_dit")
arch = cfg.arch_config
len_t, hp, wp = 2, 44, 80
B,T,HW,D = 1,2,3520,2048
num_blocks, shape = 28, (B, 6*HW, 16, 128)

adapter = _SGLTransformerAdapter(dit, arch, len_t, dtype, hp*8, wp*8)
ex = opt.OptimizedDiTExecutor(adapter, ext, dit_backend="bf16", attention_backend="auto")

caches=[]
for _ in range(num_blocks):
    c=BlockKVCache(k_shape=shape, v_shape=shape, seq_dim=1, chunk_size=HW, window_size=6*HW, sink_size=0, device=device, dtype=dtype)
    c.before_update(0); c.update(torch.randn(B, HW, 16, 128, device=device, dtype=dtype), torch.randn(B, HW, 16, 128, device=device, dtype=dtype)); c.after_update(0); c.before_update(1)
    caches.append(c)

ckv=[(torch.randn(B,512,D,device=device,dtype=dtype),torch.randn(B,512,D,device=device,dtype=dtype)) for _ in range(num_blocks)]
noisy=torch.randn(B,1,T,HW,D,device=device,dtype=dtype); mask=torch.zeros(B,1,T,HW,4,device=device,dtype=dtype)
hdmap=torch.randn(B,1,T,HW,64,device=device,dtype=dtype)
timestep=torch.tensor([0.5],device=device).expand(B).contiguous()
rf=torch.randn(T*HW,1,1,128,device=device,dtype=torch.float32)
inv=ex._ensure_invariant_tensors(ar_idx=1,timesteps=timestep); rc,rs=ex._ensure_rope_tensors(ar_idx=1,rope_freqs=rf)
ex._ensure_weights_snapshot()
ks=[c._k for c in caches]; vs=[c._v for c in caches]; kc=[kv[0] for kv in ckv]; vc=[kv[1] for kv in ckv]
rt=ex._ensure_fp8_runtime(k_cross=kc,v_cross=vc,k_self=ks,v_self=vs,tokens=T*HW,cache=None); ex._apply_runtime_config(rt)
ws=opt.compute_self_attn_write_start(caches[0])
he=ex._ensure_hdmap_tensor(ar_idx=1,input_for_ext=hdmap)
hf=hdmap if he is None else ex._empty_hdmap_tensor(device=device,dtype=dtype)

try:
    out=ex._predict_flow_ext_impl(noisy,mask,hf,he,timestep,rf,inv.t_emb,inv.t_emb_silu,inv.adaln_lora,inv.final_shift,inv.final_scale,rc,rs,inv.block_mods_sa,inv.block_mods_ca,inv.block_mods_mlp,kc,vc,ks,vs,ws)
    print("SUCCESS",list(out.shape))
except Exception as e:
    traceback.print_exc()
