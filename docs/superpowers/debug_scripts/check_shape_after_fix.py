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
dit.post_load_weights()
dit = dit.to(device, dtype=dtype).eval()
ext = load_extension(); opt = load_python_module("optimized_dit")
arch = cfg.arch_config
adapter = _SGLTransformerAdapter(dit, arch, 2, dtype, 704, 1280)

# Check x_embedder after adapter
xe_w = dit.x_embedder.proj[1].weight
print("x_embedder weight:", list(xe_w.shape))
print("x_embedder in_channels:", dit.x_embedder.in_channels)

ex = opt.OptimizedDiTExecutor(adapter, ext, dit_backend="fp8_kvcache_cudnn", attention_backend="auto")
B,T,HW=1,2,3520
timestep=torch.tensor([0.5],device=device).expand(B).contiguous()
ex._ensure_invariant_tensors(ar_idx=0,timesteps=timestep)
ex._ensure_weights_snapshot()
w=ex._optimized_weights
print()
print("x_embedder in optimized weights:")
xe_k = "x_embedder.proj.1.weight"
if xe_k in w: print(" ", xe_k, list(w[xe_k].shape))
fp8_k = xe_k + "_fp8_prepared"
if fp8_k in w: print(" ", fp8_k, list(w[fp8_k].shape))
scale_k = xe_k + "_fp8_prepared_scale"
if scale_k in w: print(" ", scale_k, list(w[scale_k].shape))
