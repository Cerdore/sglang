import torch, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import _SGLTransformerAdapter

device = torch.device("cuda"); dtype = torch.bfloat16
cfg = OmniDreamsDiTConfig()
arch = cfg.arch_config

# STEP 1: DiT after post_load_weights (this is what the pipeline does)
dit = OmniDreamsDiT(config=cfg, hf_config={})
dit.post_load_weights()
dit = dit.to(device, dtype=dtype).eval()

print("After post_load_weights:")
print("  x_embedder weight:", list(dit.x_embedder.proj[1].weight.shape))

# STEP 2: Create adapter (this is what OmniDreamsFP8DiT._ensure_executor does)
adapter = _SGLTransformerAdapter(dit, arch, 2, dtype, 704, 1280)
print("After adapter:")
print("  x_embedder weight:", list(dit.x_embedder.proj[1].weight.shape))
print("  bias:", dit.x_embedder.proj[1].bias)

# STEP 3: Check what state_dict returns
sd = dit.state_dict()
w = sd["x_embedder.proj.1.weight"]
print("state_dict x_embedder:", list(w.shape))
print("  is identity-like?", torch.allclose(w[:2048,:2048], torch.eye(2048, device=device, dtype=dtype), atol=1e-3))
print("  last 4 cols all zero?", (w[:, 2048:].abs().max() < 1e-6).item())
