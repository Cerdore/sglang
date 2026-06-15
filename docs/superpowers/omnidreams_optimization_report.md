# OmniDreams SGLang 部署优化调研报告

> 调研日期: 2026-06-14
> 论文: [NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation](https://arxiv.org/abs/2606.03159) (arXiv:2606.03159v1, 39 pages)
> 参考代码: [NVIDIA/flashdreams](https://github.com/NVIDIA/flashdreams), SGLang main `multimodal_gen/`

---

## 1. 核心概念澄清

### 1.1 DiT 是 Cosmos，不是 WAN

OmniDreams 的 DiT backbone 是 **Cosmos-Predict 2.5**（NVIDIA 视频基础模型平台），从头到尾与 WAN 无关。WAN 仅出现在 VAE 层面（Wan 2.1 VAE 的编码器/解码器），以及 FlashDreams 验证过的其他 backbone（Lingbot-World 基于 Wan2.1-14B）。

论文引用: "OmniDreams-SV and OmniDreams-MV post-train from Cosmos-Predict 2.5."

### 1.2 "Light WAN" 指的是 LightVAE，不是 LightWAN DiT

FlashDreams 中提到的 "light" 组件:
- **LightVAE**: `lightvaew2_1.pth` — Wan 2.1 VAE 编码器做 75% channel pruning（`pruning_rate=0.75`），来自 `lightx2v/Autoencoders`
- **LightTAE**: TAEHV (Tiny AutoEncoder for Hunyuan Video) 独立训练的轻量时序解码器，用 2D conv + MemBlock 1-frame temporal memory 替代 Wan VAE 的 3D causal conv
- DiT 本身无剪枝，始终是 2B Cosmos DiT

---

## 2. 模型架构详解

### 2.1 DiT (Cosmos DiT)

| 参数 | 值 |
|------|-----|
| 参数量 | ~2B |
| model_channels | 2048 |
| num_blocks | 28 |
| num_heads | 16 |
| head_dim | 128 |
| mlp_ratio | 4.0 |
| crossattn_emb_channels | 1024 |
| crossattn_proj_in_channels | 100352 (Cosmos-Reason1 full_concat) |
| patch_spatial | 2 |
| patch_temporal | 1 |
| in_channels | 16 |
| out_channels | 16 |
| use_adaln_lora | True |
| adaln_lora_dim | 256 |
| concat_padding_mask | True (推理时 fused 掉) |
| 精度 | bf16 |

每层 DiT Block 包含: Self-Attention (causal) → (可选) Cross-View Attention → Cross-Attention (text) → MLP。
所有 sublayer 通过 AdaLN 接受 time+view embedding 调制。

### 2.2 文本编码器: Cosmos-Reason1-7B

- **模型**: `nvidia/Cosmos-Reason1-7B` (Qwen2.5-VL 基础)
- **固定 revision**: `3210bec0495fdc7a8d3dbb8d58da5711eab4b423`
- **输出**: 28 层 × 3584 hidden = 100352 维 `full_concat` embedding
- **关键细节**: 不使用 attention_mask（训练时就无 mask），padding token 的 hidden state 也是 conditioning 的一部分
- **显存**: ~14GB (bf16)
- **不可替换**: DiT 的 `crossattn_proj_in_channels=100352` 是在这个特定 embedding 分布上训练的

### 2.3 VAE: 三种配置

| 配置 | 编码器 | 解码器 | 编码延迟 | 解码延迟 | 质量(FVD) |
|------|--------|--------|---------|---------|-----------|
| **Full Wan VAE** (SGLang main) | Wan 2.1 VAE (全量16ch) | Wan 2.1 VAE | 基准 | 基准 | 24.8 |
| **LightVAE + LightTAE** (FlashDreams 默认) | lightvaew2_1.pth (75%剪枝) | TAEHV (独立轻量) | ↓2-3x | ↓3-5x | 45.4 |
| **LightVAE FP8 Native** (FlashDreams 极致) | LightVAE FP8 CUDA kernel | LightTAE | 最快 | 快 | ~45 |

### 2.4 调度器: 2-step Flow Match

- **蒸馏方法**: Self Forcing + DMD (Distribution Matching Distillation)
- **去噪步数**: K=2，schedule [1000, 450]
- **flow_shift**: 5.0
- **sigma_min**: 0.0
- **CFG**: 关闭 (guidance_scale=1.0)
- **warp**: warped timestep `shift*t / (1+(shift-1)*t)`

### 2.5 AR Rollout 参数

| 参数 | OmniDreams-SV | OmniDreams-MV |
|------|---------------|---------------|
| len_t (每chunk latent帧) | 2 | 4 |
| 每chunk pixel帧 | 8 | 16 |
| window_size_t (latent帧) | 6 | 8 |
| sink_size_t | 0 | 0 |
| context_noise | 128 | - |
| h_extrapolation_ratio | 3.0 | - |
| w_extrapolation_ratio | 3.0 | - |

---

## 3. FlashDreams vs SGLang main 详细对比

### 3.1 组件矩阵

```
组件              FlashDreams                         SGLang main
─────────────────────────────────────────────────────────────────────
DiT               CosmosDiTNetwork                    OmniDreamsDiT (忠实移植)
Text Encoder      Cosmos-Reason1-7B ✅               Cosmos-Reason1-7B ✅
VAE Encoder       LightVAE (75%剪枝)                  Wan 2.1 全量 VAE
VAE Decoder       LightTAE (TAEHV)                    Wan 2.1 全量 VAE
Scheduler         2-step flow-match ✅                2-step flow-match ✅
HDMap 编码        LightVAE 或 PixelShuffle            Wan VAE (encode)
KV Cache          window=6, sink=0 ✅                 window=6, sink=0 ✅
Context Noise     128 ✅                               128 ✅
RoPE              3D NeoX (44:42:42) ✅               3D NeoX (44:42:42) ✅
Cross-Attn Caching ✅ (precompute once)               ✅ (precompute once)
CUDA Graph        ✅ DiT + VAE forward                ❌ 未实现
torch.compile     ✅ OmniDreamsBlock                  代码已有但未启用
Native FP8 VAE    ✅ (LightVAE FP8 CUDA kernels)      ❌ 未实现
Context Parallel  ✅ V×T×HW hierarchical              ❌ 有守卫禁止
Multi-View        ✅ (4-camera cross-view attn)        ❌ cross_view_attn 是 stub
Component Offload ✅ (CPU offload encoders)           ✅ (_component_residency_manager)
```

### 3.2 关键差异分析

**SGLang main 的优势**:
- 全量 Wan VAE → 更好质量 (FVD 24.8 vs 45.4)
- 代码结构清晰，与 SGLang 基础设施深度集成 (ComponentResidencyManager, FSDP, TP)
- 完善的测试覆盖 (`test_omnidreams_components.py`, `test_omnidreams_regression.py`)

**FlashDreams 的优势**:
- CUDA Graph + torch.compile → 显著更低延迟
- LightVAE/LightTAE → 更低编码/解码延迟和显存
- Native FP8 CUDA kernels → 极致推理速度
- Context Parallel → 多GPU扩展
- 生产验证: 68 FPS @720p on 1×GB300, 105 FPS @720p on 16×GB300

### 3.3 DiT 实现一致性问题（SGLang main 待修复）

| 问题 | FlashDreams (正确) | SGLang main (当前) | 影响 |
|------|-------------------|---------------------|------|
| attention_mask | 不传 mask | **已修复** (commit 4065babc) | 无 mask 时 padding token 参与 attention，否则画面 blur |
| clean frame re-inject | KV cache 写 clean chunk | **已修复** (commit 0e30f129) | 不 reinject 会导致 AR 误差累积 |
| padding mask fuse | post_load 中 fuse | ✅ 正确 | 72→68 channels |
| last-layer shuffle | post_load 中 fuse | ✅ 正确 | Cosmos channel order |
| cross_view_attn | 完整实现 | NotImplementedError stub | MV 不可用 |

---

## 4. 性能数据 (论文 + 代码)

### 4.1 论文基准 (GB300)

| 配置 | GPU | FPS | ms/chunk |
|------|-----|-----|----------|
| OmniDreams-SV (单视图) | 1×GB300 | 68 | 118 |
| OmniDreams-MV (4视图) | 16×GB300 | 105/camera | 151 |
| MV 扩展 1GPU | 1×GB300 | 12 | 1289 |
| MV 扩展 4GPU | 4×GB300 | 48 | 330 |
| MV 扩展 8GPU | 8×GB300 | 74 | 209 |

### 4.2 FlashDreams 优化收益 (论文)

| 优化 | 延迟变化 |
|------|---------|
| Wan2.1 Self Forcing (FlashDreams vs 官方) | 1.95× speedup |
| Lingbot-World on 4×H100 (FlashDreams vs 官方) | 2.49× speedup |
| 2-step 蒸馏 (vs 35-step UniPC teacher) | ~17× fewer denoising steps |

### 4.3 质量-速度权衡 (论文 Table 5)

| VAE Decoder | FVD ↓ |
|-------------|-------|
| 原版 VAE | 24.8 |
| LightTAE | 45.4 |

---

## 5. 优化路径与优先级

### 5.1 总体路线图

```
当前 ──P0──▶ ──P3──▶ ──P1──▶ ──P2──▶ [基准] ──P4/P5──▶
SGLang  CUDA    compile  LightTAE LightVAE          Native FP8
main    Graph            解码器    编码器            + CP + MV
```

### 5.2 P0: DiT CUDA Graph 支持 🔴 最高优先级

**预期收益**: 每 chunk 延迟 ↓30-50%  
**复杂度**: 中等  
**风险**: KV cache 动态 shape 需要处理  

**原因**: 当前代码有显式守卫阻止 compile:
- `OmniDreamsDiT.forward()`: `assert not torch.compiler.is_compiling()`
- `OmniDreamsDenoisingStage.forward()`: 整个 chunk loop 是 eager 模式
- FlashDreams 的 `eager_mode=True/False` 分离了 graph 生命周期和 forward 逻辑

**改动文件**:
1. `runtime/models/dits/omnidreams.py`
   - OmniDreamsBlock.forward() → 可作为 graph 单元
2. `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py`
   - OmniDreamsDenoisingStage.forward()
   - 将 `before_update`/`after_update` 移到 graph 外部
   - 仅 `predict_flow` 内部和 `ctx_latent` re-forward 在 graph 内

**实现思路**:
```python
# 伪代码: 分离 graph 生命周期
for chunk_idx in range(num_chunks):
    for c in caches: c.before_update(chunk_idx)   # 在 graph 外
    
    # CUDA Graph 捕获区域: predict_flow + clean re-forward
    clean = graph_captured_denoise(noise, ...)
    ctx_latent = graph_captured_reforward(clean, ...)
    
    for c in caches: c.after_update(chunk_idx)    # 在 graph 外
```

### 5.3 P3: torch.compile 启用 🟡 高优先级

**预期收益**: 额外 ↓10-15%  
**复杂度**: 低  
**风险**: Inductor autotune 可能增加 warmup 时间  

**改动文件**:
1. `runtime/pipelines/omnidreams_pipeline.py`
   - `load_modules()` 后对符合条件的模块调用 `torch.compile()`
   - 已有 `_compile_conditions = [lambda n, m: isinstance(m, OmniDreamsBlock)]`
2. 验证 `mode="max-autotune-no-cudagraphs"` 与 CUDA Graph 兼容性

### 5.4 P1: LightTAE (TAEHV) 解码器移植 🟡 高优先级

**预期收益**: 解码延迟 ↓3-5x，解码显存 ↓  
**复杂度**: 中等  
**风险**: Checkpoint key remap, streaming state 管理  

**TAEHV 架构要点**:
- 2D conv stack，无 3D causal convolution
- MemBlock: 1-frame temporal memory slot，用 `id(module)` 做 key
- TGrow: channel-expand + reshape 实现 temporal upsampling
- frames_to_trim: `2^sum(decoder_time_upscale) - 1` 帧（首 chunk 修剪）
- 输入: `[B, 16, T, H/8, W/8]` Wan 2.1 latent
- 输出: `[B, 3, T*4, H, W]` RGB pixels

**改动文件**:
1. **新建** `runtime/models/vaes/taehv.py`
   - TAEHV 类: 加载 checkpoint, 实现 `decode(z, cache) → pixels`
   - TAEHVCache: streaming state 容器
   - Decoder 主体: MemBlock + TGrow + Clamp
   - Checkpoint key remap: `decoder.<i>.* → decoder.blocks.<i>.*`
   - Oversize TGrow weight 截断
2. **新建** `runtime/models/vaes/omnidreams_light_vae.py`
   - `OmniDreamsLightVAE(ParallelTiledVAE)` 包装类
   - `encode()` → 转发到 Wan VAE 编码器
   - `decode()` → 转发到 TAEHV
   - `use_feature_cache = True` (OmniDreamsDenoisingStage 要求)
3. **修改** `runtime/pipelines/omnidreams_pipeline.py`
   - `load_modules()`: 新增 TAEHV checkpoint 加载路径
   - `create_pipeline_stages()`: VAE 切换逻辑
4. **修改** `configs/pipeline_configs/omnidreams.py`
   - 新增 `use_light_tae: bool = False`
   - 新增 `light_tae_checkpoint_path: str | None = None`

### 5.5 P2: LightVAE 编码器支持 🟢 中优先级

**预期收益**: HDMap/reference image 编码延迟 ↓2-3x  
**复杂度**: 低-中  
**风险**: lightvaew2_1.pth 是 flat `.pth` 格式，需要 key remap  

**改动文件**:
1. `runtime/pipelines/omnidreams_pipeline.py`
   - `_resolve_vae_path()`: 支持 `lightvaew2_1.pth` 路径
   - `_load_wan_vae()`: 加载时识别 LightVAE checkpoint
   - 可能需要 `wan_repack.py` 工具先做格式转换

### 5.6 P4: Native FP8 LightVAE 🟢 长期

**预期收益**: 终极编码延迟  
**复杂度**: 高  
**前置条件**: P1+P2 完成，有校准好的 FP8 state  

**来源**: flashdreams `omnidreams_singleview/`:
- `src/vae_streaming/` — CUDA kernels for FP8 VAE
- `src/dit_streaming/` — CUDA kernels for DiT
- 需要 `OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH` 环境变量

### 5.7 P5: 多视图 + Context Parallel 🟢 长期

**预期收益**: 4-camera 实时生成，多GPU线性扩展  
**复杂度**: 高  

**论文方案**: 三层 hierarchical CP:
```
CP degree = V × T × HW  (e.g., 4 × 4 × 1 = 16 GPUs)
```
- V=4: view axis (跨相机并行)
- T=4: temporal axis (跨时间并行)
- HW=1: spatial axis (不分割空间)

**改动文件**:
1. `runtime/models/dits/omnidreams.py`
   - 实现 `_cross_view_attn_forward()` (当前是 `NotImplementedError`)
   - Cross-view attention: 每个时间步跨所有 view 做 bidirectional attention
2. `runtime/distributed/` 或 `layers/attention/`
   - Ring attention for CP
3. `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py`
   - 移除 `assert get_sp_world_size() <= 1` 守卫
   - `_postprocess_sp_latents()` 实现 gather

---

## 6. 部署清单

### 6.1 必需模型

| 模型 | HuggingFace | 大小 | 用途 |
|------|------------|------|------|
| Cosmos DiT | `nvidia/omni-dreams-models` → `single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt` | ~4GB | DiT backbone |
| Cosmos-Reason1-7B | `nvidia/Cosmos-Reason1-7B` | ~15GB | 文本编码器 |
| Wan 2.1 VAE (diffusers) | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae/` 或任意 Wan2.1 diffusers VAE | ~1GB | VAE 编码+解码 |

### 6.2 可选加速模型

| 模型 | 来源 | 大小 | 用途 |
|------|------|------|------|
| LightVAE | `lightx2v/Autoencoders/lightvaew2_1.pth` | ~250MB | 加速 VAE 编码 |
| LightTAE | lightx2v TAEHV checkpoint | ~500MB | 加速 VAE 解码 |
| LightVAE FP8 State | 需自行校准 | - | Native FP8 编码器 |

### 6.3 目录结构建议

```
<sglang_model_path>/
├── single_view/
│   └── 2b_res720p_30fps_i2v_hdmap_distilled.pt    # DiT checkpoint
├── text_encoder/
│   ├── config.json                                   # Cosmos-Reason1-7B
│   ├── model*.safetensors
│   └── ...
├── vae/
│   ├── config.json                                   # Wan 2.1 VAE diffusers 格式
│   └── diffusion_pytorch_model*.safetensors
├── [可选] lightvae/
│   └── lightvaew2_1.pth
└── [可选] lighttae/
    └── taehv_checkpoint.pth
```

---

## 7. 风险与注意事项

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| CUDA Graph + KV cache dynamic shape | 高 | 用 static-shape cache，window ops 在 graph 外 |
| LightTAE 质量损失 (FVD=45.4 vs 24.8) | 中 | 提供配置开关，默认保持全量 VAE |
| torch.compile cold start 延迟 | 中 | 首次 chunk 用 eager，后续重放 graph |
| lightvaew2_1.pth 格式兼容性 | 低 | 写 key remap 层或先用 diffusers convert |
| SP/CP 不可用 | 低 | 保持守卫，明确错误信息 |
| Cosmos-Reason1-7B 显存占用 (~14GB) | 中 | 已支持 `--text-encoder-cpu-offload` |

---

## 8. 参考文献

- [NVIDIA OmniDreams paper (arXiv:2606.03159)](https://arxiv.org/abs/2606.03159)
- [nvidia/omni-dreams-models (HuggingFace)](https://huggingface.co/nvidia/omni-dreams-models)
- [nvidia/Cosmos-Reason1-7B (HuggingFace)](https://huggingface.co/nvidia/Cosmos-Reason1-7B)
- [NVIDIA/flashdreams (GitHub)](https://github.com/NVIDIA/flashdreams)
- [lightx2v/Autoencoders (HuggingFace)](https://huggingface.co/lightx2v/Autoencoders)
- [ModelTC/lightx2v (GitHub)](https://github.com/ModelTC/lightx2v)
- SGLang: `python/sglang/multimodal_gen/runtime/pipelines/omnidreams_pipeline.py`
- SGLang: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/omnidreams.py`
- SGLang: `python/sglang/multimodal_gen/runtime/models/dits/omnidreams.py`
- FlashDreams: `integrations/omnidreams/omnidreams/config.py`
- FlashDreams: `integrations/omnidreams/omnidreams/transformer/impl/network.py`
- FlashDreams: `flashdreams/recipes/wan/autoencoder/vae.py`
