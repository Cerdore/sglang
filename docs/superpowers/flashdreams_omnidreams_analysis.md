# FlashDreams 对 OmniDreams 的实现分析报告

## 1. 概述

**FlashDreams** 是 NVIDIA 为 **OmniDreams**（基于 Cosmos-Predict 2.5 的动作条件生成式世界模型，用于闭环自动驾驶仿真）提供的开源推理与 serving 基础设施。FlashDreams 专注于**推理端**（高效流式去噪、serving、交互式 demo），而训练流程位于独立的内部代码库中。

两者分工：

- **论文中描述的训练流程**（Diffusion Forcing 中期训练、Self Forcing 蒸馏、DMD 分布匹配、渐进式长上下文教师）—— **不在 FlashDreams 仓库中**
- **FlashDreams 中的推理基础设施**—— **完整实现**了流式自回归 DiT、KV Cache、CUDA Graph、上下文并行、VAE 编解码、serving 端点及交互式 demo

---

## 2. 架构对齐：论文声明 vs FlashDreams 实现

### 2.1 流式自回归 DiT（论文第 3、5 节）

| 论文特性 | FlashDreams 实现 | 状态 |
|-----------|---------------------|------|
| **因果扩散公式** | `CosmosDiTNetwork` in `transformer/impl/network.py:129` —— 逐块、KV Cache 自回归前向传播 | ✅ 完整实现 |
| **流式 KV Cache** | `BlockKVCache` in `core/attention/kvcache.py:26` —— 固定大小局部窗口，CUDA Graph 兼容，逐块写入，支持 sink token | ✅ 完整实现 |
| **局部窗口时间注意力** | `window_size_t` 配置参数（单视图 = 6，带 Pixel Shuffle 的四视图 = 8）。滑动窗口通过 `BlockKVCache._roll_local_window_left()` 实现 | ✅ 完整实现 |
| **Causal Transformer 骨干** | 28 层 Cosmos DiT，`model_channels=2048`，`num_heads=16`（`modules.py:477`）—— AdaLN 调制 + LoRA 分解，QK RMSNorm | ✅ 完整实现 |
| **torch.compile** | `compile_module()` in `infra/compile.py:44`，默认模式 `max-autotune-no-cudagraphs`（供 `CUDAGraphWrapper` 内部使用） | ✅ 完整实现 |
| **CUDA Graph** | `CUDAGraphWrapper` in `infra/cuda_graph.py:50` —— warmup → capture → replay（稳态），在首个完整 Cache 窗口后自动触发 | ✅ 完整实现 |
| **2-step 蒸馏 Scheduler** | `FlowMatchSchedulerConfig(num_inference_steps=2, denoising_timesteps=[1000, 450])` in `config.py:126` —— 与论文 K=2 蒸馏一致 | ✅ 完整实现 |
| **35-step UniPC 教师** | `FlowMatchUniPCSchedulerConfig(num_inference_steps=35)` in `config.py:303` —— 与 Diffusion Forcing 教师配置一致 | ✅ 完整实现 |

### 2.2 多 GPU / 上下文并行（论文第 5.2 节）

| 论文特性 | FlashDreams 实现 | 状态 |
|-----------|---------------------|------|
| **分层 V→T→HW CP** | `create_hierarchical_cp_groups()` in `transformer/impl/context_parallel.py:62` —— 完全按论文分组模式实现：V 组、T 组、THW 组、HW 组、VHW 组 | ✅ 完整实现 |
| **Ring Attention** | `ContextParallelAttention` in `core/attention/cp.py:61` 支持 `ring` 和 `ulysses` 两种方法；使用 `funcol.all_gather_tensor` + LSE merge 实现环形 KV 收集 | ✅ 完整实现 |
| **Flash / cuDNN SDPA 后端** | `torch_sdpa_cudnn()` 和 `torch_sdpa_flash()` in `cp.py:39-58` —— 可配置选择 | ✅ 完整实现 |
| **单视图：4-GPU 100 FPS** | 论文 Table 2：80ms/chunk，100 有效 FPS。可通过 `torchrun --nproc_per_node=4` 配置 | ✅ 可配置 |
| **四视图：16-GPU 105 FPS** | 论文 Table 3：151ms/chunk，每摄像头 105 有效 FPS。多视图配置已存在（`MV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE`） | ✅ 可配置 |
| **独立线程 KV Cache 更新** | 论文声称 KV Cache 更新不在热路径上；当前 `BlockKVCache` 在主线程同步执行 | ⚠️ 简化实现 |

### 2.3 VAE / Autoencoder（论文第 5.1 节）

| 论文特性 | FlashDreams 实现 | 状态 |
|-----------|---------------------|------|
| **LightVAE（编码器）** | `OmnidreamsWanVAEEncoder` in `vae_native.py:408` —— 支持原生 FP8 和 PyTorch 双路径，通过 `native_vae_backend="fp8"` 配置 | ✅ 完整实现 |
| **LightTAE（解码器）** | `TeahvVAEDecoder` from `flashdreams.recipes.taehv`，用于 `config.py:100-101` | ✅ 完整实现 |
| **Wan VAE（编码器 + 解码器）** | `SV_2STEPS_CHUNK2_LOC6_VAE_VAE` 配置使用 —— 双向均使用原始 Wan VAE | ✅ 完整实现 |
| **Pixel Shuffle VAE（HDMap 条件编码）** | `PixelShuffleVAEEncoderConfig` in `encoder/pixel_shuffle.py` —— chunk4 变体使用（`additional_concat_ch=192`） | ✅ 完整实现 |
| **解码延迟 vs 质量权衡** | 论文 Table 5：LightTAE FVD 从 24.8→45.4 vs 原始 VAE。两个选项均可用 | ✅ 可配置 |

### 2.4 条件信号（论文第 2、3、4 节）

| 论文特性 | FlashDreams 实现 | 状态 |
|-----------|---------------------|------|
| **Text Prompt（Cosmos-Reason1 7B）** | `CosmosReason1TextEncoder` — 每 session 运行一次，所有 view 共享 | ✅ 完整实现 |
| **First-frame RGB（I2V）** | `OmnidreamsPipeline.initialize_cache(image=first_frames)` — AR step 0 时通过 `_maybe_inject_image()` 注入 latent space | ✅ 完整实现 |
| **HD Map 条件** | 每 AR step 通过 `hdmap` 参数传入 `generate()` — 由 `additional_patch_embedding` 编码后与 noise latent 拼接。两分支：Wan VAE（16ch）和 Pixel Shuffle（192ch） | ✅ 完整实现 |
| **抽象世界场景（BBox + 地图）** | 转换为 HD Map 视频帧。完整场景加载器在 `conditioning/world_scenario/` 目录下，含 `clipgt_loader.py`、`data_types.py`、`ftheta.py` 处理车道线、点云、BBox | ✅ 完整实现 |
| **View-specific Embedding** | `CosmosDiTNetwork` 中的 `adaln_view_embedder` 和 `adaln_view_proj` — 7 摄像头视角查找表，向调制信号添加逐视图 AdaLN 偏移 | ✅ 完整实现 |

### 2.5 多视图生成（论文第 3.2 节）

| 论文特性 | FlashDreams 实现 | 状态 |
|-----------|---------------------|------|
| **时间注意力（逐视图）** | `SelfAttention` 每个 view 内独立运行，使用因果 KV Cache — view 在注意力前通过 flatten 为 `[BV, L, D]` | ✅ 完整实现 |
| **跨视图注意力** | `Block.forward()` 中的 `cross_view_attn` in `modules.py:712-734` — 密集注意力：reshape 为 `[B, T, V, HW, D]`，repeat context，所有 view 互相关注。CP 激活时跳过 repeat | ✅ 完整实现 |
| **跨视图 AdaLN 调制** | `view_embedding_proj` 拆分为 9 个 chunk（自注意力的 shift/scale/gate、交叉注意力的 shift/scale/gate、MLP 的 shift/scale/gate）—— 加到每个 block 的调制信号上 | ✅ 完整实现 |
| **复杂度 O(NT² + N²)** | 时间注意力：逐视图独立（无跨视图开销）。跨视图：仅每时间步密集注意力（无时间扩展）。精确匹配论文公式 | ✅ 数学对齐 |

### 2.6 自回归生成流程

FlashDreams 中的 `StreamInferencePipeline`（`infra/pipeline/base.py:111`）完整实现了论文中的因果逐块生成：

```
for each AR step i:
    1. 编码 HDMap 块 → latent space
    2. DiffusionModel.generate():
       a. 从 RNG 采样初始噪声
       b. Scheduler 运行 K=2 步去噪（每步调用 predict_flow）
       c. 每步 Transformer 前向传播：
          - cache.start(chunk_idx)  → 若已填满则滚动 KV 窗口
          - 网络前向（28 block：自注意力 → 跨视图注意力 → 交叉注意力 → MLP）
          - cache.finalize(chunk_idx) → 推进计数器
       d. 后处理：AR step 0 时注入 first-frame 图像
       e. unpatchify + decode → RGB
    3. DiffusionModel.finalize(): 用 context noise 额外运行一次前向传播推进 KV Cache
    4. 用 CUDAGraphWrapper 包裹（填充阶段 drain，稳态下 capture/replay）
```

### 2.7 训练机制 —— **不在仓库中**

| 论文特性 | FlashDreams 中的状态 |
|-----------|----------------------|
| **Diffusion Forcing 中期训练** | ❌ 未实现（仅推理） |
| **Self Forcing 蒸馏（DMD）** | ❌ 未实现（仅推理）。推理*消费*蒸馏后的 2-step checkpoint |
| **渐进式长上下文教师** | ❌ 未实现（仅推理） |
| **World-Action Model（WAM）后期训练** | ❌ 未实现（仅推理） |
| **扩散 Inpainter 后期训练** | ❌ 未实现（仅推理） |
| **OOD 物体建模（动态长方体丢弃）** | ❌ 未实现（仅推理） |
| **Rectified Flow 目标（速度预测）** | ❌ 未实现（仅推理）。推理 Scheduler 使用相同的 Flow Match 公式 |

---

## 3. OmniDreams-SV vs OmniDreams-MV 推理对比

### 3.1 是否为两个独立模型

**是的，是两个独立的模型**，使用不同的 checkpoint 文件：

```python
# config.py — 7 个 checkpoint 路径，按视图数和编码器类型区分
AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS = {
    "1view-vae-chunk2":      "...single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt",
    "1view-pshuffle-chunk4": "MISSING",
    "1view-vae-chunk3":      "MISSING",
    "4view-pshuffle-chunk4": "MISSING",       # ← 四视图独立 checkpoint
    "4view-vae-chunk4":      "MISSING",
    "1view-diffusion-forcing-chunk2": "MISSING",
    "1view-bidirectional-chunk48":    "MISSING",
}
```

当前公开可用的只有 `1view-vae-chunk2`（蒸馏后的单视图模型），四视图模型仍标注为 `MISSING`。

### 3.2 架构差异

| 维度 | OmniDreams-SV | OmniDreams-MV |
|------|---------------|---------------|
| **视图数** | 1 | 4（前广角、前望远、左交叉、右交叉） |
| **每 AR 步帧数** | 8 RGB（chunk2）或 16（chunk4） | 16 RGB × 4 views = 64 frames 总计 |
| **潜在帧数** | `len_t=2`（chunk2）或 `len_t=4`（chunk4） | `len_t=4` |
| **跨视图注意力** | `enable_cross_view_attn=False` | `enable_cross_view_attn=True` |
| **View AdaLN 调制** | 无（`view_indices=None`） | 有（7 摄像头视角 embedding table + 9-dim 投影） |
| **HDMap 编码** | Wan VAE（16ch）或 Pixel Shuffle（192ch） | Pixel Shuffle（192ch） |
| **上下文并行维度** | 仅 T 和 HW | **V、T、HW 三个维度** |
| **GPU 需求** | 单 GPU 可达实时（68 FPS @ GB300） | 需 16 GPU 达实时（105 FPS/camera @ GB300） |

### 3.3 跨视图注意力机制（MV 独有）

MV 最关键的结构差异在 `modules.py:712-734`：

```python
# 跨视图注意力仅在 enable_cross_view_attn=True 时执行
if self.enable_cross_view_attn:
    # 将 [B, V, T*HW, D] 重塑为 [B, T, V, HW, D]
    x_cv = rearrange(normed_x_cv, "b v (t hw) d -> b t v hw d", t=T, hw=HW)

    if self.cross_view_attn.is_context_parallel_enabled():
        # CP 启用时：视图已按 rank 顺序分布
        if V == 1:  # CP size == num_views，ring attention 自动收集所有 K/V
            x_context = x_cv
        else:       # CP size < num_views，先收集本地 views
            x_context = repeat(x_cv, "b t v hw d -> b t v2 (v hw) d", v2=V)
    else:
        # 无 CP：重复 context 让每个 view 关注所有 view
        x_context = repeat(x_cv, "b t v hw d -> b t v2 (v hw) d", v2=V)

    # 密集跨视图注意力（每步新建 KV，不缓存）
    cross_view_attn_kv_cache = self.cross_view_attn.compute_kv(x_context)
    cv_out = self.cross_view_attn(x_cv, kv_cache=cross_view_attn_kv_cache)
    cv_out = rearrange(cv_out, "b t v hw d -> b v (t hw) d")
    x = x + cv_out  # 残差连接
```

关键点：**跨视图注意力不缓存 KV**，每时间步重新计算，符合论文 `O(N²)` 跨视图部分的设计。

### 3.4 多视图配置入口

```python
# config.py:248-263 — MV 配置 = SV chunk4 的派生，翻转关键开关
MV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE = derive_config(
    SV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE,
    diffusion_model=dict(
        transformer=dict(
            network=dict(enable_cross_view_attn=True),    # ← 核心区别
            checkpoint_path=CHECKPOINTS["4view-pshuffle-chunk4"],  # ← 不同 checkpoint
            num_views=4,
        ),
    ),
)
```

### 3.5 分层上下文并行（MV 关键基础设施）

`create_hierarchical_cp_groups()` 按照 V→T→HW 优先级分解 world_size：

```
单视图（world_size=4, V=1, T=4）:
  V=1  → V_group=None (singleton)
  T=4  → 4 GPU 分片时间维度
  HW=1 → HW_group=None (singleton)
  CP 仅作用于 T 维度（ring attention 收集时间 KV）

多视图（world_size=16, V=4, T=4, HW=1）:
  V=4  → 4 GPU 分片视图维度（每 GPU 1 view）
  T=4  → 4 GPU 分片时间维度（同一 view 内）
  HW=1 → HW_group=None
  每 GPU 处理: 1 view × 1 时间块 = 1/16 总计算量
```

**CP 分组语义**：

- **V 组**：相同 T,HW → 用于跨视图注意力（跨 V 维度收集）
- **T 组**：相同 V,HW → 用于时间自注意力（跨 T 维度收集）
- **THW 组**：相同 V → 用于自注意力（跨 T 和 HW 收集）
- **HW 组**：相同 V,T → 用于空间注意力

### 3.6 管道中的 View 处理

`OmnidreamsPipeline` 在 `pipeline.py:162-166` 将 view 拆分工作移到 DiT 外部，确保 VAE 编码器不会在 4 个 view 上做重复工作：

```python
self.V_group = transformer.cp_groups.V_group
self.V_size = transformer.cp_groups.V_size
transformer.cp_groups.V_group = None  # ← 从 DiT 移除 V 组
transformer.config.num_views //= self.V_size  # 每 rank 见到的 view 数减少
```

---

## 4. FlashDreams 超论文范围的附加功能

### 4.1 Serving 基础设施

| 功能 | 文件 | 说明 |
|---------|------|------|
| **WebRTC Server** | `omnidreams/webrtc/server.py` | 实时流式 server，带 Web UI；浏览器连接即可接收生成帧 |
| **gRPC Server** | `omnidreams/grpc/server.py` | 实现 `WorldModelService` protobuf；客户端提供 HDMap、相机参数、动态 actor 状态 |
| **Session 管理** | `omnidreams/webrtc/session.py` | 每 session 维护 KV Cache、模拟器状态、生成历史 |
| **Session 录制** | `omnidreams/grpc/session_recorder.py` | 录制可回放 session 日志用于调试/回放 |
| **Profiling Server** | `omnidreams/grpc/profiling_server.py` | 每步时序、GPU 内存指标收集 |

### 4.2 交互式驾驶桌面 Demo

| 功能 | 文件 | 说明 |
|---------|------|------|
| **基于 Raster 后端的桌面 App** | `interactive_drive/` 包 | 全屏 OpenGL 渲染器（`backends/raster.py`），并排显示条件/生成内容 |
| **方向盘/手柄支持** | `interactive_drive/input/` | 物理输入设备集成，含方向盘校准向导（Tkinter UI） |
| **World Model 后端适配器** | `world_model/flashdreams_adapter.py` | 将 interactive-drive 的 chunking 桥接到 `OmnidreamsPipeline`，支持延迟 GPU→host 帧传输 |
| **场景加载（USDZ）** | `interactive_drive/scene_loader.py`, `scenes.py` | 从 USDZ 存档加载合成场景，可从 HuggingFace（`nvidia/omni-dreams-scenes`）拉取 |
| **Ludus Renderer** | `ludus-renderer/` | 可选的 Vulkan-backed GPU 渲染器，用于 HDMap 可视化 |

### 4.3 原生 CUDA 加速

| 功能 | 文件 | 说明 |
|---------|------|------|
| **原生 DiT 加速器** | `native/omnidreams_singleview.py` | 优化的 CUDA 扩展，支持 `fp8_kvcache_cudnn` 和 `bf16` 后端 |
| **原生 VAE 加速器（FP8 LightVAE）** | `vae_native.py` | FP8 量化的 LightVAE 编码器，含校准状态加载 |
| **原生扩展策略系统** | `native/acceleration.py` | `disabled`/`auto`/`required` 三模式策略，带符号可用性检查 |
| **Tensor 准备原语** | `native/primitives.py` | 带类型的张量验证和连续性 layout 强制 |

### 4.4 配置系统

FlashDreams 提供了 **14 个预构建 pipeline 配置** 和 **15 个 runner 配置**：

| 配置名 | Views | Chunk | 条件编码 | 解码 | Steps | Window | 特性 |
|------------|------|-------|----------|---------|-------|--------|--------------|
| `sv-2steps-chunk2-loc6-lightvae-lighttae` | 1 | 2 | LightVAE | LightTAE | 2 | 6 | 基础 chunk2 |
| `*-perf` | 1 | 2 | LightVAE（compiled） | LightTAE（compiled） | 2 | 6 | torch.compile + CUDA Graph |
| `*-native-perf` | 1 | 2 | LightVAE FP8 Native | LightTAE | 2 | 6 | FP8 VAE 编码 |
| `*-vae-vae` | 1 | 2 | Wan VAE | Wan VAE | 2 | 6 | 全 VAE（Table 4 最高质量） |
| `*-chunk3-*vae-vae` | 1 | 3 | Wan VAE | Wan VAE | 2 | 6 | 每 AR 步 12 帧 |
| `*-chunk4-loc8-pshuffle-lighttae` | 1 | 4 | Pixel Shuffle | LightTAE | 2 | 8 | 每 AR 步 16 帧 |
| `mv-*-chunk4-loc8-pshuffle-lighttae` | 4 | 4 | Pixel Shuffle | LightTAE | 2 | 8 | MV + 跨视图注意力 |
| `*-35steps-chunk2-loc24-*-mads1m` | 1 | 2 | Wan VAE | Wan VAE | 35 | 24 | CFG 教师（guidance_scale=3.0） |
| `*-35steps-chunk48-loc48-*-mads1m` | 1 | 48 | Wan VAE | Wan VAE | 35 | 48 | 双向教师（单 AR 步） |

### 4.5 关联集成包

FlashDreams 还包含独立的通用推理方法适配器，与 OmniDreams 共享核心基础设施但独立运作：

- **`integrations/self_forcing/`**：Self Forcing 的独立推理适配器（Huang et al., 2025）
- **`integrations/causal_forcing/`**：Causal Forcing 的独立推理适配器

---

## 5. 代码架构总览

```
flashdreams/
├── flashdreams/flashdreams/          # 核心推理基础设施
│   ├── infra/
│   │   ├── pipeline/base.py          # StreamInferencePipeline（encode → diffuse → decode）
│   │   ├── runner.py                 # Runner ABC + 分布式初始化
│   │   ├── diffusion/model/base.py   # DiffusionModel（scheduler + transformer 编排）
│   │   ├── diffusion/transformer/base.py  # Transformer ABC
│   │   ├── diffusion/scheduler/fm.py      # Flow Match scheduler
│   │   ├── compile.py                     # torch.compile wrapper
│   │   ├── cuda_graph.py                  # CUDAGraphWrapper
│   │   ├── encoder/                       # 编码器抽象
│   │   └── decoder/                       # 解码器抽象
│   ├── core/
│   │   ├── attention/kvcache.py           # BlockKVCache（流式，固定窗口）
│   │   ├── attention/cp.py                # ContextParallelAttention（ring + ulysses）
│   │   ├── attention/rope.py              # 3D RoPE
│   │   └── distributed/context_parallel.py # CP split/gather 原语
│   └── recipes/
│       ├── wan/autoencoder/vae.py         # Wan VAE 编解码器
│       └── taehv/                         # Taehv 解码器（LightTAE）
│
├── integrations/omnidreams/          # OmniDreams 集成
│   ├── omnidreams/
│   │   ├── config.py                      # 14 pipeline + 15 runner 配置
│   │   ├── pipeline.py                    # OmnidreamsPipeline（HDMap I2V）
│   │   ├── runner.py                      # OmnidreamsRunner（AR 循环驱动）
│   │   ├── transformer/
│   │   │   ├── __init__.py                # CosmosTransformer（CP 编排，CFG，CUDA Graph）
│   │   │   └── impl/
│   │   │       ├── network.py             # CosmosDiTNetwork（28 blocks × 2048 dim）
│   │   │       ├── modules.py             # Block, SelfAttention, CrossAttention, MLP
│   │   │       └── context_parallel.py    # 分层 V→T→HW 进程组创建
│   │   ├── vae_native.py                  # 原生 FP8 VAE 编码器
│   │   ├── native/
│   │   │   ├── acceleration.py            # 原生扩展策略/选择
│   │   │   ├── omnidreams_singleview.py   # OptimizedDiTExecutor 适配器
│   │   │   └── primitives.py              # 张量验证
│   │   ├── encoder/pixel_shuffle.py       # Pixel Shuffle HDMap 编码器
│   │   ├── conditioning/                  # 世界场景条件
│   │   │   ├── renderer.py                # HDMap 渲染器
│   │   │   └── world_scenario/            # 场景加载、车道线、点云
│   │   ├── grpc/                          # gRPC server + 客户端 stub
│   │   ├── webrtc/                        # WebRTC server + Web UI
│   │   └── interactive_drive/             # 桌面 demo 应用
│   └── ludus-renderer/                    # Vulkan HDMap 渲染器
```

---

## 6. 实现差距与已知局限

### 6.1 主要差距

1. **训练代码完全缺失。** FlashDreams 是纯推理基础设施。Diffusion Forcing、Self Forcing + DMD、渐进式蒸馏、WAM 后期训练等均位于独立的内部代码库中。FlashDreams *消费*由这些 pipeline 产出的蒸馏后 checkpoint。

2. **KV Cache 更新在热路径中。** 论文指出"KV Cache 更新（独立线程）"并报告 SV 延迟 118ms *不含* Cache 更新。FlashDreams 中 `BlockKVCache` 更新在主去噪线程内联发生。

3. **条件视频编码（世界场景）有独立延迟。** 论文 Table 2 显示 SV"World Scene Encoding"28ms。FlashDreams 将其计入 pipeline `encode` 阶段（拉取 HDMap 条件 → VAE 编码为 latent）。

4. **模拟器集成（AlpaSim）不在此仓库。** 论文将 OmniDreams 描述为通过 gRPC 与 AlpaSim 编排器集成。FlashDreams 实现了 gRPC *服务端*（`grpc/server.py`），但 AlpaSim 客户端和闭环编排逻辑位于独立仓库。

5. **未实现 Attention-sink token。** `config.py` 中 `sink_size_t=0` 显示所有配置均关闭了 sink token。论文讨论了使用来自 first-frame RGB latent 的 attention-sink token 以获得长期稳定性，FlashDreams 当前未启用。

6. **WAM 推理头未实现。** 论文第 7 节描述添加 action prediction head 用于 WAM。该 head 及其推理代码不在 FlashDreams 中——OmniDreams 仅作为 world model / simulator 运行，不作为 policy。

### 6.2 次要差异

7. **LightVAE/TAE 源于 LightX2V。** 论文引用 LightX2V 的 LightVAE 和 LightTAE。FlashDreams 使用来自 `flashdreams.recipes.wan.autoencoder.vae` 和 `flashdreams.recipes.taehv` 的 Wan VAE 变体和 Taehv 解码器，功能等效但命名不同。

8. **各组件独立 compile + CUDA Graph。** 性能预设（`*_PERF`）在 image encoder、per-AR-step encoder、decoder 上单独启用 `use_compile` 和 `use_cuda_graph`，超出论文仅对 DiT backbone 的范围。

9. **Text encoder 为 Cosmos-Reason1（~14GB bf16）。** `release_oneshot_encoders()` 在 AR 循环前卸载以回收 VRAM，这在论文中仅简略提及。

---

## 7. 总结

FlashDreams 作为 OmniDreams 的推理基础设施，完成了论文**第 5 节（推理优化）的全部实现**和三章架构描述的核心推理路径。其设计亮点包括：

- **CUDA Graph 集成**：`CUDAGraphWrapper` 自动从 eager 过渡到 capture/replay 模式
- **分层 CP**：完整实现 V→T→HW 优先级分解，与论文描述完全一致
- **配置系统**：通过 `derive_config()` 模式实现 14 种 pipeline 变体，覆盖论文 Tables 2-6
- **Serving 就绪**：WebRTC + gRPC server、session 管理、录制、profiling 开箱即用
- **交互式 Demo**：完整的桌面闭环驾驶体验，支持物理方向盘、HDMap 渲染和实时 world model 响应

训练-推理分离是刻意的设计选择——FlashDreams 面向生产推理优化，不涉及训练 pipeline。原生加速策略（`disabled/auto/required`）为内部 CUDA 优化提供后门，同时保持纯 PyTorch 路径作为开源默认行为。
