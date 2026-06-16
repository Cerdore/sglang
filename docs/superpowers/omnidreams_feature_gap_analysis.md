# OmniDreams 功能差距分析(单视角 SV)

对照三方来源,梳理 flashdreams 参考实现 / 论文有、而我们 sglang 移植版缺失或只做了一半的功能。

- **论文**:`/home/admin/workspace/2606.03159.pdf`(正文可恢复到约第 20 页,Sec. 1–9.1 + Table 1–5)
- **参考实现**:`/home/admin/workspace/flashdreams/`(`integrations/omnidreams_singleview`、`flashdreams/core`、`configs`)
- **我们的移植**:`python/sglang/multimodal_gen/`(omnidreams DiT / VAE / pipeline / configs)

> 注:多视角(MV / cross-view / 4-7 相机)相关功能**已全部排除**,因为该模型未开源,我们也拿不到 checkpoint。本文只覆盖单视角(SV)推理路径。

> **核实说明(2026-06-16)**:本文第一节已对照当前分支 `feat/omnidreams-p0-p4b-optimizations` 逐条用 subagent 查证(file:line 证据)。早期草稿大量"缺失/仅骨架"的判断**不成立**——分支实际比文档想象的完整得多。下表已修正为查证后的真实状态。

---

## 一、flashdreams && 论文都有的优化 —— 我们的真实状态

这些是论文 Sec. 5 明确归为"实时推理核心"、flashdreams 也落地了的优化。**多数已实现**,真正还缺的只有 torch.compile(及可选的异步 KV 更新)。

| # | 功能 | flashdreams / 论文做法 | 当前分支查证结果 | 证据 |
|---|---|---|---|---|
| 1 | **CUDA Graph 捕获 AR chunk** | 静态 shape 缓存使 chunk 0 懒捕获,后续每步 replay(Sec 5.1) | ✅ **已实现**。真捕获+replay,稳态判断 + warmup;门控 `SGLANG_OMNIDREAMS_CUDA_GRAPH` / `enable_cuda_graph` | `omnidreams_cuda_graph.py`;stage `omnidreams.py:851-876, 931-956` |
| 2 | **torch.compile 真正激活** | net 挂 `max-autotune-no-cudagraphs`(`flashdreams/infra/compile.py`) | ❌ **完全缺失(唯一硬缺口)**。`_compile_conditions` 定义了但 pipeline `load_modules()` 从未调用 `torch.compile()`;别的模型有 `_maybe_enable_torch_compile()`,OmniDreams 没有。即使 `--enable-torch-compile` 仍跑 eager | DiT `omnidreams.py:582`;pipeline `load_modules()` 无调用 |
| 3 | **KV cache 异步更新、移出热路径** | cache 更新放独立线程,不计入"Total"延迟(Sec 5.1,Table 2) | ⚠️ **差异(非缺陷)**。我们全同步 `copy_()`,在关键路径上;这是配合稳态 CUDA graph 静态 shape 的有意选择。异步化需权衡 | `omnidreams_kvcache.py:136-277` |
| 4 | **提取每步不变算子(hoisting)** | RoPE 频率 + patchify/unpatchify 每 chunk 算一次,而非每去噪步(Sec 5.1) | ✅ **已实现**。`rope_freqs = rope.shift_t(chunk_idx)` 每 chunk 算一次,该 chunk 3 次 forward 复用;patchify/unpatchify 在 chunk 边界 | stage `omnidreams.py:911, 943/955/987, 993` |
| 5 | **轻量解码器 LightTAE(P1)** | LightTAE(LightX2V)解码 latent→RGB,极低延迟(Sec 5.1,Table 5) | ✅ **已实现**。`LightTAEDecoder` + `OmniDreamsLightTAEDecodingStage`;`decoder_config.impl="lighttae"` | `taehv.py:447`;stage `omnidreams.py:1053-1070`;config `omnidreams.py:85-87` |
| 6 | **轻量编码器 LightVAE(P2)** | LightVAE 替换 Cosmos VAE 做条件编码(Sec 5.1) | ✅ **已实现**。streaming 编码 + PyTorch fallback + 原生 FP8;`encoder_config.impl="lightvae"` | `omnidreams_light_vae.py:308-510`;config `omnidreams.py:79-84` |

**结论:这一节里真正的待办只有第 2 项 torch.compile。** 第 3 项是设计取舍,其余均已落地。

---

## 二、flashdreams 独有(论文未提,属工程实现层)

论文可恢复正文**从未描述**权重量化、CFG 蒸馏或具体注意力 kernel。这些是 flashdreams 在工程上额外做的。**查证后发现我们大部分也已移植**,只是默认走 eager/PyTorch,FP8/native 后端需显式开启。

### 2.1 快注意力后端

| 功能 | flashdreams 位置 / flag | 当前分支查证结果 |
|---|---|---|
| **SageAttention-3**(Blackwell block FA3,FP8/BF16) | `optimized_dit.py`,`--native-attention-backend=sage3 / sage3_fp8` | ✅ FP8 路径可选(`omnidreams_fp8.py` `attention_backend`) |
| **SpargeAttn**(稀疏 block top-K,ratio 0.25) | `optimized_dit.py`,`native_dit_sparge_topk=0.25` | ✅ FP8 路径可选(`sparge_topk` 参数) |
| **cuDNN / flash 后端** | `--native-attention-backend=auto` | ✅ FP8 native 支持 flash;⚠️ eager(bf16)路径仍是纯 torch SDPA(`omnidreams.py:380`) |

> 说明:注意力量化(含 `k_cross`/`v_cross` 文本交叉注意力 KV)属**单视角路径**,`cross` 指文本 cross-attention(`crossattn_proj`),**不是** cross-view 多视角,与 MV 无关。我们的 cross-attn KV 预算(`precompute_cross_attn_kv`)已实现;FP8 量化随 native FP8 路径走。

### 2.2 量化与原生热路径

| 功能 | flashdreams 位置 | 当前分支查证结果 |
|---|---|---|
| **原生 FP8 DiT 热路径** | C++/CUDA `optimized_dit_forward`,FP8 GEMM + AdaLN-LoRA 融合 | ✅ **已接入**。`omnidreams_fp8.py`,三态 `native_dit_acceleration=disabled/auto/required` + `SGLANG_OMNIDREAMS_FP8_DIT`,仅 sm_120 |
| **snapshot 后释放 PyTorch 网络** | `_release_network_after_fp8_snapshot`:换成 `_CosmosNetworkShapeOps`(只留 shape)+ `gc.collect()` + `empty_cache()`,真省显存 | ⚠️ **部分**。native 端口里释放代码在(`native/.../optimized_dit.py:1144-1172`),但 SGL 包装 `_SGLTransformerAdapter` 仍持 `_sgl_dit` 引用;native 接管计算后该权重不再被访问,但**名义上仍占显存**。要完全等价需显式 `del self._sgl_dit` + gc/empty_cache |
| **懒加载权重 snapshot + CPU 侧量化** | 首次 `predict_flow` 才 snapshot,量化在 CPU 避免 OOM | ⚠️ 加载策略不同,待评估是否有 OOM 风险 |
| **Linear 权重预打包** | 融合 QKV + 转置 layout 供 CUTLASS GEMM | ✅ 已等价(见 2.3) |
| **原生 FP8 LightVAE(P4b)** | `vae_native.py`,`OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH` | ✅ native FP8 编码路径存在,默认回退 PyTorch |

### 2.3 缓存/内存细节

| 功能 | flashdreams 位置 | 当前分支查证结果 |
|---|---|---|
| **timestep embedding 缓存** | 按 `(ar_idx, call_idx, timestep)` 键缓存 t_emb / adaln_lora | ❌ 不缓存(用全局 `Timesteps` 对象,`omnidreams.py:122-150`),小项 |
| **临时缓存清理** | finalize 后释放每步 RoPE/hdmap 缓存,只留稳态 KV | ⚠️ 无显式清理(CUDA graph 内部管理),待评估 |
| **HDMap 投影一次复用** | `_ensure_hdmap_tensor` + `_optimized_hdmap_cache`,跨 AR step 复用 | ✅ **已等价**。native 端口 `native/omnidreams_singleview/python/optimized_dit.py:1551-1591` 完整保留缓存键与复用 |
| **Linear 权重预打包(fused QKV + CUTLASS layout)** | native CUDA `cosmos_block.cu`,`weight_prepared` / fused QKV | ✅ **已等价**。native 库内部实现一致(`native/.../kernels/cosmos_block.cu`,`cosmos_fp8_utils.py`),不暴露 Python |

---

## 三、论文独有(flashdreams SV 路径未直接暴露 / 我们也缺)

这些来自论文 Sec. 4-5,属算法层面,flashdreams 作为"已蒸馏 checkpoint + 推理 recipe"打包,部分训练侧能力不在推理代码里。

| 功能 | 论文出处 | 说明 / 我们的现状 |
|---|---|---|
| **SelfForcing + DMD 蒸馏** | Sec 4.3 | 训练侧产出 2-step checkpoint。我们直接用蒸馏后权重,**推理侧已对齐**(2-step warp scheduler)。训练复现不在范围 |
| **渐进式 teacher 蒸馏** | Sec 4.3 | 长上下文 teacher 修正学生,消除滚动 KV 超训练窗口时的"漂移伪影"。属训练,我们无 |
| **first-block latent 注入 / mask_first_block** | Sec 5.1(flashdreams `transformer/__init__.py:587-635,722-728`):AR step 0 用 `latent = latent*(1-mask) + image*mask` 强制首帧为干净 VAE latent,并用 mask 屏蔽 | ❌ **确认缺失**。`omnidreams_fp8.py:118-122` 的 `_maybe_inject_image()` 直接返回原 latent、`_select_mask()` 返回 None;首帧只作普通 I2V 条件编码(stage `omnidreams.py:334-358`),无 mask 强制注入。可能影响时序一致性 |
| **Diffusion Forcing(逐帧独立扩散时刻)** | Sec 4.2 | 训练用 Flex-Attention 因果 mask。属训练,我们无 |
| **静态 shape KV cache 预分配** | Sec 5.1 | 预分配固定 shape(配合 CUDA graph)。我们的 BlockKVCache 是固定窗口但需确认是否完全静态 shape |
| **PixelShuffle 伪 VAE 编 HDMap** | Sec 5.1 | 见下方专项说明 |

### PixelShuffle HDMap 编码器 —— 编码器免费,但 checkpoint 未开源,不可落地

flashdreams 确有**单视角**变体 `SV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE`(`integrations/omnidreams/omnidreams/config.py:222`),不是 MV 专属。要点:

- **编码器本身零成本**:`encoder/pixel_shuffle.py` 的 `PixelShuffleVAEEncoder` 是 stateless 伪 VAE(注释 "Stateless drop-in for a learned VAE"),逻辑 = 每 AR step 选一帧 + 8×8 空间 unshuffle 进通道(`[T,C,H,W]→[Tl, C*64, H/8, W/8]`),纯 `einops.rearrange`,**无权重,直接 PyTorch 即可**,<1ms。
- **但不能用在我们现有 checkpoint 上**:该配置绑定**不同的 DiT checkpoint** `1view-pshuffle-chunk4`,在 flashdreams 里标记为 **`"MISSING"`(未开源)**;且 DiT 结构改了 `additional_concat_ch=192`(VAE 路径是 16ch HDMap latent)、`len_t=4`、`window_size_t=8`,HDMap patch embedding 是按 192 通道**重训过的**。
- 我们手上可用的是 `1view-vae-chunk2`(VAE 编 HDMap、16ch),换上 pixel-shuffle 编码器后通道布局对不上,输出会坏。

**结论:不可落地。** 阻塞点是未开源的 pshuffle checkpoint + 对应 192ch DiT 配置,而非 pixel-shuffle 本身。我们 SV 用 VAE 编 HDMap 是正确且必须的。

**HF 官方 README 佐证(`nvidia/omni-dreams-models`,2026-06-02 发布)**:公开仓库只发布了 **4 个单视角 checkpoint**,全部是 chunk-2 / VAE-based:

| 文件 | 角色 |
|---|---|
| `single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt` | 发布的蒸馏 runtime 目标·chunk-2(~52 FPS@720p on GB300)= 我们在用的 `1view-vae-chunk2` |
| `single_view/distilled/e5cadda3-….pt` | L0 蒸馏(训练可续输入) |
| `single_view/teacher/3b4c21d0-….pt` | L1b teacher·双向·189 帧(蒸馏 teacher 输入) |
| `single_view/student-init/a12bf26e-….pt` | L2a student-init·Diffusion Forcing(蒸馏学生初始化) |

无任何 pshuffle / chunk-4 / MV 权重。注意:论文吹的 **68 FPS** 对应 chunk-4 pshuffle 内部配置;**公开发布的 runtime 目标是 chunk-2、~52 FPS**。`config.py` 里 6 个 checkpoint key 中只有 `1view-vae-chunk2` 有真实 URL,其余 5 个(pshuffle/MV/diffusion-forcing/bidirectional)标 `MISSING` = 内部专用。`OMNI_DREAMS_HF_ORG` 重定向仅作用于 `omni-dreams-scenes`(场景数据),不涉及模型权重。

---

## 四、我们已对齐 flashdreams/论文的部分(参考)

避免重复造轮子,以下已实现:

- AdaLN 融合(`LayerNormScaleShift`,CuTe DSL kernel)
- RoPE kernel dispatch(FlashInfer / Triton fast path)
- KV-cache split-copy(免 per-block clone)
- 文本编码器 LRU 缓存(max 32,CPU 存储)
- cross-attn KV 预算(每 prompt 投影一次,复用 28 blocks × N chunks)
- 2-step flow-match warp scheduler(shift=5,sigmas [1.0, 0.8036, 0.0])
- 无 CFG 蒸馏推理(guidance_scale=1.0)
- 3 个重模块 CPU offload(text / VAE / DiT)
- 3D NeoX RoPE(44:42:42,h/w 外推 3.0)+ Q/K RMSNorm
- 窗口 KV cache(window=6,sink=0)

---

## 五、查证后真正的待办(SV)

> 经逐条 file:line 核实,P0(CUDA graph)、P1(LightTAE)、P2(LightVAE)、P4a(FP8 DiT)、P4b(FP8 LightVAE)、RoPE/patchify hoisting、HDMap 缓存复用、QKV 预打包、cross-attn KV 预算**均已实现**。剩余真正待办很少。

### 5.1 明确该做(功能/正确性缺口)

| 项 | 缺口 | 工作量 / 收益 |
|---|---|---|
| **torch.compile 激活** | 唯一的纯性能硬缺口。`_compile_conditions` 定义了但 pipeline `load_modules()` 从未调用 `torch.compile()`;`--enable-torch-compile` 形同虚设 | 小;论文 ~10-15% |
| **first-block latent 注入 / mask_first_block** | 确认缺失,`_maybe_inject_image` 空实现、`_select_mask` 返回 None。可能影响 AR 时序一致性(正确性,不只是速度) | 中;需先评估对画质的实际影响 |

### 5.2 需要取舍(做不做取决于权衡,默认可不做)

| 项 | 取舍点 |
|---|---|
| **KV cache 异步 / 移出热路径** | flashdreams 把更新丢独立线程、不计入延迟(省 ~20-40ms/chunk)。但**与稳态 CUDA graph 的静态 shape 捕获冲突**——我们故意做成同步。要异步化得重构 graph 捕获边界,收益未必抵得过复杂度。**默认保持同步。** |
| **eager(bf16)路径快注意力后端** | flash/Sage3/Sparge 只在 FP8/sm_120 native 路径可用,bf16 eager 仍走 torch SDPA。给非 Blackwell 卡接 FlashAttention 有收益,但若目标硬件就是 sm_120 + FP8,则非必要。**取决于部署硬件。** |
| **FP8 网络释放显存** | native 端口有释放代码,但 SGL 包装 `_SGLTransformerAdapter` 仍持 `_sgl_dit` 引用(名义占显存,实际不被访问)。要省那 ~3-4 GiB 需显式 `del self._sgl_dit` + gc/empty_cache。**仅在显存吃紧时做。** |
| **timestep embedding 缓存** | 小项收益,按 `(ar_idx, timestep)` 缓存 t_emb/adaln_lora。AR step 多时才明显。**低优先。** |
| **临时缓存清理 / 懒 snapshot** | 内存优化,需先确认当前是否真有 OOM 风险,否则不必动。 |

### 5.3 不适用(训练侧 / MV,本项目范围外)

- SelfForcing + DMD 蒸馏、渐进式 teacher 蒸馏、Diffusion Forcing —— 均为训练侧,我们直接用蒸馏后 checkpoint,推理侧已对齐。
- 多视角 / cross-view / context parallelism —— 模型未开源,排除。

> 性能参考:论文 SV 2B 在 1×GB300 上 68 FPS @704×1280(118 ms / 8 帧 chunk);实时定义为 30 FPS。
