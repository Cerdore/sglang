# OmniDreams FP8 DiT — 离线预量化权重方案

> 2026-06-18 | 推理系统优化 | 面试用设计文档

## 1. 问题定义

### 1.1 现状

`OmniDreamsFP8DiT.__call__()` 首次调用时，在 GPU 推理热路径上懒执行 FP8 权重量化：

```
state_dict()                                          [BF16 GPU → CPU 拷贝]
  └─ prepare_cosmos_quantized_streaming_weights()     [CPU: QKV fuse + per-out-channel FP8 量化]
       └─ .to(device="cuda").contiguous()             [FP8 权重 CPU → GPU 回拷]
```

这三个串行操作涉及一次完整的 CPU-GPU 往返 + 28个 block × 8个 linear 的 Python 逐层量化循环。

### 1.2 量化数据

C++ native kernel (`optimized_dit_forward`) 运行 6 chunk × 28 block 的 FP8 前向只需 **1.45s**（比 eager BF16 2.2s 快 35%）。但 Python 侧的懒量化准备耗时 5-14s，导致 `OmniDreamsDenoisingStage` 实测 6.9-21.3s——**Python 开销占 78-93%**。

| 运行 | DenoisingStage | C++ kernel | Python 开销 | 占比 |
|------|---------------|------------|-------------|------|
| 进程 A (no profile) | 6.89s | ~1.45s | 5.4s | 78% |
| 进程 B (no profile) | 15.37s | ~1.45s | 13.9s | 91% |
| Profiled | 21.34s | ~1.45s | 19.9s | 93% |

> 注：三行是**三个独立进程**的冷启动测量（`sglang generate` 每次都是全新进程，懒量化按进程缓存，所以每进程都付一次冷启动代价），不是同一进程内的第 1/2 次调用。同为 "no profile" 的两行（5.4s vs 13.9s）量化工作量相同却差 2.5×，差异来自系统状态（page cache、GPU 时钟、其他进程争用）；正式立项前建议各跑 3 次取中位数，避免单点噪声误导收益估算。

### 1.3 生态位分析

SGLang 扩散系统中所有量化模型全部走离线预量化：

| 模型族 | 量化方式 | 导出工具 | 运行时 |
|--------|----------|----------|--------|
| FLUX/FLUX2/LTX2/HunyuanVideo/Qwen-Image | ModelOpt FP8 | `build_modelopt_fp8_transformer.py` — 离线读 backbone.pt → 物化 float8_e4m3fn → 写 safetensors | `modelopt_quant.py` — `from_pretrained` 加载预量化权重 |
| FLUX Nunchaku | W4A4 | `nunchaku_config.py` — 离线 SVDQ 量化 | `nunchaku_linear.py` — 运行时 dispatch |
| Cosmos-Reason1-7B (TE) | W8A8 PTQ | `export_cosmos_reason_fp8.py` — llmcompressor SmoothQuant + 256 样本校准 | Transformers 自动检测 `config.json:quantization_config` |
| **OmniDreams DiT (当前)** | **在线 FP8** | **无** | **首次 `__call__` 时在热路径上现烤** |

OmniDreams FP8 DiT 是整个 SGLang 扩散系统中唯一走在线量化的模型——这是性能差的根本原因。

### 1.4 目标

把 FP8 量化移出推理热路径，改为离线一次性完成，推理时直接加载预量化权重。

### 1.5 约束

- `sglang generate` 和 `sglang serve` 两条路径都受益
- 必须兼容 `_SGLTransformerAdapter`（SGLang 不释放 `self.network`）
- 跟随 SGLang 现有量化范式（`build_modelopt_fp8_transformer.py` 模式）
- `native_dit_acceleration=required` 时，不提供预量化权重直接 fail fast（和 ModelOpt 行为一致）
- **物理删除懒量化回退**——删掉 vendored `optimized_dit.py` 中 `_ensure_weights_snapshot` 的 FP8 懒量化分支（CPU 量化 + CPU↔GPU 搬运 + 网络释放），改为"未注入即 raise"。这是一处**有意为之的上游偏离**：`optimized_dit.py` 原本仅有 5 处 import stub 差异，本改动新增第 6 处实质差异，后续 re-vendor FlashDreams 需手动 merge 这段。`cosmos_fp8_utils.py` / `cosmos_weights.py` 仍保持 byte-identical，不动
- **热路径不再现烤**——`required` 必注入、`auto` 无文件回退 eager；FP8 路径下懒量化代码已物理移除，不存在被误触发的可能

### 1.6 非目标

- 不改动 C++ kernel
- 不改 cuDNN 注意力后端选择
- 不碰多 GPU 分发
- 不碰 VAE/TE 侧的 FP8

---

## 2. 架构总览

```
┌─ 离线（一次性）─────────────────────────────────────────┐
│                                                         │
│  tools/export_omnidreams_fp8_dit_weights.py             │
│                                                         │
│  BF16 checkpoint ──► prepare_cosmos_quantized_          │
│                       streaming_weights()               │
│                          │                              │
│                          ▼                              │
│                  omnidreams_fp8_dit.pt                   │
│                  (FP8 weights + scales)                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ 在线（推理）───────────────────────────────────────────┐
│                                                         │
│  build_fp8_dit(fp8_prepared_path=...)                   │
│    │                                                    │
│    ├─ 路径存在? → torch.load() → to(device)  [<0.5s]   │
│    └─ 路径缺失 → raise, 提示运行导出工具                  │
│                                                         │
│  OmniDreamsFP8DiT.__init__()                            │
│    │                                                    │
│    └─ 构造时 GPU 化预量化权重                             │
│                                                         │
│  OmniDreamsFP8DiT.__call__()                            │
│    │                                                    │
│    └─ _ensure_weights_snapshot() 不再做量化              │
│       权重已在 __init__ 加载完毕                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 详细设计

### 3.1 导出工具 `tools/export_omnidreams_fp8_dit_weights.py`（新增）

输入：BF16 DiT checkpoint (`single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt`)
输出：`omnidreams_fp8_dit.pt`（扁平 dict）

**关键约束：必须经过模型的 `post_load_weights` 融合，不能直接量化 raw checkpoint。**
懒量化路径快照的是 `self.network.state_dict()`，而该 state_dict 是在 `OmniDreamsDiT.post_load_weights()`
之后取的（`omnidreams.py:719-722`）——它做了三件事：①padding-mask 通道融进 `x_embedder`（72→68 入通道），
②Cosmos channel-shuffle 融进 `final_layer.linear.weight`，③重建非持久 buffer。raw `.pt` 里是**融合前**的
权重（`x_embedder` 是 72 入通道、`final_layer` 未 reshape）。若直接对 raw checkpoint 量化，导出的 `x_embedder` /
`final_layer` 与运行时不一致，会导致 key/shape 不匹配或结果错误，更谈不上和懒量化路径数值等价。

因此导出工具必须**走 SGLang 的正常加载链**，让 `post_load_weights` 跑完，再取 state_dict 量化：

```
内部链路:
  1. 用 OmniDreamsPipeline 的 loader 实例化 OmniDreamsDiT 并 load_state_dict(raw .pt)
  2. dit.post_load_weights()                               ← 融合 padding-mask / channel-shuffle（与运行时一致）
  3. state_dict = dit.state_dict()                          ← 融合后的权重快照（CPU）
  4. prepare_fp8_dit_weights(state_dict, num_blocks)        ← 复用 omnidreams_fp8.py:66 现有函数
        └─ 内部即 prepare_cosmos_quantized_streaming_weights：
           QKV fuse + per-out-channel FP8 量化 + _fp8_prepared/scale alias（单次调用，无需手动三步）
  5. torch.save({"weights": fp8_dict, "meta": {...校验信息...}}, output_path)
```

复用 vendored `cosmos_fp8_utils`（`prepare_cosmos_quantized_streaming_weights`）与 SGLang 的
`omnidreams_fp8.prepare_fp8_dit_weights()`——后者本就是"defined but never called in production，仅用于离线
预处理/测试"（见 CLAUDE.md 技术债条目），离线导出正是它的预期用途。不引入任何新量化逻辑。

**版本校验**：导出时在 `.pt` 的 `meta` 里附带 source checkpoint 的标识，推理加载时比对，避免 checkpoint 升级后静默使用过期量化结果。

> 校验方式权衡：SHA256 对 3.9GB 文件每次启动都要全量读盘+哈希（约数秒），会吃掉一部分收益，且 `build_fp8_dit` 当前签名拿不到 raw checkpoint 路径（只收到已加载的 `self.transformer`）。推荐改用**轻量指纹**：raw `.pt` 的 `(文件大小, mtime)` + 导出工具版本号，存进 `meta`，加载时由 DenoisingStage（它能从 `config`/`model_path` 解析到 raw checkpoint 路径）`os.stat` 比对，毫秒级、无需全量哈希。需要强校验时再提供 `--verify-sha256` 显式开关。无论哪种，`build_fp8_dit` 都需要新增 source checkpoint 路径参数，或把校验上移到 DenoisingStage。

**用法示例**：
```bash
python -m sglang.multimodal_gen.tools.export_omnidreams_fp8_dit_weights \
    --checkpoint /path/to/omni-dreams-models/single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt \
    --output /path/to/omni-dreams-models/omnidreams_fp8_dit.pt
```

### 3.2 `build_fp8_dit()` 改动 (`omnidreams_fp8.py`)

```python
def build_fp8_dit(
    sgl_dit, arch, *,
    mode: str,                              # "auto" | "disabled" | "required"
    fp8_prepared_path: str | None = None,   # 新增：由 DenoisingStage 解析后传入
    attention_backend: str = "auto",
    sparge_topk: float | None = None,
) -> OmniDreamsFP8DiT | None:
    if mode == "disabled":
        return None
    native = _load_native(mode)
    if native is None:                       # 原有三层回退第 2 层
        return None

    fp8_weights = None
    if fp8_prepared_path and os.path.exists(fp8_prepared_path):
        payload = torch.load(fp8_prepared_path, map_location="cpu", weights_only=True)
        fp8_weights = payload["weights"]     # meta 校验交给 DenoisingStage（见 3.1 版本校验）

    if fp8_weights is None:
        if mode == "required":
            raise FileNotFoundError(
                f"FP8 prepared weights not found at {fp8_prepared_path!r}. "
                "Run: python -m sglang.multimodal_gen.tools."
                "export_omnidreams_fp8_dit_weights --checkpoint <path> --output <path>"
            )
        # auto 模式无文件 → 返回 None，DenoisingStage 回退 eager（omnidreams.py:910-916）
        # 注意：绝不再走 vendored 懒量化分支
        return None

    return OmniDreamsFP8DiT(
        sgl_dit, arch, native,
        fp8_prepared_weights=fp8_weights,    # CPU dict，注入后在 _ensure_executor 移到 GPU（见 3.3/3.4）
        attention_backend=attention_backend,
        sparge_topk=sparge_topk,
    )
```

**与现状的两点关键差异**：
- `auto` 模式拿不到预量化文件时 **return None**（而非构造一个无权重的 wrapper）——这样才能命中 DenoisingStage 已有的 eager 回退；构造无权重 wrapper 配合"删掉懒量化"会在运行时崩。
- 默认路径**不从 `arch` 推导**（`arch` 是 `OmniDreamsDiTArchConfig`，不含 `model_path`）。路径解析放在 DenoisingStage（它能拿到 `config`/`model_path`），跟 raw checkpoint 同目录推断 `omnidreams_fp8_dit.pt`，再把结果作为 `fp8_prepared_path` 传进来（见 3.5）。

### 3.3 `OmniDreamsFP8DiT.__init__` 改动（`omnidreams_fp8.py`）

`__init__` 只**存下 CPU dict**，不在这里搬 GPU——构造发生在 DenoisingStage init 阶段，此时 DiT 可能还在 CPU
offload，且不该把设备写死成 `"cuda"`（会忽略实际 device index）。搬迁推迟到 `_ensure_executor`（首次 `__call__`），
那时设备已确定，与懒量化路径 `next(self.network.parameters()).device` 的取设备方式一致：

```python
def __init__(self, sgl_dit, arch, native, *,
             attention_backend="auto", sparge_topk=None,
             dit_backend="fp8_kvcache_cudnn",
             fp8_prepared_weights: dict | None = None):   # 新增
    ...
    self._prepared_weights = fp8_prepared_weights   # CPU dict 或 None，先不搬 GPU
```

### 3.4 注入 + 删分支

本节有两处改动：①SGLang 侧把权重注入 executor（`omnidreams_fp8.py`），②删掉 vendored 的懒量化分支（`optimized_dit.py`）。

#### (a) 注入点：`OmniDreamsFP8DiT._ensure_executor`（`omnidreams_fp8.py`）

关键事实：`_ensure_weights_snapshot()` 是 **`OptimizedDiTExecutor` 的方法**（`optimized_dit.py:1078`），而
executor 是首次 `__call__` 时由 `_ensure_executor` 懒构造的——`OmniDreamsFP8DiT` 上并没有这个方法。所以注入必须在
`_ensure_executor` 构造完 executor 后，**直接预置 `executor._optimized_weights`**（移到 GPU）。`_ensure_weights_snapshot()`
第一行 `if self._optimized_weights is not None: return ...`（`optimized_dit.py:1080`）会立即命中并返回：

```python
def _ensure_executor(self, len_t, height, width):
    if self._executor is None:
        adapter = _SGLTransformerAdapter(self._sgl_dit, self._arch,
                                         len_t, torch.bfloat16, height, width)
        self._executor = self._native.optimized_dit.OptimizedDiTExecutor(
            adapter, self._native.extension,
            dit_backend=self._dit_backend,
            attention_backend=self._attention_backend,
            sparge_topk=self._sparge_topk,
        )
        # 预量化权重为本路径唯一来源（删分支后没有别的兜底）
        assert self._prepared_weights is not None, \
            "FP8 prepared weights missing; build_fp8_dit must inject them"
        device = next(self._sgl_dit.parameters()).device
        self._executor._optimized_weights = {
            k: v.to(device=device).contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in self._prepared_weights.items()
        }
        self._prepared_weights = None   # 释放 CPU 引用
    return self._executor
```

**完整性确认**：注入的是 `prepare_cosmos_quantized_streaming_weights` 的完整输出——block 线性层是 FP8（uint8+scale），
其余（`t_embedder.*`、`t_embedding_norm`、`final_layer.adaln_modulation.*`、`blocks.*.adaln_modulation_*`、
`additional_patch_embedding.proj.1.weight` 等）原样保留为 BF16。`__call__` 里 `_ensure_invariant_tensors` /
`_ensure_hdmap_tensor` 读的就是这些 BF16 key，全部存在，注入字典自洽完整。

#### (b) 删分支：`_ensure_weights_snapshot`（`optimized_dit.py`，**vendored，第 6 处偏离**）

删掉 FP8 懒量化整段（`optimized_dit.py:1083-1112`：`prepare_cosmos_quantized_streaming_weights` 调用、CPU
state_dict 拷贝、CPU↔GPU 搬运、`_drop_redundant_bf16_prepared_weights`、`_release_network_after_fp8_snapshot`），
改为未注入即报错：

```python
def _ensure_weights_snapshot(self) -> dict[str, Tensor]:
    if self._optimized_weights is not None:
        return self._optimized_weights
    if self._uses_fp8_dit:
        raise RuntimeError(
            "FP8 weights not injected. Run the offline exporter and pass "
            "fp8_prepared_path through build_fp8_dit()."
        )
    # 非 FP8（bf16 native）路径保留原样
    self._optimized_weights = prepare_cosmos_streaming_weights(self.network.state_dict())
    return self._optimized_weights
```

> 连带删除/保留判断：`_release_network_after_fp8_snapshot`、`_snapshot_cross_cache_weights`、`_CosmosNetworkShapeOps`
> 这些只服务于"释放上游网络后重建跨场景 cross-attn cache"的逻辑，删 FP8 懒量化分支后**在 SGLang 路径里全部不再被调用**
> （adapter 始终持有真实 `OmniDreamsDiT`、cache 在 stage 层初始化）。实现时可一并删除以减少死代码，但它们也被 `_ensure_fp8_runtime`
> 末尾（`optimized_dit.py:1340`）调用——**先 grep 确认所有调用点都已失效再删**，否则只删 `_ensure_weights_snapshot` 里那一处调用即可。
> 这条要在实现时实测确认，不要凭这段描述盲删。

### 3.5 Pipeline 层对接（`omnidreams.py`）

`OmniDreamsDenoisingStage` 解析路径、（可选）校验指纹，再传给 `build_fp8_dit`。注意实际调用点已有
`attention_backend` / `sparge_topk` 参数（`omnidreams.py:903-909`），新增 `fp8_prepared_path` 不要漏掉它们：

```python
# 显式配置优先，否则跟 raw checkpoint 同目录推断
fp8_prepared_path = getattr(config, "native_dit_fp8_prepared_path", None)
if fp8_prepared_path is None:
    fp8_prepared_path = os.path.join(os.path.dirname(ckpt_path), "omnidreams_fp8_dit.pt")
# 轻量指纹校验（见 3.1）：mtime/size 不匹配则告警并忽略预量化文件，回退 eager 或报错

fp8_dit = build_fp8_dit(
    self.transformer, arch,
    mode=mode,
    fp8_prepared_path=fp8_prepared_path,
    attention_backend=getattr(config, "native_dit_backend", "auto"),
    sparge_topk=getattr(config, "fp8_dit_sparge_topk", None),
)
```

**新增配置字段**：`native_dit_fp8_prepared_path`（`OmniDreamsPipelineConfig`，可选 str，默认 None → 同目录推断）。
这是一处必须显式落地的改动，原"改动清单"漏了它。

### 3.6 范围说明

原 §3.6"`server_args.py` 保留 `decoder: True` 修复"与本设计（FP8 离线预量化）无关，属另一分支的遗留改动，
不应混进本方案。本设计不涉及 `server_args.py`（除非确实需要为 `native_dit_fp8_prepared_path` 加 CLI 暴露——
但按现状该类参数走 `pipeline_config_kwargs`/config 文件，无需 CLI flag）。如该 `decoder` 修复确属本分支，请单独立项说明。

---

## 4. 改动清单

| 文件 | 改动 | 类型 |
|------|------|------|
| `tools/export_omnidreams_fp8_dit_weights.py` | 离线导出工具：**经 OmniDreamsDiT + post_load_weights** 取 state_dict 后量化 | **新增** |
| `omnidreams_fp8.py` — `build_fp8_dit()` | 新增 `fp8_prepared_path`；无文件时 `required` 报错 / `auto` 返回 None | 修改 |
| `omnidreams_fp8.py` — `OmniDreamsFP8DiT.__init__` | 新增 `fp8_prepared_weights`，仅存 CPU dict | 修改 |
| `omnidreams_fp8.py` — `OmniDreamsFP8DiT._ensure_executor` | 预置 `executor._optimized_weights`（移 GPU），短路懒量化 | 修改 |
| `omnidreams.py` — DenoisingStage | 解析默认路径 + 指纹校验，传 `fp8_prepared_path`（保留现有 attention/sparge 参数） | 修改 |
| `configs/pipeline_configs/omnidreams.py` — `OmniDreamsPipelineConfig` | 新增 `native_dit_fp8_prepared_path` 字段 | 修改 |
| `optimized_dit.py`（vendored）— `_ensure_weights_snapshot` | **物理删除 FP8 懒量化分支**，未注入即 raise（第 6 处上游偏离） | 删代码 |

---

## 5. 预期收益

DenoisingStage 从 6.9-21.3s → 2.0-3.0s，**加速 3-7×**。

端到端生成 (`Pixel data generated`) 从 20-32s → 12-15s。

`sglang generate` 和 `sglang serve` 同时受益——不再有 per-process 冷启动量化开销。

---

## 6. 测试策略

### 6.1 离线导出正确性

- **数值等价性**：同 seed（42）推理，预量化路径 vs 当前懒量化路径，中间张量逐层 `allclose(atol=1e-3)`。
  前提是导出工具走了 §3.1 的"经 post_load_weights 取 state_dict"链——两边对**同一份融合后权重**跑**同一个**
  `prepare_cosmos_quantized_streaming_weights`（确定性量化），预期 bit-exact。**必须加一个 key-level 断言**：
  导出字典的 key 集合 == 懒量化路径 `_optimized_weights` 的 key 集合（同 `linear_policy="all"`），这是抓"导出漏了
  post_load 融合 / policy 不一致"的最直接哨兵。
- **导出可复现**：同一 checkpoint 跑两次导出，SHA256 一致
- **版本校验**：故意给旧导出文件 + 新 checkpoint，断言加载时报错提示重新导出

### 6.2 推理功能回归

- `sglang generate --config omnidreams_dit_fp8.json`：出视频、无 NaN、峰值显存不增加、无 `using eager DiT`
- `sglang serve` + 多次 `/v1/videos`：首次请求无冷启动延迟，后续请求稳定
- LightVAE/LightTAE 路径不受影响（不改 VAE 侧）
- 多 seed 可复现 (42/123/7)，Denoising 阶段方差小

### 6.3 性能验证

- `OmniDreamsDenoisingStage` 预期 ≤ 3s（C++ kernel 1.45s + 轻量 Python 桥接）
- `compare_perf.py` 对比导出前后，确认加速比 ≥ 3×
- RTX 6000D (sm_120) 和 RTX 5070 (sm_120) 两卡都跑通

### 6.4 边界情况

| 边界 | 行为 |
|------|------|
| `native_dit_acceleration=required` + 无预量化文件 | 清晰报错，提示运行导出工具 |
| 预量化文件存在但 hash 不匹配 | 报错提示 checkpoint 已更新需重新导出 |
| 空 HD-map | 正常运行 |
| 不同分辨率 (720p/1080p) | 正常运行 |
| 不同帧数 (13/29/49) | 正常运行 |
