# OmniDreams DiT FP8 (P4a) E2E 调试总结

**日期:** 2026-06-15. **分支:** `main` (working tree, 未 commit/push).

## 前置环境

- **GPU:** RTX 5070 (sm_120, 12GB), WSL2, CUDA 13.1, `ssh chen@100.87.72.4`
- **Python:** `/home/chen/.python/sglang/bin/python` (torch 2.11+cu130, editable install)
- **模型:** `/mnt/e/models/omni-dreams-models/` (DiT + Cosmos-Reason1-7B text encoder), LightVAE/TAE ckpt
- **Native ext:** `/mnt/e/gitRepo/sglang/python/sglang/multimodal_gen/native/omnidreams_singleview/build/torch_extensions/omnidreams_singleview_native_4bd22c354a24/omnidreams_singleview_native_4bd22c354a24.so` (43MB, prebuilt)
- **FP8 state:** `/home/chen/omnidreams_test/fp8_state.pt` (239 keys)
- **E2E config:** `/home/chen/omnidreams_test/e2e_dit_fp8_config.json`
- **测试数据:** `/home/chen/omnidreams_test/end_to_end/` (first_frame.png + 29 per-frame HD-map PNGs)

## 已修复的 5 个 wrapper 层 bug

| # | 错误 | 修复 | 文件 |
|---|------|------|------|
| 1 | `_ExecConfig` 缺少 `use_cuda_graph`/`cuda_graph_warmup_iters` | 加了两个属性 | `omnidreams_fp8.py:_ExecConfig` |
| 2 | SGLang wrapper 重复调用 `_ensure_fp8_runtime`(参数不全) | 删了 — `_predict_flow_ext_impl` 内部已处理 | `omnidreams_fp8.py:__call__` |
| 3 | HD-map 传入 16ch 而非 64ch(patch-embedded) | 确认 SGLang pipeline 已做 patchify,修正测试 | debug 脚本修正 |
| 4 | Cross-attn KV `[B,T,D]` 需要 reshape 到 `[B,T,16,128]`(4D) | 加了 reshape | `omnidreams_fp8.py:__call__` |
| 5 | `compute_self_attn_write_start` 需要已初始化的 `BlockKVCache` | 确认 AR loop 已初始化 | debug 脚本修正 |

## 未解决的核心问题

### 现象

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (7040x68 and 2052x2048)
```

发生在 `_predict_flow_ext_impl` → `self._native_extension.optimized_dit_forward`。

### 根因

**SGLang 和 FlashDreams 对 DiT 输入做了不同的预处理:**

1. **SGLang:** `_fuse_padding_mask_into_patch_embed()` 将 `x_embedder` 从 `[2048, 72]` 缩到 `[2048, 68]`(丢弃始终为零的 pad-mask 通道)。SGLang 的 AR loop **在外部已经做了 patchify + x_embed**,传给 `__call__` 的是 **2048-dim embedded hidden states**。

2. **Native kernel (`optimized_dit_forward`):** 内部做 `concat(noisy_2048, mask_4) = 2052` → `x_embedder(2052→2048)` → blocks → `final_layer` → unpatchify。期望的是 FlashDreams 的原始布局(respectively, 72-channel x_embedder and the post-shuffle final_layer layout from `_fuse_shuffle_op_into_last_layer`)。

**两边的 pipeline 不一致**:SGLang 传 pre-embedded tokens,native kernel 又尝试 embed → 形状不匹配。

### 尝试过的修复及其结果

| 尝试 | 方案 | 结果 | 原因 |
|------|------|------|------|
| Padding x_embedder 从 68→72 | Pad 4 个零列恢复原始 72 通道 | 错误变 `72x2048` | kernel 内部 concat noisy+mask=2052,72 不够 |
| Identity x_embedder (独立测试) | 替换为 `[2048,2052]` identity + 4 列零 padding | **独立测试通过** | SGLang 传 pre-embedded tokens → identity pass-through 正确 |
| Identity x_embedder (E2E) | 同上,在 `_SGLTransformerAdapter.__init__` 中修改 | **仍然 68** | 远端 `/mnt/e/gitRepo/.../__pycache__/omnidreams_fp8.cpython-312.pyc` 缓存了旧版本 |
| 清理所有 `__pycache__` 后重跑 | `find ... -name __pycache__ -exec rm -rf` | **运行中** (run v8) | 还有 `/mnt/e` 上的残留 `.pyc` 没删干净;最后一次 run 删了该文件但结果未知 |

### 还需要处理的 Fusion

除了 x_embedder 的 pad-mask fusion,还有 **`_fuse_shuffle_op_into_last_layer()`** 重排了 `final_layer.linear.weight` 从 `(kt kh kw c) → (c kt kh kw)` 顺序。native kernel 内部做 unpatchify 时期望原始顺序,所以 **这个也要 revert**。

### 正确修复路径

**方案 A: 绕过 `_predict_flow_ext_impl`,直接调 `optimized_dit_forward`**

在 `OmniDreamsFP8DiT.__call__` 中:
1. 不创建 `OptimizedDiTExecutor`
2. 直接用 `self._native.extension.optimized_dit_forward(...)` 传已准备好的参数
3. 这样做的好处:完全控制传入的 weights/runtime config,不需要 adapter/executor 的包装层
4. 需要确认的参数:权重 dict(已从 `_ensure_weights_snapshot` 验证 OK),runtime config,invariant tensors

**方案 B: 完整 revert 两个 fusion + 保证缓存清除**

1. 在 `_SGLTransformerAdapter.__init__` 中 undo 两个 fusion:
   - x_embedder: 替换为 `[I_{2048} | 0_{4}]` identity
   - final_layer.linear: undo shuffle `(c kt kh kw) → (kt kh kw c)` 
2. 确保 spawn worker 进程读取最新 `.py`(清除 `/mnt/e` 上的 `.pyc`)

## 调试脚本参考

远端 `/tmp/` 下保留的调试脚本(按序号):

| 脚本 | 用途 |
|------|------|
| `test_prebuilt2.py` | 验证 `_load_prebuilt_extension()` dirname 修复 |
| `test_worker_ext.py` | 验证 spawn 子进程中 `load_extension()` |
| `test_sglang_ext.py` | 完整 ext 加载诊断 |
| `debug_prebuilt.py` | 原始 `.so` 加载测试(module name 错误) |
| `debug_dit_fp8.py` | 步骤 1-13: 完整 executor 创建 + forward |
| `debug_dit_fp8_2.py` | 带真实 BlockKVCache(v1, API 错误) |
| `debug_dit_fp8_3.py` | 正确 BlockKVCache 构造(v2, 未初始化) |
| `debug_dit_fp8_4.py` | 初始化 caches(v3, after_update 导致 fail) |
| `debug_dit_fp8_5.py` | AR-loop 状态 caches(v4, hdmap 16ch bug) |
| `debug_dit_fp8_6.py` | 64ch hdmap fix → native kernel 70% 跑通 |
| `debug_full_trace.py` | 完整 traceback 版本 |
| `debug_hdmap2.py` | 隔离 `_make_cosmos_hdmap_cache` 测试 |
| `check_fp8_keys.py` | FP8 权重 dict 键名检查 |
| `check_hdmap_weight.py` | `additional_patch_embedding` 权重前后对照 |
| `check_shape_after_fix.py` | x_embedder 修复后形状验证 |
| `trace_adapter.py` | `_SGLTransformerAdapter` 每步形状追踪 |

## 已完成的优化(可正常使用)

| 组件 | 状态 | 13f 耗时 | 29f 耗时 | 说明 |
|------|------|----------|----------|------|
| P4b LightVAE-FP8 encode | ✅ | 63s | 63s | 4.3x vs PyTorch encode |
| P1 LightTAE decode | ✅ | 21s | 85s | |
| P3 torch.compile | ✅ | — | — | unit 通过 |
| P0 CUDA Graph | ✅ | — | — | unit 通过 |
| P2 LightVAE PyTorch | ✅ | — | — | unit 通过 |
| P4a DiT FP8 (unit) | ✅ | — | — | build + executor 创建 |
| P4a DiT FP8 (E2E) | ❌ | — | — | 本文档所述问题 |
| **E2E 13f** (无 DiT-FP8) | ✅ | **268s** | — | LightVAE-FP8 + LightTAE |
| **E2E 29f** (无 DiT-FP8) | ✅ | — | **560s** | 完整 29 帧通过 |
