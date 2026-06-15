# P4a: 将 OptimizedDiTExecutor 重构为 SGLang 原生实现

**Status:** deferred (blocked by: [[p4a-optimized-dit-flashdreams-import-bug]], worked around by: stub replacement in `optimized_dit.py`)

## Problem

`native/omnidreams_singleview/python/optimized_dit.py` 中的 `OptimizedDiTExecutor` 是 FlashDreams 的 vendored 代码，依赖：

- `CosmosTransformer` — FlashDreams 的 DiT 基类
- `CosmosTransformerCache` — FlashDreams 的 streaming cache
- `CosmosDiTNetworkCache` — 内部 cache 实现
- `BlockKVCache` — FlashDreams 的 KV cache
- `CUDAGraphWrapper` — FlashDreams 的 CUDA graph wrapper

SGLang 已有对应实现：
- `OmniDreamsDiT` — DiT 模型（`runtime/models/dits/omnidreams.py`）
- `BlockKVCache` — KV cache（`runtime/models/dits/omnidreams_kvcache.py`）
- `CUDAGraphWrapper` — CUDA graph（`runtime/models/dits/omnidreams_cuda_graph.py`）

## Current workaround (方案A)

Stub 替换 `optimized_dit.py` 顶部的所有 flashdreams/omnidreams import。PEP 563 让类型注解延迟求值，所以只要这些名字在模块命名空间存在即可。SGLang 的 `OmniDreamsFP8DiT.__call__` 只调用 `_predict_flow_ext_impl` → `_ensure_weights_snapshot` → `_ensure_invariant_tensors` → `_ensure_rope_tensors` → `_apply_runtime_config` → native `optimized_dit_forward`。

## Proposed fix (方案B)

重写 `OptimizedDiTExecutor` 为 SGLang 原生实现：

1. **去掉 `CosmosTransformer` 继承** — `OptimizedDiTExecutor` 目前 subclass `CosmosTransformer` 只为了复用 `predict_flow` 里的 mask/pin 逻辑。SGLang 的 AR loop 已在这些逻辑之外单独处理。

2. **直接用 SGLang 的 `BlockKVCache`** — `compute_self_attn_write_start` 应接受 SGLang 的 `BlockKVCache` 而不是 FlashDreams 的。

3. **直接使用 `sglang.multimodal_gen.runtime.models.dits.omnidreams_cuda_graph.CUDAGraphWrapper`** — 不需要 FlashDreams 的版本。

4. **去掉 CP（context parallel）stub** — `cat_outputs_cp` / `split_inputs_cp` 在 SGLang 的 SP 实现中不适用（OmniDreams 不支持 SP）。

## Scope

- `omnidreams_fp8.py` — 更新 `_SGLTransformerAdapter`，去掉对 stub 的依赖
- `optimized_dit.py` — 重构 `OptimizedDiTExecutor`，去掉 flashdreams 继承
- 可能需要微微调整 `_predict_flow_ext_impl` 的 pin/mask hook（SGLang 已在上层处理）
