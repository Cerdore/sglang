# P4a DiT FP8 — Weight Mapping Diagnosis & Fix Plan

**Date:** 2026-06-17. **Last dump:** chen WSL2, `dump_fp8_keys_v3.py`.

## Root cause (confirmed by dump)

`cosmos_run_transformer_block_streaming failed at block 0: unknown error`

### ✅ Confirmed: QKV fusion removes per-projection keys

`prepare_cosmos_quantized_streaming_weights` 把 `q_proj.weight` + `k_proj.weight` + `v_proj.weight` → 融合成 `qkv_proj.weight`，删掉了原始 3 个 key。

```
[MISSING] blocks.0.self_attn.q_proj.weight
[MISSING] blocks.0.self_attn.k_proj.weight
[MISSING] blocks.0.self_attn.v_proj.weight
[YES   ] blocks.0.self_attn.qkv_proj.weight  ← 融合版存在（6144, 2048）uint8 + scale fp16
```

其他 17/20 个 bridge 需要的 key 全部存在。

### ✅ Attempted fix: _restore_split_qkv_aliases

在 `__call__` 中 `_ensure_weights_snapshot()` 后从 qkv_proj 切分恢复单个 Q/K/V 权重 key。
**结果：** 仍 crash `block 0: unknown error`。

### ⚠️ Remaining problem: dtype mismatch after restore

恢复的 q_proj/k_proj/v_proj 是 `torch.uint8`（直接从 uint8 qkv_proj 切片），没有对应的 `_fp8_prepared_scale`。
Bridge 的 `resolve_fp8_or_bf16` 逻辑：

1. 检查 `{key}_fp8_prepared_scale` → **不存在**（只有 qkv_proj 有 scale）
2. 走 bf16 路径 → 把 uint8 数据当作 bf16 读取 → 垃圾值 → `cudaErrorUnknown`

### ⚠️ Secondary problem: precomputed block_mods shape

Streaming path 使用 precomputed block_mods（由 `_make_cosmos_block_mod_cache` 从 t_emb + AdaLN weights 预计算），
绕过了 per-block 的 `adaln_modulation.*.weight` 查找。这需要 `_supports_block_mod_cache=True`。
形状问题尚未完全验证。

### ⚠️ Other known issues (not yet root-caused)

1. **x_embedder pad shape**: SGLang DiT 的 x_embedder 是 `(2048, 68)`（pad-mask fused out），
   但 bridge 期望 `(2048, 72)`。当前 eye-pad fix 可能形状不对。
2. **cross_attn weight shapes**: bridge 期望 `(2048, 1024)` for k_proj/v_proj（cross-attn context dim ≠ hidden dim），需要验证。

## Next step: test with cosmos_forward first

`cosmos_forward` (streaming_dit_bridge.cu 的 PyTorch/ATen 路径) 使用同一份 weights dict 但在 PyTorch 中运行。
先用它验证权重 dict 本身是否正确，排除了 CUDA 特定问题后再切回 streaming。

```python
# In __call__: try cosmos_forward instead of _predict_flow_ext_impl
out = ex._native_extension.cosmos_forward(
    noisy, mask, hdmap_for_ext, hdmap_embed, timestep_b, rope_freqs,
    inv.t_emb, inv.t_emb_silu, inv.adaln_lora, inv.final_shift, inv.final_scale,
    rope_cos, rope_sin, inv.block_mods_sa, inv.block_mods_ca, inv.block_mods_mlp,
    k_cross, v_cross, k_self, v_self, write_start,
)
```

## Key files

1. `runtime/models/dits/omnidreams_fp8.py` — SGLang-side FP8 dispatch, `_SGLTransformerAdapter`, `_restore_split_qkv_aliases`
2. `native/omnidreams_singleview/python/optimized_dit.py` — vendored `OptimizedDiTExecutor`, weight snapshot, invariant tensor cache
3. `native/omnidreams_singleview/python/cosmos_fp8_utils.py` — vendored `prepare_cosmos_quantized_streaming_weights`
4. `native/omnidreams_singleview/python/cosmos_weights.py` — vendored `prepare_cosmos_streaming_weights` (BF16 path)
5. `native/omnidreams_singleview/src/dit_streaming/pyext/streaming_dit_bridge.cu` — C++ bridge, `get_w()`, `cosmos_forward()`, `optimized_dit_forward()`, `populate_block_streaming()`
6. `native/omnidreams_singleview/src/dit_streaming/kernels/cosmos_block.cu` — CUDA orchestrator: `cosmos_run_transformer_block_streaming`
7. `native/omnidreams_singleview/src/dit_streaming/kernels/cosmos_block.cuh` — `CosmosBlockWeights`/`CosmosBlockParams` structs

## Risk / Effort

- **Effort**: ~4-6 hours remaining (need GPU + CUDA debugging)
- **Risk**: SGLang DiT architecture may have subtle differences from FlashDreams DiT
  (different block naming, different AdaLN layout, different channel counts)
- **Fallback**: SGLang BF16 eager path functional; FP8 is performance-only
