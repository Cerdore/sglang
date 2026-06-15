# OmniDreams Optimization Session Log

> Date: 2026-06-14
> PRs: [#28178](https://github.com/sgl-project/sglang/pull/28178) (performance), [#28182](https://github.com/sgl-project/sglang/pull/28182) (SP)

## Summary

Two feature streams implemented and pushed, each with a separate PR:
- **4 performance optimizations** (T1-T4) targeting compute, memory, and serving latency
- **Sequence Parallelism support** (P0-P2) enabling multi-GPU distributed inference

## Feature 1: Performance Optimizations (PR #28178)

### T1: AdaLN Fusion
- **File**: `runtime/models/dits/omnidreams.py`
- Replaced `nn.LayerNorm + manual scale/shift` with `LayerNormScaleShift(x, shift, scale)`
- On CUDA dispatches to CuTe DSL `fused_norm_scale_shift` kernel

### T2: RoPE Kernel
- **Files**: `runtime/models/dits/omnidreams_rope.py` + `omnidreams.py`
- Added `to_cos_sin_cache()`, dispatch to `_apply_rotary_emb` (FlashInfer/Triton fallback)
- Fixed B-dim broadcast bug

### T3: KV-Cache Split-Copy
- **File**: `runtime/models/dits/omnidreams_kvcache.py`
- Replaced `.clone()` with two-phase `copy_()`, 6.18x CPU speedup

### T4: Text Encoder Cache
- **File**: `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py`
- `OrderedDict` LRU cache (max 32) keyed on prompt string

## Feature 2: Sequence Parallelism (PR #28182)

### P0: Split Attention
- Split `OmniDreamsAttention` into `SelfAttention(USPAttention)` + `CrossAttention(LocalAttention)`

### P1: SP-aware KV-cache
- Documentation for per-rank cache sizing

### P2: Stage Shard/Gather
- Per-rank `sp_size`, `sp_rank`, `local_chunk_tokens`
- RoPE frequency and HD-map token slicing per rank
- Latent gather after AR loop

## Tests

29 CPU-only tests in `test_omnidreams_optimizations.py`:
- T1 (AdaLN): 4 tests
- T2 (RoPE): 9 tests
- T3 (KV-cache): 8 tests
- T4 (Text cache): 4 tests
- SP (Attention split): 4 tests

## Remaining

GPU machine needed to run full test suite.

## Feature 3: Fused QK Norm+RoPE Kernel

### QKNorm+RoPE Fusion
- **File**: `runtime/models/dits/omnidreams.py`
- **What**: `OmniDreamsSelfAttention.forward` now calls `apply_qk_norm_rope()` when cos_sin_cache is available
- **Fuses**: RMSNorm(Q) + RMSNorm(K) + NeoX RoPE into single JIT kernel launch
- **Fallback**: Separate norm + `apply_rope_freqs` when cache unavailable
- **Cross-attention**: NOT fused (no RoPE at cross-attn)
- **Reference**: Mova PR #20530 `fused_rmsnorm_rope` kernel pattern
