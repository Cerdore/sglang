# P4a DiT FP8 — Block-0 Crash Diagnosis & Fix Plan

> ## ✅ 2026-06-18 SESSION 4 — RESOLVED (read this first; supersedes everything below)
>
> Native FP8 DiT now runs **end-to-end on chen's RTX 5070 (sm_120)**. Verified:
> `RESULT_OK`, `.mp4` written, 0 block failures, no arch trap, no `999`.
>
> **There were TWO independent root causes, and the earlier "consumer sm_120
> can't run sm_120a / fall back to eager" conclusion was WRONG.**
>
> ### Root cause #1 — a stale `sm_120` (non-`a`) `.so`
> The `Arch conditional MMA instruction … Aborting` trap was a build-flag bug,
> not a hardware limit. `sm_120a` is consumer Blackwell's OWN arch-specific
> target (RTX 50xx / RTX PRO 6000); datacenter Blackwell is `sm_100a`. CUTLASS
> FP8 GEMMs use arch-conditional MMA atoms that only emit real instructions
> under `sm_120a`; built as plain `sm_120` they compile to a device-side
> `CUTE_INVALID_CONTROL_PATH` trap. The loader was serving a stale `.so` from an
> earlier `TORCH_CUDA_ARCH_LIST=12.0` (no `a`) build (it loads ANY existing
> `.so` regardless of arch/fingerprint). Evidence: `cuobjdump` → `arch = sm_120`,
> `build.ninja` → `compute_120,sm_120`. Toolchain was fine all along: driver
> 591.86 (≥570), CUDA 13.0 (≥12.8), torch 2.11 maps `12.0a → compute_120a,sm_120a`.
> **Fix:** rebuild for `sm_120a` (purge `build/torch_extensions/*` first).
>
> ### Root cause #2 — cuDNN-frontend API drift (NOT sm_120-related)
> After #1, a SECOND `cudaErrorUnknown (999)` surfaced at block-0 self-attention
> (`cosmos_block.cu:2669` → `run_cudnn_fmha_packed_qkv`). cuDNN logging proved
> cuDNN DID build a valid flash-attention plan for the 5070 — the failure was a
> frontend VALIDATION error: `generate_stats attribute not set`
> (`scaled_dot_product_flash_attention.h:194`). The vendored cudnn-frontend
> (v1.14.0 / cuDNN 9.19) hard-requires the SDPA node to set `generate_stats`.
> The BF16 builder (`attention.cu:1161`) omitted it; the FP8 sibling (`:1313`)
> already had it. Would fail on ANY GPU with this frontend version.
> **Fix:** `.set_generate_stats(false)` on the BF16 SDPA node (lossless;
> `false` = inference, no softmax LSE).
>
> ### Code changes (canonical, mirrored to local repo)
> - `runtime/models/dits/omnidreams_fp8.py` — REMOVED the false `sm_120` guard.
> - `native/singleview_loader.py` — added `_normalize_blackwell_arch` (forces a
>   bare `12.0`/`12.1` arch token → `…a`; idempotent, multi-arch safe) so a
>   stray `TORCH_CUDA_ARCH_LIST=12.0` can't silently produce a trapping build.
> - `native/.../kernels/attention.cu` — `generate_stats(false)` on BF16 SDPA.
>
> ### Follow-ups (not blocking)
> - `_extension_name` should encode the arch so a stale non-`a` `.so` cannot be
>   silently reused (the mechanism behind #1).
> - Legacy WAN cuDNN SDPA sites (`attention.cu` ~1527/1767/2057) likely also need
>   `generate_stats` for this frontend version — out of scope for OmniDreams.

> ## 🔴🔴 2026-06-18 SESSION 3 — DEFINITIVE LOCALIZATION (read this first)
>
> **All three prior theories are superseded.** SESSION 2 traced the failure to
> block-0 self-attention, but stopped at `err=999`. SESSION 3 added CUTLASS GEMM
> instrumentation (`[DIAG]` fprintf at every status swallow in `ops.cu` +
> `cosmos_block.cu` + `streaming_dit_bridge.cu`), built a fresh instrumented
> `.so` on chen, and ran the E2E repro — producing **conclusive negative evidence**.
>
> ### What the instrumentation proved
>
> **Zero `[DIAG]` lines appeared.** Every CUTLASS GEMM (`rcr_fp8`, `rcr_fp8_gelu`,
> `cosmos_linear_prequantized_fp8`, `attn_batched_gemm`) completed successfully. The
> QKV projection, output-proj, MLP layer1/layer2, and cross-attn GEMMs are all **fine**.
>
> The only errors on stderr were:
> ```
> ERROR : Arch conditional MMA instruction used without targeting appropriate
> compute capability. Aborting.
> ```
>
> ### Root cause confirmed: sm_120 vs sm_120a architecture mismatch
>
> RTX 5070 is **sm_120** (consumer Blackwell). The FP8 cuDNN attention path
> (`fp8_cudnn`) uses **sm_120a** (datacenter Blackwell) MMA instructions that the
> consumer GPU's driver rejects at kernel launch time.
>
> This is a **CUDA driver-level rejection**, not a CUTLASS `Status::kSuccess`
> failure — which is why every `cutlass::Status` check returned `kSuccess` (no
> DIAG output) and the error surfaced as `cudaErrorUnknown` only when the next
> synchronization point checked `cudaGetLastError()`.
>
> The "block 0" attribution is also explained: the first kernel that touches
> sm_120a MMA instructions is the block-0 self-attention, so the async fault
> lands there. But the *cause* is architecture incompatibility, not a block-0
> code defect.
>
> ### Why FlashDreams works on the same GPU
>
> The 2026-06-15 "13f FP8 268.4s" reference was FlashDreams running on chen's
> RTX 5070. FlashDreams likely has a **sm_120 BF16 attention fallback** or uses
> cuDNN BF16 FMHA when FP8 attention isn't supported on consumer hardware. SGLang's
> vendored tree hardcodes `auto → fp8_cudnn` without the same fallback logic.
>
> ### Build environment quirks
>
> The Sage3/Sparge CUDA sources (`sage3_attention.cu`, `sparge_attention_sm89_inst.cu`,
> `sage3_blackwell_api_shim.cu`, `sage3_fp4_quant_shim.cu`) use sm_200+ MXFP4
> instructions that ptxas rejects on sm_120 targets. They must be replaced with
> stubs that export the expected C++ symbols but return `cudaErrorNotSupported`.
>
> ### 2026-06-18 build notables
>
> - **Clean build time:** ~5 min at `MAX_JOBS=16` with CUDA 13 on 20 cores.
> - **Sage3/Sparge:** Must replace `sage3_attention.cu` → `sage3_attention_stub.cu`
>   in the loader source list, stub `sparge_attention_sm89_inst.cu` with
>   exact-matching C++ signature, and set `HAS_SAGE3=0`/`HAS_SPARGE=0` macros.
> - **CUTLASS stamp:** Bypassed via `validate_thirdparty → _thirdparty_info_no_validation`.
> - **`.so` loading:** loader prefers ANY existing `.so` regardless of source
>   fingerprint. Must delete stale `.so` dirs before each rebuild.
>
> ### FP8 unit test results (chen, 2026-06-18)
>
> **22 passed, 2 skipped, 0 failed** (previously 3 failures were "no native ext
> built" — now all resolved with the Sage3/Sparge-stubbed `.so`).
>
> ### Next step: BF16 attention fallback
>
> The only viable fix on consumer sm_120 GPUs: route attention through
> `cudnn_bf16` when `native_dit_acceleration="required"`. The FP8 linear GEMMs
> (QKV/o-proj/MLP/cross-attn) all work correctly — only attention's FP8 cuDNN
> path is broken.
>
> Config change: add `"native_dit_attention_backend": "cudnn_bf16"` to the
> pipeline config JSON. This keeps 6 of 8 GEMMs per block in FP8 (all linears)
> while attention uses BF16 cuDNN FMHA. See `optimized_dit.py:1229-1233` for the
> backend routing logic.
>
> ### Diagnostic instrumentation in tree (REVERT before any commit)
>
> Three files contain uncommitted diagnostic-only edits:
> - `ops.cu`: `[DIAG] rcr_fp8 ...` / `[DIAG] rcr_fp8_gelu ...` fprintf at every
>   CUTLASS `return cudaErrorUnknown` in the SM120 paths
> - `cosmos_block.cu`: `[DIAG] cosmos_linear_prequantized_fp8 ...` fprintf at
>   fused + rcr_fp8 failure paths
> - `streaming_dit_bridge.cu`: `[DIAG] attn_batched_gemm ...` fprintf at attention
>   GEMM failure paths
>
> `git checkout -- <these files>` to revert.

---

**Date:** Original 2026-06-17, updated 2026-06-18 (SESSION 3 — definitive).
**GPU box:** chen WSL2 (`chen@100.87.72.4`, RTX 5070, sm_120, 12GB), synced to
`feat/omnidreams-p0-p4b-optimizations` @ current HEAD.

---

**Date:** 2026-06-17 (rewritten after source-level audit of the C++ bridge).
**GPU box:** chen WSL2 (`chen@100.87.72.4`, RTX 5070, sm_120, 12GB), now synced to
`feat/omnidreams-p0-p4b-optimizations` @ `81f033102`.

> ⚠️ **This supersedes the previous version of this doc.** The earlier theory
> ("QKV fusion removes per-projection keys → C++ bridge null-pointer") was
> **wrong**, and the `_restore_split_qkv_aliases` fix built on it has been
> removed from `omnidreams_fp8.py`. See "Corrected analysis" below.
>
> ⚠️⚠️ **The "CUTLASS GEMM failure" analysis in the rest of this doc is ALSO
> superseded** by the 2026-06-17 SESSION 2 box above — the failure is block-0
> self-attention, verified at runtime, not any GEMM.

## Symptom

```
cosmos_run_transformer_block_streaming failed at block 0: unknown error
```

`native_dit_acceleration="required"` (or `SGLANG_OMNIDREAMS_FP8_DIT=1`). Eager
BF16 path is unaffected.

## ❌ What the previous diagnosis got wrong

The previous doc claimed the C++ bridge looks up per-projection
`q_proj`/`k_proj`/`v_proj` keys, and that fusing them into `qkv_proj` (and
dropping the originals) left the bridge with a null pointer.

**This is false.** Confirmed by reading the bridge source:

- The streaming entry point `optimized_dit_forward` (`streaming_dit_bridge.cu:1542`)
  **expects the fused** `self_attn.qkv_proj.weight`. It explicitly:
  - checks `weights.contains("...self_attn.qkv_proj.weight")` and that it is
    `at::kByte` / uint8 FP8 (lines 1893-1894, 2646-2647),
  - reads the `_fp8_prepared` / `_fp8_prepared_scale` aliases (lines 2541, 2588, 2679),
  - validates shape `[3*K, K] = [6144, 2048]` (line 2655) — **exactly** the shape
    the dump reported.
- The separate-`q_proj`/`k_proj`/`v_proj` `get_w` calls (lines 1157-1160, 1199-1202)
  belong to the **BF16 / `cosmos_forward` (DDPM)** path, not the streaming FP8 path.

So dropping the split Q/K/V keys is **correct by design**
(`prepare_cosmos_quantized_streaming_weights` →
`drop_split_self_attn_qkv` defaults to `True`,
`cosmos_fp8_utils.py:589-590`, fusion at `:392-399`). The whole weight + bridge
contract is byte-identical to FlashDreams, which runs this exact fused-qkv FP8
path successfully **on this same RTX 5070** (the 2026-06-15 "13f FP8 268.4s"
reference number is FlashDreams).

`_restore_split_qkv_aliases` was therefore a fix for a non-existent problem. It
sliced uint8 bytes out of the fused qkv with no scale and made things worse. It
is gone from the current `omnidreams_fp8.py` (255 lines, no such method).

## ✅ Corrected analysis: it is a CUTLASS GEMM failure

`"block 0: unknown error"` is `cudaErrorUnknown`. The CUTLASS GEMM helper in the
bridge **returns `cudaErrorUnknown` whenever a `cutlass::Status` is not success**,
and **discards the real status** (`streaming_dit_bridge.cu:256-260`):

```cpp
cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
status = gemm_op(stream);
if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
return cudaSuccess;
```

This is decisive about what the bug is **not**:

- **Not a missing key.** `get_w` throws a clean `"missing weight key '<k>'"`
  (`:269-272`) — we don't see that.
- **Not a mismatched fused-qkv shape.** The streaming path has explicit
  `TORCH_CHECK` shape guards (e.g. `:2655`) that would produce a readable error —
  we don't see those either.

So the block-0 weights satisfy the key + shape contract, but **a CUTLASS GEMM in
block 0 rejects its problem at `initialize()` or execution**. Likely causes, in
rough priority:

1. **A non-shape-guarded weight is wrong** — most suspect is `x_embedder`
   (`x_embedder.proj.1.weight`, bridge `:1331`, expected `[D, C_in*pt*ph*pw]`).
   SGLang fuses the pad-mask channel out at load time → `[2048, 68]`, and the
   vendored snapshot has its own pad-mask-strip + reshape logic
   (`optimized_dit.py:37`). If these disagree, the patch-embed GEMM feeding block 0
   gets a wrong `K` and CUTLASS fails. (`final_layer.linear`, bridge `:1413`, with
   SGLang's channel-shuffle baked into the weight in `post_load_weights`, is a
   second candidate but runs after the blocks.)
2. **sm_120 / problem-size limitation in CUTLASS.** The vendored tree ships a
   `patches/cutlass/sm120-tma-pool.patch`, so sm_120 GEMM support is known-fragile.
   FlashDreams works on this GPU, though, so a pure sm_120 issue would only bite
   if SGLang feeds a different problem size.
3. **Alignment / dtype of an activation** entering the block-0 GEMM.

The divergence point is the snapshot source: `_ensure_weights_snapshot`
(`optimized_dit.py:1081`) builds the FP8 dict from
`self.network.state_dict()` where `self.network` is SGLang's `OmniDreamsDiT`
(via `_SGLTransformerAdapter`), **not** FlashDreams' `CosmosTransformer`. The
byte-identical bridge demands a CosmosTransformer-compatible layout; any
SGLang-specific weight transform (pad-mask fuse, channel-shuffle, TP sharding)
that the FlashDreams snapshot doesn't expect can produce a GEMM-incompatible
weight that passes the coarse shape guards.

## Next step (decisive): un-swallow the CUTLASS status

Make the hidden failure visible instead of guessing which GEMM:

1. In `streaming_dit_bridge.cu`, before each `return cudaErrorUnknown` at
   `:257` and `:259`, `fprintf(stderr, ...)` the real
   `cutlass::cutlassGetStatusString(status)` plus the problem dims `(M, N, K)` and
   a tag identifying the call site (qkv / out_proj / mlp1 / mlp2 / x_embedder).
2. Rebuild the extension on chen (3rdparty already synced; `.so` JIT-rebuilds).
3. Rerun the failing generate with `--text-encoder-cpu-offload --vae-cpu-offload`
   (12GB card) and `SGLANG_OMNIDREAMS_FP8_DIT=1`.

This converts "block 0: unknown error" into "GEMM <name> failed:
<cutlass status> at M×N×K = ...", pinpointing the exact failing matmul and
whether it is a shape, alignment, or sm_120 problem.

### If the failing GEMM is x_embedder / patch-embed
Compare SGLang `OmniDreamsDiT` `x_embedder.proj.1.weight` shape against the
bridge's expected `[D, C_in*pt*ph*pw]`, and check whether the snapshot's
pad-mask-strip double-applies on SGLang's already-stripped `[2048, 68]`.

### If it is a per-block linear (qkv / mlp)
Dump that block-0 prepared weight (shape, dtype, contiguity, scale shape) and
compare byte-for-byte against what FlashDreams produces for the same checkpoint.

## Key files (with the lines that matter)

1. `runtime/models/dits/omnidreams_fp8.py` — `OmniDreamsFP8DiT.__call__` (no more
   qkv restore), `_SGLTransformerAdapter`.
2. `native/omnidreams_singleview/python/optimized_dit.py` —
   `_ensure_weights_snapshot` (`:1081`), x_embedder pad-mask handling (`:37`),
   `_predict_flow_ext_impl` (`:1640`).
3. `native/omnidreams_singleview/python/cosmos_fp8_utils.py` —
   `prepare_cosmos_quantized_streaming_weights` (`:551`), drop-split default
   (`:589-590`), fusion (`:392-399`).
4. `native/omnidreams_singleview/src/dit_streaming/pyext/streaming_dit_bridge.cu` —
   CUTLASS status swallow (`:256-260`), `get_w` (`:269`), `optimized_dit_forward`
   (`:1542`), fused-qkv read + shape guard (`:1893`, `:2646-2659`), `x_embedder`
   read (`:1331`), `final_layer.linear` read (`:1413`).
5. `native/omnidreams_singleview/src/dit_streaming/kernels/cosmos_block.cu` —
   `cosmos_run_transformer_block_streaming` (emits the "block N" error).

## Risk / Effort

- **Effort:** instrument + rebuild + one generate run ≈ 30-60 min on chen.
- **Risk:** edits vendored "byte-identical" source — keep the change as a
  diagnostic-only diff, `git checkout` afterward. Real fix is likely small
  (a weight-layout adapter in the snapshot, or an sm_120 GEMM config tweak).
- **Fallback:** `native_dit_acceleration="disabled"` (default). P1+P2+LightVAE-FP8
  (P4b) deliver the bulk of the E2E speedup without DiT FP8.
