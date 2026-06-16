# P4 — OmniDreams Native FP8 Acceleration: Full Scope + Implementation Plan

> **Note:** Config fields referenced below (`use_fp8_dit`, `fp8_dit_attention_backend`, etc.) have been replaced by three-state `native_dit_acceleration` / `native_dit_backend`. See [omnidreams_config_migration.md](omnidreams_config_migration.md).

**Correction to the earlier draft:** FP8 is **not** just the LightVAE. FlashDreams'
`integrations/omnidreams/omnidreams_singleview/` ships native FP8 kernels for
**two** subsystems, and the **DiT is the dominant target** (it is the bulk of the
~43k-LOC native tree and the main latency cost):

| Subsystem | Native source | What runs in FP8 |
|---|---|---|
| **DiT (Cosmos 2B)** | `src/dit_streaming/` (~30k LOC) | The whole per-chunk DiT forward: all linear GEMMs (FP8 tensor-core, RCR contract), attention (FP8 flash / SageAttention-3 / SpargeAttn sparse), AdaLN-LoRA + modulate, streaming self/cross KV caches. |
| **VAE (LightVAE encoder)** | `src/vae_streaming/` (~6k LOC) | The pruned Wan encoder conv/GEMM + attention in FP8 (TIN16 format). |

So P4 has two halves: **P4a DiT-FP8** (big win) and **P4b VAE-FP8** (the original
draft). Both gate behind native availability + a calibrated state.

---

## 1. DiT FP8 (P4a) — what FlashDreams ships

**Entry op:** `optimized_dit_forward(x_new, condition_mask_patched, hdmap_patched,
timesteps, rope_emb, k_cross_caches[], v_cross_caches[], k_self_caches[],
v_self_caches[], self_attn_write_start, weights: dict, config: dict) ->
[B, V, T, HW, D_out]` (pre-unpatchify). This replaces the **entire** Cosmos DiT
forward for one AR chunk — i.e. the `self.transformer(...)` call in the SGLang AR
loop (the same call P0 wraps in a CUDA graph and P3 compiles).

**FP8 linear contract (RCR):** weights are `weight_fp8_u8 [out, in]` raw E4M3
bytes in PyTorch Linear layout + per-output-channel `weight_scale [out]` dequant.
Kernel variants (in `kernels/ops.cu`, `cosmos_fp8_two_gemm.cu`, `block_quant.cu`):
`linear_fp8`, `..._out_fp8`, `..._gelu_out_fp8`, `..._scaled_bf16`,
`..._residual_scaled_bf16`, with runtime **tile autotune**
(`cosmos_test_fp8_linear_tile_selection`).

**FP8 attention:** three backends, selected per shape:
- `cosmos_fp8_flash` / `cosmos_fp8_flash_tc` (`kernels/cosmos_fp8_flash*.cu`) —
  native FP8 flash attention.
- **SageAttention-3** (`kernels/sage3_attention.cu`, thirdparty `SageAttention`) —
  `use_sage3_fp8_attention`.
- **SpargeAttn** (`kernels/sparge_attention_*.cu`) — `use_sparge_attention`,
  `sparge_topk_ratio` block-sparse attention.

**Modulation in FP8:** `cosmos_adaln_lora.cu`, `cosmos_modulate.cu`,
`cosmos_block.cu`/`transformer_block.cu` — the AdaLN-LoRA + per-block shift/scale/
gate done natively.

**Streaming:** self + cross KV caches are passed in as tensor lists and written at
`self_attn_write_start` (the native analogue of `BlockKVCache`). The bridge owns a
streaming workspace (`_make_cosmos_streaming_workspace`) with pre-allocated FP8
scratch for linear + attention (`linear_fp8_scratch`, `attn_{q,k,v}_fp8`, ...).

**Python side (FlashDreams):** `python/optimized_dit.py` (1854 LOC) builds the
invariant/cache tensors + the `weights`/`config` dicts and calls the native op;
`python/cosmos_fp8_utils.py` + `python/cosmos_weights.py` do the offline FP8 weight
quantization/packing (per-output-channel E4M3 + scale).

---

## 2. VAE FP8 (P4b)

**Entry:** `omnidreams_vae_create_wan_encoder_fp8(state) -> encoder`,
`omnidreams_vae_encode_wan_fp8(encoder, input, use_cache=True) -> fp16 latent`,
`reset_cache` / `release_resources` / `is_cuda`, `omnidreams_vae_backend_status`.
Kernels: `lightvae_fp8_ops.cu`, `lightvae_fp8_warp_mma_stages.cu`,
`lightvae_fp8_direct_stages.cu`, `lightvae_fp8_attention.cu` (TIN16
`[T, C/16, H, W, 16]` uint8, `kFp8ChannelsPerSlice=16`). Calibrated state from
`scripts/export_lightvae_fp8_state.py` (per-channel E4M3, `scale_max=24`); env
`OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH`. `python/vae_weights.py` (1438 LOC) does the
weight prep. This is a drop-in FP8 backend behind the P2 `LightVAEEncoder`.

---

## 3. Build + third-party dependencies (GPU/`sm_120` only)

- **Extension:** `src/omnidreams_singleview_ext.cpp` (`PYBIND11_MODULE`), built via
  `tools/native_build.py`; **requires CUDA**, `CUDA_ARCH_LIST` = `sm_120` (consumer
  Blackwell: RTX 5090 / RTX PRO 6000). Datacenter Blackwell (`sm_100`) is NOT this
  path. The ext records source SHAs/fingerprints and refuses mismatched builds.
- **CUTLASS** pinned `f3fde58…` + `patches/cutlass/sm120-tma-pool.patch` + header
  overlays (`patches/cutlass/include`).
- **SageAttention** pinned `d1a57a5…`; **SpargeAttn**. Synced by
  `tools/sync_thirdparty.py` from `thirdparty_sources.json`.

**Consequence:** none of the kernels can be compiled or run on this CPU/macOS dev
box. Build + numerical verification happen on the `sm_120` GPU host.

---

## 4. Implementation plan into SGLang

The native code is large and GPU-only, so the implementation is **vendor the
native sources + build wiring + thin Python dispatch + flags**, validated on GPU.
No re-authoring of the CUDA — we reuse FlashDreams' kernels verbatim.

### 4a. Native extension (vendored, GPU-built)
1. Vendor `omnidreams_singleview/` (src + patches + tools + thirdparty manifest)
   under `sgl-kernel`-style native dir (or a dedicated
   `python/sglang/multimodal_gen/native/omnidreams_singleview/`), preserving the
   build + SHA-pinning. Add an opt-in build step (it is heavy + `sm_120`-only).
2. A loader (`native/acceleration.py` analogue) with the FlashDreams
   `disabled / auto / required` strategy + capability/symbol checks, so a missing
   or off-`sm_120` build cleanly falls back to the existing bf16 path.

### 4b. DiT FP8 dispatch (P4a)
3. NEW `runtime/models/dits/omnidreams_fp8.py`: an FP8 DiT path that
   - quantizes the OmniDreams DiT weights to the RCR E4M3 + per-channel scale
     layout (port `cosmos_fp8_utils.py` / `cosmos_weights.py`), once at load,
   - builds the `weights`/`config` dicts + streaming workspace and calls
     `optimized_dit_forward`, exposing the SAME signature the AR loop calls
     (`hidden_states, encoder_hidden_states, timestep, condition_video_input_mask,
     rope_freqs, hdmap_condition, kv_caches, cross_attn_kv, view_indices`).
4. Wire into the AR loop in `model_specific_stages/omnidreams.py`: when
   `use_fp8_dit`, route `_dit_call` through the FP8 path instead of
   `self.transformer(...)`. **Composes with P0/P3**: FP8 replaces the eager DiT;
   CUDA-graph capture (P0) wraps the FP8 op the same way; torch.compile (P3) is
   bypassed for the native path.
5. Map SGLang's `BlockKVCache._k/_v` to the native self/cross cache tensor lists +
   `self_attn_write_start` (the steady-state window write position).

### 4c. VAE FP8 dispatch (P4b)
6. Extend `runtime/models/vaes/omnidreams_light_vae.py::LightVAEEncoder` with an
   `fp8` backend: load the calibrated state, `create_wan_encoder_fp8`, route
   `encode` through `omnidreams_vae_encode_wan_fp8`; keep the PyTorch path as
   fallback. Port `export_lightvae_fp8_state.py` as the offline calibration tool.

### 4d. Flags + env
7. `OmniDreamsPipelineConfig`: `use_fp8_dit`, `use_light_vae_fp8`,
   `fp8_dit_state_path`, `light_vae_fp8_state_path`, plus attention-backend knobs
   (`fp8_attention="flash"|"sage3"|"sparge"`, `sparge_topk_ratio`).
8. `envs.py`: `SGLANG_OMNIDREAMS_FP8_DIT`, `SGLANG_OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH`
   (the SGLang-prefixed analogue of `OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH`).

### 5. CPU-testable vs GPU-only

| CPU-testable now | GPU/`sm_120`-only |
|---|---|
| FP8 weight quant/pack math (per-channel E4M3 + scale; RCR layout) vs a bf16 reference | Native ext build (CUTLASS+Sage+Sparge, `sm_120`) |
| `weights`/`config` dict assembly + shape/contract tests with the native op mocked | `optimized_dit_forward` numerics + latency |
| Calibration-state export round-trip (keys, scales, version fields) | `NativeWanVaeEncoderFp8` numerics + latency |
| Acceleration-strategy fallback (disabled/auto/required, missing-ext path) | FP8 vs bf16 FVD / accuracy |

### 6. Verification (GPU)
- Build the ext on `sm_120`; `*_is_built` / `*_is_runtime_supported` green.
- DiT-FP8 vs bf16: per-chunk latent tolerance (FP8 is lossy — agree a budget),
  end-to-end FVD, latency (expect the largest single speedup here).
- VAE-FP8 vs bf16 encode: latent tolerance + HD-map motion regression.
- Attention-backend sweep (flash / sage3 / sparge) for speed/quality.
- Compose with P0+P3; confirm CUDA-graph capture of the FP8 op.

---

## 7. Risks / open questions

- **Magnitude:** ~43k LOC vendored native + 3 patched third-party libs; `sm_120`
  only; entirely GPU-gated. The SGLang-side new code (Python dispatch + flags +
  weight packing) is bounded; the native is reused verbatim.
- **Calibrated states** for both DiT and VAE must be generated (port the export
  scripts) or obtained.
- **KV-cache mapping** SGLang `BlockKVCache` ↔ native self/cross cache lists +
  `self_attn_write_start` needs careful alignment (numerics gate).
- **Build integration**: how to host a heavy opt-in `sm_120` build in the SGLang
  tree without burdening the default install (separate extra / lazy build).
- **Licensing/vendoring**: SageAttention + SpargeAttn + CUTLASS are third-party;
  vendor via the existing `sync_thirdparty.py` manifest, don't copy blindly.
