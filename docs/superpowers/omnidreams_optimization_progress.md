# OmniDreams Optimization — Progress Snapshot

**Date:** 2026-06-14. **Branch:** `main` (working tree only — **NOT committed, NOT pushed, no PR**).
**Dev box:** CPU-only macOS. GPU (`sm_120`) validation happens in a later session over SSH — user will open the GPU host on request.

Plan: `docs/superpowers/omnidreams_optimization_plan.md`. FP8 design: `docs/superpowers/omnidreams_p4_fp8_design.md`.
Reference source (local): `/Users/cerdore/gitRepo/flashdreams`. Checkpoints: `/Users/cerdore/gitRepo/models/{omni-dreams-models/single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt, lighttaew2_1.pth, lightvaew2_1.pth}`.

## CPU test invariant (IMPORTANT)
- venv python: `/Users/cerdore/.python/sglang/bin/python` (torch 2.9.0, editable sglang).
- **Prefix forward-path pytest with `TORCHDYNAMO_DISABLE=1`** — Triton is absent on macOS so the T1 `@torch.compile` LayerNormScaleShift kernels fail otherwise.
- Installed into the venv this session: `imageio imageio-ffmpeg diffusers==0.37.0 moviepy opencv-python-headless av scikit-image addict cache-dit trimesh xatlas ruff` (were missing).

## Status by phase

| Phase | What | Status | Verified on CPU |
|---|---|---|---|
| **P3** torch.compile | Removed over-conservative `assert not torch.compiler.is_compiling()` in DiT `forward()` (single-chunk forward is compile-safe under fullgraph=False; AR loop lives in the stage). Compile already routes via `DenoisingStage._maybe_enable_torch_compile`. | ✅ code done | ✅ suite green; GPU smoke pending |
| **P0** CUDA Graph | Ported `CUDAGraphWrapper`+`set_or_copy` → `omnidreams_cuda_graph.py`. Wired into AR loop: capture the 3 per-chunk `self.transformer(...)` calls **only at KV steady-state**; fill phase + `pin` + `before/after_update` stay eager. Flag `enable_cuda_graph` + env `SGLANG_OMNIDREAMS_CUDA_GRAPH`. | ✅ code done | ✅ eager path (AR rollout test) green; graph capture/replay numerics GPU-pending |
| **P1** LightTAE decode | Ported TAEHV (impl+checkpoint remap) → `vaes/taehv.py` + `LightTAEDecoder` (own latent mean/std, single-pass). Custom decode stage `OmniDreamsLightTAEDecodingStage` (skips Wan scale_and_shift). Registered as `vae_decoder` module. Flags `use_light_tae`+`light_tae_path` + one-time quality warning. | ✅ code done | ✅ real ckpt loads (64/64 keys, 0 meta), decode shape `[1,16,3,8,8]→[1,3,9,64,64]` finite; FVD GPU-pending |
| **P2** LightVAE encode | Ported pruned (0.75) streaming Wan encoder → `vaes/omnidreams_light_vae.py` + `LightVAEEncoder` (`.encode().mode()` raw mu + Wan latents_mean/std). Registered as `vae_encoder`; parametrized before-stage `vae_component_name`. Flags `use_light_vae_encoder`+`light_vae_path`. | ✅ code done | ✅ real ckpt loads (0 meta), encode `[1,3,1,64,64]→[1,16,1,8,8]`, `[1,3,5,64,64]→[1,16,2,8,8]` finite; FVD GPU-pending |
| **P4a** DiT FP8 (native) | Vendored full `omnidreams_singleview/` native tree + native python loader (`native/`). `omnidreams_fp8.py`: FP8 weight prep (reuses `cosmos_fp8_utils.prepare_cosmos_quantized_streaming_weights`), `build_fp8_dit` (strategy-gated, None on CPU), `OmniDreamsFP8DiT` dispatch reusing vendored `OptimizedDiTExecutor`. Wired into `_call_dit` behind `use_fp8_dit` (mutually exclusive with P0). Flags `use_fp8_dit`+`fp8_dit_attention_backend`+`fp8_dit_sparge_topk` + env `SGLANG_OMNIDREAMS_FP8_DIT`. | ✅ code done | ✅ FP8 quant (uint8 RCR + per-ch scale), DiT key-compat, CPU fallback (build→None / required→raises); native dispatch GPU-pending |
| **P4b** LightVAE FP8 (native) | ✅ code done | ✅ 8 fp8 tests green (3 P4b: fallback/roundtrip/buffers); native dispatch GPU-pending |

## Files

**Modified (5, tracked):**
- `configs/pipeline_configs/omnidreams.py` — flags: `enable_cuda_graph`, `cuda_graph_warmup_iters`, `use_light_tae`, `light_tae_path`, `use_light_vae_encoder`, `light_vae_path`, `use_fp8_dit`, `fp8_dit_attention_backend`, `fp8_dit_sparge_topk`, `use_light_vae_fp8`, `light_vae_fp8_state_path`.
- `envs.py` — `SGLANG_OMNIDREAMS_CUDA_GRAPH`, `SGLANG_OMNIDREAMS_FP8_DIT`, `SGLANG_OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH`.
- `runtime/models/dits/omnidreams.py` — P3 assert removal (only change).
- `runtime/pipelines/omnidreams_pipeline.py` — loaders `_load_light_tae`/`_resolve_light_tae_path`/`_load_light_vae`/`_resolve_light_vae_path`/`_resolve_light_vae_fp8_state_path`; load_modules branches (`vae_decoder`/`vae_encoder` modules + warnings + P4b FP8 wiring); create_pipeline_stages decode-stage swap + encode-vae selection.
- `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py` — P0 AR-loop wiring; P4a FP8 routing; `OmniDreamsLightTAEDecodingStage`; before-stage `vae_component_name` param.

**New (code):**
- `runtime/models/dits/omnidreams_cuda_graph.py` (P0)
- `runtime/models/dits/omnidreams_fp8.py` (P4a)
- `runtime/models/vaes/taehv.py` (P1)
- `runtime/models/vaes/omnidreams_light_vae.py` (P2 + P4b: fp8_state_path, mean/inv_std buffers, _get_native_handle, _encode_native, encode() dispatch)
- `native/` — vendored `omnidreams_singleview/` (src+patches+tools+python, ~76 files) + python loader (`__init__.py`, `acceleration.py`, `singleview_loader.py` [renamed from `omnidreams_singleview.py` to avoid pkg/dir name collision], `primitives.py`). 3rdparty (CUTLASS/Sage/Sparge) NOT vendored — synced at GPU build via `tools/sync_thirdparty.py`.

**New (tests/docs):**
- `test/unit/test_omnidreams_accel.py` (P0/P1/P2, 8 tests) — all green.
- `test/unit/test_omnidreams_fp8.py` (P4a, 5 tests + P4b, 3 tests) — all green.
- `test/spikes/omnidreams_fp8_lightvae_spike.py` (P4b VAE FP8 microbench, GPU-only).
- `test/spikes/export_lightvae_fp8_state.py` (P4b LightVAE FP8 calibration export tool, GPU-only).

## P4b complete
- `docs/superpowers/omnidreams_p4_fp8_design.md`, `omnidreams_optimization_progress.md` (this file).

## Remaining: Cosmos-Reason1-7B FP8 quant (TODO)

All P0–P4b code complete + 5 bugfixes. E2E verified 13f/29f with LightVAE-FP8 + LightTAE.

### Cosmos-Reason1-7B text encoder FP8 quant

**Why:** The text encoder (Cosmos-Reason1-7B, 14GB) is the single largest component in VRAM and also the slowest stage on CPU (~270s for text encode in PyTorch path). FP8 quantizing it would: (a) reduce VRAM from 14GB to ~7GB, allowing it to fit on GPU for 12GB cards; (b) speed up the CPU-forward path via reduced memory bandwidth.

**Approach:**
- Use Nvidia ModelOpt or torchao to quantize the Qwen2.5-VL 7B model to FP8
- Wire quantized weights into `OmniDreamsPipeline._load_text_encoder()`
- Keep `--text-encoder-cpu-offload` path for safety; add `--text-encoder-fp8` flag
- Validate encoding parity (cosine similarity > 0.99 vs bf16) before E2E

**Status:** Not started. Blocked on: no dependency — purely optimization, current E2E works without it.

All CPU tests green (8 fp8 + 8 accel + 34 components/scaffold = 50 total, plus 3 pre-existing failures from env). Ruff clean on all touched files.

GPU verification: sm_120 host (chen@100.87.72.4, RTX 5070 12GB, WSL2, CUDA 13.1). Native ext built (43MB .so). **2026-06-15: 6/6 unit PASS, 13f/29f E2E PASS with LightVAE-FP8 + LightTAE.** Prior blockers resolved (see below). Durable log: `/home/chen/omnidreams_test/e2e_run12.log`.

## E2E results (2026-06-15)

| Config | Total | BeforeDenoising | Denoising | Decoding | Peak Mem |
|--------|-------|-----------------|-----------|----------|----------|
| 13f PyTorch | 744.5s | 272.9s | 385.0s | 18.3s | 30GB |
| 13f FP8 | **268.4s** | **64.1s** | **135.3s** | 21.0s | 23.3GB |
| **29f FP8** | **559.6s** | **63.0s** | **337.0s** | **84.7s** | 30.1GB |

**Speedup: 2.8× (13f) vs PyTorch encode path.** FP8 before_denoising 4.3× faster.

## Bugs fixed today (2026-06-15)

1. **`--pipeline-config-path` JSON never loaded** — `from_kwargs` only checked `pipeline_config`, not `pipeline_config_path`. Fix: `base.py` falls through to `pipeline_config_path` kwarg.
2. **Wan VAE empty dir crashes** — pipeline loads VAE weights even when LightVAE+LightTAE both active. Fix: create lightweight `nn.Module` placeholder with only `latents_mean`/`latents_std`/`use_feature_cache`.
3. **Text encoder OOM on 12GB cards** — `use_declared_component("text_encoder")` moves 14GB Cosmos-Reason1-7B onto GPU. Fix: CPU-offload paths skip residency manager, run directly on CPU.
4. **FP8 permanently disabled after one failure** — `_native_disabled` flag set on first transient error (JIT rebuild in progress). Fix: remove permanent disable, retry every call.
5. **Prebuilt `.so` module name mismatch** — `_load_prebuilt_extension()` used hardcoded `"omnidreams_singleview_native_prebuilt"`, PyTorch exports `PyInit_<dirname>()`. Fix: use dirname as module name.
6. **DiT FP8 `_ExecConfig` missing attrs** — vendored `OptimizedDiTExecutor` reads `config.use_cuda_graph`/`cuda_graph_warmup_iters`. Fix: add both to `_ExecConfig`.
7. **DiT FP8 duplicate `_ensure_fp8_runtime` call** — SGLang wrapper called it with incomplete args; `_predict_flow_ext_impl` already handles it internally. Fix: remove duplicate call.

## DiT FP8 status (P4a) — partially blocked

**Unit test:** ✅ PASS (build + executor creation).  
**E2E:** ❌ Native CUDA kernel crashes at `cosmos_run_transformer_block_streaming: block 0: unknown error` inside `optimized_dit_forward`.

Root cause (preliminary): the vendored `OptimizedDiTExecutor` (from FlashDreams) expects the FlashDreams DiT structure — state_dict key names, network sub-module layout, and `prepare_cosmos_quantized_streaming_weights` conversion all differ from SGLang's `OmniDreamsDiT`. The FP8 weights get built with wrong shapes/layouts, and the native CUDA kernel aborts on the first block.

**Next steps for DiT FP8 fix:**
1. Dump state_dict keys from both FlashDreams DiT and SGLang `OmniDreamsDiT`, map the differences
2. Add key-name translation to `_SGLTransformerAdapter` or `_ensure_weights_snapshot`
3. Verify weight shapes match BLAS GEMM expectations (K/N dimensions for FP8 kernels)
4. Handoff doc already references this gap: `p4a-optimized-dit-flashdreams-import-bug.md`

**Workaround:** `use_fp8_dit=false` (default); LightVAE-FP8 (P4b) provides the bulk of E2E speedup.

## GPU verification checklist (sm_120 host)
**Build:** vendor sync + build the native ext: drive `native/singleview_loader.load_extension(NativeAccelerationConfig(mode="required"))` (syncs CUTLASS `f3fde58`+`sm120-tma-pool.patch`, SageAttention, SpargeAttn; needs `CUDA_ARCH_LIST=sm_120`). Confirm `optimized_dit_forward`, `sage3_is_built/runtime_supported`, `sparge_is_built`, `omnidreams_vae_*` present.

1. **P3** smoke: 1-chunk `--enable-torch-compile` + `SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs` → no assert, no graph-break spam, finite output.
2. **P0** numerics: graph-on vs -off, same seed, per-chunk compare (focus fill→steady transition chunk), bf16 tol; run ≥3× for stream-sync nondeterminism; measure VRAM. Then **P0+P3 composed**.
3. **P1** quality: FVD (expect ~24.8→~45.4), frame-count, first/mid/last visual; HD-map motion regression.
4. **P2** quality: encode FVD + HD-map motion regression (meandiff in moving range, not frozen).
5. **P4a** DiT FP8: build ext; per-chunk latent FP8-vs-bf16 tol (agree budget); end-to-end FVD + latency (largest expected speedup); attention-backend sweep (flash/sage3/sparge via `fp8_dit_attention_backend`); verify KV-cache mapping (`compute_self_attn_write_start` ↔ `BlockKVCache`); confirm reshape `[B,V,T,HW,D]→[B,L,D]`.
6. **P4b** VAE FP8: generate calibrated state (export tool); encode FP8-vs-bf16 tol + HD-map motion. Spike: `python test/spikes/omnidreams_fp8_lightvae_spike.py --ckpt .../lightvaew2_1.pth`.
7. Full §8 sweep with all flags; update report numbers.

## Pre-existing CPU failures (NOT from this work; confirmed on clean main)
- `test_omnidreams_components.py::test_tiny_dit_single_chunk_forward_and_unpatchify` — Triton-absent compile; passes with `TORCHDYNAMO_DISABLE=1`.
- `test_omnidreams_regression.py::test_encode_text_normalizes_batchencoding` — test bypasses stage `__init__`, missing `_text_embed_cache`.
- Pre-existing ruff `F841` (`L` unused) at `omnidreams.py:930` (rope block) — not my code.

## Closed PRs (do NOT reopen)
#28182 (SP) and #28178 (T1–T4) were opened to the official repo by mistake and closed by the user. SP is intentionally skipped. T1–T6 already on main.
