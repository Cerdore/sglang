# OmniDreams Optimization — Progress Snapshot

> **Note:** Config fields referenced in historical sections below (`use_fp8_dit`, `use_light_vae_encoder`, etc.) have been replaced by nested Config dataclasses. See [omnidreams_config_migration.md](omnidreams_config_migration.md).

**Date:** 2026-06-17 (updated). **Branch:** `feat/omnidreams-p0-p4b-optimizations`.
**Dev box:** CPU-only macOS. GPU validation on `rtx6k` (AutoDL container, NVIDIA RTX 6000D 85GB, CUDA 12.8/13.0, torch 2.11.0+cu130). **SSH:** `ssh rtx6k` → AutoDL west-E container.

Plan: `docs/superpowers/omnidreams_optimization_plan.md`. FP8 design: `docs/superpowers/omnidreams_p4_fp8_design.md`.
Reference source (local): `/Users/cerdore/gitRepo/flashdreams`. Checkpoints: `/Users/cerdore/gitRepo/models/{omni-dreams-models/single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt, lighttaew2_1.pth, lightvaew2_1.pth}`.

## CPU test invariant (IMPORTANT)
- venv python: `/Users/cerdore/.python/sglang/bin/python` (torch 2.9.0, editable sglang).
- **Prefix forward-path pytest with `TORCHDYNAMO_DISABLE=1`** — Triton is absent on macOS so the T1 `@torch.compile` LayerNormScaleShift kernels fail otherwise.
- Installed into the venv this session: `imageio imageio-ffmpeg diffusers==0.37.0 moviepy opencv-python-headless av scikit-image addict cache-dit trimesh xatlas ruff` (were missing).

## GPU test invariant (rtx6k, AutoDL)
- venv python: `/root/autodl-tmp/sglang-venv/bin/python3` (torch 2.11.0+cu130, editable sglang at `/root/sglang`).
- Activate: `source /root/autodl-tmp/sglang-venv/bin/activate` or alias `sa`.
- Models: `/root/autodl-fs/models/` (DiT, LightVAE, LightTAE, Wan VAE, text_encoder symlink to `/root/autodl-tmp/models/Cosmos-Reason1-7B`).
- Sample data: `/root/autodl-tmp/omni-dreams-samples/data/single_view/`.
- **Model path MUST contain `omni-dreams`** for the registry detector to match. Use `/root/autodl-fs/omni-dreams-models` (symlink to `/root/autodl-fs/models`).
- CUDA toolkit: system `/usr/local/cuda` = 12.8 (too old for sm_120). pip-installed CUDA 13 packages under `/root/autodl-tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cu13/`. Set `CUDA_HOME` to that path for native extension builds.
- `huggingface-cli` not installed; gated model downloads were pre-staged.
- AutoDL academic proxy: `source /etc/network_turbo` → `http://172.20.0.113:12798` (may be unstable; direct GitHub access works but is slow).

## Status by phase

| Phase | What | Status | Verified on CPU | Verified on GPU |
|---|---|---|---|---|
| **P3** torch.compile | Removed over-conservative `assert not torch.compiler.is_compiling()` in DiT `forward()` (single-chunk forward is compile-safe under fullgraph=False; AR loop lives in the stage). Compile already routes via `DenoisingStage._maybe_enable_torch_compile`. | ✅ code done | ✅ suite green | pending |
| **P0** CUDA Graph | Ported `CUDAGraphWrapper`+`set_or_copy` → `omnidreams_cuda_graph.py`. Wired into AR loop: capture the 3 per-chunk `self.transformer(...)` calls **only at KV steady-state**; fill phase + `pin` + `before/after_update` stay eager. Flag `enable_cuda_graph` + env `SGLANG_OMNIDREAMS_CUDA_GRAPH`. | ✅ code done | ✅ eager path green | pending |
| **P1** LightTAE decode | Ported TAEHV (impl+checkpoint remap) → `vaes/taehv.py` + `LightTAEDecoder` (own latent mean/std, single-pass). Custom decode stage `OmniDreamsLightTAEDecodingStage` (skips Wan scale_and_shift). Registered as `vae_decoder` module. Flags `use_light_tae`+`light_tae_path` + one-time quality warning. | ✅ code done | ✅ decode shape finite | ✅ E2E: 13f video output correct, 20.4s total |
| **P2** LightVAE encode | Ported pruned (0.75) streaming Wan encoder → `vaes/omnidreams_light_vae.py` + `LightVAEEncoder` (`.encode().mode()` raw mu + Wan latents_mean/std). Registered as `vae_encoder`; parametrized before-stage `vae_component_name`. Flags `use_light_vae_encoder`+`light_vae_path`. | ✅ code done | ✅ encode shapes finite | ✅ E2E: LightVAE encode works in P1+P2 combined 13f test |
| **P4a** DiT FP8 (native) | Vendored full `omnidreams_singleview/` native tree + native python loader (`native/`). `omnidreams_fp8.py`: FP8 weight prep (reuses `cosmos_fp8_utils.prepare_cosmos_quantized_streaming_weights`), `build_fp8_dit` (strategy-gated, None on CPU), `OmniDreamsFP8DiT` dispatch reusing vendored `OptimizedDiTExecutor`. Wired into `_call_dit` behind `use_fp8_dit` (mutually exclusive with P0). Flags `use_fp8_dit`+`fp8_dit_attention_backend`+`fp8_dit_sparge_topk` + env `SGLANG_OMNIDREAMS_FP8_DIT`. | ✅ code done | ✅ FP8 quant + DiT key-compat | pending (needs native ext build) |
| **P4b** LightVAE FP8 (native) | ✅ code done | ✅ 8 fp8 tests green | pending (needs native ext build) |
| **Config refactoring** | Migrated from flat bool fields to nested Config dataclasses. 4 components (text_encoder, image_encoder, encoder, decoder) each get independent Config with orthogonal impl selection + FP8 acceleration (auto/disabled/required). DiT FP8 changed to three-state `native_dit_acceleration`. New Text Encoder FP8 W8A8 support. Old fields deleted with `__post_init__` migration detection. **Two bugs found+fixed via new tests:** (1) `_make_vae_config` passed `latents_mean/std` as top-level `OmniDreamsVAEConfig` kwargs (they live on `arch_config`, proxied via `__getattr__`) → would TypeError-crash every wanvae `setup()` on GPU; (2) `_DEFAULT_LATENTS_STD` in components drifted from the validated Wan values → silent VAE-normalization corruption. | ✅ code done | ✅ 14 config tests green | ✅ config loading + rehydration works |
| **E2E LightVAE+LightTAE** | First successful E2E on rtx6k with nested Config JSON pipeline. 13 frames, 1280x704, H.264. | ✅ **DONE 2026-06-17** | — | ✅ 20.4s, 28.8GB peak |

## GPU unit test results (rtx6k, 2026-06-17)

| Suite | Result | Tests |
|-------|--------|-------|
| `test_omnidreams_accel.py` | 5 passed, 3 skipped | LightTAE/LightVAE ckpt skips (checkpoint path resolution in test env) |
| `test_omnidreams_fp8.py` | 19 passed, **3 failed**, 2 skipped | 3 failures: native extension `.so` not built |
| `test_omnidreams_components.py` | 30 passed | All green |
| `test_omnidreams_regression.py` | 14 passed | All green |
| `test_omnidreams_scaffold.py` | 3 passed | All green |
| `test_omnidreams_hdmap_validation.py` | 12 passed | All green |
| `test_omnidreams_optimizations.py` | 13 passed | All green |
| **Total** | **88 passed, 3 failed, 5 skipped** | |

### 3 FP8 failures — root cause

All 3 failures (`test_native_extension_unavailable_on_cpu`, `test_build_fp8_dit_returns_none_on_cpu`, `test_build_fp8_dit_required_raises_on_cpu`) are because the native extension `.so` is not built. This depends on:

1. 3rdparty repos (CUTLASS, SageAttention, SpargeAttn, cudnn-frontend) — synced via `native/omnidreams_singleview/tools/sync_thirdparty.py`
2. CUDA toolkit ≥12.9 for sm_120 — pip-installed CUDA 13 under `nvidia/cu13/` works when `CUDA_HOME` is set

Sync tested with:
- `ghproxy.net` mirror → 2-minute timeout
- AutoDL academic proxy `source /etc/network_turbo` → also timed out
- Direct GitHub → works but very slow (CUTLASS ~15MB after 10+ minutes)
- **Verdict:** needs manual pre-staging or faster mirror

## E2E results

### 2026-06-17: rtx6k (RTX 6000D 85GB, torch 2.11.0+cu130)

| Config | Total | BeforeDenoising | Denoising | Decoding | Peak Mem | Notes |
|--------|-------|-----------------|-----------|----------|----------|-------|
| 13f LightVAE+LightTAE (PyTorch) | **20.4s** | — | — | — | 28.8GB | Python API serve + curl, 1280x704, H.264 87KB |

Serve command:
```bash
sglang serve --model-path /root/autodl-fs/omni-dreams-models \
  --pipeline-class-name OmniDreamsPipeline \
  --pipeline-config-path .../omnidreams_lightvae_lighttae.json \
  --vae-cpu-offload --text-encoder-cpu-offload --warmup false --port 30000
```

### 2026-06-15: sm_120 WSL2 host (chen@100.87.72.4, RTX 5070 12GB)

| Config | Total | BeforeDenoising | Denoising | Decoding | Peak Mem |
|--------|-------|-----------------|-----------|----------|----------|
| 13f PyTorch | 744.5s | 272.9s | 385.0s | 18.3s | 30GB |
| 13f FP8 | **268.4s** | **64.1s** | **135.3s** | 21.0s | 23.3GB |
| **29f FP8** | **559.6s** | **63.0s** | **337.0s** | **84.7s** | 30.1GB |

**Speedup: 2.8× (13f) vs PyTorch encode path.** FP8 before_denoising 4.3× faster.

## Files

**Modified (13 files):**
- `configs/pipeline_configs/omnidreams.py` — override `load_from_json`/`update_pipeline_config` to protect component-config fields from base-class `ModelConfig` recursion (2026-06-17 fix)
- `configs/models/omnidreams_components.py` — **NEW**: 3 Config dataclasses (TextEncoder, VAEEncoder, VAEDecoder) + shared loading helpers
- `envs.py` — `SGLANG_OMNIDREAMS_CUDA_GRAPH`, `SGLANG_OMNIDREAMS_FP8_DIT`, `SGLANG_OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH`, `SGLANG_OMNIDREAMS_TEXT_ENCODER_FP8_PATH`
- `runtime/models/dits/omnidreams.py` — P3 assert removal, comment cleanup
- `runtime/models/dits/omnidreams_fp8.py` — `build_fp8_dit` mode param required; three-state dispatch
- `runtime/pipelines/omnidreams_pipeline.py` — `load_modules` uses `config.X_config.setup()`; deleted old VAE/text-encoder loaders; modules dict: `image_encoder`/`encoder`/`decoder` slots
- `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py` — BeforeDenoisingStage: `image_encoder` + `encoder` params; DenoisingStage: `decoder` param; DiT FP8 three-state; **2026-06-17 fix: `self._vae_component_name`/`self.vae` → `"image_encoder"`/`self.image_encoder` and `"encoder"`/`self.encoder`**
- `native/singleview_loader.py` — cleaned shadow imports

**New (code):**
- `runtime/models/dits/omnidreams_cuda_graph.py` (P0)
- `runtime/models/dits/omnidreams_fp8.py` (P4a)
- `runtime/models/vaes/taehv.py` (P1)
- `runtime/models/vaes/omnidreams_light_vae.py` (P2 + P4b: fp8_state_path, mean/inv_std buffers, _get_native_handle, _encode_native, encode() dispatch)
- `native/` — vendored `omnidreams_singleview/` (src+patches+tools+python, ~76 files) + python loader (`__init__.py`, `acceleration.py`, `singleview_loader.py` [renamed from `omnidreams_singleview.py` to avoid pkg/dir name collision], `primitives.py`). 3rdparty (CUTLASS/Sage/Sparge) NOT vendored — synced at GPU build via `tools/sync_thirdparty.py`.

**New (tests/docs):**
- `test/unit/test_omnidreams_accel.py` (P0/P1/P2, 8 tests) — all green.
- `test/unit/test_omnidreams_fp8.py` (P4a, 5 tests + P4b, 3 tests + Config, 14 tests incl. setup() routing + three-state FP8) — all green on CPU; 3 fail on GPU (no native ext).
- `test/unit/test_omnidreams_components.py` — added SP-reject + cross-view-reject architectural guards; fixed 6 stale `_encode_hdmap` tests broken by the VAE three-role split (`self.vae` → `self.encoder`).
- `test/spikes/omnidreams_fp8_lightvae_spike.py` (P4b VAE FP8 microbench, GPU-only). **2026-06-17:** head-level GEMM: BF16 0.016ms vs FP8 0.039ms (0.41x speedup), max_abs_err=0.69, mean_rel_err=21.2% — go/no-go FAIL on this thin layer.
- `test/spikes/export_lightvae_fp8_state.py` (P4b LightVAE FP8 calibration export tool, GPU-only).
- `tools/export_cosmos_reason_fp8.py` (Text Encoder W8A8 FP8 quantization script).
- `docs/superpowers/omnidreams_config_migration.md` (Config migration guide).

## Bugs fixed (2026-06-17 session)

1. **Pipeline config JSON loading blocked E2E** — `base.py` `update_pipeline_config` iterates all fields; when it hit `image_encoder_config` (default `EncoderConfig(ModelConfig)`) with a JSON dict like `{"impl": "lightvae"}`, `update_model_config` rejected `impl` as an unknown field. The existing `_rehydrate_component_configs` ran in `__post_init__` but by then the damage was done.
   **Fix:** Override `load_from_json` + `update_pipeline_config` in `OmniDreamsPipelineConfig` → `_update_pipeline_config_with_component_protection`: pop the 4 component-config fields before delegating to base, then apply them as raw dicts + call `_rehydrate_component_configs`. Added `json` and `Any` imports.

2. **`_vae_component_name` runtime crash** — `BeforeDenoisingStage` used `self._vae_component_name` and `self.vae` in two `use_declared_component` blocks, but those attributes were never set (the stage has `self.image_encoder` + `self.encoder`). `_encode_reference_image` uses `self.image_encoder`, `_encode_hdmap` uses `self.encoder`.
   **Fix:** `component_name="image_encoder", module=self.image_encoder` and `component_name="encoder", module=self.encoder`.

## Bugs fixed (2026-06-15)

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

**Workaround:** `native_dit_acceleration="disabled"` (default); LightVAE-FP8 (P4b) provides the bulk of E2E speedup.

## Remaining tasks

### Native extension build (P4a/P4b GPU verification)
- [ ] Sync 3rdparty repos: `source /etc/network_turbo` or find working mirror → `python native/omnidreams_singleview/tools/sync_thirdparty.py sync`
- [ ] Build native ext: `CUDA_HOME=<pip-cu13> TORCH_CUDA_ARCH_LIST=12.0 python -c "from sglang.multimodal_gen.native.singleview_loader import load_extension; load_extension()"`
- [ ] Re-run 3 failing FP8 tests
- [ ] E2E with `native_dit_acceleration="required"` (dit_fp8 + mixed_fp8 configs)

### Text Encoder FP8
- [ ] Run `tools/export_cosmos_reason_fp8.py` to produce W8A8 model
- [ ] Validate encoding parity (cosine similarity > 0.99 vs bf16)
- [ ] E2E with `omnidreams_te_fp8.json` config

### GPU verification checklist (sm_120 host)
1. **P3** smoke: 1-chunk `--enable-torch-compile` + `SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs` → no assert, no graph-break spam, finite output.
2. **P0** numerics: graph-on vs -off, same seed, per-chunk compare (focus fill→steady transition chunk), bf16 tol; run ≥3× for stream-sync nondeterminism; measure VRAM. Then **P0+P3 composed**.
3. **P1** quality: FVD (expect ~24.8→~45.4), frame-count, first/mid/last visual; HD-map motion regression.
4. **P2** quality: encode FVD + HD-map motion regression (meandiff in moving range, not frozen).
5. **P4a** DiT FP8: build ext; per-chunk latent FP8-vs-bf16 tol (agree budget); end-to-end FVD + latency (largest expected speedup); attention-backend sweep (flash/sage3/sparge via `fp8_dit_attention_backend`); verify KV-cache mapping (`compute_self_attn_write_start` ↔ `BlockKVCache`); confirm reshape `[B,V,T,HW,D]→[B,L,D]`.
6. **P4b** VAE FP8: generate calibrated state (export tool); encode FP8-vs-bf16 tol + HD-map motion. Spike: `python test/spikes/omnidreams_fp8_lightvae_spike.py --ckpt .../lightvaew2_1.pth`. **2026-06-17 spike result:** head-level GEMM only — not representative of full model speedup.
7. Full §8 sweep with all flags; update report numbers.

## GPU test-readiness TODO (CI gaps found 2026-06-16)

Status of the two server cases in `test/server/gpu_cases.py` and what's needed to make OmniDreams' E2E coverage match the standard models (Wan/FLUX/Qwen).

**Blockers — the existing `omnidreams_2b_i2v` case is not fully wired:**
- [ ] **perf baseline is all zeros** (`perf_baselines.json` → `omnidreams_2b_i2v`: `expected_e2e_ms: 0.0`, `stages_ms: {}`). Run on sm_120 with `SGLANG_GEN_BASELINE=1` to populate real numbers, else the `run_perf_check=True` gate is meaningless.
- [ ] **consistency GT** must exist in the external `sgl-project/ci-data` repo (`diffusion-ci/consistency_gt/...`). Unverifiable from the dev box. Run with `SGLANG_GEN_GT=1` to (re)generate and confirm upload; prior progress note claims a 2-GPU GT exists — verify the 1-GPU case id matches.
- [ ] **HD-map case `omnidreams_2b_i2v_hdmap` is a CI no-op**: self-skips unless `SGLANG_OMNIDREAMS_FIRST_FRAME_URL`/`_HDMAP_URL` are set, and even when it runs all checks are off. Decide: either host the gated assets + add a GT baseline, or accept HD-map E2E stays unit-only.

**Standard-coverage gaps (other models have these on by default):**
- [ ] `run_models_api_check=False` → flip to `True` (cheap, universal `/v1/models` contract check).
- [ ] `run_component_accuracy_check=False` → OmniDreams has zero full-model numeric-parity verification at the server layer. Hard (non-Diffusers flat `.pt` + custom AR loop + Cosmos-Reason1 TE → generic Diffusers reference loader can't materialize it). Needs a skip-registry entry or a bespoke reference; defer unless prioritized.

**New-feature E2E coverage (this refactor added the config surface, but no server case exercises it):**
- [x] **Acceleration-path server cases prepared (opt-in).** 4 JSON pipeline-configs under `test/test_files/` (`omnidreams_dit_fp8.json`, `omnidreams_lightvae_lighttae.json`, `omnidreams_te_fp8.json`, `omnidreams_mixed_fp8.json`) + 4 `DiffusionTestCase`s in `gpu_cases.py`, registered only when `SGLANG_OMNIDREAMS_ACCEL_E2E=1` (sm_120 + extra assets needed; CI stays green without it). Perf/consistency off (no baselines yet) — first goal is clean no-crash generation per path. Verified on CPU: all 4 JSON configs load + rehydrate into valid component configs.
  - **E2E verified 2026-06-17 (rtx6k):** `omnidreams_lightvae_lighttae.json` — 13f, 1280x704, 20.4s, 28.8GB peak. **This is the first full E2E with nested Config pipeline.**
  - **Heterogeneous-VAE case considered and dropped.** FlashDreams' only shipped heterogeneous wiring pairs a full-WanVAE image_encoder with a **PixelShuffle** HD-map encoder (`SV_..._PSHUFFLE_LIGHTTAE`), which is bound to the unreleased `1view-pshuffle-chunk4` checkpoint (192-ch DiT, len_t=4 — see [[omnidreams-pixelshuffle-variant]]); not landable with our `1view-vae-chunk2` (16-ch) checkpoint. A synthetic wanvae-image + lightvae-HDMap substitute isn't a reference config and validates nothing the `lightvae_lighttae` case + the CPU plumbing tests (rehydration, `_wanvae_cache` impl-keyed sharing, shipped-JSON loader) don't already cover. So no heterogeneous E2E case.
  - **Bug found+fixed 2026-06-17:** `update_pipeline_config` in base `PipelineConfig` rejected nested component config dicts because `image_encoder_config`'s default `EncoderConfig(ModelConfig)` couldn't handle fields like `impl`. Fixed by overriding `update_pipeline_config`/`load_from_json` in `OmniDreamsPipelineConfig` to intercept the 4 component-config fields before base-class recursion. **Second bug:** `_vae_component_name` + `self.vae` never set in `BeforeDenoisingStage` → runtime crash. Fixed by using correct `"image_encoder"`/`"encoder"` component names.
  - **To activate on rtx6k:** ensure LightVAE/LightTAE `.pth` in model dir; for FP8 set `SGLANG_OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH`; for TE-FP8 set `SGLANG_OMNIDREAMS_TEXT_ENCODER_FP8_PATH` (W8A8 dir from `tools/export_cosmos_reason_fp8.py`). Model path MUST contain `omni-dreams` substring for registry detector.
- [ ] Once each path generates cleanly, turn on `run_consistency_check` (generate GT with `SGLANG_GEN_GT=1`) + `run_perf_check` (baseline with `SGLANG_GEN_BASELINE=1`) per accel case.

## Pre-existing CPU failures (NOTE: re-verified 2026-06-16 — now all pass)
- `test_omnidreams_components.py::test_tiny_dit_single_chunk_forward_and_unpatchify` — passes with `TORCHDYNAMO_DISABLE=1` (always required on this Triton-less box).
- `test_omnidreams_regression.py::test_encode_text_normalizes_batchencoding` — now passes (stage `_text_embed_cache` set via `__new__` path).
- Full run 2026-06-16: **106 passed, 5 skipped** (skips = real lightvae/lighttae `.pth` not on the CPU box). Zero failures.

## Closed PRs (do NOT reopen)
#28182 (SP) and #28178 (T1–T4) were opened to the official repo by mistake and closed by the user. SP is intentionally skipped. T1–T6 already on main.
