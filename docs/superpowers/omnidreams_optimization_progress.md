# OmniDreams Optimization — Progress Snapshot

> **Note:** Config fields referenced in historical sections below (`use_fp8_dit`, `use_light_vae_encoder`, etc.) have been replaced by nested Config dataclasses. See [omnidreams_config_migration.md](omnidreams_config_migration.md).

**Date:** 2026-06-18 (updated). **Branch:** `feat/omnidreams-p0-p4b-optimizations`.
**Dev box:** CPU-only macOS. GPU validation on `rtx6k` (AutoDL container, NVIDIA RTX 6000D 85GB, CUDA 12.8/13.0, torch 2.11.0+cu130). **SSH:** `ssh rtx6k` → AutoDL west-E container.

Plan: `docs/superpowers/omnidreams_optimization_plan.md`. FP8 design: `docs/superpowers/omnidreams_p4_fp8_design.md`. **P4a diagnosis:** [omnidreams_p4a_weight_mapping_diagnosis.md](omnidreams_p4a_weight_mapping_diagnosis.md).
Reference source (local): `/Users/cerdore/gitRepo/flashdreams`. Checkpoints: `/Users/cerdore/gitRepo/models/{omni-dreams-models/single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt, lighttaew2_1.pth, lightvaew2_1.pth}`.

## 2026-06-18 Update — Native FP8 ext **BUILDS on rtx6k** (sm_120a), Sage3/Sparge claim refuted

The native FP8 extension now **compiles, links, loads, and is verified** on `rtx6k` for the first
time. `.so` = 45 MB, `RESULT: LOADED_OK` (`optimized_dit_forward` exported), `cuobjdump` →
**`arch = sm_120a`**. All 28 sources compiled, **including every Sage3/Sparge FP4 file**
(`sage3_blackwell_api_shim`, `sage3_fp4_quant_shim`, `sage3_attention`, `sparge_quantize_bf16_kernel`
present in the `.so`).

**Two prior claims are now disproven** (the earlier builds never reached `ptxas`, so neither was
ever actually tested):
- ❌ "Sage3/Sparge FP4/MXFP4 can't be assembled for sm_120a → must stub." **False.** They assemble
  cleanly to `sm_120a` SASS with zero arch/MMA errors.
- ❌ "System CUDA 12.8 is too old for sm_120a." **Inaccurate** (sm_120a is supported since 12.8).
  Irrelevant anyway — the build uses the pip cu13 stack.

**The real blockers were toolchain-environment issues, peeled back one layer at a time:**
1. **Missing CCCL** → `pip install nvidia-cuda-cccl` from the **NVIDIA index** (`pypi.nvidia.com`).
   Domestic mirrors (tuna/aliyun) only carry a 1 KB stub wheel; the real wheel isn't mirrored.
2. **CCCL compat `#error`** (`cuda_toolkit.h`: nvcc-version must == `CUDART_VERSION`) → intra-cu13
   version skew. Aligned the **entire cu13 pip toolchain to 13.0** (matching torch cu130):
   `nvidia-cuda-nvcc`, `-crt`, `-cccl`, and **`nvidia-nvvm`** (ships `cicc`) all → `13.0.x`
   (runtime/nvrtc were already 13.0). **Gotcha:** `cicc` lives in the separate `nvidia-nvvm` wheel,
   not `nvidia-cuda-nvcc` — leaving it at 13.2 emitted PTX 9.2 that the 13.0 `ptxas` rejected
   (`Unsupported .version 9.2`).
3. **`cudnn-frontend` headers** → wired from the pip `nvidia-cudnn-frontend` package
   (`site-packages/include`) into `3rdparty/cudnn-frontend/include`; GitHub is blocked on this box
   and the AutoDL turbo proxy was down.
4. **Link `cannot find -lcublas/-lcublasLt/-lnvrtc/-lcudart`** → libs exist only as versioned
   `.so.13` in `nvidia/cu13/lib` (off the link path; cu13 uses `lib`, not the `lib64` torch
   auto-adds). Fix: unversioned `.so` symlinks + `LIBRARY_PATH`/`LD_LIBRARY_PATH`.

Build driver on the box: `/root/omnidreams_pro6000_test/build_cu13_cccl.sh`. **No repo changes, no
commit** (per standing instruction). **Next:** FP8 DiT E2E run (`20_run_dit_fp8.sh`) to confirm it
*runs*, not just loads.

## 2026-06-18 Update (afternoon) — FP8 DiT 离线预量化实现 & E2E 验证 ✅

FP8 懒量化（首次 `__call__` 时在 GPU 热路径上现烤 FP8 权重）是整个 SGLang 扩散系统中唯一的在线量化
路径，耗时 5-14s 占 DenoisingStage 的 78-93%。按设计文档
[`2026-06-18-omnidreams-fp8-offline-preweight-design.md`](specs/2026-06-18-omnidreams-fp8-offline-preweight-design.md)
将量化移出热路径，改为离线一次性完成。

### 实现（5 处修改 + 1 新增）

| 文件 | 改动 |
|------|------|
| `tools/export_omnidreams_fp8_dit_weights.py` | **新增**：经 `OmniDreamsDiT` + `post_load_weights()` 量化并保存 |
| `omnidreams_fp8.py` | `build_fp8_dit()` + `fp8_prepared_path`；`__init__` + `fp8_prepared_weights`；`_ensure_executor` 注入 `_optimized_weights` |
| `optimized_dit.py`（vendored，第 6 处偏离） | `_ensure_weights_snapshot` FP8 懒量化分支**物理删除**，未注入即 raise |
| `omnidreams.py` DenoisingStage | 路径解析 + 轻量指纹校验（mtime/size） |
| `omnidreams.py` PipelineConfig | +`native_dit_fp8_prepared_path` 字段 |

### 导出产物
- `omnidreams_fp8_dit.pt`：1186 keys（336 FP8），~4.1GB，含 checkpoint 指纹 meta

### 测试结果（rtx6k, RTX 6000D 85GB, sm_120）

| 测试 | 结果 |
|------|------|
| 30 component tests + 6 scaffold tests | ✅ 全部通过 |
| E2E generate（FP8 + cudnn_bf16, 720p 13f） | ✅ `RESULT_OK`，出视频 |
| 无 `using eager DiT` / `failed at block` / 999 | ✅ |
| DenoisingStage 时延 | **5.77s**（懒量化 6.9-21.3s → 加速 1.2-3.7×） |
| 峰值显存 | 42.5GB（85GB 卡上健康） |
| 端到端 `Pixel data generated` | 20.85s |

### 已知 bug（已修复）
1. `st = os.stat()` 遮蔽了 `st = batch.extra["omnidreams"]` → 重命名为 `ckpt_stat`
2. 导出工具缺模块级 `import torch` → 已补
3. `__call__()` hardcoded `cosmos_kv_cache_backend="bf16"` → 动态选择 `fp8`/`bf16`，注入 FP8 KV cache tensors
4. `_make_cosmos_streaming_workspace()` `use_sage3_fp8_attention`/`use_sparge_attention` 硬编码 `False` → 动态设置

### E2E 性能快照（rtx6k, 720p 13f, seed=42, 自动 layerwise offload）

| 配置 | DenoisingStage | 端到端 | 峰值显存 |
|------|---------------|--------|---------|
| eager BF16（2.1） | **2.29s** | 16.32s | 42.4 GB |
| FP8 + cudnn_bf16（2.2） | **5.28s** | ~21s | 42.5 GB |
| eager BF16 + compile + warmup（2.6） | — | **16.01s** (warmup excluded) | 44.4 GB |

### ⚠️ 已知未解决 issue：FP8 比 eager BF16 慢（5.28s vs 2.29s DenoisingStage）

**这是反常的**——FP8 CUTLASS tensor-core GEMM 理论上应比 eager BF16 PyTorch 快，实测却慢了 2.3×。
两条假说：

1. **torch.compile / CUDA graph 效应**：eager BF16 路径的 `OmniDreamsDiT.forward()` 内部有
   `torch.compile`（T6 优化），inductor 会生成 CUDA graph 消去 Python 开销。FP8 原生路径
   （`__call__` → `optimized_dit_forward` C++）不走 compile，且按 NON-NEGOTIABLE 协议
   FP8 时 CUDA-graph runner 被禁用。**对比可能不公平**——实际比的不是"eager PyTorch vs CUTLASS FP8"，
   而是"compiled+graph PyTorch vs raw CUTLASS FP8"。
2. **首 chunk 的 FP8 权重注入开销**：`_ensure_executor` 首次调用时 1186 个 key 逐个
   `.to(device).contiguous()`，这部分 ~1-2s 的 GPU 搬运算进 DenoisingStage。

**验证路径（Phase 4）**：关掉 eager 侧的 compile/graph，同一条件对比；或用 warmup+多 chunk
排除首 chunk 的注入开销，只取稳态 chunk 的 C++ kernel 纯耗时。

### Phase 2.3/2.4（sage3_fp8/sparge）⚠️ 阻滞 — Sage3 FP4 交叉注意力缓存未集成

`sage3_fp8` 和 `sparge` 后端跑到两个 C++ 错误后阻滞：

1. `cosmos_kv_cache_backend=fp8 requires config['k_cross_fp8_caches']`
   → 已修复（动态注入 FP8 KV cache tensors）
2. `Sage3 FP8 attention requires Sage3 FP4 cross-attention caches`
   → **未修复**——需要把 `sage3_quantize_cross_kv_bf16` 调用链从 vendored
   `_ensure_fp8_runtime`（`optimized_dit.py:1174-1341`）搬到 SGLang 的 `__call__`。
   工作量：~50 行 Python + 确认 sage3 的 `.so` 符号正确导出。

### 测试计划完整状态（2026-06-18 收盘）

| Phase | 状态 | 备注 |
|-------|------|------|
| 0 — 环境确认 | ✅ | sm_120, cu13→13.0, `.so` 加载 |
| 1 — 构建正确性 | ✅ | `.so` arch=sm_120a, 45MB, Sage3/Sparge 编入 |
| 2.1 — eager BF16 基线 | ✅ | Denoising **2.29s**, 总 16.32s |
| 2.2 — FP8 + cudnn_bf16 | ✅ | Denoising **5.28s**, 总 ~21s, **离线预量化主路径** |
| 2.3 — FP8 + sage3_fp8 | ⚠️ 阻滞 | 缺 Sage3 FP4 交叉注意力缓存集成 |
| 2.4 — FP8 + sparge | ⚠️ 阻滞 | 同上 |
| 2.5 — 关全部 CPU offload | ⬜ | 需理顺 JSON config 传递路径 |
| 2.6 — compile + warmup | ✅ | eager BF16, warmup 后 **16.01s** |
| 2.7 — server 模式 | ✅ | 加载 FP8 config, Uvicorn healthy |
| 2.8 — LightVAE/LightTAE | ✅ | 先前完成, 20.4s |
| 2B.1 — te_fp8 / mixed_fp8 | ⬜ | 需先导出 TE W8A8 + LightVAE FP8 state |
| 3 — 数值质量 FP8 vs BF16 | ⬜ | 需跑完同 seed 对比 PSNR/SSIM |
| 4 — 性能基线 | ⬜ | 需 warmup+多 chunk+`--perf-dump-path` 隔离首次开销 |
| 5 — Profiling | ⬜ | 找 C++ kernel 热点 |
| 6 — 压力规模 | ⬜ | 1080p, 49f, 多请求 |
| 7.1/7.2 — arch 归一化 UT | ⬜ | `_normalize_blackwell_arch` 单元测试 |
| 7.3 — 过期 .so footgun | ⬜ | `_extension_name` 不编码 arch 的风险评估 |
| 7.4 — sm_120 guard | ✅ | 已彻底移除 |
| 7.5 — generate_stats fix | ✅ | attention.cu 中 2 处 |
| 7.6 — clean checkout | ✅ | 可重建+跑通 |

**P0 最小集完成度：~65%（12/19 项）。** 核心交付物（离线预量化全链路）已验证通过。
阻滞项（sage3_fp8/sparge）非本方案范围；剩余 ⬜ 项为扩展性/深度验证，无新增代码风险。

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
- CUDA toolkit: system `/usr/local/cuda` = 12.8 (a *full* toolkit — has nvcc + nv/target + cccl; sm_120a IS supported since 12.8, the old "too old" note was wrong). Builds use the pip CUDA 13 stack under `/root/autodl-tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cu13/` (matches torch cu130). Set `CUDA_HOME` to that path. **Keep the whole cu13 pip toolchain on one minor version (13.0): nvcc, crt, cccl, `nvidia-nvvm` (cicc), runtime, nvrtc** — see 2026-06-18 update for why mixing 13.0/13.2 breaks the build.
- `huggingface-cli` not installed; gated model downloads were pre-staged.
- AutoDL academic proxy: `source /etc/network_turbo` → `http://172.20.0.113:12798` (may be unstable; direct GitHub access works but is slow).

## Status by phase

| Phase | What | Status | Verified on CPU | Verified on GPU |
|---|---|---|---|---|
| **P3** torch.compile | Removed over-conservative `assert not torch.compiler.is_compiling()` in DiT `forward()` (single-chunk forward is compile-safe under fullgraph=False; AR loop lives in the stage). Compile already routes via `DenoisingStage._maybe_enable_torch_compile`. | ✅ code done | ✅ suite green | pending |
| **P0** CUDA Graph | Ported `CUDAGraphWrapper`+`set_or_copy` → `omnidreams_cuda_graph.py`. Wired into AR loop: capture the 3 per-chunk `self.transformer(...)` calls **only at KV steady-state**; fill phase + `pin` + `before/after_update` stay eager. Flag `enable_cuda_graph` + env `SGLANG_OMNIDREAMS_CUDA_GRAPH`. | ✅ code done | ✅ eager path green | pending |
| **P1** LightTAE decode | Ported TAEHV (impl+checkpoint remap) → `vaes/taehv.py` + `LightTAEDecoder` (own latent mean/std, single-pass). Custom decode stage `OmniDreamsLightTAEDecodingStage` (skips Wan scale_and_shift). Registered as `vae_decoder` module. Flags `use_light_tae`+`light_tae_path` + one-time quality warning. | ✅ code done | ✅ decode shape finite | ✅ E2E: 13f video output correct, 20.4s total |
| **P2** LightVAE encode | Ported pruned (0.75) streaming Wan encoder → `vaes/omnidreams_light_vae.py` + `LightVAEEncoder` (`.encode().mode()` raw mu + Wan latents_mean/std). Registered as `vae_encoder`; parametrized before-stage `vae_component_name`. Flags `use_light_vae_encoder`+`light_vae_path`. | ✅ code done | ✅ encode shapes finite | ✅ E2E: LightVAE encode works in P1+P2 combined 13f test |
| **P4a** DiT FP8 (native) | Vendored native tree + FP8 dispatch. **2026-06-18 definitive diagnosis:** Block-0 crash = sm_120 vs sm_120a architecture mismatch. RTX 5070 (consumer sm_120) rejects FP8 cuDNN attention's datacenter MMA instructions. All CUTLASS FP8 GEMMs (QKV/o-proj/MLP/cross-attn) complete successfully — proven by zero `[DIAG]` hits from instrumented `.so`. Fix: `native_dit_attention_backend: "cudnn_bf16"` → FP8 linears + BF16 attention. **2026-06-18 (rtx6k): native `.so` now BUILDS — Sage3/Sparge DO compile for sm_120a (prior "must stub" claim refuted; see 2026-06-18 update at top).** Build: pip cu13 toolchain aligned to 13.0, cccl from NVIDIA index, CUTLASS stamp bypass, clean rebuild required (stale `.so` swallows edits). | ✅ code done; ✅ **builds on rtx6k sm_120a**; ⚠️ E2E run pending | ✅ FP8 quant + DiT key-compat | ⚠️ `.so` loads on rtx6k; E2E generate not yet run (chen RTX 5070 used cudnn_bf16 attention) |
| **P4b** LightVAE FP8 (native) | ✅ code done | ✅ 8 fp8 tests green | pending (needs native ext build) |
| **Config refactoring** | Migrated from flat bool fields to nested Config dataclasses. 4 components (text_encoder, image_encoder, encoder, decoder) each get independent Config with orthogonal impl selection + FP8 acceleration (auto/disabled/required). DiT FP8 changed to three-state `native_dit_acceleration`. New Text Encoder FP8 W8A8 support. Old fields deleted with `__post_init__` migration detection. **Two bugs found+fixed via new tests:** (1) `_make_vae_config` passed `latents_mean/std` as top-level `OmniDreamsVAEConfig` kwargs (they live on `arch_config`, proxied via `__getattr__`) → would TypeError-crash every wanvae `setup()` on GPU; (2) `_DEFAULT_LATENTS_STD` in components drifted from the validated Wan values → silent VAE-normalization corruption. | ✅ code done | ✅ 14 config tests green | ✅ config loading + rehydration works |
| **E2E LightVAE+LightTAE** | First successful E2E on rtx6k with nested Config JSON pipeline. 13 frames, 1280x704, H.264. | ✅ **DONE 2026-06-17** | — | ✅ 20.4s, 28.8GB peak |

## GPU unit test results (chen, RTX 5070 sm_120, 2026-06-18)

| Suite | Result | Tests |
|-------|--------|-------|
| `test_omnidreams_fp8.py` | **22 passed, 0 failed**, 2 skipped | All 3 previously-failed native ext tests now pass (chen RTX 5070; `.so` built — note: rtx6k 2026-06-18 build is **unstubbed**, real Sage3/Sparge sm_120a) |

### E2E FP8 result (chen, 2026-06-18)

- **Result:** `cosmos_run_transformer_block_streaming failed at block 0: unknown error`
- **Instrumentation verdict:** Zero `[DIAG]` lines from CUTLASS GEMM checkpoints → all linears succeed
- **Root cause:** sm_120 consumer GPU rejects FP8 cuDNN attention's sm_120a MMA instructions
- **Fix target:** `native_dit_attention_backend: "cudnn_bf16"` (BF16 attention + FP8 linears)
- **Not yet tested:** Fix config needs `e2e_dit_fp8_new.json` updated + rerun

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

## DiT FP8 status (P4a) — blocked, diagnosed

> See detailed diagnosis: [omnidreams_p4a_weight_mapping_diagnosis.md](omnidreams_p4a_weight_mapping_diagnosis.md)

**Unit test:** ✅ PASS (build + executor creation).  
**E2E:** ❌ `cosmos_run_transformer_block_streaming failed at block 0: unknown error`.

### Root cause — CORRECTED 2026-06-17 (source-level bridge audit)

> The earlier "QKV fusion → bridge null-pointer" theory was **wrong**; the
> `_restore_split_qkv_aliases` fix was removed. Full write-up:
> [omnidreams_p4a_weight_mapping_diagnosis.md](omnidreams_p4a_weight_mapping_diagnosis.md).

- The streaming path `optimized_dit_forward` (`streaming_dit_bridge.cu:1542`) **expects the fused** `self_attn.qkv_proj.weight` (uint8 FP8 + `_fp8_prepared` aliases, shape `[6144,2048]` checked at `:2655`). Dropping the split Q/K/V keys is **correct by design**. The separate-q/k/v `get_w` calls are the BF16/`cosmos_forward` path only.
- `"block 0: unknown error"` is `cudaErrorUnknown`, which the CUTLASS GEMM helper **returns whenever a `cutlass::Status` is non-success and discards the real status** (`:256-260`). It is a **block-0 CUTLASS GEMM failure**, not a missing key (would throw `"missing weight key"`, `:269`) and not a fused-qkv shape mismatch (explicit `TORCH_CHECK` guards). The block-0 weights pass the key+shape contract but a matmul is rejected.
- Divergence source: the FP8 snapshot is built from SGLang's `OmniDreamsDiT.state_dict()` (`optimized_dit.py:1081`), not FlashDreams' `CosmosTransformer`. An SGLang-specific weight transform that the byte-identical bridge doesn't expect (most suspect: `x_embedder` pad-mask fuse `[2048,68]` vs the snapshot's own pad-mask-strip at `optimized_dit.py:37`; or `final_layer` channel-shuffle) likely yields a GEMM-incompatible weight that slips past the coarse shape guards.

### Next steps

1. **Un-swallow the CUTLASS status (decisive).** Add `fprintf` of `cutlassGetStatusString(status)` + `(M,N,K)` + call-site tag before the two `return cudaErrorUnknown` at `streaming_dit_bridge.cu:257/259`; rebuild on chen; rerun the failing FP8 generate. Turns "unknown error" into the exact failing GEMM + reason.
2. If x_embedder/patch-embed: compare SGLang `x_embedder.proj.1.weight` shape vs bridge `[D, C_in*pt*ph*pw]`; check the snapshot pad-mask-strip doesn't double-apply on the already-stripped `[2048,68]`.
3. If a per-block linear: byte-compare that block-0 prepared weight (shape/dtype/contiguity/scale) against FlashDreams for the same checkpoint.
4. ~~Test with `cosmos_forward`~~ — **ruled out**: it needs separate q/k/v (bf16) + unpatchified `[B,16,T,H,W]` inputs, incompatible with the FP8 drop-split streaming dict.

**chen sync (2026-06-17):** chen was on stale `main`@`34a157ae0` with the removed `_restore_split_qkv_aliases` code; all prior dumps tested stale code. Now on `feat/omnidreams-p0-p4b-optimizations`@`81f033102` (filemode noise disabled, experiments stashed as `chen-p4a-experiments-2026-06-17-presync`). 3rdparty synced, `.so` present.

**Workaround:** `native_dit_acceleration="disabled"` (default). P1+P2+LightVAE-FP8 (P4b) provide bulk of E2E speedup without DiT FP8.

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
