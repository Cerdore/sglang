# OmniDreams Optimization Implementation Plan (P0 + P3 + P1 + P2 + P4-spike)

**Status:** Plan + review only — NO code changes yet. Awaiting approval.
**Scope:** P0 (CUDA Graph) + P3 (torch.compile) + P1 (LightTAE decoder) + P2 (LightVAE encoder) full acceleration, accepting the FVD 24.8→45.4 quality tradeoff; plus P4 (Native FP8 LightVAE) **design + feasibility spike** only.
**Source of truth:** `docs/superpowers/omnidreams_optimization_report.md` (research report) + `docs/superpowers/flashdreams.md` (FlashDreams reference-runtime research) + current `main` code.
**Target GPU:** Blackwell. See §0 for the GPU/VRAM answer.

> **Evidence labelling convention** (per the active research-protocol directive): every load-bearing claim about current SGLang code is tagged `[verified L<file>:<line>]`; claims about FlashDreams behavior are tagged `[flashdreams-doc]` (from the research doc, not re-read from source this session); claims about compiler/CUDA-graph/Dynamo runtime behavior that have NOT been empirically tested are tagged `[UNTESTED-SPECULATION]` and MUST be validated on GPU before being relied upon.

---

## 0. GPU & VRAM answer (Blackwell)

**Question asked:** beyond GB200/GB300, are RTX 5090 / RTX PRO 6000 supported, and how much VRAM is needed?

- **RTX 5090 (GB202) and RTX PRO 6000 Blackwell are `sm_120`** — consumer/workstation Blackwell. The FlashDreams FP8 kernels (`sm120-tma-pool.patch`, CUTLASS `sm_120` path, TIN16 format) target **exactly this ISA** `[flashdreams-doc]`. So for **P4 (FP8)** these cards are the *ideal* match — better-supported than GB200/GB300, which are `sm_100` (datacenter Blackwell) and would need a *different* CUTLASS path `[flashdreams-doc]`.
- **P0/P3/P1/P2 are ISA-agnostic** — CUDA Graph, torch.compile, and the LightTAE/LightVAE ports are pure PyTorch/bf16 and run on any CUDA GPU (Ada, Hopper, Blackwell). No `sm_120` requirement for P0–P2.
- **VRAM:** measured peak **38 GB** for 704×1280 / 29-frame single-view on a 46 GB card `[verified: memory omnidreams-run-command]`. Practical minimums:
  - **≥ 32 GB comfortable** (RTX 5090 = 32 GB → fits 704×1280 29f, but with thin headroom; may need `--vae-cpu-offload` for longer rollouts).
  - **RTX PRO 6000 = 96 GB → ample** (no offload, room for MV/CP later).
  - CUDA Graph (P0) *adds* a fixed VRAM cost: static input/output buffers + the captured graph pool. Estimate +1–3 GB `[UNTESTED-SPECULATION]` — measure in §8.
- **Recommendation:** develop/verify P0–P2 on either card; do the P4 FP8 spike on an `sm_120` card (5090 or PRO 6000). PRO 6000 preferred for the spike (headroom for calibration dumps + dual-model A/B).

---

## 1. Strategy summary

OmniDreams inference cost splits into two buckets:

1. **DiT denoise loop** — `num_chunks × 2 flow steps × 1 context re-forward` calls through the 28-block 2B DiT. This is the dominant latency. → attack with **P0 CUDA Graph** (eliminate per-call CPU launch overhead across the repeated identical-shape calls) and **P3 torch.compile** (fuse the hot blocks).
2. **VAE encode/decode** — full Wan 2.1 VAE today (FVD 24.8, slow). → attack with **P1 LightTAE decoder** (decode ↓3-5×) and **P2 LightVAE encoder** (encode ↓2-3×), accepting FVD→45.4.

The two buckets are **independent** and gated by **separate flags**, so we can land + verify them incrementally and roll back either without touching the other. Ordering is chosen so each step is independently verifiable on GPU before the next is stacked:

```
P3 (torch.compile)  ──►  P0 (CUDA Graph)  ──►  P1 (LightTAE)  ──►  P2 (LightVAE)  ──►  P4 (FP8 spike)
   relax 1 assert        capture predict_flow    new decoder       new encoder        design+microbench
   lowest risk           biggest single win      VAE swap          VAE swap           no production wire-up
```

**Why P3 before P0:** P3 only requires relaxing one assert and flipping an existing flag `[verified L878 omnidreams.py]`; it's the cheapest to land and gives a clean baseline. P0 is the bigger win but riskier (static buffers, KV-cache pointer stability), so it goes second on top of a known-good compiled baseline. **Critical interaction:** torch.compile mode MUST be `max-autotune-no-cudagraphs` so inductor does *not* install its own CUDA graphs that would collide with our external P0 wrapper `[verified L296 denoising.py: SGLANG_TORCH_COMPILE_MODE default]` `[flashdreams-doc: drain() fires autotune before external capture]`.

**Quality gate:** P1/P2 change pixel output (FVD 24.8→45.4 expected). Every VAE-swap step is gated behind a flag defaulting to **off**, so the default install keeps full-VAE quality. P0/P3 are **numerically lossless** (same math, faster launch) and may default on after verification.

**Loud quality opt-in (critic finding):** when `use_light_tae` or `use_light_vae_encoder` is enabled, the loader MUST emit a one-time `logger.warning("OmniDreams: LightTAE/LightVAE enabled — trades quality (FVD ~24.8→~45.4) for speed. Disable for full-quality output.")`. This prevents a user from silently shipping the degraded path. Flags default off + visible warning when on = no accidental quality regression.

---

## 2. Current-code grounding (the facts the plan stands on)

| Fact | Evidence |
|---|---|
| DiT `forward()` hard-blocks torch.compile via `assert not torch.compiler.is_compiling()` | `[verified L878-882 omnidreams.py]` |
| DiT `forward()` requires `post_load_weights()` first (fuses padding-mask + shuffle) | `[verified L871-874 omnidreams.py]` |
| DiT `forward()` raises on SP `_sp_size>1` | `[verified L885-889 omnidreams.py]` |
| Hot blocks already opt into compile via `_compile_conditions = [lambda n,m: isinstance(m, OmniDreamsBlock)]` | `[verified L573 omnidreams.py — from prior read]` |
| `_maybe_enable_torch_compile` compiles the WHOLE transformer, gated by `server_args.enable_torch_compile`, mode from `SGLANG_TORCH_COMPILE_MODE` (default `max-autotune-no-cudagraphs`) | `[verified L284-309 denoising.py]` |
| AR loop drives KV-cache lifecycle OUTSIDE the DiT forward: `c.before_update(chunk_idx)` then `predict_flow` then clean re-forward then `c.after_update(chunk_idx)` | `[verified L815-847 model_specific_stages/omnidreams.py]` |
| `predict_flow(noisy, t)` is a closure capturing `rope_freqs, cond_mask, hdmap_chunk, caches, cross_attn_kv, view_indices` — the natural CUDA-graph capture unit | `[verified L800-813 model_specific_stages/omnidreams.py]` |
| Two distinct DiT invocations per chunk: the **2-step denoise** (`scheduler.sample`→`predict_flow`) and the **context re-forward** at `ctx_noise_t` | `[verified L821, L834-844 model_specific_stages/omnidreams.py]` |
| `BlockKVCache._k/_v` are static buffers allocated once in `__post_init__` (pointer-stable → CUDA-graph-friendly) | `[verified L126-127 omnidreams_kvcache.py — from prior read]` |
| Wan VAE swap point: `_load_wan_vae` + `_resolve_vae_path` (`_VAE_RELDIRS=("vae","wan_vae","Wan2.1_VAE")`), `create_pipeline_stages` uses `add_standard_decoding_stage` | `[verified L105-206, L349-375 omnidreams_pipeline.py — from prior read]` |
| VAE encode entry: `_vae_encode_normalized` (stage L325-346); HDMap encoded as one causal clip | `[verified model_specific_stages/omnidreams.py — from prior read]` |
| Pipeline config has `dit_precision=bf16, vae_precision=fp32, flow_shift=5.0, denoising_timesteps=(1000,450)` | `[verified configs/pipeline_configs/omnidreams.py — from prior read]` |
| Env vars MUST be registered in `environ.py` with `SGLANG_` prefix | `[verified .claude/rules + env-var-conventions skill]` |

**Concrete dataflow walk-through (per research-protocol directive), num_frames=13, len_t=2:**
`_compute_num_chunks(13, len_t=2) = 2` `[verified test param (13→2)]`. So the loop at L786 runs `chunk_idx ∈ {0,1}`. For each chunk: `scheduler.sample` calls `predict_flow` **2× (timesteps 1000,450)** `[verified denoising_timesteps]`, then **1× context re-forward** at `ctx_noise_t`. Total DiT calls = `2 chunks × (2+1) = 6` DiT forwards for a 13-frame clip. CUDA Graph (P0) must therefore capture/replay a unit that is invoked 6× with **identical tensor shapes** within a request (the chunk token count `chunk_tokens` is constant across chunks for a fixed h/w). This is what makes P0 high-value: 6 identical-shape launches → 6 graph replays.

---

## 3. P3 — torch.compile (land first, lowest risk)

**Goal:** allow the existing `enable_torch_compile` path to compile the OmniDreams DiT blocks without the `forward()` assert aborting the run. Expected ↓10-15% `[flashdreams-doc / report]`.

**Key insight `[verified L875-882]`:** the assert exists because the *outer* forward has a dynamic chunk loop + KV ops that break `fullgraph`. But the *blocks* already opt into compile via `_compile_conditions` `[verified L573]`. The whole-module `module.compile(fullgraph=False, ...)` `[verified L302 denoising.py]` compiles submodules independently — so the block-level compile is what we want, and the top-level forward should run eager.

**The problem `[UNTESTED-SPECULATION]`:** with `module.compile()` on the parent, when a compiled child block runs, `torch.compiler.is_compiling()` returns True *inside the parent forward during tracing of children*? — This needs empirical confirmation. The assert at L878 fires if the parent `forward()` body is ever entered under compilation. With `_compile_conditions` targeting only blocks and `fullgraph=False`, the parent forward should run eager and only the block `__call__`s enter compiled regions — meaning `is_compiling()` is False in the parent body. **MUST verify with a 1-chunk GPU smoke test before trusting.**

### Files & changes

1. **`runtime/models/dits/omnidreams.py`** — relax the assert (L878-882).
   - **Replace** the unconditional `assert not torch.compiler.is_compiling()` with a guard that only blocks the *unsupported* path. Two candidate designs (pick after the smoke test):
     - **(A) Remove the assert entirely** if the smoke test confirms the parent forward never runs under compilation (block-only compile). Add a one-line comment: `# Parent forward runs eager; only OmniDreamsBlock children are compiled (_compile_conditions).`
     - **(B) Downgrade to a targeted guard** if compilation *can* reach the parent: gate the dynamic-loop-unsafe portions, or assert only when `kv_caches is not None and torch.compiler.is_compiling()` (the genuinely unsafe combination). Prefer (A) if the test allows — it's cleaner.
   - Keep the `post_load_weights` assert (L871) and SP guard (L885) untouched — both still valid.

2. **No new flag needed** — `server_args.enable_torch_compile` + `SGLANG_TORCH_COMPILE_MODE` already exist `[verified L284-309 denoising.py]`. Confirm `add_standard_decoding_stage` for OmniDreams routes through `_maybe_enable_torch_compile` (it does, via the shared DenoisingStage base — verify call path on GPU).

3. **`configs/pipeline_configs/omnidreams.py`** — optionally add a doc note that `enable_torch_compile` is supported + the required `max-autotune-no-cudagraphs` mode. No behavior change.

**Risk:** if (A) is wrong and the parent forward *does* enter under compile, the dynamic chunk loop will graph-break repeatedly → slow or crash. Mitigation: design (B) fallback; the smoke test in §8.1 catches this before any benchmark.

---

## 4. P0 — CUDA Graph (biggest single win)

**Goal:** capture the repeated identical-shape DiT calls (`predict_flow` denoise steps + context re-forward) into a CUDA graph and replay, eliminating per-launch CPU overhead. Expected ↓30-50% `[report]`.

**Capture unit:** the raw `self.transformer(...)` call **only** — i.e. the DiT `forward()` body invoked inside the `predict_flow` closure `[verified L803-813]` and the context re-forward `[verified L834-844]`. *Not* the whole `predict_flow` closure: the closure also does the `pin`/`inject_mask` mutation (L801-802) which MUST stay eager (see risk #4). Both DiT calls have shapes constant across chunks (for fixed h/w). The KV-cache `before_update`/`after_update` ops `[verified L815-847]` are OUTSIDE the captured region — so the graph captures **only** the pure DiT forward, and the cache window-roll happens in eager Python between replays. This is the clean split that makes P0 tractable here.

**CUDA stream synchronization (critic finding):** the eager KV-window roll (`before_update`/`after_update` at L815-847) writes the static `_k`/`_v` buffers that the captured graph then reads. These run on the (default) stream; the graph replay runs on the capture stream. The runner MUST ensure ordering — either replay on the same stream as the eager cache ops, or insert an explicit `torch.cuda.current_stream().wait_stream(...)` / event sync between the eager roll and `g.replay()`. A missing sync = the graph reads a half-written window → silent corruption. Add this to the runner design and to numerical test §8.3.

**Why the KV cache doesn't break capture `[mixed]`:** `BlockKVCache._k/_v` are pointer-stable static buffers `[verified L126-127 kvcache — prior read]`. During capture, `cached_k`/`cached_v` reads + the in-place `update` writes hit the *same* memory addresses on every replay. **BUT** `cached_k`/`cached_v` return a *variable-length slice* of the static buffer during the fill phase and a *static* slice only in steady-state `[verified L284-290 kvcache per critic]` `[UNTESTED-SPECULATION on graph-capture interaction]`. → **This is the central P0 risk (both reviewers' #1):** the slice length changes per chunk during fill (chunk 0 sees a shorter window than chunk 5). A captured graph bakes the slice shape at capture time, so a graph captured at one window length replays wrong at another.

**Two-layer mitigation:**
1. **Steady-state-only capture:** run the fill-phase chunks (first `window_size_t/len_t` chunks) eager; capture once the window is steady and the slice length is constant. Mirrors FlashDreams' `drain()` + capture-at-steady-state lifecycle `[flashdreams-doc: capture AR idx = chunks_total//len_t]`.
2. **Full-buffer read with external length (critic refinement):** the attention inside the captured region should read the **full static `_k`/`_v` buffer** (fixed shape) and receive the valid window length as an externally-managed value (a captured constant in steady-state, or masked), rather than relying on a Python-level variable slice that bakes a shape. This makes the captured region shape-invariant by construction and is more robust than (1) alone. Confirm the actual `cached_k`/`cached_v` slicing at L284-290 supports this before committing; if the slice is unavoidable, fall back to (1) strictly.

**FlashDreams reference `[flashdreams-doc]`:** `CUDAGraphWrapper` with `drain()` (warm-up that triggers lazy init + autotune, L205-217), `capture`/`replay` (L219-255), `reset` (L114-120), `set_or_copy` (static-buffer copy-in, L26-47). The `eager_mode=True/False` split separates KV-cache window updates from `predict_flow` (network.py forward L411-500). We replicate this lifecycle in SGLang.

### Files & changes

1. **NEW `runtime/models/dits/omnidreams_cuda_graph.py`** — a `OmniDreamsCUDAGraphRunner` (SGLang-native port of FlashDreams `CUDAGraphWrapper`):
   - Static input buffers: `noisy [B, chunk_tokens, in_d]`, `timestep [.]`, plus references to the (already-static) `caches`, and the per-chunk-varying `rope_freqs`/`hdmap_chunk`/`cond_mask`. **Key design question:** `rope_freqs` changes per chunk (`rope.shift_t(chunk_idx)` L787) and `hdmap_chunk` changes per chunk (L795). For a single captured graph these must become **static buffers copied-in per replay** (`set_or_copy` pattern). `cond_mask` toggles only first-vs-rest (`cond_mask_c0` vs `cond_mask_zero` L789) → 2 graphs, or copy-in.
   - `drain(predict_flow)`: run N eager warm-up iterations to fire lazy allocations + (if P3 on) inductor autotune `[flashdreams-doc]`.
   - `capture(predict_flow)`: `torch.cuda.graph(g)` over one `predict_flow` call with static inputs.
   - `replay(noisy, t, rope, hdmap, cond)`: copy-in to static buffers, `g.replay()`, return static output (cloned).
   - `reset()`: drop graph + buffers (per-request or per-resolution-change).

2. **`runtime/pipelines_core/stages/model_specific_stages/omnidreams.py`** — wire the runner into the AR loop (L786-850):
   - Construct the runner lazily on first steady-state chunk (after the KV window fills).
   - Route both the `scheduler.sample`→`predict_flow` denoise calls AND the context re-forward through `runner.replay(...)` when in steady-state + flag on; else call the DiT eager (current behavior, fill phase + fallback).
   - **Preserve exact semantics:** the `pin`/`inject_mask` first-frame injection (L801-802, L822-823, L832-833) must stay in eager Python *around* the graph (it mutates `noisy`/`clean`/`ctx_latent` before/after the DiT call). The graph captures only the raw `self.transformer(...)`.
   - `before_update`/`after_update` stay eager between replays (unchanged).

3. **NEW flag** `enable_omnidreams_cuda_graph` (or reuse a generic `enable_cuda_graph` if one exists for diffusion — check `server_args.py`). Register the env override in **`environ.py`** as `SGLANG_OMNIDREAMS_CUDA_GRAPH` per `env-var-conventions` `[verified .claude/rules]`. Default **off** until §8 verifies.

4. **`configs/pipeline_configs/omnidreams.py`** — add `enable_cuda_graph: bool = False` config field + plumb to the stage.

**Risks:**
- **Variable KV-window slice during fill** (above) → capture only at steady-state. **Highest risk; gates the whole P0.**
- **Per-chunk-varying `rope_freqs`/`hdmap`** → static copy-in buffers (`set_or_copy`). If a tensor we assumed static is actually re-allocated per chunk, replay reads stale memory → silent wrong output. Mitigation: numerical-alignment test §8.3 (graph-on vs graph-off bitwise/tolerance compare).
- **First-frame `pin` injection inside the captured region** would bake `image_full` into the graph → MUST stay eager outside. `[verified the pin ops are outside self.transformer at L801-802/L834]` — design preserves this.
- **VRAM** for graph pool +1-3GB `[UNTESTED-SPECULATION]` — measure §8.
- **CUDA Graph + torch.compile collision** → mode `max-autotune-no-cudagraphs` mandatory `[verified default]`.

---

## 5. P1 — LightTAE decoder (decode ↓3-5×, FVD tradeoff)

**Goal:** replace the Wan VAE *decode* path with the LightTAE (TAEHV) tiny decoder. Latent space is unchanged (16-ch, Wan 2.1 mean/std `[verified project-memory note + wanvae config L21-56]`), so only the decoder swaps.

**FlashDreams reference `[flashdreams-doc]`:** TAEHV at `recipes/taehv/impl.py` — `MemBlock` (L62), `TGrow` (L110), `Decoder` (L123-224), `frames_to_trim = 2**sum(decoder_time_upscale)-1` (L317). Checkpoint remap `recipes/taehv/checkpoint.py`: `legacy_to_blocks_keys` (L36), `truncate_oversize_tgrow_weights` (L57), channels `(256,128,64,64)`. Checkpoint: `lighttaew2_1.pth` from `lightx2v/Autoencoders` `[project-memory note]`. Porting difficulty: **LOW-MEDIUM** `[flashdreams-doc]`.

### Files & changes

1. **NEW `runtime/models/vaes/taehv.py`** — port TAEHV decoder (MemBlock, TGrow, Decoder). Pure PyTorch, bf16/fp32. Expose `decode(latents) -> pixels` matching the `AutoencoderKLWan.decode` signature the decoding stage expects, including the causal temporal feature-cache contract (`causal_decode`/`reset_causal_decode_state` `[verified L887-920 wanvae.py — prior read]`) **or** the simpler frame-trim contract (`frames_to_trim`) — decide by reading how `add_standard_decoding_stage` calls decode.

2. **NEW `configs/models/vaes/taehv.py`** — `LightTAEConfig` (channels `(256,128,64,64)`, time-upscale factors → `frames_to_trim`).

3. **`runtime/pipelines/omnidreams_pipeline.py`** — add `_load_light_tae(path)` mirroring `_load_wan_vae` (L170-206); use the checkpoint remap (`legacy_to_blocks_keys`, `truncate_oversize_tgrow_weights`). In `load_modules` (L275-347), branch on config flag: load LightTAE decoder instead of (or alongside) Wan VAE decoder. Resolve path via a new `_LIGHTTAE_RELDIRS` or explicit config path.

4. **`configs/pipeline_configs/omnidreams.py`** — add `use_light_tae: bool = False` + `light_tae_path: str | None = None`.

5. **`create_pipeline_stages` (L349-375)** — when `use_light_tae`, build the decoding stage with the LightTAE decoder. Keep `add_standard_decoding_stage` if the decoder honors the same interface; else a thin custom decode stage.

**Risk:** frame-count / trim mismatch (`frames_to_trim=2**sum(...)-1`) → off-by-N frames or temporal seam. Mitigation: frame-count assertion test + visual check §8. Checkpoint key remap is the fiddly part — port `checkpoint.py` faithfully and add a unit test (state-dict key coverage) mirroring `test_omnidreams_regression.py` patterns `[verified test file style]`.

---

## 6. P2 — LightVAE encoder (encode ↓2-3×, FVD tradeoff)

**Goal:** replace the Wan VAE *encode* path (used for the first-frame image + the HDMap causal clip `[verified _vae_encode_normalized + _encode_hdmap — prior read]`) with the 75%-pruned LightVAE encoder. Latent space unchanged.

**FlashDreams reference `[flashdreams-doc]`:** LightVAE `pruning_rate=0.75` at `vae.py:1034`. Checkpoint `lightvaew2_1.pth` from `lightx2v/Autoencoders` `[project-memory note]`. Porting difficulty: not separately rated but encoder is structurally simpler than the FP8 path.

### Files & changes

1. **NEW `runtime/models/vaes/omnidreams_light_vae.py`** — port the pruned LightVAE encoder. Expose `encode(pixels) -> latents` matching `AutoencoderKLWan.encode` (L922+ `[prior read]`), same normalization (Wan mean/std).

2. **`runtime/pipelines/omnidreams_pipeline.py`** — `_load_light_vae_encoder(path)`; branch in `load_modules`. Two sub-cases:
   - **encoder-only swap:** LightVAE encoder + Wan decoder (or LightTAE decoder from P1).
   - The HDMap encode path `_encode_hdmap` (L376-471 `[prior read]`) and first-frame encode `_vae_encode_normalized` (L325-346) both call the VAE encoder → both must route to LightVAE when flagged.

3. **`configs/pipeline_configs/omnidreams.py`** — add `use_light_vae_encoder: bool = False` + `light_vae_path`.

4. **Stage wiring** — `OmniDreamsBeforeDenoisingStage` holds the encoder; swap the instance based on flag.

**Risk:** the HDMap clip is VAE-encoded as ONE causal clip then sliced `[verified memory note]` — LightVAE must honor the same causal/tiling contract or the per-chunk HDMap slices misalign → motion control degrades. Mitigation: HDMap-motion regression (frame-to-frame meandiff, the `>0` motion check from the run-command memory) §8.

---

## 7. P4 — Native FP8 LightVAE (DESIGN + FEASIBILITY SPIKE ONLY)

**Scope (user-chosen):** *design + feasibility spike*, **NOT** a full kernel port. Deliverable = a written design + a microbenchmark that proves the `sm_120` FP8 path runs and is faster, *without* wiring it into production inference.

**FlashDreams reference `[flashdreams-doc]`:** FP8 LightVAE in `omnidreams_singleview/src/vae_streaming/` (~5900 lines). TIN16 format `[T, C/16, H, W, 16]` uint8, `kFp8ChannelsPerSlice=16`, `sm_120` CUTLASS, `sm120-tma-pool.patch`. Env var `OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH` (calibrated state). Calibration script `scripts/export_lightvae_fp8_state.py`. Porting difficulty: **HIGH** `[flashdreams-doc]`.

### Feasibility-spike deliverables (no production wire-up)

1. **Design doc section** (append to this plan or a sibling): the TIN16 layout, the `sm_120` CUTLASS dependency, the calibrated-state contract (`OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH` → SGLang `SGLANG_OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH` via `environ.py`), and where it would slot into P2's encoder (`omnidreams_light_vae.py` as an FP8 backend behind a third flag `use_light_vae_fp8`).
2. **Spike script** (NOT in the inference path) under `scripts/` or `test/`: load the FlashDreams FP8 kernels (or a minimal CUTLASS `sm_120` GEMM), run a single conv/GEMM layer in FP8 vs bf16, measure latency + numerical error on an `sm_120` card. Prove: (a) kernels build on the target GPU, (b) FP8 is faster, (c) error within tolerance vs bf16 reference using a calibrated state.
3. **Go/no-go criteria** for a future full port: speedup ≥1.5× over bf16 LightVAE AND FVD delta ≤ agreed budget AND build reproducible on `sm_120`.

**Risks:** HIGH porting cost (5900 LOC), `sm_120`-only (no datacenter Blackwell), CUTLASS/TMA patch maintenance, calibration-state availability (need a calibrated `lightvae_fp8_state` — must obtain or generate via the export script). The spike intentionally de-risks before committing to the port.

---

## 8. GPU verification plan (latency / correctness / numerical alignment)

Run on the `gpu` host, model at `/root/blockdata/omni-dreams`, 704×1280 / 29f, seed 42, HD-map mandatory `[verified memory omnidreams-run-command]`. Baseline command from memory (serve or `sglang generate` CLI). Always read the `warmup excluded` timing line `[verified multimodal_gen CLAUDE.md perf-measurement]`.

### 8.1 P3 smoke (BEFORE benchmarking)
- 1-chunk generate with `enable_torch_compile=1`, `SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs`. **Pass = no assert, no fullgraph break spam, output finite.** Confirms the relaxed-assert design (A vs B §3).

### 8.2 Latency (per phase, cumulative)
- Measure ms/chunk + end-to-end FPS (warmup-excluded) for: **baseline → +P3 → +P0 → +P1 → +P2**.
- Record VRAM peak at each step (catch P0 graph-pool growth).
- Expected (from report, to be confirmed): P3 ↓10-15%, P0 ↓30-50%, P1 decode ↓3-5×, P2 encode ↓2-3×.

### 8.3 Numerical alignment (lossless steps P0/P3)
- **P0 graph-on vs graph-off:** same seed, compare DiT output tensors — must match within bf16 tolerance (e.g. `rtol=1e-2, atol=1e-2`; tighten if exact). Any divergence = static-buffer/stale-read/stream-sync bug §4.
- **Per-chunk boundary comparison (critic finding — required):** comparing only *final* latents can hide a single corrupted chunk that gets averaged/overwritten by later chunks. Capture and compare the DiT output **per chunk**, with explicit attention to the **fill→steady-state transition chunk** (where an off-by-one in the steady-state threshold corrupts exactly one chunk). The first steady-state chunk and the last fill chunk are the highest-signal comparison points.
- **P3 compile-on vs off:** same seed, compare final latents.
- **P3+P0 composition (critic finding):** verify both stacked together (compile ON + graph ON) against baseline — the inductor-cudagraph-mode interaction (§4 risk #5) only surfaces when both are active.
- **Stream-sync stress:** run the P0 graph path with multiple back-to-back chunks at high occupancy; a missing eager-roll↔replay sync (§4) shows as intermittent divergence, not deterministic — run the comparison ≥3× to catch nondeterminism.
- Use `SGLANG_OMNIDREAMS_DIAGNOSTICS` (if present; else add a temporary dump hook) to capture per-chunk latents.

### 8.4 Quality (lossy steps P1/P2)
- **FVD:** generate the verified sample clip (`23599139-...`) at baseline (full VAE) and with LightTAE/LightVAE; compute FVD against GT. Expect ~24.8 (full) → ~45.4 (light). Confirm within tolerance of the reported numbers.
- **HDMap motion regression:** frame-to-frame meandiff must stay in the moving range (~0.59-5.66, avg ~2.0) — NOT ~0 (frozen) `[verified memory]`. Per-frame std stable ~51 (no gray-mush) `[verified memory]`.
- Visual spot-check first/mid/last frame.

### 8.5 Regression tests (CPU, no GPU)
- Extend `test/unit/test_omnidreams_regression.py` `[verified style]`: LightTAE checkpoint-key coverage, LightVAE encode shape, `frames_to_trim` math, CUDA-graph runner buffer-shape invariants (meta/CPU where possible).

---

## 9. Ordered execution sequence (after approval)

1. **P3:** GPU smoke §8.1 → choose assert design (A/B) → relax assert in `omnidreams.py` → confirm compile path → latency + numerical §8.2/8.3.
2. **P0:** port `omnidreams_cuda_graph.py` → wire steady-state capture into AR loop → flag + `environ.py` → numerical alignment §8.3 (the critical gate) → latency §8.2 + VRAM.
3. **P1:** port `taehv.py` + config + checkpoint remap → loader branch → decode flag → FVD + frame-count §8.4 + unit tests §8.5.
4. **P2:** port `omnidreams_light_vae.py` + config → encoder branch (first-frame + HDMap) → encode flag → HDMap-motion regression §8.4.
5. **P4 spike:** design section + spike script on `sm_120` → go/no-go.
6. Final: full §8 sweep with all flags, update report numbers, summary to user.

---

## 10. Files touched (summary)

**New:**
- `runtime/models/dits/omnidreams_cuda_graph.py` (P0 runner)
- `runtime/models/vaes/taehv.py` + `configs/models/vaes/taehv.py` (P1)
- `runtime/models/vaes/omnidreams_light_vae.py` (P2)
- `scripts/` spike script (P4)
- test additions in `test/unit/test_omnidreams_regression.py`

**Modified:**
- `runtime/models/dits/omnidreams.py` (relax assert L878-882) — P3
- `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py` (wire P0 into AR loop L786-850) — P0
- `runtime/pipelines/omnidreams_pipeline.py` (loaders + stage wiring) — P1/P2
- `configs/pipeline_configs/omnidreams.py` (flags: `enable_cuda_graph`, `use_light_tae`, `use_light_vae_encoder`, paths) — P0/P1/P2
- `environ.py` (register `SGLANG_OMNIDREAMS_CUDA_GRAPH`, FP8 state path) — P0/P4

**Unchanged (deliberately):** `omnidreams_kvcache.py` (static buffers already CUDA-graph-friendly), `wanvae.py` (kept as default/fallback).

---

## 11. Creative enhancements (optional, post-baseline)

- **Dual-graph for first-vs-rest chunk** (cond_mask_c0 vs cond_mask_zero) instead of copy-in — fewer copies, 2 small graphs.
- **Capture the 2 denoise steps as a single mega-graph** (both timesteps + the context re-forward in one replay) — eliminates 3 launches/chunk instead of 1. Higher risk (timestep must be a copy-in static buffer).
- **Auto-mode select:** pick LightTAE/LightVAE automatically when latency budget is set (tie into the `sglang-diffusion-performance` skill's flag-recommendation logic).
- **FP8 DiT (separate from P4 VAE FP8):** the report's P-series stops at VAE FP8; a ModelOpt FP8 DiT (`sglang-diffusion-modelopt-quant` skill) is an orthogonal future win on `sm_120`.

---

## 12. Risk register (consolidated)

| # | Risk | Phase | Severity | Mitigation | Evidence tag |
|---|---|---|---|---|---|
| 1 | Parent forward enters under compile → dynamic-loop graph-break | P3 | Med | smoke §8.1, design B fallback | `[UNTESTED-SPECULATION]` |
| 2 | Variable KV-window slice during fill breaks fixed-shape graph | P0 | **High** | capture only at steady-state | `[verified slice behavior]` `[UNTESTED interaction]` |
| 3 | Per-chunk rope/hdmap assumed static → stale replay read | P0 | **High** | static copy-in + numerical §8.3 | `[verified per-chunk L787/795]` |
| 4 | `pin` injection baked into graph | P0 | Med | keep pin eager outside capture | `[verified L801-833]` |
| 5 | CUDA graph ↔ inductor cudagraph collision | P0+P3 | Med | `max-autotune-no-cudagraphs` | `[verified default]` |
| 6 | LightTAE frame-trim off-by-N / seam | P1 | Med | frame-count test + FVD | `[flashdreams-doc frames_to_trim]` |
| 7 | LightTAE checkpoint key remap errors | P1 | Med | port `checkpoint.py` + key-coverage test | `[flashdreams-doc]` |
| 8 | LightVAE HDMap causal-clip contract mismatch → frozen/misaligned motion | P2 | **High** | HDMap-motion regression §8.4 | `[verified memory motion check]` |
| 9 | FVD regression beyond 45.4 | P1+P2 | Med | FVD gate §8.4, flags default off | `[report]` |
| 10 | P4 FP8 build fails on non-`sm_120` | P4 | Low (spike) | spike isolates before port | `[flashdreams-doc sm_120]` |
| 11 | VRAM growth from graph pool exceeds card | P0 | Med | measure §8, offload fallback | `[UNTESTED-SPECULATION]` |
| 12 | Missing stream sync: eager cache roll ↔ graph replay → intermittent corruption | P0 | **High** | same-stream replay or explicit event sync; nondeterminism stress §8.3 | `[verified eager ops L815-847]` `[UNTESTED interaction]` |
| 13 | Per-chunk corruption hidden by final-only comparison | P0 verify | Med | per-chunk + fill→steady transition compare §8.3 | `[critic]` |
| 14 | User silently ships degraded LightVAE path | P1/P2 | Med | default-off flag + one-time `logger.warning` §1 | `[critic]` |
| 15 | Mid-request rollback (graph fails mid-rollout) leaves partial state | P0 | Low | per-request `runner.reset()` on exception → fall back to eager for remainder | `[critic]` |

---

## 13. Open questions to resolve during execution (not blockers)

- Does `module.compile()` cause the *parent* OmniDreamsDiT.forward to run under `is_compiling()`? → §8.1 settles P3 design A vs B. `[UNTESTED-SPECULATION]`
- Exact decode interface `add_standard_decoding_stage` expects (causal-cache vs frame-trim) → read at P1 start.
- Is there an existing generic diffusion `enable_cuda_graph` flag to reuse vs a new OmniDreams-specific one? → check `server_args.py` at P0 start.
- Calibrated LightVAE FP8 state availability for the P4 spike (obtain vs generate via `export_lightvae_fp8_state.py`).
