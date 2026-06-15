# P4b — LightVAE FP8 (native) Implementation Plan

**Goal:** route the OmniDreams encode-side LightVAE through the vendored native FP8
streaming encoder (`omnidreams_vae_encode_wan_fp8`), with a clean PyTorch fallback
on CPU / no-ext / non-conforming input. Mirrors P4a (DiT FP8) in structure and the
FlashDreams reference `integrations/omnidreams/omnidreams/vae_native.py`.

**Dev/verify constraints:** CPU-only dev box; native ext is `sm_120`-only (GPU
session, later, over SSH). No commit / no push. Test runs delegated to a haiku
sub-agent with `TORCHDYNAMO_DISABLE=1`.

---

## Key facts that shape the design (verified by reading the vendored sources)

1. **Native output is NORMALIZED, PyTorch `encode()` contract is RAW.**
   The native kernel ends in `lightvae_fp8_extract_mu_normalize_tin16(..., mean,
   inv_std, h/8, w/8)` — it extracts mu (first 16 of 32 ch) **and** applies
   `(mu - mean) * inv_std`. SGLang's `_vae_encode_normalized`
   (`stages/.../omnidreams.py:356`) calls `vae.encode(x).mode()` then applies
   `(z - latents_mean) / latents_std` itself. So `LightVAEEncoder.encode()` must
   return **raw** mu. ⇒ the native path must **de-normalize**:
   `raw = normalized * latents_std + latents_mean` (per-channel, 16 ch).
   This keeps `encode()`'s contract identical for both backends and requires
   **zero** changes to the stage / `_vae_encode_normalized`.

2. **The native handle is built from the live model + a calibrated FP8 state.**
   `vae_weights.build_lightvae_encoder_fp8_staged_state(model, fp8_state, ext)`:
   - reads **weights** from the live module: `model.encoder` (Encoder3d:
     `.conv1/.downsamples/.middle/.head`), `model.conv1` (post quant conv),
     `model.mean` (16), `model.inv_std` (16).
   - reads **scales** from `fp8_state` (the calibration export): `*.weight_scale`
     (per-out-channel) and `*.activation_scale` keyed by module path.
   - calls `ext.lightvae_fp8_prepare_conv2d_weight_krsc(...)` (GPU symbol).
   - returns a prepared dict (`scale_input/conv1_w/blocks/ds0..2/mid_attn/mean/
     inv_std/...`) consumed by `ext.omnidreams_vae_create_wan_encoder_fp8(state)`.
   `_require_lightvae_layout`: needs `encoder.downsamples` len **11** and
   `encoder.conv1.out_channels == 24`. Our `Encoder3d(base_dim=96,pruning=0.75)`
   already satisfies both (dims `[24,24,48,96,96]` → 3+3+3+2 = 11 downsamples;
   `encoder.conv1 = CausalConv3d(3,24,3)`). ⇒ **must add `model.mean`/`model.inv_std`
   buffers** to `LightVAEEncoder`; everything else already lines up.

3. **Native I/O boundary:** input must be `[1,3,T,H,W]`, CUDA, **fp16**, H&W % 8 == 0
   (asserted in `lightvae_encode_fp8_native`). Output is fp16 normalized
   `[1,16,Tl,h,w]`. Batch must be 1.

4. **Streaming chunking matches the PyTorch path exactly** (FlashDreams
   `_NativeWanVAEEncoderExecutor.try_encode`): reset the handle, encode a 1-frame
   causal seed, then `TEMPORAL_WINDOW(=4)`-frame body chunks + a short tail, with
   `use_cache=True` so the handle's internal streaming cache carries left-context.
   Per `encode()` call we run a **fresh** logical pass ⇒ call
   `omnidreams_vae_reset_wan_encoder_fp8(handle)` at the start of every `encode()`.

5. **Public native API** (from `multimodal_gen/native/__init__.py`):
   `load_extension()` (no-arg → ext or None; build fails on CPU → None),
   `singleview_loader.load_python_module("vae_weights")`, and ext symbols
   `omnidreams_vae_backend_status / _create_wan_encoder_fp8 / _encode_wan_fp8 /
   _reset_wan_encoder_fp8`.
   NOTE: P4a's `omnidreams_fp8.py` calls `load_extension(NativeAccelerationConfig(...))`,
   but the re-exported `load_extension` signature is `(build_root=None,*,max_jobs,
   verbose)` — passing a config as `build_root`. Harmless on CPU (returns None via
   the build-failure path) but **latent-buggy on GPU**. P4b will call
   `load_extension()` with **no args** (correct). Flag P4a separately; do not fix
   here.

---

## Step 1 — `LightVAEEncoder` FP8 backend
File: `python/sglang/multimodal_gen/runtime/models/vaes/omnidreams_light_vae.py`

### Interface changes
```python
class LightVAEEncoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        latents_mean: list[float],
        latents_std: list[float],
        dtype: torch.dtype = torch.float32,
        base_dim: int = 96,
        z_dim: int = 16,
        pruning_rate: float = 0.75,
        temperal_downsample: tuple[bool, bool, bool] = (False, True, True),
        fp8_state_path: str | None = None,   # NEW: calibrated FP8 state (.pt)
        fp8_required: bool = False,          # NEW: raise instead of falling back
    ) -> None: ...
```

### `__init__` additions (after the existing weight load + `self.to(dtype)`)
- Register normalization buffers consumed by the native staged-state builder:
  ```python
  self.register_buffer("mean",
      torch.tensor(self.latents_mean, dtype=torch.float32).reshape(-1),
      persistent=False)
  self.register_buffer("inv_std",
      (1.0 / torch.tensor(self.latents_std, dtype=torch.float32)).reshape(-1),
      persistent=False)
  ```
  (Real tensors, not on meta; moved with `.to(device/dtype)` by `_load_light_vae`.
  `strict=False` load ignores them; builder checks `numel()==16`.)
- Native lazy state (no work at init — model may still be on CPU pre-residency):
  ```python
  self._fp8_state_path = fp8_state_path
  self._fp8_required = bool(fp8_required)
  self._fp8_enabled = fp8_state_path is not None
  self._native_ext = None        # cached extension module
  self._native_handle = None     # _OmnidreamsNativeWanVaeEncoderFp8
  self._native_device = None     # device the handle was built on
  self._native_disabled = False  # set after a failed attempt (auto mode)
  ```

### New helper: `_get_native_handle(device) -> handle | None`
Lazy build, cached per device; mirrors FlashDreams `_get_native_encoder`.
```python
def _get_native_handle(self, device):
    if not self._fp8_enabled or self._native_disabled:
        return None
    if self._native_handle is not None and self._native_device == device:
        return self._native_handle
    try:
        from sglang.multimodal_gen.native import load_extension
        from sglang.multimodal_gen.native.singleview_loader import load_python_module
        ext = self._native_ext or load_extension()
        if ext is None:
            raise RuntimeError("native ext unavailable (sm_120 build required)")
        status = ext.omnidreams_vae_backend_status("vae_encoder", "fp8")
        if not bool(status.get("available", False)):
            raise RuntimeError(f"native vae_encoder fp8 unavailable: {status.get('reason')}")
        vae_weights = load_python_module("vae_weights")
        fp8_state = vae_weights.load_lightvae_fp8_state(self._fp8_state_path)
        staged = vae_weights.build_lightvae_encoder_fp8_staged_state(self, fp8_state, ext)
        handle = ext.omnidreams_vae_create_wan_encoder_fp8(staged)
        self._native_ext = ext
        self._native_handle = handle
        self._native_device = device
        return handle
    except Exception as e:           # noqa: BLE001
        if self._fp8_required:
            raise
        logger.warning("OmniDreams LightVAE FP8 native unavailable (%s); using PyTorch encode.", e)
        self._native_disabled = True
        return None
```
(Add module-level `logger = init_logger(__name__)` import.)

### New helper: `_encode_native(handle, x) -> torch.Tensor` (raw mu)
```python
def _encode_native(self, handle, x):
    ext = self._native_ext
    ext.omnidreams_vae_reset_wan_encoder_fp8(handle)        # fresh streaming pass
    xf = x.to(torch.float16).contiguous()                  # [1,3,T,H,W] cuda fp16
    outs = [ext.omnidreams_vae_encode_wan_fp8(handle, xf[:, :, :1], True)]
    xf = xf[:, :, 1:]
    t = xf.shape[2]
    body = (t // TEMPORAL_WINDOW) * TEMPORAL_WINDOW
    for i in range(0, body, TEMPORAL_WINDOW):
        outs.append(ext.omnidreams_vae_encode_wan_fp8(handle, xf[:, :, i:i+TEMPORAL_WINDOW], True))
    if body < t:
        outs.append(ext.omnidreams_vae_encode_wan_fp8(handle, xf[:, :, body:], True))
    z_norm = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)  # [1,16,Tl,h,w] normalized
    # de-normalize -> raw mu (downstream _vae_encode_normalized re-applies (z-mean)/std)
    mean = self.mean.to(z_norm.device, z_norm.dtype).view(1, -1, 1, 1, 1)
    std = (1.0 / self.inv_std).to(z_norm.device, z_norm.dtype).view(1, -1, 1, 1, 1)
    return z_norm * std + mean
```

### `encode()` dispatch (guard native eligibility, else existing PyTorch path)
```python
@torch.inference_mode()
def encode(self, x):
    if self._fp8_enabled and x.is_cuda and x.shape[0] == 1:
        handle = self._get_native_handle(x.device)
        if handle is not None:
            return _LatentDist(self._encode_native(handle, x))
    # ---- existing PyTorch streaming path (unchanged) ----
    ...
```
Native eligibility = fp8 requested **and** CUDA **and** batch 1. CPU / B>1 / build
failure ⇒ PyTorch path (or raise if `fp8_required`). H/W % 8 is enforced natively;
if violated and required→ propagate, else the except in `_get_native_handle` won't
catch it (it's in `_encode_native`) — so also wrap the native call site: on
`Exception` with `not fp8_required`, log once and fall through to PyTorch **only if
the handle's stream is still fresh** (we reset at call start, so a first-chunk
failure is safe to fall back; mid-stream failure cannot — match FlashDreams by
letting it raise). Simplest: catch around the whole `_encode_native`; on failure
with `not required`, set `_native_disabled=True` and fall back.

---

## Step 2 — pipeline wiring
File: `python/sglang/multimodal_gen/runtime/pipelines/omnidreams_pipeline.py`

### `_load_light_vae` signature + body
```python
@staticmethod
def _load_light_vae(ckpt_path, vae_config, device, dtype,
                    fp8_state_path: str | None = None,
                    fp8_required: bool = False) -> nn.Module:
    encoder = LightVAEEncoder(
        checkpoint_path=ckpt_path,
        latents_mean=list(vae_config.latents_mean),
        latents_std=list(vae_config.latents_std),
        dtype=dtype,
        fp8_state_path=fp8_state_path,
        fp8_required=fp8_required,
    )
    return encoder.to(device).eval()
```

### New resolver (mirrors `_resolve_light_vae_path`)
```python
@staticmethod
def _resolve_light_vae_fp8_state_path(model_path, explicit) -> str | None:
    from sglang.multimodal_gen import envs
    cand = explicit or envs.SGLANG_OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH
    if not cand:
        return None
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError(f"light_vae_fp8_state_path not found: {cand}")
```

### `load_modules` — LightVAE branch (where `use_light_vae_encoder` is handled)
```python
fp8_state_path = None
if getattr(pipeline_config, "use_light_vae_fp8", False):
    fp8_state_path = self._resolve_light_vae_fp8_state_path(
        model_path, getattr(pipeline_config, "light_vae_fp8_state_path", None))
    logger.info("OmniDreams: LightVAE FP8 encode enabled (state=%s)", fp8_state_path)
light_vae = self._load_light_vae(
    lv_path, pipeline_config.vae_config, vae_device, vae_dtype,
    fp8_state_path=fp8_state_path,
    fp8_required=False,  # auto: fall back to PyTorch LightVAE if native missing
)
```
`use_light_vae_fp8` implies `use_light_vae_encoder` (config-level invariant; if the
latter is off, `light_vae` isn't built ⇒ fp8 has no effect — log a warning if
`use_light_vae_fp8 and not use_light_vae_encoder`). No stage-wiring change: the fp8
path lives entirely inside `LightVAEEncoder.encode`, and the encoder still occupies
the `vae_encoder` module slot.

---

## Step 3 — calibration export tool (GPU-only at runtime)
File: `python/sglang/multimodal_gen/test/spikes/export_lightvae_fp8_state.py`
(spikes/ to match `omnidreams_fp8_lightvae_spike.py`; not a unit test.)

Port FlashDreams `export_lightvae_fp8_state.py`, but build the **SGLang**
`LightVAEEncoder` so hooked module paths match what
`build_lightvae_encoder_fp8_staged_state` looks up (`encoder.downsamples.{i}.
residual.{j}`, `encoder.middle.{i}.*`, `encoder.head.*`, `encoder.conv1`, `conv1`).

### CLI
```
--ckpt           lightvaew2_1.pth (required)
--out            output .pt (required)
--calibration-video <mp4>  (required; per-frame RGB → [1,3,T,H,W] in [-1,1])
--device cuda  --height 720 --width 1280  --frames 13  --scale-max 24.0
```
(Drop the FlashDreams `omnidreams.config`/`derive_config` deps — instantiate
`LightVAEEncoder(checkpoint_path=ckpt, latents_mean/std from OmniDreamsVAEConfig,
dtype=fp16).to("cuda")`.)

### Logic (reused verbatim from the reference, retargeted)
- `_quantize_fp8_per_channel(w, channel_dim=0, scale_max)` → `(uint8 RCR bytes,
  fp16 scale[out])`.
- forward/pre hooks over `model.encoder.named_modules()` + `cache_step` wrapping
  → per-channel activation amax → `{name}.activation_scale` (fp16, 1-D).
  Manually seed `encoder.conv1.input` / `encoder.input` / `input` and `latent`.
- run one streaming encode (`model.encode(video_bcthw)`) to populate amax.
- `_build_fp8_state(model.state_dict(), activation_scales, scale_max)`:
  every floating `.weight` with `dim>=2` → `{name}` (uint8) + `{name}.weight_scale`;
  other floats → fp16; ints → as-is. Plus header keys:
  `__omnidreams_vae_fp8_version__=1`, `__omnidreams_vae_fp8_model_kind__=1`,
  `__omnidreams_vae_fp8_scale_max__=scale_max`.
- `torch.save(state, out)`.

Hook compatibility note: SGLang `LightVAEEncoder.encode(x)` takes no `cache=`
kwarg and runs its own internal streaming (fresh cache per call) — fine for
calibration (single pass). Use `model.encode(video_bcthw)` directly; the
`cache_step` calls still fire and get recorded.

---

## Step 4 — CPU tests
File: `python/sglang/multimodal_gen/test/unit/test_omnidreams_fp8.py` (extend)

1. `test_light_vae_fp8_falls_back_to_pytorch_on_cpu`: build
   `LightVAEEncoder(ckpt=<tiny synthetic or skip if no ckpt>, fp8_state_path=
   "/nonexistent")`; on CPU `encode([1,3,5,64,64])` returns finite `_LatentDist`
   with mode shape `[1,16,2,8,8]` (native disabled ⇒ PyTorch path). Assert
   `_native_disabled` becomes True after first call, and a 1-frame input →
   `[1,16,1,8,8]`.
   — If a real `lightvaew2_1.pth` isn't reachable in CI, construct the encoder on
   `meta`→random init is not possible (it loads a ckpt). So gate with
   `pytest.importorskip`/`skipif` on checkpoint presence, OR refactor the test to
   monkeypatch `torch.load` to return a synthetic state dict matching the pruned
   layout. Prefer: small helper that fabricates a state dict by instantiating the
   module on CPU (meta=False) with random weights and `state_dict()` — load that.
2. `test_light_vae_fp8_state_roundtrip` (pure-CPU, no native): exercise the
   export tool's `_quantize_fp8_per_channel` + `_build_fp8_state` header keys on a
   synthetic state dict → assert `__omnidreams_vae_fp8_{version,model_kind,
   scale_max}__` present, `weight_scale` shape `[out]`, uint8 weight bytes, version
   round-trips via `vae_weights.load_lightvae_fp8_state`.
3. `test_mean_inv_std_buffers`: `LightVAEEncoder` exposes `mean`/`inv_std`
   buffers, both `numel()==16`, `inv_std ≈ 1/std`.

Run: haiku sub-agent, `TORCHDYNAMO_DISABLE=1 <venv>/bin/python -m pytest
test/unit/test_omnidreams_fp8.py -x -q`.

---

## Step 5 — verify + handoff
- haiku sub-agent runs the OmniDreams CPU suite (`test_omnidreams_accel.py`,
  `test_omnidreams_fp8.py`, components/scaffold) with `TORCHDYNAMO_DISABLE=1`, plus
  `ruff check` on the touched files.
- Update `omnidreams_optimization_progress.md`: flip P4b to ✅ code-done /
  CPU-verified; list new files; refresh the GPU checklist (export state → encode
  FP8-vs-bf16 tol + HD-map motion).
- Notify user: P4b code-complete, GPU verification pending. **No commit / push.**

---

## Risks / open items
- **Activation-scale key naming** must match between the export tool (Step 3) and
  the staged-state builder. Mitigated by running the export against the *same*
  SGLang `LightVAEEncoder` module tree. Cannot be CPU-verified end-to-end (build is
  GPU-only) — flagged as a GPU checklist item.
- **De-normalize precision:** raw↔normalized round-trip in fp16. Negligible
  (values O(1), per-channel affine); downstream casts to dit_dtype anyway.
- **`_native_disabled` caching** prevents per-call retry storms after a failed
  build, but means a transient failure disables native for the encoder's lifetime.
  Acceptable (matches "fall back to PyTorch" intent); a fresh pipeline rebuilds.
- **P4a `load_extension` signature mismatch** noted above — out of scope for P4b,
  flag to user.
```
