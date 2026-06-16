# OmniDreams Config Migration Guide

## Overview

OmniDreams pipeline config migrated from flat boolean fields to nested Config
dataclasses with orthogonal impl selection + FP8 acceleration. Old fields are
removed — `__post_init__` raises `ValueError` with migration guidance.

## Field Mapping

| Old Field (removed) | New Config |
|---|---|
| `use_light_vae_encoder` | `image_encoder_config.impl` / `encoder_config.impl` |
| `light_vae_path` | `image_encoder_config.checkpoint_path` / `encoder_config.checkpoint_path` |
| `use_light_vae_fp8` | `encoder_config.native_acceleration` + `encoder_config.fp8_state_path` |
| `light_vae_fp8_state_path` | `encoder_config.fp8_state_path` / `image_encoder_config.fp8_state_path` |
| `use_light_tae` | `decoder_config.impl` |
| `light_tae_path` | `decoder_config.checkpoint_path` |
| `use_fp8_dit` | `native_dit_acceleration` |
| `fp8_dit_attention_backend` | `native_dit_backend` |

## Before / After

### Old (flat booleans)

```python
config = OmniDreamsPipelineConfig(
    use_light_vae_encoder=True,
    light_vae_path="/path/lightvae.pth",
    use_light_vae_fp8=True,
    light_vae_fp8_state_path="/path/fp8.pt",
    use_light_tae=True,
    light_tae_path="/path/lighttae.pth",
    use_fp8_dit=True,
    fp8_dit_attention_backend="cudnn",
)
```

### New (nested Config)

```python
from sglang.multimodal_gen.configs.models.omnidreams_components import (
    OmniDreamsTextEncoderConfig,
    OmniDreamsVAEDecoderConfig,
    OmniDreamsVAEEncoderConfig,
)

config = OmniDreamsPipelineConfig(
    # image_encoder and encoder are independent instances
    image_encoder_config=OmniDreamsVAEEncoderConfig(
        impl="lightvae",
        checkpoint_path="/path/lightvae.pth",
        native_acceleration="required",
        fp8_state_path="/path/fp8.pt",
    ),
    encoder_config=OmniDreamsVAEEncoderConfig(
        impl="lightvae",
        checkpoint_path="/path/lightvae.pth",
        native_acceleration="required",
        fp8_state_path="/path/fp8.pt",
    ),
    decoder_config=OmniDreamsVAEDecoderConfig(
        impl="lighttae",
        checkpoint_path="/path/lighttae.pth",
    ),
    native_dit_acceleration="required",
    native_dit_backend="cudnn",
)
```

## New: Text Encoder FP8

```python
# Offline: quantize Cosmos-Reason1-7B to W8A8 FP8
python -m sglang.multimodal_gen.tools.export_cosmos_reason_fp8 \
    --model-id nvidia/Cosmos-Reason1-7B \
    --save-dir ./Cosmos-Reason1-7B-W8A8-FP8

# Config
config = OmniDreamsPipelineConfig(
    text_encoder_config=OmniDreamsTextEncoderConfig(
        impl="fp8_w8a8",
        fp8_model_path="./Cosmos-Reason1-7B-W8A8-FP8",
    ),
)
```

## Three-State Acceleration Mode

All FP8-capable components use `NativeAccelerationMode = Literal["auto", "disabled", "required"]`:

| Mode | Behavior |
|---|---|
| `"disabled"` | No FP8 (default) |
| `"auto"` | Try FP8, fall back to PyTorch on failure |
| `"required"` | FP8 required, raise on failure |

## Composable Mixed Precision

Each component is independently configurable:

```python
config = OmniDreamsPipelineConfig(
    text_encoder_config=OmniDreamsTextEncoderConfig(impl="fp8_w8a8", ...),
    image_encoder_config=OmniDreamsVAEEncoderConfig(impl="wanvae"),  # full quality
    encoder_config=OmniDreamsVAEEncoderConfig(impl="lightvae", native_acceleration="required"),
    decoder_config=OmniDreamsVAEDecoderConfig(impl="lighttae"),
    native_dit_acceleration="required",
)
```
