# SPDX-License-Identifier: Apache-2.0
"""Pipeline config for NVIDIA OmniDreams.

Phase 0 wires the static structure (DiT config, VAE reuse, task type) and the
2-step flow-match sigma schedule. The denoising/decoding callbacks used at GPU
time are added in later phases.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
from sglang.multimodal_gen.configs.models.vaes.wanvae import OmniDreamsVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


def warp_flow_match_sigmas(
    denoising_timesteps: tuple[int, ...] = (1000, 450),
    flow_shift: float = 5.0,
    sigma_min: float = 0.0,
) -> list[float]:
    """OmniDreams 2-step flow-match sigma schedule.

    Each raw timestep ``t`` maps to ``s = t / 1000`` then is warped by
    ``shift*s / (1 + (shift-1)*s)``; ``sigma_min`` is appended as the final
    target. With the distilled defaults this yields ``[1.0, 0.8036, 0.0]``.
    """
    sigmas = [
        flow_shift * (t / 1000.0) / (1.0 + (flow_shift - 1.0) * (t / 1000.0))
        for t in denoising_timesteps
    ]
    sigmas.append(sigma_min)
    return sigmas


@dataclass
class OmniDreamsPipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2V
    # CFG disabled for the distilled checkpoint.
    should_use_guidance: bool = False
    # Native bf16 DiT; VAE in fp32 for numerical stability.
    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    # Flow-match warp shift (also drives warp_flow_match_sigmas).
    flow_shift: float | None = 5.0

    dit_config: OmniDreamsDiTConfig = field(default_factory=OmniDreamsDiTConfig)
    # A.5: OmniDreams uses a Cosmos-Predict2.5-based latent space; the
    # latents_mean/std defaults match Wan 2.1 (safe fallback). Override in
    # OmniDreamsVAEArchConfig once GPU validation confirms the correct values.
    vae_config: OmniDreamsVAEConfig = field(default_factory=OmniDreamsVAEConfig)

    # 2-step distilled flow-match schedule.
    denoising_timesteps: tuple[int, ...] = (1000, 450)
    sigma_min: float = 0.0

    # P0: capture the steady-state AR-rollout DiT calls into a CUDA graph and
    # replay (eliminates per-launch CPU overhead across the repeated
    # identical-shape calls). Numerically lossless. Default off until GPU
    # verification; env SGLANG_OMNIDREAMS_CUDA_GRAPH force-enables. The fill-phase
    # chunks (before the KV window is steady) always run eager.
    enable_cuda_graph: bool = False
    # Eager warmup iterations before graph capture (drains lazy allocs +
    # torch.compile autotune when --enable-torch-compile is also on).
    cuda_graph_warmup_iters: int = 2

    # P1 (LOSSY): swap the Wan VAE *decode* for the LightX2V LightTAE (TAEHV)
    # tiny decoder. Large decode speedup, FVD ~24.8 -> ~45.4 (paper Table 5).
    # Encode stays on the Wan VAE unless use_light_vae_encoder is also set.
    # Default off; the loader emits a one-time quality warning when enabled.
    use_light_tae: bool = False
    # Path to the LightTAE checkpoint (lighttaew2_1.pth). None -> resolve from
    # the model dir / a sibling of the DiT checkpoint.
    light_tae_path: str | None = None

    # P2 (LOSSY): swap the Wan VAE *encode* for the 75%-pruned LightVAE encoder
    # (first-frame image + HD-map). Encode speedup, same FVD tradeoff family.
    use_light_vae_encoder: bool = False
    light_vae_path: str | None = None

    # P4a (LOSSY, native, sm_120): run the DiT through the native FP8
    # optimized_dit_forward (FP8 tensor-core GEMMs + FP8 attention + AdaLN).
    # Biggest single speedup; requires the built native ext (else falls back to
    # the eager/bf16 DiT, or raises when SGLANG_OMNIDREAMS_FP8_DIT is set).
    # Mutually exclusive with enable_cuda_graph (the native op owns its own graph).
    use_fp8_dit: bool = False
    # FP8 attention backend: "auto" | "cudnn" | "sage3" | "sage3_fp8" | "sparge".
    fp8_dit_attention_backend: str = "auto"
    # Block-sparse top-k ratio for the "sparge" attention backend (0, 1].
    fp8_dit_sparge_topk: float | None = None

    # P4b (LOSSY, native, sm_120): native FP8 LightVAE encoder. Requires P2
    # (use_light_vae_encoder) + a calibrated state (light_vae_fp8_state_path or
    # SGLANG_OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH). Falls back to the PyTorch
    # LightVAE encoder when the native ext / state is unavailable.
    use_light_vae_fp8: bool = False
    light_vae_fp8_state_path: str | None = None

    def denoising_sigmas(self) -> list[float]:
        return warp_flow_match_sigmas(
            self.denoising_timesteps,
            self.flow_shift if self.flow_shift is not None else 5.0,
            self.sigma_min,
        )
