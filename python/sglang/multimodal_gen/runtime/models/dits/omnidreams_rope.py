# SPDX-License-Identifier: Apache-2.0
"""3D rotary position embedding for OmniDreams (NeoX, T:H:W = 44:42:42 split).

The per-axis frequency construction reuses the shared multi-axis builder
:class:`sglang.multimodal_gen.runtime.layers.rotary_embedding.mrope.NDRotaryEmbedding`:
OmniDreams supplies the per-axis dims (:func:`rope_dims`) and the per-axis NTK
extrapolation ratios (H/W = 3.0, T = 1.0). NDRotaryEmbedding builds the per-axis
cos/sin and column-concatenates them (``[cos_t|cos_h|cos_w]`` / ``[sin_t|sin_h|sin_w]``),
which matches the FlashDreams (T, H, W) layout.

``shift_t(ar_idx)`` builds the ``(t + ar_idx*len_t, h, w)`` position grid (the
autoregressive time offset) and returns the ``[L, D]`` cos|sin cache (first D/2 cos,
second D/2 sin) consumed by :func:`apply_rope_freqs`. Keys are rotated *before*
being written to the KV-cache (standard RoPE, not the cache-relative variant).
"""

from __future__ import annotations

import torch
from torch import Tensor

from sglang.multimodal_gen.runtime.layers.rotary_embedding.mrope import (
    NDRotaryEmbedding,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding.utils import (
    _apply_rotary_emb,
)


def rope_dims(head_dim: int) -> tuple[int, int, int]:
    """Return the (T, H, W) split of a head dim for 3D RoPE.

    For head_dim=128 this yields (44, 42, 42): each spatial axis takes
    ``head_dim // 6 * 2`` and time takes the remainder.
    """
    dim_h = dim_w = head_dim // 6 * 2
    dim_t = head_dim - dim_h - dim_w
    return dim_t, dim_h, dim_w


def _compute_freqs(
    dim: int,
    extrapolation_ratio: float = 1.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Base RoPE frequencies for one axis with NTK extrapolation (shape [dim//2]).

    Kept for the native FP8 path, which consumes the raw-angle ``[L,1,1,head_dim]``
    tensor (see :meth:`RotaryPositionEmbedding3D.shift_t_freqs`). The eager path
    instead uses :class:`NDRotaryEmbedding` via :meth:`shift_t`.
    """
    dim_range = (
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim
    )
    ntk_factor = extrapolation_ratio ** (dim / (dim - 2))
    theta = 10000.0 * ntk_factor
    return 1.0 / (theta**dim_range)


class RotaryPositionEmbedding3D:
    """Standard 3D NeoX RoPE with unbounded autoregressive time positions.

    ``shift_t`` returns a ``[L, D]`` cos|sin cache (first D/2 = cos, second D/2 =
    sin, memory layout (T, H, W)) suitable for :func:`apply_rope_freqs`.
    """

    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.head_dim = head_dim
        self.len_h = len_h
        self.len_w = len_w
        self.len_t = len_t
        self.device = device

        dim_t, dim_h, dim_w = rope_dims(head_dim)
        # Per-axis NTK extrapolation: theta_rescale_factor = ratio**(dim/(dim-2)).
        self._rope = NDRotaryEmbedding(
            rope_dim_list=[dim_t, dim_h, dim_w],
            rope_theta=10000.0,
            theta_rescale_factor=[
                t_extrapolation_ratio,
                h_extrapolation_ratio,
                w_extrapolation_ratio,
            ],
        )
        # Raw per-axis frequencies for the native FP8 path's [L,1,1,head_dim]
        # angle tensor (shift_t_freqs). The eager path uses self._rope above.
        self._raw_freqs_t = _compute_freqs(dim_t, t_extrapolation_ratio, device)
        self._raw_freqs_h = _compute_freqs(dim_h, h_extrapolation_ratio, device)
        self._raw_freqs_w = _compute_freqs(dim_w, w_extrapolation_ratio, device)

    def _positions(self, autoregressive_index: int) -> Tensor:
        """``[L, 3]`` (t, h, w) integer coordinates in (t h w) flatten order."""
        offset = autoregressive_index * self.len_t
        t = torch.arange(self.len_t, device=self.device) + offset
        h = torch.arange(self.len_h, device=self.device)
        w = torch.arange(self.len_w, device=self.device)
        tt, hh, ww = torch.meshgrid(t, h, w, indexing="ij")
        return torch.stack([tt.reshape(-1), hh.reshape(-1), ww.reshape(-1)], dim=-1)

    def shift_t(self, autoregressive_index: int = 0) -> Tensor:
        """``[L, D]`` cos|sin cache for AR chunk ``autoregressive_index``.

        The chunk's absolute time positions are offset by ``autoregressive_index *
        len_t`` so cached K and current Q keep the correct relative rotation.
        """
        cos, sin = self._rope.forward(self._positions(autoregressive_index))
        return torch.cat([cos, sin], dim=-1)

    def shift_t_freqs(self, autoregressive_index: int = 0) -> Tensor:
        """``[L, 1, 1, head_dim]`` raw-angle tensor for the native FP8 DiT path.

        The vendored native kernel (``_make_cosmos_rope_cache``) expects the raw
        RoPE *angles* (position x frequency), not the cos|sin cache that
        :meth:`shift_t` returns for the eager path; it takes ``cos``/``sin`` of
        these angles itself. Layout is NeoX non-interleaved: per-axis angle blocks
        ``[t, h, w]`` repeated twice along the last dim, in ``(t h w)`` flatten
        order (matching :meth:`_positions`).
        """
        pos = self._positions(autoregressive_index).to(torch.float32)  # [L, 3]
        ft = torch.outer(pos[:, 0], self._raw_freqs_t)  # [L, dim_t//2]
        fh = torch.outer(pos[:, 1], self._raw_freqs_h)  # [L, dim_h//2]
        fw = torch.outer(pos[:, 2], self._raw_freqs_w)  # [L, dim_w//2]
        full = torch.cat([ft, fh, fw, ft, fh, fw], dim=-1)  # [L, head_dim]
        return full.reshape(full.shape[0], 1, 1, full.shape[-1])


def apply_rope_freqs(x: Tensor, cos_sin: Tensor) -> Tensor:
    """Apply NeoX 3D RoPE to ``x`` from a precomputed cos|sin cache.

    Delegates the rotation to the shared backend
    :func:`...rotary_embedding.utils._apply_rotary_emb` (FlashInfer on CUDA, pure
    torch otherwise), matching the convention of the rest of the diffusion stack.

    Args:
        x: ``[B, S, H, D]`` query or key.
        cos_sin: ``[S, D]`` cache from :meth:`RotaryPositionEmbedding3D.shift_t`
            (first D/2 columns cos, second D/2 sin).
    Returns:
        Rotated tensor of shape ``[B, S, H, D]``.
    """
    B, S, H, D = x.shape
    half = D // 2
    cos = cos_sin[:, :half].to(x.dtype)
    sin = cos_sin[:, half:].to(x.dtype)
    # cos/sin are shared across batch and heads (_apply_rotary_emb broadcasts heads).
    cos = cos.unsqueeze(0).expand(B, -1, -1).reshape(B * S, half)
    sin = sin.unsqueeze(0).expand(B, -1, -1).reshape(B * S, half)
    # NeoX (non-interleaved) rotation: the rotated pair is (d, d + D/2).
    return _apply_rotary_emb(
        x.reshape(B * S, H, D), cos, sin, is_neox_style=True, interleaved=False
    ).reshape(B, S, H, D)
