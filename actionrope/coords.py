"""Known / newly revealed regions in world coordinates (backbone-agnostic).

A latent (or token) grid cell (i, j) of latent frame k looks at the world position (i + dy_k / p, j + dx_k / p), where
(dx_k, dy_k) is the camera offset of that latent frame in pixels and p the number of pixels per grid cell. The cell is
*known* when this position lies inside the footprint of the first frame, [-0.5, h - 0.5] x [-0.5, w - 0.5] (closed),
and *newly revealed* otherwise. All functions here are pure geometry and work on any grid.
"""
from __future__ import annotations

import torch


def footprint_distance(offset_px: torch.Tensor, h: int, w: int, px_per_cell: float) -> torch.Tensor:
    """Distance (in grid cells) of every cell's world position to the first-frame footprint; 0 inside.

    offset_px: [b, f, 2] camera offset (dx, dy) of each latent frame in pixels. Returns float64 [b, f, h, w].
    """
    off = offset_px.to(torch.float64) / px_per_cell
    b, f = off.shape[:2]
    dx, dy = off[..., 0].view(b, f, 1, 1), off[..., 1].view(b, f, 1, 1)
    wi = torch.arange(h, device=off.device, dtype=torch.float64).view(1, 1, h, 1) + dy
    wj = torch.arange(w, device=off.device, dtype=torch.float64).view(1, 1, 1, w) + dx
    dh = torch.clamp(torch.maximum(-0.5 - wi, wi - (h - 0.5)), min=0.0)
    dw = torch.clamp(torch.maximum(-0.5 - wj, wj - (w - 0.5)), min=0.0)
    return torch.sqrt(dh * dh + dw * dw)


def known_mask(offset_px: torch.Tensor, h: int, w: int, px_per_cell: float) -> torch.Tensor:
    """Boolean [b, f, h, w]: True where the cell's world position lies inside the first-frame footprint."""
    return footprint_distance(offset_px, h, w, px_per_cell) <= 0


def knownness(offset_px: torch.Tensor, h: int, w: int, px_per_cell: float) -> torch.Tensor:
    """kappa as float32 [b, 1, f, h, w]: 1 for known cells, 0 for newly revealed cells."""
    return known_mask(offset_px, h, w, px_per_cell).to(torch.float32).unsqueeze(1)


def loss_weight_map(offset_px: torch.Tensor, h: int, w: int, px_per_cell: float, new_weight: float = 2.0) -> torch.Tensor:
    """Per-cell loss weight, float32 [b, 1, f, h, w]: 1 for known cells, ``new_weight`` for newly revealed cells."""
    known = known_mask(offset_px, h, w, px_per_cell)
    return torch.where(known, 1.0, float(new_weight)).to(torch.float32).unsqueeze(1)
