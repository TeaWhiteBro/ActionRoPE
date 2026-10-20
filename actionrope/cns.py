"""Coordinate-Conditioned Noise Scheduling (CNS).

Every backbone samples with a shifted flow-matching schedule sigma = S u / (1 + (S - 1) u), u in [0, 1]. CNS gives each
token its own shift according to its knownness kappa (1 = inside the first-frame footprint, 0 = newly revealed):

    u       = sigma / (S - (S - 1) sigma)                 (invert the base schedule)
    S_tok   = S * (S_known / S) ** (alpha * kappa)
    sigma_t = S_tok u / (1 + (S_tok - 1) u)

Known tokens follow a schedule with the smaller shift S_known, so they reach low noise earlier during sampling;
newly revealed tokens keep the base schedule (sigma_t = sigma for kappa = 0). sigma_t(1) = 1 and sigma_t(0) = 0 for
every kappa. The same function is used to noise the training samples and to integrate the sampler.
"""
from __future__ import annotations

import torch


def cns_sigma(sigma, kappa: torch.Tensor, shift: float, shift_known: float, alpha: float = 1.0) -> torch.Tensor:
    """Per-token noise level for the global noise level ``sigma`` (float or tensor) and knownness ``kappa``."""
    S = float(shift)
    u = sigma / (S - (S - 1.0) * sigma)
    S_tok = S * (float(shift_known) / S) ** (alpha * kappa)
    return S_tok * u / (1.0 + (S_tok - 1.0) * u)
