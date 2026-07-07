from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from rlef.utils.tensor_ops import gradient_magnitude, luminance


def image_luminance_stats(low: torch.Tensor) -> torch.Tensor:
    """Return deployable image statistics for gauge calibration.

    Output columns are: mean(Y_l), std(Y_l), dark_ratio, sat_ratio,
    grad_mean, local_contrast.
    """
    if low.ndim != 4:
        raise ValueError(f'low must be a BCHW tensor, got {tuple(low.shape)}')
    y = luminance(low.clamp(0, 1))
    mean = y.mean(dim=(1, 2, 3))
    std = y.std(dim=(1, 2, 3), unbiased=False)
    dark_ratio = (y < 0.10).to(y.dtype).mean(dim=(1, 2, 3))
    sat_ratio = (y > (254.0 / 255.0)).to(y.dtype).mean(dim=(1, 2, 3))
    grad_mean = gradient_magnitude(y).mean(dim=(1, 2, 3))
    local_mean = F.avg_pool2d(y, kernel_size=5, stride=1, padding=2)
    local_contrast = (y - local_mean).abs().mean(dim=(1, 2, 3))
    return torch.stack([mean, std, dark_ratio, sat_ratio, grad_mean, local_contrast], dim=1)


class ImageStatsGaugeHead(nn.Module):
    """Small deployable MLP that maps image statistics to exposure gauge."""

    def __init__(self, in_dim: int = 6, hidden: int = 32, mu_min: float = -1.0, mu_max: float = 2.5):
        super().__init__()
        if mu_max <= mu_min:
            raise ValueError('mu_max must be greater than mu_min')
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, low_or_stats: torch.Tensor) -> torch.Tensor:
        stats = image_luminance_stats(low_or_stats) if low_or_stats.ndim == 4 else low_or_stats
        raw = self.net(stats)
        mu = self.mu_min + (self.mu_max - self.mu_min) * torch.sigmoid(raw)
        return mu.view(-1, 1, 1, 1)
