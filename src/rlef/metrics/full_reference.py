from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    return 10.0 * torch.log10(torch.tensor(1.0, device=pred.device, dtype=pred.dtype) / (mse + eps))


def _avg_pool(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)


def ssim_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = pred.clamp(0, 1)
    y = target.clamp(0, 1)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mux, muy = _avg_pool(x), _avg_pool(y)
    sigx = _avg_pool(x * x) - mux * mux
    sigy = _avg_pool(y * y) - muy * muy
    sigxy = _avg_pool(x * y) - mux * muy
    ssim = ((2 * mux * muy + c1) * (2 * sigxy + c2)) / ((mux * mux + muy * muy + c1) * (sigx + sigy + c2) + eps)
    return ssim.mean()
