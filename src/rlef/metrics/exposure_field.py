from __future__ import annotations

import torch
import torch.nn.functional as F
from rlef.metrics.full_reference import psnr_torch
from rlef.utils.tensor_ops import luminance


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av = a.flatten(1)
    bv = b.flatten(1)
    av = av - av.mean(dim=1, keepdim=True)
    bv = bv - bv.mean(dim=1, keepdim=True)
    denom = av.norm(dim=1) * bv.norm(dim=1) + 1e-8
    return ((av * bv).sum(dim=1) / denom).mean()


def exposure_field_metrics(e_pred: torch.Tensor, e_gt: torch.Tensor) -> dict[str, float]:
    if e_gt.shape[-2:] != e_pred.shape[-2:]:
        e_gt = F.interpolate(e_gt, size=e_pred.shape[-2:], mode='bilinear', align_corners=False)
    mae = (e_pred - e_gt).abs().mean()
    ep = e_pred - e_pred.mean(dim=(2, 3), keepdim=True)
    eg = e_gt - e_gt.mean(dim=(2, 3), keepdim=True)
    aligned = (ep - eg).abs().mean()
    corr = _corr(e_pred, e_gt)
    return {'E_MAE': float(mae.detach().cpu()), 'E_MAE_aligned': float(aligned.detach().cpu()), 'E_corr': float(corr.detach().cpu())}


def local_exposure_error(pred: torch.Tensor, target: torch.Tensor, patch: int = 16, eps: float = 1e-4) -> torch.Tensor:
    yp = luminance(pred.clamp(0, 1))
    yt = luminance(target.clamp(0, 1))
    if yp.shape[-1] >= patch and yp.shape[-2] >= patch:
        yp = F.avg_pool2d(yp, kernel_size=patch, stride=patch)
        yt = F.avg_pool2d(yt, kernel_size=patch, stride=patch)
    return (torch.log(yp + eps) - torch.log(yt + eps)).abs().mean()


def saturation_rate(x: torch.Tensor, low: float = 1/255, high: float = 254/255) -> dict[str, torch.Tensor]:
    return {'under': (x <= low).float().mean(), 'over': (x >= high).float().mean()}


def noise_amplification_index(pred: torch.Tensor, low: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dark = (luminance(low) < 0.1).float()
    if dark.sum() < 1:
        dark = torch.ones_like(dark)
    def hf(z):
        return z - F.avg_pool2d(z, 5, stride=1, padding=2)
    hp = hf(pred) * dark
    hl = hf(low) * dark
    return hp.std() / (hl.std() + eps)


def identity_drop(pred: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return psnr_torch(pred, high) - psnr_torch(low, high)


def q_ece(q: torch.Tensor, err: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    # q is recoverability; 1-q should match normalized error.
    conf_err = (1.0 - q).detach() if q.requires_grad else (1.0 - q)
    true_err = err.detach() if err.requires_grad else err
    total = torch.zeros((), device=q.device, dtype=q.dtype)
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (conf_err >= lo) & (conf_err < hi if i < n_bins - 1 else conf_err <= hi)
        if mask.any():
            total = total + mask.float().mean() * (conf_err[mask].mean() - true_err[mask].mean()).abs()
    return total


def normalized_abs_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = (pred - target).abs().mean(dim=1, keepdim=True)
    q95 = torch.quantile(err.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return (err / q95).clamp(0, 1)
