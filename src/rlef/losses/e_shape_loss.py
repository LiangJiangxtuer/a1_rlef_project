from __future__ import annotations

import torch
import torch.nn.functional as F

from rlef.utils.tensor_ops import gradient_xy


def _match_spatial(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if target.shape[-2:] == reference.shape[-2:]:
        return target
    return F.interpolate(target, size=reference.shape[-2:], mode='bilinear', align_corners=False)


def _lowpass(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        raise ValueError(f'kernel_size must be odd for centered padding, got {kernel_size}')
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=0)


def _centered_corr_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av = a.flatten(1)
    bv = b.flatten(1)
    av = av - av.mean(dim=1, keepdim=True)
    bv = bv - bv.mean(dim=1, keepdim=True)
    corr = (av * bv).sum(dim=1) / (av.norm(dim=1) * bv.norm(dim=1) + 1e-8)
    return (1.0 - corr).mean()


def _grad_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay = gradient_xy(a)
    bx, by = gradient_xy(b)
    return (ax - bx).abs().mean() + (ay - by).abs().mean()


def gauge_invariant_e_shape_loss(
    e_pred: torch.Tensor,
    e_target: torch.Tensor,
    kernel_size: int = 7,
    beta: float = 0.1,
) -> torch.Tensor:
    """Gauge-invariant exposure-shape loss.

    The loss ignores additive exposure-gauge shifts by comparing low-pass,
    zero-mean spatial shape plus a gradient term. It is intended for DGB-RLEF's
    ``S(x)`` branch, where absolute mean/gauge is handled separately.
    """
    if e_pred.ndim != 4 or e_target.ndim != 4:
        raise ValueError('e_pred and e_target must be BCHW tensors')
    e_target = _match_spatial(e_target, e_pred).to(dtype=e_pred.dtype, device=e_pred.device)
    pred_lp = _lowpass(e_pred, kernel_size)
    target_lp = _lowpass(e_target, kernel_size)
    pred_centered = pred_lp - pred_lp.mean(dim=(2, 3), keepdim=True)
    target_centered = target_lp - target_lp.mean(dim=(2, 3), keepdim=True)
    corr = _centered_corr_loss(pred_centered, target_centered)
    grad = _grad_l1(e_pred, e_target)
    return corr + float(beta) * grad


def e_shape_loss(e_pred: torch.Tensor, e_target: torch.Tensor, kernel_size: int = 7, beta: float = 0.1) -> torch.Tensor:
    """Alias kept short for config/loss call sites."""
    return gauge_invariant_e_shape_loss(e_pred, e_target, kernel_size=kernel_size, beta=beta)
