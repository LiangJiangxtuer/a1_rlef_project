from __future__ import annotations

import torch
import torch.nn.functional as F

from rlef.metrics.exposure_field import (
    exposure_field_metrics,
    identity_drop,
    local_exposure_error,
    noise_amplification_index,
    normalized_abs_error,
    q_ece,
)
from rlef.metrics.full_reference import psnr_torch, ssim_torch


def _to_float(x: torch.Tensor | float | int) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def _match_spatial(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode='bilinear', align_corners=False)


def _centered_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.flatten(1)
    bv = b.flatten(1)
    av = av - av.mean(dim=1, keepdim=True)
    bv = bv - bv.mean(dim=1, keepdim=True)
    corr = (av * bv).sum(dim=1) / (av.norm(dim=1) * bv.norm(dim=1) + 1e-8)
    return _to_float(corr.mean())


def _binary_auc(score: torch.Tensor, target: torch.Tensor) -> float | None:
    score = score.flatten().detach()
    target = (target.flatten().detach() > 0.5)
    pos = score[target]
    neg = score[~target]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    cmp = (pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()
    return _to_float(cmp.mean())


def _route_entropy(a: torch.Tensor) -> float:
    p = a.clamp(1e-6, 1.0 - 1e-6)
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return _to_float(ent.mean())


def _unsafe_overenhance(pred: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> float:
    near_identity = (high - low).abs().mean(dim=1, keepdim=True) < 0.05
    over_delta = (pred - low).abs().mean(dim=1, keepdim=True) > 0.10
    if near_identity.sum() < 1:
        return 0.0
    return _to_float((near_identity & over_delta).float().mean())


def compute_domain_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    low: torch.Tensor,
    *,
    e_pred: torch.Tensor | None = None,
    e_gt: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
    a_gt: torch.Tensor | None = None,
    q: torch.Tensor | None = None,
) -> dict[str, float | None | list[str]]:
    """Compute the Phase-1 DGB metric ledger for one domain/eval split."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    low = low.clamp(0, 1)
    notes: list[str] = []
    metrics: dict[str, float | None | list[str]] = {
        'psnr': _to_float(psnr_torch(pred, target)),
        'ssim': _to_float(ssim_torch(pred, target)),
        'lee': _to_float(local_exposure_error(pred, target)),
        'nai': _to_float(noise_amplification_index(pred, low)),
        'identity_drop': _to_float(identity_drop(pred, low, target)),
        'unsafe_overenhance': _unsafe_overenhance(pred, low, target),
    }
    if e_pred is not None and e_gt is not None:
        e_target = _match_spatial(e_gt, e_pred).to(dtype=e_pred.dtype, device=e_pred.device)
        metrics.update(exposure_field_metrics(e_pred, e_target))
        gauge_pred = e_pred.mean(dim=(2, 3), keepdim=True)
        gauge_gt = e_target.mean(dim=(2, 3), keepdim=True)
        metrics['Gauge_MAE'] = _to_float((gauge_pred - gauge_gt).abs().mean())
        s_pred = e_pred - gauge_pred
        s_gt = e_target - gauge_gt
        metrics['S_corr'] = _centered_corr(s_pred, s_gt)
    else:
        notes.append('missing_e_gt_or_e_pred')
        metrics.update({'E_MAE': None, 'E_MAE_aligned': None, 'E_corr': None, 'Gauge_MAE': None, 'S_corr': None})

    if a is not None:
        a_gate = a.clamp(0, 1)
        metrics['route_entropy'] = _route_entropy(a_gate)
        if a_gt is not None:
            a_target = _match_spatial(a_gt, a_gate).to(dtype=a_gate.dtype, device=a_gate.device)
            auc = _binary_auc(a_gate, a_target)
            if auc is None:
                notes.append('a_auc_requires_positive_and_negative_targets')
            metrics['A_AUC'] = auc
        else:
            notes.append('missing_a_gt_or_a')
            metrics['A_AUC'] = None
    else:
        notes.append('missing_a_gt_or_a')
        metrics['A_AUC'] = None
        metrics['route_entropy'] = None

    if q is not None:
        metrics['Q_ECE'] = _to_float(q_ece(q, normalized_abs_error(pred, target)))
    else:
        notes.append('missing_q')
        metrics['Q_ECE'] = None

    metrics['notes'] = notes
    return metrics
