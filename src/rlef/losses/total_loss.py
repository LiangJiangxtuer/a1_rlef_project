from __future__ import annotations

import torch
import torch.nn.functional as F
from rlef.metrics.full_reference import ssim_torch
from rlef.metrics.exposure_field import normalized_abs_error
from rlef.utils.tensor_ops import luminance, gradient_xy, laplacian


def _oracle_e(low: torch.Tensor, high: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.log(luminance(high) + eps) - torch.log(luminance(low) + eps)


def _grad_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay = gradient_xy(a); bx, by = gradient_xy(b)
    return (ax - bx).abs().mean() + (ay - by).abs().mean()


def compute_total_loss(out: dict, batch: dict, weights: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    low, high = batch['low'], batch['high']
    pred = out['I_hat']
    total = pred.new_zeros(())
    scalars: dict[str, torch.Tensor] = {}
    if weights.get('rec', 0) > 0:
        l1 = F.l1_loss(pred, high)
        ssim_loss = 1.0 - ssim_torch(pred, high)
        rec = l1 + 0.2 * ssim_loss
        total = total + float(weights['rec']) * rec
        scalars['rec'] = rec.detach()
    if weights.get('phys', 0) > 0 and 'I_phys' in out:
        phys = F.l1_loss(out['I_phys'], high)
        total = total + float(weights['phys']) * phys
        scalars['phys'] = phys.detach()
    e_target = batch.get('E_gt')
    if e_target is None:
        e_target = _oracle_e(low, high).detach()
    if e_target.shape[-2:] != out['E'].shape[-2:]:
        e_target = F.interpolate(e_target, size=out['E'].shape[-2:], mode='bilinear', align_corners=False)
    if weights.get('poisson', 0) > 0:
        # Use gradient-domain target as a stable Poisson-equivalent smoke objective.
        poi = _grad_l1(out['E'], e_target)
        total = total + float(weights['poisson']) * poi
        scalars['poisson'] = poi.detach()
    if weights.get('gauge', 0) > 0:
        gauge = (out['E'].mean(dim=(2,3), keepdim=True) - e_target.mean(dim=(2,3), keepdim=True)).abs().mean()
        total = total + float(weights['gauge']) * gauge
        scalars['gauge'] = gauge.detach()
    if weights.get('id', 0) > 0:
        tau = float(weights.get('id_tau', 0.05))
        w = torch.exp(-(high - low).abs().mean(dim=1, keepdim=True) / tau)
        ident = (w * (pred - low).abs().mean(dim=1, keepdim=True)).mean()
        total = total + float(weights['id']) * ident
        scalars['id'] = ident.detach()
    if weights.get('gate', 0) > 0 and 'A' in out:
        a_target = batch.get('A_gt')
        if a_target is None:
            a_target = (e_target.abs() / (torch.quantile(e_target.abs().flatten(1), 0.95, dim=1).view(-1,1,1,1).clamp_min(1e-6))).clamp(0,1).detach()
        if a_target.shape[-2:] != out['A'].shape[-2:]:
            a_target = F.interpolate(a_target, size=out['A'].shape[-2:], mode='bilinear', align_corners=False)
        gate = F.l1_loss(out['A'], a_target)
        total = total + float(weights['gate']) * gate
        scalars['gate'] = gate.detach()
    if weights.get('q', 0) > 0 and 'Q' in out:
        q_target = batch.get('Q_gt')
        if q_target is None:
            q_target = 1.0 - normalized_abs_error(pred.detach(), high)
        if q_target.shape[-2:] != out['Q'].shape[-2:]:
            q_target = F.interpolate(q_target, size=out['Q'].shape[-2:], mode='bilinear', align_corners=False)
        q = F.l1_loss(out['Q'], q_target.clamp(0, 1))
        total = total + float(weights['q']) * q
        scalars['q'] = q.detach()
    if weights.get('wtv', 0) > 0:
        ex, ey = gradient_xy(out['E'])
        gx, gy = gradient_xy(luminance(low))
        wx, wy = torch.exp(-10.0 * gx.abs()), torch.exp(-10.0 * gy.abs())
        wtv = (wx * ex.abs()).mean() + (wy * ey.abs()).mean()
        total = total + float(weights['wtv']) * wtv
        scalars['wtv'] = wtv.detach()
    return total, scalars
