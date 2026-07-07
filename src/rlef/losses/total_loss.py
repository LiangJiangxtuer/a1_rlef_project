from __future__ import annotations

import torch
import torch.nn.functional as F
from rlef.metrics.full_reference import ssim_torch
from rlef.metrics.exposure_field import normalized_abs_error
from rlef.losses.e_shape_loss import e_shape_loss as gauge_invariant_e_shape_loss
from rlef.utils.tensor_ops import luminance, gradient_xy, laplacian


def _oracle_e(low: torch.Tensor, high: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.log(luminance(high) + eps) - torch.log(luminance(low) + eps)


def _grad_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay = gradient_xy(a); bx, by = gradient_xy(b)
    return (ax - bx).abs().mean() + (ay - by).abs().mean()


def _lowpass_centered_corr_loss(a: torch.Tensor, b: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size > 1:
        pad = kernel_size // 2
        a = F.avg_pool2d(a, kernel_size=kernel_size, stride=1, padding=pad)
        b = F.avg_pool2d(b, kernel_size=kernel_size, stride=1, padding=pad)
    av = a.flatten(1)
    bv = b.flatten(1)
    av = av - av.mean(dim=1, keepdim=True)
    bv = bv - bv.mean(dim=1, keepdim=True)
    corr = (av * bv).sum(dim=1) / (av.norm(dim=1) * bv.norm(dim=1) + 1e-8)
    return (1.0 - corr).mean()


def _structure_preserve_loss(
    pred: torch.Tensor,
    high: torch.Tensor,
    floor: float = 0.25,
    grad_weight: float = 0.5,
) -> torch.Tensor:
    """Generic contrast-weighted luminance/gradient preservation loss.

    The weighting is derived only from the target image structure, not from
    dataset labels, so it can preserve difficult real-image detail without
    using `rec_by_dataset` or teacher distillation.
    """
    y_pred = luminance(pred)
    y_high = luminance(high)
    hx, hy = gradient_xy(y_high)
    contrast = (hx.abs() + hy.abs()).detach()
    q = torch.quantile(contrast.flatten(1), 0.90, dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    floor = min(max(float(floor), 0.0), 1.0)
    weight = floor + (1.0 - floor) * (contrast / q).clamp(0, 1)
    lum_loss = (weight * (y_pred - y_high).abs()).mean()
    px, py = gradient_xy(y_pred)
    grad_loss = (weight * (px - hx).abs()).mean() + (weight * (py - hy).abs()).mean()
    return lum_loss + float(grad_weight) * grad_loss


def _image_stat_gate_prior_target(
    low: torch.Tensor,
    base: float = 0.35,
    dark_gain: float = 0.30,
    contrast_gain: float = 0.05,
    min_value: float = 0.25,
    max_value: float = 0.75,
) -> torch.Tensor:
    """Deployable gate prior from low-image statistics only.

    Very dark inputs receive a higher restoration-route target; contrast adds a
    small boost. No dataset labels or teacher predictions are used.
    """
    y = luminance(low.clamp(0, 1))
    dark_ratio = (y < 0.10).float().mean(dim=(2, 3), keepdim=True)
    grad_mean = gradient_xy(y)[0].abs().mean(dim=(2, 3), keepdim=True) + gradient_xy(y)[1].abs().mean(dim=(2, 3), keepdim=True)
    target = float(base) + float(dark_gain) * dark_ratio + float(contrast_gain) * grad_mean.clamp(0, 1)
    return target.clamp(float(min_value), float(max_value))


def _dataset_names(batch: dict, n: int) -> list[str]:
    names = batch.get('dataset')
    if names is None:
        return [''] * n
    if isinstance(names, str):
        return [names] * n
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    try:
        return [str(x) for x in list(names)]
    except TypeError:
        return [str(names)] * n


def _canonical_domain_name(name: str) -> str:
    text = str(name).lower()
    if 'synthetic' in text:
        return 'synthetic'
    if 'real' in text:
        return 'real'
    if 'uefb' in text:
        return 'uefb'
    return text


def _rgb_distill_loss(pred: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, teacher) + 0.2 * (1.0 - ssim_torch(pred, teacher))


def compute_total_loss(out: dict, batch: dict, weights: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    low, high = batch['low'], batch['high']
    pred = out['I_hat']
    total = pred.new_zeros(())
    scalars: dict[str, torch.Tensor] = {}
    rec_by_dataset = weights.get('rec_by_dataset') or {}
    if isinstance(rec_by_dataset, dict) and rec_by_dataset:
        names = _dataset_names(batch, pred.shape[0])
        weighted_rec = pred.new_zeros(())
        for dataset_name, dataset_weight in rec_by_dataset.items():
            scalar_key = f'rec_{dataset_name}'
            scalars[scalar_key] = pred.new_zeros(())
            dataset_weight = float(dataset_weight)
            if dataset_weight == 0:
                continue
            mask = torch.tensor([name == dataset_name for name in names], device=pred.device, dtype=torch.bool)
            if not mask.any():
                continue
            domain_rec = _rgb_distill_loss(pred[mask], high[mask])
            domain_weighted = dataset_weight * domain_rec
            weighted_rec = weighted_rec + domain_weighted
            scalars[scalar_key] = domain_weighted.detach()
        if weights.get('rec', 0) > 0:
            fallback = _rgb_distill_loss(pred, high)
            weighted_rec = weighted_rec + float(weights['rec']) * fallback
        total = total + weighted_rec
        scalars['rec'] = weighted_rec.detach()
    elif weights.get('rec', 0) > 0:
        l1 = F.l1_loss(pred, high)
        ssim_loss = 1.0 - ssim_torch(pred, high)
        rec = l1 + 0.2 * ssim_loss
        total = total + float(weights['rec']) * rec
        scalars['rec'] = rec.detach()
    if weights.get('structure_preserve', 0) > 0:
        structure_preserve = _structure_preserve_loss(
            pred,
            high,
            floor=float(weights.get('structure_preserve_floor', 0.25)),
            grad_weight=float(weights.get('structure_preserve_grad', 0.5)),
        )
        total = total + float(weights['structure_preserve']) * structure_preserve
        scalars['structure_preserve'] = structure_preserve.detach()
    if (weights.get('distill', 0) > 0 or weights.get('distill_by_dataset')) and batch.get('teacher') is not None:
        teacher = batch['teacher']
        if teacher.shape[-2:] != pred.shape[-2:]:
            teacher = F.interpolate(teacher, size=pred.shape[-2:], mode='bilinear', align_corners=False)
        teacher = teacher.clamp(0, 1)
        distill_by_dataset = weights.get('distill_by_dataset') or {}
        if isinstance(distill_by_dataset, dict) and distill_by_dataset:
            names = _dataset_names(batch, pred.shape[0])
            weighted = pred.new_zeros(())
            for dataset_name, dataset_weight in distill_by_dataset.items():
                scalar_key = f'distill_{dataset_name}'
                scalars[scalar_key] = pred.new_zeros(())
                dataset_weight = float(dataset_weight)
                if dataset_weight == 0:
                    continue
                mask = torch.tensor([name == dataset_name for name in names], device=pred.device, dtype=torch.bool)
                if not mask.any():
                    continue
                domain_loss = _rgb_distill_loss(pred[mask], teacher[mask])
                domain_weighted = dataset_weight * domain_loss
                weighted = weighted + domain_weighted
                scalars[scalar_key] = domain_weighted.detach()
            if weights.get('distill', 0) > 0:
                fallback = _rgb_distill_loss(pred, teacher)
                weighted = weighted + float(weights['distill']) * fallback
            total = total + weighted
            scalars['distill'] = weighted.detach()
        else:
            distill = _rgb_distill_loss(pred, teacher)
            total = total + float(weights['distill']) * distill
            scalars['distill'] = distill.detach()
    anchor_by_dataset = weights.get('domain_head_anchor_by_dataset') or {}
    if isinstance(anchor_by_dataset, dict) and anchor_by_dataset and out.get('domain_head_bias_l2') is not None:
        names = [_canonical_domain_name(x) for x in _dataset_names(batch, pred.shape[0])]
        head_l2 = out['domain_head_bias_l2'].reshape(-1)
        weighted_anchor = pred.new_zeros(())
        for dataset_name, dataset_weight in anchor_by_dataset.items():
            domain = _canonical_domain_name(dataset_name)
            scalar_key = f'domain_head_anchor_{domain}'
            scalars[scalar_key] = pred.new_zeros(())
            dataset_weight = float(dataset_weight)
            if dataset_weight == 0:
                continue
            mask = torch.tensor([name == domain for name in names], device=pred.device, dtype=torch.bool)
            if not mask.any():
                continue
            domain_anchor = dataset_weight * head_l2[mask].mean()
            weighted_anchor = weighted_anchor + domain_anchor
            scalars[scalar_key] = domain_anchor.detach()
        total = total + weighted_anchor
        scalars['domain_head_anchor'] = weighted_anchor.detach()
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
    if weights.get('e_shape', 0) > 0:
        e_shape = gauge_invariant_e_shape_loss(
            out['E'],
            e_target,
            kernel_size=int(weights.get('e_shape_kernel', 7)),
            beta=float(weights.get('e_shape_beta', 0.1)),
        )
        total = total + float(weights['e_shape']) * e_shape
        scalars['e_shape'] = e_shape.detach()
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
    if weights.get('gate_prior', 0) > 0 and 'A' in out:
        gate_prior_target = _image_stat_gate_prior_target(
            low,
            base=float(weights.get('gate_prior_base', 0.35)),
            dark_gain=float(weights.get('gate_prior_dark_gain', 0.30)),
            contrast_gain=float(weights.get('gate_prior_contrast_gain', 0.05)),
            min_value=float(weights.get('gate_prior_min', 0.25)),
            max_value=float(weights.get('gate_prior_max', 0.75)),
        ).detach()
        if gate_prior_target.shape[-2:] != out['A'].shape[-2:]:
            gate_prior_target = gate_prior_target.expand(-1, -1, out['A'].shape[-2], out['A'].shape[-1])
        gate_prior = F.l1_loss(out['A'], gate_prior_target)
        total = total + float(weights['gate_prior']) * gate_prior
        scalars['gate_prior'] = gate_prior.detach()
        scalars['gate_prior_target'] = gate_prior_target.mean().detach()
    if weights.get('gate_identity', 0) > 0 and 'A' in out:
        tau = float(weights.get('gate_identity_tau', 0.05))
        identity_weight = torch.exp(-(high - low).abs().mean(dim=1, keepdim=True) / tau).detach()
        gate_identity = (identity_weight * out['A']).mean()
        total = total + float(weights['gate_identity']) * gate_identity
        scalars['gate_identity'] = gate_identity.detach()
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
