"""External/black-box exposure-field adapters for UEFB-G.

The adapters convert an enhanced RGB output into a defensible exposure-field
proxy by measuring the luminance gain that the method applied relative to the
low-light input:

    E_hat = log(Y_pred + eps) - log(Y_low + eps)

When a paired reference image is available, the oracle diagnostic target is:

    E_gt = log(Y_high + eps) - log(Y_low + eps)

This does not claim that a black-box method internally predicts an exposure
field. It only asks whether the method-induced luminance transformation is
consistent with a physically meaningful exposure-field proxy.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


@dataclass(frozen=True)
class ExternalAdapterSpec:
    name: str
    smoothing_radius: float = 0.0
    eps: float = 1e-3


def canonical_key(path: Path) -> str:
    stem = path.stem
    for suffix in ('_kindle_v2', '_Zero_DCE++', '_zero_dce_pp'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    for prefix in ('normal', 'low', 'high'):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem


def image_files(path: Path) -> list[Path]:
    return sorted([p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def index_by_key(path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in image_files(path):
        out[canonical_key(p)] = p
    return out


def load_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0


def crop_to_common(*arrays: np.ndarray) -> list[np.ndarray]:
    h = min(a.shape[0] for a in arrays)
    w = min(a.shape[1] for a in arrays)
    return [a[:h, :w, ...] for a in arrays]


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] < 3:
        raise ValueError(f'expected RGB array, got shape={arr.shape}')
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def smooth_field(field: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return np.asarray(field, dtype=np.float32)
    # Dependency-light separable box low-pass. This avoids scipy/cv2 and works
    # on float fields; `radius` is interpreted as a window half-size in pixels.
    arr = np.asarray(field, dtype=np.float32)
    r = max(1, int(round(float(radius))))

    def blur_axis(x: np.ndarray, axis: int) -> np.ndarray:
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (r, r)
        padded = np.pad(x, pad_width, mode='edge')
        padded = np.moveaxis(padded, axis, 0)
        csum = np.cumsum(padded, axis=0, dtype=np.float64)
        csum = np.concatenate([np.zeros_like(csum[:1]), csum], axis=0)
        win = csum[2 * r + 1:] - csum[:-(2 * r + 1)]
        out = win / float(2 * r + 1)
        return np.moveaxis(out, 0, axis).astype(np.float32)

    return blur_axis(blur_axis(arr, 0), 1)


def log_luminance_ratio(numerator_rgb: np.ndarray, denominator_rgb: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    y_num = np.clip(rgb_to_luma(numerator_rgb), 0.0, 1.0)
    y_den = np.clip(rgb_to_luma(denominator_rgb), 0.0, 1.0)
    return (np.log(y_num + eps) - np.log(y_den + eps)).astype(np.float32)


def compute_adapter_fields(
    low_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    high_rgb: np.ndarray,
    smoothing_radius: float = 0.0,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(E_gt_proxy, E_pred_proxy)` for a black-box enhancement output."""
    low, pred, high = crop_to_common(low_rgb, pred_rgb, high_rgb)
    e_gt = log_luminance_ratio(high, low, eps=eps)
    e_pred = log_luminance_ratio(pred, low, eps=eps)
    if smoothing_radius > 0:
        e_gt = smooth_field(e_gt, smoothing_radius)
        e_pred = smooth_field(e_pred, smoothing_radius)
    return e_gt.astype(np.float32), e_pred.astype(np.float32)


def compute_gauge_metrics(e_pred: np.ndarray, e_gt: np.ndarray) -> dict[str, float | str]:
    ep = np.asarray(e_pred, dtype=np.float64)
    eg = np.asarray(e_gt, dtype=np.float64)
    if ep.shape != eg.shape:
        raise ValueError(f'shape mismatch: {ep.shape} vs {eg.shape}')
    mu_p = float(ep.mean())
    mu_g = float(eg.mean())
    sp = ep - mu_p
    sg = eg - mu_g
    denom = float(np.linalg.norm(sp.ravel()) * np.linalg.norm(sg.ravel()))
    if denom < 1e-12:
        s_corr: float | str = 'N/A_constant_shape'
    else:
        s_corr = float(np.dot(sp.ravel(), sg.ravel()) / (denom + 1e-12))
    return {
        'E_MAE': float(np.mean(np.abs(ep - eg))),
        'S_MAE': float(np.mean(np.abs(sp - sg))),
        'Gauge_MAE': float(abs(mu_p - mu_g)),
        'S_corr': s_corr,
        'mu_pred_proxy': mu_p,
        'mu_gt_proxy': mu_g,
        'pred_proxy_std': float(ep.std()),
        'gt_proxy_std': float(eg.std()),
    }


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, str) and x.strip() in {'', 'N/A', 'NA', 'nan', 'None', 'N/A_constant_shape'}:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _mean(vals: Iterable[float]) -> float | None:
    xs = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return float(np.mean(xs)) if xs else None


def _std(vals: Iterable[float]) -> float | None:
    xs = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    return float(np.std(xs, ddof=1))


def evaluate_adapter_triplet(
    method: str,
    dataset: str,
    key: str,
    pred_path: Path,
    low_path: Path,
    high_path: Path,
    adapter: ExternalAdapterSpec,
    output_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    low = load_rgb01(low_path)
    pred = load_rgb01(pred_path)
    high = load_rgb01(high_path)
    e_gt, e_pred = compute_adapter_fields(
        low,
        pred,
        high,
        smoothing_radius=adapter.smoothing_radius,
        eps=adapter.eps,
    )
    metrics = compute_gauge_metrics(e_pred, e_gt)
    row: dict[str, Any] = {
        'variant_id': f'{method}__{adapter.name}',
        'display': f'{method} ({adapter.name})',
        'role': 'external field-aware adapter',
        'method': method,
        'dataset': dataset,
        'adapter': adapter.name,
        'name': key,
        'key': key,
        'reporting_mode': 'field_aware_adapter',
        'adapter_definition': 'log_luminance_ratio(pred,low) vs log_luminance_ratio(high,low)',
        'smoothing_radius': adapter.smoothing_radius,
        'pred_path': str(pred_path),
        'low_path': str(low_path),
        'high_path': str(high_path),
        'height': int(e_pred.shape[0]),
        'width': int(e_pred.shape[1]),
        **metrics,
    }
    if output_metrics:
        for k, v in output_metrics.items():
            if k not in row:
                row[k] = v
    return row


def summarize_adapter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row['method']), str(row['dataset']), str(row['adapter']))].append(row)
    metrics = ['psnr', 'ssim', 'lee', 'nai', 'identity_drop', 'E_MAE', 'S_MAE', 'Gauge_MAE', 'S_corr', 'mu_pred_proxy', 'mu_gt_proxy', 'pred_proxy_std', 'gt_proxy_std']
    out: list[dict[str, Any]] = []
    for (method, dataset, adapter), rs in sorted(grouped.items()):
        base: dict[str, Any] = {
            'method': method,
            'dataset': dataset,
            'adapter': adapter,
            'n': len(rs),
            'variant_id': rs[0].get('variant_id', f'{method}__{adapter}'),
            'reporting_mode': 'field_aware_adapter',
        }
        for metric in metrics:
            vals = [safe_float(r.get(metric)) for r in rs]
            vals = [v for v in vals if v is not None]
            if vals:
                base[f'{metric}_mean'] = _mean(vals)
                base[f'{metric}_std'] = _std(vals)
            else:
                base[f'{metric}_mean'] = 'N/A'
                base[f'{metric}_std'] = 'N/A'
        out.append(base)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fields})
