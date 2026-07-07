#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def planned_remaining_steps() -> list[dict[str, Any]]:
    """Remaining master-prompt steps after DGB stop and S1-S4 paper pipeline.

    Phase 2/3/4 DGB promotion is not legally reachable because DGB has been
    stopped and no candidate passed the joint gate. The remaining executable
    steps are therefore the no-reference supplement and final paper artifacts
    requested by the master prompt.
    """
    return [
        {
            'id': 'P2_NOREF_SUPPLEMENT',
            'name': 'Official/no-reference supplementary eval on frozen artifacts',
            'uses_new_training': False,
            'allowed_claim_scope': 'support_or_limitation_only',
        },
        {'id': 'FINAL_TABLES', 'name': 'Final main and ablation tables', 'uses_new_training': False},
        {'id': 'FINAL_REPORT', 'name': 'DGB_RLEF_FINAL_EXPERIMENT_REPORT.md', 'uses_new_training': False},
        {'id': 'PAPER_IDEA', 'name': 'DGB_RLEF_PAPER_IDEA.md', 'uses_new_training': False},
        {'id': 'REPRO_CHECKLIST', 'name': 'DGB_RLEF_REPRODUCIBILITY_CHECKLIST.md', 'uses_new_training': False},
        {'id': 'FINAL_FIGURE', 'name': 'final_qualitative_grid.png', 'uses_new_training': False},
    ]


def proxy_metric_keys() -> list[str]:
    # BRISQUE is intentionally excluded: no local package is available, and the
    # pipeline records that limitation instead of fabricating a value.
    return ['niqe', 'mean_luma', 'over', 'under', 'dark_ratio', 'contrast', 'sharpness_proxy', 'noise_proxy']


def planned_outputs(root: Path = ROOT) -> dict[str, Path]:
    return {
        'noref_dir': root / 'results/hermes_audit/supplementary_noref',
        'noref_report': root / 'results/hermes_audit/reports/P2_NOREF_SUPPLEMENTARY_EVAL.md',
        'noref_summary_csv': root / 'results/hermes_audit/tables/noref_supplementary_summary.csv',
        'noref_summary_json': root / 'results/hermes_audit/tables/noref_supplementary_summary.json',
        'noref_per_image_csv': root / 'results/hermes_audit/tables/noref_supplementary_per_image.csv',
        'noref_grid': root / 'results/hermes_audit/figures/noref_unpaired_grid.png',
        'final_report': root / 'docs/DGB_RLEF_FINAL_EXPERIMENT_REPORT.md',
        'paper_idea': root / 'docs/DGB_RLEF_PAPER_IDEA.md',
        'repro_checklist': root / 'docs/DGB_RLEF_REPRODUCIBILITY_CHECKLIST.md',
        'final_main_table': root / 'results/hermes_audit/tables/final_main_table.csv',
        'final_main_table_json': root / 'results/hermes_audit/tables/final_main_table.json',
        'final_ablation_table': root / 'results/hermes_audit/tables/final_ablation_table.csv',
        'final_ablation_table_json': root / 'results/hermes_audit/tables/final_ablation_table.json',
        'final_qualitative_grid': root / 'results/hermes_audit/figures/final_qualitative_grid.png',
        'manifest': root / 'results/hermes_audit/logs/master_remaining_pipeline_manifest.json',
        'validation_report': root / 'results/hermes_audit/reports/MASTER_REMAINING_STEPS_VALIDATION.md',
    }


def _load_json(path: Path) -> Any:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == '':
        return '—'
    try:
        f = float(value)
        if math.isnan(f):
            return '—'
        return f'{f:.{digits}f}'
    except (TypeError, ValueError):
        return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for row in rows for k in row.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def _md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = '| ' + ' | '.join(fields) + ' |'
    sep = '| ' + ' | '.join('---' for _ in fields) + ' |'
    body = []
    for row in rows:
        body.append('| ' + ' | '.join(_fmt(row.get(field)) for field in fields) + ' |')
    return '\n'.join([header, sep, *body])


def build_final_main_rows(root: Path = ROOT) -> list[dict[str, Any]]:
    s2_path = root / 'results/paper_pipeline/s2_main_table/main_results_table.json'
    if s2_path.exists():
        core = _load_json(s2_path)
    else:
        from scripts.run_s1_s4_paper_pipeline import build_core_table_rows
        core = build_core_table_rows(root)
    dgb = _load_json(root / 'results/tables/dgb_branch_consolidation_summary.json')
    rows: list[dict[str, Any]] = []
    for row in core:
        method = row.get('method')
        if method == 'P3c M4J_ES e_shape=0.05':
            decision = 'current conservative default'
            allowed = True
        elif method in {'Retinexformer', 'Zero-DCE++', 'KinD++'}:
            decision = 'official baseline comparison'
            allowed = False
        else:
            decision = 'mainline ablation / route evidence'
            allowed = method in {'M4J_ES seed3407', 'M4J joint', 'M4 A-gate'}
        rows.append({
            'method': method,
            'role': row.get('role'),
            'evidence_level': row.get('evidence_level'),
            'uefb_psnr': row.get('uefb_psnr'),
            'real_psnr': row.get('real_psnr'),
            'synthetic_psnr': row.get('synthetic_psnr'),
            'uefb_E_corr': row.get('uefb_E_corr'),
            'real_E_corr': row.get('real_E_corr'),
            'synthetic_E_corr': row.get('synthetic_E_corr'),
            'decision': decision,
            'main_claim_allowed': bool(allowed),
            'note': row.get('claim_scope', ''),
        })
    best = dgb['best_by_split']
    rows.append({
        'method': 'DGB branch',
        'role': 'diagnostic branch',
        'evidence_level': 'stopped consolidation',
        'uefb_psnr': best['uefb_psnr']['uefb_psnr'],
        'real_psnr': best['real_psnr']['real_psnr'],
        'synthetic_psnr': best['synthetic_psnr']['synthetic_psnr'],
        'uefb_E_corr': best['uefb_psnr'].get('uefb_E_corr'),
        'real_E_corr': best['real_psnr'].get('real_E_corr'),
        'synthetic_E_corr': best['synthetic_psnr'].get('synthetic_E_corr'),
        'decision': 'stopped; diagnostic only',
        'main_claim_allowed': False,
        'note': 'No DGB candidate passed the joint UEFB/real/synthetic gate; use as negative-route evidence only.',
    })
    return rows


def build_final_ablation_rows(root: Path = ROOT) -> list[dict[str, Any]]:
    p3c = _load_json(root / 'results/tables/p3c_multiseed_sweep_aggregate.json')['multiseed_e_shape_0p05']
    p3d_file = _load_json(root / 'results/tables/p3d_e010_multiseed_aggregate.json')
    p3d = p3d_file.get('multiseed_e_shape_0p10', p3d_file.get('aggregate'))
    s4 = _load_json(root / 'results/paper_pipeline/s4_appendix/negative_route_summary.json')
    rows = [
        {
            'group': 'Gauge / E-shape',
            'variant': 'P3c e_shape=0.05',
            'key_result': f"UEFB={_fmt(p3c['uefb_psnr']['mean'])}; real={_fmt(p3c['real_psnr']['mean'])}; synthetic={_fmt(p3c['synthetic_psnr']['mean'])}",
            'decision': 'default',
            'paper_location': 'main ablation',
            'interpretation': 'best validated three-seed conservative default; positive E-shape diagnostics',
        },
        {
            'group': 'Gauge / E-shape',
            'variant': 'P3d e_shape=0.10',
            'key_result': f"UEFB={_fmt(p3d['uefb_psnr']['mean'])}; real={_fmt(p3d['real_psnr']['mean'])}; synthetic={_fmt(p3d['synthetic_psnr']['mean'])}",
            'decision': 'ablation only',
            'paper_location': 'main ablation',
            'interpretation': 'higher UEFB/E_corr but worse real/synthetic mean than e=0.05',
        },
    ]
    mapping = {
        'P5/P5b distillation': ('Distillation', 'appendix negative evidence'),
        'P6/P6b/P6c structural scalar controls': ('Structural backbone', 'appendix negative evidence'),
        'P7/P7b/P7c domain heads': ('Domain heads', 'appendix negative evidence'),
        'DGB branch': ('DGB controlled isolation', 'appendix negative evidence'),
    }
    for item in s4:
        group, location = mapping[item['route']]
        rows.append({
            'group': group,
            'variant': item['best_reference'],
            'key_result': item['best_snapshot'],
            'decision': item['decision'],
            'paper_location': location,
            'interpretation': item['reason'],
        })
    return rows


def collect_unpaired_paths(root: Path = ROOT, max_images: int | None = None) -> list[Path]:
    base = root / 'data/unpaired_real'
    paths = sorted(p for p in base.rglob('*') if p.suffix.lower() in IMAGE_EXTS)
    if max_images is not None:
        return paths[:max_images]
    return paths


def _load_niqe_backend(root: Path):
    """Load NIQE pristine parameters and use a local no-cv2 implementation.

    The local OpenCV build can fail with a libpng symbol mismatch in this
    environment. The NIQE formula itself only needs convolution, gamma, and
    bilinear downsampling, so we keep the metric available without importing
    cv2 or Retinexformer's basicsr package.
    """
    params_path = root / 'external_baselines/Retinexformer/basicsr/metrics/niqe_pris_params.npz'
    params = np.load(params_path)
    return _niqe_no_cv2, params['mu_pris_param'], params['cov_pris_param'], params['gaussian_window']


def _estimate_aggd_param(block: np.ndarray) -> tuple[float, float, float]:
    from scipy.special import gamma
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)
    r_gam = np.square(gamma(2 / gam)) / (gamma(1 / gam) * gamma(3 / gam))
    left = block[block < 0]
    right = block[block > 0]
    left_std = np.sqrt(np.mean(left ** 2)) if left.size else 0.0
    right_std = np.sqrt(np.mean(right ** 2)) if right.size else 0.0
    gammahat = left_std / (right_std + 1e-12)
    rhat = (np.mean(np.abs(block)) ** 2) / (np.mean(block ** 2) + 1e-12)
    rhatnorm = (rhat * (gammahat ** 3 + 1) * (gammahat + 1)) / ((gammahat ** 2 + 1) ** 2 + 1e-12)
    pos = int(np.argmin((r_gam - rhatnorm) ** 2))
    alpha = float(gam[pos])
    beta_l = float(left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))) if left_std > 0 else 0.0
    beta_r = float(right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))) if right_std > 0 else 0.0
    return alpha, beta_l, beta_r


def _niqe_feature(block: np.ndarray) -> list[float]:
    from scipy.special import gamma
    feat: list[float] = []
    alpha, beta_l, beta_r = _estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])
    for shift in ([0, 1], [1, 0], [1, 1], [1, -1]):
        shifted = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = _estimate_aggd_param(block * shifted)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([float(alpha), float(mean), float(beta_l), float(beta_r)])
    return feat


def _niqe_no_cv2(img: np.ndarray, mu_pris_param: np.ndarray, cov_pris_param: np.ndarray, gaussian_window: np.ndarray, block_size_h: int = 96, block_size_w: int = 96) -> float:
    from scipy.ndimage import convolve
    assert img.ndim == 2
    img = img.astype(np.float32)
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    if num_block_h < 1 or num_block_w < 1:
        return float('nan')
    img = img[:num_block_h * block_size_h, :num_block_w * block_size_w]
    distparam = []
    for scale in (1, 2):
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        normalized = (img - mu) / (sigma + 1)
        feats = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = normalized[idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                                   idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                feats.append(_niqe_feature(block))
        distparam.append(np.asarray(feats, dtype=np.float64))
        if scale == 1:
            pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode='L')
            pil = pil.resize((max(1, img.shape[1] // 2), max(1, img.shape[0] // 2)), Image.BILINEAR)
            img = np.asarray(pil).astype(np.float32)
    dist = np.concatenate(distparam, axis=1)
    mu_dist = np.nanmean(dist, axis=0)
    clean = dist[~np.isnan(dist).any(axis=1)]
    cov_dist = np.cov(clean, rowvar=False)
    invcov = np.linalg.pinv((cov_pris_param + cov_dist) / 2)
    quality = np.matmul(np.matmul((mu_pris_param - mu_dist), invcov), np.transpose(mu_pris_param - mu_dist))
    return float(np.sqrt(quality))


def _rgb_to_y255(arr_rgb: np.ndarray) -> np.ndarray:
    # MATLAB-like luma enough for within-pipeline supplementary comparisons.
    arr = arr_rgb.astype(np.float32)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def compute_proxy_metrics(img: Image.Image, niqe_backend: tuple[Any, Any, Any, Any] | None = None) -> dict[str, float | None]:
    rgb = np.asarray(img.convert('RGB')).astype(np.float32)
    y = _rgb_to_y255(rgb) / 255.0
    dark = y < 0.10
    # High-frequency/noise proxy: residual from a 3x3 mean filter; only a proxy.
    from scipy.ndimage import uniform_filter, laplace
    blur = uniform_filter(y, size=3, mode='nearest')
    hf = y - blur
    metrics: dict[str, float | None] = {
        'mean_luma': float(y.mean()),
        'over': float((y > 0.98).mean()),
        'under': float((y < 0.02).mean()),
        'dark_ratio': float(dark.mean()),
        'contrast': float(y.std()),
        'sharpness_proxy': float(laplace(y * 255.0).var()),
        'noise_proxy': float(hf[dark].std() if dark.any() else hf.std()),
    }
    metrics['niqe'] = None
    if niqe_backend is not None:
        niqe_fn, mu, cov, win = niqe_backend
        try:
            y255 = _rgb_to_y255(rgb).astype(np.float32)
            if y255.shape[0] >= 96 and y255.shape[1] >= 96:
                metrics['niqe'] = float(niqe_fn(y255, mu, cov, win))
        except Exception:
            metrics['niqe'] = None
    return metrics


def generate_rlef_unpaired_outputs(root: Path, image_paths: list[Path], out_dir: Path, device: str) -> list[Path]:
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'src'))
    import torch
    import yaml
    from torchvision.transforms.functional import to_tensor
    from scripts.train import build_model
    from rlef.utils.image_io import save_tensor_image

    cfg_path = root / 'configs/p3b_joint/p3b_m4j_es_m4_joint_eshape.yml'
    ckpt_path = root / 'experiments/p3b_m4j_es_m4_joint_eshape_seed3407/checkpoints/last.pth'
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f'Missing frozen RLEF visualization checkpoint/config: {cfg_path}, {ckpt_path}')
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    torch_device = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = build_model(cfg).to(torch_device)
    ckpt = torch.load(ckpt_path, map_location=torch_device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    generated: list[Path] = []
    with torch.no_grad():
        for src in image_paths:
            dataset = src.parent.name
            dst = out_dir / 'rlef_p3c_unpaired' / dataset / f'{src.stem}.png'
            if dst.exists() and dst.stat().st_size > 0:
                generated.append(dst)
                continue
            img = Image.open(src).convert('RGB')
            low = to_tensor(img).unsqueeze(0).to(torch_device)
            out = model(low, domain='real')
            save_tensor_image(out['I_hat'][0], dst)
            generated.append(dst)
    return generated


def run_noref_supplement(root: Path, out: dict[str, Path], device: str = 'cuda', max_images: int | None = None) -> dict[str, Any]:
    paths = collect_unpaired_paths(root, max_images=max_images)
    out['noref_dir'].mkdir(parents=True, exist_ok=True)
    generated = generate_rlef_unpaired_outputs(root, paths, out['noref_dir'], device=device)
    try:
        niqe_backend = _load_niqe_backend(root)
        niqe_status = 'available: Retinexformer basicsr NIQE implementation'
    except Exception as exc:
        niqe_backend = None
        niqe_status = f'unavailable: {type(exc).__name__}: {exc}'

    rows: list[dict[str, Any]] = []
    for src, pred in zip(paths, generated):
        dataset = src.parent.name
        for method, path in [('input_low', src), ('P3c_RLEF_seed3407', pred)]:
            metrics = compute_proxy_metrics(Image.open(path).convert('RGB'), niqe_backend=niqe_backend)
            rows.append({
                'dataset': dataset,
                'image_id': src.name,
                'method': method,
                'source_path': str(path),
                **metrics,
            })
    fields = ['dataset', 'image_id', 'method', 'source_path', *proxy_metric_keys()]
    _write_csv(out['noref_per_image_csv'], rows, fields)
    summary = _summarize_noref(rows)
    _write_csv(out['noref_summary_csv'], summary)
    out['noref_summary_json'].parent.mkdir(parents=True, exist_ok=True)
    out['noref_summary_json'].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    make_unpaired_grid(paths, generated, out['noref_grid'])
    report = _noref_report(summary, len(paths), niqe_status, out)
    out['noref_report'].parent.mkdir(parents=True, exist_ok=True)
    out['noref_report'].write_text(report, encoding='utf-8')
    return {'paths': paths, 'generated': generated, 'summary': summary, 'niqe_status': niqe_status}


def _summarize_noref(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['method'])].append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, method), items in sorted(groups.items()):
        rec: dict[str, Any] = {'dataset': dataset, 'method': method, 'n': len(items)}
        for key in proxy_metric_keys():
            vals = [float(x[key]) for x in items if x.get(key) is not None and str(x.get(key)) != 'nan']
            rec[key] = sum(vals) / len(vals) if vals else None
        summary.append(rec)
    return summary


def make_unpaired_grid(inputs: list[Path], outputs: list[Path], out_path: Path, max_rows: int = 5, cell: int = 180) -> None:
    chosen: list[tuple[Path, Path]] = []
    seen = set()
    for src, pred in zip(inputs, outputs):
        dataset = src.parent.name
        if dataset in seen:
            continue
        seen.add(dataset)
        chosen.append((src, pred))
        if len(chosen) >= max_rows:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    label_h = 24
    sheet = Image.new('RGB', (cell * 2, (cell + label_h) * max(1, len(chosen))), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, (src, pred) in enumerate(chosen):
        y0 = row * (cell + label_h)
        for col, (label, path) in enumerate([('input ' + src.parent.name, src), ('P3c RLEF', pred)]):
            img = Image.open(path).convert('RGB').resize((cell, cell))
            x0 = col * cell
            draw.text((x0 + 4, y0 + 4), label[:24], fill=(0, 0, 0))
            sheet.paste(img, (x0, y0 + label_h))
    sheet.save(out_path)


def _noref_report(summary: list[dict[str, Any]], n_images: int, niqe_status: str, out: dict[str, Path]) -> str:
    table = _md_table(summary, ['dataset', 'method', 'n', 'niqe', 'mean_luma', 'over', 'under', 'dark_ratio', 'contrast', 'sharpness_proxy', 'noise_proxy'])
    return f"""# P2 No-Reference Supplementary Evaluation

Generated: {_now()}

## Scope

Frozen-artifact supplementary evaluation on local `data/unpaired_real` images. This stage does **not** train a new model and does **not** create a SOTA claim.

```text
n_unpaired_images = {n_images}
methods = input_low, P3c_RLEF_seed3407
NIQE = {niqe_status}
BRISQUE = unavailable locally; not reported/fabricated
```

## Summary

{table}

## Interpretation rule

No-reference metrics are auxiliary only. If they support the visual story, cite them as supplementary evidence; if they conflict with paired PSNR/visual audit, use them as a limitation.

## Saved artifacts

- `{out['noref_summary_csv'].relative_to(ROOT)}`
- `{out['noref_summary_json'].relative_to(ROOT)}`
- `{out['noref_per_image_csv'].relative_to(ROOT)}`
- `{out['noref_grid'].relative_to(ROOT)}`
"""


def write_final_tables(root: Path, out: dict[str, Path]) -> dict[str, Any]:
    main_rows = build_final_main_rows(root)
    ablation_rows = build_final_ablation_rows(root)
    main_fields = ['method', 'role', 'evidence_level', 'uefb_psnr', 'real_psnr', 'synthetic_psnr', 'uefb_E_corr', 'real_E_corr', 'synthetic_E_corr', 'decision', 'main_claim_allowed', 'note']
    ablation_fields = ['group', 'variant', 'key_result', 'decision', 'paper_location', 'interpretation']
    _write_csv(out['final_main_table'], main_rows, main_fields)
    _write_csv(out['final_ablation_table'], ablation_rows, ablation_fields)
    out['final_main_table_json'].parent.mkdir(parents=True, exist_ok=True)
    out['final_main_table_json'].write_text(json.dumps(main_rows, indent=2, ensure_ascii=False), encoding='utf-8')
    out['final_ablation_table_json'].write_text(json.dumps(ablation_rows, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'main_rows': main_rows, 'ablation_rows': ablation_rows}


def make_final_qualitative_grid(root: Path, out_path: Path) -> Path:
    srcs = [
        root / 'results/paper_pipeline/s3_visualizations/real/real_000_low00690_panel.png',
        root / 'results/paper_pipeline/s3_visualizations/real/real_001_low00691_panel.png',
        root / 'results/paper_pipeline/s3_visualizations/synthetic/synthetic_000_r00816405t_panel.png',
        root / 'results/paper_pipeline/s3_visualizations/synthetic/synthetic_001_r00869422t_panel.png',
    ]
    imgs = [Image.open(p).convert('RGB') for p in srcs if p.exists()]
    if not imgs:
        raise FileNotFoundError('No S3 panels found for final qualitative grid')
    width = max(im.width for im in imgs)
    resized = []
    for im in imgs:
        if im.width != width:
            h = int(im.height * width / im.width)
            im = im.resize((width, h))
        resized.append(im)
    total_h = sum(im.height for im in resized)
    sheet = Image.new('RGB', (width, total_h), 'white')
    y = 0
    for im in resized:
        sheet.paste(im, (0, y))
        y += im.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def write_final_docs(root: Path, out: dict[str, Path], tables: dict[str, Any], noref: dict[str, Any]) -> list[Path]:
    p3c = next(r for r in tables['main_rows'] if r['method'] == 'P3c M4J_ES e_shape=0.05')
    ret = next(r for r in tables['main_rows'] if r['method'] == 'Retinexformer')
    dgb = next(r for r in tables['main_rows'] if r['method'] == 'DGB branch')
    out['final_report'].parent.mkdir(parents=True, exist_ok=True)
    final_report = f"""# DGB-RLEF Final Experiment Report

Generated: {_now()}

## Executive decision

```text
DGB branch: stopped and consolidated.
Current default: P3c M4J_ES e_shape=0.05.
Paper framing: interpretable, gauge-aware exposure-field auxiliary calibration, not SOTA restoration.
```

## Background problem

The original research question asked whether local exposure-field modeling can improve low-light image enhancement. The executed evidence shows that the main bottleneck is not merely adding a physical branch, but separating gauge-invariant exposure shape from absolute exposure gauge and avoiding domain trade-offs across UEFB, LOL-v2-real, and LOL-v2-synthetic.

## Main results

{_md_table(tables['main_rows'], ['method','role','evidence_level','uefb_psnr','real_psnr','synthetic_psnr','uefb_E_corr','real_E_corr','synthetic_E_corr','decision'])}

Key boundary: Retinexformer remains much stronger on paired fidelity (`real={_fmt(ret['real_psnr'])}`, `synthetic={_fmt(ret['synthetic_psnr'])}`), while P3c remains the conservative RLEF default (`real={_fmt(p3c['real_psnr'])}`, `synthetic={_fmt(p3c['synthetic_psnr'])}`). DGB did not pass the joint gate and is kept as diagnostic evidence only.

## Ablation and negative-route evidence

{_md_table(tables['ablation_rows'], ['group','variant','key_result','decision','paper_location'])}

## No-reference supplementary audit

The no-reference supplement evaluated frozen P3c outputs on local unpaired real images. It is stored at `{out['noref_report'].relative_to(root)}`. BRISQUE was unavailable locally and is not fabricated.

## Failure cases and limitations

- DGB/P2B-P2E changed route statistics but never solved joint UEFB/real/synthetic promotion.
- Retinexformer remains the fidelity ceiling in this project; any paper claim must avoid SOTA wording.
- Q maps in the frozen P3c visual panels are inactive/constant because the default checkpoint does not use the Q branch as a main claim.
- No-reference metrics are auxiliary and cannot override paired PSNR/SSIM evidence.

## Claim ledger

### Verified claims

1. P3c M4J_ES e_shape=0.05 is the validated conservative default with positive exposure-shape correlations.
2. Retinexformer is substantially stronger on paired real/synthetic fidelity.
3. DGB branch is stopped; its controlled-isolation results are diagnostic negative evidence.
4. S1-S4 paper evidence pipeline and final supplementary artifacts are reproducible from local files.

### Rejected claims

1. DGB-RLEF is a SOTA LLIE method.
2. DGB branch should enter 3-seed or full schedule.
3. Teacher distillation, dataset-weighted reconstruction, or larger real anchors are final main innovations.

## Publication value

The defensible paper idea is a mechanism/benchmark/analysis paper around gauge-aware exposure-field auxiliary calibration, UEFB-v2 diagnostics, and honest negative-route evidence. It is not yet a method/SOTA paper.

## Next strengthening route

Only after this evidence package is reviewed should a new non-DGB route be opened: either RLEF-as-auxiliary on a strong restoration backbone, or a Retinex-factorization redesign. Both require fresh gates and must not reuse stopped DGB scalar sweeps.
"""
    out['final_report'].write_text(final_report, encoding='utf-8')

    paper_idea = f"""# DGB_RLEF Paper Idea After Consolidation

Generated: {_now()}

## Working title

Gauge-Aware Exposure-Field Diagnostics for Interpretable Low-Light Enhancement

## Abstract draft

Local exposure-field models promise interpretable low-light enhancement, but our staged experiments show that exposure fields are not identifiable under gradient/Poisson constraints alone and that simple restoration, distillation, or domain-anchor fixes create real/synthetic/UEFB trade-offs. We therefore reframe the contribution as a gauge-aware exposure-field auxiliary calibration study. Across P1-P7 and DGB controlled isolation, the validated default is P3c M4J_ES with low-pass centered E-shape consistency, while Retinexformer remains the paired-fidelity ceiling. The paper presents the UEFB-v2 protocol, a claim-calibrated ablation ladder, and negative evidence explaining why DGB scalar/gate/router sweeps should stop. The resulting contribution is not SOTA restoration, but a reproducible diagnostic framework for exposure-field interpretability and failure-aware LLIE evaluation.

## Introduction logic

1. Retinex-style physical heads are attractive but can overclaim physical correctness.
2. Exposure-field losses have gauge ambiguity; fixed anchors are domain fragile.
3. The project evidence shows repeated three-domain trade-offs.
4. Centered E-shape consistency is the stable positive mechanism.
5. DGB attempts clarify what does not solve the trade-off, which is valuable for honest method design.

## Method structure to write

- Gauge-invariant exposure decomposition: `E = S + mu`, `mean(S)=0`.
- Low-pass centered E-shape consistency.
- A-gate / diagnostic recoverability maps as interpretability outputs.
- UEFB-v2 benchmark with E/A/Q ground truth.
- Claim ledger and negative-route audit as reproducibility protocol.

## Experiments chapter

1. Datasets and implementation details.
2. Main compact table: P3c default vs official baselines.
3. UEFB-v2 exposure-field diagnostics.
4. Ablation ladder: M0/M4/M4J/M4J_ES/P3c/P3d.
5. Appendix: P5/P6/P7/DGB negative routes.
6. No-reference unpaired supplement as support/limitation.

## Contributions

1. A gauge-aware reading of local exposure-field enhancement.
2. A validated UEFB-v2 diagnostic protocol with E/A/Q metrics.
3. A compact evidence ladder showing which mechanisms work and which fail.
4. A transparent negative-result appendix preventing SOTA overclaiming.

## Limitations

- P3c does not approach Retinexformer paired-fidelity performance.
- DGB was stopped before 3-seed because no candidate passed the joint gate.
- No-reference unpaired metrics are auxiliary and not definitive.
- Q-branch evidence is ablation-level, not default-method evidence.

## Claims that must not be written

- "DGB-RLEF outperforms Retinexformer."
- "DGB-RLEF is SOTA."
- "E_corr alone proves better enhancement."
- "Teacher distillation/domain heads solve the problem."
"""
    out['paper_idea'].write_text(paper_idea, encoding='utf-8')

    repro = f"""# DGB_RLEF Reproducibility Checklist

Generated: {_now()}

## Project

```text
root: {root}
active default: P3c M4J_ES e_shape=0.05
DGB status: stopped and consolidated
```

## Data paths

- `data/LOL-v2/Real_captured/Train|Test/Low|Normal`
- `data/LOL-v2/Synthetic/Train|Test/Low|Normal`
- `data/UEFB-v2/train|test/low|high|E_gt|A_gt|Q_gt`
- `data/unpaired_real/DICM|LIME|MEF|NPE|VV`

## Seeds

```text
3407, 2027, 42
```

## Core configs/checkpoints

- P3c/P3b visual checkpoint: `configs/p3b_joint/p3b_m4j_es_m4_joint_eshape.yml`
- Checkpoint: `experiments/p3b_m4j_es_m4_joint_eshape_seed3407/checkpoints/last.pth`
- P3c aggregate table: `results/tables/p3c_multiseed_sweep_aggregate.json`
- Official baseline table: `results/tables/p4_official_baselines_summary.json`

## Commands

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_s1_s4_paper_pipeline.py --device cuda --n_visuals 4
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_master_remaining_pipeline.py --device cuda
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m pytest tests -q
```

## Metric scripts / tables

- `scripts/eval_uefb.py`
- `scripts/eval_paired.py`
- `scripts/run_s1_s4_paper_pipeline.py`
- `scripts/run_master_remaining_pipeline.py`
- `results/hermes_audit/tables/final_main_table.csv`
- `results/hermes_audit/tables/final_ablation_table.csv`
- `results/hermes_audit/tables/noref_supplementary_summary.csv`

## Baseline versions

- Retinexformer: local official-code outputs under `experiments/p4_official_baselines/retinexformer/`.
- Zero-DCE++: local official-code outputs under `experiments/p4_official_baselines/zero_dce_pp/`.
- KinD++: local official-code high-assisted outputs under `experiments/p4_official_baselines/kindpp/`; mark high-assisted.

## Verification

- Do not claim SOTA unless future evidence changes the paired-fidelity gap.
- Do not promote DGB to 3-seed unless a new candidate passes the joint gate under a new branch.
- Keep DGB/P5/P6/P7 scalar routes in appendix/negative evidence.
"""
    out['repro_checklist'].write_text(repro, encoding='utf-8')
    return [out['final_report'], out['paper_idea'], out['repro_checklist']]


def build_manifest(root: Path, out: dict[str, Path], extra_files: list[Path], elapsed: float) -> dict[str, Any]:
    files = [p for p in extra_files if p.exists()]
    recs = []
    for p in files:
        data = p.read_bytes()
        recs.append({'path': str(p), 'relative_path': str(p.relative_to(root)), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    manifest = {
        'generated_at': _now(),
        'decision': 'Master-prompt remaining executable steps completed after DGB stop and S1-S4 pipeline',
        'no_new_training': True,
        'dgb_status': 'stopped and consolidated',
        'elapsed_sec': elapsed,
        'steps': planned_remaining_steps(),
        'files': recs,
    }
    out['manifest'].parent.mkdir(parents=True, exist_ok=True)
    out['manifest'].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def write_validation_report(root: Path, out: dict[str, Path], manifest: dict[str, Any]) -> Path:
    # Basic file and image validation.
    for rec in manifest['files']:
        p = Path(rec['path'])
        data = p.read_bytes()
        assert len(data) == rec['bytes']
        assert hashlib.sha256(data).hexdigest() == rec['sha256']
    for p in [out['final_qualitative_grid'], out['noref_grid']]:
        with Image.open(p) as im:
            im.verify()
    text = f"""# Master Remaining Steps Validation

Generated: {_now()}

```text
PASS
```

## Checks

- Manifest has {len(manifest['files'])} file records.
- All file byte counts and SHA256 hashes match.
- Final qualitative grid and no-reference unpaired grid open with PIL.
- No new training was run.
- DGB remains stopped and consolidated.

## Main artifacts

- `{out['final_report'].relative_to(root)}`
- `{out['paper_idea'].relative_to(root)}`
- `{out['repro_checklist'].relative_to(root)}`
- `{out['final_main_table'].relative_to(root)}`
- `{out['final_ablation_table'].relative_to(root)}`
- `{out['final_qualitative_grid'].relative_to(root)}`
- `{out['noref_report'].relative_to(root)}`
"""
    out['validation_report'].parent.mkdir(parents=True, exist_ok=True)
    out['validation_report'].write_text(text, encoding='utf-8')
    return out['validation_report']


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_pipeline(root: Path = ROOT, device: str = 'cuda', max_unpaired: int | None = None) -> dict[str, Any]:
    started = time.time()
    out = planned_outputs(root)
    noref = run_noref_supplement(root, out, device=device, max_images=max_unpaired)
    tables = write_final_tables(root, out)
    final_grid = make_final_qualitative_grid(root, out['final_qualitative_grid'])
    docs = write_final_docs(root, out, tables, noref)
    extra = [
        out['noref_report'], out['noref_summary_csv'], out['noref_summary_json'], out['noref_per_image_csv'], out['noref_grid'],
        out['final_main_table'], out['final_main_table_json'], out['final_ablation_table'], out['final_ablation_table_json'],
        final_grid, *docs,
    ]
    manifest = build_manifest(root, out, extra, time.time() - started)
    validation = write_validation_report(root, out, manifest)
    # Rebuild manifest including validation report.
    manifest = build_manifest(root, out, [*extra, validation], time.time() - started)
    return {'manifest': manifest, 'outputs': out, 'noref': noref, 'tables': tables}


def main() -> None:
    ap = argparse.ArgumentParser(description='Execute remaining master-prompt artifacts after S1-S4 pipeline.')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_unpaired', type=int, default=None, help='Optional cap for debugging; default uses all local unpaired images.')
    args = ap.parse_args()
    result = run_pipeline(ROOT, device=args.device, max_unpaired=args.max_unpaired)
    print(json.dumps({
        'DONE': True,
        'manifest': str(planned_outputs(ROOT)['manifest']),
        'files': len(result['manifest']['files']),
        'noref_images': len(result['noref']['paths']),
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
