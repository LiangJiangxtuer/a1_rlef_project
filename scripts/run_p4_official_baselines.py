#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rlef.metrics.full_reference import psnr_torch, ssim_torch
from rlef.metrics.exposure_field import identity_drop, local_exposure_error, noise_amplification_index, saturation_rate

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def p4_baseline_plan() -> list[dict]:
    return [
        {
            'method': 'Retinexformer',
            'source': 'official repo caiyuanhao1998/Retinexformer',
            'required_runtime': 'PyTorch + local BasicSR package from official repo',
            'status': 'executed_or_evaluable',
        },
        {
            'method': 'Zero-DCE++',
            'source': 'official repo Li-Chongyi/Zero-DCE_extension, official Epoch99.pth',
            'required_runtime': 'PyTorch official model.py with dataset wrapper preserving official scale_factor=12',
            'status': 'executed_or_evaluable',
        },
        {
            'method': 'KinD++',
            'source': 'official repo zhangyhuaee/KinD_plus',
            'required_runtime': 'tensorflow.compat.v1 plus official checkpoint',
            'status': 'attempt_or_blocker',
        },
    ]


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


def _index_by_key(path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in image_files(path):
        out[canonical_key(p)] = p
    return out


def _load_tensor(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _crop_to_common(pred: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = min(pred.shape[-2], low.shape[-2], high.shape[-2])
    w = min(pred.shape[-1], low.shape[-1], high.shape[-1])
    return pred[..., :h, :w], low[..., :h, :w], high[..., :h, :w]


def evaluate_image_outputs(pred_dir: str | Path, low_dir: str | Path, high_dir: str | Path) -> tuple[dict, list[dict]]:
    pred_dir, low_dir, high_dir = Path(pred_dir), Path(low_dir), Path(high_dir)
    pred_idx, low_idx, high_idx = _index_by_key(pred_dir), _index_by_key(low_dir), _index_by_key(high_dir)
    keys = sorted(set(pred_idx) & set(low_idx) & set(high_idx))
    if not keys:
        raise FileNotFoundError(f'no paired images for pred={pred_dir}, low={low_dir}, high={high_dir}')
    rows = []
    for key in keys:
        pred, low, high = _crop_to_common(_load_tensor(pred_idx[key]), _load_tensor(low_idx[key]), _load_tensor(high_idx[key]))
        sat = saturation_rate(pred)
        row = {
            'key': key,
            'pred_path': str(pred_idx[key]),
            'low_path': str(low_idx[key]),
            'high_path': str(high_idx[key]),
            'height': int(pred.shape[-2]),
            'width': int(pred.shape[-1]),
            'psnr': float(psnr_torch(pred, high)),
            'ssim': float(ssim_torch(pred, high)),
            'lee': float(local_exposure_error(pred, high)),
            'nai': float(noise_amplification_index(pred, low)),
            'input_psnr': float(psnr_torch(low, high)),
            'identity_drop': float(identity_drop(pred, low, high)),
            'over': float(sat['over']),
            'under': float(sat['under']),
        }
        rows.append(row)
    metrics = {'n': len(rows)}
    for k in ['psnr', 'ssim', 'lee', 'nai', 'input_psnr', 'identity_drop', 'over', 'under']:
        metrics[k] = float(sum(r[k] for r in rows) / len(rows))
    return metrics, rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        fields = sorted({k for r in rows for k in r})
        preferred = ['method', 'dataset', 'n', 'psnr', 'ssim', 'lee', 'nai', 'input_psnr', 'identity_drop', 'over', 'under', 'status', 'note']
        ordered = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader(); w.writerows(rows)


def run_zero_dce_pp(device: str = 'cuda') -> dict:
    base = ROOT / 'external_baselines/Zero-DCE_extension/Zero-DCE++'
    model_path = base / 'model.py'
    ckpt = base / 'snapshots_Zero_DCE++/Epoch99.pth'
    if not model_path.exists() or not ckpt.exists():
        raise FileNotFoundError(f'Zero-DCE++ official code or checkpoint missing: {model_path}, {ckpt}')
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(base)
        sys.path.insert(0, str(base))
        spec = importlib.util.spec_from_file_location('zero_dce_pp_model', model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot import {model_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dev = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
        scale_factor = 12
        net = module.enhance_net_nopool(scale_factor).to(dev)
        net.load_state_dict(torch.load(ckpt, map_location=dev))
        net.eval()
        out_root = ROOT / 'experiments/p4_official_baselines/zero_dce_pp/results'
        specs = [
            ('real', ROOT / 'data/LOL-v2/Real_captured/Test/Low'),
            ('synthetic', ROOT / 'data/LOL-v2/Synthetic/Test/Low'),
        ]
        counts = {}
        with torch.no_grad():
            for dataset, low_dir in specs:
                out_dir = out_root / dataset
                out_dir.mkdir(parents=True, exist_ok=True)
                count = 0
                for img_path in sorted([p for p in low_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]):
                    arr = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255.0
                    h = (arr.shape[0] // scale_factor) * scale_factor
                    w = (arr.shape[1] // scale_factor) * scale_factor
                    arr = arr[:h, :w, :]
                    x = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).to(dev)
                    y, _ = net(x)
                    y_np = y.clamp(0, 1).squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
                    Image.fromarray((y_np * 255.0 + 0.5).astype(np.uint8)).save(out_dir / img_path.name)
                    count += 1
                counts[dataset] = count
        return counts
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def retinexformer_pred_dirs() -> dict[str, Path]:
    base = ROOT / 'experiments/p4_official_baselines/retinexformer/results'
    return {
        'real': base / 'LOL_v2_real/RetinexFormer_LOL_v2_real/LOL_v2_real',
        'synthetic': base / 'LOL_v2_synthetic/RetinexFormer_LOL_v2_synthetic/LOL_v2_synthetic',
    }


def low_high_dirs(dataset: str) -> tuple[Path, Path]:
    if dataset == 'real':
        return ROOT / 'data/LOL-v2/Real_captured/Test/Low', ROOT / 'data/LOL-v2/Real_captured/Test/Normal'
    if dataset == 'synthetic':
        return ROOT / 'data/LOL-v2/Synthetic/Test/Low', ROOT / 'data/LOL-v2/Synthetic/Test/Normal'
    raise ValueError(dataset)


def evaluate_p4(run_zero_dce: bool = False, device: str = 'cuda') -> list[dict]:
    if run_zero_dce:
        run_zero_dce_pp(device=device)
    rows: list[dict] = []
    detail_root = ROOT / 'results/tables/p4_details'
    methods = {
        'Retinexformer': retinexformer_pred_dirs(),
        'Zero-DCE++': {
            'real': ROOT / 'experiments/p4_official_baselines/zero_dce_pp/results/real',
            'synthetic': ROOT / 'experiments/p4_official_baselines/zero_dce_pp/results/synthetic',
        },
        'KinD++': {
            'real': ROOT / 'experiments/p4_official_baselines/kindpp/results/real/LOLdataset',
            'synthetic': ROOT / 'experiments/p4_official_baselines/kindpp/results/synthetic/LOLdataset',
        },
    }
    for method, datasets in methods.items():
        for dataset, pred_dir in datasets.items():
            low_dir, high_dir = low_high_dirs(dataset)
            if not pred_dir.exists():
                rows.append({'method': method, 'dataset': dataset, 'status': 'missing_outputs', 'note': str(pred_dir)})
                continue
            metrics, per_image = evaluate_image_outputs(pred_dir, low_dir, high_dir)
            row = {'method': method, 'dataset': dataset, 'status': 'ok', **metrics, 'pred_dir': str(pred_dir)}
            rows.append(row)
            _write_rows(detail_root / f'{method.replace("+", "p").replace("-", "_").lower()}_{dataset}_per_image.csv', per_image)
    out_csv = ROOT / 'results/tables/p4_official_baselines_summary.csv'
    _write_rows(out_csv, rows)
    (ROOT / 'results/tables/p4_official_baselines_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-zero-dce', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    rows = evaluate_p4(run_zero_dce=args.run_zero_dce, device=args.device)
    print(json.dumps({'P4_DONE': True, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
