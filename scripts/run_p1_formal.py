#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

PY = '/home/user/miniconda3/envs/cutler_dinov3/bin/python'
ROOT = Path(__file__).resolve().parents[1]


def paired_spec(dataset: str, split: str, train_crop: int | None = 128):
    if dataset == 'real':
        base = ROOT / 'data/LOL-v2/Real_captured'
    elif dataset == 'synthetic':
        base = ROOT / 'data/LOL-v2/Synthetic'
    else:
        raise ValueError(dataset)
    folder = 'Train' if split == 'train' else 'Test'
    return {
        'type': 'paired_rgb',
        'low_dir': str(base / folder / 'Low'),
        'high_dir': str(base / folder / 'Normal'),
        # Formal eval uses full images. Training still uses random crops.
        'crop_size': train_crop if split == 'train' else None,
        'augment': split == 'train',
        'max_images': None,
        'name': f'LOL-v2-{dataset}-{split}-formal',
    }


def planned_runs() -> list[dict[str, str]]:
    return [
        {'dataset': dataset, 'mode': mode, 'tag': f'p1formal_{dataset}_{mode}'}
        for dataset in ['real', 'synthetic']
        for mode in ['nogauge', 'fixed0p02', 'adaptive']
    ]


def make_cfg(dataset: str, mode: str, max_steps: int = 1000, train_crop: int = 128, batch_size: int = 8):
    if mode == 'nogauge':
        adaptive = False
        fixed = None
        gauge_w = 0.0
    elif mode == 'fixed0p02':
        adaptive = False
        fixed = 0.02
        gauge_w = 0.05
    elif mode == 'adaptive':
        adaptive = True
        fixed = None
        gauge_w = 0.10
    else:
        raise ValueError(mode)
    return {
        'seed': 3407,
        'protocol': {
            'stage': 'P1 formal',
            'dataset': dataset,
            'mode': mode,
            'notes': '1000-step full-train/full-test gauge replication; val crop_size=None means full-resolution test images.',
        },
        'model': {
            'base_channels': 24,
            'exposure_branch': True,
            'adaptive_gauge': adaptive,
            'fixed_gauge': fixed,
            'physics_branch': True,
            'gate_branch': False,
            'q_branch': False,
        },
        'training': {
            'batch_size': batch_size,
            'max_steps': max_steps,
            'lr': 2e-4,
            'weight_decay': 1e-4,
            'crop_size': train_crop,
            'log_every': 100,
            'num_workers': 0,
            'grad_clip': 1.0,
        },
        'eval': {'batch_size': 1},
        'data': {
            'train': paired_spec(dataset, 'train', train_crop=train_crop),
            'val': paired_spec(dataset, 'val', train_crop=train_crop),
        },
        'loss': {
            'rec': 1.0,
            'phys': 0.10,
            'poisson': 0.03,
            'gauge': gauge_w,
            'wtv': 0.01,
        },
    }


def run_one(run: dict[str, str], max_steps: int, train_crop: int, batch_size: int, gpu: int) -> dict:
    tag = run['tag']
    cfg = make_cfg(run['dataset'], run['mode'], max_steps=max_steps, train_crop=train_crop, batch_size=batch_size)
    cfg_dir = ROOT / 'configs/p1_formal'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f'{tag}.yml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    out = ROOT / 'experiments' / f'{tag}_seed3407'
    log_path = ROOT / 'logs' / 'p1_formal' / f'{tag}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, 'scripts/train.py', '--config', str(cfg_path), '--output_dir', str(out), '--device', 'cuda', '--max_steps', str(max_steps)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    start = time.time()
    with log_path.open('w', encoding='utf-8') as log:
        log.write('+ ' + ' '.join(cmd) + f'  # CUDA_VISIBLE_DEVICES={gpu}\n')
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'{tag} failed with code {proc.returncode}; see {log_path}')
    metrics_path = out / 'metrics/eval_metrics.json'
    metrics = json.loads(metrics_path.read_text())
    return {
        'run': tag,
        'dataset': run['dataset'],
        'mode': run['mode'],
        'steps': max_steps,
        'train_crop': train_crop,
        'batch_size': batch_size,
        'gpu': gpu,
        'elapsed_sec': time.time() - start,
        **metrics,
        'run_dir': str(out),
        'log': str(log_path),
    }


def write_summary(rows: list[dict]) -> Path:
    rows = sorted(rows, key=lambda r: (r['dataset'], r['mode']))
    table = ROOT / 'results/tables/p1_formal_1000_summary.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (ROOT / 'results/tables/p1_formal_1000_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_steps', type=int, default=1000)
    ap.add_argument('--train_crop', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--parallel', type=int, default=2)
    args = ap.parse_args()
    runs = planned_runs()
    rows = []
    print(json.dumps({'P1_FORMAL_START': True, 'runs': runs, 'max_steps': args.max_steps, 'train_crop': args.train_crop, 'batch_size': args.batch_size, 'parallel': args.parallel}, indent=2), flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = []
        for i, run in enumerate(runs):
            futs.append(ex.submit(run_one, run, args.max_steps, args.train_crop, args.batch_size, i % args.parallel))
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            print(json.dumps({'DONE_RUN': row['run'], 'dataset': row['dataset'], 'mode': row['mode'], 'psnr': row.get('psnr'), 'ssim': row.get('ssim'), 'elapsed_sec': row['elapsed_sec']}, ensure_ascii=False), flush=True)
    table = write_summary(rows)
    print(json.dumps({'P1_FORMAL_DONE': True, 'summary': str(table), 'rows': rows}, indent=2), flush=True)


if __name__ == '__main__':
    main()
