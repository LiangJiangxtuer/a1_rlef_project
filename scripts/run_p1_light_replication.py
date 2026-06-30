#!/usr/bin/env python3
from __future__ import annotations

import csv, json, subprocess, sys
from pathlib import Path
import yaml

PY = '/home/user/miniconda3/envs/cutler_dinov3/bin/python'
ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=ROOT)


def paired_spec(dataset: str, split: str, max_images: int):
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
        'crop_size': 64 if split == 'train' else 96,
        'augment': split == 'train',
        'max_images': max_images,
        'name': f'LOL-v2-{dataset}-{split}',
    }


def make_cfg(dataset: str, mode: str):
    fixed = None if mode == 'nogauge' else 0.02
    adaptive = False
    return {
        'seed': 3407,
        'model': {
            'base_channels': 24,
            'exposure_branch': True,
            'adaptive_gauge': adaptive,
            'fixed_gauge': fixed,
            'physics_branch': True,
            'gate_branch': False,
            'q_branch': False,
        },
        'training': {'batch_size': 8, 'max_steps': 30, 'lr': 2e-4, 'crop_size': 64, 'log_every': 10, 'num_workers': 0},
        'eval': {'batch_size': 1},
        'data': {'train': paired_spec(dataset, 'train', 64), 'val': paired_spec(dataset, 'val', 12)},
        'loss': {'rec': 1.0, 'phys': 0.10, 'poisson': 0.03, 'gauge': 0.05 if fixed is not None else 0.0, 'wtv': 0.01},
    }


def main():
    rows = []
    cfg_dir = ROOT / 'configs/p1_light'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ['real', 'synthetic']:
        for mode in ['nogauge', 'fixed0p02']:
            tag = f'p1_{dataset}_{mode}'
            cfg = make_cfg(dataset, mode)
            cfg_path = cfg_dir / f'{tag}.yml'
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
            out = ROOT / 'experiments' / f'{tag}_seed3407'
            run([PY, 'scripts/train.py', '--config', cfg_path, '--output_dir', out, '--device', 'cuda', '--max_steps', '30'])
            metrics = json.loads((out / 'metrics/eval_metrics.json').read_text())
            rows.append({'run': tag, 'dataset': dataset, 'mode': mode, **metrics, 'run_dir': str(out)})
    table = ROOT / 'results/tables/p1_light_replication_summary.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (ROOT / 'results/tables/p1_light_replication_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps({'summary': str(table), 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
