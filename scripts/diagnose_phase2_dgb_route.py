#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import torch
import yaml
from torch.utils.data import DataLoader

from scripts.train import build_model, build_dataset


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5


def diagnose_loader(model, loader, device) -> dict:
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch['low'], domain=batch.get('dataset'))
            a = out['A'].clamp(1e-6, 1 - 1e-6)
            entropy = -(a * torch.log(a) + (1 - a) * torch.log(1 - a)).mean()
            safe_gap = (out.get('I_safe', batch['low']) - out['I_rest']).abs().mean()
            route_safe_fraction = (1.0 - out['A']).mean()
            mu = out['mu_E']
            rows.append({
                'A_mean': float(out['A'].mean().cpu()),
                'A_std': float(out['A'].std(unbiased=False).cpu()),
                'route_entropy': float(entropy.cpu()),
                'safe_fraction': float(route_safe_fraction.cpu()),
                'safe_rest_l1': float(safe_gap.cpu()),
                'mu_mean': float(mu.mean().cpu()),
                'mu_std': float(mu.std(unbiased=False).cpu()),
                'mu_min': float(mu.min().cpu()),
                'mu_max': float(mu.max().cpu()),
            })
    if not rows:
        return {'n_batches': 0}
    mean_keys = ['A_mean', 'A_std', 'route_entropy', 'safe_fraction', 'safe_rest_l1', 'mu_mean']
    payload = {k: _mean([r[k] for r in rows]) for k in mean_keys}
    payload['mu_dataset_std'] = _std([r['mu_mean'] for r in rows])
    payload['mu_min'] = min(r['mu_min'] for r in rows)
    payload['mu_max'] = max(r['mu_max'] for r in rows)
    payload['n_batches'] = len(rows)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    specs = {
        'uefb': cfg['data']['val'],
        'real': {'type': 'paired_rgb', 'low_dir': str(ROOT / 'data/LOL-v2/Real_captured/Test/Low'), 'high_dir': str(ROOT / 'data/LOL-v2/Real_captured/Test/Normal'), 'name': 'real'},
        'synthetic': {'type': 'paired_rgb', 'low_dir': str(ROOT / 'data/LOL-v2/Synthetic/Test/Low'), 'high_dir': str(ROOT / 'data/LOL-v2/Synthetic/Test/Normal'), 'name': 'synthetic'},
    }
    payload = {}
    for name, spec in specs.items():
        ds = build_dataset(spec, 'val')
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        payload[name] = diagnose_loader(model, loader, device)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
