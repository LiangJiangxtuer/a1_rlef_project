#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

PY = '/home/user/miniconda3/envs/cutler_dinov3/bin/python'
ROOT = Path(__file__).resolve().parents[1]


BASE_M4_MODEL = {
    'base_channels': 24,
    'e_max': 3.5,
    'exposure_branch': True,
    'adaptive_gauge': True,
    'fixed_gauge': None,
    'physics_branch': True,
    'gate_branch': True,
    'q_branch': False,
}

BASE_M4_LOSS = {
    'rec': 1.0,
    'phys': 0.15,
    'poisson': 0.05,
    'gauge': 0.10,
    'id': 0.02,
    'gate': 0.02,
    'wtv': 0.02,
}


def planned_runs() -> list[dict]:
    eshape = dict(BASE_M4_LOSS)
    eshape.update({'e_shape': 0.05, 'e_shape_kernel': 7})
    return [
        {'id': 'M4J', 'name': 'm4_joint', 'model': dict(BASE_M4_MODEL), 'loss': dict(BASE_M4_LOSS)},
        {'id': 'M4J_ES', 'name': 'm4_joint_eshape', 'model': dict(BASE_M4_MODEL), 'loss': eshape},
    ]


def paired_eval_specs() -> list[dict]:
    return [
        {
            'dataset': 'real',
            'low_dir': str(ROOT / 'data/LOL-v2/Real_captured/Test/Low'),
            'high_dir': str(ROOT / 'data/LOL-v2/Real_captured/Test/Normal'),
            'max_images': None,
        },
        {
            'dataset': 'synthetic',
            'low_dir': str(ROOT / 'data/LOL-v2/Synthetic/Test/Low'),
            'high_dir': str(ROOT / 'data/LOL-v2/Synthetic/Test/Normal'),
            'max_images': None,
        },
    ]


def make_cfg(run: dict, max_steps: int = 1000, train_crop: int = 128, batch_size: int = 8) -> dict:
    return {
        'seed': 3407,
        'protocol': {
            'stage': 'P3b joint training',
            'variant_id': run['id'],
            'variant_name': run['name'],
            'notes': 'M4 gate joint training on UEFB-v2 + LOL-v2-real + LOL-v2-synthetic; optional e_shape is an E spatial-shape repair diagnostic.',
        },
        'model': run['model'],
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
            'train': [
                {'type': 'uefb', 'root': str(ROOT / 'data/UEFB-v2/train'), 'crop_size': train_crop, 'augment': True, 'max_images': None},
                {'type': 'paired_rgb', 'low_dir': str(ROOT / 'data/LOL-v2/Real_captured/Train/Low'), 'high_dir': str(ROOT / 'data/LOL-v2/Real_captured/Train/Normal'), 'crop_size': train_crop, 'augment': True, 'max_images': None, 'name': 'LOL-v2-real-train'},
                {'type': 'paired_rgb', 'low_dir': str(ROOT / 'data/LOL-v2/Synthetic/Train/Low'), 'high_dir': str(ROOT / 'data/LOL-v2/Synthetic/Train/Normal'), 'crop_size': train_crop, 'augment': True, 'max_images': None, 'name': 'LOL-v2-synthetic-train'},
            ],
            'val': {'type': 'uefb', 'root': str(ROOT / 'data/UEFB-v2/test'), 'crop_size': None, 'augment': False, 'max_images': None},
        },
        'loss': run['loss'],
    }


def _run(cmd: list[str | Path], log_path: Path, env: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write('+ ' + ' '.join(map(str, cmd)) + '\n')
        log.flush()
        proc = subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'command failed with code {proc.returncode}; see {log_path}')


def _prefix_metrics(prefix: str, metrics: dict) -> dict:
    return {f'{prefix}_{k}': v for k, v in metrics.items()}


def training_complete(run_dir: Path) -> bool:
    return (run_dir / 'checkpoints/last.pth').exists() and (run_dir / 'metrics/eval_metrics.json').exists()


def run_one(run: dict, max_steps: int, train_crop: int, batch_size: int, gpu: int, resume: bool = False) -> dict:
    tag = f"p3b_{run['id'].lower()}_{run['name']}"
    cfg = make_cfg(run, max_steps=max_steps, train_crop=train_crop, batch_size=batch_size)
    cfg_dir = ROOT / 'configs/p3b_joint'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f'{tag}.yml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    out = ROOT / 'experiments' / f'{tag}_seed3407'
    log_path = ROOT / 'logs/p3b_joint' / f'{tag}.log'
    if log_path.exists() and not resume:
        log_path.unlink()
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    start = time.time()
    if resume and training_complete(out):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'RESUME: found existing checkpoint and UEFB metrics under {out}\n')
    else:
        _run([PY, 'scripts/train.py', '--config', cfg_path, '--output_dir', out, '--device', 'cuda', '--max_steps', str(max_steps)], log_path, env)
    row = {
        'run': tag,
        'variant_id': run['id'],
        'variant_name': run['name'],
        'steps': max_steps,
        'train_crop': train_crop,
        'batch_size': batch_size,
        'gpu': gpu,
        'elapsed_sec_train': time.time() - start,
        'run_dir': str(out),
        'log': str(log_path),
    }
    row.update(_prefix_metrics('uefb', json.loads((out / 'metrics/eval_metrics.json').read_text())))
    ckpt = out / 'checkpoints/last.pth'
    for spec in paired_eval_specs():
        pair_out = out / 'paired_eval' / spec['dataset']
        cmd = [PY, 'scripts/eval_paired.py', '--config', cfg_path, '--checkpoint', ckpt, '--low_dir', spec['low_dir'], '--high_dir', spec['high_dir'], '--output_dir', pair_out, '--device', 'cuda']
        if spec['max_images'] is not None:
            cmd += ['--max_images', str(spec['max_images'])]
        _run(cmd, log_path, env)
        row.update(_prefix_metrics(spec['dataset'], json.loads((pair_out / 'metrics.json').read_text())))
    row['elapsed_sec_total'] = time.time() - start
    return row


def write_summary(rows: list[dict]) -> Path:
    rows = sorted(rows, key=lambda r: r['variant_id'])
    table = ROOT / 'results/tables/p3b_joint_m4_summary.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    preferred = ['run','variant_id','variant_name','steps','train_crop','batch_size','gpu','elapsed_sec_train','elapsed_sec_total','uefb_psnr','uefb_ssim','uefb_E_MAE','uefb_E_MAE_aligned','uefb_E_corr','uefb_identity_drop','uefb_q_ece','real_psnr','real_ssim','real_lee','real_nai','synthetic_psnr','synthetic_ssim','synthetic_lee','synthetic_nai','run_dir','log']
    ordered = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with table.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader(); w.writerows(rows)
    (ROOT / 'results/tables/p3b_joint_m4_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_steps', type=int, default=1000)
    ap.add_argument('--train_crop', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--parallel', type=int, default=2)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if not (ROOT / 'data/UEFB-v2/train/meta.json').exists():
        raise FileNotFoundError('UEFB-v2 formal data missing')
    runs = planned_runs()
    print(json.dumps({'P3B_START': True, 'runs': [(r['id'], r['name']) for r in runs], 'max_steps': args.max_steps, 'parallel': args.parallel, 'resume': args.resume}, indent=2), flush=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(run_one, run, args.max_steps, args.train_crop, args.batch_size, i % args.parallel, args.resume) for i, run in enumerate(runs)]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(json.dumps({'DONE_RUN': row['run'], 'uefb_psnr': row.get('uefb_psnr'), 'uefb_E_corr': row.get('uefb_E_corr'), 'real_psnr': row.get('real_psnr'), 'synthetic_psnr': row.get('synthetic_psnr')}, ensure_ascii=False), flush=True)
    table = write_summary(rows)
    print(json.dumps({'P3B_DONE': True, 'summary': str(table), 'rows': rows}, indent=2), flush=True)


if __name__ == '__main__':
    main()
