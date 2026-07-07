#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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

BASE_M4J_ES_LOSS = {
    'rec': 1.0,
    'phys': 0.15,
    'poisson': 0.05,
    'gauge': 0.10,
    'id': 0.02,
    'gate': 0.02,
    'wtv': 0.02,
    'e_shape': 0.05,
    'e_shape_kernel': 7,
}


def planned_jobs() -> list[dict]:
    jobs = []
    seen: set[tuple[int, float]] = set()
    for seed, e_shape in [(3407, 0.05), (2027, 0.05), (42, 0.05), (3407, 0.02), (3407, 0.10)]:
        key = (seed, float(e_shape))
        if key in seen:
            continue
        seen.add(key)
        jobs.append({'seed': seed, 'e_shape': float(e_shape), 'e_shape_kernel': 7})
    return jobs


def multiseed_jobs(jobs: list[dict] | None = None) -> list[dict]:
    jobs = planned_jobs() if jobs is None else jobs
    return [j for j in jobs if abs(float(j['e_shape']) - 0.05) < 1e-12]


def sweep_jobs(jobs: list[dict] | None = None) -> list[dict]:
    jobs = planned_jobs() if jobs is None else jobs
    return [j for j in jobs if int(j['seed']) == 3407]


def job_tag(job: dict) -> str:
    return f"p3c_m4j_es_seed{int(job['seed'])}_e{int(round(float(job['e_shape']) * 1000)):04d}"


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


def make_cfg(job: dict, max_steps: int = 1000, train_crop: int = 128, batch_size: int = 8) -> dict:
    loss = dict(BASE_M4J_ES_LOSS)
    loss['e_shape'] = float(job['e_shape'])
    loss['e_shape_kernel'] = int(job.get('e_shape_kernel', 7))
    return {
        'seed': int(job['seed']),
        'protocol': {
            'stage': 'P3c M4J_ES multi-seed and e_shape sweep',
            'variant_id': 'M4J_ES',
            'variant_name': 'm4_joint_eshape',
            'seed': int(job['seed']),
            'e_shape': float(job['e_shape']),
            'notes': 'M4 gate joint-training on UEFB-v2 + LOL-v2-real + LOL-v2-synthetic with low-pass E-shape consistency; P3c validates 3-seed stability and e_shape trade-off.',
        },
        'model': dict(BASE_M4_MODEL),
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
        'loss': loss,
    }


def reuse_run_dir(job: dict) -> Path | None:
    if int(job['seed']) == 3407 and abs(float(job['e_shape']) - 0.05) < 1e-12:
        p = ROOT / 'experiments/p3b_m4j_es_m4_joint_eshape_seed3407'
        if p.exists():
            return p
    return None


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


def run_one(job: dict, max_steps: int, train_crop: int, batch_size: int, gpu: int, resume: bool = False) -> dict:
    tag = job_tag(job)
    cfg = make_cfg(job, max_steps=max_steps, train_crop=train_crop, batch_size=batch_size)
    cfg_dir = ROOT / 'configs/p3c_multiseed_sweep'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f'{tag}.yml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    reused = reuse_run_dir(job)
    out = reused if reused is not None else ROOT / 'experiments' / tag
    log_path = ROOT / 'logs/p3c_multiseed_sweep' / f'{tag}.log'
    if log_path.exists() and not resume:
        log_path.unlink()
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    start = time.time()
    if reused is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'REUSE: seed=3407 e_shape=0.05 imported from {out}\n')
    elif resume and training_complete(out):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'RESUME: found existing checkpoint and UEFB metrics under {out}\n')
    else:
        _run([PY, 'scripts/train.py', '--config', cfg_path, '--output_dir', out, '--device', 'cuda', '--max_steps', str(max_steps)], log_path, env)
    if not training_complete(out):
        raise FileNotFoundError(f'missing checkpoint or UEFB metrics under {out}')
    row = {
        'run': tag,
        'variant_id': 'M4J_ES',
        'variant_name': 'm4_joint_eshape',
        'seed': int(job['seed']),
        'e_shape': float(job['e_shape']),
        'e_shape_kernel': int(job.get('e_shape_kernel', 7)),
        'steps': max_steps,
        'train_crop': train_crop,
        'batch_size': batch_size,
        'gpu': gpu,
        'reused_from_p3b': bool(reused is not None),
        'elapsed_sec_train_or_reuse': time.time() - start,
        'run_dir': str(out),
        'config': str(cfg_path),
        'log': str(log_path),
    }
    row.update(_prefix_metrics('uefb', json.loads((out / 'metrics/eval_metrics.json').read_text())))
    ckpt = out / 'checkpoints/last.pth'
    for spec in paired_eval_specs():
        pair_out = out / 'paired_eval' / spec['dataset']
        if (pair_out / 'metrics.json').exists() and (resume or reused is not None):
            with log_path.open('a', encoding='utf-8') as log:
                log.write(f'REUSE_EVAL: {pair_out / "metrics.json"}\n')
        else:
            cmd = [PY, 'scripts/eval_paired.py', '--config', cfg_path, '--checkpoint', ckpt, '--low_dir', spec['low_dir'], '--high_dir', spec['high_dir'], '--output_dir', pair_out, '--device', 'cuda']
            if spec['max_images'] is not None:
                cmd += ['--max_images', str(spec['max_images'])]
            _run(cmd, log_path, env)
        row.update(_prefix_metrics(spec['dataset'], json.loads((pair_out / 'metrics.json').read_text())))
    row['elapsed_sec_total'] = time.time() - start
    return row


def _mean_std(vals: list[float]) -> dict:
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
    return {'mean': mean, 'std': math.sqrt(var), 'n': len(vals)}


def aggregate(rows: list[dict]) -> dict:
    metrics = ['uefb_psnr', 'uefb_E_MAE', 'uefb_E_MAE_aligned', 'uefb_E_corr', 'real_psnr', 'real_E_corr', 'synthetic_psnr', 'synthetic_E_corr']
    seed_rows = [r for r in rows if abs(float(r['e_shape']) - 0.05) < 1e-12]
    sweep_rows = sorted([r for r in rows if int(r['seed']) == 3407], key=lambda r: float(r['e_shape']))
    return {
        'multiseed_e_shape_0p05': {m: _mean_std([float(r[m]) for r in seed_rows]) for m in metrics},
        'sweep_seed3407': [{m: r[m] for m in ['seed','e_shape'] + metrics if m in r} for r in sweep_rows],
    }


def write_summary(rows: list[dict]) -> Path:
    rows = sorted(rows, key=lambda r: (int(r['seed']), float(r['e_shape'])))
    table = ROOT / 'results/tables/p3c_multiseed_sweep_summary.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    preferred = ['run','variant_id','variant_name','seed','e_shape','e_shape_kernel','steps','train_crop','batch_size','gpu','reused_from_p3b','elapsed_sec_train_or_reuse','elapsed_sec_total','uefb_psnr','uefb_ssim','uefb_E_MAE','uefb_E_MAE_aligned','uefb_E_corr','uefb_identity_drop','uefb_q_ece','real_psnr','real_ssim','real_lee','real_nai','real_E_corr','synthetic_psnr','synthetic_ssim','synthetic_lee','synthetic_nai','synthetic_E_corr','run_dir','config','log']
    ordered = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with table.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader(); w.writerows(rows)
    (ROOT / 'results/tables/p3c_multiseed_sweep_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (ROOT / 'results/tables/p3c_multiseed_sweep_aggregate.json').write_text(json.dumps(aggregate(rows), indent=2), encoding='utf-8')
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
    jobs = planned_jobs()
    print(json.dumps({'P3C_START': True, 'jobs': jobs, 'max_steps': args.max_steps, 'parallel': args.parallel, 'resume': args.resume}, indent=2), flush=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(run_one, job, args.max_steps, args.train_crop, args.batch_size, i % args.parallel, args.resume) for i, job in enumerate(jobs)]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(json.dumps({'DONE_RUN': row['run'], 'seed': row['seed'], 'e_shape': row['e_shape'], 'reused': row['reused_from_p3b'], 'uefb_E_corr': row.get('uefb_E_corr'), 'real_psnr': row.get('real_psnr'), 'synthetic_psnr': row.get('synthetic_psnr')}, ensure_ascii=False), flush=True)
    table = write_summary(rows)
    print(json.dumps({'P3C_DONE': True, 'summary': str(table), 'aggregate': str(ROOT / 'results/tables/p3c_multiseed_sweep_aggregate.json'), 'rows': rows}, indent=2), flush=True)


if __name__ == '__main__':
    main()
