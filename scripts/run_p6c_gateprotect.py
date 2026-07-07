#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_p3c_multiseed_sweep import (  # noqa: E402
    BASE_M4J_ES_LOSS,
    PY,
    _prefix_metrics,
    _run,
    paired_eval_specs,
    training_complete,
)
from scripts.run_p6_structural_backbone import P6_MODEL  # noqa: E402


def _gateprotect_loss(gate: float = 0.02, gate_identity: float = 0.0) -> dict:
    loss = dict(BASE_M4J_ES_LOSS)
    loss['rec'] = 1.0
    loss['gate'] = float(gate)
    loss.pop('rec_by_dataset', None)
    loss.pop('distill', None)
    loss.pop('distill_by_dataset', None)
    if gate_identity > 0:
        loss['gate_identity'] = float(gate_identity)
        loss['gate_identity_tau'] = 0.05
    return loss


def _model_with_blocks(blocks: int) -> dict:
    model = dict(P6_MODEL)
    model['backbone_blocks'] = int(blocks)
    return model


def planned_runs() -> list[dict]:
    return [
        {
            'id': 'P6C_MS_GATEID005',
            'name': 'm4j_eshape_multiscale_gate_identity005',
            'seed': 3407,
            'model': _model_with_blocks(3),
            'loss': _gateprotect_loss(gate=0.02, gate_identity=0.005),
            'gate_weight': 0.02,
            'gate_identity_weight': 0.005,
            'rationale': 'P6 multiscale backbone with light identity-weighted A-gate penalty; no teacher/output distillation and no synthetic-rec upweighting.',
        },
        {
            'id': 'P6C_MS_GATEID010',
            'name': 'm4j_eshape_multiscale_gate_identity010',
            'seed': 3407,
            'model': _model_with_blocks(3),
            'loss': _gateprotect_loss(gate=0.02, gate_identity=0.010),
            'gate_weight': 0.02,
            'gate_identity_weight': 0.010,
            'rationale': 'P6 multiscale backbone with stronger identity-weighted A-gate penalty; no teacher/output distillation and no synthetic-rec upweighting.',
        },
        {
            'id': 'P6C_MS_GATELOW005',
            'name': 'm4j_eshape_multiscale_gate_weight005',
            'seed': 3407,
            'model': _model_with_blocks(3),
            'loss': _gateprotect_loss(gate=0.005, gate_identity=0.0),
            'gate_weight': 0.005,
            'gate_identity_weight': 0.0,
            'rationale': 'P6 multiscale backbone with lower supervised gate loss; no teacher/output distillation and no synthetic-rec upweighting.',
        },
        {
            'id': 'P6C_MS_B2_GATEID005',
            'name': 'm4j_eshape_multiscale_blocks2_gate_identity005',
            'seed': 3407,
            'model': _model_with_blocks(2),
            'loss': _gateprotect_loss(gate=0.02, gate_identity=0.005),
            'gate_weight': 0.02,
            'gate_identity_weight': 0.005,
            'rationale': 'Slightly smaller P6 multiscale backbone plus light identity-weighted A-gate penalty; no teacher/output distillation and no synthetic-rec upweighting.',
        },
    ]


def make_cfg(run: dict, max_steps: int = 1000, train_crop: int = 128, batch_size: int = 8) -> dict:
    return {
        'seed': int(run.get('seed', 3407)),
        'protocol': {
            'stage': 'P6c gate-protection structural route',
            'variant_id': run['id'],
            'variant_name': run['name'],
            'gate_weight': float(run['gate_weight']),
            'gate_identity_weight': float(run['gate_identity_weight']),
            'backbone_blocks': int(run['model']['backbone_blocks']),
            'notes': 'Keep P6 multiscale structural route. Protect synthetic via A-gate aggressiveness/capacity controls, not teacher distillation and not dataset-weighted reconstruction.',
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


def should_promote_to_multiseed(row: dict, p3c_aggregate: dict) -> bool:
    return (
        float(row.get('uefb_psnr', float('-inf'))) > float(p3c_aggregate['uefb_psnr']['mean'])
        and float(row.get('real_psnr', float('-inf'))) > float(p3c_aggregate['real_psnr']['mean'])
        and float(row.get('synthetic_psnr', float('-inf'))) > float(p3c_aggregate['synthetic_psnr']['mean'])
    )


def gpu_for_run(index: int, parallel: int, gpu_start: int = 0) -> int:
    return int(gpu_start) + (int(index) % max(1, int(parallel)))


def run_one(run: dict, max_steps: int, train_crop: int, batch_size: int, gpu: int, resume: bool = False) -> dict:
    tag = f"p6c_{run['id'].lower()}_{run['name']}"
    cfg = make_cfg(run, max_steps=max_steps, train_crop=train_crop, batch_size=batch_size)
    cfg_dir = ROOT / 'configs/p6c_gateprotect'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f'{tag}.yml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    out = ROOT / 'experiments' / f'{tag}_seed{int(run.get("seed", 3407))}'
    log_path = ROOT / 'logs/p6c_gateprotect' / f'{tag}.log'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    start = time.time()
    if resume and training_complete(out):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'RESUME: found existing checkpoint and UEFB metrics under {out}\n')
    else:
        _run([PY, 'scripts/train.py', '--config', cfg_path, '--output_dir', out, '--device', 'cuda', '--max_steps', str(max_steps)], log_path, env)
    if not training_complete(out):
        raise FileNotFoundError(f'missing checkpoint or UEFB metrics under {out}')
    row = {
        'run': tag,
        'variant_id': run['id'],
        'variant_name': run['name'],
        'seed': int(run.get('seed', 3407)),
        'backbone': run['model']['backbone'],
        'backbone_blocks': int(run['model']['backbone_blocks']),
        'gate_weight': float(run['gate_weight']),
        'gate_identity_weight': float(run['gate_identity_weight']),
        'steps': max_steps,
        'train_crop': train_crop,
        'batch_size': batch_size,
        'gpu': gpu,
        'elapsed_sec_train': time.time() - start,
        'run_dir': str(out),
        'config': str(cfg_path),
        'log': str(log_path),
    }
    row.update(_prefix_metrics('uefb', json.loads((out / 'metrics/eval_metrics.json').read_text())))
    ckpt = out / 'checkpoints/last.pth'
    for spec in paired_eval_specs():
        pair_out = out / 'paired_eval' / spec['dataset']
        if (pair_out / 'metrics.json').exists() and resume:
            with log_path.open('a', encoding='utf-8') as log:
                log.write(f'REUSE_EVAL: {pair_out / "metrics.json"}\n')
        else:
            cmd = [PY, 'scripts/eval_paired.py', '--config', cfg_path, '--checkpoint', ckpt, '--low_dir', spec['low_dir'], '--high_dir', spec['high_dir'], '--output_dir', pair_out, '--device', 'cuda']
            if spec.get('max_images') is not None:
                cmd += ['--max_images', str(spec['max_images'])]
            _run(cmd, log_path, env)
        row.update(_prefix_metrics(spec['dataset'], json.loads((pair_out / 'metrics.json').read_text())))
    row['elapsed_sec_total'] = time.time() - start
    return row


def write_summary(rows: list[dict]) -> Path:
    order = {run['id']: i for i, run in enumerate(planned_runs())}
    rows = sorted(rows, key=lambda r: order.get(r['variant_id'], 999))
    table = ROOT / 'results/tables/p6c_gateprotect_summary.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    preferred = ['run','variant_id','variant_name','seed','backbone','backbone_blocks','gate_weight','gate_identity_weight','steps','train_crop','batch_size','gpu','elapsed_sec_train','elapsed_sec_total','uefb_psnr','uefb_ssim','uefb_E_MAE','uefb_E_MAE_aligned','uefb_E_corr','real_psnr','real_ssim','real_E_corr','synthetic_psnr','synthetic_ssim','synthetic_E_corr','run_dir','config','log']
    ordered = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with table.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader(); w.writerows(rows)
    p3c_path = ROOT / 'results/tables/p3c_multiseed_sweep_aggregate.json'
    p3c = json.loads(p3c_path.read_text())['multiseed_e_shape_0p05'] if p3c_path.exists() else {}
    payload = {'runs': rows, 'promotion': [should_promote_to_multiseed(r, p3c) for r in rows] if p3c else []}
    (ROOT / 'results/tables/p6c_gateprotect_summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_steps', type=int, default=1000)
    ap.add_argument('--train_crop', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--gpu_start', type=int, default=0, help='First physical CUDA device index for child runs; use 1 to skip a busy GPU 0.')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if not (ROOT / 'data/UEFB-v2/train/meta.json').exists():
        raise FileNotFoundError('UEFB-v2 formal data missing')
    runs = planned_runs()
    print(json.dumps({'P6C_START': True, 'runs': [(r['id'], r['name']) for r in runs], 'max_steps': args.max_steps, 'parallel': args.parallel, 'gpu_start': args.gpu_start}, indent=2), flush=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futures = [ex.submit(run_one, run, args.max_steps, args.train_crop, args.batch_size, gpu_for_run(i, args.parallel, args.gpu_start), args.resume) for i, run in enumerate(runs)]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(json.dumps({'DONE_RUN': row['run'], 'gate_weight': row.get('gate_weight'), 'gate_identity_weight': row.get('gate_identity_weight'), 'uefb_psnr': row.get('uefb_psnr'), 'real_psnr': row.get('real_psnr'), 'synthetic_psnr': row.get('synthetic_psnr')}, ensure_ascii=False), flush=True)
    table = write_summary(rows)
    print(json.dumps({'P6C_DONE': True, 'summary': str(table), 'rows': rows}, indent=2), flush=True)


if __name__ == '__main__':
    main()
