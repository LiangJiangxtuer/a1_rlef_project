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
    PY,
    _prefix_metrics,
    _run,
    paired_eval_specs,
    training_complete,
)
from scripts.run_phase2_dgb_candidate import (  # noqa: E402
    _p7b_near_miss_reference,
    should_promote_to_multiseed,
)
from scripts.run_p2b_controlled_isolation import _gauge_only_loss, _gauge_only_model  # noqa: E402


def _gate_floor_model(route_floor: float) -> dict:
    model = _gauge_only_model()
    model['route_floor'] = float(route_floor)
    model['route_type'] = 'identity_gate'
    return model


def _gate_floor_loss() -> dict:
    loss = _gauge_only_loss()
    loss.pop('distill', None)
    loss.pop('distill_by_dataset', None)
    loss.pop('rec_by_dataset', None)
    loss.pop('domain_head_anchor_by_dataset', None)
    return loss


def _e_shape_model() -> dict:
    model = _gauge_only_model()
    model['route_floor'] = 0.0
    model['route_type'] = 'identity_gate'
    return model


def _e_shape_loss(weight: float) -> dict:
    loss = _gate_floor_loss()
    loss['e_shape'] = float(weight)
    return loss


def planned_runs() -> list[dict]:
    return [
        {
            'id': 'P2C_DGB_GATE_FLOOR015',
            'name': 'image_stats_gauge_original_gate_floor015',
            'seed': 3407,
            'base_variant': 'P2B_DGB_GAUGE_ONLY',
            'intervention': 'global_route_floor',
            'model': _gate_floor_model(0.15),
            'loss': _gate_floor_loss(),
            'rationale': 'P2C UEFB recovery: global route floor 0.15 to reduce low-route conservatism without dataset-specific reweighting.',
        },
        {
            'id': 'P2C_DGB_GATE_FLOOR025',
            'name': 'image_stats_gauge_original_gate_floor025',
            'seed': 3407,
            'base_variant': 'P2B_DGB_GAUGE_ONLY',
            'intervention': 'global_route_floor',
            'model': _gate_floor_model(0.25),
            'loss': _gate_floor_loss(),
            'rationale': 'P2C UEFB recovery: stronger global route floor 0.25 to test whether more restoration contribution recovers UEFB.',
        },
        {
            'id': 'P2C_DGB_ESHAPE025',
            'name': 'image_stats_gauge_original_route_eshape025',
            'seed': 3407,
            'base_variant': 'P2B_DGB_GAUGE_ONLY',
            'intervention': 'reduced_e_shape_weight',
            'model': _e_shape_model(),
            'loss': _e_shape_loss(0.025),
            'rationale': 'P2C UEFB recovery: reduce gauge-invariant e_shape weight from 0.05 to 0.025 while keeping Gauge-only route unchanged.',
        },
        {
            'id': 'P2C_DGB_ESHAPE010',
            'name': 'image_stats_gauge_original_route_eshape010',
            'seed': 3407,
            'base_variant': 'P2B_DGB_GAUGE_ONLY',
            'intervention': 'reduced_e_shape_weight',
            'model': _e_shape_model(),
            'loss': _e_shape_loss(0.010),
            'rationale': 'P2C UEFB recovery: reduce gauge-invariant e_shape weight from 0.05 to 0.010 to test whether UEFB PSNR is shape-overconstrained.',
        },
    ]


def make_cfg(run: dict, max_steps: int = 1000, train_crop: int = 128, batch_size: int = 8) -> dict:
    return {
        'seed': int(run.get('seed', 3407)),
        'protocol': {
            'stage': 'P2C UEFB recovery from Gauge-only',
            'variant_id': run['id'],
            'variant_name': run['name'],
            'base_variant': run['base_variant'],
            'intervention': run['intervention'],
            'notes': run['rationale'],
            'phase2c_uefb_recovery': True,
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
            'phase2c_uefb_recovery': True,
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
        'forbidden': {
            'teacher_distill': True,
            'rec_by_dataset': True,
            'real_anchor_above_0p010': True,
        },
    }


def _phase2_reference() -> dict:
    path = ROOT / 'results/tables/phase2_dgb_candidate_summary.json'
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    runs = payload.get('runs') or []
    return runs[0] if runs else {}


def _p2b_gauge_reference() -> dict:
    path = ROOT / 'results/tables/p2b_controlled_isolation_summary.json'
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    for row in payload.get('runs', []):
        if row.get('variant_id') == 'P2B_DGB_GAUGE_ONLY':
            return row
    return {}


def run_one(run: dict, max_steps: int, train_crop: int, batch_size: int, gpu: int, resume: bool = False) -> dict:
    tag = f"p2c_{run['id'].lower()}_{run['name']}"
    cfg = make_cfg(run, max_steps=max_steps, train_crop=train_crop, batch_size=batch_size)
    cfg_dir = ROOT / 'configs/dgb_rlef'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f'{tag}.yml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    out = ROOT / 'experiments' / f'{tag}_seed{int(run.get("seed", 3407))}'
    log_path = ROOT / 'logs/p2c_uefb_recovery' / f'{tag}.log'
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
    if not training_complete(out):
        raise FileNotFoundError(f'missing checkpoint or UEFB metrics under {out}')
    row = {
        'run': tag,
        'variant_id': run['id'],
        'variant_name': run['name'],
        'base_variant': run['base_variant'],
        'intervention': run['intervention'],
        'seed': int(run.get('seed', 3407)),
        'backbone': run['model']['backbone'],
        'backbone_blocks': int(run['model']['backbone_blocks']),
        'gauge_mode': run['model']['gauge_mode'],
        'route_type': run['model']['route_type'],
        'route_floor': float(run['model'].get('route_floor', 0.0)),
        'safe_alpha': float(run['model']['safe_alpha']),
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
            cmd = [PY, 'scripts/eval_paired.py', '--config', cfg_path, '--checkpoint', ckpt, '--low_dir', spec['low_dir'], '--high_dir', spec['high_dir'], '--output_dir', pair_out, '--device', 'cuda', '--domain', spec['dataset']]
            if spec.get('max_images') is not None:
                cmd += ['--max_images', str(spec['max_images'])]
            _run(cmd, log_path, env)
        row.update(_prefix_metrics(spec['dataset'], json.loads((pair_out / 'metrics.json').read_text())))
    row['elapsed_sec_total'] = time.time() - start
    return row


def write_summary(rows: list[dict]) -> Path:
    order = {run['id']: i for i, run in enumerate(planned_runs())}
    rows = sorted(rows, key=lambda r: order.get(r['variant_id'], 999))
    table = ROOT / 'results/tables/p2c_uefb_recovery_summary.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    preferred = ['run','variant_id','variant_name','base_variant','intervention','seed','backbone','backbone_blocks','gauge_mode','route_type','route_floor','safe_alpha','steps','train_crop','batch_size','gpu','elapsed_sec_train','elapsed_sec_total','uefb_psnr','uefb_ssim','uefb_E_MAE','uefb_E_MAE_aligned','uefb_E_corr','uefb_identity_drop','uefb_q_ece','real_psnr','real_ssim','real_lee','real_nai','real_E_corr','real_identity_drop','real_q_ece','synthetic_psnr','synthetic_ssim','synthetic_lee','synthetic_nai','synthetic_E_corr','synthetic_identity_drop','synthetic_q_ece','run_dir','config','log']
    ordered = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with table.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader(); w.writerows(rows)
    p3c_path = ROOT / 'results/tables/p3c_multiseed_sweep_aggregate.json'
    p3c = json.loads(p3c_path.read_text())['multiseed_e_shape_0p05'] if p3c_path.exists() else {}
    p7b = _p7b_near_miss_reference()
    payload = {
        'runs': rows,
        'p3c_reference': p3c,
        'p7b_near_miss_reference': p7b,
        'phase2_dgb_reference': _phase2_reference(),
        'p2b_gauge_only_reference': _p2b_gauge_reference(),
        'promotion': [should_promote_to_multiseed(r, p3c, p7b) for r in rows] if p3c else [],
    }
    (ROOT / 'results/tables/p2c_uefb_recovery_summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_steps', type=int, default=1000)
    ap.add_argument('--train_crop', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--gpu_start', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if not (ROOT / 'data/UEFB-v2/train/meta.json').exists():
        raise FileNotFoundError('UEFB-v2 formal data missing')
    runs = planned_runs()
    print(json.dumps({'P2C_START': True, 'runs': [(r['id'], r['name']) for r in runs], 'max_steps': args.max_steps, 'parallel': args.parallel, 'gpu_start': args.gpu_start}, indent=2), flush=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futures = [ex.submit(run_one, run, args.max_steps, args.train_crop, args.batch_size, args.gpu_start + (i % max(1, args.parallel)), args.resume) for i, run in enumerate(runs)]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(json.dumps({'DONE_RUN': row['run'], 'route_floor': row.get('route_floor'), 'uefb_psnr': row.get('uefb_psnr'), 'real_psnr': row.get('real_psnr'), 'synthetic_psnr': row.get('synthetic_psnr'), 'uefb_E_corr': row.get('uefb_E_corr')}, ensure_ascii=False), flush=True)
    table = write_summary(rows)
    print(json.dumps({'P2C_DONE': True, 'summary': str(table), 'rows': rows}, indent=2), flush=True)


if __name__ == '__main__':
    main()
