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


def _dgb_model() -> dict:
    model = dict(P6_MODEL)
    model.update({
        'type': 'DGB_RLEF_MINIMAL',
        'q_branch': True,
        'gauge_mode': 'image_stats',
        'gauge_mu_min': -1.0,
        'gauge_mu_max': 2.5,
        'route_type': 'recoverability_safe_router',
        'safe_alpha': 0.70,
    })
    return model


def _dgb_loss() -> dict:
    loss = dict(BASE_M4J_ES_LOSS)
    loss['gauge'] = 0.0
    loss['gauge_schedule'] = {
        'ramp_start': 300,
        'full_start': 700,
        'max_weight': 0.005,
        'hard_cap': 0.010,
    }
    loss['e_shape'] = 0.05
    loss['e_shape_kernel'] = 7
    loss['e_shape_beta'] = 0.1
    loss['q'] = 0.02
    loss.pop('distill', None)
    loss.pop('distill_by_dataset', None)
    loss.pop('rec_by_dataset', None)
    loss.pop('domain_head_anchor_by_dataset', None)
    loss.pop('gate_identity', None)
    return loss


def planned_runs() -> list[dict]:
    return [
        {
            'id': 'DGB_RLEF_MINIMAL_S3407',
            'name': 'multiscale_image_stats_gauge_safe_router',
            'seed': 3407,
            'model': _dgb_model(),
            'loss': _dgb_loss(),
            'rationale': 'Phase 2 candidate: P6 multiscale trunk + P3c e_shape=0.05 + image-stat gauge head with warm gauge schedule + recoverability safe router. No teacher, no rec_by_dataset, no RA>0.010.',
        }
    ]


def make_cfg(run: dict, max_steps: int = 1000, train_crop: int = 128, batch_size: int = 8) -> dict:
    return {
        'seed': int(run.get('seed', 3407)),
        'protocol': {
            'stage': 'Phase 2 DGB-RLEF candidate training',
            'variant_id': run['id'],
            'variant_name': run['name'],
            'notes': run['rationale'],
            'phase2_candidate_training': True,
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
            'phase2_candidate_training': True,
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


def _p7b_near_miss_reference() -> dict:
    table = ROOT / 'results/tables/p7b_realanchor_summary.csv'
    if not table.exists():
        return {}
    with table.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get('variant_id') == 'P7B_DHEAD_RA010':
            return {
                'uefb_psnr': float(row['uefb_psnr']),
                'real_psnr': float(row['real_psnr']),
                'synthetic_psnr': float(row['synthetic_psnr']),
            }
    return {}


def should_promote_to_multiseed(row: dict, p3c_aggregate: dict, p7b_reference: dict | None = None) -> bool:
    p7b_reference = p7b_reference or {}
    for key in ['uefb_psnr', 'real_psnr', 'synthetic_psnr']:
        value = float(row.get(key, float('-inf')))
        if value <= float(p3c_aggregate[key]['mean']):
            return False
        if p7b_reference and value <= float(p7b_reference[key]):
            return False
    return True


def run_one(run: dict, max_steps: int, train_crop: int, batch_size: int, gpu: int, resume: bool = False) -> dict:
    tag = f"phase2_{run['id'].lower()}_{run['name']}"
    cfg = make_cfg(run, max_steps=max_steps, train_crop=train_crop, batch_size=batch_size)
    cfg_dir = ROOT / 'configs/dgb_rlef'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f'{tag}.yml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    out = ROOT / 'experiments' / f'{tag}_seed{int(run.get("seed", 3407))}'
    log_path = ROOT / 'logs/phase2_dgb_candidate' / f'{tag}.log'
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
        'seed': int(run.get('seed', 3407)),
        'backbone': run['model']['backbone'],
        'backbone_blocks': int(run['model']['backbone_blocks']),
        'gauge_mode': run['model']['gauge_mode'],
        'route_type': run['model']['route_type'],
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
    rows = sorted(rows, key=lambda r: r['variant_id'])
    table = ROOT / 'results/tables/phase2_dgb_candidate_summary.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    preferred = ['run','variant_id','variant_name','seed','backbone','backbone_blocks','gauge_mode','route_type','safe_alpha','steps','train_crop','batch_size','gpu','elapsed_sec_train','elapsed_sec_total','uefb_psnr','uefb_ssim','uefb_E_MAE','uefb_E_MAE_aligned','uefb_E_corr','uefb_identity_drop','uefb_q_ece','real_psnr','real_ssim','real_lee','real_nai','real_E_corr','real_identity_drop','real_q_ece','synthetic_psnr','synthetic_ssim','synthetic_lee','synthetic_nai','synthetic_E_corr','synthetic_identity_drop','synthetic_q_ece','run_dir','config','log']
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
        'promotion': [should_promote_to_multiseed(r, p3c, p7b) for r in rows] if p3c else [],
    }
    (ROOT / 'results/tables/phase2_dgb_candidate_summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
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
    print(json.dumps({'PHASE2_DGB_START': True, 'runs': [(r['id'], r['name']) for r in runs], 'max_steps': args.max_steps, 'parallel': args.parallel, 'gpu_start': args.gpu_start}, indent=2), flush=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futures = [ex.submit(run_one, run, args.max_steps, args.train_crop, args.batch_size, args.gpu_start + (i % max(1, args.parallel)), args.resume) for i, run in enumerate(runs)]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(json.dumps({'DONE_RUN': row['run'], 'uefb_psnr': row.get('uefb_psnr'), 'real_psnr': row.get('real_psnr'), 'synthetic_psnr': row.get('synthetic_psnr'), 'uefb_E_corr': row.get('uefb_E_corr')}, ensure_ascii=False), flush=True)
    table = write_summary(rows)
    print(json.dumps({'PHASE2_DGB_DONE': True, 'summary': str(table), 'rows': rows}, indent=2), flush=True)


if __name__ == '__main__':
    main()
