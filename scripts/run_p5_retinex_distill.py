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
RETINEX_ROOT = ROOT / 'external_baselines/Retinexformer'

BASE_M4_ES_MODEL = {
    'base_channels': 24,
    'e_max': 3.5,
    'exposure_branch': True,
    'adaptive_gauge': True,
    'fixed_gauge': None,
    'physics_branch': True,
    'gate_branch': True,
    'q_branch': False,
}

BASE_M4_ES_LOSS = {
    'rec': 0.70,
    'distill': 0.30,
    'phys': 0.15,
    'poisson': 0.05,
    'gauge': 0.10,
    'id': 0.02,
    'gate': 0.02,
    'wtv': 0.02,
    'e_shape': 0.05,
    'e_shape_kernel': 7,
}


def planned_runs() -> list[dict]:
    loss_t01 = dict(BASE_M4_ES_LOSS)
    loss_t01['distill'] = 0.10
    loss_t03 = dict(BASE_M4_ES_LOSS)
    loss_t03['distill'] = 0.30
    return [
        {
            'id': 'P5_RD_T01',
            'name': 'm4j_eshape_retinexformer_distill_t01',
            'model': dict(BASE_M4_ES_MODEL),
            'loss': loss_t01,
            'teacher': 'Retinexformer LOL-v2 official pretrained outputs on train splits',
        },
        {
            'id': 'P5_RD_T03',
            'name': 'm4j_eshape_retinexformer_distill_t03',
            'model': dict(BASE_M4_ES_MODEL),
            'loss': loss_t03,
            'teacher': 'Retinexformer LOL-v2 official pretrained outputs on train splits',
        }
    ]


def teacher_tag(spec: dict) -> str:
    return f"retinexformer_train_{spec['dataset']}"


def teacher_export_specs() -> list[dict]:
    return [
        {
            'dataset': 'real',
            'low_dir': str(ROOT / 'data/LOL-v2/Real_captured/Train/Low'),
            'high_dir': str(ROOT / 'data/LOL-v2/Real_captured/Train/Normal'),
            'base_opt': str(RETINEX_ROOT / 'Options/RetinexFormer_LOL_v2_real.yml'),
            'weights': str(RETINEX_ROOT / 'pretrained_weights/LOL_v2_real.pth'),
            'dataset_arg': 'LOL_v2_real_train_teacher',
            'output_dir': str(ROOT / 'experiments/p5_retinexformer_train_teacher/real'),
        },
        {
            'dataset': 'synthetic',
            'low_dir': str(ROOT / 'data/LOL-v2/Synthetic/Train/Low'),
            'high_dir': str(ROOT / 'data/LOL-v2/Synthetic/Train/Normal'),
            'base_opt': str(RETINEX_ROOT / 'Options/RetinexFormer_LOL_v2_synthetic.yml'),
            'weights': str(RETINEX_ROOT / 'pretrained_weights/LOL_v2_synthetic.pth'),
            'dataset_arg': 'LOL_v2_synthetic_train_teacher',
            'output_dir': str(ROOT / 'experiments/p5_retinexformer_train_teacher/synthetic'),
        },
    ]


def _image_count(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for x in p.iterdir() if x.is_file() and x.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'})


def teacher_outputs_complete(spec: dict) -> bool:
    return _image_count(spec['output_dir']) >= _image_count(spec['low_dir']) > 0


def write_teacher_option(spec: dict) -> Path:
    base = yaml.safe_load(Path(spec['base_opt']).read_text(encoding='utf-8'))
    if 'datasets' not in base or 'val' not in base['datasets']:
        raise KeyError(f"Retinexformer option lacks datasets.val: {spec['base_opt']}")
    base['datasets']['val']['dataroot_lq'] = spec['low_dir']
    base['datasets']['val']['dataroot_gt'] = spec['high_dir']
    out_dir = ROOT / 'configs/p5_retinex_teacher'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{teacher_tag(spec)}.yml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding='utf-8')
    return path


def _run(cmd: list[str | Path], log_path: Path, env: dict, cwd: Path = ROOT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write('+ ' + ' '.join(map(str, cmd)) + '\n')
        log.flush()
        proc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'command failed with code {proc.returncode}; see {log_path}')


def export_teacher(spec: dict, gpu: int = 0, resume: bool = False) -> dict:
    cfg = write_teacher_option(spec)
    out_dir = Path(spec['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / 'logs/p5_retinex_teacher' / f"{teacher_tag(spec)}.log"
    if teacher_outputs_complete(spec) and resume:
        return {'dataset': spec['dataset'], 'status': 'skipped_complete', 'output_dir': str(out_dir), 'n': _image_count(out_dir), 'log': str(log_path)}
    env = os.environ.copy()
    env['PYTHONPATH'] = str(RETINEX_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [
        PY,
        'Enhancement/test_from_dataset.py',
        '--opt', cfg,
        '--weights', spec['weights'],
        '--dataset', spec['dataset_arg'],
        '--gpus', str(gpu),
        '--output_dir', out_dir,
        '--result_dir', str(ROOT / 'experiments/p5_retinexformer_train_teacher/retinexformer_raw_results'),
    ]
    _run(cmd, log_path, env, cwd=RETINEX_ROOT)
    return {'dataset': spec['dataset'], 'status': 'exported', 'output_dir': str(out_dir), 'n': _image_count(out_dir), 'expected': _image_count(spec['low_dir']), 'log': str(log_path)}


def paired_eval_specs() -> list[dict]:
    return [
        {
            'dataset': 'real',
            'low_dir': str(ROOT / 'data/LOL-v2/Real_captured/Test/Low'),
            'high_dir': str(ROOT / 'data/LOL-v2/Real_captured/Test/Normal'),
        },
        {
            'dataset': 'synthetic',
            'low_dir': str(ROOT / 'data/LOL-v2/Synthetic/Test/Low'),
            'high_dir': str(ROOT / 'data/LOL-v2/Synthetic/Test/Normal'),
        },
    ]


def make_cfg(run: dict, max_steps: int = 1000, train_crop: int = 128, batch_size: int = 8) -> dict:
    specs = {s['dataset']: s for s in teacher_export_specs()}
    return {
        'seed': 3407,
        'protocol': {
            'stage': 'P5 Retinexformer teacher distillation',
            'variant_id': run['id'],
            'variant_name': run['name'],
            'notes': 'Keep RLEF M4J_ES heads; distill paired LOL-v2 training samples from official Retinexformer outputs while UEFB remains supervised by GT.',
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
                {'type': 'paired_rgb', 'low_dir': str(ROOT / 'data/LOL-v2/Real_captured/Train/Low'), 'high_dir': str(ROOT / 'data/LOL-v2/Real_captured/Train/Normal'), 'teacher_dir': specs['real']['output_dir'], 'crop_size': train_crop, 'augment': True, 'max_images': None, 'name': 'LOL-v2-real-train-retinex-teacher'},
                {'type': 'paired_rgb', 'low_dir': str(ROOT / 'data/LOL-v2/Synthetic/Train/Low'), 'high_dir': str(ROOT / 'data/LOL-v2/Synthetic/Train/Normal'), 'teacher_dir': specs['synthetic']['output_dir'], 'crop_size': train_crop, 'augment': True, 'max_images': None, 'name': 'LOL-v2-synthetic-train-retinex-teacher'},
            ],
            'val': {'type': 'uefb', 'root': str(ROOT / 'data/UEFB-v2/test'), 'crop_size': None, 'augment': False, 'max_images': None},
        },
        'loss': run['loss'],
    }


def _prefix_metrics(prefix: str, metrics: dict) -> dict:
    return {f'{prefix}_{k}': v for k, v in metrics.items()}


def training_complete(run_dir: Path) -> bool:
    return (run_dir / 'checkpoints/last.pth').exists() and (run_dir / 'metrics/eval_metrics.json').exists()


def run_one(run: dict, max_steps: int, train_crop: int, batch_size: int, gpu: int, resume: bool = False) -> dict:
    tag = f"p5_{run['id'].lower()}_{run['name']}"
    cfg = make_cfg(run, max_steps=max_steps, train_crop=train_crop, batch_size=batch_size)
    cfg_dir = ROOT / 'configs/p5_retinex_distill'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f'{tag}.yml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    out = ROOT / 'experiments' / f'{tag}_seed3407'
    log_path = ROOT / 'logs/p5_retinex_distill' / f'{tag}.log'
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
        _run([PY, 'scripts/eval_paired.py', '--config', cfg_path, '--checkpoint', ckpt, '--low_dir', spec['low_dir'], '--high_dir', spec['high_dir'], '--output_dir', pair_out, '--device', 'cuda'], log_path, env)
        row.update(_prefix_metrics(spec['dataset'], json.loads((pair_out / 'metrics.json').read_text())))
    row['elapsed_sec_total'] = time.time() - start
    return row


def write_summary(rows: list[dict], teacher_rows: list[dict]) -> Path:
    table = ROOT / 'results/tables/p5_retinex_distill_summary.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    preferred = ['run','variant_id','variant_name','steps','train_crop','batch_size','gpu','elapsed_sec_train','elapsed_sec_total','uefb_psnr','uefb_ssim','uefb_E_MAE','uefb_E_MAE_aligned','uefb_E_corr','uefb_identity_drop','uefb_q_ece','real_psnr','real_ssim','real_lee','real_nai','synthetic_psnr','synthetic_ssim','synthetic_lee','synthetic_nai','run_dir','log']
    ordered = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with table.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader(); w.writerows(rows)
    payload = {'teacher_exports': teacher_rows, 'runs': rows}
    (ROOT / 'results/tables/p5_retinex_distill_summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_steps', type=int, default=1000)
    ap.add_argument('--train_crop', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--skip_teacher_export', action='store_true')
    args = ap.parse_args()
    if not (ROOT / 'data/UEFB-v2/train/meta.json').exists():
        raise FileNotFoundError('UEFB-v2 formal data missing')
    print(json.dumps({'P5_START': True, 'runs': [(r['id'], r['name']) for r in planned_runs()], 'max_steps': args.max_steps, 'resume': args.resume}, indent=2), flush=True)
    teacher_rows: list[dict] = []
    if args.skip_teacher_export:
        for spec in teacher_export_specs():
            if not teacher_outputs_complete(spec):
                raise FileNotFoundError(f"teacher outputs missing/incomplete for {spec['dataset']}: {spec['output_dir']}")
            teacher_rows.append({'dataset': spec['dataset'], 'status': 'already_present', 'output_dir': spec['output_dir'], 'n': _image_count(spec['output_dir'])})
    else:
        for i, spec in enumerate(teacher_export_specs()):
            row = export_teacher(spec, gpu=i % 2, resume=args.resume)
            teacher_rows.append(row)
            print(json.dumps({'TEACHER_DONE': row}, indent=2), flush=True)
    rows: list[dict] = []
    runs = planned_runs()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futures = [ex.submit(run_one, run, args.max_steps, args.train_crop, args.batch_size, i % max(1, args.parallel), args.resume) for i, run in enumerate(runs)]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(json.dumps({'DONE_RUN': row['run'], 'uefb_psnr': row.get('uefb_psnr'), 'uefb_E_corr': row.get('uefb_E_corr'), 'real_psnr': row.get('real_psnr'), 'synthetic_psnr': row.get('synthetic_psnr')}, ensure_ascii=False), flush=True)
    table = write_summary(rows, teacher_rows)
    print(json.dumps({'P5_DONE': True, 'summary': str(table), 'teacher_exports': teacher_rows, 'rows': rows}, indent=2), flush=True)


if __name__ == '__main__':
    main()
