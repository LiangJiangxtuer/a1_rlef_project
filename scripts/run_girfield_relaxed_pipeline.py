#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

SOURCE_GUIDANCE = Path('/home/user/下载/A1_after_master_prompt_QA_experiment_audit_next_top_tier_innovations_20260706.md')
OUT_ROOT = ROOT / 'results/girfield_relaxed'
DOC_SOURCE_DIR = ROOT / 'docs/source'
DOC_REPORT = ROOT / 'docs/GIR_FIELD_RELAXED_PIPELINE_REPORT.md'


def relaxed_protocol() -> dict[str, Any]:
    """Relaxed experimental protocol requested by the user.

    This intentionally performs no new main-model training and does not revive
    DGB/P2F. It converts the audited next-route guidance into a bounded,
    executable evidence pipeline using frozen checkpoints and existing official
    baseline outputs.
    """
    return {
        'name': 'GIR-Field relaxed evidence pipeline',
        'claim_scope': 'relaxed-routing-evidence-not-paper-grade-final',
        'no_main_training': True,
        'dgb_revival_allowed': False,
        'uses_frozen_checkpoints': True,
        'max_images_per_split': 64,
        'bootstrap_samples': 1000,
        'risk_patch_size': 32,
        'risk_max_patches': 8000,
        'lpips_policy': 'not_computed_if_package_missing; do_not_fabricate',
        'brisque_policy': 'appendix_only_if_available; do_not_fabricate',
        'external_baseline_policy': 'reuse locally verified Retinexformer/Zero-DCE++/KinD++ artifacts; no web SOTA update in relaxed run',
    }


def planned_phases() -> list[dict[str, str]]:
    return [
        {'id': 'N0', 'name': 'evidence_cleaning_and_freeze'},
        {'id': 'N1', 'name': 'p3c_m4j_m4j_es_statistical_reaudit'},
        {'id': 'N2', 'name': 'uefbg_gauge_perturbation_benchmark'},
        {'id': 'N3', 'name': 'external_baseline_registry_and_output_only_eval'},
        {'id': 'N4', 'name': 'recoverability_risk_calibration_probe'},
        {'id': 'N5', 'name': 'paper_grade_tables_figures_claim_report'},
    ]


def planned_outputs(root: Path = ROOT) -> dict[str, Path]:
    out = root / 'results/girfield_relaxed'
    return {
        'source_copy': root / 'docs/source/A1_after_master_prompt_QA_experiment_audit_next_top_tier_innovations_20260706.md',
        'execution_plan_md': root / 'docs/GIR_FIELD_RELAXED_EXECUTION_PLAN_ZH.md',
        'evidence_manifest_csv': out / 'N0_evidence/evidence_manifest.csv',
        'evidence_manifest_json': out / 'N0_evidence/evidence_manifest.json',
        'claim_ledger_csv': out / 'N0_evidence/claim_ledger.csv',
        'claim_ledger_json': out / 'N0_evidence/claim_ledger.json',
        'per_image_csv': out / 'N1_statistics/per_image_metrics.csv',
        'per_image_json': out / 'N1_statistics/per_image_metrics.json',
        'summary_csv': out / 'N1_statistics/variant_dataset_summary.csv',
        'summary_json': out / 'N1_statistics/variant_dataset_summary.json',
        'stat_tests_csv': out / 'N1_statistics/statistical_tests.csv',
        'stat_tests_json': out / 'N1_statistics/statistical_tests.json',
        'gauge_perturbation_csv': out / 'N2_uefbg/gauge_perturbation_metrics.csv',
        'gauge_perturbation_json': out / 'N2_uefbg/gauge_perturbation_metrics.json',
        'uefbg_metadata_csv': out / 'N2_uefbg/uefbg_metadata.csv',
        'external_registry_csv': out / 'N3_external/external_baseline_registry.csv',
        'external_registry_json': out / 'N3_external/external_baseline_registry.json',
        'risk_patch_csv': out / 'N4_risk/risk_patch_dataset.csv',
        'risk_calibration_csv': out / 'N4_risk/risk_calibration_summary.csv',
        'risk_calibration_json': out / 'N4_risk/risk_calibration_summary.json',
        'main_table_csv': out / 'N5_paper/girfield_main_table.csv',
        'main_table_json': out / 'N5_paper/girfield_main_table.json',
        'main_table_md': out / 'N5_paper/girfield_main_table.md',
        'stat_figure': out / 'N5_paper/fig_psnr_vs_scorr.png',
        'gauge_figure': out / 'N5_paper/fig_gauge_perturbation_bar.png',
        'risk_figure': out / 'N5_paper/fig_risk_reliability.png',
        'failure_grid': out / 'N5_paper/fig_psnr_misranking_failure_grid.png',
        'report_md': root / 'docs/GIR_FIELD_RELAXED_PIPELINE_REPORT.md',
        'manifest_json': out / 'GIR_FIELD_RELAXED_PIPELINE_MANIFEST.json',
        'validation_md': out / 'GIR_FIELD_RELAXED_VALIDATION.md',
    }


def planned_variants(root: Path = ROOT) -> list[dict[str, Any]]:
    return [
        {
            'id': 'M4', 'display': 'M4 A-gate', 'seed': 3407, 'e_shape': None,
            'role': 'counterexample',
            'config': root / 'configs/p3_formal/p3formal_m4_gate.yml',
            'checkpoint': root / 'experiments/p3formal_m4_gate_seed3407/checkpoints/last.pth',
            'selection_rule': 'P3 formal M4 single-seed: PSNR-high but exposure-shape wrong counterexample.',
        },
        {
            'id': 'M4J', 'display': 'M4J joint', 'seed': 3407, 'e_shape': None,
            'role': 'joint_baseline',
            'config': root / 'configs/p3b_joint/p3b_m4j_m4_joint.yml',
            'checkpoint': root / 'experiments/p3b_m4j_m4_joint_seed3407/checkpoints/last.pth',
            'selection_rule': 'Joint-training baseline without centered E-shape.',
        },
        {
            'id': 'M4J_ES', 'display': 'M4J_ES seed3407', 'seed': 3407, 'e_shape': 0.05,
            'role': 'centered_shape_positive',
            'config': root / 'configs/p3b_joint/p3b_m4j_es_m4_joint_eshape.yml',
            'checkpoint': root / 'experiments/p3b_m4j_es_m4_joint_eshape_seed3407/checkpoints/last.pth',
            'selection_rule': 'Centered E-shape positive evidence; reused as P3c seed3407/e=0.05 default checkpoint.',
        },
        {
            'id': 'P3C_E0050_S42', 'display': 'P3c e=0.05 seed42', 'seed': 42, 'e_shape': 0.05,
            'role': 'default_3seed',
            'config': root / 'configs/p3c_multiseed_sweep/p3c_m4j_es_seed42_e0050.yml',
            'checkpoint': root / 'experiments/p3c_m4j_es_seed42_e0050/checkpoints/last.pth',
            'selection_rule': 'P3c e=0.05 3-seed stability member.',
        },
        {
            'id': 'P3C_E0050_S2027', 'display': 'P3c e=0.05 seed2027', 'seed': 2027, 'e_shape': 0.05,
            'role': 'default_3seed',
            'config': root / 'configs/p3c_multiseed_sweep/p3c_m4j_es_seed2027_e0050.yml',
            'checkpoint': root / 'experiments/p3c_m4j_es_seed2027_e0050/checkpoints/last.pth',
            'selection_rule': 'P3c e=0.05 3-seed stability member.',
        },
        {
            'id': 'P3C_E0050_S3407', 'display': 'P3c e=0.05 seed3407', 'seed': 3407, 'e_shape': 0.05,
            'role': 'default_3seed',
            'config': root / 'configs/p3c_multiseed_sweep/p3c_m4j_es_seed3407_e0050.yml',
            'checkpoint': root / 'experiments/p3b_m4j_es_m4_joint_eshape_seed3407/checkpoints/last.pth',
            'selection_rule': 'P3c e=0.05 seed3407 reuses P3b M4J_ES checkpoint by design.',
        },
        {
            'id': 'P3D_E0100_S42', 'display': 'P3d e=0.10 seed42', 'seed': 42, 'e_shape': 0.10,
            'role': 'stronger_shape_control',
            'config': root / 'configs/p3d_e010_multiseed/p3d_m4j_es_seed42_e0100.yml',
            'checkpoint': root / 'experiments/p3d_m4j_es_seed42_e0100/checkpoints/last.pth',
            'selection_rule': 'P3d e=0.10 stronger E-shape weight trade-off control.',
        },
        {
            'id': 'P3D_E0100_S2027', 'display': 'P3d e=0.10 seed2027', 'seed': 2027, 'e_shape': 0.10,
            'role': 'stronger_shape_control',
            'config': root / 'configs/p3d_e010_multiseed/p3d_m4j_es_seed2027_e0100.yml',
            'checkpoint': root / 'experiments/p3d_m4j_es_seed2027_e0100/checkpoints/last.pth',
            'selection_rule': 'P3d e=0.10 stronger E-shape weight trade-off control.',
        },
        {
            'id': 'P3D_E0100_S3407', 'display': 'P3d e=0.10 seed3407', 'seed': 3407, 'e_shape': 0.10,
            'role': 'stronger_shape_control',
            'config': root / 'configs/p3d_e010_multiseed/p3d_m4j_es_seed3407_e0100.yml',
            'checkpoint': root / 'experiments/p3c_m4j_es_seed3407_e0100/checkpoints/last.pth',
            'selection_rule': 'P3d e=0.10 seed3407 reuses P3c e=0.10 checkpoint.',
        },
    ]


def dataset_specs(root: Path = ROOT) -> list[dict[str, Any]]:
    return [
        {'id': 'uefb', 'display': 'UEFB-v2-test', 'type': 'uefb', 'root': root / 'data/UEFB-v2/test'},
        {'id': 'real', 'display': 'LOL-v2-real-test', 'type': 'paired', 'low_dir': root / 'data/LOL-v2/Real_captured/Test/Low', 'high_dir': root / 'data/LOL-v2/Real_captured/Test/Normal', 'domain': 'real'},
        {'id': 'synthetic', 'display': 'LOL-v2-synthetic-test', 'type': 'paired', 'low_dir': root / 'data/LOL-v2/Synthetic/Test/Low', 'high_dir': root / 'data/LOL-v2/Synthetic/Test/Normal', 'domain': 'synthetic'},
    ]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    ensure_parent(path)
    if fields is None:
        preferred: list[str] = []
        for row in rows:
            for k in row.keys():
                if k not in preferred:
                    preferred.append(k)
        fields = preferred
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fields})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_float(x: Any) -> float | None:
    if x is None or x == '':
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return float(np.mean(vals)) if vals else None


def std(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0 if vals else None


def compute_gauge_metrics_np(e_pred: np.ndarray, e_gt: np.ndarray) -> dict[str, float]:
    ep = np.asarray(e_pred, dtype=np.float64)
    eg = np.asarray(e_gt, dtype=np.float64)
    if ep.ndim == 2:
        ep = ep[None, ...]
    if eg.ndim == 2:
        eg = eg[None, ...]
    if ep.shape != eg.shape:
        raise ValueError(f'shape mismatch: {ep.shape} vs {eg.shape}')
    mu_p = float(ep.mean())
    mu_g = float(eg.mean())
    sp = ep - mu_p
    sg = eg - mu_g
    denom = float(np.linalg.norm(sp.ravel()) * np.linalg.norm(sg.ravel()) + 1e-12)
    s_corr = float(np.dot(sp.ravel(), sg.ravel()) / denom)
    return {
        'e_mae': float(np.mean(np.abs(ep - eg))),
        's_mae': float(np.mean(np.abs(sp - sg))),
        'gauge_mae': float(abs(mu_p - mu_g)),
        'mu_pred': mu_p,
        'mu_gt': mu_g,
        's_corr': s_corr,
    }


def apply_bh_fdr(pvals: list[float | None]) -> list[float | None]:
    indexed = [(i, float(p)) for i, p in enumerate(pvals) if p is not None and not math.isnan(float(p))]
    out: list[float | None] = [None for _ in pvals]
    if not indexed:
        return out
    indexed_sorted = sorted(indexed, key=lambda x: x[1])
    m = len(indexed_sorted)
    prev = 1.0
    q_by_idx: dict[int, float] = {}
    for rank_from_end, (i, p) in enumerate(reversed(indexed_sorted), start=1):
        rank = m - rank_from_end + 1
        q = min(prev, p * m / rank)
        prev = q
        q_by_idx[i] = min(1.0, max(p, q))
    for i, q in q_by_idx.items():
        out[i] = q
    return out


def torch_gauge_metrics(e_pred, e_gt) -> dict[str, float]:
    import torch
    import torch.nn.functional as F
    if e_gt.shape[-2:] != e_pred.shape[-2:]:
        e_gt = F.interpolate(e_gt, size=e_pred.shape[-2:], mode='bilinear', align_corners=False)
    ep = e_pred.detach()
    eg = e_gt.detach()
    mu_p = ep.mean(dim=(1, 2, 3))
    mu_g = eg.mean(dim=(1, 2, 3))
    sp = ep - mu_p.view(-1, 1, 1, 1)
    sg = eg - mu_g.view(-1, 1, 1, 1)
    denom = sp.flatten(1).norm(dim=1) * sg.flatten(1).norm(dim=1) + 1e-8
    s_corr = ((sp.flatten(1) * sg.flatten(1)).sum(dim=1) / denom).mean()
    return {
        'E_MAE': float((ep - eg).abs().mean().cpu()),
        'S_MAE': float((sp - sg).abs().mean().cpu()),
        'Gauge_MAE': float((mu_p - mu_g).abs().mean().cpu()),
        'mu_pred': float(mu_p.mean().cpu()),
        'mu_gt': float(mu_g.mean().cpu()),
        'S_corr': float(s_corr.cpu()),
        'E_corr': float(s_corr.cpu()),
        'E_MAE_aligned': float((sp - sg).abs().mean().cpu()),
    }


def make_dataloader(spec: dict[str, Any], max_images: int | None):
    from torch.utils.data import DataLoader
    from rlef.datasets.uefb_dataset import UEFBPairedDataset
    from rlef.datasets.paired_rgb_dataset import PairedRGBDataset
    if spec['type'] == 'uefb':
        ds = UEFBPairedDataset(spec['root'], max_images=max_images)
    else:
        ds = PairedRGBDataset(spec['low_dir'], spec['high_dir'], max_images=max_images, name=spec.get('domain'))
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


def load_model(variant: dict[str, Any], device: str):
    import torch
    import yaml
    from scripts.train import build_model
    cfg = yaml.safe_load(variant['config'].read_text(encoding='utf-8'))
    model = build_model(cfg).to(device)
    ckpt = torch.load(variant['checkpoint'], map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def evaluate_variant(variant: dict[str, Any], spec: dict[str, Any], device: str, max_images: int | None, visual_dir: Path | None = None) -> list[dict[str, Any]]:
    import torch
    from rlef.metrics.full_reference import psnr_torch, ssim_torch
    from rlef.metrics.exposure_field import local_exposure_error, noise_amplification_index, saturation_rate, identity_drop, q_ece, normalized_abs_error
    from rlef.utils.image_io import make_contact_sheet
    model = load_model(variant, device)
    loader = make_dataloader(spec, max_images)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            domain = batch_dev.get('dataset')
            out = model(batch_dev['low'], domain=domain)
            sr = saturation_rate(out['I_hat'])
            name = batch.get('name')
            if isinstance(name, (list, tuple)):
                name = name[0]
            row: dict[str, Any] = {
                'variant_id': variant['id'],
                'display': variant['display'],
                'role': variant['role'],
                'seed': variant['seed'],
                'e_shape': variant['e_shape'] if variant['e_shape'] is not None else '',
                'dataset': spec['id'],
                'sample_index': idx,
                'name': str(name),
                'psnr': float(psnr_torch(out['I_hat'], batch_dev['high']).cpu()),
                'ssim': float(ssim_torch(out['I_hat'], batch_dev['high']).cpu()),
                'lee': float(local_exposure_error(out['I_hat'], batch_dev['high']).cpu()),
                'nai': float(noise_amplification_index(out['I_hat'], batch_dev['low']).cpu()),
                'input_psnr': float(psnr_torch(batch_dev['low'], batch_dev['high']).cpu()),
                'identity_drop': float(identity_drop(out['I_hat'], batch_dev['low'], batch_dev['high']).cpu()),
                'q_ece': float(q_ece(out['Q'], normalized_abs_error(out['I_hat'], batch_dev['high'])).cpu()),
                'over': float(sr['over'].cpu()),
                'under': float(sr['under'].cpu()),
                'A_mean': float(out['A'].mean().cpu()),
                'Q_mean': float(out['Q'].mean().cpu()),
                'mu_E': float(out['mu_E'].mean().cpu()),
            }
            if 'E_gt' in batch_dev:
                row.update(torch_gauge_metrics(out['E'], batch_dev['E_gt']))
            if visual_dir is not None and spec['id'] == 'uefb' and idx < 2 and variant['id'] in {'M4', 'M4J', 'M4J_ES', 'P3C_E0050_S3407'}:
                make_contact_sheet({
                    'low': batch_dev['low'][0:1], 'pred': out['I_hat'][0:1], 'gt': batch_dev['high'][0:1],
                    'E': out['E'][0:1], 'A': out['A'][0:1], 'Q': out['Q'][0:1]
                }, visual_dir / f"{variant['id']}_{idx:03d}.png", cell=128)
            rows.append(row)
    return rows


def build_execution_plan(protocol: dict[str, Any]) -> str:
    return f"""# GIR-Field relaxed pipeline 执行计划

来源指导：`{SOURCE_GUIDANCE}`

## 放松设置

- 不训练新的主干模型：`{protocol['no_main_training']}`
- 不恢复 DGB/P2F：`{not protocol['dgb_revival_allowed']}`
- 使用 frozen checkpoints 与已有 official-baseline outputs。
- 每个 split 最多评估 `{protocol['max_images_per_split']}` 张图，用于 relaxed routing evidence。
- bootstrap samples: `{protocol['bootstrap_samples']}`。
- LPIPS/BRISQUE 若本地包不可用则不报告，禁止伪造。

## Pipeline

1. N0: 证据清洗、source guidance 冻结、claim ledger。
2. N1: M4/M4J/M4J_ES/P3c/P3d per-image 统计复核。
3. N2: UEFB-G gauge perturbation，验证 gauge shift 与 shape distortion 可区分。
4. N3: 外部基线 registry，区分 black-box output-only 与 internal-field metrics。
5. N4: recoverability risk calibration probe；轻量 logistic risk probe，不改主干。
6. N5: 论文主表、图、报告、manifest。

## Claim 边界

这不是 SOTA 方法实验，也不是 DGB 复活实验。输出只支持 GIR-Field/UEFB-G 机制与 benchmark 的 relaxed evidence。
"""


def phase_n0(outputs: dict[str, Path], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    DOC_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_GUIDANCE.exists():
        shutil.copy2(SOURCE_GUIDANCE, outputs['source_copy'])
    outputs['execution_plan_md'].write_text(build_execution_plan(protocol), encoding='utf-8')

    sources = [
        ROOT / 'results/tables/p3_formal_uefb_m0_m5_summary.json',
        ROOT / 'results/tables/p3b_joint_m4_summary.json',
        ROOT / 'results/tables/p3c_multiseed_sweep_summary.json',
        ROOT / 'results/tables/p3c_multiseed_sweep_aggregate.json',
        ROOT / 'results/tables/p3d_e010_multiseed_summary.csv',
        ROOT / 'results/tables/p4_official_baselines_summary.json',
        ROOT / 'results/tables/dgb_branch_consolidation_summary.json',
        ROOT / 'results/hermes_audit/tables/final_main_table.csv',
        ROOT / 'results/hermes_audit/tables/final_ablation_table.csv',
        ROOT / 'results/hermes_audit/tables/noref_supplementary_summary.csv',
    ]
    evidence_rows = []
    for path in sources:
        evidence_rows.append({
            'artifact_id': path.stem,
            'path': str(path.relative_to(ROOT)) if path.exists() else str(path),
            'exists': path.exists(),
            'kind': path.suffix.lstrip('.'),
            'size_bytes': path.stat().st_size if path.exists() else '',
            'sha256': sha256_file(path) if path.exists() else '',
            'evidence_role': 'source_metric_or_prior_audit',
        })
    for v in planned_variants(ROOT):
        for key in ['config', 'checkpoint']:
            path = v[key]
            evidence_rows.append({
                'artifact_id': f"{v['id']}_{key}",
                'path': str(path.relative_to(ROOT)),
                'exists': path.exists(),
                'kind': key,
                'size_bytes': path.stat().st_size if path.exists() else '',
                'sha256': sha256_file(path) if path.exists() else '',
                'evidence_role': v['role'],
            })
    write_csv(outputs['evidence_manifest_csv'], evidence_rows)
    write_json(outputs['evidence_manifest_json'], evidence_rows)

    claim_rows = [
        {'claim_id': 'C1_centered_shape_positive', 'claim': 'Centered/gauge-invariant E-shape improves exposure-shape identifiability relative to M4J.', 'status': 'to_test_in_N1', 'primary_artifacts': 'N1_statistics/statistical_tests.csv', 'allowed_in_main': True, 'forbidden_overclaim': 'E_corr improvement guarantees superior restoration'},
        {'claim_id': 'C2_psnr_can_mislead', 'claim': 'UEFB-G/E-A-Q exposes PSNR-only misranking such as M4 high PSNR but weak/wrong shape.', 'status': 'to_test_in_N1_N2', 'primary_artifacts': 'N1_statistics, N2_uefbg', 'allowed_in_main': True, 'forbidden_overclaim': 'PSNR is invalid for all LLIE'},
        {'claim_id': 'C3_dgb_no_joint_resolution', 'claim': 'DGB/P5/P6/P7 move trade-offs but do not resolve tri-domain balance under current evidence.', 'status': 'frozen_prior_audit', 'primary_artifacts': 'dgb_branch_consolidation_summary; final_ablation_table', 'allowed_in_main': False, 'forbidden_overclaim': 'DGB-RLEF is SOTA or resolves tri-domain trade-off'},
        {'claim_id': 'C4_external_blackbox_boundary', 'claim': 'External black-box baselines can be compared on output metrics but not internal E/S/Gauge metrics.', 'status': 'to_register_in_N3', 'primary_artifacts': 'N3_external/external_baseline_registry.csv', 'allowed_in_main': True, 'forbidden_overclaim': 'black-box methods failed E_corr because they lack internal E-map'},
        {'claim_id': 'C5_recoverability_risk_probe', 'claim': 'A lightweight output/field risk probe can test whether recoverability risk is calibratable without reviving DGB router.', 'status': 'to_test_in_N4', 'primary_artifacts': 'N4_risk/risk_calibration_summary.csv', 'allowed_in_main': False, 'forbidden_overclaim': 'risk head solves enhancement'},
    ]
    write_csv(outputs['claim_ledger_csv'], claim_rows)
    write_json(outputs['claim_ledger_json'], claim_rows)
    return evidence_rows, claim_rows


def summarize_per_image(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ['psnr', 'ssim', 'lee', 'nai', 'identity_drop', 'E_MAE', 'S_MAE', 'Gauge_MAE', 'S_corr', 'q_ece', 'over', 'under', 'A_mean', 'mu_E']
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r['variant_id'], r['dataset'])].append(r)
    out = []
    for (variant_id, dataset), rs in sorted(grouped.items()):
        base = {
            'variant_id': variant_id,
            'dataset': dataset,
            'n': len(rs),
            'display': rs[0].get('display', ''),
            'role': rs[0].get('role', ''),
            'seed': rs[0].get('seed', ''),
            'e_shape': rs[0].get('e_shape', ''),
        }
        for m in metrics:
            vals = [safe_float(r.get(m)) for r in rs]
            vals = [v for v in vals if v is not None]
            if vals:
                base[m] = mean(vals)
                base[m + '_std'] = std(vals)
        out.append(base)
    return out


def bootstrap_ci(diffs: np.ndarray, samples: int, seed: int = 3407) -> tuple[float, float]:
    if diffs.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(samples, diffs.size))
    means = diffs[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def wilcoxon_pvalue(diffs: np.ndarray) -> float | None:
    try:
        from scipy.stats import wilcoxon
        nz = diffs[np.abs(diffs) > 1e-12]
        if nz.size < 2:
            return None
        return float(wilcoxon(nz, alternative='two-sided').pvalue)
    except Exception:
        return None


def statistical_tests(rows: list[dict[str, Any]], bootstrap_samples: int) -> list[dict[str, Any]]:
    by_key = {(r['variant_id'], r['dataset'], r['name']): r for r in rows}
    comparisons = [
        ('M4J', 'M4J_ES', ['psnr', 'S_corr', 'Gauge_MAE', 'E_MAE', 'identity_drop']),
        ('M4', 'M4J_ES', ['psnr', 'S_corr', 'Gauge_MAE', 'E_MAE', 'identity_drop']),
    ]
    test_rows: list[dict[str, Any]] = []
    for a, b, metrics in comparisons:
        datasets = sorted({r['dataset'] for r in rows if r['variant_id'] in {a, b}})
        for dataset in datasets:
            names = sorted({r['name'] for r in rows if r['dataset'] == dataset and r['variant_id'] in {a, b}})
            for metric in metrics:
                av, bv = [], []
                for name in names:
                    ra = by_key.get((a, dataset, name))
                    rb = by_key.get((b, dataset, name))
                    if not ra or not rb:
                        continue
                    va, vb = safe_float(ra.get(metric)), safe_float(rb.get(metric))
                    if va is None or vb is None:
                        continue
                    av.append(va); bv.append(vb)
                if len(av) < 2:
                    continue
                arr_a = np.asarray(av, dtype=np.float64)
                arr_b = np.asarray(bv, dtype=np.float64)
                diffs = arr_b - arr_a
                ci_lo, ci_hi = bootstrap_ci(diffs, bootstrap_samples)
                p = wilcoxon_pvalue(diffs)
                paired_std = float(diffs.std(ddof=1)) if diffs.size > 1 else math.nan
                effect = float(diffs.mean() / (paired_std + 1e-12)) if paired_std and not math.isnan(paired_std) else math.nan
                sign_effect = float((np.sum(diffs > 0) - np.sum(diffs < 0)) / diffs.size)
                test_rows.append({
                    'comparison': f'{b}_minus_{a}',
                    'dataset': dataset,
                    'metric': metric,
                    'n_pairs': int(diffs.size),
                    'mean_a': float(arr_a.mean()),
                    'mean_b': float(arr_b.mean()),
                    'mean_delta': float(diffs.mean()),
                    'bootstrap95_lo': ci_lo,
                    'bootstrap95_hi': ci_hi,
                    'wilcoxon_p': p if p is not None else '',
                    'paired_cohen_d': effect,
                    'sign_effect': sign_effect,
                    'win_rate_b_gt_a': float(np.mean(diffs > 0)),
                })
    qvals = apply_bh_fdr([safe_float(r.get('wilcoxon_p')) for r in test_rows])
    for r, q in zip(test_rows, qvals):
        r['bh_fdr_q'] = q if q is not None else ''
    return test_rows


def phase_n1(outputs: dict[str, Path], device: str, max_images: int, bootstrap_samples: int, force: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if outputs['per_image_csv'].exists() and outputs['summary_csv'].exists() and outputs['stat_tests_csv'].exists() and not force:
        per_rows = list(csv.DictReader(outputs['per_image_csv'].open(encoding='utf-8')))
        summary_rows = list(csv.DictReader(outputs['summary_csv'].open(encoding='utf-8')))
        stat_rows = list(csv.DictReader(outputs['stat_tests_csv'].open(encoding='utf-8')))
        return per_rows, summary_rows, stat_rows
    all_rows: list[dict[str, Any]] = []
    visual_dir = OUT_ROOT / 'N1_statistics/visual_smoke'
    for variant in planned_variants(ROOT):
        for spec in dataset_specs(ROOT):
            print(json.dumps({'phase': 'N1', 'eval': variant['id'], 'dataset': spec['id'], 'max_images': max_images}, ensure_ascii=False), flush=True)
            all_rows.extend(evaluate_variant(variant, spec, device, max_images=max_images, visual_dir=visual_dir))
    fields = ['variant_id', 'display', 'role', 'seed', 'e_shape', 'dataset', 'sample_index', 'name', 'psnr', 'ssim', 'lee', 'nai', 'input_psnr', 'identity_drop', 'q_ece', 'over', 'under', 'A_mean', 'Q_mean', 'mu_E', 'E_MAE', 'E_MAE_aligned', 'S_MAE', 'Gauge_MAE', 'S_corr', 'E_corr', 'mu_pred', 'mu_gt']
    write_csv(outputs['per_image_csv'], all_rows, fields=fields)
    write_json(outputs['per_image_json'], all_rows)
    summary_rows = summarize_per_image(all_rows)
    write_csv(outputs['summary_csv'], summary_rows)
    write_json(outputs['summary_json'], summary_rows)
    stat_rows = statistical_tests(all_rows, bootstrap_samples)
    write_csv(outputs['stat_tests_csv'], stat_rows)
    write_json(outputs['stat_tests_json'], stat_rows)
    return all_rows, summary_rows, stat_rows


def load_uefb_gt_rows(max_images: int) -> list[dict[str, Any]]:
    rows = []
    low_dir = ROOT / 'data/UEFB-v2/test/low'
    e_dir = ROOT / 'data/UEFB-v2/test/E_gt'
    q_dir = ROOT / 'data/UEFB-v2/test/Q_gt'
    for idx, lp in enumerate(sorted(low_dir.glob('*.png'))[:max_images]):
        name = lp.stem
        e = np.load(e_dir / f'{name}.npy').astype('float32')
        q = np.load(q_dir / f'{name}.npy').astype('float32') if (q_dir / f'{name}.npy').exists() else np.ones_like(e)
        arr = np.asarray(Image.open(lp).convert('L'), dtype=np.float32) / 255.0
        rows.append({'index': idx, 'name': name, 'E_gt': e, 'Q_gt': q, 'low_luma': arr})
    return rows


def make_shape_delta(shape: tuple[int, ...], amp: float = 0.5) -> np.ndarray:
    h, w = shape[-2], shape[-1]
    yy, xx = np.mgrid[0:h, 0:w]
    delta = np.sin(2 * np.pi * xx / max(w, 1)) * np.cos(2 * np.pi * yy / max(h, 1))
    delta = delta.astype('float32')
    delta = delta - float(delta.mean())
    delta = delta / (float(np.abs(delta).max()) + 1e-8) * amp
    return delta.reshape((1, h, w))


def classify_stratum(e: np.ndarray, q: np.ndarray, low_luma: np.ndarray) -> dict[str, Any]:
    e_mean = float(np.mean(e))
    e_std = float(np.std(e))
    low_mean = float(np.mean(low_luma))
    low_std = float(np.std(low_luma))
    q_mean = float(np.mean(q))
    if e_std > 0.55:
        primary = 'local_uneven'
    elif e_mean > 0.75:
        primary = 'global_underexposed'
    elif q_mean > 0.85 and abs(e_mean) < 0.20:
        primary = 'near_identity'
    elif low_std > 0.22:
        primary = 'high_structure_or_noise'
    else:
        primary = 'mixed_regular'
    return {'e_mean': e_mean, 'e_std': e_std, 'low_mean': low_mean, 'low_std': low_std, 'q_mean': q_mean, 'stratum': primary}


def phase_n2(outputs: dict[str, Path], max_images: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for item in load_uefb_gt_rows(max_images):
        e = item['E_gt']
        meta = {'index': item['index'], 'name': item['name'], **classify_stratum(e, item['Q_gt'], item['low_luma'])}
        meta_rows.append(meta)
        delta = make_shape_delta(e.shape, amp=0.5)
        variants = {
            'identity': e,
            'global_shift_plus_0p5': e + 0.5,
            'global_shift_minus_0p5': e - 0.5,
            'local_shape_distortion': e + delta,
            'mixed_shift_plus_shape': e + 0.5 + delta,
        }
        for kind, pred in variants.items():
            gm = compute_gauge_metrics_np(pred, e)
            metric_rows.append({'index': item['index'], 'name': item['name'], 'perturbation': kind, **gm, **{k: meta[k] for k in ['stratum', 'e_mean', 'e_std', 'low_mean', 'q_mean']}})
    write_csv(outputs['gauge_perturbation_csv'], metric_rows)
    write_json(outputs['gauge_perturbation_json'], metric_rows)
    write_csv(outputs['uefbg_metadata_csv'], meta_rows)
    return metric_rows, meta_rows


def phase_n3(outputs: dict[str, Path]) -> list[dict[str, Any]]:
    p4 = read_json(ROOT / 'results/tables/p4_official_baselines_summary.json')
    rows = []
    for r in p4:
        rows.append({
            'method': r.get('method'),
            'dataset': r.get('dataset'),
            'n': r.get('n'),
            'status': r.get('status'),
            'psnr': r.get('psnr'),
            'ssim': r.get('ssim'),
            'lee': r.get('lee'),
            'nai': r.get('nai'),
            'identity_drop': r.get('identity_drop'),
            'internal_field_metrics': 'N/A_black_box',
            'output_only_metrics': True,
            'pred_dir': r.get('pred_dir'),
            'claim_boundary': 'external black-box baseline; do not compare E/S/Gauge metrics',
        })
    write_csv(outputs['external_registry_csv'], rows)
    write_json(outputs['external_registry_json'], rows)
    return rows


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return None
    # Mann-Whitney U / rank AUC, high score = positive.
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    n_pos = labels.sum(); n_neg = labels.size - n_pos
    rank_sum_pos = ranks[labels].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def ece_score(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float | None:
    if labels.size == 0:
        return None
    total = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if mask.any():
            total += float(mask.mean()) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return total


def extract_risk_patches_for_variant(variant: dict[str, Any], device: str, max_images: int, patch: int, max_patches: int) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from rlef.utils.tensor_ops import luminance, gradient_magnitude
    model = load_model(variant, device)
    rows = []
    for spec in dataset_specs(ROOT):
        loader = make_dataloader(spec, max_images=max_images)
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(batch_dev['low'], domain=batch_dev.get('dataset'))
                low, high, pred = batch_dev['low'], batch_dev['high'], out['I_hat']
                mse_pred = (pred - high).pow(2).mean(dim=1, keepdim=True)
                mse_low = (low - high).pow(2).mean(dim=1, keepdim=True)
                benefit = F.avg_pool2d(mse_low - mse_pred, patch, stride=patch)
                harm = (benefit < -1e-4).float()
                beneficial = (benefit > 1e-4).float()
                lum_low = luminance(low)
                lum_pred = luminance(pred)
                grad_low = gradient_magnitude(lum_low)
                def pool(x):
                    return F.avg_pool2d(x, patch, stride=patch).flatten().detach().cpu().numpy()
                def flat(x):
                    return x.flatten().detach().cpu().numpy()
                pooled = {
                    'benefit': flat(benefit),
                    'harmful': flat(harm),
                    'beneficial': flat(beneficial),
                    'mean_low': pool(lum_low),
                    'mean_pred': pool(lum_pred),
                    'grad_low': pool(grad_low),
                    'A_mean': pool(out['A']),
                    'Q_risk': pool(1.0 - out['Q']),
                    'E_abs': pool(out['E'].abs()),
                    'S_abs': pool(out['S'].abs()),
                    'sat_pred': pool(((pred <= 1/255) | (pred >= 254/255)).float().mean(dim=1, keepdim=True)),
                }
                name = batch.get('name')
                if isinstance(name, (list, tuple)):
                    name = name[0]
                n = len(pooled['benefit'])
                for j in range(n):
                    rows.append({
                        'variant_id': variant['id'], 'dataset': spec['id'], 'image_index': idx, 'name': str(name), 'patch_index': j,
                        **{k: float(v[j]) for k, v in pooled.items()}
                    })
                    if len(rows) >= max_patches:
                        return rows
    return rows


def train_logistic_probe(patch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    if not patch_rows:
        return {'status': 'no_patch_rows'}
    features = ['mean_low', 'mean_pred', 'grad_low', 'A_mean', 'Q_risk', 'E_abs', 'S_abs', 'sat_pred']
    X = np.asarray([[float(r[f]) for f in features] for r in patch_rows], dtype=np.float32)
    y = np.asarray([float(r['harmful']) for r in patch_rows], dtype=np.float32)
    if y.sum() < 5 or y.sum() > len(y) - 5:
        return {'status': 'single_class_or_too_few_positive', 'n': int(len(y)), 'positive_rate': float(y.mean())}
    rng = np.random.default_rng(3407)
    idx = np.arange(len(y)); rng.shuffle(idx)
    split = int(0.7 * len(idx))
    tr, te = idx[:split], idx[split:]
    mu = X[tr].mean(axis=0, keepdims=True); sd = X[tr].std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    xt = torch.from_numpy(Xs[tr]); yt = torch.from_numpy(y[tr]).view(-1, 1)
    model = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)
    pos_weight = torch.tensor([(len(yt) - yt.sum()).clamp_min(1.0) / yt.sum().clamp_min(1.0)])
    for _ in range(240):
        logits = model(xt)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(Xs[te]))).numpy().reshape(-1)
    yte = y[te].astype(int)
    auc = auc_score(probs, yte)
    ece = ece_score(probs, yte)
    acc = float(((probs >= 0.5).astype(int) == yte).mean())
    # Baseline proxy AUCs on the same split.
    proxies = {}
    for f in ['A_mean', 'Q_risk', 'sat_pred', 'E_abs', 'S_abs']:
        proxies[f + '_auc'] = auc_score(X[te, features.index(f)], yte)
    return {
        'status': 'ok', 'n_train': int(len(tr)), 'n_test': int(len(te)), 'positive_rate_train': float(y[tr].mean()), 'positive_rate_test': float(yte.mean()),
        'logistic_auc': auc if auc is not None else '', 'logistic_ece': ece if ece is not None else '', 'logistic_acc@0.5': acc,
        **{k: (v if v is not None else '') for k, v in proxies.items()},
        'features': ','.join(features),
    }


def phase_n4(outputs: dict[str, Path], device: str, max_images: int, patch: int, max_patches: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_ids = {'M4J_ES', 'P3C_E0050_S3407'}
    patch_rows: list[dict[str, Any]] = []
    for v in planned_variants(ROOT):
        if v['id'] in selected_ids:
            print(json.dumps({'phase': 'N4', 'risk_extract': v['id'], 'max_patches': max_patches}, ensure_ascii=False), flush=True)
            patch_rows.extend(extract_risk_patches_for_variant(v, device, max_images=max_images, patch=patch, max_patches=max_patches // len(selected_ids)))
    write_csv(outputs['risk_patch_csv'], patch_rows)
    summaries = []
    for vid in sorted(selected_ids):
        rows = [r for r in patch_rows if r['variant_id'] == vid]
        result = train_logistic_probe(rows)
        result['variant_id'] = vid
        summaries.append(result)
    write_csv(outputs['risk_calibration_csv'], summaries)
    write_json(outputs['risk_calibration_json'], summaries)
    return patch_rows, summaries


def grouped_summary_lookup(summary_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r['variant_id'], r['dataset']): r for r in summary_rows}


def aggregate_variant_group(summary_rows: list[dict[str, Any]], ids: list[str], label: str, role: str) -> dict[str, Any]:
    out = {'method': label, 'role': role, 'source_variant_ids': ','.join(ids)}
    for dataset in ['uefb', 'real', 'synthetic']:
        rs = [r for r in summary_rows if r['variant_id'] in ids and r['dataset'] == dataset]
        if not rs:
            continue
        for metric in ['psnr', 'ssim', 'S_corr', 'Gauge_MAE', 'identity_drop', 'q_ece', 'over', 'under']:
            vals = [safe_float(r.get(metric)) for r in rs]
            vals = [v for v in vals if v is not None]
            if vals:
                out[f'{dataset}_{metric}'] = mean(vals)
                out[f'{dataset}_{metric}_std_across_rows'] = std(vals)
                out[f'{dataset}_n_rows'] = len(vals)
    return out


def phase_n5(outputs: dict[str, Path], summary_rows: list[dict[str, Any]], stat_rows: list[dict[str, Any]], perturb_rows: list[dict[str, Any]], risk_summary: list[dict[str, Any]], external_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table_rows = [
        aggregate_variant_group(summary_rows, ['M4'], 'M4 A-gate', 'PSNR-high counterexample'),
        aggregate_variant_group(summary_rows, ['M4J'], 'M4J joint', 'joint baseline'),
        aggregate_variant_group(summary_rows, ['M4J_ES'], 'M4J_ES seed3407', 'centered E-shape positive'),
        aggregate_variant_group(summary_rows, ['P3C_E0050_S42', 'P3C_E0050_S2027', 'P3C_E0050_S3407'], 'P3c e=0.05 3-seed', 'default 3-seed'),
        aggregate_variant_group(summary_rows, ['P3D_E0100_S42', 'P3D_E0100_S2027', 'P3D_E0100_S3407'], 'P3d e=0.10 3-seed', 'stronger shape control'),
    ]
    for method in sorted({r['method'] for r in external_rows}):
        row = {'method': method, 'role': 'external black-box output-only', 'source_variant_ids': 'p4_official_baselines'}
        for r in external_rows:
            if r['method'] != method:
                continue
            ds = r['dataset']
            row[f'{ds}_psnr'] = r.get('psnr')
            row[f'{ds}_ssim'] = r.get('ssim')
            row[f'{ds}_identity_drop'] = r.get('identity_drop')
            row[f'{ds}_S_corr'] = 'N/A'
            row[f'{ds}_Gauge_MAE'] = 'N/A'
        table_rows.append(row)
    write_csv(outputs['main_table_csv'], table_rows)
    write_json(outputs['main_table_json'], table_rows)
    outputs['main_table_md'].write_text(table_to_markdown(table_rows), encoding='utf-8')

    draw_psnr_scorr_figure(summary_rows, outputs['stat_figure'])
    draw_gauge_bar_figure(perturb_rows, outputs['gauge_figure'])
    draw_risk_figure(risk_summary, outputs['risk_figure'])
    draw_failure_grid(outputs['failure_grid'])
    return table_rows


def table_to_markdown(rows: list[dict[str, Any]]) -> str:
    fields = ['method', 'role', 'uefb_psnr', 'uefb_S_corr', 'uefb_Gauge_MAE', 'real_psnr', 'real_S_corr', 'real_Gauge_MAE', 'synthetic_psnr', 'synthetic_S_corr', 'synthetic_Gauge_MAE']
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    for r in rows:
        vals = []
        for f in fields:
            v = r.get(f, '')
            if isinstance(v, float):
                vals.append(f'{v:.4f}')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines) + '\n'


def draw_text_image(path: Path, title: str, lines: list[str], size=(1100, 700)) -> None:
    ensure_parent(path)
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), title, fill='black')
    y = 60
    for line in lines:
        draw.text((20, y), line, fill='black')
        y += 24
    img.save(path)


def draw_psnr_scorr_figure(summary_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [r for r in summary_rows if r.get('dataset') == 'uefb' and safe_float(r.get('psnr')) is not None and safe_float(r.get('S_corr')) is not None]
    if not rows:
        draw_text_image(path, 'PSNR vs S_corr', ['No rows available'])
        return
    w, h = 1000, 700
    img = Image.new('RGB', (w, h), 'white'); draw = ImageDraw.Draw(img)
    draw.text((30, 20), 'UEFB relaxed eval: PSNR vs gauge-free S_corr', fill='black')
    xs = [float(r['psnr']) for r in rows]; ys = [float(r['S_corr']) for r in rows]
    xmin, xmax = min(xs)-0.2, max(xs)+0.2; ymin, ymax = min(ys)-0.05, max(ys)+0.05
    left, top, right, bottom = 80, 80, 940, 620
    draw.rectangle((left, top, right, bottom), outline='black')
    for r in rows:
        x = (float(r['psnr']) - xmin) / (xmax - xmin + 1e-9)
        y = (float(r['S_corr']) - ymin) / (ymax - ymin + 1e-9)
        px = left + x * (right-left); py = bottom - y * (bottom-top)
        color = 'red' if r['variant_id'] == 'M4' else 'blue' if 'P3C' in r['variant_id'] else 'green'
        draw.ellipse((px-5, py-5, px+5, py+5), fill=color)
        draw.text((px+7, py-7), str(r['variant_id']), fill='black')
    draw.text((left, bottom+20), 'PSNR', fill='black')
    draw.text((10, top), 'S_corr', fill='black')
    ensure_parent(path); img.save(path)


def draw_gauge_bar_figure(perturb_rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in perturb_rows:
        grouped[r['perturbation']].append(r)
    lines = []
    for k in ['identity', 'global_shift_plus_0p5', 'local_shape_distortion', 'mixed_shift_plus_shape']:
        rs = grouped.get(k, [])
        if not rs:
            continue
        lines.append(f"{k}: E_MAE={mean([r['e_mae'] for r in rs]):.3f}, S_MAE={mean([r['s_mae'] for r in rs]):.3f}, Gauge={mean([r['gauge_mae'] for r in rs]):.3f}, S_corr={mean([r['s_corr'] for r in rs]):.3f}")
    draw_text_image(path, 'Gauge perturbation separates global gauge from shape error', lines)


def draw_risk_figure(risk_summary: list[dict[str, Any]], path: Path) -> None:
    lines = []
    for r in risk_summary:
        lines.append(f"{r.get('variant_id')}: status={r.get('status')}, AUC={r.get('logistic_auc')}, ECE={r.get('logistic_ece')}, positive_rate_test={r.get('positive_rate_test')}")
        for k in ['A_mean_auc', 'Q_risk_auc', 'sat_pred_auc', 'E_abs_auc', 'S_abs_auc']:
            if k in r:
                lines.append(f"  {k}: {r.get(k)}")
    draw_text_image(path, 'Recoverability risk relaxed probe', lines)


def draw_failure_grid(path: Path) -> None:
    # Use already generated contact sheets from N1 as a compact failure/misranking grid.
    visual_dir = OUT_ROOT / 'N1_statistics/visual_smoke'
    imgs = sorted(visual_dir.glob('*.png'))[:8]
    if not imgs:
        draw_text_image(path, 'Failure grid', ['No visual smoke files available'])
        return
    thumbs = [Image.open(p).convert('RGB').resize((512, 160)) for p in imgs]
    w = 1024; h = 40 + 160 * math.ceil(len(thumbs) / 2)
    canvas = Image.new('RGB', (w, h), 'white'); draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), 'Relaxed visual smoke: low/pred/gt/E/A/Q contact sheets for PSNR-vs-shape cases', fill='black')
    for i, im in enumerate(thumbs):
        x = (i % 2) * 512; y = 40 + (i // 2) * 160
        canvas.paste(im, (x, y))
    ensure_parent(path); canvas.save(path)


def write_report(outputs: dict[str, Path], protocol: dict[str, Any], evidence_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], stat_rows: list[dict[str, Any]], perturb_rows: list[dict[str, Any]], risk_summary: list[dict[str, Any]], table_rows: list[dict[str, Any]]) -> str:
    def find_stat(comp: str, dataset: str, metric: str) -> dict[str, Any] | None:
        for r in stat_rows:
            if r.get('comparison') == comp and r.get('dataset') == dataset and r.get('metric') == metric:
                return r
        return None
    m4j_es_scorr = find_stat('M4J_ES_minus_M4J', 'uefb', 'S_corr')
    m4j_es_psnr_real = find_stat('M4J_ES_minus_M4J', 'real', 'psnr')
    perturb_means = defaultdict(list)
    for r in perturb_rows:
        perturb_means[r['perturbation']].append(r)
    lines = [
        '# GIR-Field relaxed pipeline report',
        '',
        '## Protocol',
        '',
        f"- Claim scope: `{protocol['claim_scope']}`",
        f"- No main-model training: `{protocol['no_main_training']}`",
        f"- DGB revival allowed: `{protocol['dgb_revival_allowed']}`",
        f"- Max images per split: `{protocol['max_images_per_split']}`",
        f"- Bootstrap samples: `{protocol['bootstrap_samples']}`",
        '',
        '## Key relaxed-run findings',
        '',
    ]
    if m4j_es_scorr:
        lines.append(f"- M4J_ES vs M4J on UEFB S_corr: delta={float(m4j_es_scorr['mean_delta']):.4f}, 95% bootstrap CI=[{float(m4j_es_scorr['bootstrap95_lo']):.4f}, {float(m4j_es_scorr['bootstrap95_hi']):.4f}], Wilcoxon p={m4j_es_scorr.get('wilcoxon_p')}, FDR q={m4j_es_scorr.get('bh_fdr_q')}.")
    if m4j_es_psnr_real:
        lines.append(f"- M4J_ES vs M4J on real PSNR: delta={float(m4j_es_psnr_real['mean_delta']):.4f} dB, 95% CI=[{float(m4j_es_psnr_real['bootstrap95_lo']):.4f}, {float(m4j_es_psnr_real['bootstrap95_hi']):.4f}].")
    if perturb_means:
        for kind in ['global_shift_plus_0p5', 'local_shape_distortion', 'mixed_shift_plus_shape']:
            rs = perturb_means.get(kind, [])
            if rs:
                lines.append(f"- Perturbation `{kind}`: mean E_MAE={mean([r['e_mae'] for r in rs]):.4f}, mean S_MAE={mean([r['s_mae'] for r in rs]):.4f}, mean Gauge_MAE={mean([r['gauge_mae'] for r in rs]):.4f}, mean S_corr={mean([r['s_corr'] for r in rs]):.4f}.")
    for r in risk_summary:
        lines.append(f"- N4 risk probe `{r.get('variant_id')}`: status={r.get('status')}, logistic_auc={r.get('logistic_auc')}, logistic_ece={r.get('logistic_ece')}.")
    lines.extend([
        '',
        '## Claim calibration',
        '',
        '- This relaxed run supports GIR-Field/UEFB-G as a mechanism + benchmark route, not a SOTA LLIE method claim.',
        '- DGB/P2F was not resumed; DGB remains stopped and consolidated.',
        '- LPIPS/BRISQUE were not fabricated when packages were unavailable.',
        '- External black-box baselines are output-only; internal E/S/Gauge metrics are N/A for them.',
        '',
        '## Main artifacts',
        '',
    ])
    for key in ['claim_ledger_csv', 'per_image_csv', 'stat_tests_csv', 'gauge_perturbation_csv', 'external_registry_csv', 'risk_calibration_csv', 'main_table_csv', 'stat_figure', 'gauge_figure', 'risk_figure', 'failure_grid', 'manifest_json']:
        lines.append(f"- `{outputs[key].relative_to(ROOT)}`")
    text = '\n'.join(lines) + '\n'
    ensure_parent(outputs['report_md'])
    outputs['report_md'].write_text(text, encoding='utf-8')
    return text


def build_manifest(outputs: dict[str, Path], protocol: dict[str, Any]) -> dict[str, Any]:
    files = []
    for key, path in outputs.items():
        if path.exists() and path.is_file():
            files.append({'key': key, 'relative_path': str(path.relative_to(ROOT)), 'size_bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    manifest = {
        'pipeline': 'GIR-Field relaxed pipeline',
        'created_at_unix': time.time(),
        'protocol': protocol,
        'phases': planned_phases(),
        'file_count': len(files),
        'files': files,
    }
    write_json(outputs['manifest_json'], manifest)
    return manifest


def validate_outputs(outputs: dict[str, Path], manifest: dict[str, Any]) -> str:
    required = ['claim_ledger_csv', 'per_image_csv', 'summary_csv', 'stat_tests_csv', 'gauge_perturbation_csv', 'external_registry_csv', 'risk_calibration_csv', 'main_table_csv', 'report_md', 'manifest_json']
    errors = []
    for key in required:
        if not outputs[key].exists() or outputs[key].stat().st_size == 0:
            errors.append(f'missing_or_empty:{key}')
    for key in ['claim_ledger_json', 'per_image_json', 'summary_json', 'stat_tests_json', 'gauge_perturbation_json', 'external_registry_json', 'risk_calibration_json', 'main_table_json', 'manifest_json']:
        try:
            read_json(outputs[key])
        except Exception as e:
            errors.append(f'json_error:{key}:{e}')
    for key in ['stat_figure', 'gauge_figure', 'risk_figure', 'failure_grid']:
        try:
            Image.open(outputs[key]).verify()
        except Exception as e:
            errors.append(f'image_error:{key}:{e}')
    for item in manifest.get('files', []):
        p = ROOT / item['relative_path']
        if p.exists() and sha256_file(p) != item['sha256']:
            errors.append(f'sha_mismatch:{item["relative_path"]}')
    status = 'PASS' if not errors else 'FAIL'
    text = '# GIR-Field relaxed validation\n\n' + f'- status: `{status}`\n' + f'- manifest files: {manifest.get("file_count")}\n' + ''.join(f'- error: {e}\n' for e in errors)
    outputs['validation_md'].write_text(text, encoding='utf-8')
    if errors:
        raise RuntimeError('validation failed: ' + '; '.join(errors))
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-images', type=int, default=relaxed_protocol()['max_images_per_split'])
    ap.add_argument('--bootstrap-samples', type=int, default=relaxed_protocol()['bootstrap_samples'])
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    protocol = relaxed_protocol()
    protocol['max_images_per_split'] = int(args.max_images)
    protocol['bootstrap_samples'] = int(args.bootstrap_samples)
    outputs = planned_outputs(ROOT)
    for p in outputs.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    evidence_rows, claim_rows = phase_n0(outputs, protocol)
    per_rows, summary_rows, stat_rows = phase_n1(outputs, args.device, args.max_images, args.bootstrap_samples, force=args.force)
    perturb_rows, meta_rows = phase_n2(outputs, args.max_images)
    external_rows = phase_n3(outputs)
    patch_rows, risk_summary = phase_n4(outputs, args.device, args.max_images, protocol['risk_patch_size'], protocol['risk_max_patches'])
    table_rows = phase_n5(outputs, summary_rows, stat_rows, perturb_rows, risk_summary, external_rows)
    report = write_report(outputs, protocol, evidence_rows, summary_rows, stat_rows, perturb_rows, risk_summary, table_rows)
    manifest = build_manifest(outputs, protocol)
    validation = validate_outputs(outputs, manifest)
    print(json.dumps({'DONE': True, 'report': str(outputs['report_md']), 'manifest': str(outputs['manifest_json']), 'validation': validation.splitlines()[2] if len(validation.splitlines()) > 2 else 'PASS'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
