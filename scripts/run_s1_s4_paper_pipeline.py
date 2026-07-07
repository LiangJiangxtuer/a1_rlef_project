#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Public contract used by tests
# -----------------------------------------------------------------------------

def planned_stages() -> list[dict[str, str]]:
    return [
        {'id': 'S1', 'name': 'Evidence freeze + paper framing'},
        {'id': 'S2', 'name': 'Manuscript-grade main table'},
        {'id': 'S3', 'name': 'Qualitative diagnostic visualization'},
        {'id': 'S4', 'name': 'Appendix negative-route diagnostics'},
    ]


def planned_outputs(root: Path = ROOT) -> dict[str, Path]:
    out = root / 'results/paper_pipeline'
    return {
        'root': out,
        'manifest': out / 'S1_S4_PIPELINE_MANIFEST.json',
        'report': out / 'S1_S4_PIPELINE_REPORT.md',
        's1_dir': out / 's1_evidence_freeze',
        's2_dir': out / 's2_main_table',
        's3_dir': out / 's3_visualizations',
        's4_dir': out / 's4_appendix',
    }


def build_core_table_rows(root: Path = ROOT) -> list[dict[str, Any]]:
    p3 = _load_json(root / 'results/tables/p3_formal_uefb_m0_m5_summary.json')
    p3b = _load_json(root / 'results/tables/p3b_joint_m4_summary.json')
    p3c = _load_json(root / 'results/tables/p3c_multiseed_sweep_aggregate.json')['multiseed_e_shape_0p05']
    p4 = _load_json(root / 'results/tables/p4_official_baselines_summary.json')

    def by_variant(rows: list[dict[str, Any]], variant_id: str) -> dict[str, Any]:
        for row in rows:
            if row.get('variant_id') == variant_id:
                return row
        raise KeyError(variant_id)

    def baseline(method: str) -> dict[str, Any]:
        real = next(row for row in p4 if row.get('method') == method and row.get('dataset') == 'real')
        syn = next(row for row in p4 if row.get('method') == method and row.get('dataset') == 'synthetic')
        return {
            'row_id': method.upper().replace('+', 'P').replace('-', '_') + '_OFFICIAL',
            'method': method,
            'role': 'official baseline',
            'evidence_level': 'official-code eval',
            'uefb_psnr': None,
            'real_psnr': real.get('psnr'),
            'synthetic_psnr': syn.get('psnr'),
            'uefb_E_corr': None,
            'real_E_corr': None,
            'synthetic_E_corr': None,
            'claim_scope': 'restoration baseline only; no RLEF interpretability heads',
        }

    m0 = by_variant(p3, 'M0')
    m4 = by_variant(p3, 'M4')
    m4j = by_variant(p3b, 'M4J')
    m4jes = by_variant(p3b, 'M4J_ES')
    rows = [
        _method_row('M0_RESTORER_ONLY', 'M0 restorer_only', 'minimal restorer baseline', 'single-seed ablation', m0, 'baseline for mechanism ablation'),
        _method_row('M4_GATE', 'M4 A-gate', 'strong UEFB mechanism', 'single-seed ablation', m4, 'A-gate improves UEFB but no paired-domain maturity'),
        _method_row('M4J', 'M4J joint', 'joint-training bridge', 'single-seed route evidence', m4j, 'joint UEFB+LOL training improves paired fidelity'),
        _method_row('M4J_ES_SEED3407', 'M4J_ES seed3407', 'E-shape route evidence', 'single-seed route evidence', m4jes, 'low-pass centered E-shape turns E correlations positive'),
        {
            'row_id': 'P3C_M4J_ES_E005_MEAN',
            'method': 'P3c M4J_ES e_shape=0.05',
            'role': 'conservative default',
            'evidence_level': '3-seed default',
            'uefb_psnr': p3c['uefb_psnr']['mean'],
            'real_psnr': p3c['real_psnr']['mean'],
            'synthetic_psnr': p3c['synthetic_psnr']['mean'],
            'uefb_E_corr': p3c['uefb_E_corr']['mean'],
            'real_E_corr': p3c['real_E_corr']['mean'],
            'synthetic_E_corr': p3c['synthetic_E_corr']['mean'],
            'claim_scope': 'main RLEF default; interpretable exposure-field auxiliary calibration',
            'uefb_psnr_std': p3c['uefb_psnr'].get('std'),
            'real_psnr_std': p3c['real_psnr'].get('std'),
            'synthetic_psnr_std': p3c['synthetic_psnr'].get('std'),
        },
        baseline('Retinexformer'),
        baseline('Zero-DCE++'),
        baseline('KinD++'),
    ]
    # Normalize row id for exact test contract.
    for row in rows:
        if row.get('method') == 'Retinexformer':
            row['row_id'] = 'RETINEXFORMER_OFFICIAL'
        elif row.get('method') == 'Zero-DCE++':
            row['row_id'] = 'ZERO_DCE_PP_OFFICIAL'
        elif row.get('method') == 'KinD++':
            row['row_id'] = 'KINDPP_OFFICIAL'
    return rows


def build_appendix_rows(root: Path = ROOT) -> list[dict[str, Any]]:
    p5 = _load_json(root / 'results/tables/p5_retinex_distill_summary.json')['runs']
    p5b = _load_json(root / 'results/tables/p5b_domain_distill_summary.json')['runs']
    p6 = _load_json(root / 'results/tables/p6_structural_backbone_summary.json')['runs']
    p6b = _load_json(root / 'results/tables/p6b_synprotect_summary.json')['runs']
    p6c = _load_json(root / 'results/tables/p6c_gateprotect_summary.json')['runs']
    p7 = _load_json(root / 'results/tables/p7_domainhead_summary.json')['runs']
    p7b = _load_json(root / 'results/tables/p7b_realanchor_summary.json')['runs']
    p7c = _load_json(root / 'results/tables/p7c_realanchor_fine_summary.json')['runs']
    dgb = _load_json(root / 'results/tables/dgb_branch_consolidation_summary.json')

    p5_best_real = _best(p5, 'real_psnr')
    p6_best_syn = _best(p6b, 'synthetic_psnr')
    p6c_best_real = _best(p6c, 'real_psnr')
    p7b_near = next(row for row in p7b if row.get('variant_id') == 'P7B_DHEAD_RA010')
    return [
        {
            'route': 'P5/P5b distillation',
            'best_reference': p5_best_real['variant_id'],
            'best_snapshot': _short_metrics(p5_best_real),
            'decision': 'appendix only',
            'reason': 'output-level teacher distillation shifts real/synthetic trade-off and is not a stable deployable mainline',
            'n_runs': len(p5) + len(p5b),
        },
        {
            'route': 'P6/P6b/P6c structural scalar controls',
            'best_reference': f"{p6[0]['variant_id']} / {p6_best_syn['variant_id']} / {p6c_best_real['variant_id']}",
            'best_snapshot': f"P6 real={_fmt(p6[0].get('real_psnr'))}; P6b synthetic={_fmt(p6_best_syn.get('synthetic_psnr'))}; P6c real={_fmt(p6c_best_real.get('real_psnr'))}",
            'decision': 'appendix only',
            'reason': 'structural backbone is useful, but scalar synthetic/gate protection over-corrects and does not solve three-domain trade-off',
            'n_runs': len(p6) + len(p6b) + len(p6c),
        },
        {
            'route': 'P7/P7b/P7c domain heads',
            'best_reference': p7b_near['variant_id'],
            'best_snapshot': _short_metrics(p7b_near),
            'decision': 'near-miss reference only',
            'reason': 'P7B_RA010 is closest but misses real P3c mean; stronger P7c anchors degrade',
            'n_runs': len(p7) + len(p7b) + len(p7c),
        },
        {
            'route': 'DGB branch',
            'best_reference': dgb['best_by_split']['real_psnr']['variant_id'],
            'best_snapshot': f"best real={_fmt(dgb['best_by_split']['real_psnr']['real_psnr'])}; best synthetic={_fmt(dgb['best_by_split']['synthetic_psnr']['synthetic_psnr'])}; joint_gate_any={dgb['joint_gate_any']}",
            'decision': 'stopped and consolidated',
            'reason': 'joint gate failed across Phase2/P2B/P2C/P2D/P2E; routing changes did not recover real-domain quality',
            'n_runs': len(dgb.get('runs', [])),
        },
    ]


# -----------------------------------------------------------------------------
# Pipeline internals
# -----------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def _method_row(row_id: str, method: str, role: str, evidence_level: str, src: dict[str, Any], claim_scope: str) -> dict[str, Any]:
    return {
        'row_id': row_id,
        'method': method,
        'role': role,
        'evidence_level': evidence_level,
        'uefb_psnr': src.get('uefb_psnr'),
        'real_psnr': src.get('real_psnr'),
        'synthetic_psnr': src.get('synthetic_psnr'),
        'uefb_E_corr': src.get('uefb_E_corr'),
        'real_E_corr': src.get('real_E_corr'),
        'synthetic_E_corr': src.get('synthetic_E_corr'),
        'claim_scope': claim_scope,
    }


def _best(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row.get(key, -1e9)))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return '—'
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return str(value)


def _short_metrics(row: dict[str, Any]) -> str:
    return f"UEFB={_fmt(row.get('uefb_psnr'))}; real={_fmt(row.get('real_psnr'))}; synthetic={_fmt(row.get('synthetic_psnr'))}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def _markdown_table(rows: list[dict[str, Any]], fields: list[str], labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    header = '| ' + ' | '.join(labels.get(field, field) for field in fields) + ' |'
    sep = '| ' + ' | '.join('---' for _ in fields) + ' |'
    body = []
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field)
            if value is None:
                cells.append('—')
            elif isinstance(value, float):
                cells.append(_fmt(value))
            else:
                cells.append(str(value))
        body.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join([header, sep, *body])


def _latex_table(rows: list[dict[str, Any]]) -> str:
    cols = ['method', 'role', 'uefb_psnr', 'real_psnr', 'synthetic_psnr', 'uefb_E_corr', 'real_E_corr', 'synthetic_E_corr']
    lines = [
        '% Auto-generated by scripts/run_s1_s4_paper_pipeline.py',
        '\\begin{tabular}{llrrrrrr}',
        '\\toprule',
        'Method & Role & UEFB PSNR & Real PSNR & Synthetic PSNR & UEFB $E_{corr}$ & Real $E_{corr}$ & Synthetic $E_{corr}$ \\\\',
        '\\midrule',
    ]
    for row in rows:
        vals = [str(row.get('method', '')), str(row.get('role', ''))] + [_fmt(row.get(c)) for c in cols[2:]]
        vals = [v.replace('_', '\\_') for v in vals]
        lines.append(' & '.join(vals) + ' \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}', ''])
    return '\n'.join(lines)


def run_s1(root: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    claims = [
        {
            'claim_id': 'C1',
            'claim': 'RLEF should be framed as interpretable exposure-field auxiliary calibration, not SOTA restoration.',
            'status': 'supported as framing',
            'evidence': 'P4 official baselines show Retinexformer real/synthetic PSNR gap; P3c has stable E diagnostics.',
            'allowed_location': 'main text framing + limitations',
        },
        {
            'claim_id': 'C2',
            'claim': 'Adaptive/gauge-aware exposure modeling is preferable to fixed absolute gauge.',
            'status': 'supported',
            'evidence': 'P1 formal: adaptive is best on real/synthetic; fixed0p02 is domain-fragile.',
            'allowed_location': 'method motivation',
        },
        {
            'claim_id': 'C3',
            'claim': 'A-gate is the strongest early mechanism in UEFB ablation.',
            'status': 'supported single-seed',
            'evidence': 'P3 M4 gate gives best UEFB PSNR among M0-M5.',
            'allowed_location': 'ablation discussion',
        },
        {
            'claim_id': 'C4',
            'claim': 'Low-pass centered e_shape is the core mechanism for positive spatial E diagnostics.',
            'status': 'supported',
            'evidence': 'P3b M4J_ES and P3c 3-seed E_corr stay positive.',
            'allowed_location': 'main method + diagnostics',
        },
        {
            'claim_id': 'C5',
            'claim': 'DGB is not a final route.',
            'status': 'stopped',
            'evidence': 'DGB consolidation: no Phase2/P2B/P2C/P2D/P2E variant passes joint gate.',
            'allowed_location': 'appendix negative diagnostics',
        },
    ]
    _write_csv(out_dir / 'claim_matrix.csv', claims, ['claim_id', 'claim', 'status', 'evidence', 'allowed_location'])
    (out_dir / 'claim_matrix.json').write_text(json.dumps(claims, indent=2, ensure_ascii=False), encoding='utf-8')
    md = f"""# S1 Evidence Freeze + Paper Framing

Generated: {_now()}

## Framing decision

```text
Main story: interpretable, gauge-aware exposure-field auxiliary calibration.
Not allowed: SOTA / Retinexformer-level restoration claim.
```

## Frozen default

```text
P3c M4J_ES e_shape=0.05
```

## Claim matrix

{_markdown_table(claims, ['claim_id','claim','status','evidence','allowed_location'])}

## Source anchors

- `docs/CLAIM_LEDGER.md`
- `results/hermes_audit/reports/STAGE_EXPERIMENT_SUMMARY_AND_ADJUSTED_PLAN.md`
- `results/hermes_audit/reports/DGB_BRANCH_STOP_CONSOLIDATION_REPORT.md`
"""
    (out_dir / 'EVIDENCE_FREEZE.md').write_text(md, encoding='utf-8')
    return [out_dir / 'claim_matrix.csv', out_dir / 'claim_matrix.json', out_dir / 'EVIDENCE_FREEZE.md']


def run_s2(root: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_core_table_rows(root)
    fields = ['row_id', 'method', 'role', 'evidence_level', 'uefb_psnr', 'real_psnr', 'synthetic_psnr', 'uefb_E_corr', 'real_E_corr', 'synthetic_E_corr', 'claim_scope']
    _write_csv(out_dir / 'main_results_table.csv', rows, fields)
    (out_dir / 'main_results_table.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    md_table = _markdown_table(rows, ['method','role','evidence_level','uefb_psnr','real_psnr','synthetic_psnr','uefb_E_corr','real_E_corr','synthetic_E_corr'])
    md = f"""# S2 Manuscript-Grade Main Table

This table intentionally keeps the main paper compact. Distillation, scalar controls, domain-head near-misses, and DGB diagnostics are moved to S4 appendix artifacts.

{md_table}

## Table-use rule

Use this as the main experimental table. Do not mix stopped DGB/P5/P6/P7 scalar-route rows into the main text unless discussing negative diagnostics.
"""
    (out_dir / 'main_results_table.md').write_text(md, encoding='utf-8')
    (out_dir / 'main_results_table.tex').write_text(_latex_table(rows), encoding='utf-8')
    return [out_dir / 'main_results_table.csv', out_dir / 'main_results_table.json', out_dir / 'main_results_table.md', out_dir / 'main_results_table.tex']


def run_s3(root: Path, out_dir: Path, device: str = 'cuda', n: int = 4, skip_visuals: bool = False) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if skip_visuals:
        note = out_dir / 'VISUALIZATION_INDEX.md'
        note.write_text('# S3 Visualization skipped\n\nPipeline was run with `--skip_visuals`.\n', encoding='utf-8')
        return [note]
    outputs = _generate_visual_panels(root, out_dir, device=device, n=n)
    index_lines = [
        '# S3 Qualitative Diagnostic Visualization',
        '',
        f'Generated: {_now()}',
        '',
        'Representative frozen checkpoint: `P3b/P3c M4J_ES seed3407 e_shape=0.05` for visual examples. P3c aggregate remains the numeric default.',
        '',
        'Columns use available frozen artifacts: low input, RLEF prediction, Retinexformer official output, Zero-DCE++ official output, GT, predicted E map, predicted A map, and Q map. For the P3c/M4J_ES default, Q branch is inactive, so Q is expected to appear as a constant map; Q-branch claims remain ablation-only.',
        '',
        '## Panels',
    ]
    for path in outputs:
        index_lines.append(f'- `{path.relative_to(root)}`')
    index = out_dir / 'VISUALIZATION_INDEX.md'
    index.write_text('\n'.join(index_lines) + '\n', encoding='utf-8')
    return [index, *outputs]


def run_s4(root: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_appendix_rows(root)
    fields = ['route', 'best_reference', 'best_snapshot', 'decision', 'reason', 'n_runs']
    _write_csv(out_dir / 'negative_route_summary.csv', rows, fields)
    (out_dir / 'negative_route_summary.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    table = _markdown_table(rows, fields)
    md = f"""# S4 Appendix: Negative Routes and Diagnostic Findings

Generated: {_now()}

These rows should be used as appendix evidence / negative-route diagnostics, not as main-method claims.

{table}

## Appendix writing policy

- Present these as decisions that prevented overclaiming.
- Emphasize diagnostic learning, not failure.
- Do not revive scalar sweeps unless a new non-DGB hypothesis and validation gate are explicitly defined.
"""
    (out_dir / 'APPENDIX_NEGATIVE_RESULTS.md').write_text(md, encoding='utf-8')
    return [out_dir / 'negative_route_summary.csv', out_dir / 'negative_route_summary.json', out_dir / 'APPENDIX_NEGATIVE_RESULTS.md']


def run_pipeline(root: Path = ROOT, device: str = 'cuda', n_visuals: int = 4, skip_visuals: bool = False) -> dict[str, Any]:
    outputs = planned_outputs(root)
    outputs['root'].mkdir(parents=True, exist_ok=True)
    started = time.time()
    generated: list[Path] = []
    stage_records = []
    for stage in planned_stages():
        stage_start = time.time()
        if stage['id'] == 'S1':
            files = run_s1(root, outputs['s1_dir'])
        elif stage['id'] == 'S2':
            files = run_s2(root, outputs['s2_dir'])
        elif stage['id'] == 'S3':
            files = run_s3(root, outputs['s3_dir'], device=device, n=n_visuals, skip_visuals=skip_visuals)
        elif stage['id'] == 'S4':
            files = run_s4(root, outputs['s4_dir'])
        else:  # pragma: no cover
            raise ValueError(stage)
        generated.extend(files)
        stage_records.append({**stage, 'files': [str(p) for p in files], 'elapsed_sec': time.time() - stage_start})
    report_path = outputs['report']
    manifest_path = outputs['manifest']
    report_path.write_text(_pipeline_report(root, stage_records, generated, skip_visuals=skip_visuals), encoding='utf-8')
    generated.append(report_path)
    manifest = {
        'generated_at': _now(),
        'root': str(root),
        'decision': 'S1-S4 executed as full evidence-to-paper pipeline',
        'skip_visuals': bool(skip_visuals),
        'n_visuals_per_split': int(n_visuals),
        'elapsed_sec': time.time() - started,
        'stages': stage_records,
        'files': [_file_record(root, path) for path in generated if path.exists()],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    _validate_manifest(manifest)
    return {'manifest': manifest, 'manifest_path': manifest_path, 'report_path': report_path}


def _pipeline_report(root: Path, stage_records: list[dict[str, Any]], files: list[Path], skip_visuals: bool) -> str:
    stage_lines = ['| Stage | Name | Files | Elapsed sec |', '|---|---|---:|---:|']
    for stage in stage_records:
        stage_lines.append(f"| {stage['id']} | {stage['name']} | {len(stage['files'])} | {_fmt(stage['elapsed_sec'], 2)} |")
    file_lines = ['| Artifact |', '|---|']
    for path in files:
        if path.exists():
            file_lines.append(f"| `{path.relative_to(root)}` |")
    visual_note = 'S3 generated real/synthetic qualitative panels.' if not skip_visuals else 'S3 visual generation was skipped by CLI flag.'
    return f"""# S1-S4 Paper Evidence Pipeline Report

Generated: {_now()}  
Project root: `{root}`

## Pipeline decision

```text
S1 → S2 → S3 → S4 completed as a reproducible local pipeline.
No training, no GitHub push, no local git commit.
```

{visual_note}

## Stage summary

{chr(10).join(stage_lines)}

## Core output locations

- S1 evidence freeze: `results/paper_pipeline/s1_evidence_freeze/EVIDENCE_FREEZE.md`
- S2 main table: `results/paper_pipeline/s2_main_table/main_results_table.md`
- S3 visuals: `results/paper_pipeline/s3_visualizations/VISUALIZATION_INDEX.md`
- S4 appendix: `results/paper_pipeline/s4_appendix/APPENDIX_NEGATIVE_RESULTS.md`
- Manifest: `results/paper_pipeline/S1_S4_PIPELINE_MANIFEST.json`

## Generated artifacts

{chr(10).join(file_lines)}
"""


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        'path': str(path),
        'relative_path': str(path.relative_to(root)),
        'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
    }


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required = {'S1', 'S2', 'S3', 'S4'}
    got = {stage['id'] for stage in manifest['stages']}
    missing = required - got
    if missing:
        raise RuntimeError(f'Missing stages: {missing}')
    missing_files = [record['path'] for record in manifest['files'] if not Path(record['path']).exists()]
    if missing_files:
        raise FileNotFoundError(missing_files)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# Visualization helpers imported only when needed
# -----------------------------------------------------------------------------

def _generate_visual_panels(root: Path, out_dir: Path, device: str, n: int) -> list[Path]:
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'src'))
    import torch
    import yaml
    from PIL import Image, ImageDraw
    from torchvision.transforms.functional import to_tensor
    from scripts.train import build_model

    cfg_path = root / 'configs/p3b_joint/p3b_m4j_es_m4_joint_eshape.yml'
    ckpt_path = root / 'experiments/p3b_m4j_es_m4_joint_eshape_seed3407/checkpoints/last.pth'
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f'Missing visualization checkpoint/config: {cfg_path}, {ckpt_path}')
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    torch_device = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = build_model(cfg).to(torch_device)
    ckpt = torch.load(ckpt_path, map_location=torch_device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    split_specs = {
        'real': {
            'low_dir': root / 'data/LOL-v2/Real_captured/Test/Low',
            'high_dir': root / 'data/LOL-v2/Real_captured/Test/Normal',
            'retinex_dir': root / 'experiments/p4_official_baselines/retinexformer/results/LOL_v2_real/RetinexFormer_LOL_v2_real/LOL_v2_real',
            'zero_dir': root / 'experiments/p4_official_baselines/zero_dce_pp/results/real',
        },
        'synthetic': {
            'low_dir': root / 'data/LOL-v2/Synthetic/Test/Low',
            'high_dir': root / 'data/LOL-v2/Synthetic/Test/Normal',
            'retinex_dir': root / 'experiments/p4_official_baselines/retinexformer/results/LOL_v2_synthetic/RetinexFormer_LOL_v2_synthetic/LOL_v2_synthetic',
            'zero_dir': root / 'experiments/p4_official_baselines/zero_dce_pp/results/synthetic',
        },
    }
    generated: list[Path] = []
    for split, spec in split_specs.items():
        low_paths = sorted(spec['low_dir'].glob('*'))[:n]
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        for i, low_path in enumerate(low_paths):
            high_path = _high_path_for_low(low_path, spec['high_dir'])
            ret_path = spec['retinex_dir'] / low_path.name
            zero_path = spec['zero_dir'] / low_path.name
            low_img = Image.open(low_path).convert('RGB')
            low_tensor = to_tensor(low_img).unsqueeze(0).to(torch_device)
            with torch.no_grad():
                out = model(low_tensor, domain=split)
            rlef_img = _tensor_to_pil(out['I_hat'][0])
            e_map = _map_to_pil(out['E'][0])
            a_map = _map_to_pil(out['A'][0])
            q_map = _map_to_pil(out['Q'][0])
            tiles: list[tuple[str, Image.Image]] = [('low', low_img), ('RLEF', rlef_img)]
            if ret_path.exists():
                tiles.append(('Retinexformer', Image.open(ret_path).convert('RGB')))
            if zero_path.exists():
                tiles.append(('Zero-DCE++', Image.open(zero_path).convert('RGB')))
            if high_path.exists():
                tiles.append(('GT', Image.open(high_path).convert('RGB')))
            tiles.extend([('E map', e_map), ('A map', a_map), ('Q map/inactive', q_map)])
            panel_path = split_out / f'{split}_{i:03d}_{low_path.stem}_panel.png'
            _make_pil_contact_sheet(tiles, panel_path)
            generated.append(panel_path)
    return generated


def _high_path_for_low(low_path: Path, high_dir: Path) -> Path:
    candidates = [high_dir / low_path.name]
    if low_path.name.startswith('low'):
        candidates.append(high_dir / ('normal' + low_path.name[len('low'):]))
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _tensor_to_pil(tensor: Any) -> Any:
    from PIL import Image
    x = tensor.detach().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype('uint8')
    return Image.fromarray(arr)


def _map_to_pil(tensor: Any) -> Any:
    from PIL import Image
    x = tensor.detach().cpu().float()
    if x.ndim == 3:
        x = x[0]
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-8:
        y = x * 0 + 0.5
    else:
        y = (x - lo) / (hi - lo)
    arr = (y.numpy() * 255.0 + 0.5).clip(0, 255).astype('uint8')
    return Image.fromarray(arr, mode='L').convert('RGB')


def _make_pil_contact_sheet(tiles: list[tuple[str, Any]], path: Path, cell: int = 160) -> None:
    from PIL import Image, ImageDraw
    path.parent.mkdir(parents=True, exist_ok=True)
    label_h = 24
    sheet = Image.new('RGB', (cell * len(tiles), cell + label_h), 'white')
    draw = ImageDraw.Draw(sheet)
    for idx, (label, img) in enumerate(tiles):
        img = img.convert('RGB').resize((cell, cell))
        x = idx * cell
        sheet.paste(img, (x, label_h))
        draw.text((x + 4, 4), label[:22], fill=(0, 0, 0))
    sheet.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description='Run S1-S4 paper evidence pipeline for A1-RLEF.')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n_visuals', type=int, default=4)
    ap.add_argument('--skip_visuals', action='store_true')
    args = ap.parse_args()
    result = run_pipeline(ROOT, device=args.device, n_visuals=args.n_visuals, skip_visuals=args.skip_visuals)
    print(json.dumps({'DONE': True, 'manifest': str(result['manifest_path']), 'report': str(result['report_path'])}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
