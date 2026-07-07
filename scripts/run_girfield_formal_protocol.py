#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import scripts.run_girfield_relaxed_pipeline as base

ROOT = base.ROOT
FORMAL_OUT = ROOT / 'results/girfield_formal'
SOURCE_GUIDANCE = base.SOURCE_GUIDANCE


def full_dataset_counts(root: Path = ROOT) -> dict[str, int]:
    counts = {
        'uefb': len(list((root / 'data/UEFB-v2/test/low').glob('*.png'))),
        'real': len([p for p in (root / 'data/LOL-v2/Real_captured/Test/Low').glob('*') if p.is_file()]),
        'synthetic': len([p for p in (root / 'data/LOL-v2/Synthetic/Test/Low').glob('*') if p.is_file()]),
    }
    counts['total_per_variant'] = counts['uefb'] + counts['real'] + counts['synthetic']
    counts['expected_per_image_rows_for_9_variants'] = 9 * counts['total_per_variant']
    return counts


def formal_protocol(root: Path = ROOT) -> dict[str, Any]:
    p = base.relaxed_protocol()
    p.update({
        'name': 'GIR-Field formal full frozen-evidence protocol',
        'claim_scope': 'formal-full-frozen-evidence',
        'max_images_per_split': None,
        'bootstrap_samples': 5000,
        'risk_max_patches': 20000,
        'dataset_counts': full_dataset_counts(root),
        'formal_scope_note': 'Full local UEFB-v2 test, LOL-v2-real test, and LOL-v2-synthetic test; frozen checkpoints only; no new model training.',
    })
    return p


def planned_formal_phases() -> list[dict[str, str]]:
    return base.planned_phases()


def planned_formal_outputs(root: Path = ROOT) -> dict[str, Path]:
    outputs = base.planned_outputs(root)
    remapped: dict[str, Path] = {}
    for key, path in outputs.items():
        text = str(path)
        text = text.replace('/results/girfield_relaxed/', '/results/girfield_formal/')
        text = text.replace('/docs/GIR_FIELD_RELAXED_EXECUTION_PLAN_ZH.md', '/docs/GIR_FIELD_FORMAL_PROTOCOL_PLAN_ZH.md')
        text = text.replace('/docs/GIR_FIELD_RELAXED_PIPELINE_REPORT.md', '/docs/GIR_FIELD_FORMAL_PROTOCOL_REPORT.md')
        text = text.replace('/results/girfield_formal/GIR_FIELD_RELAXED_PIPELINE_MANIFEST.json', '/results/girfield_formal/GIR_FIELD_FORMAL_PROTOCOL_MANIFEST.json')
        text = text.replace('/results/girfield_formal/GIR_FIELD_RELAXED_VALIDATION.md', '/results/girfield_formal/GIR_FIELD_FORMAL_PROTOCOL_VALIDATION.md')
        remapped[key] = Path(text)
    return remapped


def formal_execution_plan(protocol: dict[str, Any]) -> str:
    counts = protocol['dataset_counts']
    return f"""# GIR-Field formal full protocol plan

Source guidance: `{SOURCE_GUIDANCE}`

## Scope

- Claim scope: `{protocol['claim_scope']}`
- No main-model training: `{protocol['no_main_training']}`
- DGB revival allowed: `{protocol['dgb_revival_allowed']}`
- Frozen checkpoints only: `{protocol['uses_frozen_checkpoints']}`
- Bootstrap samples: `{protocol['bootstrap_samples']}`
- Risk max patches: `{protocol['risk_max_patches']}`

## Full local dataset counts

- UEFB-v2 test: {counts['uefb']}
- LOL-v2 real test: {counts['real']}
- LOL-v2 synthetic test: {counts['synthetic']}
- Per variant total: {counts['total_per_variant']}
- Expected per-image rows for 9 variants: {counts['expected_per_image_rows_for_9_variants']}

## Formal phases

N0 evidence freeze; N1 full per-image statistical audit; N2 full UEFB-G gauge perturbation; N3 external output-only registry; N4 larger recoverability-risk calibration probe; N5 formal tables/figures/report/manifest.

## Guardrails

This formal protocol validates GIR-Field/UEFB-G mechanism and benchmark evidence. It does not claim SOTA LLIE, does not revive DGB, and does not compare internal E/S/Gauge metrics for black-box baselines.
"""


def rows_in_csv(path: Path) -> int:
    with path.open(encoding='utf-8') as f:
        return max(0, sum(1 for _ in f) - 1)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def find_stat(stat_rows: list[dict[str, Any]], comparison: str, dataset: str, metric: str) -> dict[str, Any] | None:
    for r in stat_rows:
        if r.get('comparison') == comparison and r.get('dataset') == dataset and r.get('metric') == metric:
            return r
    return None


def write_formal_report(outputs: dict[str, Path], protocol: dict[str, Any], table_rows: list[dict[str, Any]], risk_summary: list[dict[str, Any]]) -> None:
    stat_rows = read_csv_rows(outputs['stat_tests_csv'])
    counts = protocol['dataset_counts']
    key_stats = [
        ('M4J_ES_minus_M4J', 'uefb', 'S_corr'),
        ('M4J_ES_minus_M4J', 'real', 'psnr'),
        ('M4J_ES_minus_M4J', 'synthetic', 'psnr'),
        ('M4J_ES_minus_M4', 'uefb', 'S_corr'),
    ]
    lines = [
        '# GIR-Field formal full protocol report',
        '',
        '## Scope',
        '',
        f"- Claim scope: `{protocol['claim_scope']}`",
        '- Full local evaluation: UEFB-v2 test + LOL-v2 real test + LOL-v2 synthetic test.',
        f"- Dataset counts: UEFB={counts['uefb']}, real={counts['real']}, synthetic={counts['synthetic']}, expected per-image rows={counts['expected_per_image_rows_for_9_variants']}.",
        f"- Bootstrap samples: {protocol['bootstrap_samples']}.",
        '- No main-model training; frozen checkpoints only; DGB/P2F not resumed.',
        '',
        '## Key formal statistics',
        '',
    ]
    for comp, dataset, metric in key_stats:
        row = find_stat(stat_rows, comp, dataset, metric)
        if row:
            unit = ' dB' if metric == 'psnr' else ''
            lines.append(f"- {comp}, {dataset}, {metric}: delta={float(row['mean_delta']):.4f}{unit}, 95% CI=[{float(row['bootstrap95_lo']):.4f}, {float(row['bootstrap95_hi']):.4f}], Wilcoxon p={row.get('wilcoxon_p')}, FDR q={row.get('bh_fdr_q')}, win_rate={float(row['win_rate_b_gt_a']):.3f}.")
    lines.extend([
        '',
        '## Risk calibration probe',
        '',
    ])
    for r in risk_summary:
        lines.append(f"- {r.get('variant_id')}: status={r.get('status')}, AUC={r.get('logistic_auc')}, ECE={r.get('logistic_ece')}, test positive rate={r.get('positive_rate_test')}.")
    lines.extend([
        '',
        '## Claim calibration',
        '',
        '- Supported: centered E-shape improves gauge-free exposure-shape identifiability over M4J under full local eval.',
        '- Supported: UEFB-G gauge perturbation separates global gauge shifts from local shape distortions.',
        '- Diagnostic only: recoverability risk has signal but remains calibration-limited.',
        '- Not supported: DGB-RLEF SOTA, DGB tri-domain resolution, Retinexformer outperformance.',
        '',
        '## Artifacts',
        '',
        f"- `{outputs['per_image_csv'].relative_to(ROOT)}`",
        f"- `{outputs['stat_tests_csv'].relative_to(ROOT)}`",
        f"- `{outputs['gauge_perturbation_csv'].relative_to(ROOT)}`",
        f"- `{outputs['risk_calibration_csv'].relative_to(ROOT)}`",
        f"- `{outputs['main_table_md'].relative_to(ROOT)}`",
        f"- `{outputs['manifest_json'].relative_to(ROOT)}`",
    ])
    outputs['report_md'].parent.mkdir(parents=True, exist_ok=True)
    outputs['report_md'].write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_formal_manifest(outputs: dict[str, Path], protocol: dict[str, Any]) -> dict[str, Any]:
    files = []
    for key, path in outputs.items():
        if path.exists() and path.is_file():
            files.append({'key': key, 'relative_path': str(path.relative_to(ROOT)), 'size_bytes': path.stat().st_size, 'sha256': base.sha256_file(path)})
    manifest = {
        'pipeline': 'GIR-Field formal full frozen-evidence protocol',
        'created_at_unix': time.time(),
        'protocol': protocol,
        'phases': planned_formal_phases(),
        'file_count': len(files),
        'files': files,
    }
    base.write_json(outputs['manifest_json'], manifest)
    return manifest


def validate_formal_outputs(outputs: dict[str, Path], manifest: dict[str, Any], protocol: dict[str, Any]) -> str:
    errors = []
    expected = protocol['dataset_counts']['expected_per_image_rows_for_9_variants']
    actual = rows_in_csv(outputs['per_image_csv'])
    if actual != expected:
        errors.append(f'per_image_rows_expected_{expected}_actual_{actual}')
    for key in ['claim_ledger_csv', 'per_image_csv', 'summary_csv', 'stat_tests_csv', 'gauge_perturbation_csv', 'risk_calibration_csv', 'main_table_csv', 'report_md', 'manifest_json']:
        if not outputs[key].exists() or outputs[key].stat().st_size == 0:
            errors.append(f'missing_or_empty:{key}')
    for key in ['per_image_json', 'summary_json', 'stat_tests_json', 'gauge_perturbation_json', 'external_registry_json', 'risk_calibration_json', 'main_table_json', 'manifest_json']:
        try:
            base.read_json(outputs[key])
        except Exception as e:
            errors.append(f'json_error:{key}:{e}')
    for key in ['stat_figure', 'gauge_figure', 'risk_figure', 'failure_grid']:
        try:
            from PIL import Image
            Image.open(outputs[key]).verify()
        except Exception as e:
            errors.append(f'image_error:{key}:{e}')
    status = 'PASS' if not errors else 'FAIL'
    text = '# GIR-Field formal protocol validation\n\n' + f'- status: `{status}`\n' + f'- expected per-image rows: {expected}\n' + f'- actual per-image rows: {actual}\n' + f'- manifest files: {manifest.get("file_count")}\n' + ''.join(f'- error: {e}\n' for e in errors)
    outputs['validation_md'].write_text(text, encoding='utf-8')
    if errors:
        raise RuntimeError('; '.join(errors))
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--bootstrap-samples', type=int, default=5000)
    ap.add_argument('--risk-max-patches', type=int, default=20000)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    # Repoint relaxed-pipeline helper globals so visual grids and reports land in formal output root.
    base.OUT_ROOT = FORMAL_OUT
    outputs = planned_formal_outputs(ROOT)
    for p in outputs.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    protocol = formal_protocol(ROOT)
    protocol['bootstrap_samples'] = int(args.bootstrap_samples)
    protocol['risk_max_patches'] = int(args.risk_max_patches)

    evidence_rows, claim_rows = base.phase_n0(outputs, protocol)
    outputs['execution_plan_md'].write_text(formal_execution_plan(protocol), encoding='utf-8')
    per_rows, summary_rows, stat_rows = base.phase_n1(outputs, args.device, None, protocol['bootstrap_samples'], force=args.force)
    perturb_rows, meta_rows = base.phase_n2(outputs, None)
    external_rows = base.phase_n3(outputs)
    patch_rows, risk_summary = base.phase_n4(outputs, args.device, None, protocol['risk_patch_size'], protocol['risk_max_patches'])
    table_rows = base.phase_n5(outputs, summary_rows, stat_rows, perturb_rows, risk_summary, external_rows)
    write_formal_report(outputs, protocol, table_rows, risk_summary)
    manifest = build_formal_manifest(outputs, protocol)
    validation = validate_formal_outputs(outputs, manifest, protocol)
    print(json.dumps({'DONE': True, 'report': str(outputs['report_md']), 'manifest': str(outputs['manifest_json']), 'validation': validation.splitlines()[2]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
