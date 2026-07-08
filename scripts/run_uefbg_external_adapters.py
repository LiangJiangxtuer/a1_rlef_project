#!/usr/bin/env python3
"""Run I2 external field-aware adapter study for UEFB-G.

This script reuses frozen P4 official baseline output images and computes an
adapter-derived exposure-field proxy for each black-box method:

    E_hat = log(Y_pred + eps) - log(Y_low + eps)
    E_gt  = log(Y_high + eps) - log(Y_low + eps)

The resulting rows are valid `field_aware` submissions to the UEFB-G v1 public
evaluator, but the report explicitly frames them as adapter/proxy evidence, not
as proof that the external method internally predicts this field.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlef.adapters.exposure_field_adapters import (  # noqa: E402
    ExternalAdapterSpec,
    evaluate_adapter_triplet,
    summarize_adapter_rows,
    write_csv,
)
from scripts.uefbg_eval import write_evaluation_bundle  # noqa: E402

DETAIL_FILES = {
    ('Retinexformer', 'real'): ROOT / 'results/tables/p4_details/retinexformer_real_per_image.csv',
    ('Retinexformer', 'synthetic'): ROOT / 'results/tables/p4_details/retinexformer_synthetic_per_image.csv',
    ('Zero-DCE++', 'real'): ROOT / 'results/tables/p4_details/zero_dcepp_real_per_image.csv',
    ('Zero-DCE++', 'synthetic'): ROOT / 'results/tables/p4_details/zero_dcepp_synthetic_per_image.csv',
    ('KinD++', 'real'): ROOT / 'results/tables/p4_details/kindpp_real_per_image.csv',
    ('KinD++', 'synthetic'): ROOT / 'results/tables/p4_details/kindpp_synthetic_per_image.csv',
}

DEFAULT_ADAPTERS = [
    ExternalAdapterSpec(name='log_ratio_raw', smoothing_radius=0.0),
    ExternalAdapterSpec(name='log_ratio_lowpass_r8', smoothing_radius=8.0),
]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def output_metric_subset(row: dict[str, Any]) -> dict[str, Any]:
    keep = ['psnr', 'ssim', 'lee', 'nai', 'input_psnr', 'identity_drop', 'over', 'under']
    return {k: row[k] for k in keep if k in row and row[k] != ''}


def collect_external_adapter_rows(max_images: int | None = None) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for (method, dataset), detail_path in DETAIL_FILES.items():
        if not detail_path.exists():
            raise FileNotFoundError(f'missing P4 detail file: {detail_path}')
        source_rows = read_csv_rows(detail_path)
        if max_images is not None:
            source_rows = source_rows[:max_images]
        for adapter in DEFAULT_ADAPTERS:
            for idx, row in enumerate(source_rows):
                all_rows.append(
                    evaluate_adapter_triplet(
                        method=method,
                        dataset=dataset,
                        key=str(row.get('key') or row.get('name') or idx),
                        pred_path=Path(row['pred_path']),
                        low_path=Path(row['low_path']),
                        high_path=Path(row['high_path']),
                        adapter=adapter,
                        output_metrics=output_metric_subset(row),
                    )
                )
    return all_rows


def format_num(x: Any, nd: int = 4) -> str:
    try:
        return f'{float(x):.{nd}f}'
    except Exception:
        return str(x)


def write_report(out_dir: Path, per_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], validation: dict[str, Any]) -> Path:
    report = out_dir / 'EXTERNAL_FIELD_ADAPTER_STUDY_REPORT.md'
    lines = [
        '# I2 External field-aware adapter study',
        '',
        'Date: 2026-07-08',
        '',
        '## Scope',
        '',
        'This study evaluates whether frozen P4 external baseline outputs can be mapped into UEFB-G field-aware reporting through a defensible luminance-gain adapter.',
        '',
        'Adapter definition:',
        '',
        '```text',
        'E_hat = log(Y_pred + eps) - log(Y_low + eps)',
        'E_gt  = log(Y_high + eps) - log(Y_low + eps)',
        '```',
        '',
        'Two adapter variants are reported:',
        '',
        '- `log_ratio_raw`: raw per-pixel luminance-gain proxy.',
        '- `log_ratio_lowpass_r8`: dependency-light box low-pass illumination-scale proxy, radius 8 px.',
        '',
        'Important claim guardrail: these are adapter-derived proxies. They do not prove that Retinexformer, Zero-DCE++, or KinD++ internally predicts this exact field.',
        '',
        '## Execution summary',
        '',
        f'- Per-image rows: {len(per_rows)}',
        f"- UEFB-G validation status: `{validation.get('status')}`",
        f"- Methods: {', '.join(sorted({r['method'] for r in per_rows}))}",
        f"- Datasets: {', '.join(sorted({r['dataset'] for r in per_rows}))}",
        '',
        '## Summary table',
        '',
        '| Method | Dataset | Adapter | n | PSNR↑ | S_corr↑ | S_MAE↓ | Gauge_MAE↓ | E_MAE↓ |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in summary_rows:
        lines.append(
            '| {method} | {dataset} | {adapter} | {n} | {psnr} | {scorr} | {smae} | {gauge} | {emae} |'.format(
                method=r['method'],
                dataset=r['dataset'],
                adapter=r['adapter'],
                n=r['n'],
                psnr=format_num(r.get('psnr_mean')),
                scorr=format_num(r.get('S_corr_mean')),
                smae=format_num(r.get('S_MAE_mean')),
                gauge=format_num(r.get('Gauge_MAE_mean')),
                emae=format_num(r.get('E_MAE_mean')),
            )
        )
    # Simple evidence rule for go/no-go: blind methods, both splits, lowpass S_corr >= 0.5.
    lowpass_rows = [r for r in summary_rows if r['adapter'] == 'log_ratio_lowpass_r8']
    nontrivial = [r for r in lowpass_rows if r['method'] != 'KinD++' and isinstance(r.get('S_corr_mean'), float) and r['S_corr_mean'] >= 0.5]
    methods_with_signal = sorted({r['method'] for r in nontrivial})
    lines.extend([
        '',
        '## Interpretation',
        '',
        f'Blind external methods with lowpass adapter `S_corr >= 0.5`: {methods_with_signal if methods_with_signal else "none"}.',
        '',
    ])
    if len(methods_with_signal) >= 2:
        lines.extend([
            '**GO.** At least two blind external methods produce nontrivial adapter-field shape signal, so UEFB-G can be framed as applicable beyond the internal RLEF family when adapter scope is clearly stated.',
        ])
    elif len(methods_with_signal) == 1:
        lines.extend([
            '**PARTIAL GO.** One blind external method produces nontrivial adapter-field shape signal. UEFB-G can include an adapter study, but the main benchmark claim should remain conservative: explicit-field methods are primary; black-box methods are adapter/proxy analyses.',
        ])
    else:
        lines.extend([
            '**NO-GO for broad black-box field-aware claims.** External adapters are too weak/arbitrary for main-paper benchmark expansion; keep black-box methods output-only.',
        ])
    lines.extend([
        '',
        '## Artifact list',
        '',
        '```text',
        'external_adapter_per_image.csv',
        'external_adapter_summary.csv',
        'uefbg_v1_bundle/',
        'EXTERNAL_FIELD_ADAPTER_STUDY_REPORT.md',
        '```',
        '',
    ])
    report.write_text('\n'.join(lines), encoding='utf-8')
    return report


def run(out_dir: Path, max_images: int | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_rows = collect_external_adapter_rows(max_images=max_images)
    summary_rows = summarize_adapter_rows(per_rows)
    per_csv = out_dir / 'external_adapter_per_image.csv'
    summary_csv = out_dir / 'external_adapter_summary.csv'
    write_csv(per_csv, per_rows)
    write_csv(summary_csv, summary_rows)
    (out_dir / 'external_adapter_summary.json').write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding='utf-8')
    bundle = write_evaluation_bundle(
        per_rows,
        out_dir=out_dir / 'uefbg_v1_bundle',
        protocol={
            'name': 'UEFB-G external field-aware adapter protocol',
            'version': 'i2-v1',
            'adapter_claim_boundary': 'adapter-derived luminance gain proxy, not internal field proof',
        },
        source_metrics_path=str(per_csv),
    )
    report = write_report(out_dir, per_rows, summary_rows, bundle['validation'])
    payload = {
        'DONE': True,
        'out_dir': str(out_dir),
        'n_rows': len(per_rows),
        'n_summary_rows': len(summary_rows),
        'validation_status': bundle['validation']['status'],
        'per_image_csv': str(per_csv),
        'summary_csv': str(summary_csv),
        'report': str(report),
        'uefbg_bundle': str(out_dir / 'uefbg_v1_bundle'),
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results/uefbg_external_adapters'))
    ap.add_argument('--max-images', type=int, default=None)
    args = ap.parse_args()
    print(json.dumps(run(Path(args.out), max_images=args.max_images), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
