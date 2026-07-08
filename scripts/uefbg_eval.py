#!/usr/bin/env python3
"""Standalone UEFB-G benchmark/evaluator utilities.

This module is intentionally independent from RLEF model/training internals. It
accepts public-style per-image metric submissions and optional field-aware
metrics, validates the reporting mode, and writes JSON/CSV/Markdown report
cards. Image/field-array evaluation can be added on top of the same schema; the
current v1 path hardens the formal frozen-evidence tables into a public
benchmark artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]

BASE_REQUIRED_COLUMNS = ('variant_id', 'dataset', 'name')
OUTPUT_METRICS = ('psnr', 'ssim', 'lee', 'nai', 'identity_drop', 'q_ece', 'over', 'under')
FIELD_METRICS = ('E_MAE', 'S_MAE', 'Gauge_MAE', 'S_corr')
AUX_FIELD_METRICS = ('E_MAE_aligned', 'E_corr', 'mu_pred', 'mu_gt', 'A_mean', 'Q_mean', 'mu_E')
SUMMARY_METRICS = OUTPUT_METRICS + FIELD_METRICS + AUX_FIELD_METRICS


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, str) and x.strip() in {'', 'N/A', 'NA', 'nan', 'None'}:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def mean(vals: Iterable[float]) -> float | None:
    xs = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return sum(xs) / len(xs) if xs else None


def std(vals: Iterable[float]) -> float | None:
    xs = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _shape_and_flatten_array(x: Any) -> tuple[tuple[int, ...], list[float]]:
    """Return a conservative shape plus flattened floats without requiring numpy.

    The evaluator is intended to be a public, lightweight checker. It should run
    even in a plain Python environment; numpy arrays are accepted when available,
    but numpy is not required by the CLI.
    """
    if hasattr(x, 'shape') and hasattr(x, 'ravel'):
        shape = tuple(int(v) for v in getattr(x, 'shape'))
        return shape, [float(v) for v in x.ravel()]
    if isinstance(x, (list, tuple)):
        if not x:
            return (0,), []
        if isinstance(x[0], (list, tuple)):
            child_shapes = []
            flat: list[float] = []
            for child in x:
                s, vals = _shape_and_flatten_array(child)
                child_shapes.append(s)
                flat.extend(vals)
            first = child_shapes[0]
            if any(s != first for s in child_shapes):
                raise ValueError('ragged array is not a valid exposure field')
            return (len(x),) + first, flat
        return (len(x),), [float(v) for v in x]
    return (), [float(x)]


def compute_gauge_metrics_np(e_pred: Any, e_gt: Any) -> dict[str, float]:
    """Compute UEFB-G gauge-aware field metrics for prediction and target arrays.

    `E = S + mu`, where `mu` is the scalar image gauge and `S` is the
    mean-centered shape. A global additive shift should change only E/Gauge
    metrics, not the centered shape metrics.
    """
    shape_p, ep = _shape_and_flatten_array(e_pred)
    shape_g, eg = _shape_and_flatten_array(e_gt)
    if shape_p != shape_g:
        raise ValueError(f'shape mismatch: {shape_p} vs {shape_g}')
    if not ep:
        raise ValueError('empty exposure field')
    mu_p = sum(ep) / len(ep)
    mu_g = sum(eg) / len(eg)
    sp = [v - mu_p for v in ep]
    sg = [v - mu_g for v in eg]
    norm_p = math.sqrt(sum(v * v for v in sp))
    norm_g = math.sqrt(sum(v * v for v in sg))
    denom = norm_p * norm_g + 1e-12
    s_corr = sum(a * b for a, b in zip(sp, sg)) / denom
    e_mae = sum(abs(a - b) for a, b in zip(ep, eg)) / len(ep)
    s_mae = sum(abs(a - b) for a, b in zip(sp, sg)) / len(sp)
    gauge_mae = abs(mu_p - mu_g)
    return {
        'E_MAE': e_mae,
        'S_MAE': s_mae,
        'Gauge_MAE': gauge_mae,
        'S_corr': float(s_corr),
        # Lowercase aliases make the public perturbation CSV easy to consume.
        'e_mae': e_mae,
        's_mae': s_mae,
        'gauge_mae': gauge_mae,
        's_corr': float(s_corr),
        'mu_pred': mu_p,
        'mu_gt': mu_g,
    }


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _has_value(row: dict[str, Any], key: str) -> bool:
    return safe_float(row.get(key)) is not None


def row_reporting_mode(row: dict[str, Any]) -> str:
    present = [m for m in FIELD_METRICS if _has_value(row, m)]
    if not present:
        return 'output_only'
    if len(present) == len(FIELD_METRICS):
        return 'field_aware'
    return 'partial_field'


def validate_metrics_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if not rows:
        errors.append('empty_submission')
    output_only = 0
    field_aware = 0
    partial = 0
    datasets: set[str] = set()
    variants: set[str] = set()
    for i, row in enumerate(rows):
        for col in BASE_REQUIRED_COLUMNS:
            if not str(row.get(col, '')).strip():
                errors.append(f'row_{i}_missing_required_column:{col}')
        mode = row_reporting_mode(row)
        if mode == 'output_only':
            output_only += 1
        elif mode == 'field_aware':
            field_aware += 1
        else:
            partial += 1
            present = [m for m in FIELD_METRICS if _has_value(row, m)]
            missing = [m for m in FIELD_METRICS if not _has_value(row, m)]
            errors.append(f'row_{i}_partial_field_metrics:present={present}:missing={missing}')
        if not any(_has_value(row, m) for m in OUTPUT_METRICS):
            errors.append(f'row_{i}_missing_all_output_metrics')
        if row.get('dataset'):
            datasets.add(str(row['dataset']))
        if row.get('variant_id'):
            variants.add(str(row['variant_id']))
    return {
        'status': 'PASS' if not errors else 'FAIL',
        'errors': errors,
        'n_rows': len(rows),
        'n_variants': len(variants),
        'n_datasets': len(datasets),
        'variants': sorted(variants),
        'datasets': sorted(datasets),
        'output_only_rows': output_only,
        'field_aware_rows': field_aware,
        'partial_field_rows': partial,
    }


def summarize_metrics_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get('variant_id', '')), str(row.get('dataset', '')))].append(row)
    out: list[dict[str, Any]] = []
    for (variant_id, dataset), rs in sorted(grouped.items()):
        modes = [row_reporting_mode(r) for r in rs]
        if any(m == 'field_aware' for m in modes):
            reporting_mode = 'field_aware'
        else:
            reporting_mode = 'output_only'
        summary: dict[str, Any] = {
            'variant_id': variant_id,
            'display': rs[0].get('display', variant_id),
            'role': rs[0].get('role', ''),
            'dataset': dataset,
            'n': len(rs),
            'reporting_mode': reporting_mode,
        }
        for metric in SUMMARY_METRICS:
            vals = [safe_float(r.get(metric)) for r in rs]
            vals = [v for v in vals if v is not None]
            if vals:
                summary[f'{metric}_mean'] = mean(vals)
                summary[f'{metric}_std'] = std(vals)
            else:
                summary[f'{metric}_mean'] = 'N/A' if metric in FIELD_METRICS + AUX_FIELD_METRICS else ''
                summary[f'{metric}_std'] = 'N/A' if metric in FIELD_METRICS + AUX_FIELD_METRICS else ''
        out.append(summary)
    return out


def report_cards_markdown(summary_rows: list[dict[str, Any]], validation: dict[str, Any]) -> str:
    fields = ['variant_id', 'dataset', 'n', 'reporting_mode', 'psnr_mean', 'ssim_mean', 'S_corr_mean', 'Gauge_MAE_mean']
    lines = [
        '# UEFB-G v1 report cards',
        '',
        f"Validation status: `{validation['status']}`",
        '',
        f"Rows: {validation['n_rows']} | Variants: {validation['n_variants']} | Datasets: {validation['n_datasets']}",
        '',
        '| ' + ' | '.join(fields) + ' |',
        '| ' + ' | '.join(['---'] * len(fields)) + ' |',
    ]
    for row in summary_rows:
        vals = []
        for f in fields:
            v = row.get(f, '')
            vals.append(f'{v:.4f}' if isinstance(v, float) else str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    lines.extend([
        '',
        '## Reporting rules',
        '',
        '- `output_only` rows are valid for black-box enhancers and report only RGB/output metrics.',
        '- `field_aware` rows must provide all core field metrics: `E_MAE`, `S_MAE`, `Gauge_MAE`, and `S_corr`.',
        '- Partial field reports are rejected to avoid mixing output-only and internal-field claims.',
    ])
    return '\n'.join(lines) + '\n'


def read_protocol(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {'name': 'UEFB-G public evaluator', 'version': 'v1'}
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        return json.loads(text)
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Conservative tiny YAML fallback for the protocol_v1.yaml shape.
    out: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith('  - ') and current_list_key:
            out.setdefault(current_list_key, []).append(line[4:].strip().strip('"\''))
            continue
        if ':' in line and not line.startswith(' '):
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if value == '':
                out[key] = []
                current_list_key = key
            else:
                current_list_key = None
                out[key] = value.strip('"\'')
    return out


def write_evaluation_bundle(
    rows: list[dict[str, Any]],
    out_dir: Path,
    protocol: dict[str, Any] | None = None,
    source_metrics_path: str | None = None,
) -> dict[str, Any]:
    protocol = protocol or {'name': 'UEFB-G public evaluator', 'version': 'v1'}
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_metrics_rows(rows)
    summary_rows = summarize_metrics_rows(rows)

    input_csv = out_dir / 'input_metrics.csv'
    summary_csv = out_dir / 'summary.csv'
    summary_json = out_dir / 'summary.json'
    validation_json = out_dir / 'validation.json'
    report_md = out_dir / 'report_cards.md'
    manifest_json = out_dir / 'manifest.json'

    write_csv(input_csv, rows)
    write_csv(summary_csv, summary_rows)
    write_json(summary_json, summary_rows)
    write_json(validation_json, validation)
    report_md.write_text(report_cards_markdown(summary_rows, validation), encoding='utf-8')

    files = []
    for key, path in {
        'input_metrics_csv': input_csv,
        'summary_csv': summary_csv,
        'summary_json': summary_json,
        'validation_json': validation_json,
        'report_cards_md': report_md,
    }.items():
        files.append({
            'key': key,
            'path': str(path),
            'size_bytes': path.stat().st_size,
            'sha256': sha256_file(path),
        })
    manifest = {
        'benchmark': 'UEFB-G',
        'version': protocol.get('version', 'v1'),
        'created_at_unix': time.time(),
        'protocol': protocol,
        'source_metrics_path': source_metrics_path,
        'n_input_rows': len(rows),
        'validation': validation,
        'files': files,
    }
    write_json(manifest_json, manifest)
    manifest['files'].append({
        'key': 'manifest_json',
        'path': str(manifest_json),
        'size_bytes': manifest_json.stat().st_size,
        'sha256': sha256_file(manifest_json),
    })
    return {
        'validation': validation,
        'summary_rows': summary_rows,
        'manifest': manifest,
        'out_dir': str(out_dir),
    }


def run_cli(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description='UEFB-G v1 standalone public evaluator')
    ap.add_argument('--input-metrics', required=True, help='Per-image metrics CSV with output metrics and optional field metrics.')
    ap.add_argument('--protocol', default=None, help='Protocol YAML/JSON metadata file.')
    ap.add_argument('--out', required=True, help='Output directory for public report bundle.')
    args = ap.parse_args(argv)

    input_path = Path(args.input_metrics)
    out_dir = Path(args.out)
    protocol = read_protocol(Path(args.protocol) if args.protocol else None)
    rows = read_csv_rows(input_path)
    bundle = write_evaluation_bundle(rows, out_dir=out_dir, protocol=protocol, source_metrics_path=str(input_path))
    payload = {
        'DONE': True,
        'out_dir': str(out_dir),
        'validation_status': bundle['validation']['status'],
        'n_rows': bundle['validation']['n_rows'],
        'n_variants': bundle['validation']['n_variants'],
        'n_datasets': bundle['validation']['n_datasets'],
        'manifest': str(out_dir / 'manifest.json'),
        'report': str(out_dir / 'report_cards.md'),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    run_cli()


if __name__ == '__main__':
    main()
