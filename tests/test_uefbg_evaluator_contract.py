import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.uefbg_eval import (
    compute_gauge_metrics_np,
    summarize_metrics_rows,
    validate_metrics_rows,
    write_evaluation_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def test_gauge_metrics_separate_global_shift_from_shape_error():
    gt = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    shifted = gt + 0.5
    distorted = gt.copy()
    distorted[0, 0] += 1.0
    distorted[1, 1] -= 1.0

    global_metrics = compute_gauge_metrics_np(shifted, gt)
    assert abs(global_metrics['gauge_mae'] - 0.5) < 1e-6
    assert global_metrics['s_mae'] < 1e-6
    assert global_metrics['s_corr'] > 0.999

    shape_metrics = compute_gauge_metrics_np(distorted, gt)
    assert shape_metrics['gauge_mae'] < 1e-6
    assert shape_metrics['s_mae'] > 0.0
    assert shape_metrics['s_corr'] < 0.999


def test_validate_metrics_rows_accepts_black_box_and_rejects_partial_field_rows():
    output_only = [
        {'variant_id': 'Retinexformer', 'dataset': 'real', 'name': '0001', 'psnr': '22.0', 'ssim': '0.88'},
    ]
    validation = validate_metrics_rows(output_only)
    assert validation['status'] == 'PASS'
    assert validation['field_aware_rows'] == 0
    assert validation['output_only_rows'] == 1

    partial_field = [
        {'variant_id': 'BadField', 'dataset': 'uefb', 'name': '0001', 'psnr': '18.0', 'S_corr': '0.3'},
    ]
    validation = validate_metrics_rows(partial_field)
    assert validation['status'] == 'FAIL'
    assert any('partial_field_metrics' in e for e in validation['errors'])


def test_summarize_metrics_rows_groups_by_variant_dataset_and_marks_field_mode():
    rows = [
        {'variant_id': 'A', 'display': 'A model', 'dataset': 'uefb', 'name': '1', 'psnr': '10', 'ssim': '0.5', 'S_corr': '0.1', 'Gauge_MAE': '0.2', 'E_MAE': '0.3', 'S_MAE': '0.4'},
        {'variant_id': 'A', 'display': 'A model', 'dataset': 'uefb', 'name': '2', 'psnr': '14', 'ssim': '0.7', 'S_corr': '0.3', 'Gauge_MAE': '0.4', 'E_MAE': '0.5', 'S_MAE': '0.6'},
        {'variant_id': 'B', 'display': 'B blackbox', 'dataset': 'real', 'name': '1', 'psnr': '20', 'ssim': '0.8'},
    ]
    summary = summarize_metrics_rows(rows)
    by_key = {(r['variant_id'], r['dataset']): r for r in summary}
    assert by_key[('A', 'uefb')]['n'] == 2
    assert abs(by_key[('A', 'uefb')]['psnr_mean'] - 12.0) < 1e-9
    assert abs(by_key[('A', 'uefb')]['S_corr_mean'] - 0.2) < 1e-9
    assert by_key[('A', 'uefb')]['reporting_mode'] == 'field_aware'
    assert by_key[('B', 'real')]['reporting_mode'] == 'output_only'
    assert by_key[('B', 'real')]['S_corr_mean'] == 'N/A'


def test_write_evaluation_bundle_reruns_current_formal_table_into_public_report(tmp_path):
    source = ROOT / 'results/girfield_formal/N1_statistics/per_image_metrics.csv'
    rows = []
    with source.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['variant_id'] in {'M4', 'M4J_ES'} and row['dataset'] == 'uefb' and int(row['sample_index']) < 5:
                rows.append(row)
    assert rows

    bundle = write_evaluation_bundle(
        rows,
        out_dir=tmp_path,
        protocol={'name': 'UEFB-G test protocol', 'version': 'v1'},
        source_metrics_path=str(source),
    )
    assert bundle['validation']['status'] == 'PASS'
    assert (tmp_path / 'summary.csv').exists()
    assert (tmp_path / 'report_cards.md').exists()
    report = (tmp_path / 'report_cards.md').read_text(encoding='utf-8')
    assert 'M4' in report
    assert 'M4J_ES' in report
    manifest = json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['protocol']['version'] == 'v1'
    assert manifest['n_input_rows'] == len(rows)


def test_public_cli_generates_internal_method_bundle(tmp_path):
    out_dir = tmp_path / 'bundle'
    cmd = [
        'python', 'scripts/run_uefbg_benchmark_v1.py',
        '--input-metrics', 'results/girfield_formal/N1_statistics/per_image_metrics.csv',
        '--protocol', 'configs/uefbg/protocol_v1.yaml',
        '--out', str(out_dir),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(completed.stdout)
    assert payload['DONE'] is True
    assert payload['validation_status'] == 'PASS'
    assert (out_dir / 'summary.csv').exists()
    assert (out_dir / 'report_cards.md').exists()
    assert (out_dir / 'validation.json').exists()
