import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from rlef.adapters.exposure_field_adapters import (
    ExternalAdapterSpec,
    compute_adapter_fields,
    compute_gauge_metrics,
    evaluate_adapter_triplet,
    summarize_adapter_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def test_log_ratio_adapter_is_perfect_when_prediction_matches_high_image(tmp_path):
    low = np.full((4, 4, 3), 0.10, dtype=np.float32)
    high = np.full((4, 4, 3), 0.40, dtype=np.float32)
    low_path = tmp_path / 'low.png'
    high_path = tmp_path / 'high.png'
    pred_path = tmp_path / 'pred.png'
    _write_rgb(low_path, low)
    _write_rgb(high_path, high)
    _write_rgb(pred_path, high)

    row = evaluate_adapter_triplet(
        method='toy',
        dataset='toyset',
        key='toy_0001',
        pred_path=pred_path,
        low_path=low_path,
        high_path=high_path,
        adapter=ExternalAdapterSpec(name='log_ratio_raw', smoothing_radius=0.0),
    )

    assert row['reporting_mode'] == 'field_aware_adapter'
    assert row['E_MAE'] < 1e-6
    assert row['S_MAE'] < 1e-6
    assert row['Gauge_MAE'] < 1e-6
    # Constant fields have no shape energy, so correlation is intentionally N/A.
    assert row['S_corr'] == 'N/A_constant_shape'


def test_adapter_metrics_separate_global_gain_from_local_shape():
    low = np.full((2, 2, 3), 0.10, dtype=np.float32)
    high = np.array(
        [
            [[0.20, 0.20, 0.20], [0.30, 0.30, 0.30]],
            [[0.40, 0.40, 0.40], [0.50, 0.50, 0.50]],
        ],
        dtype=np.float32,
    )
    pred_global = np.clip(high * 1.25, 0, 1)
    pred_local = high.copy()
    pred_local[0, 0, :] = 0.50
    pred_local[1, 1, :] = 0.20

    gt_field, global_field = compute_adapter_fields(low, pred_global, high, smoothing_radius=0.0)
    global_metrics = compute_gauge_metrics(global_field, gt_field)
    assert global_metrics['Gauge_MAE'] > 0.1
    # The adapter uses a small log epsilon for dark-pixel stability, so a pure
    # multiplicative gain is nearly—but not mathematically exactly—a constant
    # gauge shift.
    assert global_metrics['S_MAE'] < 1e-3
    assert global_metrics['S_corr'] > 0.999

    gt_field, local_field = compute_adapter_fields(low, pred_local, high, smoothing_radius=0.0)
    local_metrics = compute_gauge_metrics(local_field, gt_field)
    assert local_metrics['S_MAE'] > 0.1
    assert local_metrics['S_corr'] < 0.5


def test_summarize_adapter_rows_keeps_method_dataset_adapter_groups():
    rows = [
        {'method': 'A', 'dataset': 'real', 'adapter': 'raw', 'psnr': 10.0, 'S_corr': 0.1, 'Gauge_MAE': 0.2, 'E_MAE': 0.3, 'S_MAE': 0.4},
        {'method': 'A', 'dataset': 'real', 'adapter': 'raw', 'psnr': 14.0, 'S_corr': 0.3, 'Gauge_MAE': 0.4, 'E_MAE': 0.5, 'S_MAE': 0.6},
        {'method': 'B', 'dataset': 'synthetic', 'adapter': 'raw', 'psnr': 20.0, 'S_corr': 'N/A_constant_shape', 'Gauge_MAE': 0.1, 'E_MAE': 0.1, 'S_MAE': 0.0},
    ]
    summary = summarize_adapter_rows(rows)
    by_key = {(r['method'], r['dataset'], r['adapter']): r for r in summary}
    assert by_key[('A', 'real', 'raw')]['n'] == 2
    assert abs(by_key[('A', 'real', 'raw')]['psnr_mean'] - 12.0) < 1e-9
    assert abs(by_key[('A', 'real', 'raw')]['S_corr_mean'] - 0.2) < 1e-9
    assert by_key[('B', 'synthetic', 'raw')]['S_corr_mean'] == 'N/A'


def test_external_adapter_cli_writes_v1_bundle(tmp_path):
    out_dir = tmp_path / 'external_adapter_smoke'
    cmd = [
        '/home/user/miniconda3/envs/cutler_dinov3/bin/python',
        'scripts/run_uefbg_external_adapters.py',
        '--max-images', '3',
        '--out', str(out_dir),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(completed.stdout)
    assert payload['DONE'] is True
    assert payload['validation_status'] == 'PASS'
    assert payload['n_rows'] == 36  # 3 methods * 2 datasets * 2 adapters * 3 images
    assert (out_dir / 'external_adapter_per_image.csv').exists()
    assert (out_dir / 'external_adapter_summary.csv').exists()
    assert (out_dir / 'uefbg_v1_bundle/validation.json').exists()
    rows = list(csv.DictReader((out_dir / 'external_adapter_per_image.csv').open(encoding='utf-8')))
    assert {r['method'] for r in rows} >= {'Retinexformer', 'Zero-DCE++', 'KinD++'}
    assert {r['adapter'] for r in rows} == {'log_ratio_raw', 'log_ratio_lowpass_r8'}
