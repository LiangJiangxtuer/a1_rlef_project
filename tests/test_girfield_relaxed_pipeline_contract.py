from pathlib import Path

from scripts.run_girfield_relaxed_pipeline import (
    ROOT,
    apply_bh_fdr,
    compute_gauge_metrics_np,
    planned_phases,
    planned_outputs,
    planned_variants,
    relaxed_protocol,
)


def test_relaxed_protocol_forbids_dgb_revival_and_main_training():
    protocol = relaxed_protocol()
    assert protocol['no_main_training'] is True
    assert protocol['dgb_revival_allowed'] is False
    assert protocol['max_images_per_split'] > 0
    assert protocol['bootstrap_samples'] > 0
    assert protocol['claim_scope'] == 'relaxed-routing-evidence-not-paper-grade-final'


def test_pipeline_has_n0_to_n5_and_required_outputs():
    phases = [p['id'] for p in planned_phases()]
    assert phases == ['N0', 'N1', 'N2', 'N3', 'N4', 'N5']
    outputs = planned_outputs(ROOT)
    required = [
        'claim_ledger_csv',
        'per_image_csv',
        'stat_tests_csv',
        'gauge_perturbation_csv',
        'risk_calibration_csv',
        'main_table_csv',
        'report_md',
        'manifest_json',
    ]
    for key in required:
        assert key in outputs
        assert str(outputs[key]).startswith(str(ROOT))


def test_variants_include_required_positive_and_control_routes():
    variants = planned_variants(ROOT)
    ids = {v['id'] for v in variants}
    assert {'M4', 'M4J', 'M4J_ES', 'P3C_E0050_S42', 'P3C_E0050_S2027', 'P3C_E0050_S3407', 'P3D_E0100_S42', 'P3D_E0100_S2027', 'P3D_E0100_S3407'} <= ids
    for variant in variants:
        assert variant['config'].exists(), variant['config']
        assert variant['checkpoint'].exists(), variant['checkpoint']
        assert variant['role'] in {'counterexample', 'joint_baseline', 'centered_shape_positive', 'default_3seed', 'stronger_shape_control'}


def test_gauge_metrics_separate_global_gauge_from_shape_error():
    import numpy as np
    e = np.linspace(-1, 1, 16, dtype='float32').reshape(1, 4, 4)
    shifted = e + 0.75
    distorted = e.copy()
    distorted[:, :2, :] += 0.5
    distorted = distorted - distorted.mean() + e.mean()
    global_metrics = compute_gauge_metrics_np(shifted, e)
    shape_metrics = compute_gauge_metrics_np(distorted, e)
    assert global_metrics['gauge_mae'] > 0.7
    assert global_metrics['s_mae'] < 1e-5
    assert global_metrics['s_corr'] > 0.999
    assert shape_metrics['gauge_mae'] < 1e-5
    assert shape_metrics['s_mae'] > 0.1
    assert shape_metrics['s_corr'] < 0.999


def test_bh_fdr_monotone_and_not_smaller_than_raw_p():
    pvals = [0.001, 0.04, 0.02, 0.20]
    qvals = apply_bh_fdr(pvals)
    assert len(qvals) == len(pvals)
    for p, q in zip(pvals, qvals):
        assert q >= p
        assert 0 <= q <= 1
    ordered = sorted(zip(pvals, qvals))
    assert [q for _, q in ordered] == sorted(q for _, q in ordered)
