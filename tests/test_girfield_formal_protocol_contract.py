from pathlib import Path

from scripts.run_girfield_formal_protocol import (
    ROOT,
    formal_protocol,
    full_dataset_counts,
    planned_formal_outputs,
    planned_formal_phases,
)


def test_formal_protocol_is_full_frozen_and_not_dgb_revival():
    protocol = formal_protocol(ROOT)
    assert protocol['claim_scope'] == 'formal-full-frozen-evidence'
    assert protocol['no_main_training'] is True
    assert protocol['dgb_revival_allowed'] is False
    assert protocol['uses_frozen_checkpoints'] is True
    assert protocol['max_images_per_split'] is None
    assert protocol['bootstrap_samples'] >= 5000
    assert protocol['risk_max_patches'] >= 20000


def test_formal_outputs_are_separate_from_relaxed_outputs():
    outputs = planned_formal_outputs(ROOT)
    assert outputs['manifest_json'] == ROOT / 'results/girfield_formal/GIR_FIELD_FORMAL_PROTOCOL_MANIFEST.json'
    assert outputs['report_md'] == ROOT / 'docs/GIR_FIELD_FORMAL_PROTOCOL_REPORT.md'
    assert outputs['per_image_csv'].parts[-3:] == ('girfield_formal', 'N1_statistics', 'per_image_metrics.csv')
    assert 'girfield_relaxed' not in str(outputs['manifest_json'])


def test_formal_phases_preserve_n0_to_n5():
    assert [p['id'] for p in planned_formal_phases()] == ['N0', 'N1', 'N2', 'N3', 'N4', 'N5']


def test_full_dataset_counts_are_actual_local_full_counts():
    counts = full_dataset_counts(ROOT)
    assert counts['uefb'] >= 500
    assert counts['real'] == 100
    assert counts['synthetic'] == 100
    assert counts['total_per_variant'] == counts['uefb'] + counts['real'] + counts['synthetic']
    assert counts['expected_per_image_rows_for_9_variants'] == 9 * counts['total_per_variant']
