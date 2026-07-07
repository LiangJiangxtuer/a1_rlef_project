from pathlib import Path

from scripts.run_master_remaining_pipeline import (
    ROOT,
    build_final_ablation_rows,
    build_final_main_rows,
    planned_outputs,
    planned_remaining_steps,
    proxy_metric_keys,
)


def test_remaining_steps_follow_current_master_prompt_after_s1_s4():
    steps = planned_remaining_steps()
    assert [step['id'] for step in steps] == [
        'P2_NOREF_SUPPLEMENT',
        'FINAL_TABLES',
        'FINAL_REPORT',
        'PAPER_IDEA',
        'REPRO_CHECKLIST',
        'FINAL_FIGURE',
    ]
    assert steps[0]['uses_new_training'] is False
    assert steps[0]['allowed_claim_scope'] == 'support_or_limitation_only'


def test_proxy_metric_contract_is_no_reference_and_available_without_brisque():
    keys = proxy_metric_keys()
    assert {'niqe', 'over', 'under', 'dark_ratio', 'contrast', 'sharpness_proxy', 'noise_proxy'} <= set(keys)
    assert 'psnr' not in keys
    assert 'brisque' not in keys  # package unavailable; report as limitation, do not fabricate it


def test_final_main_rows_preserve_honest_claim_boundary():
    rows = build_final_main_rows(ROOT)
    methods = [row['method'] for row in rows]
    assert 'P3c M4J_ES e_shape=0.05' in methods
    assert 'Retinexformer' in methods
    assert 'DGB branch' in methods
    dgb = next(row for row in rows if row['method'] == 'DGB branch')
    assert dgb['decision'] == 'stopped; diagnostic only'
    assert dgb['main_claim_allowed'] is False


def test_final_ablation_rows_route_negative_evidence_to_appendix():
    rows = build_final_ablation_rows(ROOT)
    groups = {row['group'] for row in rows}
    assert {'Gauge / E-shape', 'Distillation', 'Structural backbone', 'Domain heads', 'DGB controlled isolation'} <= groups
    assert all(row['paper_location'] in {'main ablation', 'appendix negative evidence', 'limitation'} for row in rows)


def test_planned_outputs_match_master_prompt_final_targets():
    outputs = planned_outputs(ROOT)
    assert outputs['final_report'].relative_to(ROOT) == Path('docs/DGB_RLEF_FINAL_EXPERIMENT_REPORT.md')
    assert outputs['paper_idea'].relative_to(ROOT) == Path('docs/DGB_RLEF_PAPER_IDEA.md')
    assert outputs['repro_checklist'].relative_to(ROOT) == Path('docs/DGB_RLEF_REPRODUCIBILITY_CHECKLIST.md')
    assert outputs['final_main_table'].relative_to(ROOT) == Path('results/hermes_audit/tables/final_main_table.csv')
    assert outputs['final_ablation_table'].relative_to(ROOT) == Path('results/hermes_audit/tables/final_ablation_table.csv')
    assert outputs['final_qualitative_grid'].relative_to(ROOT) == Path('results/hermes_audit/figures/final_qualitative_grid.png')
