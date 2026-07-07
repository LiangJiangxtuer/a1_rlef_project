from pathlib import Path

from scripts.run_s1_s4_paper_pipeline import (
    ROOT,
    build_appendix_rows,
    build_core_table_rows,
    planned_stages,
    planned_outputs,
)


def test_pipeline_declares_exact_s1_to_s4_stages():
    stages = planned_stages()
    assert [stage['id'] for stage in stages] == ['S1', 'S2', 'S3', 'S4']
    assert stages[0]['name'] == 'Evidence freeze + paper framing'
    assert stages[1]['name'] == 'Manuscript-grade main table'
    assert stages[2]['name'] == 'Qualitative diagnostic visualization'
    assert stages[3]['name'] == 'Appendix negative-route diagnostics'


def test_core_table_is_compact_mainline_not_negative_route_dump():
    rows = build_core_table_rows(ROOT)
    row_ids = [row['row_id'] for row in rows]
    assert 'M0_RESTORER_ONLY' in row_ids
    assert 'M4_GATE' in row_ids
    assert 'M4J' in row_ids
    assert 'M4J_ES_SEED3407' in row_ids
    assert 'P3C_M4J_ES_E005_MEAN' in row_ids
    assert 'RETINEXFORMER_OFFICIAL' in row_ids
    assert 'ZERO_DCE_PP_OFFICIAL' in row_ids
    assert not any(row_id.startswith('DGB_') for row_id in row_ids)
    assert not any(row_id.startswith('P7') for row_id in row_ids)
    p3c = next(row for row in rows if row['row_id'] == 'P3C_M4J_ES_E005_MEAN')
    assert p3c['evidence_level'] == '3-seed default'
    assert p3c['role'] == 'conservative default'


def test_appendix_rows_capture_stopped_routes():
    rows = build_appendix_rows(ROOT)
    route_names = {row['route'] for row in rows}
    assert {'P5/P5b distillation', 'P6/P6b/P6c structural scalar controls', 'P7/P7b/P7c domain heads', 'DGB branch'} <= route_names
    dgb = next(row for row in rows if row['route'] == 'DGB branch')
    assert dgb['decision'] == 'stopped and consolidated'
    assert 'joint gate failed' in dgb['reason']


def test_planned_outputs_are_under_results_paper_pipeline():
    outputs = planned_outputs(ROOT)
    assert outputs['root'].relative_to(ROOT) == Path('results/paper_pipeline')
    for key, path in outputs.items():
        if key == 'root':
            continue
        assert str(path).startswith(str(outputs['root']))
