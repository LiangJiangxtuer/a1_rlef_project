from pathlib import Path

from scripts.run_p5_retinex_distill import make_cfg, planned_runs, teacher_export_specs, teacher_tag


def test_p5_plans_single_retinexformer_distillation_route():
    runs = planned_runs()
    assert [r['id'] for r in runs] == ['P5_RD_T01', 'P5_RD_T03']
    by_id = {r['id']: r for r in runs}
    assert by_id['P5_RD_T01']['loss']['distill'] == 0.10
    assert by_id['P5_RD_T03']['loss']['distill'] == 0.30
    for run in runs:
        assert run['model']['gate_branch'] is True
        assert run['model']['exposure_branch'] is True
        assert run['loss']['e_shape'] == 0.05


def test_p5_teacher_export_specs_are_train_splits_with_official_weights():
    specs = teacher_export_specs()
    assert {s['dataset'] for s in specs} == {'real', 'synthetic'}
    for spec in specs:
        assert spec['low_dir'].endswith('/Train/Low')
        assert spec['high_dir'].endswith('/Train/Normal')
        assert spec['output_dir'].endswith(f"retinexformer_train_teacher/{spec['dataset']}")
        assert Path(spec['weights']).name.startswith('LOL_v2_')
        assert teacher_tag(spec) == f"retinexformer_train_{spec['dataset']}"


def test_p5_config_uses_teacher_dirs_only_for_lol_paired_sets():
    cfg = make_cfg(planned_runs()[0], max_steps=7, train_crop=64, batch_size=3)
    assert cfg['protocol']['stage'] == 'P5 Retinexformer teacher distillation'
    assert cfg['loss']['distill'] == 0.10
    assert cfg['training']['max_steps'] == 7
    assert cfg['training']['batch_size'] == 3
    train_specs = cfg['data']['train']
    assert train_specs[0]['type'] == 'uefb'
    paired = [s for s in train_specs if s['type'] == 'paired_rgb']
    assert len(paired) == 2
    assert all('teacher_dir' in s for s in paired)
    assert paired[0]['teacher_dir'].endswith('retinexformer_train_teacher/real')
    assert paired[1]['teacher_dir'].endswith('retinexformer_train_teacher/synthetic')
