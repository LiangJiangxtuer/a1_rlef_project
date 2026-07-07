from scripts.run_p2c_uefb_recovery import make_cfg, planned_runs, should_promote_to_multiseed


def test_p2c_plans_global_gate_floor_and_e_shape_variants_from_gauge_only_base():
    runs = planned_runs()
    assert [r['id'] for r in runs] == [
        'P2C_DGB_GATE_FLOOR015',
        'P2C_DGB_GATE_FLOOR025',
        'P2C_DGB_ESHAPE025',
        'P2C_DGB_ESHAPE010',
    ]
    by_id = {r['id']: r for r in runs}
    assert by_id['P2C_DGB_GATE_FLOOR015']['model']['route_floor'] == 0.15
    assert by_id['P2C_DGB_GATE_FLOOR025']['model']['route_floor'] == 0.25
    assert by_id['P2C_DGB_ESHAPE025']['model']['route_floor'] == 0.0
    assert by_id['P2C_DGB_ESHAPE010']['model']['route_floor'] == 0.0
    assert by_id['P2C_DGB_ESHAPE025']['loss']['e_shape'] == 0.025
    assert by_id['P2C_DGB_ESHAPE010']['loss']['e_shape'] == 0.010
    for run in runs:
        assert run['seed'] == 3407
        assert run['base_variant'] == 'P2B_DGB_GAUGE_ONLY'
        assert run['model']['gauge_mode'] == 'image_stats'
        assert run['model']['route_type'] == 'identity_gate'
        assert run['model']['q_branch'] is True
        assert run['loss']['q'] == 0.02
        assert run['loss']['gauge_schedule']['max_weight'] <= 0.005
        assert run['loss']['gauge_schedule']['hard_cap'] <= 0.010
        assert 'distill' not in run['loss']
        assert 'rec_by_dataset' not in run['loss']
        assert 'domain_head_anchor_by_dataset' not in run['loss']


def test_p2c_config_records_global_gate_floor_not_dataset_specific_reweighting():
    cfg = make_cfg(planned_runs()[0], max_steps=23, train_crop=96, batch_size=4)
    assert cfg['protocol']['stage'] == 'P2C UEFB recovery from Gauge-only'
    assert cfg['protocol']['base_variant'] == 'P2B_DGB_GAUGE_ONLY'
    assert cfg['protocol']['intervention'] == 'global_route_floor'
    assert cfg['training']['phase2c_uefb_recovery'] is True
    assert cfg['model']['route_floor'] == 0.15
    assert cfg['loss'].get('rec_by_dataset') is None
    assert cfg['loss'].get('distill') is None
    assert [s['type'] for s in cfg['data']['train']] == ['uefb', 'paired_rgb', 'paired_rgb']
    assert all('teacher_dir' not in s for s in cfg['data']['train'])


def test_p2c_promotion_rule_requires_all_three_psnr_gates():
    p3c = {
        'uefb_psnr': {'mean': 17.9},
        'real_psnr': {'mean': 20.0},
        'synthetic_psnr': {'mean': 17.7},
    }
    p7b = {'uefb_psnr': 18.08, 'real_psnr': 19.84, 'synthetic_psnr': 17.96}
    assert should_promote_to_multiseed({'uefb_psnr': 18.2, 'real_psnr': 20.1, 'synthetic_psnr': 18.1}, p3c, p7b)
    assert not should_promote_to_multiseed({'uefb_psnr': 17.95, 'real_psnr': 20.1, 'synthetic_psnr': 18.1}, p3c, p7b)
