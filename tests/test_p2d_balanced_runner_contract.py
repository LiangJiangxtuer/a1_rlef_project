from scripts.run_p2d_balanced_eshape_real_preserve import make_cfg, planned_runs, should_promote_to_multiseed


def test_p2d_plans_balanced_eshape010_structure_preserve_variants():
    runs = planned_runs()
    assert [r['id'] for r in runs] == ['P2D_ESHAPE010_STRUCT002', 'P2D_ESHAPE010_STRUCT005']
    weights = [r['loss']['structure_preserve'] for r in runs]
    assert weights == [0.02, 0.05]
    for run in runs:
        assert run['seed'] == 3407
        assert run['base_variant'] == 'P2C_DGB_ESHAPE010'
        assert run['model']['gauge_mode'] == 'image_stats'
        assert run['model']['route_type'] == 'identity_gate'
        assert run['model'].get('route_floor', 0.0) == 0.0
        assert run['loss']['e_shape'] == 0.010
        assert run['loss']['q'] == 0.02
        assert run['loss']['structure_preserve_grad'] == 0.5
        assert run['loss']['gauge_schedule']['max_weight'] <= 0.005
        assert 'distill' not in run['loss']
        assert 'rec_by_dataset' not in run['loss']
        assert 'domain_head_anchor_by_dataset' not in run['loss']


def test_p2d_config_records_non_dataset_label_real_preserve_intervention():
    cfg = make_cfg(planned_runs()[0], max_steps=31, train_crop=80, batch_size=2)
    assert cfg['protocol']['stage'] == 'P2D balanced e_shape010 real preservation'
    assert cfg['protocol']['base_variant'] == 'P2C_DGB_ESHAPE010'
    assert cfg['protocol']['intervention'] == 'generic_structure_preserve'
    assert cfg['training']['phase2d_balanced_eshape_real_preserve'] is True
    assert cfg['loss']['e_shape'] == 0.010
    assert cfg['loss']['structure_preserve'] == 0.02
    assert cfg['loss'].get('rec_by_dataset') is None
    assert cfg['loss'].get('distill') is None
    assert [s['type'] for s in cfg['data']['train']] == ['uefb', 'paired_rgb', 'paired_rgb']
    assert all('teacher_dir' not in s for s in cfg['data']['train'])


def test_p2d_promotion_rule_requires_all_three_psnr_gates():
    p3c = {
        'uefb_psnr': {'mean': 17.9},
        'real_psnr': {'mean': 20.0},
        'synthetic_psnr': {'mean': 17.7},
    }
    p7b = {'uefb_psnr': 18.08, 'real_psnr': 19.84, 'synthetic_psnr': 17.96}
    assert should_promote_to_multiseed({'uefb_psnr': 18.2, 'real_psnr': 20.1, 'synthetic_psnr': 18.1}, p3c, p7b)
    assert not should_promote_to_multiseed({'uefb_psnr': 18.2, 'real_psnr': 19.7, 'synthetic_psnr': 18.1}, p3c, p7b)
