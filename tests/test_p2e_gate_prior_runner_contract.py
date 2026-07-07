from scripts.run_p2e_gate_prior_balance import make_cfg, planned_runs, should_promote_to_multiseed


def test_p2e_plans_image_stat_gate_prior_variants():
    runs = planned_runs()
    assert [r['id'] for r in runs] == ['P2E_ESHAPE010_GATEPRI002', 'P2E_ESHAPE010_GATEPRI005']
    weights = [r['loss']['gate_prior'] for r in runs]
    assert weights == [0.02, 0.05]
    for run in runs:
        assert run['seed'] == 3407
        assert run['base_variant'] == 'P2C_DGB_ESHAPE010'
        assert run['intervention'] == 'image_stat_gate_prior'
        assert run['model']['gauge_mode'] == 'image_stats'
        assert run['model']['route_type'] == 'identity_gate'
        assert run['loss']['e_shape'] == 0.010
        assert run['loss']['gate_prior_base'] == 0.35
        assert run['loss']['gate_prior_dark_gain'] == 0.30
        assert run['loss']['gate_prior_max'] <= 0.75
        assert 'distill' not in run['loss']
        assert 'rec_by_dataset' not in run['loss']
        assert 'domain_head_anchor_by_dataset' not in run['loss']


def test_p2e_config_records_gate_prior_without_dataset_label_reweighting():
    cfg = make_cfg(planned_runs()[0], max_steps=29, train_crop=80, batch_size=2)
    assert cfg['protocol']['stage'] == 'P2E image-stat gate prior balance'
    assert cfg['protocol']['base_variant'] == 'P2C_DGB_ESHAPE010'
    assert cfg['protocol']['intervention'] == 'image_stat_gate_prior'
    assert cfg['training']['phase2e_gate_prior_balance'] is True
    assert cfg['loss']['gate_prior'] == 0.02
    assert cfg['loss']['e_shape'] == 0.010
    assert cfg['loss'].get('rec_by_dataset') is None
    assert cfg['loss'].get('distill') is None
    assert [s['type'] for s in cfg['data']['train']] == ['uefb', 'paired_rgb', 'paired_rgb']
    assert all('teacher_dir' not in s for s in cfg['data']['train'])


def test_p2e_promotion_rule_requires_all_three_psnr_gates():
    p3c = {
        'uefb_psnr': {'mean': 17.9},
        'real_psnr': {'mean': 20.0},
        'synthetic_psnr': {'mean': 17.7},
    }
    p7b = {'uefb_psnr': 18.08, 'real_psnr': 19.84, 'synthetic_psnr': 17.96}
    assert should_promote_to_multiseed({'uefb_psnr': 18.2, 'real_psnr': 20.1, 'synthetic_psnr': 18.1}, p3c, p7b)
    assert not should_promote_to_multiseed({'uefb_psnr': 18.2, 'real_psnr': 19.8, 'synthetic_psnr': 18.1}, p3c, p7b)
