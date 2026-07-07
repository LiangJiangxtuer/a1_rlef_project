from scripts.run_p2b_controlled_isolation import make_cfg, planned_runs, should_promote_to_multiseed


def test_p2b_plans_two_controlled_isolation_variants_only():
    runs = planned_runs()
    assert [r['id'] for r in runs] == ['P2B_DGB_GAUGE_ONLY', 'P2B_DGB_ROUTE_ONLY']

    gauge_only, route_only = runs
    assert gauge_only['model']['gauge_mode'] == 'image_stats'
    assert gauge_only['model']['route_type'] == 'identity_gate'
    assert gauge_only['model']['safe_alpha'] == 0.70
    assert gauge_only['loss']['gauge_schedule']['max_weight'] <= 0.005

    assert route_only['model']['gauge_mode'] == 'adaptive_head'
    assert route_only['model']['route_type'] == 'recoverability_safe_router'
    assert route_only['model']['safe_alpha'] == 0.70
    assert 'gauge_schedule' not in route_only['loss']
    assert route_only['loss']['gauge'] == 0.10

    for run in runs:
        assert run['seed'] == 3407
        assert run['model']['backbone'] == 'multiscale'
        assert run['model']['backbone_blocks'] == 3
        assert run['model']['q_branch'] is True
        assert run['loss']['e_shape'] == 0.05
        assert run['loss']['q'] == 0.02
        assert 'distill' not in run['loss']
        assert 'rec_by_dataset' not in run['loss']
        assert 'domain_head_anchor_by_dataset' not in run['loss']


def test_p2b_config_records_isolation_factor_and_joint_data_protocol():
    cfg = make_cfg(planned_runs()[0], max_steps=19, train_crop=80, batch_size=3)
    assert cfg['protocol']['stage'] == 'P2B DGB controlled isolation'
    assert cfg['protocol']['isolation_factor'] == 'gauge_only'
    assert cfg['training']['max_steps'] == 19
    assert cfg['training']['phase2_controlled_isolation'] is True
    assert [s['type'] for s in cfg['data']['train']] == ['uefb', 'paired_rgb', 'paired_rgb']
    assert all('teacher_dir' not in s for s in cfg['data']['train'])


def test_p2b_promotion_rule_still_requires_p3c_and_p7b_psnr_gain():
    p3c = {
        'uefb_psnr': {'mean': 17.9},
        'real_psnr': {'mean': 20.0},
        'synthetic_psnr': {'mean': 17.7},
    }
    p7b = {'uefb_psnr': 18.08, 'real_psnr': 19.84, 'synthetic_psnr': 17.96}
    assert should_promote_to_multiseed({'uefb_psnr': 18.2, 'real_psnr': 20.1, 'synthetic_psnr': 18.1}, p3c, p7b)
    assert not should_promote_to_multiseed({'uefb_psnr': 18.0, 'real_psnr': 20.1, 'synthetic_psnr': 18.1}, p3c, p7b)
    assert not should_promote_to_multiseed({'uefb_psnr': 18.2, 'real_psnr': 19.9, 'synthetic_psnr': 18.1}, p3c, p7b)
