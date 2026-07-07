from scripts.run_p6_structural_backbone import make_cfg, planned_runs, should_promote_to_multiseed


def test_p6_plans_multiscale_structural_route_without_teacher_distillation():
    runs = planned_runs()
    assert [r['id'] for r in runs] == ['P6_MS_M4J_ES']
    run = runs[0]
    assert run['model']['backbone'] == 'multiscale'
    assert run['model']['backbone_blocks'] >= 3
    assert run['model']['gate_branch'] is True
    assert run['loss']['e_shape'] == 0.05
    assert 'distill' not in run['loss']


def test_p6_config_uses_same_joint_data_protocol_as_p3c():
    cfg = make_cfg(planned_runs()[0], max_steps=13, train_crop=96, batch_size=5)
    assert cfg['protocol']['stage'] == 'P6 stronger-backbone RLEF integration'
    assert cfg['training']['max_steps'] == 13
    assert cfg['training']['batch_size'] == 5
    assert cfg['model']['backbone'] == 'multiscale'
    train_specs = cfg['data']['train']
    assert [s['type'] for s in train_specs] == ['uefb', 'paired_rgb', 'paired_rgb']
    assert all('teacher_dir' not in s for s in train_specs)


def test_p6_promotion_rule_requires_real_synthetic_and_uefb_gain():
    p3c = {
        'uefb_psnr': {'mean': 17.9},
        'real_psnr': {'mean': 20.0},
        'synthetic_psnr': {'mean': 17.7},
    }
    assert should_promote_to_multiseed({'uefb_psnr': 18.0, 'real_psnr': 20.2, 'synthetic_psnr': 17.9}, p3c)
    assert not should_promote_to_multiseed({'uefb_psnr': 18.0, 'real_psnr': 19.9, 'synthetic_psnr': 17.9}, p3c)
    assert not should_promote_to_multiseed({'uefb_psnr': 17.8, 'real_psnr': 20.2, 'synthetic_psnr': 17.9}, p3c)
