from scripts.run_p7_domainhead import make_cfg, planned_runs, should_promote_to_multiseed, gpu_for_run


def test_p7_plans_domain_conditioned_multiscale_variants():
    runs = planned_runs()
    assert [r['id'] for r in runs] == [
        'P7_MS_DGATE',
        'P7_MS_DHEAD',
        'P7_MS_DAFFINE_GATE',
    ]
    for run in runs:
        assert run['model']['backbone'] == 'multiscale'
        assert run['model']['backbone_blocks'] == 3
        assert run['model']['domain_conditioning'] is True
        assert run['model']['domain_names'] == ['uefb', 'real', 'synthetic']
        assert run['loss']['rec'] == 1.0
        assert 'rec_by_dataset' not in run['loss']
        assert 'distill' not in run['loss']
        assert 'distill_by_dataset' not in run['loss']
    assert planned_runs()[0]['model']['domain_adapter'] == 'gate_bias'
    assert planned_runs()[1]['model']['domain_adapter'] == 'head_bias'
    assert planned_runs()[2]['model']['domain_adapter'] == 'feature_affine+gate_bias'


def test_p7_config_uses_domain_conditioning_and_same_joint_data_protocol():
    cfg = make_cfg(planned_runs()[0], max_steps=17, train_crop=96, batch_size=5)
    assert cfg['protocol']['stage'] == 'P7 domain-conditioned head structural route'
    assert cfg['model']['domain_conditioning'] is True
    assert cfg['model']['domain_adapter'] == 'gate_bias'
    assert cfg['training']['max_steps'] == 17
    assert cfg['training']['batch_size'] == 5
    train_specs = cfg['data']['train']
    assert [s['name'] if s['type'] == 'paired_rgb' else 'UEFB' for s in train_specs] == ['UEFB', 'LOL-v2-real-train', 'LOL-v2-synthetic-train']
    assert all('teacher_dir' not in s for s in train_specs)


def test_p7_promotion_rule_requires_balanced_psnr_gain():
    p3c = {
        'uefb_psnr': {'mean': 17.9},
        'real_psnr': {'mean': 20.0},
        'synthetic_psnr': {'mean': 17.7},
    }
    assert should_promote_to_multiseed({'uefb_psnr': 18.1, 'real_psnr': 20.1, 'synthetic_psnr': 17.8}, p3c)
    assert not should_promote_to_multiseed({'uefb_psnr': 18.1, 'real_psnr': 20.1, 'synthetic_psnr': 17.6}, p3c)


def test_p7_gpu_assignment_can_skip_busy_gpu_zero():
    assert gpu_for_run(index=0, parallel=1, gpu_start=1) == 1
    assert [gpu_for_run(i, parallel=2, gpu_start=1) for i in range(4)] == [1, 2, 1, 2]
