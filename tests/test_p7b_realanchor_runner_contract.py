from scripts.run_p7b_realanchor import make_cfg, planned_runs, should_promote_to_multiseed, gpu_for_run


def test_p7b_plans_dhead_real_anchor_without_scalar_rec_or_teacher():
    runs = planned_runs()
    assert [r['id'] for r in runs] == [
        'P7B_DHEAD_RA001',
        'P7B_DHEAD_RA005',
        'P7B_DHEAD_RA010',
    ]
    for run in runs:
        assert run['model']['backbone'] == 'multiscale'
        assert run['model']['backbone_blocks'] == 3
        assert run['model']['domain_conditioning'] is True
        assert run['model']['domain_adapter'] == 'head_bias'
        assert run['model']['domain_names'] == ['uefb', 'real', 'synthetic']
        assert run['loss']['rec'] == 1.0
        assert run['loss']['domain_head_anchor_by_dataset']['real'] == run['real_anchor_weight']
        assert run['loss']['domain_head_anchor_by_dataset']['uefb'] == 0.0
        assert run['loss']['domain_head_anchor_by_dataset']['synthetic'] == 0.0
        assert 'rec_by_dataset' not in run['loss']
        assert 'distill' not in run['loss']
        assert 'distill_by_dataset' not in run['loss']


def test_p7b_config_marks_real_anchor_protocol_and_keeps_joint_data():
    cfg = make_cfg(planned_runs()[1], max_steps=17, train_crop=96, batch_size=5)
    assert cfg['protocol']['stage'] == 'P7b domain-head real-anchor route'
    assert cfg['protocol']['real_anchor_weight'] == 0.005
    assert cfg['model']['domain_adapter'] == 'head_bias'
    assert cfg['loss']['domain_head_anchor_by_dataset']['real'] == 0.005
    assert cfg['training']['max_steps'] == 17
    assert cfg['training']['batch_size'] == 5
    train_specs = cfg['data']['train']
    assert [s['name'] if s['type'] == 'paired_rgb' else 'UEFB' for s in train_specs] == ['UEFB', 'LOL-v2-real-train', 'LOL-v2-synthetic-train']
    assert all('teacher_dir' not in s for s in train_specs)


def test_p7b_promotion_rule_requires_balanced_psnr_gain():
    p3c = {
        'uefb_psnr': {'mean': 17.9},
        'real_psnr': {'mean': 20.0},
        'synthetic_psnr': {'mean': 17.7},
    }
    assert should_promote_to_multiseed({'uefb_psnr': 18.1, 'real_psnr': 20.1, 'synthetic_psnr': 17.8}, p3c)
    assert not should_promote_to_multiseed({'uefb_psnr': 18.1, 'real_psnr': 19.9, 'synthetic_psnr': 17.8}, p3c)


def test_p7b_gpu_assignment_can_skip_busy_gpu_zero():
    assert gpu_for_run(index=0, parallel=1, gpu_start=1) == 1
    assert [gpu_for_run(i, parallel=2, gpu_start=1) for i in range(4)] == [1, 2, 1, 2]
