from scripts.run_p6c_gateprotect import make_cfg, planned_runs, should_promote_to_multiseed, gpu_for_run


def test_p6c_plans_gate_protection_without_teacher_or_synthetic_rec_upweight():
    runs = planned_runs()
    assert [r['id'] for r in runs] == [
        'P6C_MS_GATEID005',
        'P6C_MS_GATEID010',
        'P6C_MS_GATELOW005',
        'P6C_MS_B2_GATEID005',
    ]
    for run in runs:
        assert run['model']['backbone'] == 'multiscale'
        assert run['loss'].get('distill', 0) == 0
        assert 'distill_by_dataset' not in run['loss']
        assert 'rec_by_dataset' not in run['loss']
        assert run['loss']['rec'] == 1.0
        assert run['loss']['gate'] <= 0.02
    assert planned_runs()[0]['loss']['gate_identity'] == 0.005
    assert planned_runs()[1]['loss']['gate_identity'] == 0.010
    assert planned_runs()[2]['loss']['gate'] == 0.005
    assert planned_runs()[3]['model']['backbone_blocks'] == 2


def test_p6c_config_uses_p6_data_and_gateprotect_protocol():
    cfg = make_cfg(planned_runs()[0], max_steps=17, train_crop=96, batch_size=5)
    assert cfg['protocol']['stage'] == 'P6c gate-protection structural route'
    assert cfg['model']['backbone'] == 'multiscale'
    assert cfg['training']['max_steps'] == 17
    assert cfg['training']['batch_size'] == 5
    assert cfg['loss']['gate_identity'] == 0.005
    assert 'rec_by_dataset' not in cfg['loss']
    train_specs = cfg['data']['train']
    assert [s['name'] if s['type'] == 'paired_rgb' else 'UEFB' for s in train_specs] == ['UEFB', 'LOL-v2-real-train', 'LOL-v2-synthetic-train']
    assert all('teacher_dir' not in s for s in train_specs)


def test_p6c_promotion_rule_requires_balanced_psnr_gain():
    p3c = {
        'uefb_psnr': {'mean': 17.9},
        'real_psnr': {'mean': 20.0},
        'synthetic_psnr': {'mean': 17.7},
    }
    assert should_promote_to_multiseed({'uefb_psnr': 18.1, 'real_psnr': 20.1, 'synthetic_psnr': 17.8}, p3c)
    assert not should_promote_to_multiseed({'uefb_psnr': 18.1, 'real_psnr': 20.1, 'synthetic_psnr': 17.6}, p3c)


def test_p6c_gpu_assignment_can_skip_busy_gpu_zero():
    assert gpu_for_run(index=0, parallel=1, gpu_start=1) == 1
    assert [gpu_for_run(i, parallel=2, gpu_start=1) for i in range(4)] == [1, 2, 1, 2]
