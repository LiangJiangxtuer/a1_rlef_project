from scripts.run_p6b_synprotect import make_cfg, planned_runs, should_promote_to_multiseed, gpu_for_run


def test_p6b_plans_synthetic_protection_loss_weight_sweep():
    runs = planned_runs()
    assert [r['id'] for r in runs] == ['P6B_MS_SYN105', 'P6B_MS_SYN110', 'P6B_MS_SYN125', 'P6B_MS_SYN150']
    for run in runs:
        assert run['model']['backbone'] == 'multiscale'
        assert run['model']['backbone_blocks'] == 3
        assert 'distill' not in run['loss']
        rec_map = run['loss']['rec_by_dataset']
        assert rec_map['UEFB'] == 1.0
        assert rec_map['LOL-v2-real-train'] == 1.0
        assert rec_map['LOL-v2-synthetic-train'] > rec_map['LOL-v2-real-train']


def test_p6b_config_uses_p6_data_and_dataset_weighted_rec():
    cfg = make_cfg(planned_runs()[0], max_steps=17, train_crop=96, batch_size=5)
    assert cfg['protocol']['stage'] == 'P6b synthetic-protection structural route'
    assert cfg['model']['backbone'] == 'multiscale'
    assert cfg['training']['max_steps'] == 17
    assert cfg['training']['batch_size'] == 5
    assert cfg['loss']['rec'] == 0.0
    assert cfg['loss']['rec_by_dataset']['LOL-v2-synthetic-train'] == 1.05
    train_specs = cfg['data']['train']
    assert [s['name'] if s['type'] == 'paired_rgb' else 'UEFB' for s in train_specs] == ['UEFB', 'LOL-v2-real-train', 'LOL-v2-synthetic-train']
    assert all('teacher_dir' not in s for s in train_specs)


def test_p6b_promotion_rule_requires_balanced_psnr_gain():
    p3c = {
        'uefb_psnr': {'mean': 17.9},
        'real_psnr': {'mean': 20.0},
        'synthetic_psnr': {'mean': 17.7},
    }
    assert should_promote_to_multiseed({'uefb_psnr': 18.1, 'real_psnr': 20.1, 'synthetic_psnr': 17.8}, p3c)
    assert not should_promote_to_multiseed({'uefb_psnr': 18.1, 'real_psnr': 20.1, 'synthetic_psnr': 17.6}, p3c)


def test_p6b_gpu_assignment_can_skip_busy_gpu_zero():
    assert gpu_for_run(index=0, parallel=1, gpu_start=1) == 1
    assert [gpu_for_run(i, parallel=2, gpu_start=1) for i in range(4)] == [1, 2, 1, 2]
