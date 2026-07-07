from scripts.run_phase2_dgb_candidate import make_cfg, planned_runs, should_promote_to_multiseed


def test_phase2_plans_single_dgb_candidate_without_rejected_routes():
    runs = planned_runs()
    assert [r['id'] for r in runs] == ['DGB_RLEF_MINIMAL_S3407']
    run = runs[0]
    assert run['seed'] == 3407
    assert run['model']['backbone'] == 'multiscale'
    assert run['model']['backbone_blocks'] == 3
    assert run['model']['gauge_mode'] == 'image_stats'
    assert run['model']['route_type'] == 'recoverability_safe_router'
    assert run['model']['safe_alpha'] == 0.70
    assert run['model']['q_branch'] is True
    assert run['loss']['e_shape'] == 0.05
    assert run['loss']['q'] == 0.02
    assert run['loss']['gauge_schedule']['max_weight'] <= 0.005
    assert run['loss']['gauge_schedule']['hard_cap'] <= 0.010
    assert 'distill' not in run['loss']
    assert 'distill_by_dataset' not in run['loss']
    assert 'rec_by_dataset' not in run['loss']
    assert 'domain_head_anchor_by_dataset' not in run['loss']


def test_phase2_config_uses_joint_data_protocol_and_training_enabled():
    cfg = make_cfg(planned_runs()[0], max_steps=17, train_crop=96, batch_size=4)
    assert cfg['protocol']['stage'] == 'Phase 2 DGB-RLEF candidate training'
    assert cfg['protocol']['variant_id'] == 'DGB_RLEF_MINIMAL_S3407'
    assert cfg['training']['max_steps'] == 17
    assert cfg['training']['phase2_candidate_training'] is True
    assert cfg['model']['gauge_mode'] == 'image_stats'
    assert cfg['model']['route_type'] == 'recoverability_safe_router'
    train_specs = cfg['data']['train']
    assert [s['type'] for s in train_specs] == ['uefb', 'paired_rgb', 'paired_rgb']
    assert all('teacher_dir' not in s for s in train_specs)


def test_phase2_promotion_rule_compares_p3c_and_p7b_near_miss():
    p3c = {
        'uefb_psnr': {'mean': 17.9},
        'real_psnr': {'mean': 20.0},
        'synthetic_psnr': {'mean': 17.7},
    }
    p7b = {'uefb_psnr': 18.08, 'real_psnr': 19.84, 'synthetic_psnr': 17.96}
    assert should_promote_to_multiseed({'uefb_psnr': 18.2, 'real_psnr': 20.1, 'synthetic_psnr': 18.0}, p3c, p7b)
    assert not should_promote_to_multiseed({'uefb_psnr': 18.2, 'real_psnr': 20.1, 'synthetic_psnr': 17.8}, p3c, p7b)
    assert not should_promote_to_multiseed({'uefb_psnr': 18.2, 'real_psnr': 19.9, 'synthetic_psnr': 18.0}, p3c, p7b)
