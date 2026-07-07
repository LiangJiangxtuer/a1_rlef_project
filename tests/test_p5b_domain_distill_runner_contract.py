from scripts.run_p5b_domain_distill import make_cfg, planned_runs, should_promote_to_multiseed


def test_p5b_plans_domain_conditioned_distillation_route():
    runs = planned_runs()
    assert [r['id'] for r in runs] == ['P5B_DW_R03_S005']
    run = runs[0]
    weights = run['loss']['distill_by_dataset']
    assert weights['LOL-v2-real-train-retinex-teacher'] == 0.30
    assert weights['LOL-v2-synthetic-train-retinex-teacher'] == 0.05
    assert weights['UEFB'] == 0.0
    assert 'distill' not in run['loss'] or run['loss']['distill'] == 0.0


def test_p5b_config_preserves_teacher_dirs_and_domain_weight_map():
    cfg = make_cfg(planned_runs()[0], max_steps=11, train_crop=96, batch_size=5)
    assert cfg['protocol']['stage'] == 'P5b domain-conditioned Retinexformer distillation'
    assert cfg['training']['max_steps'] == 11
    assert cfg['training']['batch_size'] == 5
    weights = cfg['loss']['distill_by_dataset']
    paired_names = [s['name'] for s in cfg['data']['train'] if s['type'] == 'paired_rgb']
    assert paired_names == ['LOL-v2-real-train-retinex-teacher', 'LOL-v2-synthetic-train-retinex-teacher']
    assert set(weights) == {'UEFB', *paired_names}


def test_p5b_promotion_rule_requires_real_and_synthetic_gain_over_p3c():
    assert should_promote_to_multiseed({'real_psnr': 20.2, 'synthetic_psnr': 17.9}, {'real_psnr': {'mean': 20.0}, 'synthetic_psnr': {'mean': 17.7}})
    assert not should_promote_to_multiseed({'real_psnr': 20.2, 'synthetic_psnr': 17.6}, {'real_psnr': {'mean': 20.0}, 'synthetic_psnr': {'mean': 17.7}})
