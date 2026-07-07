from scripts.run_p3d_e010_multiseed import make_cfg, planned_jobs, job_tag, should_enter_p4


def test_p3d_plans_e010_three_seeds_only():
    jobs = planned_jobs()
    assert len(jobs) == 3
    assert {(j['seed'], j['e_shape']) for j in jobs} == {(3407, 0.10), (2027, 0.10), (42, 0.10)}


def test_p3d_config_sets_e010_and_seed():
    cfg = make_cfg({'seed': 42, 'e_shape': 0.10, 'e_shape_kernel': 7}, max_steps=1000, train_crop=128, batch_size=8)
    assert cfg['seed'] == 42
    assert cfg['loss']['e_shape'] == 0.10
    assert cfg['loss']['e_shape_kernel'] == 7
    assert cfg['model']['gate_branch'] is True
    assert cfg['model']['q_branch'] is False
    assert [d['type'] for d in cfg['data']['train']] == ['uefb', 'paired_rgb', 'paired_rgb']


def test_p3d_job_tag_reuses_p3c_naming_family():
    assert job_tag({'seed': 3407, 'e_shape': 0.10}) == 'p3d_m4j_es_seed3407_e0100'


def test_p3d_p4_gate_thresholds_are_explicit():
    aggregate = {
        'real_psnr': {'mean': 20.1},
        'synthetic_psnr': {'mean': 18.0},
        'uefb_E_corr': {'mean': 0.4},
        'synthetic_E_corr': {'mean': 0.7},
    }
    assert should_enter_p4(aggregate, synthetic_baseline_mean=17.678)
    aggregate['synthetic_psnr']['mean'] = 17.0
    assert not should_enter_p4(aggregate, synthetic_baseline_mean=17.678)
