from scripts.run_p3c_multiseed_sweep import make_cfg, planned_jobs, job_tag, multiseed_jobs, sweep_jobs


def test_p3c_plans_three_seed_and_eshape_sweep_without_duplicate_center_job():
    jobs = planned_jobs()
    keys = {(j['seed'], j['e_shape']) for j in jobs}
    assert len(jobs) == 5
    assert keys == {(3407, 0.05), (2027, 0.05), (42, 0.05), (3407, 0.02), (3407, 0.10)}
    assert {(j['seed'], j['e_shape']) for j in multiseed_jobs(jobs)} == {(3407, 0.05), (2027, 0.05), (42, 0.05)}
    assert {(j['seed'], j['e_shape']) for j in sweep_jobs(jobs)} == {(3407, 0.02), (3407, 0.05), (3407, 0.10)}


def test_p3c_config_sets_seed_and_eshape_weight():
    job = {'seed': 2027, 'e_shape': 0.10, 'e_shape_kernel': 7}
    cfg = make_cfg(job, max_steps=1000, train_crop=128, batch_size=8)
    assert cfg['seed'] == 2027
    assert cfg['loss']['e_shape'] == 0.10
    assert cfg['loss']['e_shape_kernel'] == 7
    assert cfg['model']['gate_branch'] is True
    assert cfg['model']['q_branch'] is False
    assert isinstance(cfg['data']['train'], list)
    assert [d['type'] for d in cfg['data']['train']] == ['uefb', 'paired_rgb', 'paired_rgb']


def test_p3c_job_tag_is_filesystem_safe_and_precise():
    assert job_tag({'seed': 3407, 'e_shape': 0.05}) == 'p3c_m4j_es_seed3407_e0050'
    assert job_tag({'seed': 42, 'e_shape': 0.1}) == 'p3c_m4j_es_seed42_e0100'
