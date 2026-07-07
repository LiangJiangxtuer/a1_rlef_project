from scripts.run_p3b_joint import make_cfg, planned_runs, paired_eval_specs


def test_p3b_joint_plans_m4_joint_and_eshape_diagnostic():
    runs = planned_runs()
    assert [r['id'] for r in runs] == ['M4J', 'M4J_ES']
    assert runs[0]['name'] == 'm4_joint'
    assert runs[1]['loss']['e_shape'] > 0


def test_p3b_joint_config_mixes_uefb_real_and_synthetic_train_sets():
    cfg = make_cfg(planned_runs()[0], max_steps=1000, train_crop=128, batch_size=8)
    train = cfg['data']['train']
    assert isinstance(train, list)
    assert [d['type'] for d in train] == ['uefb', 'paired_rgb', 'paired_rgb']
    assert train[0]['root'].endswith('/data/UEFB-v2/train')
    assert train[1]['name'] == 'LOL-v2-real-train'
    assert train[2]['name'] == 'LOL-v2-synthetic-train'
    assert cfg['data']['val']['root'].endswith('/data/UEFB-v2/test')
    assert cfg['model']['gate_branch'] is True
    assert cfg['model']['q_branch'] is False


def test_p3b_paired_eval_specs_are_full_test():
    specs = paired_eval_specs()
    assert {s['dataset'] for s in specs} == {'real', 'synthetic'}
    assert all(s['max_images'] is None for s in specs)
