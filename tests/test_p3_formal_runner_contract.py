from scripts.run_p3_formal import make_cfg, planned_variants, paired_eval_specs, training_complete


def test_p3_formal_has_m0_to_m5_variants():
    variants = planned_variants()
    assert [v['id'] for v in variants] == ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
    assert variants[0]['name'] == 'restorer_only'
    assert variants[-1]['name'] == 'recoverability_q'


def test_p3_formal_config_uses_uefb_formal_and_full_eval():
    cfg = make_cfg(planned_variants()[3], max_steps=1000, train_crop=128, batch_size=8)
    assert cfg['training']['max_steps'] == 1000
    assert cfg['data']['train']['type'] == 'uefb'
    assert cfg['data']['train']['root'].endswith('/data/UEFB-v2/train')
    assert cfg['data']['val']['root'].endswith('/data/UEFB-v2/test')
    assert cfg['data']['val']['crop_size'] is None
    assert cfg['model']['adaptive_gauge'] is True
    assert cfg['loss']['gauge'] > 0


def test_p3_formal_paired_eval_specs_are_full_test():
    specs = paired_eval_specs()
    assert {s['dataset'] for s in specs} == {'real', 'synthetic'}
    for spec in specs:
        assert spec['low_dir'].endswith('/Test/Low')
        assert spec['high_dir'].endswith('/Test/Normal')
        assert spec['max_images'] is None


def test_training_complete_detects_checkpoint_and_metrics(tmp_path):
    run_dir = tmp_path / 'run'
    assert not training_complete(run_dir)
    (run_dir / 'checkpoints').mkdir(parents=True)
    (run_dir / 'metrics').mkdir()
    (run_dir / 'checkpoints' / 'last.pth').write_bytes(b'fake')
    assert not training_complete(run_dir)
    (run_dir / 'metrics' / 'eval_metrics.json').write_text('{}')
    assert training_complete(run_dir)
