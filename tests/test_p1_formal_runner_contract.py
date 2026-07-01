from scripts.run_p1_formal import make_cfg, planned_runs


def test_p1_formal_has_real_synthetic_and_three_gauge_modes():
    runs = planned_runs()
    assert len(runs) == 6
    assert {(r['dataset'], r['mode']) for r in runs} == {
        ('real', 'nogauge'), ('real', 'fixed0p02'), ('real', 'adaptive'),
        ('synthetic', 'nogauge'), ('synthetic', 'fixed0p02'), ('synthetic', 'adaptive'),
    }


def test_p1_formal_config_uses_full_val_and_1000_steps():
    cfg = make_cfg('real', 'adaptive', max_steps=1000)
    assert cfg['training']['max_steps'] == 1000
    assert cfg['data']['train']['max_images'] is None
    assert cfg['data']['val']['max_images'] is None
    assert cfg['data']['val']['crop_size'] is None
    assert cfg['model']['adaptive_gauge'] is True
    assert cfg['model']['fixed_gauge'] is None
    assert cfg['loss']['gauge'] > 0
