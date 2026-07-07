from pathlib import Path

import numpy as np
from PIL import Image

from scripts.run_p4_official_baselines import canonical_key, evaluate_image_outputs, p4_baseline_plan


def _save_rgb(path: Path, value: int, size=(12, 10)):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_p4_canonical_key_strips_known_baseline_suffixes():
    assert canonical_key(Path('00690_kindle_v2.png')) == '00690'
    assert canonical_key(Path('r0001.png')) == 'r0001'
    assert canonical_key(Path('low00771.png')) == '00771'
    assert canonical_key(Path('normal00771.png')) == '00771'


def test_p4_evaluate_image_outputs_pairs_and_crops_to_prediction_size(tmp_path):
    pred = tmp_path / 'pred'
    low = tmp_path / 'low'
    high = tmp_path / 'high'
    _save_rgb(pred / 'a_kindle_v2.png', 100, size=(12, 12))
    _save_rgb(low / 'a.png', 50, size=(16, 16))
    _save_rgb(high / 'a.png', 100, size=(16, 16))
    metrics, rows = evaluate_image_outputs(pred, low, high)
    assert metrics['n'] == 1
    assert len(rows) == 1
    assert metrics['psnr'] > 70
    assert rows[0]['key'] == 'a'


def test_p4_baseline_plan_marks_kindpp_dependency_separately():
    plan = p4_baseline_plan()
    names = {p['method'] for p in plan}
    assert {'Retinexformer', 'Zero-DCE++', 'KinD++'} <= names
    kind = next(p for p in plan if p['method'] == 'KinD++')
    assert 'tensorflow' in kind['required_runtime']
