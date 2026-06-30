from pathlib import Path
import json
from PIL import Image

from rlef.datasets.uefb_dataset import UEFBPairedDataset
from scripts.make_uefb_v2 import generate_uefb


def test_generate_uefb_writes_low_high_maps_and_meta(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for i in range(3):
        Image.new('RGB', (32, 24), color=(80 + i * 30, 100, 120)).save(source / f'{i:03d}.png')
    out = tmp_path / 'UEFB-smoke'
    generate_uefb(source, out, num_train=2, num_test=1, image_size=32, seed=123)
    for split, expected in [('train', 2), ('test', 1)]:
        for sub in ['low', 'high', 'E_gt', 'A_gt', 'Q_gt']:
            files = sorted((out / split / sub).glob('*.png'))
            assert len(files) == expected, (split, sub, files)
        meta = json.loads((out / split / 'meta.json').read_text())
        assert len(meta['samples']) == expected
    ds = UEFBPairedDataset(out / 'train', crop_size=16, training=True)
    item = ds[0]
    assert item['low'].shape == (3, 16, 16)
    assert item['E_gt'].shape == (1, 16, 16)
    assert item['A_gt'].min() >= 0 and item['A_gt'].max() <= 1
    assert item['Q_gt'].min() >= 0 and item['Q_gt'].max() <= 1
