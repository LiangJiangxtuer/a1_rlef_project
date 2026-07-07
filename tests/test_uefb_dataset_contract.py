import numpy as np
from PIL import Image
import torch

from rlef.datasets.uefb_dataset import UEFBPairedDataset


def test_uefb_dataset_returns_high_as_default_teacher_for_distillation_batches(tmp_path):
    root = tmp_path / 'uefb'
    for sub in ['low', 'high', 'E_gt', 'A_gt', 'Q_gt']:
        (root / sub).mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (12, 12), color=(30, 30, 30)).save(root / 'low' / '000.png')
    Image.new('RGB', (12, 12), color=(90, 90, 90)).save(root / 'high' / '000.png')
    for sub, value in [('E_gt', 0.5), ('A_gt', 0.7), ('Q_gt', 1.0)]:
        np.save(root / sub / '000.npy', np.full((1, 12, 12), value, dtype='float32'))
        Image.new('L', (12, 12), color=int(value * 255)).save(root / sub / '000.png')

    sample = UEFBPairedDataset(root)[0]

    assert 'teacher' in sample
    assert torch.allclose(sample['teacher'], sample['high'])
