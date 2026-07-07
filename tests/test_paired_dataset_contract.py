from pathlib import Path
from PIL import Image
import torch

from rlef.datasets.paired_rgb_dataset import PairedRGBDataset


def test_paired_rgb_dataset_returns_oracle_supervision_maps(tmp_path):
    low_dir = tmp_path / 'low'; high_dir = tmp_path / 'high'
    low_dir.mkdir(); high_dir.mkdir()
    Image.new('RGB', (16, 16), color=(32, 32, 32)).save(low_dir / '000.png')
    Image.new('RGB', (16, 16), color=(96, 96, 96)).save(high_dir / '000.png')
    sample = PairedRGBDataset(low_dir, high_dir)[0]
    assert sample['E_gt'].shape == (1, 16, 16)
    assert sample['A_gt'].shape == (1, 16, 16)
    assert sample['Q_gt'].shape == (1, 16, 16)
    assert torch.isfinite(sample['E_gt']).all()
    assert float(sample['A_gt'].min()) >= 0 and float(sample['A_gt'].max()) <= 1
    assert torch.allclose(sample['Q_gt'], torch.ones_like(sample['Q_gt']))


def test_paired_rgb_dataset_can_return_teacher_targets_with_matching_crop(tmp_path):
    low_dir = tmp_path / 'low'; high_dir = tmp_path / 'high'; teacher_dir = tmp_path / 'teacher'
    low_dir.mkdir(); high_dir.mkdir(); teacher_dir.mkdir()
    Image.new('RGB', (20, 20), color=(32, 32, 32)).save(low_dir / 'low0001.png')
    Image.new('RGB', (20, 20), color=(96, 96, 96)).save(high_dir / 'normal0001.png')
    Image.new('RGB', (20, 20), color=(128, 64, 32)).save(teacher_dir / '0001.png')

    sample = PairedRGBDataset(low_dir, high_dir, teacher_dir=teacher_dir, crop_size=12, training=False)[0]

    assert sample['low'].shape == sample['high'].shape == sample['teacher'].shape == (3, 12, 12)
    assert sample['name'] == 'low0001'
    assert torch.allclose(sample['teacher'][0], torch.full((12, 12), 128 / 255), atol=1e-4)
