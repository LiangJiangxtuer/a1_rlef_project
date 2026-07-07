from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor


def _load_map(path: Path) -> torch.Tensor:
    npy = path.with_suffix('.npy')
    if npy.exists():
        arr = np.load(npy).astype('float32')
        if arr.ndim == 2:
            arr = arr[None]
        return torch.from_numpy(arr)
    img = to_tensor(Image.open(path).convert('L'))
    return img


class UEFBPairedDataset(Dataset):
    def __init__(self, root: str | Path, crop_size: int | None = None, training: bool = False, augment: bool = False, max_images: int | None = None):
        self.root = Path(root)
        self.crop_size, self.training, self.augment = crop_size, training, augment
        self.low_paths = sorted((self.root / 'low').glob('*.png'))
        if max_images is not None:
            self.low_paths = self.low_paths[:max_images]
        if not self.low_paths:
            raise FileNotFoundError(f'No UEFB low images under {self.root}')

    def __len__(self): return len(self.low_paths)

    def _crop(self, *xs):
        if self.crop_size is None:
            return xs
        _, h, w = xs[0].shape
        cs = min(self.crop_size, h, w)
        if self.training and (h > cs or w > cs):
            top = int(torch.randint(0, h - cs + 1, (1,)).item()) if h > cs else 0
            left = int(torch.randint(0, w - cs + 1, (1,)).item()) if w > cs else 0
        else:
            top, left = max(0, (h - cs)//2), max(0, (w - cs)//2)
        return tuple(x[:, top:top+cs, left:left+cs] for x in xs)

    def _aug(self, *xs):
        if not (self.training and self.augment): return xs
        if torch.rand(()) < 0.5: xs = tuple(torch.flip(x, [2]) for x in xs)
        if torch.rand(()) < 0.5: xs = tuple(torch.flip(x, [1]) for x in xs)
        k = int(torch.randint(0, 4, (1,)).item())
        if k: xs = tuple(torch.rot90(x, k, [1,2]) for x in xs)
        return xs

    def __getitem__(self, idx):
        lp = self.low_paths[idx]
        name = lp.stem
        low = to_tensor(Image.open(lp).convert('RGB'))
        high = to_tensor(Image.open(self.root / 'high' / f'{name}.png').convert('RGB'))
        e = _load_map(self.root / 'E_gt' / f'{name}.png')
        a = _load_map(self.root / 'A_gt' / f'{name}.png').clamp(0, 1)
        q = _load_map(self.root / 'Q_gt' / f'{name}.png').clamp(0, 1)
        low, high, e, a, q = self._crop(low, high, e, a, q)
        low, high, e, a, q = self._aug(low, high, e, a, q)
        return {'low': low.contiguous(), 'high': high.contiguous(), 'teacher': high.contiguous(), 'E_gt': e.contiguous(), 'A_gt': a.contiguous(), 'Q_gt': q.contiguous(), 'name': name, 'dataset': 'UEFB'}
