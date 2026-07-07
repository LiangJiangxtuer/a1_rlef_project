from __future__ import annotations

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG'}


def list_images(path: str | Path) -> list[Path]:
    p = Path(path)
    return sorted([x for x in p.rglob('*') if x.is_file() and x.suffix in IMG_EXTS])


def _key(p: Path) -> str:
    s = p.stem.lower()
    for prefix in ('low', 'normal', 'high', 'gt', 'input', 'target'):
        if s.startswith(prefix) and len(s) > len(prefix):
            return s[len(prefix):]
    return s


def _luminance_chw(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 1:
        return x
    w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x[:3] * w).sum(dim=0, keepdim=True)


def _oracle_maps(low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e = torch.log(_luminance_chw(high) + 1e-4) - torch.log(_luminance_chw(low) + 1e-4)
    q95 = torch.quantile(e.abs().flatten(), 0.95).clamp_min(1e-6)
    a = (e.abs() / q95).clamp(0, 1)
    q = torch.ones_like(e)
    return e.contiguous(), a.contiguous(), q.contiguous()


class PairedRGBDataset(Dataset):
    def __init__(self, low_dir: str | Path, high_dir: str | Path, crop_size: int | None = None, training: bool = False, augment: bool = False, name: str | None = None, max_images: int | None = None, teacher_dir: str | Path | None = None):
        self.low_dir, self.high_dir = Path(low_dir), Path(high_dir)
        self.teacher_dir = Path(teacher_dir) if teacher_dir is not None else None
        self.crop_size, self.training, self.augment = crop_size, training, augment
        self.dataset_name = name or self.low_dir.parent.name
        highs = {_key(p): p for p in list_images(self.high_dir)}
        teachers = {_key(p): p for p in list_images(self.teacher_dir)} if self.teacher_dir is not None else None
        pairs = []
        for lp in list_images(self.low_dir):
            key = _key(lp)
            if key not in highs:
                continue
            if teachers is not None:
                if key not in teachers:
                    continue
                pairs.append((lp, highs[key], teachers[key]))
            else:
                pairs.append((lp, highs[key], None))
        if max_images is not None:
            pairs = pairs[:max_images]
        if not pairs:
            raise FileNotFoundError(f'No pairs under {self.low_dir} and {self.high_dir}')
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

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
        if not (self.training and self.augment):
            return xs
        if torch.rand(()) < 0.5:
            xs = tuple(torch.flip(x, [2]) for x in xs)
        if torch.rand(()) < 0.5:
            xs = tuple(torch.flip(x, [1]) for x in xs)
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            xs = tuple(torch.rot90(x, k, [1, 2]) for x in xs)
        return xs

    def __getitem__(self, idx):
        low_path, high_path, teacher_path = self.pairs[idx]
        low = to_tensor(Image.open(low_path).convert('RGB'))
        high = to_tensor(Image.open(high_path).convert('RGB'))
        if teacher_path is not None:
            teacher = to_tensor(Image.open(teacher_path).convert('RGB'))
            low, high, teacher = self._crop(low, high, teacher)
            low, high, teacher = self._aug(low, high, teacher)
        else:
            low, high = self._crop(low, high)
            low, high = self._aug(low, high)
            teacher = None
        e, a, q = _oracle_maps(low, high)
        sample = {'low': low.contiguous(), 'high': high.contiguous(), 'teacher': (teacher if teacher is not None else high).contiguous(), 'E_gt': e, 'A_gt': a, 'Q_gt': q, 'name': low_path.stem, 'dataset': self.dataset_name}
        return sample
