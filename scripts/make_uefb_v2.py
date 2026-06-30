#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from rlef.datasets.paired_rgb_dataset import list_images


def _save_rgb(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (t.clamp(0,1).permute(1,2,0).numpy() * 255 + 0.5).astype('uint8')
    Image.fromarray(arr).save(path)


def _save_map(t: torch.Tensor, path: Path, png_min: float = 0.0, png_max: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = t.squeeze().numpy().astype('float32')
    np.save(path.with_suffix('.npy'), arr)
    norm = np.clip((arr - png_min) / (png_max - png_min + 1e-8), 0, 1)
    Image.fromarray((norm * 255 + 0.5).astype('uint8')).save(path)


def _resize_square(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert('RGB')
    w, h = img.size
    m = min(w, h)
    left, top = (w-m)//2, (h-m)//2
    img = img.crop((left, top, left+m, top+m)).resize((size, size), Image.BICUBIC)
    return to_tensor(img)


def _smooth_noise(rng: np.random.Generator, size: int, scale: float) -> torch.Tensor:
    small = torch.from_numpy(rng.normal(0, 1, size=(1,1,8,8)).astype('float32'))
    field = F.interpolate(small, size=(size, size), mode='bicubic', align_corners=False)[0,0]
    field = field / (field.std() + 1e-6) * scale
    return field


def _synthesize(high: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    _, h, w = high.shape
    yy, xx = torch.meshgrid(torch.linspace(-1,1,h), torch.linspace(-1,1,w), indexing='ij')
    e = torch.ones((h,w)) * float(rng.uniform(0.25, 1.8))
    e = e + float(rng.uniform(-0.5,0.5)) * xx + float(rng.uniform(-0.5,0.5)) * yy
    if rng.random() < 0.8:
        e = e + float(rng.uniform(0.0,0.8)) * (xx*xx + yy*yy)
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = rng.uniform(-0.8,0.8), rng.uniform(-0.8,0.8)
        amp = rng.uniform(-0.5, 0.9)
        sig = rng.uniform(0.15, 0.55)
        e = e + float(amp) * torch.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*sig*sig))
    e = e + _smooth_noise(rng, h, float(rng.uniform(0.0, 0.25)))
    near_identity = bool(rng.random() < 0.22)
    if near_identity:
        e = e * float(rng.uniform(0.0, 0.15))
    e = e.clamp(0.0, 3.0)
    low = high * torch.exp(-e).unsqueeze(0)
    noise = torch.from_numpy(rng.normal(0, rng.uniform(0.0, 0.018), size=tuple(low.shape)).astype('float32'))
    low = (low + noise).clamp(0,1)
    # Random unrecoverable clipping regions.
    clip_mask = torch.zeros((h,w))
    if rng.random() < 0.35:
        cx, cy = rng.uniform(-0.9,0.9), rng.uniform(-0.9,0.9)
        sig = rng.uniform(0.12, 0.35)
        blob = torch.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*sig*sig))
        clip_mask = (blob > 0.55).float()
        if rng.random() < 0.5:
            low = low * (1 - clip_mask) + 0.0 * clip_mask
        else:
            low = low * (1 - clip_mask) + 1.0 * clip_mask
    a = (e.abs() / torch.quantile(e.abs().flatten(), 0.95).clamp_min(1e-6)).clamp(0,1).unsqueeze(0)
    q = (1.0 - clip_mask.unsqueeze(0) * 0.85).clamp(0,1)
    return low.float(), e.unsqueeze(0).float(), a.float(), q.float(), {'near_identity': near_identity, 'E_mean': float(e.mean()), 'E_max': float(e.max()), 'clip_ratio': float(clip_mask.mean())}


def generate_uefb(source: str | Path, output: str | Path, num_train: int = 20, num_test: int = 20, image_size: int = 256, seed: int = 3407) -> None:
    source, output = Path(source), Path(output)
    imgs = list_images(source)
    if not imgs:
        raise FileNotFoundError(f'No source images under {source}')
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(imgs), size=num_train + num_test, replace=(len(imgs) < num_train + num_test))
    for split, ids in [('train', indices[:num_train]), ('test', indices[num_train:])]:
        samples = []
        for j, idx in enumerate(ids):
            high = _resize_square(Image.open(imgs[int(idx)]), image_size)
            low, e, a, q, meta = _synthesize(high, rng)
            name = f'{split}_{j:05d}'
            _save_rgb(low, output / split / 'low' / f'{name}.png')
            _save_rgb(high, output / split / 'high' / f'{name}.png')
            _save_map(e, output / split / 'E_gt' / f'{name}.png', png_min=0.0, png_max=3.0)
            _save_map(a, output / split / 'A_gt' / f'{name}.png')
            _save_map(q, output / split / 'Q_gt' / f'{name}.png')
            samples.append({'name': name, 'source': str(imgs[int(idx)]), **meta})
        (output / split).mkdir(parents=True, exist_ok=True)
        (output / split / 'meta.json').write_text(json.dumps({'seed': seed, 'split': split, 'samples': samples}, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--num_train', type=int, default=20)
    ap.add_argument('--num_test', type=int, default=20)
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=3407)
    args = ap.parse_args()
    generate_uefb(args.source, args.output, args.num_train, args.num_test, args.image_size, args.seed)
    print(json.dumps({'output': args.output, 'num_train': args.num_train, 'num_test': args.num_test, 'seed': args.seed}, ensure_ascii=False))

if __name__ == '__main__':
    main()
