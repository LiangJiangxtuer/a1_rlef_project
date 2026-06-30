from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw
import torch
from torchvision.transforms.functional import to_tensor
from rlef.utils.tensor_ops import normalize01


def read_rgb(path: str | Path) -> torch.Tensor:
    return to_tensor(Image.open(path).convert('RGB'))


def save_tensor_image(x: torch.Tensor, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype('uint8')
    Image.fromarray(arr).save(path)


def make_contact_sheet(items: dict[str, torch.Tensor], path: str | Path, cell: int = 160) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    labels = []
    for label, tensor in items.items():
        t = tensor.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.shape[0] == 1:
            t = normalize01(t.unsqueeze(0))[0].repeat(3, 1, 1)
        else:
            t = t.clamp(0, 1)
        arr = (t.permute(1, 2, 0).numpy() * 255 + 0.5).astype('uint8')
        img = Image.fromarray(arr).resize((cell, cell))
        tiles.append(img); labels.append(label)
    sheet = Image.new('RGB', (cell * len(tiles), cell + 20), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, img in enumerate(tiles):
        sheet.paste(img, (i * cell, 20))
        draw.text((i * cell + 4, 3), labels[i], fill=(0, 0, 0))
    sheet.save(path)
