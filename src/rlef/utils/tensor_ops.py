from __future__ import annotations

import torch
import torch.nn.functional as F


def luminance(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x[:, :3] * w).sum(dim=1, keepdim=True)


def gradient_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    dx, dy = gradient_xy(x)
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def laplacian(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return F.conv2d(x, k, padding=1)


def normalize01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    mn = x.amin(dim=dims, keepdim=True)
    mx = x.amax(dim=dims, keepdim=True)
    return (x - mn) / (mx - mn + eps)
