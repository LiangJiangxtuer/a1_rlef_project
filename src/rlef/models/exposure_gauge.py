from __future__ import annotations

import torch


def _as_image_scalar(mu: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    """Normalize a provided gauge value to [B, 1, 1, 1]."""
    mu_tensor = torch.as_tensor(mu, dtype=reference.dtype, device=reference.device)
    if mu_tensor.ndim == 0:
        return mu_tensor.view(1, 1, 1, 1).expand(reference.shape[0], 1, 1, 1)
    if mu_tensor.ndim == 1:
        return mu_tensor.view(-1, 1, 1, 1)
    if mu_tensor.ndim == 2 and mu_tensor.shape[1] == 1:
        return mu_tensor.view(-1, 1, 1, 1)
    if mu_tensor.ndim == 4 and mu_tensor.shape[-2:] == (1, 1):
        return mu_tensor
    return mu_tensor.mean(dim=tuple(range(1, mu_tensor.ndim)), keepdim=True).view(-1, 1, 1, 1)


def decompose_exposure(e_raw: torch.Tensor, mu: torch.Tensor | float | None = None) -> dict[str, torch.Tensor]:
    """Split an exposure field into zero-mean shape and absolute gauge.

    DGB-RLEF treats the spatial exposure shape as gauge-invariant and the image
    mean as the calibrated gauge. If ``mu`` is omitted, the decomposition is an
    identity-preserving split: ``E == e_raw`` and ``mu == mean(e_raw)``.
    """
    if e_raw.ndim != 4 or e_raw.shape[1] != 1:
        raise ValueError(f'e_raw must have shape [B, 1, H, W], got {tuple(e_raw.shape)}')
    shape = e_raw - e_raw.mean(dim=(2, 3), keepdim=True)
    gauge = e_raw.mean(dim=(2, 3), keepdim=True) if mu is None else _as_image_scalar(mu, e_raw)
    if gauge.shape[0] == 1 and e_raw.shape[0] > 1:
        gauge = gauge.expand(e_raw.shape[0], 1, 1, 1)
    if gauge.shape != (e_raw.shape[0], 1, 1, 1):
        raise ValueError(f'mu must broadcast to [B, 1, 1, 1], got {tuple(gauge.shape)} for B={e_raw.shape[0]}')
    return {'S': shape, 'mu': gauge, 'E': shape + gauge}
