from __future__ import annotations

import torch


def safe_recovery_image(low: torch.Tensor, i_phys: torch.Tensor, safe_alpha: float = 0.70) -> torch.Tensor:
    """Blend low input with physics path for conservative fallback recovery."""
    alpha = float(safe_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f'safe_alpha must be in [0, 1], got {safe_alpha}')
    return (alpha * low.clamp(0, 1) + (1.0 - alpha) * i_phys.clamp(0, 1)).clamp(0, 1)


def recoverability_route(
    i_rest: torch.Tensor,
    i_phys: torch.Tensor,
    low: torch.Tensor,
    a: torch.Tensor,
    safe_alpha: float = 0.70,
) -> dict[str, torch.Tensor]:
    """Route between strong restoration and a safe conservative path.

    ``a`` is the recoverability gate: 1 uses the learned restorer, 0 uses the
    safe path. This separates the route from scalar gate penalties rejected in
    earlier P6c experiments.
    """
    if i_rest.shape != low.shape:
        raise ValueError(f'i_rest and low must have same shape, got {tuple(i_rest.shape)} vs {tuple(low.shape)}')
    if i_phys.shape != low.shape:
        raise ValueError(f'i_phys and low must have same shape, got {tuple(i_phys.shape)} vs {tuple(low.shape)}')
    if a.ndim != 4 or a.shape[0] != low.shape[0] or a.shape[-2:] != low.shape[-2:]:
        raise ValueError(f'a must have shape [B, 1, H, W] compatible with low, got {tuple(a.shape)}')
    gate = a.clamp(0, 1)
    if gate.shape[1] != 1:
        gate = gate.mean(dim=1, keepdim=True)
    i_safe = safe_recovery_image(low, i_phys, safe_alpha=safe_alpha)
    i_hat = (gate * i_rest.clamp(0, 1) + (1.0 - gate) * i_safe).clamp(0, 1)
    return {'I_hat': i_hat, 'I_safe': i_safe, 'A_route': gate}
