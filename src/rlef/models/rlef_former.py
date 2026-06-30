from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from rlef.utils.tensor_ops import luminance, gradient_magnitude


class ResidualBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        groups = 4 if c >= 4 else 1
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(groups, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(groups, c))
    def forward(self, x):
        return F.silu(x + self.net(x))


class TinyRestormerLite(nn.Module):
    def __init__(self, in_channels: int = 5, base_channels: int = 32, blocks: int = 2):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(nn.Conv2d(in_channels, c, 3, padding=1), nn.SiLU(), ResidualBlock(c))
        self.down = nn.Sequential(nn.Conv2d(c, c*2, 3, stride=2, padding=1), nn.SiLU(), ResidualBlock(c*2))
        self.mid = nn.Sequential(*[ResidualBlock(c*2) for _ in range(max(1, blocks))])
        self.up = nn.Sequential(nn.Conv2d(c*2 + c, c, 3, padding=1), nn.SiLU(), ResidualBlock(c))
        self.out_channels = c
    def forward(self, x):
        s = self.stem(x)
        d = self.down(s)
        m = self.mid(d)
        u = F.interpolate(m, size=s.shape[-2:], mode='bilinear', align_corners=False)
        return self.up(torch.cat([u, s], dim=1))


class RLEFFormer(nn.Module):
    def __init__(self, base_channels: int = 32, e_max: float = 3.5,
                 exposure_branch: bool = True, adaptive_gauge: bool = True,
                 fixed_gauge: float | None = None, physics_branch: bool = True,
                 gate_branch: bool = False, q_branch: bool = False):
        super().__init__()
        self.e_max = float(e_max)
        self.exposure_branch = exposure_branch
        self.adaptive_gauge = adaptive_gauge
        self.fixed_gauge = fixed_gauge
        self.physics_branch = physics_branch
        self.gate_branch = gate_branch
        self.q_branch = q_branch
        self.backbone = TinyRestormerLite(in_channels=5, base_channels=base_channels)
        c = self.backbone.out_channels
        self.rest_head = nn.Conv2d(c, 3, 3, padding=1)
        self.e_head = nn.Conv2d(c, 1, 3, padding=1)
        self.mu_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1), nn.SiLU(), nn.Conv2d(c, 1, 1))
        self.r_head = nn.Conv2d(c, 3, 3, padding=1)
        self.l_head = nn.Conv2d(c, 1, 3, padding=1)
        self.a_head = nn.Conv2d(c, 1, 3, padding=1)
        self.q_head = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, low: torch.Tensor) -> dict:
        low = low.clamp(0, 1)
        lum = luminance(low)
        grad = gradient_magnitude(lum)
        feat = self.backbone(torch.cat([low, lum, grad], dim=1))
        i_rest = torch.sigmoid(self.rest_head(feat)).clamp(0, 1)
        b, _, h, w = low.shape
        if self.exposure_branch:
            e_raw = torch.tanh(self.e_head(feat)) * self.e_max
            if self.adaptive_gauge:
                mu = torch.tanh(self.mu_head(feat)) * self.e_max
            elif self.fixed_gauge is not None:
                mu = torch.full((b,1,1,1), float(self.fixed_gauge), device=low.device, dtype=low.dtype)
            else:
                mu = e_raw.mean(dim=(2,3), keepdim=True)
            e = e_raw - e_raw.mean(dim=(2,3), keepdim=True) + mu
        else:
            e_raw = torch.zeros((b,1,h,w), device=low.device, dtype=low.dtype)
            mu = torch.zeros((b,1,1,1), device=low.device, dtype=low.dtype)
            e = e_raw
        if self.physics_branch:
            r = torch.sigmoid(self.r_head(feat))
            l_low = torch.sigmoid(self.l_head(feat)).clamp_min(1e-4)
            i_phys = (r * torch.clamp(l_low * torch.exp(e), 0, 1)).clamp(0, 1)
        else:
            r = low
            l_low = lum.clamp_min(1e-4)
            i_phys = low
        a = torch.sigmoid(self.a_head(feat)) if self.gate_branch else torch.ones((b,1,h,w), device=low.device, dtype=low.dtype)
        q = torch.sigmoid(self.q_head(feat)) if self.q_branch else torch.ones((b,1,h,w), device=low.device, dtype=low.dtype)
        i_hat = (a * i_rest + (1.0 - a) * low).clamp(0, 1) if self.gate_branch else i_rest
        return {'I_hat': i_hat, 'I_rest': i_rest, 'I_phys': i_phys, 'E_raw': e_raw, 'E': e, 'mu_E': mu, 'A': a, 'Q': q, 'R': r, 'L_low': l_low}
