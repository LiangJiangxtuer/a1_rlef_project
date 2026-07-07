from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from rlef.models.exposure_gauge import decompose_exposure
from rlef.models.gauge_head import ImageStatsGaugeHead
from rlef.models.recoverability_router import recoverability_route
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


class MultiScaleRestorationBackbone(nn.Module):
    """Two-level encoder/decoder trunk for P6 structural routing.

    It keeps the same feature contract as TinyRestormerLite: input is the
    5-channel low/luminance/gradient tensor and output is a full-resolution
    feature map consumed by the existing restoration and RLEF heads.
    """
    def __init__(self, in_channels: int = 5, base_channels: int = 32, blocks: int = 3):
        super().__init__()
        c = base_channels
        blocks = max(1, int(blocks))
        self.stem = nn.Sequential(nn.Conv2d(in_channels, c, 3, padding=1), nn.SiLU(), ResidualBlock(c))
        self.down1 = nn.Sequential(nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.SiLU(), ResidualBlock(c * 2))
        self.down2 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.SiLU(), ResidualBlock(c * 4))
        self.mid = nn.Sequential(*[ResidualBlock(c * 4) for _ in range(blocks)])
        self.up2 = nn.Sequential(nn.Conv2d(c * 4 + c * 2, c * 2, 3, padding=1), nn.SiLU(), ResidualBlock(c * 2))
        self.up1 = nn.Sequential(nn.Conv2d(c * 2 + c, c, 3, padding=1), nn.SiLU(), ResidualBlock(c))
        self.out_channels = c

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        m = self.mid(s2)
        u2 = F.interpolate(m, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, s1], dim=1))
        u1 = F.interpolate(u2, size=s0.shape[-2:], mode='bilinear', align_corners=False)
        return self.up1(torch.cat([u1, s0], dim=1))


def _canonical_domain_name(name: str) -> str:
    text = str(name).lower()
    if 'synthetic' in text:
        return 'synthetic'
    if 'real' in text:
        return 'real'
    if 'uefb' in text:
        return 'uefb'
    return text


class RLEFFormer(nn.Module):
    def __init__(self, base_channels: int = 32, e_max: float = 3.5,
                 exposure_branch: bool = True, adaptive_gauge: bool = True,
                 fixed_gauge: float | None = None, physics_branch: bool = True,
                 gate_branch: bool = False, q_branch: bool = False,
                 backbone: str = 'tiny', backbone_blocks: int = 2,
                 domain_conditioning: bool = False,
                 domain_names: list[str] | tuple[str, ...] | None = None,
                 domain_embed_dim: int = 8,
                 domain_adapter: str = 'gate_bias',
                 gauge_mode: str | None = None,
                 gauge_mu_min: float = -1.0,
                 gauge_mu_max: float = 2.5,
                 route_type: str = 'identity_gate',
                 safe_alpha: float = 0.70,
                 route_floor: float = 0.0):
        super().__init__()
        self.e_max = float(e_max)
        self.exposure_branch = exposure_branch
        self.adaptive_gauge = adaptive_gauge
        self.fixed_gauge = fixed_gauge
        self.physics_branch = physics_branch
        self.gate_branch = gate_branch
        self.q_branch = q_branch
        self.domain_conditioning = bool(domain_conditioning)
        self.domain_adapter = str(domain_adapter or '')
        if gauge_mode is None:
            gauge_mode = 'adaptive_head' if adaptive_gauge else ('fixed' if fixed_gauge is not None else 'raw_mean')
        self.gauge_mode = str(gauge_mode)
        self.route_type = str(route_type or 'identity_gate')
        self.safe_alpha = float(safe_alpha)
        self.route_floor = min(max(float(route_floor), 0.0), 1.0)
        self.backbone_kind = str(backbone)
        if self.backbone_kind in {'tiny', 'tiny_restormer_lite'}:
            self.backbone = TinyRestormerLite(in_channels=5, base_channels=base_channels, blocks=int(backbone_blocks))
        elif self.backbone_kind in {'multiscale', 'multi_scale'}:
            self.backbone = MultiScaleRestorationBackbone(in_channels=5, base_channels=base_channels, blocks=int(backbone_blocks))
        else:
            raise ValueError(f'Unknown RLEF backbone: {backbone}')
        c = self.backbone.out_channels
        self.rest_head = nn.Conv2d(c, 3, 3, padding=1)
        self.e_head = nn.Conv2d(c, 1, 3, padding=1)
        self.mu_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1), nn.SiLU(), nn.Conv2d(c, 1, 1))
        if self.gauge_mode in {'image_stats', 'image_stats_head'}:
            self.image_stats_gauge_head = ImageStatsGaugeHead(in_dim=6, hidden=32, mu_min=float(gauge_mu_min), mu_max=float(gauge_mu_max))
        self.r_head = nn.Conv2d(c, 3, 3, padding=1)
        self.l_head = nn.Conv2d(c, 1, 3, padding=1)
        self.a_head = nn.Conv2d(c, 1, 3, padding=1)
        self.q_head = nn.Conv2d(c, 1, 3, padding=1)

        self.domain_names = [_canonical_domain_name(x) for x in (domain_names or ['uefb', 'real', 'synthetic'])]
        self.domain_to_idx = {name: i for i, name in enumerate(self.domain_names)}
        if self.domain_conditioning:
            dim = int(domain_embed_dim)
            self.domain_embedding = nn.Embedding(len(self.domain_names), dim)
            adapters = {x.strip() for x in self.domain_adapter.replace(',', '+').split('+') if x.strip()}
            self._domain_adapters = adapters
            if 'feature_affine' in adapters:
                self.domain_feat_affine = nn.Linear(dim, c * 2)
            if 'gate_bias' in adapters:
                self.domain_gate_bias = nn.Linear(dim, 1)
            if 'rest_bias' in adapters or 'head_bias' in adapters:
                self.domain_rest_bias = nn.Linear(dim, 3)
            if 'e_bias' in adapters or 'head_bias' in adapters:
                self.domain_e_bias = nn.Linear(dim, 1)
        else:
            self._domain_adapters = set()

    def _domain_ids(self, domain, batch_size: int, device: torch.device) -> torch.Tensor:
        if not self.domain_conditioning:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if domain is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if torch.is_tensor(domain):
            ids = domain.to(device=device, dtype=torch.long).flatten()
            if ids.numel() == 1:
                ids = ids.repeat(batch_size)
            return ids[:batch_size].clamp(0, max(0, len(self.domain_names) - 1))
        if isinstance(domain, str):
            values = [domain] * batch_size
        else:
            values = list(domain)
            if len(values) == 1 and batch_size > 1:
                values = values * batch_size
        ids = [self.domain_to_idx.get(_canonical_domain_name(v), 0) for v in values[:batch_size]]
        if len(ids) < batch_size:
            ids.extend([0] * (batch_size - len(ids)))
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _domain_vec(self, domain, batch_size: int, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor]:
        ids = self._domain_ids(domain, batch_size, device)
        if not self.domain_conditioning:
            return None, ids
        return self.domain_embedding(ids), ids

    def forward(self, low: torch.Tensor, domain=None) -> dict:
        low = low.clamp(0, 1)
        lum = luminance(low)
        grad = gradient_magnitude(lum)
        feat = self.backbone(torch.cat([low, lum, grad], dim=1))
        b, _, h, w = low.shape
        domain_vec, domain_id = self._domain_vec(domain, b, low.device)
        if domain_vec is not None and 'feature_affine' in self._domain_adapters:
            gamma_beta = torch.tanh(self.domain_feat_affine(domain_vec)).view(b, 2, feat.shape[1], 1, 1)
            gamma, beta = gamma_beta[:, 0], gamma_beta[:, 1]
            feat = feat * (1.0 + 0.1 * gamma) + 0.1 * beta
        head_bias_l2_terms = []
        rest_logits = self.rest_head(feat)
        if domain_vec is not None and hasattr(self, 'domain_rest_bias'):
            rest_bias = self.domain_rest_bias(domain_vec)
            rest_logits = rest_logits + rest_bias.view(b, 3, 1, 1)
            head_bias_l2_terms.append(rest_bias.square().mean(dim=1))
        i_rest = torch.sigmoid(rest_logits).clamp(0, 1)
        if self.exposure_branch:
            e_logits = self.e_head(feat)
            if domain_vec is not None and hasattr(self, 'domain_e_bias'):
                e_bias = self.domain_e_bias(domain_vec)
                e_logits = e_logits + e_bias.view(b, 1, 1, 1)
                head_bias_l2_terms.append(e_bias.square().mean(dim=1))
            e_raw = torch.tanh(e_logits) * self.e_max
            if self.gauge_mode in {'image_stats', 'image_stats_head'}:
                mu = self.image_stats_gauge_head(low)
            elif self.gauge_mode in {'adaptive_head', 'adaptive'} and self.adaptive_gauge:
                mu = torch.tanh(self.mu_head(feat)) * self.e_max
            elif self.gauge_mode == 'fixed' or self.fixed_gauge is not None:
                mu = torch.full((b,1,1,1), float(self.fixed_gauge or 0.0), device=low.device, dtype=low.dtype)
            else:
                mu = e_raw.mean(dim=(2,3), keepdim=True)
            exposure_parts = decompose_exposure(e_raw, mu=mu)
            s = exposure_parts['S']
            mu = exposure_parts['mu']
            e = exposure_parts['E']
        else:
            e_raw = torch.zeros((b,1,h,w), device=low.device, dtype=low.dtype)
            mu = torch.zeros((b,1,1,1), device=low.device, dtype=low.dtype)
            s = e_raw
            e = e_raw
        if self.physics_branch:
            r = torch.sigmoid(self.r_head(feat))
            l_low = torch.sigmoid(self.l_head(feat)).clamp_min(1e-4)
            i_phys = (r * torch.clamp(l_low * torch.exp(e), 0, 1)).clamp(0, 1)
        else:
            r = low
            l_low = lum.clamp_min(1e-4)
            i_phys = low
        if self.gate_branch:
            a_logits = self.a_head(feat)
            if domain_vec is not None and hasattr(self, 'domain_gate_bias'):
                a_logits = a_logits + self.domain_gate_bias(domain_vec).view(b, 1, 1, 1)
            a = torch.sigmoid(a_logits)
            if self.route_floor > 0.0:
                a = (self.route_floor + (1.0 - self.route_floor) * a).clamp(0, 1)
        else:
            a = torch.ones((b,1,h,w), device=low.device, dtype=low.dtype)
        q = torch.sigmoid(self.q_head(feat)) if self.q_branch else torch.ones((b,1,h,w), device=low.device, dtype=low.dtype)
        i_safe = None
        if self.gate_branch and self.route_type in {'recoverability_safe_router', 'recoverability', 'safe_router'}:
            routed = recoverability_route(i_rest, i_phys, low, a, safe_alpha=self.safe_alpha)
            i_hat = routed['I_hat']
            i_safe = routed['I_safe']
            a = routed['A_route']
        else:
            i_hat = (a * i_rest + (1.0 - a) * low).clamp(0, 1) if self.gate_branch else i_rest
        out = {'I_hat': i_hat, 'I_rest': i_rest, 'I_phys': i_phys, 'E_raw': e_raw, 'S': s, 'E': e, 'mu_E': mu, 'A': a, 'Q': q, 'R': r, 'L_low': l_low}
        if i_safe is not None:
            out['I_safe'] = i_safe
        if head_bias_l2_terms:
            out['domain_head_bias_l2'] = torch.stack(head_bias_l2_terms, dim=0).sum(dim=0)
        if self.domain_conditioning:
            out['domain_id'] = domain_id.detach()
        return out
