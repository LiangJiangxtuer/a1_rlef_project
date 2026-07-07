import pytest
import torch

from rlef.losses.gauge_scheduler import warm_gauge_anchor_weight
from rlef.models.gauge_head import ImageStatsGaugeHead, image_luminance_stats


def test_warm_gauge_anchor_weight_respects_warmup_ramp_and_cap():
    assert warm_gauge_anchor_weight(0) == 0.0
    assert warm_gauge_anchor_weight(299) == 0.0
    assert warm_gauge_anchor_weight(300) == 0.0
    assert warm_gauge_anchor_weight(500) == pytest.approx(0.0025)
    assert warm_gauge_anchor_weight(700) == pytest.approx(0.005)
    assert warm_gauge_anchor_weight(1000) == pytest.approx(0.005)

    with pytest.raises(ValueError, match='exceeds hard cap'):
        warm_gauge_anchor_weight(1000, max_weight=0.02, hard_cap=0.010)


def test_image_luminance_stats_returns_deployable_six_stat_vector():
    low = torch.zeros(2, 3, 8, 8)
    low[0] = 0.05
    low[1] = 1.0

    stats = image_luminance_stats(low)

    assert stats.shape == (2, 6)
    # mean(Y), std(Y), dark_ratio, sat_ratio, grad_mean, local_contrast
    assert stats[0, 0] < stats[1, 0]
    assert stats[0, 2] == pytest.approx(1.0)
    assert stats[1, 3] == pytest.approx(1.0)
    assert torch.all(stats >= 0)


def test_image_stats_gauge_head_outputs_bounded_image_scalar():
    low = torch.rand(4, 3, 16, 16)
    head = ImageStatsGaugeHead(in_dim=6, hidden=12, mu_min=-1.0, mu_max=2.5)

    mu = head(low)

    assert mu.shape == (4, 1, 1, 1)
    assert float(mu.min()) >= -1.0
    assert float(mu.max()) <= 2.5
    mu.sum().backward()
    assert all(p.grad is not None for p in head.parameters())
