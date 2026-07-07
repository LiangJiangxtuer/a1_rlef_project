import torch

from rlef.models.rlef_former import RLEFFormer
from scripts.train import scheduled_loss_weights


def test_dgb_rlef_model_uses_image_stats_gauge_and_safe_recoverability_route():
    model = RLEFFormer(
        base_channels=8,
        backbone='multiscale',
        backbone_blocks=1,
        exposure_branch=True,
        adaptive_gauge=True,
        physics_branch=True,
        gate_branch=True,
        q_branch=True,
        gauge_mode='image_stats',
        gauge_mu_min=-1.0,
        gauge_mu_max=2.5,
        route_type='recoverability_safe_router',
        safe_alpha=0.70,
    )
    low = torch.rand(2, 3, 24, 24) * 0.4

    out = model(low)

    assert out['I_hat'].shape == low.shape
    assert out['I_safe'].shape == low.shape
    assert out['S'].shape == (2, 1, 24, 24)
    assert out['Q'].shape == (2, 1, 24, 24)
    assert float(out['mu_E'].min()) >= -1.0
    assert float(out['mu_E'].max()) <= 2.5
    assert torch.allclose(out['S'].mean(dim=(2, 3), keepdim=True), torch.zeros_like(out['mu_E']), atol=1e-5)
    assert torch.allclose(out['E'], out['S'] + out['mu_E'], atol=1e-6)
    expected_safe = 0.70 * low.clamp(0, 1) + 0.30 * out['I_phys'].clamp(0, 1)
    expected_hat = out['A'] * out['I_rest'] + (1.0 - out['A']) * expected_safe
    assert torch.allclose(out['I_safe'], expected_safe.clamp(0, 1), atol=1e-6)
    assert torch.allclose(out['I_hat'], expected_hat.clamp(0, 1), atol=1e-6)


def test_scheduled_loss_weights_apply_warm_gauge_without_mutating_base_weights():
    base = {
        'rec': 1.0,
        'gauge': 0.0,
        'gauge_schedule': {
            'ramp_start': 300,
            'full_start': 700,
            'max_weight': 0.005,
            'hard_cap': 0.010,
        },
    }

    early = scheduled_loss_weights(base, 299)
    mid = scheduled_loss_weights(base, 500)
    late = scheduled_loss_weights(base, 700)

    assert early['gauge'] == 0.0
    assert mid['gauge'] == 0.0025
    assert late['gauge'] == 0.005
    assert base['gauge'] == 0.0
    assert 'gauge_schedule' not in late
