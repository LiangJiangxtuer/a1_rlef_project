import torch

from rlef.models.recoverability_router import recoverability_route, safe_recovery_image


def test_recoverability_router_endpoints_match_restorer_and_safe_path():
    low = torch.full((1, 3, 4, 4), 0.2)
    i_rest = torch.full_like(low, 0.8)
    i_phys = torch.full_like(low, 0.5)

    open_route = recoverability_route(i_rest, i_phys, low, torch.ones(1, 1, 4, 4), safe_alpha=0.70)
    closed_route = recoverability_route(i_rest, i_phys, low, torch.zeros(1, 1, 4, 4), safe_alpha=0.70)
    expected_safe = 0.70 * low + 0.30 * i_phys.clamp(0, 1)

    assert torch.allclose(open_route['I_hat'], i_rest)
    assert torch.allclose(closed_route['I_hat'], expected_safe)
    assert torch.allclose(closed_route['I_safe'], expected_safe)


def test_recoverability_router_clamps_inputs_and_gate_range():
    low = torch.full((1, 3, 2, 2), 0.1)
    i_rest = torch.full_like(low, 2.0)
    i_phys = torch.full_like(low, -1.0)
    a = torch.full((1, 1, 2, 2), 0.5)

    out = recoverability_route(i_rest, i_phys, low, a, safe_alpha=0.70)

    assert out['I_hat'].shape == low.shape
    assert float(out['I_hat'].min()) >= 0.0
    assert float(out['I_hat'].max()) <= 1.0
    assert torch.allclose(safe_recovery_image(low, i_phys, safe_alpha=0.70), torch.full_like(low, 0.07))
