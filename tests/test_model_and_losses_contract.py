import torch

from rlef.models.rlef_former import RLEFFormer
from rlef.losses.total_loss import compute_total_loss


def test_rlef_full_outputs_have_expected_shapes_and_ranges():
    model = RLEFFormer(base_channels=8, exposure_branch=True, adaptive_gauge=True, physics_branch=True, gate_branch=True, q_branch=True)
    x = torch.rand(2, 3, 32, 32)
    out = model(x)
    for key in ['I_hat', 'I_rest', 'I_phys', 'E_raw', 'E', 'mu_E', 'A', 'Q']:
        assert key in out
    assert out['I_hat'].shape == x.shape
    assert out['E'].shape == (2, 1, 32, 32)
    assert out['mu_E'].shape == (2, 1, 1, 1)
    assert float(out['I_hat'].min()) >= 0 and float(out['I_hat'].max()) <= 1
    assert float(out['A'].min()) >= 0 and float(out['A'].max()) <= 1
    assert float(out['Q'].min()) >= 0 and float(out['Q'].max()) <= 1
    assert torch.allclose(out['E'].mean(dim=(2,3), keepdim=True), out['mu_E'], atol=1e-5)


def test_total_loss_returns_requested_scalars():
    model = RLEFFormer(base_channels=8, exposure_branch=True, adaptive_gauge=True, physics_branch=True, gate_branch=True, q_branch=True)
    low = torch.rand(2, 3, 24, 24) * 0.4
    high = (low + 0.2).clamp(0, 1)
    batch = {
        'low': low,
        'high': high,
        'E_gt': torch.ones(2, 1, 24, 24) * 0.5,
        'A_gt': torch.ones(2, 1, 24, 24) * 0.7,
        'Q_gt': torch.ones(2, 1, 24, 24) * 0.9,
    }
    out = model(low)
    loss, scalars = compute_total_loss(out, batch, {
        'rec': 1.0, 'phys': 0.1, 'poisson': 0.05, 'gauge': 0.1, 'id': 0.02, 'gate': 0.02, 'q': 0.02, 'wtv': 0.01,
    })
    assert torch.isfinite(loss)
    for key in ['rec', 'phys', 'poisson', 'gauge', 'id', 'gate', 'q', 'wtv']:
        assert key in scalars
        assert torch.isfinite(scalars[key])
