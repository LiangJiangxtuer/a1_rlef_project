import torch

from rlef.models.exposure_gauge import decompose_exposure


def test_decompose_exposure_returns_zero_mean_shape_and_scalar_mu():
    e_raw = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).view(2, 1, 4, 4)
    mu = torch.tensor([[[[0.25]]], [[[1.25]]]])

    parts = decompose_exposure(e_raw, mu=mu)

    assert set(parts) == {'S', 'mu', 'E'}
    assert parts['S'].shape == e_raw.shape
    assert parts['mu'].shape == (2, 1, 1, 1)
    assert parts['E'].shape == e_raw.shape
    assert torch.allclose(parts['S'].mean(dim=(2, 3), keepdim=True), torch.zeros_like(mu), atol=1e-6)
    assert torch.allclose(parts['E'], parts['S'] + parts['mu'])


def test_decompose_exposure_default_mu_preserves_original_field_mean():
    e_raw = torch.randn(3, 1, 5, 7)

    parts = decompose_exposure(e_raw)

    expected_mu = e_raw.mean(dim=(2, 3), keepdim=True)
    assert torch.allclose(parts['mu'], expected_mu)
    assert torch.allclose(parts['E'], e_raw, atol=1e-6)
