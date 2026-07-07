import torch

from rlef.losses.e_shape_loss import e_shape_loss, gauge_invariant_e_shape_loss


def test_e_shape_loss_is_invariant_to_additive_gauge_shift():
    yy = torch.linspace(-1, 1, 17).view(1, 1, 17, 1)
    xx = torch.linspace(-1, 1, 19).view(1, 1, 1, 19)
    target = torch.sin(xx * 2.0) + torch.cos(yy * 1.5)
    pred = target + 0.05 * torch.sin(xx * yy)

    base = e_shape_loss(pred, target)
    shifted = e_shape_loss(pred + 4.0, target)

    assert torch.isfinite(base)
    assert torch.allclose(base, shifted, atol=1e-6)


def test_e_shape_loss_defaults_match_public_gauge_invariant_api():
    target = torch.randn(2, 1, 13, 15)
    pred = target.flip(-1)

    loss_a = e_shape_loss(pred, target)
    loss_b = gauge_invariant_e_shape_loss(pred, target, kernel_size=7, beta=0.1)

    assert torch.isfinite(loss_a)
    assert loss_a >= 0
    assert torch.allclose(loss_a, loss_b)
    assert e_shape_loss(target, target) < loss_a
