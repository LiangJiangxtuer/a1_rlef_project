import torch

from rlef.metrics.full_reference import psnr_torch, ssim_torch
from rlef.metrics.exposure_field import exposure_field_metrics, identity_drop, q_ece


def test_psnr_identity_is_large_and_ssim_identity_is_one():
    x = torch.rand(2, 3, 16, 16)
    assert psnr_torch(x, x).item() > 60
    assert abs(ssim_torch(x, x).item() - 1.0) < 1e-5


def test_exposure_field_metrics_separate_gauge_offset_from_shape():
    e_gt = torch.randn(2, 1, 8, 8)
    e_pred = e_gt + 2.0
    m = exposure_field_metrics(e_pred, e_gt)
    assert m['E_MAE'] > 1.9
    assert m['E_MAE_aligned'] < 1e-5
    assert m['E_corr'] > 0.999


def test_identity_drop_positive_when_output_matches_gt_better_than_input():
    high = torch.ones(1, 3, 8, 8) * 0.6
    low = torch.ones(1, 3, 8, 8) * 0.3
    pred = torch.ones(1, 3, 8, 8) * 0.55
    assert identity_drop(pred, low, high).item() > 0


def test_q_ece_is_low_for_perfect_error_confidence():
    err = torch.linspace(0, 1, 16).view(1, 1, 4, 4)
    q = 1.0 - err
    assert q_ece(q, err, n_bins=4).item() < 1e-6
