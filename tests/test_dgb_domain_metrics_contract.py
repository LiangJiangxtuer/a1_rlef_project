import torch

from rlef.metrics.domain_metrics import compute_domain_metrics


def test_domain_metrics_returns_required_keys_with_optional_dgb_fields():
    low = torch.full((1, 3, 4, 4), 0.2)
    high = torch.full_like(low, 0.6)
    pred = torch.full_like(low, 0.55)
    e_gt = torch.linspace(0, 1, 16).view(1, 1, 4, 4)
    e_pred = e_gt + 2.0
    a = torch.linspace(0, 1, 16).view(1, 1, 4, 4)
    a_gt = (a > 0.5).float()
    q = torch.full((1, 1, 4, 4), 0.8)

    metrics = compute_domain_metrics(pred, high, low, e_pred=e_pred, e_gt=e_gt, a=a, a_gt=a_gt, q=q)

    required = {
        'psnr', 'ssim', 'lee', 'nai',
        'E_MAE', 'E_MAE_aligned', 'E_corr',
        'Gauge_MAE', 'S_corr',
        'A_AUC', 'Q_ECE', 'identity_drop', 'unsafe_overenhance', 'route_entropy',
    }
    assert required.issubset(metrics)
    assert metrics['Gauge_MAE'] == torch.tensor(2.0).item()
    assert metrics['S_corr'] > 0.99
    assert 0.0 <= metrics['A_AUC'] <= 1.0
    assert metrics['route_entropy'] > 0.0
    assert metrics['notes'] == []


def test_domain_metrics_marks_missing_optional_ground_truth_explicitly():
    low = torch.full((1, 3, 4, 4), 0.2)
    high = torch.full_like(low, 0.6)
    pred = torch.full_like(low, 0.5)

    metrics = compute_domain_metrics(pred, high, low)

    assert metrics['E_MAE'] is None
    assert metrics['Gauge_MAE'] is None
    assert metrics['A_AUC'] is None
    assert metrics['Q_ECE'] is None
    assert 'missing_e_gt_or_e_pred' in metrics['notes']
    assert 'missing_a_gt_or_a' in metrics['notes']
    assert 'missing_q' in metrics['notes']
