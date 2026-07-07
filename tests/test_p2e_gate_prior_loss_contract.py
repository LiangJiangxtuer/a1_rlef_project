import torch

from rlef.losses.total_loss import compute_total_loss


def _case(low_value, a_value):
    low = torch.full((2, 3, 8, 8), low_value)
    high = torch.full((2, 3, 8, 8), 0.6)
    pred = torch.full((2, 3, 8, 8), 0.5)
    return {
        'I_hat': pred,
        'I_phys': pred,
        'E': torch.zeros(2, 1, 8, 8),
        'A': torch.full((2, 1, 8, 8), a_value),
        'Q': torch.ones(2, 1, 8, 8),
    }, {'low': low, 'high': high, 'dataset': ['real', 'uefb']}


def test_image_stat_gate_prior_penalizes_under_routing_dark_inputs_more_than_bright_inputs():
    weights = {
        'gate_prior': 1.0,
        'gate_prior_base': 0.35,
        'gate_prior_dark_gain': 0.30,
        'gate_prior_contrast_gain': 0.0,
        'gate_prior_min': 0.25,
        'gate_prior_max': 0.75,
    }
    dark_out, dark_batch = _case(0.02, 0.20)
    bright_out, bright_batch = _case(0.55, 0.20)

    dark_loss, dark_scalars = compute_total_loss(dark_out, dark_batch, weights)
    bright_loss, bright_scalars = compute_total_loss(bright_out, bright_batch, weights)

    assert 'gate_prior' in dark_scalars
    assert 'gate_prior' in bright_scalars
    assert dark_loss > bright_loss
    assert dark_scalars['gate_prior'] > bright_scalars['gate_prior']


def test_image_stat_gate_prior_is_generic_not_dataset_specific():
    out, batch = _case(0.03, 0.10)

    loss, scalars = compute_total_loss(out, batch, {'gate_prior': 0.05})

    assert loss.item() > 0
    assert 'gate_prior' in scalars
    assert 'rec_by_dataset' not in scalars
    assert not any(key.startswith('gate_prior_real') for key in scalars)
    assert not any(key.startswith('gate_prior_uefb') for key in scalars)
