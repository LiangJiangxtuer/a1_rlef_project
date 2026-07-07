import torch

from rlef.losses.total_loss import compute_total_loss


def _batch_and_out(pred):
    low = torch.zeros_like(pred) + 0.1
    high = torch.zeros_like(pred) + 0.1
    high[:, :, :, 4:] = 0.9
    batch = {'low': low, 'high': high, 'dataset': ['uefb'] * pred.shape[0]}
    out = {
        'I_hat': pred,
        'I_phys': pred,
        'E': torch.zeros(pred.shape[0], 1, pred.shape[2], pred.shape[3], dtype=pred.dtype),
        'A': torch.ones(pred.shape[0], 1, pred.shape[2], pred.shape[3], dtype=pred.dtype),
        'Q': torch.ones(pred.shape[0], 1, pred.shape[2], pred.shape[3], dtype=pred.dtype),
    }
    return out, batch


def test_structure_preserve_loss_is_zero_for_perfect_prediction_and_exposes_scalar():
    high = torch.zeros(2, 3, 8, 8) + 0.1
    high[:, :, :, 4:] = 0.9
    out, batch = _batch_and_out(high.clone())

    loss, scalars = compute_total_loss(out, batch, {'structure_preserve': 0.25})

    assert 'structure_preserve' in scalars
    assert torch.isclose(scalars['structure_preserve'], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_structure_preserve_loss_is_generic_not_dataset_label_weighted():
    pred = torch.zeros(2, 3, 8, 8) + 0.2
    out, batch = _batch_and_out(pred)
    batch['dataset'] = ['uefb', 'real']

    loss, scalars = compute_total_loss(out, batch, {
        'structure_preserve': 0.25,
        'structure_preserve_grad': 0.5,
        'structure_preserve_floor': 0.25,
    })

    assert 'structure_preserve' in scalars
    assert loss.item() > 0
    assert 'rec_by_dataset' not in scalars
    assert not any(key.startswith('structure_preserve_uefb') for key in scalars)
    assert not any(key.startswith('structure_preserve_real') for key in scalars)
