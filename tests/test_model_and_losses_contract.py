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


def test_rlef_multiscale_backbone_preserves_heads_and_increases_capacity():
    tiny = RLEFFormer(base_channels=8, backbone='tiny', backbone_blocks=1, exposure_branch=True, adaptive_gauge=True, physics_branch=True, gate_branch=True, q_branch=True)
    multi = RLEFFormer(base_channels=8, backbone='multiscale', backbone_blocks=2, exposure_branch=True, adaptive_gauge=True, physics_branch=True, gate_branch=True, q_branch=True)
    x = torch.rand(1, 3, 32, 32)
    out = multi(x)
    assert out['I_hat'].shape == x.shape
    assert out['E'].shape == (1, 1, 32, 32)
    assert out['A'].shape == (1, 1, 32, 32)
    tiny_params = sum(p.numel() for p in tiny.parameters())
    multi_params = sum(p.numel() for p in multi.parameters())
    assert multi_params > tiny_params


def test_rlef_domain_conditioned_gate_bias_changes_same_image_by_domain():
    model = RLEFFormer(
        base_channels=8,
        backbone='multiscale',
        backbone_blocks=1,
        exposure_branch=True,
        adaptive_gauge=True,
        physics_branch=True,
        gate_branch=True,
        q_branch=False,
        domain_conditioning=True,
        domain_names=['uefb', 'real', 'synthetic'],
        domain_embed_dim=4,
        domain_adapter='gate_bias',
    )
    x = torch.rand(2, 3, 32, 32)
    out_real = model(x, domain=['LOL-v2-real-train', 'real'])
    out_syn = model(x, domain=['LOL-v2-synthetic-train', 'synthetic'])
    assert out_real['I_hat'].shape == x.shape
    assert out_real['domain_id'].tolist() == [1, 1]
    assert out_syn['domain_id'].tolist() == [2, 2]
    assert not torch.allclose(out_real['A'], out_syn['A'])


def test_rlef_domain_conditioned_head_bias_reports_anchor_penalty():
    model = RLEFFormer(
        base_channels=8,
        backbone='multiscale',
        backbone_blocks=1,
        exposure_branch=True,
        adaptive_gauge=True,
        physics_branch=True,
        gate_branch=True,
        q_branch=False,
        domain_conditioning=True,
        domain_names=['uefb', 'real', 'synthetic'],
        domain_embed_dim=4,
        domain_adapter='head_bias',
    )
    x = torch.rand(2, 3, 32, 32)
    out = model(x, domain=['real', 'synthetic'])
    assert 'domain_head_bias_l2' in out
    assert out['domain_head_bias_l2'].shape == (2,)
    assert torch.all(out['domain_head_bias_l2'] >= 0)


def test_total_loss_can_anchor_real_domain_head_bias_only():
    pred = torch.zeros(2, 3, 8, 8)
    out = {
        'I_hat': pred,
        'E': torch.zeros(2, 1, 8, 8),
        'domain_head_bias_l2': torch.tensor([2.0, 4.0]),
    }
    batch = {
        'low': pred,
        'high': pred,
        'dataset': ['LOL-v2-real-train', 'LOL-v2-synthetic-train'],
    }
    loss, scalars = compute_total_loss(out, batch, {
        'domain_head_anchor_by_dataset': {
            'real': 0.1,
            'synthetic': 0.0,
        }
    })
    assert torch.isfinite(loss)
    assert scalars['domain_head_anchor_real'] == torch.tensor(0.2)
    assert scalars['domain_head_anchor_synthetic'] == 0
    assert scalars['domain_head_anchor'] == torch.tensor(0.2)


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


def test_total_loss_can_include_lowpass_shape_correlation_term():
    model = RLEFFormer(base_channels=8, exposure_branch=True, adaptive_gauge=True, physics_branch=True, gate_branch=True, q_branch=False)
    low = torch.rand(2, 3, 24, 24) * 0.4
    high = (low + 0.2).clamp(0, 1)
    batch = {
        'low': low,
        'high': high,
        'E_gt': torch.linspace(0, 1, 24).view(1, 1, 1, 24).repeat(2, 1, 24, 1),
        'A_gt': torch.ones(2, 1, 24, 24) * 0.7,
    }
    out = model(low)
    loss, scalars = compute_total_loss(out, batch, {
        'rec': 1.0, 'e_shape': 0.05, 'e_shape_kernel': 5,
    })
    assert torch.isfinite(loss)
    assert 'e_shape' in scalars
    assert torch.isfinite(scalars['e_shape'])


def test_total_loss_can_distill_from_teacher_target():
    model = RLEFFormer(base_channels=8, exposure_branch=True, adaptive_gauge=True, physics_branch=True, gate_branch=True, q_branch=False)
    low = torch.rand(2, 3, 24, 24) * 0.4
    high = (low + 0.2).clamp(0, 1)
    teacher = (low + 0.35).clamp(0, 1)
    batch = {'low': low, 'high': high, 'teacher': teacher}
    out = model(low)
    loss, scalars = compute_total_loss(out, batch, {'rec': 1.0, 'distill': 0.25})
    assert torch.isfinite(loss)
    assert 'distill' in scalars
    assert torch.isfinite(scalars['distill'])


def test_total_loss_can_use_dataset_weighted_reconstruction_for_synthetic_protection():
    pred = torch.zeros(2, 3, 8, 8)
    high = torch.ones_like(pred) * 0.5
    out = {'I_hat': pred, 'E': torch.zeros(2, 1, 8, 8)}
    batch = {
        'low': pred,
        'high': high,
        'dataset': ['LOL-v2-real-train', 'LOL-v2-synthetic-train'],
    }
    loss, scalars = compute_total_loss(out, batch, {
        'rec_by_dataset': {
            'LOL-v2-real-train': 1.0,
            'LOL-v2-synthetic-train': 1.25,
            'UEFB': 0.0,
        }
    })
    assert torch.isfinite(loss)
    assert 'rec' in scalars
    assert 'rec_LOL-v2-real-train' in scalars
    assert 'rec_LOL-v2-synthetic-train' in scalars
    assert 'rec_UEFB' in scalars
    assert scalars['rec_LOL-v2-synthetic-train'] > scalars['rec_LOL-v2-real-train']
    assert scalars['rec_UEFB'] == 0


def test_total_loss_can_penalize_gate_opening_on_near_identity_pixels():
    pred = torch.zeros(2, 3, 8, 8)
    low = torch.zeros_like(pred)
    high_near = low.clone()
    high_far = torch.ones_like(pred)
    out = {
        'I_hat': pred,
        'E': torch.zeros(2, 1, 8, 8),
        'A': torch.ones(2, 1, 8, 8) * 0.75,
    }
    near_loss, near_scalars = compute_total_loss(out, {'low': low, 'high': high_near}, {'gate_identity': 1.0})
    far_loss, far_scalars = compute_total_loss(out, {'low': low, 'high': high_far}, {'gate_identity': 1.0})
    assert torch.isfinite(near_loss)
    assert 'gate_identity' in near_scalars
    assert near_scalars['gate_identity'] > far_scalars['gate_identity']


def test_total_loss_can_use_domain_conditioned_distill_weights():
    pred = torch.zeros(2, 3, 8, 8)
    teacher = torch.stack([torch.ones(3, 8, 8), torch.ones(3, 8, 8) * 0.5])
    low = torch.zeros_like(pred)
    high = teacher.clone()
    out = {'I_hat': pred, 'E': torch.zeros(2, 1, 8, 8)}
    batch = {
        'low': low,
        'high': high,
        'teacher': teacher,
        'dataset': ['LOL-v2-real-train-retinex-teacher', 'LOL-v2-synthetic-train-retinex-teacher'],
    }
    loss, scalars = compute_total_loss(out, batch, {
        'distill_by_dataset': {
            'LOL-v2-real-train-retinex-teacher': 0.30,
            'LOL-v2-synthetic-train-retinex-teacher': 0.05,
        }
    })
    assert torch.isfinite(loss)
    assert 'distill' in scalars
    assert 'distill_LOL-v2-real-train-retinex-teacher' in scalars
    assert 'distill_LOL-v2-synthetic-train-retinex-teacher' in scalars
    assert scalars['distill_LOL-v2-real-train-retinex-teacher'] > scalars['distill_LOL-v2-synthetic-train-retinex-teacher']


def test_domain_conditioned_distill_logs_zero_for_absent_domains():
    pred = torch.zeros(1, 3, 8, 8)
    teacher = torch.ones_like(pred)
    out = {'I_hat': pred, 'E': torch.zeros(1, 1, 8, 8)}
    batch = {'low': pred, 'high': teacher, 'teacher': teacher, 'dataset': ['UEFB']}
    _, scalars = compute_total_loss(out, batch, {
        'distill_by_dataset': {
            'LOL-v2-real-train-retinex-teacher': 0.30,
            'LOL-v2-synthetic-train-retinex-teacher': 0.05,
            'UEFB': 0.0,
        }
    })
    assert scalars['distill_LOL-v2-real-train-retinex-teacher'] == 0
    assert scalars['distill_LOL-v2-synthetic-train-retinex-teacher'] == 0
    assert scalars['distill_UEFB'] == 0
