import torch

from rlef.models.rlef_former import RLEFFormer
from scripts.train import build_model


def test_route_floor_is_global_minimum_restoration_weight_for_identity_route():
    model = RLEFFormer(
        base_channels=8,
        backbone='tiny',
        backbone_blocks=1,
        exposure_branch=True,
        adaptive_gauge=True,
        physics_branch=True,
        gate_branch=True,
        q_branch=True,
        gauge_mode='image_stats',
        route_type='identity_gate',
        route_floor=0.25,
    )
    low = torch.rand(2, 3, 20, 20) * 0.4

    out = model(low)

    assert out['A'].shape == (2, 1, 20, 20)
    assert float(out['A'].min()) >= 0.25 - 1e-6
    expected = out['A'] * out['I_rest'] + (1.0 - out['A']) * low.clamp(0, 1)
    assert torch.allclose(out['I_hat'], expected.clamp(0, 1), atol=1e-6)


def test_build_model_maps_nested_route_floor_config():
    model = build_model({
        'model': {
            'base_channels': 8,
            'backbone': 'tiny',
            'backbone_blocks': 1,
            'gate_branch': True,
            'q_branch': True,
            'gauge': {'mode': 'image_stats'},
            'route': {'type': 'identity_gate', 'floor': 0.15},
        }
    })

    assert model.route_type == 'identity_gate'
    assert model.route_floor == 0.15
