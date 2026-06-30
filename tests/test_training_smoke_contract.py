import json
import subprocess
from pathlib import Path
from PIL import Image
import yaml

PY = '/home/user/miniconda3/envs/cutler_dinov3/bin/python'


def test_train_and_eval_scripts_complete_on_tiny_uefb(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    for i in range(4):
        Image.new('RGB', (40, 40), color=(70 + i * 20, 90 + i * 10, 120)).save(source / f'{i:03d}.png')
    data = tmp_path / 'UEFB-smoke'
    subprocess.run([PY, 'scripts/make_uefb_v2.py', '--source', str(source), '--output', str(data), '--num_train', '3', '--num_test', '1', '--image_size', '32', '--seed', '1'], check=True)
    cfg = tmp_path / 'cfg.yml'
    cfg_obj = {
        'seed': 1,
        'model': {'base_channels': 8, 'exposure_branch': True, 'adaptive_gauge': True, 'physics_branch': True, 'gate_branch': True, 'q_branch': True},
        'training': {'batch_size': 2, 'max_steps': 2, 'lr': 0.0002, 'crop_size': 24, 'log_every': 1},
        'data': {
            'train': {'type': 'uefb', 'root': str(data / 'train'), 'crop_size': 24},
            'val': {'type': 'uefb', 'root': str(data / 'test'), 'crop_size': 24},
        },
        'loss': {'rec': 1.0, 'phys': 0.1, 'poisson': 0.05, 'gauge': 0.1, 'id': 0.02, 'gate': 0.02, 'q': 0.02, 'wtv': 0.01},
    }
    cfg.write_text(yaml.safe_dump(cfg_obj), encoding='utf-8')
    out = tmp_path / 'run'
    subprocess.run([PY, 'scripts/train.py', '--config', str(cfg), '--output_dir', str(out), '--device', 'cpu', '--max_steps', '2'], check=True)
    metrics = json.loads((out / 'metrics' / 'eval_metrics.json').read_text())
    assert 'psnr' in metrics and 'E_MAE' in metrics
    assert (out / 'checkpoints' / 'last.pth').exists()
