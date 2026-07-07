import subprocess

PY = '/home/user/miniconda3/envs/cutler_dinov3/bin/python'


def test_eval_paired_cli_imports_before_argument_parsing():
    proc = subprocess.run([PY, 'scripts/eval_paired.py', '--help'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr
    assert '--checkpoint' in proc.stdout
    assert '--domain' in proc.stdout


def test_eval_uefb_cli_imports_before_argument_parsing():
    proc = subprocess.run([PY, 'scripts/eval_uefb.py', '--help'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr
    assert '--checkpoint' in proc.stdout
