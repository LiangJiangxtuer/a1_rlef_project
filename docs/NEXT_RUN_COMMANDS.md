# Formal RLEF-v2 next-run commands

These commands are intentionally not marked as executed until their result tables exist.

```bash
cd /home/user/a1_rlef_project
PY=/home/user/miniconda3/envs/cutler_dinov3/bin/python

# P2 formal UEFB-v2 (adjust source if FiveK/SICE becomes available)
$PY scripts/make_uefb_v2.py   --source data/LOL-v2/Real_captured/Train/Normal   --output data/UEFB-v2   --num_train 3000 --num_test 500 --image_size 256 --seed 3407

# P3 formal single-seed smoke-to-formal promotion (example)
for cfg in configs/rlef_m0_restorer_smoke.yml configs/rlef_m3_adaptive_gauge_smoke.yml configs/rlef_m4_gate_smoke.yml configs/rlef_m5_recoverability_smoke.yml; do
  tag=$(basename "$cfg" .yml)
  $PY scripts/train.py --config "$cfg" --output_dir "experiments/formal_${tag}_seed3407" --device cuda --max_steps 1000
done
```
