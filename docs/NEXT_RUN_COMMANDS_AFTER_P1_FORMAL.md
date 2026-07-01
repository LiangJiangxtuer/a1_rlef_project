# Next commands after P1 formal

P1 formal 1000-step has been executed. Results:

```text
results/tables/p1_formal_1000_summary.csv
docs/P1_FORMAL_1000_REPORT.md
```

## Execute P2 formal UEFB-v2 generation

```bash
cd /home/user/a1_rlef_project
PY=/home/user/miniconda3/envs/cutler_dinov3/bin/python
$PY scripts/make_uefb_v2.py   --source data/LOL-v2/Real_captured/Train/Normal   --output data/UEFB-v2   --num_train 3000 --num_test 500 --image_size 256 --seed 3407
```

## Then execute P3 formal M0-M5 ablation

Use adaptive gauge as the primary gauge candidate. Fixed0p02 remains only a fragile baseline.
