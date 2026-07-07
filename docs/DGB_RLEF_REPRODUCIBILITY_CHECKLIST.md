# DGB_RLEF Reproducibility Checklist

Generated: 2026-07-05T03:06:24.533914+00:00

## Project

```text
root: /home/user/a1_rlef_project
active default: P3c M4J_ES e_shape=0.05
DGB status: stopped and consolidated
```

## Data paths

- `data/LOL-v2/Real_captured/Train|Test/Low|Normal`
- `data/LOL-v2/Synthetic/Train|Test/Low|Normal`
- `data/UEFB-v2/train|test/low|high|E_gt|A_gt|Q_gt`
- `data/unpaired_real/DICM|LIME|MEF|NPE|VV`

## Seeds

```text
3407, 2027, 42
```

## Core configs/checkpoints

- P3c/P3b visual checkpoint: `configs/p3b_joint/p3b_m4j_es_m4_joint_eshape.yml`
- Checkpoint: `experiments/p3b_m4j_es_m4_joint_eshape_seed3407/checkpoints/last.pth`
- P3c aggregate table: `results/tables/p3c_multiseed_sweep_aggregate.json`
- Official baseline table: `results/tables/p4_official_baselines_summary.json`

## Commands

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_s1_s4_paper_pipeline.py --device cuda --n_visuals 4
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_master_remaining_pipeline.py --device cuda
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m pytest tests -q
```

## Metric scripts / tables

- `scripts/eval_uefb.py`
- `scripts/eval_paired.py`
- `scripts/run_s1_s4_paper_pipeline.py`
- `scripts/run_master_remaining_pipeline.py`
- `results/hermes_audit/tables/final_main_table.csv`
- `results/hermes_audit/tables/final_ablation_table.csv`
- `results/hermes_audit/tables/noref_supplementary_summary.csv`

## Baseline versions

- Retinexformer: local official-code outputs under `experiments/p4_official_baselines/retinexformer/`.
- Zero-DCE++: local official-code outputs under `experiments/p4_official_baselines/zero_dce_pp/`.
- KinD++: local official-code high-assisted outputs under `experiments/p4_official_baselines/kindpp/`; mark high-assisted.

## Verification

- Do not claim SOTA unless future evidence changes the paired-fidelity gap.
- Do not promote DGB to 3-seed unless a new candidate passes the joint gate under a new branch.
- Keep DGB/P5/P6/P7 scalar routes in appendix/negative evidence.
