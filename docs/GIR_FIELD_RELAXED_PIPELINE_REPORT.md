# GIR-Field relaxed pipeline report

## Protocol

- Claim scope: `relaxed-routing-evidence-not-paper-grade-final`
- No main-model training: `True`
- DGB revival allowed: `False`
- Max images per split: `64`
- Bootstrap samples: `1000`

## Key relaxed-run findings

- M4J_ES vs M4J on UEFB S_corr: delta=0.1504, 95% bootstrap CI=[0.1030, 0.1978], Wilcoxon p=5.62739581705274e-08, FDR q=1.5347443137416562e-07.
- M4J_ES vs M4J on real PSNR: delta=0.4904 dB, 95% CI=[0.1812, 0.7964].
- Perturbation `global_shift_plus_0p5`: mean E_MAE=0.5000, mean S_MAE=0.0000, mean Gauge_MAE=0.5000, mean S_corr=1.0000.
- Perturbation `local_shape_distortion`: mean E_MAE=0.2026, mean S_MAE=0.2026, mean Gauge_MAE=0.0000, mean S_corr=0.6027.
- Perturbation `mixed_shift_plus_shape`: mean E_MAE=0.5000, mean S_MAE=0.2026, mean Gauge_MAE=0.5000, mean S_corr=0.6027.
- N4 risk probe `M4J_ES`: status=ok, logistic_auc=0.7781900434532432, logistic_ece=0.2561714644978444.
- N4 risk probe `P3C_E0050_S3407`: status=ok, logistic_auc=0.7778973789162483, logistic_ece=0.25622736087068915.

## Claim calibration

- This relaxed run supports GIR-Field/UEFB-G as a mechanism + benchmark route, not a SOTA LLIE method claim.
- DGB/P2F was not resumed; DGB remains stopped and consolidated.
- LPIPS/BRISQUE were not fabricated when packages were unavailable.
- External black-box baselines are output-only; internal E/S/Gauge metrics are N/A for them.

## Main artifacts

- `results/girfield_relaxed/N0_evidence/claim_ledger.csv`
- `results/girfield_relaxed/N1_statistics/per_image_metrics.csv`
- `results/girfield_relaxed/N1_statistics/statistical_tests.csv`
- `results/girfield_relaxed/N2_uefbg/gauge_perturbation_metrics.csv`
- `results/girfield_relaxed/N3_external/external_baseline_registry.csv`
- `results/girfield_relaxed/N4_risk/risk_calibration_summary.csv`
- `results/girfield_relaxed/N5_paper/girfield_main_table.csv`
- `results/girfield_relaxed/N5_paper/fig_psnr_vs_scorr.png`
- `results/girfield_relaxed/N5_paper/fig_gauge_perturbation_bar.png`
- `results/girfield_relaxed/N5_paper/fig_risk_reliability.png`
- `results/girfield_relaxed/N5_paper/fig_psnr_misranking_failure_grid.png`
- `results/girfield_relaxed/GIR_FIELD_RELAXED_PIPELINE_MANIFEST.json`
