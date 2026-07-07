# GIR-Field formal full protocol report

## Scope

- Claim scope: `formal-full-frozen-evidence`
- Full local evaluation: UEFB-v2 test + LOL-v2 real test + LOL-v2 synthetic test.
- Dataset counts: UEFB=500, real=100, synthetic=100, expected per-image rows=6300.
- Bootstrap samples: 5000.
- No main-model training; frozen checkpoints only; DGB/P2F not resumed.

## Key formal statistics

- M4J_ES_minus_M4J, uefb, S_corr: delta=0.1581, 95% CI=[0.1369, 0.1791], Wilcoxon p=1.4511464133386487e-44, FDR q=2.176719620007973e-43, win_rate=0.826.
- M4J_ES_minus_M4J, real, psnr: delta=0.6963 dB, 95% CI=[0.4141, 0.9664], Wilcoxon p=1.2033085813778238e-05, FDR q=1.5695329322319443e-05, win_rate=0.660.
- M4J_ES_minus_M4J, synthetic, psnr: delta=0.6321 dB, 95% CI=[0.4182, 0.8559], Wilcoxon p=3.625288140148871e-08, FDR q=6.042146900248119e-08, win_rate=0.760.
- M4J_ES_minus_M4, uefb, S_corr: delta=0.5792, 95% CI=[0.5361, 0.6230], Wilcoxon p=1.1735135854226827e-65, FDR q=3.5205407562680485e-64, win_rate=0.858.

## Risk calibration probe

- M4J_ES: status=ok, AUC=0.7658007881167234, ECE=0.2807872076183558, test positive rate=0.14133333333333334.
- P3C_E0050_S3407: status=ok, AUC=0.7654675231454353, ECE=0.2809492608259121, test positive rate=0.14133333333333334.

## Claim calibration

- Supported: centered E-shape improves gauge-free exposure-shape identifiability over M4J under full local eval.
- Supported: UEFB-G gauge perturbation separates global gauge shifts from local shape distortions.
- Diagnostic only: recoverability risk has signal but remains calibration-limited.
- Not supported: DGB-RLEF SOTA, DGB tri-domain resolution, Retinexformer outperformance.

## Artifacts

- `results/girfield_formal/N1_statistics/per_image_metrics.csv`
- `results/girfield_formal/N1_statistics/statistical_tests.csv`
- `results/girfield_formal/N2_uefbg/gauge_perturbation_metrics.csv`
- `results/girfield_formal/N4_risk/risk_calibration_summary.csv`
- `results/girfield_formal/N5_paper/girfield_main_table.md`
- `results/girfield_formal/GIR_FIELD_FORMAL_PROTOCOL_MANIFEST.json`
