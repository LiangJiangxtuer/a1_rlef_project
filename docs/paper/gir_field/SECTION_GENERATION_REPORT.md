# GIR-Field paper section generation report

Generated from formal artifacts:

- `docs/GIR_FIELD_FORMAL_PROTOCOL_REPORT.md`
- `results/girfield_formal/N5_paper/girfield_main_table.md`
- `results/girfield_formal/N1_statistics/statistical_tests.csv`
- `results/girfield_formal/N4_risk/risk_calibration_summary.csv`

## Generated files

- Chinese argument draft: `docs/paper/gir_field/METHOD_AND_EXPERIMENTS_ZH.md`
- English LaTeX method section: `docs/paper/gir_field/sections/method.tex`
- English LaTeX experiments section: `docs/paper/gir_field/sections/experiments.tex`
- Standalone LaTeX wrapper: `docs/paper/gir_field/gir_field_method_experiments_draft.tex`

## Evidence used

- Full local evaluation: UEFB-v2 test 500, LOL-v2-real test 100, LOL-v2-synthetic test 100.
- 9 internal variants, 6300 per-image records.
- 5000 bootstrap samples, paired Wilcoxon signed-rank tests, Benjamini-Hochberg FDR.
- Key formal statistics:
  - M4J_ES vs M4J UEFB S_corr: `+0.1581`, 95% CI `[0.1369, 0.1791]`, FDR q `2.18e-43`.
  - M4J_ES vs M4J Real PSNR: `+0.6963 dB`, 95% CI `[0.4141, 0.9664]`, FDR q `1.57e-05`.
  - M4J_ES vs M4J Synthetic PSNR: `+0.6321 dB`, 95% CI `[0.4182, 0.8559]`, FDR q `6.04e-08`.
  - M4J_ES vs M4 UEFB S_corr: `+0.5792`, 95% CI `[0.5361, 0.6230]`, FDR q `3.52e-64`.
  - Risk probe AUC about `0.766`, ECE about `0.281`.

## Claim boundaries preserved

Allowed:

- Centered/gauge-invariant exposure-shape calibration improves gauge-free exposure-field identifiability.
- UEFB-G exposes PSNR-only misranking.
- Global gauge and local shape errors can be separated by S/Gauge metrics.
- Recoverability risk has diagnostic signal but remains calibration-limited.

Not claimed:

- Retinexformer outperformance.
- DGB-RLEF SOTA or tri-domain resolution.
- Risk head solves enhancement.
- E/S metrics apply to black-box baselines.

## Validation

Generated artifact sizes/line counts:

- `METHOD_AND_EXPERIMENTS_ZH.md`: 164 lines, 10258 bytes.
- `sections/method.tex`: 53 lines, 5079 bytes.
- `sections/experiments.tex`: 79 lines, 6506 bytes.
- `gir_field_method_experiments_draft.tex`: 22 lines, 876 bytes.
- `SECTION_GENERATION_REPORT.md`: 48 lines before this validation patch.

Checks passed:

- standalone wrapper includes `sections/method` and `sections/experiments`.
- LaTeX `equation` and `table` environments are balanced.
- brace counts are balanced in generated TeX files.
- required evidence strings are present: `+0.1581`, `2.18e-43`, `Retinexformer`, `不能写`, `Gauge_MAE`.

## Compilation note

`pdflatex` is not installed in the current environment, so the LaTeX wrapper was not PDF-compiled here. File existence, include references, and lightweight syntax checks were performed instead.
