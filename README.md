# A1-RLEF / GIR-Field

Gauge-identifiable exposure-field learning for uneven low-light enhancement.

This repository contains the latest code, experiment runners, compact experiment evidence, analysis reports, and paper draft artifacts for the A1-RLEF → GIR-Field research line.

## Current research position

The final claim-calibrated direction is:

```text
GIR-Field = gauge-identifiable exposure-field learning + UEFB-G benchmark/evaluation + failure-audit evidence
```

The repository does **not** claim SOTA low-light image enhancement and does **not** revive the stopped DGB/P2F route. Retinexformer remains stronger on RGB fidelity; GIR-Field is positioned as a mechanism/evaluation contribution for exposure-field identifiability.

## Latest paper artifacts

Compiled drafts:

```text
docs/paper/gir_field/build_i2_ieee/gir_field_ieee_conference.pdf      # latest IEEE-style draft with I2 external adapter study, 8 pages
docs/paper/gir_field/build_ieee_v3/gir_field_ieee_conference.pdf      # previous IEEE-style conference draft, 8 pages
docs/paper/gir_field/build_article_v3/gir_field_full_paper_skeleton.pdf # article-style draft, 12 pages
```

Main TeX sources:

```text
docs/paper/gir_field/gir_field_ieee_conference.tex
docs/paper/gir_field/gir_field_full_paper_skeleton.tex
docs/paper/gir_field/sections/
docs/paper/gir_field/references.bib
```

Final package report and validation:

```text
docs/paper/gir_field/FINAL_TEX_TEMPLATE_AND_REFRESH_REPORT.md
docs/paper/gir_field/FINAL_PACKAGE_VALIDATION.json
```

## Key formal evidence

Formal full frozen-evidence protocol:

```text
docs/GIR_FIELD_FORMAL_PROTOCOL_PLAN_ZH.md
docs/GIR_FIELD_FORMAL_PROTOCOL_REPORT.md
results/girfield_formal/
scripts/run_girfield_formal_protocol.py
tests/test_girfield_formal_protocol_contract.py
```

Core formal results:

```text
M4J_ES vs M4J UEFB S_corr: +0.1581, 95% CI [0.1369, 0.1791], FDR q=2.18e-43
M4J_ES vs M4J Real PSNR: +0.6963 dB, 95% CI [0.4141, 0.9664], FDR q=1.57e-05
M4J_ES vs M4J Synthetic PSNR: +0.6321 dB, 95% CI [0.4182, 0.8559], FDR q=6.04e-08
M4J_ES vs M4 UEFB S_corr: +0.5792, 95% CI [0.5361, 0.6230], FDR q=3.52e-64
```

Validation summary:

```text
formal validation: PASS
focused/full repository tests from the final protocol stage: PASS
paper package validation: PASS
```

## Latest benchmark-hardening experiments

Recent audit/iteration artifacts:

```text
docs/GIR_FIELD_FULL_AUDIT_AND_ITERATION_PLAN_20260707.md
docs/GIR_FIELD_RECENT_EXPERIMENTS_AUDIT_AND_SYNC_SUMMARY_20260708.md
docs/UEFB_G_V1_PUBLIC_EVALUATOR_REPORT.md
docs/I2_EXTERNAL_FIELD_ADAPTER_STUDY_REPORT_20260708.md
```

I1 UEFB-G public evaluator:

```text
scripts/uefbg_eval.py
scripts/run_uefbg_benchmark_v1.py
configs/uefbg/protocol_v1.yaml
results/uefbg_v1/internal_methods/
```

I2 external field-aware adapter study:

```text
src/rlef/adapters/exposure_field_adapters.py
scripts/run_uefbg_external_adapters.py
docs/EXTERNAL_FIELD_ADAPTER_PROTOCOL.md
results/uefbg_external_adapters/
```

Key I2 low-pass adapter result:

```text
Retinexformer S_corr: real=0.9547, synthetic=0.9893
Zero-DCE++ S_corr:   real=0.6590, synthetic=0.7998
Full repository regression after I2: 133 passed in 24.07s
```

## Important directories

```text
src/rlef/                         # model, loss, gauge/exposure-field components
scripts/                          # experiment runners and paper/evidence pipelines
configs/                          # experiment configs by phase
results/tables/                   # compact cross-phase metric summaries
results/girfield_formal/          # formal protocol outputs
results/girfield_relaxed/         # relaxed protocol outputs
docs/                             # phase reports and final paper reports
docs/paper/gir_field/             # final paper TeX/PDF/citation/appendix package
tests/                            # contract and runner tests
logs/                             # compact run logs for reproducibility
```

## GitHub-safe boundary

Tracked: source code, configs, tests, docs, compact tables, selected logs, final paper PDFs, and validation artifacts.

Ignored/local-only: datasets, checkpoints, generated image dumps, third-party baseline checkouts, local virtual environments, local compiler toolchains, and tarballs.

See also:

```text
docs/github/GITHUB_UPLOAD_MANIFEST_20260707.md
```
