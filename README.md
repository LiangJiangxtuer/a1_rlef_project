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
docs/paper/gir_field/build_ieee_v3/gir_field_ieee_conference.pdf      # IEEE-style conference draft, 8 pages
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
