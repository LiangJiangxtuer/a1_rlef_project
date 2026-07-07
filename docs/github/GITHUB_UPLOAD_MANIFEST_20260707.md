# GitHub upload manifest — 2026-07-07

This repository upload is intended to preserve the latest complete A1-RLEF / GIR-Field research state: code, experiment runners, compact experiment outputs, analysis reports, and final paper artifacts.

## Included categories

```text
src/rlef/                         # research implementation
scripts/                          # experiment runners and paper pipelines
configs/                          # configs for DGB/P2/P3/P4/P5/P6/P7/GIR-Field experiments
tests/                            # contract tests and runner tests
docs/                             # phase reports, analysis reports, paper reports
results/tables/                   # compact phase metric tables
results/girfield_formal/          # full formal protocol outputs
results/girfield_relaxed/         # relaxed protocol outputs
results/paper_pipeline/           # paper pipeline artifacts
results/hermes_audit/             # audit artifacts
logs/                             # compact reproducibility logs
```

## Final paper files

```text
docs/paper/gir_field/gir_field_ieee_conference.tex
docs/paper/gir_field/build_ieee_v3/gir_field_ieee_conference.pdf
docs/paper/gir_field/gir_field_full_paper_skeleton.tex
docs/paper/gir_field/build_article_v3/gir_field_full_paper_skeleton.pdf
docs/paper/gir_field/FINAL_TEX_TEMPLATE_AND_REFRESH_REPORT.md
docs/paper/gir_field/FINAL_PACKAGE_VALIDATION.json
docs/paper/gir_field/LITERATURE_REFRESH_2024_2026.md
docs/paper/gir_field/LITERATURE_REFRESH_VERIFICATION_LEDGER.md
```

## Excluded/local-only categories

These are intentionally excluded for GitHub safety and reproducibility hygiene:

```text
data/                             # datasets
experiments/                      # checkpoints/run directories
results/visuals/                  # generated image dumps
results/paired/                   # generated paired images
results/uefb/                     # generated UEFB images
results/unpaired/                 # generated unpaired images
external_baselines/               # third-party baseline checkouts/venvs/pretrained files
tools/tectonic/                   # local compiler binary/toolchain
*.pth, *.pt, *.ckpt               # model checkpoints
*.tar.gz, *.zip                   # archives/downloads
```

Official-baseline evidence is still preserved via compact logs and tables under `logs/`, `results/tables/`, and `docs/P4_OFFICIAL_BASELINES_REPORT.md`; the third-party repositories themselves are not vendored into this project.

## Authentication boundary

GitHub credentials were read from `/home/token` only for the explicit user-requested upload. The token file is outside this repository and is not tracked.

## Claim boundary

The upload preserves the final claim boundary:

```text
GIR-Field is a gauge-identifiability / benchmark / failure-audit contribution.
It is not uploaded as a SOTA LLIE claim.
DGB/P2F remains stopped.
Recoverability risk remains diagnostic, not solved.
```
