# GitHub push status — 2026-07-08

Repository:

```text
https://github.com/LiangJiangxtuer/a1_rlef_project
```

Purpose:

```text
Sync recent GIR-Field audit, UEFB-G public evaluator, I2 external field-aware adapter study, compact experiment results/data, analysis reports, and updated paper draft.
```

This file is committed before the final push, so the exact final remote HEAD should be verified with:

```bash
git ls-remote origin refs/heads/main
```

The final response in the Hermes session reports the exact pushed SHA after verification.

## Local verification before push

```text
Full repository regression: 133 passed in 23.09s
JSON validation: PASS
I2 IEEE TeX compile: PASS
Staged hard-secret scan: PASS
Large-file scan: no >10MB GitHub-unsafe staged files outside ignored data/experiment areas
```

## Key synced artifacts

```text
docs/GIR_FIELD_RECENT_EXPERIMENTS_AUDIT_AND_SYNC_SUMMARY_20260708.md
docs/GIR_FIELD_FULL_AUDIT_AND_ITERATION_PLAN_20260707.md
docs/UEFB_G_V1_PUBLIC_EVALUATOR_REPORT.md
docs/I2_EXTERNAL_FIELD_ADAPTER_STUDY_REPORT_20260708.md
scripts/uefbg_eval.py
scripts/run_uefbg_benchmark_v1.py
scripts/run_uefbg_external_adapters.py
src/rlef/adapters/exposure_field_adapters.py
results/uefbg_v1/internal_methods/
results/uefbg_external_adapters/
docs/paper/gir_field/build_i2_ieee/gir_field_ieee_conference.pdf
```

## Credential boundary

No credentials are committed. `/home/token` is outside the repository and is used only for the explicit authenticated GitHub push when needed.
