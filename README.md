# A1-RLEF v2

Recoverability/Gauge-aware Local Exposure Field research scaffold for low-light and uneven-exposure enhancement.

Main execution plan:

```text
docs/EXECUTION_PLAN_RLEF_V2_ZH.md
```

Current executed evidence:

```text
docs/EXPERIMENT_PROCESS_REPORT.md        # P0/P1-light scaffold evidence
docs/P1_FORMAL_1000_REPORT.md            # P1 formal 1000-step full-train/full-test evidence
results/tables/p1_formal_1000_summary.csv
```

P1 formal conclusion: fixed `e_mean=0.02` is not a main positive contribution in the new scaffold; adaptive gauge is the better candidate and fixed anchor is retained as a fragile baseline. No SOTA/Retinexformer-level or physical E-field correctness claim is made yet.

This repo is intentionally GitHub-safe: source code, configs, tests, docs, compact metric tables, and compact run logs are tracked; datasets/checkpoints/generated images are local-only.
