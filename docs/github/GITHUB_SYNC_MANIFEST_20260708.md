# GitHub sync manifest — 2026-07-08

Repository:

```text
https://github.com/LiangJiangxtuer/a1_rlef_project
```

Purpose:

```text
Sync recent GIR-Field benchmark-hardening experiments after the previous upload:
- full work audit and iteration plan
- I1 UEFB-G public evaluator package
- I2 external field-aware adapter study
- compact experiment results/data/logs
- updated paper draft and analysis
```

## Included categories

### Code

```text
scripts/uefbg_eval.py
scripts/run_uefbg_benchmark_v1.py
scripts/run_uefbg_external_adapters.py
src/rlef/adapters/exposure_field_adapters.py
src/rlef/adapters/__init__.py
```

### Configs

```text
configs/uefbg/protocol_v1.yaml
```

### Tests

```text
tests/test_uefbg_evaluator_contract.py
tests/test_external_field_adapters_contract.py
```

### Experiment results/data

```text
results/uefbg_v1/internal_methods/
results/uefbg_external_adapters/
```

These are compact CSV/JSON/Markdown experiment artifacts, not raw image datasets or checkpoints.

### Logs

```text
logs/uefbg_v1_internal_methods_20260707.log
logs/uefbg_v1_full_pytest_20260707.log
logs/uefbg_external_adapters_i2_20260708.log
logs/i2_external_adapters_full_pytest_20260708.log
```

### Analysis and reports

```text
docs/GIR_FIELD_FULL_AUDIT_AND_ITERATION_PLAN_20260707.md
docs/GIR_FIELD_ITERATION_PLAN_20260707.json
docs/GIR_FIELD_RECENT_EXPERIMENTS_AUDIT_AND_SYNC_SUMMARY_20260708.md
docs/UEFB_G_DATASET_CARD.md
docs/UEFB_G_EVALUATOR_SPEC.md
docs/UEFB_G_BASELINE_SUBMISSION_FORMAT.md
docs/UEFB_G_V1_PUBLIC_EVALUATOR_REPORT.md
docs/EXTERNAL_FIELD_ADAPTER_PROTOCOL.md
docs/I2_EXTERNAL_FIELD_ADAPTER_STUDY_REPORT_20260708.md
```

### Paper update

```text
docs/paper/gir_field/sections/experiments.tex
docs/paper/gir_field/tables/external_adapter_lowpass_table.tex
docs/paper/gir_field/build_i2_ieee/gir_field_ieee_conference.pdf
docs/paper/gir_field/build_i2_ieee/tectonic_i2_ieee_compile.log
```

## Excluded / still local-only

```text
data/
experiments/
external_baselines/
tools/tectonic/
*.pth, *.pt, *.ckpt
*.zip, *.tar, *.tar.gz
```

No credentials or token files are included. `/home/token` is used only for the authenticated push when needed and must not be committed or printed.

## Verification before sync

Latest completed verification before this manifest:

```text
I2 full repository regression: 133 passed in 24.07s
I2 TeX compile: PASS, PDF generated, no fatal errors
UEFB-G public evaluator validation: PASS
External adapter UEFB-G validation: PASS
```

Final commit/push verification will be recorded in `docs/github/PUSH_STATUS_20260708.md` after the remote push.
