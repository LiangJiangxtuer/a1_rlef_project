# UEFB-G v1 public evaluator execution report

Date: 2026-07-07

## Scope

This implements the first step from `GIR_FIELD_FULL_AUDIT_AND_ITERATION_PLAN_20260707.md`:

```text
I1. UEFB-G Benchmark v1 public package
Immediate next step: Build UEFB-G public evaluator v1 and rerun current internal methods through it.
```

## Implemented artifacts

```text
scripts/uefbg_eval.py
scripts/run_uefbg_benchmark_v1.py
configs/uefbg/protocol_v1.yaml
docs/UEFB_G_DATASET_CARD.md
docs/UEFB_G_EVALUATOR_SPEC.md
docs/UEFB_G_BASELINE_SUBMISSION_FORMAT.md
tests/test_uefbg_evaluator_contract.py
```

## Public evaluator properties

The evaluator is intentionally independent from RLEF training/model internals. It accepts public-style per-image metric submissions with:

```text
required identity columns: variant_id,dataset,name
output metrics: psnr,ssim,lee,nai,identity_drop,q_ece,over,under
optional full field metric quartet: E_MAE,S_MAE,Gauge_MAE,S_corr
```

Reporting modes:

```text
output_only  : valid for black-box methods; field metrics are N/A
field_aware  : valid when the full field metric quartet is present
partial_field: rejected, because it cannot separate gauge and shape failures
```

## Internal formal evidence rerun

Command executed:

```bash
python scripts/run_uefbg_benchmark_v1.py \
  --input-metrics results/girfield_formal/N1_statistics/per_image_metrics.csv \
  --protocol configs/uefbg/protocol_v1.yaml \
  --out results/uefbg_v1/internal_methods
```

Result:

```text
DONE: true
validation_status: PASS
n_rows: 6300
n_variants: 9
n_datasets: 3
```

Generated bundle:

```text
results/uefbg_v1/internal_methods/input_metrics.csv
results/uefbg_v1/internal_methods/summary.csv
results/uefbg_v1/internal_methods/summary.json
results/uefbg_v1/internal_methods/validation.json
results/uefbg_v1/internal_methods/report_cards.md
results/uefbg_v1/internal_methods/manifest.json
results/uefbg_v1/internal_methods/formal_match_validation.json
```

## Formal-protocol match check

The public evaluator summary was compared against the existing formal protocol summary:

```text
formal source: results/girfield_formal/N1_statistics/variant_dataset_summary.csv
public output: results/uefbg_v1/internal_methods/summary.csv
variant-dataset keys: 27 vs 27
max_abs_diff across checked metrics: 0.0
status: PASS
```

This means the public evaluator reproduces the existing formal summary metrics exactly for the current metric-table mode.

## Contract tests

New tests:

```text
tests/test_uefbg_evaluator_contract.py
```

They cover:

```text
- gauge metric separation for global shift vs local shape distortion
- black-box output-only row acceptance
- partial field metric rejection
- variant/dataset summary aggregation
- report bundle generation
- public CLI execution on the formal internal table
```

Specific UEFB-G evaluator test result:

```text
5 passed in 0.57s
```

Full repository regression result:

```text
129 passed in 21.08s
log: logs/uefbg_v1_full_pytest_20260707.log
```

## Claim boundary

This is benchmark/evaluator hardening only. It does not add new training and does not revive DGB/P2F. It improves the top-tier readiness of UEFB-G by making the current formal evidence consumable through a standalone public-facing evaluator interface.

## Next recommended step

Proceed to I2:

```text
External field-aware adapter study
```

Minimum next deliverables:

```text
src/rlef/adapters/exposure_field_adapters.py
scripts/run_uefbg_external_adapters.py
docs/EXTERNAL_FIELD_ADAPTER_PROTOCOL.md
results/uefbg_external_adapters/
```

Go/no-go:

```text
GO if at least two external/semiexternal methods can produce defensible field proxies with nontrivial S/Gauge behavior.
NO-GO if adapters are too arbitrary; then UEFB-G should be explicitly scoped to explicit-field methods only.
```
