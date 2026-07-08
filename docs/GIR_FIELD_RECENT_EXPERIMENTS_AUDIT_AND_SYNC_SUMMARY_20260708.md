# GIR-Field recent experiments audit, summary, and GitHub sync plan

Date: 2026-07-08 09:57:44 CST  
Repository: `/home/user/a1_rlef_project`  
Scope: the recent audit/iteration work after the GitHub upload, especially I1 UEFB-G public evaluator and I2 external field-aware adapter study.

## 1. Executive summary

The recent work meaningfully improves GIR-Field's top-tier readiness. Before these steps, the strongest weakness was that UEFB-G looked like a local/internal protocol for the author's own method. After I1 and I2, the project now has:

```text
1. A standalone UEFB-G v1 public evaluator interface.
2. Explicit output_only vs field_aware reporting modes.
3. Partial-field rejection to avoid overclaiming incomplete physical metrics.
4. A public-style rerun of all internal formal evidence with exact agreement to the previous formal summary.
5. An external adapter study showing that black-box baselines can be evaluated through a clearly bounded log-luminance-gain proxy.
6. Updated paper text/table and a freshly compiled IEEE-style draft.
```

The correct claim calibration is now:

```text
GIR-Field is not SOTA LLIE.
GIR-Field is a gauge-identifiable exposure-field learning + benchmark/evaluation + failure-audit contribution.
UEFB-G supports explicit internal-field evaluation and adapter-derived black-box field analysis under a strict proxy boundary.
```

## 2. Code audit

### 2.1 New evaluator code

Implemented files:

```text
scripts/uefbg_eval.py
scripts/run_uefbg_benchmark_v1.py
configs/uefbg/protocol_v1.yaml
tests/test_uefbg_evaluator_contract.py
```

Audit verdict:

```text
PASS
```

Reasons:

- The evaluator is model-independent and does not import RLEF training/model internals.
- It has a simple CSV-in/report-bundle-out interface suitable for public benchmark use.
- It supports black-box `output_only` submissions and `field_aware` submissions.
- It rejects partial field metric rows, preventing unsupported physical claims.
- It reproduces the original formal summary exactly in metric-table mode.

Known limitation:

- v1 consumes per-image metrics rather than raw image folders for arbitrary users. This is acceptable for the current benchmark-hardening stage but should eventually be extended to direct image/field-folder evaluation.

### 2.2 New external adapter code

Implemented files:

```text
src/rlef/adapters/exposure_field_adapters.py
src/rlef/adapters/__init__.py
scripts/run_uefbg_external_adapters.py
tests/test_external_field_adapters_contract.py
```

Audit verdict:

```text
PASS, with claim-boundary caveat
```

Reasons:

- The adapter definition is physically interpretable as method-induced luminance gain:

```text
E_hat = log(Y_pred + eps) - log(Y_low + eps)
E_gt  = log(Y_high + eps) - log(Y_low + eps)
```

- It computes the same gauge decomposition metrics as UEFB-G:

```text
E_MAE, S_MAE, Gauge_MAE, S_corr
```

- It reports both raw and low-pass variants.
- It keeps KinD++ caveated as high-assisted under the executed official script.

Critical caveat:

```text
These are adapter-derived proxy fields. They are not evidence that Retinexformer, Zero-DCE++, or KinD++ internally predicts the same field.
```

## 3. Experiment result audit

### 3.1 I1: UEFB-G v1 public evaluator

Primary outputs:

```text
results/uefbg_v1/internal_methods/input_metrics.csv
results/uefbg_v1/internal_methods/summary.csv
results/uefbg_v1/internal_methods/summary.json
results/uefbg_v1/internal_methods/validation.json
results/uefbg_v1/internal_methods/report_cards.md
results/uefbg_v1/internal_methods/manifest.json
results/uefbg_v1/internal_methods/formal_match_validation.json
```

Execution summary:

```text
DONE: true
validation_status: PASS
n_rows: 6300
n_variants: 9
n_datasets: 3
```

Formal equivalence check:

```text
formal source: results/girfield_formal/N1_statistics/variant_dataset_summary.csv
public output: results/uefbg_v1/internal_methods/summary.csv
formal keys: 27
new keys: 27
max_abs_diff: 0.0
status: PASS
```

Audit verdict:

```text
PASS
```

I1 successfully converts UEFB-G from a local protocol artifact into a public-facing evaluator/report-bundle interface.

### 3.2 I2: external field-aware adapter study

Primary outputs:

```text
results/uefbg_external_adapters/external_adapter_per_image.csv
results/uefbg_external_adapters/external_adapter_summary.csv
results/uefbg_external_adapters/external_adapter_summary.json
results/uefbg_external_adapters/EXTERNAL_FIELD_ADAPTER_STUDY_REPORT.md
results/uefbg_external_adapters/uefbg_v1_bundle/
```

Execution summary:

```text
DONE: true
n_rows: 1200
n_summary_rows: 12
validation_status: PASS
```

UEFB-G validation:

```text
status: PASS
n_rows: 1200
n_variants: 6
n_datasets: 2
output_only_rows: 0
field_aware_rows: 1200
partial_field_rows: 0
```

Low-pass adapter summary:

| Method | Dataset | PSNR↑ | S_corr↑ | S_MAE↓ | Gauge_MAE↓ | E_MAE↓ |
|---|---:|---:|---:|---:|---:|---:|
| Retinexformer | real | 22.794 | 0.955 | 0.059 | 0.183 | 0.190 |
| Retinexformer | synthetic | 25.669 | 0.989 | 0.066 | 0.127 | 0.136 |
| Zero-DCE++ | real | 18.491 | 0.659 | 0.174 | 0.486 | 0.497 |
| Zero-DCE++ | synthetic | 17.576 | 0.800 | 0.291 | 0.481 | 0.498 |
| KinD++ | real | 22.211 | 0.944 | 0.083 | 0.116 | 0.152 |
| KinD++ | synthetic | 19.259 | 0.923 | 0.168 | 0.194 | 0.246 |

Go/no-go verdict:

```text
GO, guarded.
```

Reason:

```text
Retinexformer and Zero-DCE++ are blind external methods and both have nontrivial low-pass adapter S_corr.
```

KinD++ remains caveated because the executed official path is high-assisted.

## 4. Experiment data audit

The synced experiment data is compact and GitHub-safe:

```text
results/uefbg_v1/internal_methods/*.csv/*.json/*.md
results/uefbg_external_adapters/*.csv/*.json/*.md
logs/uefbg_v1_*.log
logs/uefbg_external_adapters_i2_20260708.log
logs/i2_external_adapters_full_pytest_20260708.log
```

Raw datasets and large local artifacts remain excluded by `.gitignore`:

```text
data/
experiments/
external_baselines/
tools/tectonic/
*.pth, *.pt, *.ckpt
*.zip, *.tar, *.tar.gz
```

Audit verdict:

```text
PASS
```

This preserves reproducible experimental evidence without uploading raw LOL/UEFB datasets, checkpoints, third-party repos, or compiler toolchains.

## 5. Experiment analysis audit

New analysis/report files:

```text
docs/GIR_FIELD_FULL_AUDIT_AND_ITERATION_PLAN_20260707.md
docs/GIR_FIELD_ITERATION_PLAN_20260707.json
docs/UEFB_G_V1_PUBLIC_EVALUATOR_REPORT.md
docs/UEFB_G_DATASET_CARD.md
docs/UEFB_G_EVALUATOR_SPEC.md
docs/UEFB_G_BASELINE_SUBMISSION_FORMAT.md
docs/EXTERNAL_FIELD_ADAPTER_PROTOCOL.md
docs/I2_EXTERNAL_FIELD_ADAPTER_STUDY_REPORT_20260708.md
```

Audit verdict:

```text
PASS
```

The analysis correctly separates:

```text
- explicit-field evaluation
- adapter-derived black-box proxy evaluation
- output-only baseline evaluation
- unsupported SOTA/restoration claims
- high-assisted KinD++ caveat
```

## 6. Paper audit

Updated manuscript files:

```text
docs/paper/gir_field/sections/experiments.tex
docs/paper/gir_field/tables/external_adapter_lowpass_table.tex
docs/paper/gir_field/build_i2_ieee/gir_field_ieee_conference.pdf
```

Compiled PDF:

```text
pages: 8
bytes: 320168
sha256: 9bead10603d6644e7f2b578942a849dd9a114208da25f07abe934c8c5784c9dd
```

TeX status:

```text
compile: PASS
fatal errors: none
remaining warnings: minor underfull/overfull text-layout warnings only
```

Audit verdict:

```text
PASS
```

The new paper subsection is correctly scoped as:

```text
External field-aware adapter study
```

not as a new restoration method or black-box internal-field proof.

## 7. Verification status

Recent executed tests:

```text
UEFB-G evaluator tests: 5 passed
I2 external adapter tests: 4 passed
Full repository regression after I2: 133 passed in 24.07s
```

Full test log:

```text
logs/i2_external_adapters_full_pytest_20260708.log
```

Additional validation files:

```text
results/uefbg_v1/internal_methods/formal_match_validation.json
results/uefbg_external_adapters/uefbg_v1_bundle/validation.json
```

## 8. Claim boundary after these experiments

Safe claims:

```text
1. UEFB-G now has a standalone public evaluator/report-bundle interface.
2. UEFB-G reproduces the original formal summary exactly in public-evaluator metric-table mode.
3. External black-box outputs can be analyzed through a log-luminance-gain adapter when paired references are available.
4. Retinexformer shows strong adapter-field shape agreement; Zero-DCE++ shows moderate shape agreement but large gauge error.
5. GIR-Field remains a benchmark/evaluation/identifiability contribution, not SOTA LLIE.
```

Unsafe claims:

```text
1. GIR-Field is SOTA low-light enhancement.
2. GIR-Field outperforms Retinexformer.
3. Retinexformer/Zero-DCE++ internally predicts the exact UEFB-G exposure field.
4. KinD++ is a blind baseline under the executed high-assisted official script.
5. Recoverability risk is solved or calibrated.
```

## 9. GitHub sync inclusion plan

Include:

```text
source code: scripts/, src/rlef/adapters/
configs: configs/uefbg/protocol_v1.yaml
tests: tests/test_uefbg_evaluator_contract.py, tests/test_external_field_adapters_contract.py
docs: audit reports, protocol docs, evaluator docs, paper update
results: compact CSV/JSON/MD bundles under results/uefbg_v1 and results/uefbg_external_adapters
logs: compact execution/test logs for I1/I2
paper: updated TeX table/section and compiled I2 IEEE PDF/log artifacts
```

Exclude/remain ignored:

```text
raw datasets
external baseline repositories
local toolchains
checkpoints/weights
large generated image dumps
archives
credentials/tokens
```

## 10. Final audit verdict before GitHub sync

```text
READY_TO_SYNC_TO_GITHUB
```

The code, experiment results, compact experiment data, and analysis are internally consistent, verified by tests and evaluator validation, and claim-calibrated for GitHub publication.
