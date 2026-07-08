# I2 External Field-Aware Adapter Study — Execution Report

Date: 2026-07-08

## 1. Audit-plan target

This executes I2 from the full audit iteration plan:

```text
External field-aware adapter study
```

Original motivation:

```text
If UEFB-G only evaluates GIR-Field/RLEF internal fields, reviewers may treat it
as a self-serving metric. I2 tests whether external black-box enhancement outputs
can be mapped into a defensible exposure-field proxy and evaluated through the
same gauge/shape decomposition.
```

## 2. Implementation artifacts

```text
src/rlef/adapters/exposure_field_adapters.py
src/rlef/adapters/__init__.py
scripts/run_uefbg_external_adapters.py
docs/EXTERNAL_FIELD_ADAPTER_PROTOCOL.md
tests/test_external_field_adapters_contract.py
```

Primary output directory:

```text
results/uefbg_external_adapters/
```

Generated files:

```text
results/uefbg_external_adapters/external_adapter_per_image.csv
results/uefbg_external_adapters/external_adapter_summary.csv
results/uefbg_external_adapters/external_adapter_summary.json
results/uefbg_external_adapters/EXTERNAL_FIELD_ADAPTER_STUDY_REPORT.md
results/uefbg_external_adapters/uefbg_v1_bundle/
```

Execution log:

```text
logs/uefbg_external_adapters_i2_20260708.log
```

## 3. Adapter definition

For low-light input `L`, external method prediction `P`, and paired normal-light reference `H`:

```text
E_hat = log(Y_P + eps) - log(Y_L + eps)
E_gt  = log(Y_H + eps) - log(Y_L + eps)
```

where `Y` is luminance and `eps=1e-3`.

Two adapter variants were executed:

```text
log_ratio_raw
log_ratio_lowpass_r8
```

`log_ratio_lowpass_r8` applies a dependency-light separable box low-pass with radius 8 px to both predicted and target log-ratio fields.

## 4. Run result

Command:

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_uefbg_external_adapters.py \
  --out results/uefbg_external_adapters
```

Result:

```text
DONE: true
n_rows: 1200
n_summary_rows: 12
validation_status: PASS
```

UEFB-G v1 validation:

```text
status: PASS
n_rows: 1200
n_variants: 6
n_datasets: 2
output_only_rows: 0
field_aware_rows: 1200
partial_field_rows: 0
```

## 5. Main quantitative findings

| Method | Dataset | Adapter | PSNR↑ | S_corr↑ | S_MAE↓ | Gauge_MAE↓ | E_MAE↓ |
|---|---|---|---:|---:|---:|---:|---:|
| Retinexformer | real | lowpass r8 | 22.794 | 0.955 | 0.059 | 0.183 | 0.190 |
| Retinexformer | synthetic | lowpass r8 | 25.669 | 0.989 | 0.066 | 0.127 | 0.136 |
| Zero-DCE++ | real | lowpass r8 | 18.491 | 0.659 | 0.174 | 0.486 | 0.497 |
| Zero-DCE++ | synthetic | lowpass r8 | 17.576 | 0.800 | 0.291 | 0.481 | 0.498 |
| KinD++ | real | lowpass r8 | 22.211 | 0.944 | 0.083 | 0.116 | 0.152 |
| KinD++ | synthetic | lowpass r8 | 19.259 | 0.923 | 0.168 | 0.194 | 0.246 |

Raw adapter results are also stored in `external_adapter_summary.csv`. Low-pass consistently improves shape correlation, especially for Zero-DCE++ real:

```text
Zero-DCE++ real S_corr: raw 0.398 -> lowpass 0.659
Zero-DCE++ synthetic S_corr: raw 0.728 -> lowpass 0.800
```

## 6. Interpretation

### What I2 supports

I2 supports a guarded expansion of UEFB-G beyond internal RLEF fields:

```text
UEFB-G can evaluate adapter-derived exposure-field proxies for external
enhancement outputs when paired low/high references are available.
```

The strongest blind external result is Retinexformer:

```text
Retinexformer lowpass S_corr:
real      0.955
synthetic 0.989
```

Zero-DCE++ is weaker but still nontrivial under low-pass:

```text
Zero-DCE++ lowpass S_corr:
real      0.659
synthetic 0.800
```

Thus the I2 go/no-go criterion is satisfied for blind external methods:

```text
GO: Retinexformer and Zero-DCE++ both produce nontrivial low-pass adapter-field shape signal.
```

### What I2 does not support

I2 does **not** support saying:

```text
Retinexformer internally predicts an exposure field.
Zero-DCE++ internally predicts an exposure field.
UEFB-G proves black-box methods have physically correct latent fields.
```

Correct language:

```text
adapter-derived luminance-gain field
method-induced exposure proxy
external field-aware adapter analysis
```

### KinD++ caveat

KinD++ has strong adapter metrics, but the P4 report established that the executed official KinD++ script is high-assisted/non-blind under this evaluation path. Therefore KinD++ should be kept as an appendix/caveat row, not counted as blind external support.

## 7. Paper-level implication

Before I2, the strongest reviewer attack was:

```text
UEFB-G only works for your own method's internal field.
```

After I2, the safer response is:

```text
UEFB-G has two modes:
1. explicit-field evaluation for methods that expose an internal field;
2. adapter-field evaluation for black-box methods via log-luminance-gain proxies.
```

This makes the benchmark contribution substantially more defensible, provided the main paper clearly distinguishes explicit internal fields from adapter-derived proxies.

## 8. Recommended manuscript update

Add a subsection:

```text
External Field-Aware Adapter Study
```

Core claims:

```text
- Retinexformer and Zero-DCE++ outputs can be evaluated under UEFB-G via a log-luminance-gain adapter.
- Retinexformer has high adapter-field shape agreement, matching its strong PSNR/SSIM.
- Zero-DCE++ has moderate shape agreement but large gauge error, explaining lower fidelity and calibration quality.
- Low-pass adapter improves field-shape agreement, suggesting illumination-scale rather than texture-scale comparison is the appropriate external adapter level.
```

Do not promote this as a new method. It is benchmark/protocol evidence.

## 9. Verification

Specific I2 tests:

```text
tests/test_external_field_adapters_contract.py
4 passed
```

Required full-regression command remains:

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m pytest tests -q
```
