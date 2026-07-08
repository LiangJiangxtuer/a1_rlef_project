# External Field-Aware Adapter Protocol

Version: I2-v1  
Date: 2026-07-08

## Purpose

This protocol tests whether UEFB-G can evaluate methods beyond the internal GIR-Field/RLEF family by converting frozen black-box enhancement outputs into a defensible exposure-field proxy.

It does **not** claim that a black-box method internally predicts this field. The claim is weaker and safer:

```text
Given low-light input L and enhanced output P, the method-induced luminance gain
can be interpreted as an adapter-derived exposure proxy and compared with the
paired reference-induced exposure proxy.
```

## Adapter definition

Let `Y(.)` be luminance. For each image:

```text
E_hat = log(Y_pred + eps) - log(Y_low + eps)
E_gt  = log(Y_high + eps) - log(Y_low + eps)
```

where:

```text
Y_low  = luminance of the low-light input
Y_pred = luminance of the external method output
Y_high = luminance of the paired normal-light reference
eps    = 1e-3 for dark-pixel numerical stability
```

UEFB-G field metrics are then computed on `(E_hat, E_gt)`:

```text
E_MAE     = mean(abs(E_hat - E_gt))
Gauge_MAE = abs(mean(E_hat) - mean(E_gt))
S_MAE     = mean(abs((E_hat - mean(E_hat)) - (E_gt - mean(E_gt))))
S_corr    = corr(E_hat - mean(E_hat), E_gt - mean(E_gt))
```

## Adapter variants

```text
log_ratio_raw
```

Raw per-pixel log-luminance gain. This is sensitive to texture/detail differences but preserves the direct transformation made by the method.

```text
log_ratio_lowpass_r8
```

A dependency-light separable box low-pass with radius 8 px applied to both `E_hat` and `E_gt`. This is intended as an illumination-scale proxy rather than a texture-level proxy.

## Methods and data

Frozen P4 official baseline outputs are reused:

```text
Retinexformer: official blind outputs, LOL-v2 real/synthetic
Zero-DCE++: official blind outputs, LOL-v2 real/synthetic
KinD++: official-code outputs but high-assisted under the executed script
```

The study uses paired LOL-v2 real and synthetic references. It does not use UEFB-v2 external outputs because P4 official baselines were generated on LOL-v2 only.

## Reporting rule

Rows are valid UEFB-G `field_aware` submissions because all four field metrics are provided. However, all paper text must label them as:

```text
external field-aware adapter / adapter-derived proxy
```

not as:

```text
external method internal field prediction
```

## Go/no-go criterion

A broad adapter claim is allowed only if at least two blind external methods produce nontrivial shape signal under the low-pass adapter:

```text
S_corr >= 0.5 on at least one paired split, with results reported for both splits.
```

KinD++ is excluded from the blind-method count because the executed official script is high-assisted.

## Artifact locations

```text
src/rlef/adapters/exposure_field_adapters.py
scripts/run_uefbg_external_adapters.py
tests/test_external_field_adapters_contract.py
results/uefbg_external_adapters/
logs/uefbg_external_adapters_i2_20260708.log
```
