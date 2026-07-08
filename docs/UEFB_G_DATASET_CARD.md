# UEFB-G Dataset / Protocol Card

Version: v1  
Date: 2026-07-07

## Purpose

UEFB-G is a gauge-controlled evaluation protocol for low-light / uneven-exposure enhancement methods that claim, expose, or can defensibly adapt an internal exposure/illumination field.

It is designed to separate three questions that are often conflated:

```text
1. Does the enhanced RGB image improve?                  output-only metrics
2. Is the predicted exposure-field global gauge correct? Gauge_MAE / E_MAE
3. Is the local exposure-field spatial shape correct?    S_MAE / S_corr
```

## Dataset/protocol scope

Current v1 is a protocol wrapper around the local GIR-Field formal evidence set:

```text
UEFB-v2 test: 500 images
LOL-v2 real test: 100 images
LOL-v2 synthetic test: 100 images
```

The public evaluator in `scripts/uefbg_eval.py` is model-independent and consumes per-image metric submissions. It does not import RLEF model internals.

## Reporting modes

### output_only

For black-box methods that provide enhanced images but no exposure field. Valid metrics include:

```text
psnr, ssim, lee, nai, identity_drop, q_ece, over, under
```

Internal-field metrics are reported as `N/A`.

### field_aware

For methods with explicit or defensibly adapted exposure/illumination fields. Required field metric quartet:

```text
E_MAE, S_MAE, Gauge_MAE, S_corr
```

Partial field reporting is rejected because it mixes physical claims without enough evidence to separate global-gauge and local-shape failures.

## Gauge decomposition

The exposure field is decomposed as:

```text
E = S + μ
S = E - mean(E)
μ = mean(E)
```

A global additive shift should change `E_MAE` and `Gauge_MAE`, but should not change centered shape metrics. A local zero-mean shape distortion should change `S_MAE` and `S_corr`, while leaving `Gauge_MAE` approximately unchanged.

## Claim boundary

UEFB-G does not claim that a method is SOTA LLIE. It evaluates whether an output/fidelity claim and an exposure-field physical claim agree or conflict.
