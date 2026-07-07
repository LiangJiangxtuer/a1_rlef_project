# P2C UEFB Recovery Report

Generated: 2026-07-03T05:41:09.528386+00:00
Project root: `/home/user/a1_rlef_project`

## Scope

Executed P2C from the P2B `Gauge-only` base. The goal was UEFB recovery while preserving paired-domain gains, under the same guardrails:

- No teacher distillation.
- No `rec_by_dataset`.
- No real anchor > `0.010`.
- No SOTA claim.
- No GitHub commit/push.

P2C interventions tested:

1. `P2C_DGB_GATE_FLOOR015`: image-stat gauge + original route + global restoration-weight floor `0.15`.
2. `P2C_DGB_GATE_FLOOR025`: image-stat gauge + original route + global restoration-weight floor `0.25`.
3. `P2C_DGB_ESHAPE025`: image-stat gauge + original route + lower `e_shape=0.025`.
4. `P2C_DGB_ESHAPE010`: image-stat gauge + original route + lower `e_shape=0.010`.

All use seed `3407`, 1000 training steps, P6 multiscale trunk, q branch, and joint UEFB + LOL-v2 real + LOL-v2 synthetic training.

## Commands executed

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_p2c_uefb_recovery.py   --max_steps 1000 --train_crop 128 --batch_size 8 --parallel 2 --gpu_start 0

/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_p2c_uefb_recovery.py   --max_steps 1000 --train_crop 128 --batch_size 8 --parallel 2 --gpu_start 0 --resume
```

Verification:

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m pytest tests -q
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m compileall -q src scripts
```

## Main comparison

| Metric | P3c mean | P7B_RA010 | Phase2 full DGB | P2B Gauge-only | P2C floor015 | P2C floor025 | P2C e025 | P2C e010 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UEFB PSNR | 17.915 | 18.085 | 17.754 | 17.719 | 17.695 | 17.684 | 17.655 | 17.792 |
| Real PSNR | 20.021 | 19.847 | 19.939 | 20.432 | 19.658 | 20.297 | 19.873 | 19.604 |
| Synthetic PSNR | 17.678 | 17.965 | 18.479 | 18.402 | 17.635 | 18.356 | 18.187 | 18.588 |
| UEFB E_corr | 0.436 | — | 0.528 | 0.508 | 0.482 | 0.534 | 0.488 | 0.472 |
| Real E_corr | 0.707 | — | 0.679 | 0.712 | 0.748 | 0.651 | 0.708 | 0.730 |
| Synthetic E_corr | 0.841 | — | 0.818 | 0.839 | 0.866 | 0.792 | 0.839 | 0.813 |

## Deltas vs P2B Gauge-only base

| Variant | Intervention | Δ UEFB PSNR vs P2B | Δ Real PSNR vs P2B | Δ Synthetic PSNR vs P2B | Promotion |
|---|---|---:|---:|---:|---|
| P2C_DGB_GATE_FLOOR015 | global_route_floor | -0.023 | -0.774 | -0.767 | False |
| P2C_DGB_GATE_FLOOR025 | global_route_floor | -0.035 | -0.134 | -0.046 | False |
| P2C_DGB_ESHAPE025 | reduced_e_shape_weight | -0.064 | -0.559 | -0.215 | False |
| P2C_DGB_ESHAPE010 | reduced_e_shape_weight | 0.073 | -0.828 | 0.186 | False |

## Route/gauge diagnostics

For these identity-route variants, `low fraction` is `(1-A)`, i.e. how much the output keeps the original low input instead of `I_rest`.

| Variant | Split | A_mean | low fraction | route_entropy | mu_mean | mu_min | mu_max |
|---|---|---:|---:|---:|---:|---:|---:|
| P2C_DGB_GATE_FLOOR015 | uefb | 0.457 | 0.543 | 0.659 | 1.193 | 0.837 | 1.531 |
| P2C_DGB_GATE_FLOOR015 | real | 0.479 | 0.521 | 0.660 | 1.458 | 1.195 | 1.534 |
| P2C_DGB_GATE_FLOOR015 | synthetic | 0.442 | 0.558 | 0.662 | 1.236 | 0.849 | 1.487 |
| P2C_DGB_GATE_FLOOR025 | uefb | 0.501 | 0.499 | 0.674 | 1.184 | 0.864 | 1.490 |
| P2C_DGB_GATE_FLOOR025 | real | 0.507 | 0.493 | 0.674 | 1.424 | 1.185 | 1.493 |
| P2C_DGB_GATE_FLOOR025 | synthetic | 0.485 | 0.515 | 0.677 | 1.223 | 0.874 | 1.450 |
| P2C_DGB_ESHAPE025 | uefb | 0.467 | 0.533 | 0.643 | 1.201 | 0.801 | 1.574 |
| P2C_DGB_ESHAPE025 | real | 0.494 | 0.506 | 0.650 | 1.495 | 1.205 | 1.577 |
| P2C_DGB_ESHAPE025 | synthetic | 0.437 | 0.563 | 0.638 | 1.248 | 0.814 | 1.527 |
| P2C_DGB_ESHAPE010 | uefb | 0.464 | 0.536 | 0.644 | 1.201 | 0.759 | 1.608 |
| P2C_DGB_ESHAPE010 | real | 0.500 | 0.500 | 0.657 | 1.523 | 1.208 | 1.612 |
| P2C_DGB_ESHAPE010 | synthetic | 0.442 | 0.558 | 0.649 | 1.252 | 0.773 | 1.557 |

## Interpretation

### 1. Global gate floors do not recover UEFB

Both route-floor variants decrease UEFB PSNR relative to P2B Gauge-only:

```text
P2B Gauge-only UEFB PSNR = 17.719
P2C floor015 UEFB PSNR = 17.695
P2C floor025 UEFB PSNR = 17.684
```

This falsifies the simple hypothesis that UEFB was mainly under-restored due to low route weight. Forcing more restoration globally worsens UEFB and, at floor015, also hurts paired domains.

### 2. Reducing e_shape to 0.010 partially recovers UEFB and improves synthetic, but breaks real

Best UEFB among P2C is:

```text
P2C_DGB_ESHAPE010 UEFB PSNR = 17.792
```

This improves over P2B Gauge-only and Phase2 full DGB on UEFB, and it gives the best synthetic PSNR among all DGB variants tested:

```text
P2C_DGB_ESHAPE010 Synthetic PSNR = 18.588
```

But real PSNR drops below P3c/P7B:

```text
P2C_DGB_ESHAPE010 Real PSNR = 19.604
P3c Real PSNR = 20.021
P7B_RA010 Real PSNR = 19.847
```

So reduced e_shape is useful evidence, but not promotable.

### 3. P2B Gauge-only remains the best paired-domain base

`P2B_DGB_GAUGE_ONLY` still has the best real PSNR among the DGB branch:

```text
P2B Gauge-only Real PSNR = 20.432
Best P2C Real PSNR = 20.297 (P2C_DGB_GATE_FLOOR025)
```

P2C did not produce a candidate that simultaneously keeps P2B's real-domain gain and recovers UEFB.

## Promotion decision

Decision: **NO-GO for multiseed promotion**.

Reason: all P2C candidates fail at least one PSNR gate versus both P3c aggregate and P7B_RA010. The closest direction is `P2C_DGB_ESHAPE010` for UEFB/synthetic, but its real PSNR is too low.

## Next recommendation

Do **not** promote P2C. The next controlled branch should not continue global gate floors. A more promising follow-up is a **balanced e_shape/gate objective** that preserves real-domain behavior from P2B Gauge-only while keeping the `e_shape=0.010` UEFB/synthetic benefit. Keep the same guardrails: no teacher distillation, no `rec_by_dataset`, no RA>0.010.

Candidate next branch name:

```text
P2D_BALANCED_ESHAPE010_REAL_PRESERVE
```

Possible first test: `e_shape=0.010` plus a small global real-preservation objective that is not dataset-label reweighting, e.g. image-stat conditioned gate calibration or a generic high-contrast restoration consistency term.

## Artifacts

- Summary CSV: `results/tables/p2c_uefb_recovery_summary.csv`
- Summary JSON: `results/tables/p2c_uefb_recovery_summary.json`
- Route diagnostics: `results/tables/p2c_uefb_recovery_route_diagnostics.json`
- Report: `results/hermes_audit/reports/P2C_UEFB_RECOVERY_REPORT.md`

Run directories:

- `P2C_DGB_GATE_FLOOR015`: `/home/user/a1_rlef_project/experiments/p2c_p2c_dgb_gate_floor015_image_stats_gauge_original_gate_floor015_seed3407`
- `P2C_DGB_GATE_FLOOR025`: `/home/user/a1_rlef_project/experiments/p2c_p2c_dgb_gate_floor025_image_stats_gauge_original_gate_floor025_seed3407`
- `P2C_DGB_ESHAPE025`: `/home/user/a1_rlef_project/experiments/p2c_p2c_dgb_eshape025_image_stats_gauge_original_route_eshape025_seed3407`
- `P2C_DGB_ESHAPE010`: `/home/user/a1_rlef_project/experiments/p2c_p2c_dgb_eshape010_image_stats_gauge_original_route_eshape010_seed3407`


## Verification logs

Full suite:

```text
........................................................................ [ 75%]
........................                                                 [100%]
96 passed in 19.75s
```

Compile:

```text
passed (no output)
```
