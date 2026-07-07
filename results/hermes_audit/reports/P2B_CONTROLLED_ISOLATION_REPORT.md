# P2B Controlled Isolation Report

Generated: 2026-07-03T03:44:53.515881+00:00
Project root: `/home/user/a1_rlef_project`

## Scope

Executed P2B controlled isolation to split the two suspected factors from Phase2 full DGB:

1. `P2B_DGB_GAUGE_ONLY`: image-stat gauge + original low-input gate route (`route_type=identity_gate`).
2. `P2B_DGB_ROUTE_ONLY`: adaptive feature-head gauge + recoverability safe router (`safe_alpha=0.70`).

Both runs use the same 1000-step joint UEFB + LOL-v2 real + LOL-v2 synthetic protocol, seed 3407, P6 multiscale trunk, `e_shape=0.05`, q branch, no teacher distillation, no `rec_by_dataset`, and no real anchor >0.010.

No GitHub commit/push was made.

## Commands executed

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_p2b_controlled_isolation.py   --max_steps 1000 --train_crop 128 --batch_size 8 --parallel 2 --gpu_start 0
```

Diagnostics:

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/diagnose_phase2_dgb_route.py ...
```

Verification:

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m pytest tests -q
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m compileall -q src scripts
```

## Main comparison

| Metric | P3c mean | P7B_RA010 | Phase2 full DGB | P2B Gauge-only | P2B Route-only |
|---|---:|---:|---:|---:|---:|
| UEFB PSNR | 17.915 | 18.085 | 17.754 | 17.719 | 17.713 |
| Real PSNR | 20.021 | 19.847 | 19.939 | 20.432 | 19.967 |
| Synthetic PSNR | 17.678 | 17.965 | 18.479 | 18.402 | 17.945 |
| UEFB E_corr | 0.436 | 0.438 | 0.528 | 0.508 | 0.465 |
| Real E_corr | 0.707 | 0.729 | 0.679 | 0.712 | 0.664 |
| Synthetic E_corr | 0.841 | 0.841 | 0.818 | 0.839 | 0.837 |

## Deltas vs Phase2 full DGB

| Variant | Δ UEFB PSNR vs Phase2 | Δ Real PSNR vs Phase2 | Δ Synthetic PSNR vs Phase2 | Promotion |
|---|---:|---:|---:|---|
| P2B_DGB_GAUGE_ONLY | -0.035 | 0.493 | -0.078 | False |
| P2B_DGB_ROUTE_ONLY | -0.041 | 0.028 | -0.534 | False |

## Route/gauge diagnostics

For `P2B_DGB_GAUGE_ONLY`, the fraction column is effectively the low-input blend fraction `(1-A)` under the original route. For `P2B_DGB_ROUTE_ONLY`, it is the safe-path blend fraction under the recoverability router.

| Variant | Split | A_mean | low/safe fraction | entropy | mu_mean | mu_min | mu_max |
|---|---|---:|---:|---:|---:|---:|---:|
| P2B_DGB_GAUGE_ONLY | uefb | 0.469 | 0.531 | 0.641 | 1.202 | 0.812 | 1.569 |
| P2B_DGB_GAUGE_ONLY | real | 0.522 | 0.478 | 0.648 | 1.491 | 1.205 | 1.572 |
| P2B_DGB_GAUGE_ONLY | synthetic | 0.443 | 0.557 | 0.647 | 1.249 | 0.824 | 1.523 |
| P2B_DGB_ROUTE_ONLY | uefb | 0.413 | 0.587 | 0.624 | 1.051 | 0.104 | 3.148 |
| P2B_DGB_ROUTE_ONLY | real | 0.465 | 0.535 | 0.650 | 2.205 | 1.138 | 3.273 |
| P2B_DGB_ROUTE_ONLY | synthetic | 0.395 | 0.605 | 0.630 | 1.397 | -0.007 | 2.695 |

## Interpretation

### 1. Safe-router conservatism is a real-domain bottleneck, but not the whole UEFB problem

Removing the safe router while keeping image-stat gauge gives the strongest paired-domain result:

```text
P2B_DGB_GAUGE_ONLY real PSNR = 20.432
P2B_DGB_GAUGE_ONLY synthetic PSNR = 18.402
```

This beats both P3c and P7B_RA010 on real/synthetic PSNR, so the full Phase2 safe router was indeed suppressing real-domain restoration. However, UEFB remains below P3c and P7B even after removing the safe router.

### 2. Image-stat gauge is not the paired-domain failure source

Gauge-only keeps image-stat gauge and improves real/synthetic strongly. Therefore image-stat gauge is viable for paired-domain performance. Its remaining problem is UEFB, not paired LOL-v2.

### 3. Route-only does not rescue UEFB and weakens synthetic

`P2B_DGB_ROUTE_ONLY` keeps the recoverability safe router but reverts to adaptive feature-head gauge. It still fails UEFB and loses most of the synthetic gain from Phase2 full DGB. This indicates the Phase2 synthetic improvement mostly came from image-stat gauge plus the safe route interaction, not from the router alone.

## Promotion decision

Decision: **NO-GO for multiseed promotion**.

Reason: neither isolation variant beats both P3c aggregate and P7B_RA010 on all three PSNR gates. `P2B_DGB_GAUGE_ONLY` is the most promising paired-domain route, but its UEFB PSNR is still below the promotion threshold.

## Next recommendation

Recommended next branch: **P2C UEFB recovery from Gauge-only**, not another safe-router variant.

Use `P2B_DGB_GAUGE_ONLY` as the base because it fixes real/synthetic best. The next controlled intervention should target UEFB without using forbidden scalar routes:

1. Keep image-stat gauge and original low route.
2. Test an UEFB-specific exposure-shape/route calibration that does **not** use teacher distillation, `rec_by_dataset`, or RA>0.010.
3. First smoke candidate: reduce gate conservatism / increase restoration contribution on UEFB while preserving paired real/synthetic numbers.
4. Promotion still requires UEFB, real, and synthetic PSNR all beating P3c mean and P7B_RA010.

## Artifacts

- Summary CSV: `results/tables/p2b_controlled_isolation_summary.csv`
- Summary JSON: `results/tables/p2b_controlled_isolation_summary.json`
- Route diagnostics: `results/tables/p2b_controlled_isolation_route_diagnostics.json`
- Gauge-only run: `/home/user/a1_rlef_project/experiments/p2b_p2b_dgb_gauge_only_image_stats_gauge_original_gate_route_seed3407`
- Route-only run: `/home/user/a1_rlef_project/experiments/p2b_p2b_dgb_route_only_adaptive_gauge_safe_router_seed3407`

## Verification logs

Full suite:

```text
........................................................................ [ 79%]
...................                                                      [100%]
91 passed in 19.00s
```

Compile:

```text
passed (no output)
```
