# P2E Image-Stat Gate Prior Balance Report

Generated: 2026-07-04T11:07:00.640386+00:00
Project root: `/home/user/a1_rlef_project`

## Scope

Executed P2E after P2D. The hypothesis was that the P2C/P2D bottleneck is route/gate calibration under `e_shape=0.010`, not generic structure reconstruction. P2E adds a deployable low-image-stat gate prior that does not use dataset labels or teacher predictions.

Guardrails:

- No teacher distillation.
- No `rec_by_dataset`.
- No real anchor > `0.010`.
- No SOTA claim.
- No GitHub commit/push.

Implementation added an image-stat gate prior target derived from low-image luminance/darkness and contrast only:

```text
A_prior = clamp(base + dark_gain * dark_ratio(low) + contrast_gain * grad_mean(low), min, max)
gate_prior_loss = L1(A, A_prior)
```

P2E candidates:

1. `P2E_ESHAPE010_GATEPRI002`: `e_shape=0.010`, `gate_prior=0.02`.
2. `P2E_ESHAPE010_GATEPRI005`: `e_shape=0.010`, `gate_prior=0.05`.

Both use seed `3407`, 1000 steps, P6 multiscale trunk, image-stat gauge, original identity route, q branch, and joint UEFB + LOL-v2 real + synthetic training.

## Commands executed

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_p2e_gate_prior_balance.py   --max_steps 1000 --train_crop 128 --batch_size 8 --parallel 2 --gpu_start 0
```

Verification:

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m pytest tests -q
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m compileall -q src scripts
```

## Main comparison

| Metric | P3c mean | P7B_RA010 | P2B Gauge-only | P2C e010 | P2D best-UEFB | P2E gp002 | P2E gp005 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UEFB PSNR | 17.915 | 18.085 | 17.719 | 17.792 | 17.863 | 17.850 | 17.660 |
| Real PSNR | 20.021 | 19.847 | 20.432 | 19.604 | 19.541 | 19.487 | 19.789 |
| Synthetic PSNR | 17.678 | 17.965 | 18.402 | 18.588 | 18.399 | 18.683 | 18.248 |
| UEFB E_corr | 0.436 | — | 0.508 | 0.472 | 0.472 | 0.493 | 0.451 |
| Real E_corr | 0.707 | — | 0.712 | 0.730 | 0.656 | 0.666 | 0.701 |
| Synthetic E_corr | 0.841 | — | 0.839 | 0.813 | 0.784 | 0.804 | 0.815 |

## Deltas

| Variant | gate_prior | Δ UEFB vs P2C e010 | Δ Real vs P2C e010 | Δ Synthetic vs P2C e010 | Δ UEFB vs P2D best | Δ Real vs P2D best | Promotion |
|---|---:|---:|---:|---:|---:|---:|---|
| P2E_ESHAPE010_GATEPRI002 | 0.020 | 0.058 | -0.117 | 0.096 | -0.014 | -0.054 | False |
| P2E_ESHAPE010_GATEPRI005 | 0.050 | -0.131 | 0.185 | -0.340 | -0.203 | 0.248 | False |

## Route/gauge diagnostics

For these identity-route variants, `low fraction` is `(1-A)`.

| Variant | Split | A_mean | low fraction | route_entropy | mu_mean | mu_min | mu_max |
|---|---|---:|---:|---:|---:|---:|---:|
| P2E_ESHAPE010_GATEPRI002 | uefb | 0.525 | 0.475 | 0.667 | 1.202 | 0.770 | 1.600 |
| P2E_ESHAPE010_GATEPRI002 | real | 0.600 | 0.400 | 0.667 | 1.516 | 1.208 | 1.604 |
| P2E_ESHAPE010_GATEPRI002 | synthetic | 0.523 | 0.477 | 0.677 | 1.251 | 0.783 | 1.550 |
| P2E_ESHAPE010_GATEPRI005 | uefb | 0.492 | 0.508 | 0.674 | 1.202 | 0.769 | 1.603 |
| P2E_ESHAPE010_GATEPRI005 | real | 0.585 | 0.415 | 0.674 | 1.518 | 1.208 | 1.606 |
| P2E_ESHAPE010_GATEPRI005 | synthetic | 0.503 | 0.497 | 0.680 | 1.252 | 0.782 | 1.552 |

## Interpretation

### 1. The gate-prior objective changes routing as intended

P2E increased real split restoration-route usage relative to P2D/P2C diagnostics:

```text
P2E gp002 real A_mean = 0.600
P2E gp005 real A_mean = 0.585
```

This verifies that the image-stat prior is active and pushes dark real images toward more restoration routing.

### 2. More real routing did not translate into real PSNR recovery

Best P2E real is:

```text
P2E_ESHAPE010_GATEPRI005 Real PSNR = 19.789
```

This is slightly above P2C e010 real (19.604) but remains below P7B (19.847), P3c (20.021), and far below P2B Gauge-only (20.432). Thus the route prior improves routing statistics but not enough to recover real-domain output quality.

### 3. gp002 is useful for synthetic; gp005 is a weak real-recovery attempt

`P2E_ESHAPE010_GATEPRI002` gives the best synthetic PSNR in the DGB line:

```text
P2E gp002 Synthetic PSNR = 18.683
P2C e010 Synthetic PSNR = 18.588
```

But gp002's real PSNR is poor. gp005 improves real vs gp002 and P2C e010, but damages UEFB and synthetic. This is still a trade-off, not a promotion candidate.

### 4. Best P2E per split

```text
Best UEFB: P2E_ESHAPE010_GATEPRI002 = 17.850
Best Real: P2E_ESHAPE010_GATEPRI005 = 19.789
Best Synthetic: P2E_ESHAPE010_GATEPRI002 = 18.683
```

None passes the joint promotion gates.

## Promotion decision

Decision: **NO-GO for multiseed promotion**.

Reason: both P2E candidates fail at least one PSNR gate against P3c aggregate and P7B_RA010. P2E confirms the gate-prior mechanism affects routing, but the resulting restoration quality is insufficient.

## Next recommendation

Do **not** promote P2E. Do not simply increase `gate_prior` weight; stronger prior trades away UEFB/synthetic and still does not recover real.

Recommended next direction:

```text
P2F_ROUTE_QUALITY_COUPLING_OR_STOP_DGB
```

Two options:

1. Route-quality coupling: couple the gate prior to output-quality signals already available without labels at inference/training time, e.g. Q-gated or recoverability-aware gate regularization while avoiding the safe-router bottleneck.
2. Stop the DGB branch and consolidate: P2B Gauge-only remains the best real-domain DGB base, while lower-shape variants give partial UEFB/synthetic improvements but cannot satisfy all gates.

## Artifacts

- Summary CSV: `results/tables/p2e_gate_prior_balance_summary.csv`
- Summary JSON: `results/tables/p2e_gate_prior_balance_summary.json`
- Route diagnostics: `results/tables/p2e_gate_prior_balance_route_diagnostics.json`
- Report: `results/hermes_audit/reports/P2E_GATE_PRIOR_BALANCE_REPORT.md`

Run directories:

- `P2E_ESHAPE010_GATEPRI002`: `/home/user/a1_rlef_project/experiments/p2e_p2e_eshape010_gatepri002_eshape010_image_stat_gate_prior002_seed3407`
- `P2E_ESHAPE010_GATEPRI005`: `/home/user/a1_rlef_project/experiments/p2e_p2e_eshape010_gatepri005_eshape010_image_stat_gate_prior005_seed3407`


## Verification logs

Full suite:

```text
........................................................................ [ 67%]
..................................                                       [100%]
106 passed in 19.66s
```

Compile:

```text
passed (no output)
```
