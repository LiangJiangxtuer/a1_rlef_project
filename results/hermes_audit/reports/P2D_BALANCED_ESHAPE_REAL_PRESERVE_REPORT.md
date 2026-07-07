# P2D Balanced e_shape010 Real-Preservation Report

Generated: 2026-07-03T07:51:40.023387+00:00
Project root: `/home/user/a1_rlef_project`

## Scope

Executed P2D as the follow-up to P2C: keep the useful `e_shape=0.010` direction, then add a generic real-preservation objective that does **not** use dataset labels.

Guardrails:

- No teacher distillation.
- No `rec_by_dataset`.
- No real anchor > `0.010`.
- No SOTA claim.
- No GitHub commit/push.

Implementation added a generic contrast-weighted structure-preservation loss:

```text
structure_preserve = weighted luminance L1 + grad_weight * weighted luminance-gradient L1
```

The weight map is computed from target-image contrast only; it is not conditioned on dataset names.

P2D candidates:

1. `P2D_ESHAPE010_STRUCT002`: `e_shape=0.010`, `structure_preserve=0.02`.
2. `P2D_ESHAPE010_STRUCT005`: `e_shape=0.010`, `structure_preserve=0.05`.

Both use seed `3407`, 1000 steps, P6 multiscale trunk, image-stat gauge, original identity route, q branch, and joint UEFB + LOL-v2 real + synthetic training.

## Commands executed

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_p2d_balanced_eshape_real_preserve.py   --max_steps 1000 --train_crop 128 --batch_size 8 --parallel 2 --gpu_start 0
```

Verification:

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m pytest tests -q
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m compileall -q src scripts
```

## Main comparison

| Metric | P3c mean | P7B_RA010 | P2B Gauge-only | P2C e010 | P2D struct002 | P2D struct005 |
|---|---:|---:|---:|---:|---:|---:|
| UEFB PSNR | 17.915 | 18.085 | 17.719 | 17.792 | 17.733 | 17.863 |
| Real PSNR | 20.021 | 19.847 | 20.432 | 19.604 | 19.781 | 19.541 |
| Synthetic PSNR | 17.678 | 17.965 | 18.402 | 18.588 | 18.542 | 18.399 |
| UEFB E_corr | 0.436 | — | 0.508 | 0.472 | 0.475 | 0.472 |
| Real E_corr | 0.707 | — | 0.712 | 0.730 | 0.715 | 0.656 |
| Synthetic E_corr | 0.841 | — | 0.839 | 0.813 | 0.810 | 0.784 |

## Deltas

| Variant | structure_preserve | Δ UEFB vs P2C e010 | Δ Real vs P2C e010 | Δ Synthetic vs P2C e010 | Δ UEFB vs P2B | Δ Real vs P2B | Promotion |
|---|---:|---:|---:|---:|---:|---:|---|
| P2D_ESHAPE010_STRUCT002 | 0.020 | -0.059 | 0.178 | -0.046 | 0.014 | -0.650 | False |
| P2D_ESHAPE010_STRUCT005 | 0.050 | 0.071 | -0.063 | -0.189 | 0.144 | -0.891 | False |

## Route/gauge diagnostics

For these identity-route variants, `low fraction` is `(1-A)`.

| Variant | Split | A_mean | low fraction | route_entropy | mu_mean | mu_min | mu_max |
|---|---|---:|---:|---:|---:|---:|---:|
| P2D_ESHAPE010_STRUCT002 | uefb | 0.510 | 0.490 | 0.650 | 1.200 | 0.771 | 1.597 |
| P2D_ESHAPE010_STRUCT002 | real | 0.540 | 0.460 | 0.655 | 1.513 | 1.206 | 1.601 |
| P2D_ESHAPE010_STRUCT002 | synthetic | 0.476 | 0.524 | 0.654 | 1.250 | 0.784 | 1.547 |
| P2D_ESHAPE010_STRUCT005 | uefb | 0.477 | 0.523 | 0.643 | 1.203 | 0.776 | 1.597 |
| P2D_ESHAPE010_STRUCT005 | real | 0.515 | 0.485 | 0.653 | 1.514 | 1.208 | 1.601 |
| P2D_ESHAPE010_STRUCT005 | synthetic | 0.455 | 0.545 | 0.649 | 1.252 | 0.789 | 1.548 |

## Interpretation

### 1. Generic structure preservation gives partial trade-off control, but not promotion

`P2D_ESHAPE010_STRUCT002` improves real PSNR over P2C e010:

```text
P2C e010 Real PSNR = 19.604
P2D struct002 Real PSNR = 19.781
```

But it loses UEFB and synthetic relative to P2C e010. It also remains below P7B on real PSNR.

`P2D_ESHAPE010_STRUCT005` gives the best UEFB among the DGB line so far:

```text
P2D struct005 UEFB PSNR = 17.863
P3c UEFB PSNR = 17.915
P7B_RA010 UEFB PSNR = 18.085
```

It is still below both UEFB gates and its real PSNR is worse than P2C e010.

### 2. The structure-preserve objective did not solve the real-preservation problem

The intended real-preservation mechanism was too weak or misaligned. It improved real for the weak-weight setting but not enough; stronger weighting moved UEFB up but damaged real. This means the real drop in `e_shape=0.010` is not fixed by simply adding generic high-contrast reconstruction.

### 3. No P2D candidate dominates the active references

Best P2D per split:

```text
Best UEFB: P2D_ESHAPE010_STRUCT005 = 17.863
Best Real: P2D_ESHAPE010_STRUCT002 = 19.781
Best Synthetic: P2D_ESHAPE010_STRUCT002 = 18.542
```

None beats all of P3c, P7B_RA010, and P2B Gauge-only simultaneously.

## Promotion decision

Decision: **NO-GO for multiseed promotion**.

Reason: both P2D candidates fail at least one PSNR gate against P3c aggregate and P7B_RA010. `P2D_ESHAPE010_STRUCT005` is closest on UEFB, but it is still below the UEFB gate and has low real PSNR.

## Next recommendation

Do **not** promote P2D. Do not continue simply increasing `structure_preserve` weight.

Recommended next branch:

```text
P2E_GATE_PRIOR_UEFB_REAL_BALANCE
```

Rationale: the successful/failed routes suggest the bottleneck is not just reconstruction structure, but routing/calibration under `e_shape=0.010`. A better next step is an image-stat-conditioned gate prior or calibration term that can keep P2B's real-domain restoration while allowing the P2C/P2D lower-shape UEFB/synthetic gains — still without dataset-label reweighting, teacher distillation, or RA>0.010.

## Artifacts

- Summary CSV: `results/tables/p2d_balanced_eshape_real_preserve_summary.csv`
- Summary JSON: `results/tables/p2d_balanced_eshape_real_preserve_summary.json`
- Route diagnostics: `results/tables/p2d_balanced_eshape_real_preserve_route_diagnostics.json`
- Report: `results/hermes_audit/reports/P2D_BALANCED_ESHAPE_REAL_PRESERVE_REPORT.md`

Run directories:

- `P2D_ESHAPE010_STRUCT002`: `/home/user/a1_rlef_project/experiments/p2d_p2d_eshape010_struct002_eshape010_structure_preserve002_seed3407`
- `P2D_ESHAPE010_STRUCT005`: `/home/user/a1_rlef_project/experiments/p2d_p2d_eshape010_struct005_eshape010_structure_preserve005_seed3407`


## Verification logs

Full suite:

```text
........................................................................ [ 71%]
.............................                                            [100%]
101 passed in 21.16s
```

Compile:

```text
passed (no output)
```
