# Phase 2 DGB-RLEF Candidate Training Report

Generated: 2026-07-03T03:01:09.614224+00:00
Project root: `/home/user/a1_rlef_project`

## Scope

Executed Phase 2 candidate training for the first minimal DGB-RLEF candidate recommended by Phase 1:

```text
DGB_RLEF_MINIMAL_S3407
base: P6 multiscale trunk
shape: P3c e_shape=0.05, public gauge-invariant loss
calibration: image-stat gauge head with warm gauge schedule
route: recoverability safe router, safe_alpha=0.70
```

No GitHub commit/push was made.

## Commands executed

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_phase2_dgb_candidate.py   --max_steps 1000 --train_crop 128 --batch_size 8 --gpu_start 0
```

Verification:

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m pytest tests -q
/home/user/miniconda3/envs/cutler_dinov3/bin/python -m compileall -q src scripts
```

## Artifacts

- Config: `/home/user/a1_rlef_project/configs/dgb_rlef/phase2_dgb_rlef_minimal_s3407_multiscale_image_stats_gauge_safe_router.yml`
- Run dir: `/home/user/a1_rlef_project/experiments/phase2_dgb_rlef_minimal_s3407_multiscale_image_stats_gauge_safe_router_seed3407`
- Checkpoint: `/home/user/a1_rlef_project/experiments/phase2_dgb_rlef_minimal_s3407_multiscale_image_stats_gauge_safe_router_seed3407/checkpoints/last.pth`
- Train/eval log: `/home/user/a1_rlef_project/logs/phase2_dgb_candidate/phase2_dgb_rlef_minimal_s3407_multiscale_image_stats_gauge_safe_router.log`
- Summary CSV: `results/tables/phase2_dgb_candidate_summary.csv`
- Summary JSON: `results/tables/phase2_dgb_candidate_summary.json`
- Route diagnostics: `results/tables/phase2_dgb_route_diagnostics.json`

## Result table

| Metric | P3c mean | P7B_RA010 | Phase2 DGB | Δ vs P3c | Δ vs P7B |
|---|---:|---:|---:|---:|---:|
| UEFB PSNR | 17.915 | 18.085 | 17.754 | -0.160 | -0.331 |
| Real PSNR | 20.021 | 19.847 | 19.939 | -0.082 | 0.092 |
| Synthetic PSNR | 17.678 | 17.965 | 18.479 | 0.801 | 0.514 |
| UEFB E_corr | 0.436 | 0.438 | 0.528 | 0.092 | 0.089 |
| Real E_corr | 0.707 | 0.729 | 0.679 | -0.029 | -0.050 |
| Synthetic E_corr | 0.841 | 0.841 | 0.818 | -0.023 | -0.023 |

## Promotion decision

Decision: **NO-GO for multiseed promotion**.

Reason: promotion gate requires beating both P3c aggregate and P7B_RA010 near-miss on UEFB, real, and synthetic PSNR. Phase2 DGB improves synthetic PSNR strongly and slightly exceeds P7B_RA010 real PSNR, but it fails UEFB PSNR and remains below P3c real PSNR.

Promotion payload:

```json
{
  "promotion": [
    false
  ],
  "p7b_near_miss_reference": {
    "uefb_psnr": 18.0850629196167,
    "real_psnr": 19.84705744743347,
    "synthetic_psnr": 17.96549958229065
  }
}
```

## Route diagnostics

| Split | A_mean | safe_fraction | route_entropy | safe_rest_l1 | mu_mean | mu_min | mu_max |
|---|---:|---:|---:|---:|---:|---:|---:|
| uefb | 0.393 | 0.607 | 0.632 | 0.375 | 1.172 | 0.892 | 1.435 |
| real | 0.426 | 0.574 | 0.650 | 0.442 | 1.378 | 1.173 | 1.438 |
| synthetic | 0.359 | 0.641 | 0.623 | 0.337 | 1.206 | 0.902 | 1.402 |


Interpretation: the router is conservative across all splits (`safe_fraction` around 0.57–0.64). This likely protects synthetic PSNR but suppresses UEFB/real gains by blending heavily toward the safe path.

## Guardrail audit

- Teacher distillation: **not used**.
- `rec_by_dataset`: **not used**.
- Real anchor above `0.010`: **not used**.
- `gauge_schedule.max_weight`: `0.005` with hard cap `0.010`.
- Retinexformer/SOTA claim: **not made**.

## Verification logs

Full suite:

```text
........................................................................ [ 81%]
................                                                         [100%]
88 passed in 19.10s
```

Compile:

```text
passed (no output)
```

## Recommended next step

Do **not** promote this exact Phase2 candidate to 3 seeds. The next controlled experiment should isolate which DGB component caused the UEFB/real drop:

1. `P2B_DGB_GAUGE_ONLY`: image-stat gauge + original low-input gate route; no safe router.
2. `P2B_DGB_ROUTE_ONLY`: adaptive feature gauge + recoverability safe router.
3. If (1) recovers UEFB/real while preserving synthetic, tune router calibration; if (2) fails similarly, safe-router conservatism is the bottleneck.

This remains a Phase2 diagnostic ladder, not a SOTA route.
