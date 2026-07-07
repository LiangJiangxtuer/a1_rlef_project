# DGB Branch Stop & Consolidation Report

Generated: 2026-07-04T13:07:30.013047+00:00  
Project root: `/home/user/a1_rlef_project`

## Decision

```text
STOP_DGB_BRANCH_AND_CONSOLIDATE
```

The DGB branch is stopped. Do not run further DGB scalar/gate/router sweeps unless a future user request explicitly reopens it with a new hypothesis and a new validation gate.

## Guardrails

- No GitHub commit/push.
- No local git commit.
- No new model training in this consolidation step.
- No teacher distillation added.
- No `rec_by_dataset` added.
- No real anchor above `0.010` added.
- No SOTA / Retinexformer-level claim.

## Stop rationale

Across Phase2, P2B, P2C, P2D, and P2E, the branch produced useful diagnostics but no promotable method. Every follow-up changed the trade-off surface rather than solving the joint gate:

- Full DGB improved synthetic and E diagnostics but failed UEFB/real promotion gates.
- P2B isolated the safe-router bottleneck; Gauge-only was best real, but not jointly promotable.
- P2C lower `e_shape=0.010` helped UEFB/synthetic direction but hurt real.
- P2D generic structure preservation did not recover real.
- P2E image-stat gate prior changed routing as intended but did not recover real PSNR enough, and stronger prior damaged UEFB/synthetic.

## Promotion reference gates

| Reference | UEFB PSNR | Real PSNR | Synthetic PSNR |
|---|---:|---:|---:|
| P3c M4J_ES e_shape=0.05 mean | 17.915 | 20.021 | 17.678 |
| P7B_DHEAD_RA010 near-miss | 18.085 | 19.847 | 17.965 |

Joint promotion required beating the relevant P3c/P7B gates on UEFB, real, and synthetic. No DGB variant satisfies this.

## Consolidated DGB evidence table

| Stage | Variant | Intervention | UEFB PSNR | Real PSNR | Synthetic PSNR | UEFB E_corr | Real E_corr | Synthetic E_corr | Promote |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Phase2 full DGB | `DGB_RLEF_MINIMAL_S3407` | multiscale_image_stats_gauge_safe_router | 17.754 | 19.939 | 18.479 | 0.528 | 0.679 | 0.818 | False |
| P2B controlled isolation | `P2B_DGB_GAUGE_ONLY` | gauge_only | 17.719 | 20.432 | 18.402 | 0.508 | 0.712 | 0.839 | False |
| P2B controlled isolation | `P2B_DGB_ROUTE_ONLY` | route_only | 17.713 | 19.967 | 17.945 | 0.465 | 0.664 | 0.837 | False |
| P2C UEFB recovery | `P2C_DGB_GATE_FLOOR015` | global_route_floor | 17.695 | 19.658 | 17.635 | 0.482 | 0.748 | 0.866 | False |
| P2C UEFB recovery | `P2C_DGB_GATE_FLOOR025` | global_route_floor | 17.684 | 20.297 | 18.356 | 0.534 | 0.651 | 0.792 | False |
| P2C UEFB recovery | `P2C_DGB_ESHAPE025` | reduced_e_shape_weight | 17.655 | 19.873 | 18.187 | 0.488 | 0.708 | 0.839 | False |
| P2C UEFB recovery | `P2C_DGB_ESHAPE010` | reduced_e_shape_weight | 17.792 | 19.604 | 18.588 | 0.472 | 0.730 | 0.813 | False |
| P2D real-preserve | `P2D_ESHAPE010_STRUCT002` | generic_structure_preserve | 17.733 | 19.781 | 18.542 | 0.475 | 0.715 | 0.810 | False |
| P2D real-preserve | `P2D_ESHAPE010_STRUCT005` | generic_structure_preserve | 17.863 | 19.541 | 18.399 | 0.472 | 0.656 | 0.784 | False |
| P2E gate-prior | `P2E_ESHAPE010_GATEPRI002` | image_stat_gate_prior | 17.850 | 19.487 | 18.683 | 0.493 | 0.666 | 0.804 | False |
| P2E gate-prior | `P2E_ESHAPE010_GATEPRI005` | image_stat_gate_prior | 17.660 | 19.789 | 18.248 | 0.451 | 0.701 | 0.815 | False |

## Best split-wise DGB results and why they are not final

| Split winner | Stage | Variant | UEFB PSNR | Real PSNR | Synthetic PSNR | Why not final |
|---|---|---|---:|---:|---:|---|
| UEFB | P2D real-preserve | `P2D_ESHAPE010_STRUCT005` | 17.863 | 19.541 | 18.399 | UEFB still below P7B_RA010; real below P3c/P7B/P2B. |
| Real | P2B controlled isolation | `P2B_DGB_GAUGE_ONLY` | 17.719 | 20.432 | 18.402 | Best real but UEFB below P3c/P7B and synthetic below P2E/P2C; diagnostic only. |
| Synthetic | P2E gate-prior | `P2E_ESHAPE010_GATEPRI002` | 17.850 | 19.487 | 18.683 | Best synthetic but real fails strongly. |

## Final interpretation

DGB generated three useful findings:

1. Image-stat gauge and gauge-only calibration are deployable and can preserve or improve paired metrics in some splits.
2. The recoverability safe-router is too conservative for the current restoration objective and suppresses UEFB/real improvements.
3. Route/gate statistics can be manipulated, but routing changes alone do not guarantee real-domain output quality.

However, these findings are diagnostic, not a final method. The core failure mode is persistent three-way trade-off among UEFB, real, and synthetic PSNR.

## Consolidated default after stopping

Recommended default for the current paper/project state:

```text
P3c M4J_ES e_shape=0.05 remains the conservative validated default.
```

Keep as references only:

- `P7B_DHEAD_RA010`: best P7-family near-miss / domain-head reference.
- `P2B_DGB_GAUGE_ONLY`: best real-domain DGB diagnostic, not a final DGB promotion.
- `P2E_ESHAPE010_GATEPRI002`: best synthetic DGB diagnostic, not a final DGB promotion.

## What not to do next

Do not continue:

- larger `gate_prior` sweeps;
- more global `route_floor` sweeps;
- stronger generic `structure_preserve` sweeps;
- more safe-router scalar tuning;
- DGB multiseed promotion.

## If a new route is needed later

Start a new branch outside DGB. Recommended direction:

```text
stronger restoration backbone / Retinex-factorization redesign,
with RLEF exposure-field heads retained as auxiliary interpretability/calibration.
```

This should be treated as a new route, not P2F continuation of DGB.

## Verification

Consolidation artifacts were parsed with `json.tool`, the compact archive was listed successfully, and the full test suite still passes:

```text
106 passed in 19.29s
```

Compact archive SHA256 is recorded next to the archive in:

```text
results/hermes_audit/checkpoints/dgb_branch_stopped_20260704.tar.gz.sha256
```

## Artifacts

- Consolidation report: `results/hermes_audit/reports/DGB_BRANCH_STOP_CONSOLIDATION_REPORT.md`
- Consolidation CSV: `results/tables/dgb_branch_consolidation_summary.csv`
- Consolidation JSON: `results/tables/dgb_branch_consolidation_summary.json`
- Frozen compact bundle directory: `results/hermes_audit/checkpoints/dgb_branch_stopped_20260704/`
- Compact archive: `results/hermes_audit/checkpoints/dgb_branch_stopped_20260704.tar.gz`
- Verification log: `results/hermes_audit/logs/dgb_consolidation_full_suite.log`
