# S1 Evidence Freeze + Paper Framing

Generated: 2026-07-04T16:22:47.568908+00:00

## Framing decision

```text
Main story: interpretable, gauge-aware exposure-field auxiliary calibration.
Not allowed: SOTA / Retinexformer-level restoration claim.
```

## Frozen default

```text
P3c M4J_ES e_shape=0.05
```

## Claim matrix

| claim_id | claim | status | evidence | allowed_location |
| --- | --- | --- | --- | --- |
| C1 | RLEF should be framed as interpretable exposure-field auxiliary calibration, not SOTA restoration. | supported as framing | P4 official baselines show Retinexformer real/synthetic PSNR gap; P3c has stable E diagnostics. | main text framing + limitations |
| C2 | Adaptive/gauge-aware exposure modeling is preferable to fixed absolute gauge. | supported | P1 formal: adaptive is best on real/synthetic; fixed0p02 is domain-fragile. | method motivation |
| C3 | A-gate is the strongest early mechanism in UEFB ablation. | supported single-seed | P3 M4 gate gives best UEFB PSNR among M0-M5. | ablation discussion |
| C4 | Low-pass centered e_shape is the core mechanism for positive spatial E diagnostics. | supported | P3b M4J_ES and P3c 3-seed E_corr stay positive. | main method + diagnostics |
| C5 | DGB is not a final route. | stopped | DGB consolidation: no Phase2/P2B/P2C/P2D/P2E variant passes joint gate. | appendix negative diagnostics |

## Source anchors

- `docs/CLAIM_LEDGER.md`
- `results/hermes_audit/reports/STAGE_EXPERIMENT_SUMMARY_AND_ADJUSTED_PLAN.md`
- `results/hermes_audit/reports/DGB_BRANCH_STOP_CONSOLIDATION_REPORT.md`
