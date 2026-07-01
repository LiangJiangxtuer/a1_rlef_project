# A1-RLEF v2 Claim Ledger

This ledger separates executed evidence from pending protocols and hypotheses.

## Executed evidence

- Project scaffold exists and tests pass: `8 passed`.
- UEFB-smoke generation executed with 20 train / 20 test samples.
- P0 M0/M3/M4/M5 smoke runs executed for 30 steps each.
- P1 real/synthetic nogauge/fixed0p02 light runs executed for 30 steps each.

## Supported only at smoke level

- M3/M4/M5 reduce absolute E_MAE on UEFB-smoke relative to M0.
- M5 improves Q_ECE on UEFB-smoke relative to M0/M3/M4.
- M4/M5 reduce NAI on UEFB-smoke.

## Not supported yet

- Spatial E correctness: P0 E_corr remains negative for E branches.
- Synthetic domain tension in the new scaffold: P1-light did not reproduce anchor hurt.
- SOTA or Retinexformer-level performance.
- CRF/noise/RAW physical inversion.

## Pending protocols before paper claims

- Full P1 real/synthetic replication at 1000-3000 steps.
- Formal UEFB-v2 generation and M0-M5 ablation.
- Full-resolution paired benchmark against official baselines.
- Unpaired real visual/no-reference evaluation.
- 3-seed mean±std and significance tests.


## P1 formal 1000-step update — 2026-07-01

Executed six full-train/full-test LOL-v2 gauge runs with seed 3407:

| Dataset | nogauge PSNR | fixed0p02 PSNR | adaptive PSNR | Decision |
|---|---:|---:|---:|---|
| LOL-v2-real | 18.748 | 18.734 | 19.050 | fixed real gain did not reproduce; adaptive best |
| LOL-v2-synthetic | 20.116 | 19.906 | 20.269 | fixed anchor hurts; adaptive best PSNR |

Claim calibration after this update:

- Supported at single-seed P1 formal level: fixed gauge is domain-fragile on synthetic; adaptive gauge is a better candidate than fixed anchor.
- Not supported: fixed `e_mean=0.02` as a main positive contribution; SOTA/Retinexformer-level claims; physical E spatial correctness.
- Next required evidence: UEFB-v2 formal E/A/Q metrics and M0-M5 ablation.
