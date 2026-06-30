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
