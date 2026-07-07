# Claim Ledger — Phase 0 Baseline Reproduction

Generated: 2026-07-02T16:45:01.235719+00:00

## Current best method

Method name: not selected in Phase 0. This phase freezes baselines only.  
Default baseline remains: `P3c M4J_ES e_shape=0.05` until a later phase produces a verified balanced improvement.  
Relevant near-miss: `P7B_DHEAD_RA010`.

## Verified claims

1. **The current project contains evaluable P3c/P6/P7B checkpoints and configs.**
   - Evidence: paths listed in `PHASE0_BASELINE_REPRODUCTION.md` and summary CSV.
   - Whether writable in paper正文: yes, as reproducibility evidence / experiment setup.

2. **Phase 0 reproduction matches recorded expectations within threshold.**
   - User-targeted baselines P3c/P6/P7B: PASS.
   - Full prompt-listed rows: PASS.
   - Evidence: `/home/user/a1_rlef_project/results/hermes_audit/tables/phase0_baselines_summary.csv`.

3. **P3c remains the frozen conservative default baseline before new-method work.**
   - Evidence: P3c 3-seed aggregate reproduced and P6/P7/P7B still exhibit the previously documented trade-offs.

## Partially supported claims

1. **P6/P7/P7B are useful near-miss diagnostics.**
   - Evidence: reproduced metrics show partial domain passes, but they are single-seed routes and not final methods.
   - Next required experiment: Phase 1/2 only after this Phase 0 ledger is accepted.

## Rejected claims

1. **Do not claim RLEF/DGB-RLEF is SOTA over Retinexformer.**
   - Retinexformer official blind baseline remains far stronger on LOL-v2 paired PSNR.

2. **Do not claim P7B_DHEAD_RA010 solves tri-domain balance.**
   - It remains a near-miss; real PSNR is below P3c mean in the recorded/reproduced baseline chain.

## Comparison to P3c/P6/P7B

- P3c is the frozen default reference.
- P6 provides multiscale trunk evidence but synthetic remains the weak domain.
- P7_MS_DHEAD improves UEFB/synthetic but hurts real.
- P7B_DHEAD_RA010 is the strongest P7-family near-miss but does not replace P3c.

## Comparison to Retinexformer

- Retinexformer official blind remains the strong SOTA reference for LOL-v2 real/synthetic paired fidelity.
- Paper language should be mechanism/benchmark/calibration-oriented unless later phases close the fidelity gap.

## Final decision

Decision: **phase0_pass_freeze_baselines_stop_here**  
Reason: all Phase 0 rows are within deterministic reproduction thresholds; P3c seed rows are traceability rows and the threshold applies to the aggregate mean. Phase 1 was not executed and requires a separate user instruction.
