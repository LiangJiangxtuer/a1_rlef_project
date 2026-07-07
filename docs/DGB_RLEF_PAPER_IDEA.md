# DGB_RLEF Paper Idea After Consolidation

Generated: 2026-07-05T03:06:24.533855+00:00

## Working title

Gauge-Aware Exposure-Field Diagnostics for Interpretable Low-Light Enhancement

## Abstract draft

Local exposure-field models promise interpretable low-light enhancement, but our staged experiments show that exposure fields are not identifiable under gradient/Poisson constraints alone and that simple restoration, distillation, or domain-anchor fixes create real/synthetic/UEFB trade-offs. We therefore reframe the contribution as a gauge-aware exposure-field auxiliary calibration study. Across P1-P7 and DGB controlled isolation, the validated default is P3c M4J_ES with low-pass centered E-shape consistency, while Retinexformer remains the paired-fidelity ceiling. The paper presents the UEFB-v2 protocol, a claim-calibrated ablation ladder, and negative evidence explaining why DGB scalar/gate/router sweeps should stop. The resulting contribution is not SOTA restoration, but a reproducible diagnostic framework for exposure-field interpretability and failure-aware LLIE evaluation.

## Introduction logic

1. Retinex-style physical heads are attractive but can overclaim physical correctness.
2. Exposure-field losses have gauge ambiguity; fixed anchors are domain fragile.
3. The project evidence shows repeated three-domain trade-offs.
4. Centered E-shape consistency is the stable positive mechanism.
5. DGB attempts clarify what does not solve the trade-off, which is valuable for honest method design.

## Method structure to write

- Gauge-invariant exposure decomposition: `E = S + mu`, `mean(S)=0`.
- Low-pass centered E-shape consistency.
- A-gate / diagnostic recoverability maps as interpretability outputs.
- UEFB-v2 benchmark with E/A/Q ground truth.
- Claim ledger and negative-route audit as reproducibility protocol.

## Experiments chapter

1. Datasets and implementation details.
2. Main compact table: P3c default vs official baselines.
3. UEFB-v2 exposure-field diagnostics.
4. Ablation ladder: M0/M4/M4J/M4J_ES/P3c/P3d.
5. Appendix: P5/P6/P7/DGB negative routes.
6. No-reference unpaired supplement as support/limitation.

## Contributions

1. A gauge-aware reading of local exposure-field enhancement.
2. A validated UEFB-v2 diagnostic protocol with E/A/Q metrics.
3. A compact evidence ladder showing which mechanisms work and which fail.
4. A transparent negative-result appendix preventing SOTA overclaiming.

## Limitations

- P3c does not approach Retinexformer paired-fidelity performance.
- DGB was stopped before 3-seed because no candidate passed the joint gate.
- No-reference unpaired metrics are auxiliary and not definitive.
- Q-branch evidence is ablation-level, not default-method evidence.

## Claims that must not be written

- "DGB-RLEF outperforms Retinexformer."
- "DGB-RLEF is SOTA."
- "E_corr alone proves better enhancement."
- "Teacher distillation/domain heads solve the problem."
