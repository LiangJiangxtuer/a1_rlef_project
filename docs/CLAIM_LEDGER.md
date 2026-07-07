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


## P2/P3 formal UEFB ablation update — 2026-07-01

Executed P2 UEFB-v2 formal generation and P3 M0-M5 single-seed 1000-step UEFB ablation.

P2 dataset:

- UEFB-v2 train: 3000 samples, test: 500 samples.
- Each sample includes low/high image plus `E_gt`, `A_gt`, `Q_gt`, and metadata.

P3 key results:

| Variant | UEFB PSNR | E_MAE | E_corr | Q_ECE | Real PSNR | Synthetic PSNR |
|---|---:|---:|---:|---:|---:|---:|
| M0 restorer_only | 17.379 | 1.076 | 0.000 | 0.412 | 15.306 | 12.564 |
| M1 phys_aux | 17.913 | 1.039 | -0.366 | 0.407 | 16.396 | 13.178 |
| M2 poisson | 17.891 | 1.833 | -0.104 | 0.398 | 16.689 | 13.340 |
| M3 adaptive_gauge | 17.670 | 0.302 | -0.095 | 0.398 | 16.111 | 13.586 |
| M4 gate | 18.653 | 0.292 | -0.139 | 0.394 | 17.411 | 13.963 |
| M5 recoverability_q | 18.535 | 0.285 | -0.243 | 0.389 | 17.029 | 13.778 |

Claim calibration after P3:

- Supported at single-seed P3 level: adaptive gauge reduces absolute E_MAE; A gate (M4) is the strongest current image-quality module; Q branch improves Q_ECE slightly but costs PSNR.
- Not supported: exposure-field spatial correctness, because E_corr remains negative for E-branch variants; SOTA/Retinexformer-level claims, because P4 official baseline comparison has not run and UEFB-only training has low LOL-v2 paired PSNR.
- Next required evidence: M4 joint-training with UEFB + LOL-v2-real/synthetic, E-shape repair diagnostics, and 3-seed verification before paper claims.


## P3b joint-training and E-shape repair update — 2026-07-01

Executed two single-seed 1000-step P3b runs:

| Variant | UEFB PSNR | UEFB E_corr | Real PSNR | Real E_corr | Synthetic PSNR | Synthetic E_corr |
|---|---:|---:|---:|---:|---:|---:|
| M4J joint | 18.082 | 0.282 | 19.581 | 0.501 | 16.508 | 0.369 |
| M4J_ES joint+e_shape | 17.981 | 0.440 | 20.277 | 0.648 | 17.141 | 0.799 |

Claim calibration after P3b:

- Supported at single-seed routing level: joint training improves paired fidelity over UEFB-only M4; low-pass centered E-shape consistency turns E_corr positive and improves real/synthetic PSNR.
- Still not supported: SOTA/Retinexformer-level claims, physical correctness in real images, or multi-seed stability.
- Next required evidence: M4J_ES 3-seed validation and e_shape weight sweep before P4 official baselines.


## P3c multi-seed and e_shape sweep update — 2026-07-01

Executed M4J_ES 3-seed validation at `e_shape=0.05` and seed3407 sweep for `e_shape ∈ {0.02, 0.05, 0.10}`.

3-seed e_shape=0.05:

| Metric | Mean±Std |
|---|---:|
| UEFB PSNR | 17.915±0.275 |
| UEFB E_corr | 0.436±0.003 |
| Real PSNR | 20.021±0.223 |
| Synthetic PSNR | 17.678±0.735 |
| Synthetic E_corr | 0.841±0.037 |

Sweep seed3407 conclusion:

- `e_shape=0.10` gives the best single-seed UEFB E_corr (0.457) and synthetic PSNR (17.821), with real PSNR 20.179.
- `e_shape=0.05` remains the validated 3-seed default; `e_shape=0.10` should be promoted to a 3-seed validation before P4 baselines.

Claim calibration after P3c:

- Supported: M4J_ES e=0.05 has stable positive E_corr across 3 seeds and improves paired fidelity over UEFB-only M4.
- Promising but not yet final: e_shape=0.10 as a potentially better default.
- Still not supported: SOTA/Retinexformer-level claims or real-image physical E-field correctness.


## P3d e_shape=0.10 multi-seed update — 2026-07-01

Executed M4J_ES 3-seed validation at `e_shape=0.10`.

| Metric | e=0.10 Mean±Std | e=0.05 Mean±Std |
|---|---:|---:|
| UEFB PSNR | 18.111±0.201 | 17.915±0.275 |
| UEFB E_corr | 0.449±0.007 | 0.436±0.003 |
| Real PSNR | 19.875±0.427 | 20.021±0.223 |
| Synthetic PSNR | 17.626±0.517 | 17.678±0.735 |

Decision: `e_shape=0.10` is not promoted as final default because synthetic PSNR mean does not exceed `e_shape=0.05` and real PSNR mean is lower. `e_shape=0.05` remains the P4 default; `e_shape=0.10` remains a useful ablation/robustness variant.


## P4 official baseline update — 2026-07-01

Executed official-code baselines on local LOL-v2 real/synthetic test outputs and re-evaluated with the project evaluator.

| Method | Real PSNR | Synthetic PSNR | Note |
|---|---:|---:|---|
| Retinexformer | 22.794 | 25.669 | official pretrained, blind eval |
| KinD++ | 22.211 | 19.259 | official code, high-assisted ratio uses high image |
| Zero-DCE++ | 18.491 | 17.576 | official pretrained, blind eval |
| RLEF M4J_ES e=0.05 | 20.021±0.223 | 17.678±0.735 | local 3-seed default, blind eval |

Decision: RLEF is not SOTA / not Retinexformer-level. It beats Zero-DCE++ in this protocol and has stable positive E-field diagnostics, so the defensible story is interpretable exposure-field auxiliary calibration, not raw restoration SOTA.


## P5 Retinexformer teacher distillation update — 2026-07-01

Executed P5 output-level Retinexformer train-teacher distillation while preserving RLEF M4J_ES heads.

| Variant | Real PSNR | Synthetic PSNR | UEFB PSNR | Decision |
|---|---:|---:|---:|---|
| P5_RD_T01 distill=0.10 | 19.588 | 17.960 | 17.948 | balanced but not default; real below P3c mean |
| P5_RD_T03 distill=0.30 | 20.416 | 16.996 | 18.169 | real improves but synthetic drops; not default |

Decision: simple output-level Retinexformer distillation is not a sufficient final route. Keep P3c M4J_ES e=0.05 as conservative default; pursue P5b domain-conditioned distillation or stronger-backbone integration.


## P5b domain-conditioned distillation update — 2026-07-01

Executed single-seed P5B_DW_R03_S005: real distill=0.30, synthetic distill=0.05, UEFB distill=0.00.

| Variant | Real PSNR | Synthetic PSNR | Real E_corr | Synthetic E_corr | Decision |
|---|---:|---:|---:|---:|---|
| P5B_DW_R03_S005 | 19.366 | 17.473 | 0.728 | 0.865 | no 3-seed promotion |
| P3c default mean | 20.021±0.223 | 17.678±0.735 | 0.707±0.053 | 0.841±0.037 | remains default |

Decision: P5b improves paired E_corr but fails both real/synthetic PSNR go/no-go gates versus P3c. Do not promote. Stop scalar output-level distillation sweeps; next route should be stronger-backbone / feature-level RLEF integration.


## P6 structural-backbone update — 2026-07-01

Executed P6_MS_M4J_ES: M4J_ES heads/losses with a two-level multiscale restoration backbone and no teacher/output-level distillation.

| Variant | UEFB PSNR | Real PSNR | Synthetic PSNR | UEFB E_corr | Real E_corr | Synthetic E_corr | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| P6_MS_M4J_ES | 18.015 | 20.197 | 17.598 | 0.464 | 0.629 | 0.812 | no immediate 3-seed promotion |
| P3c default mean | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | 0.436±0.003 | 0.707±0.053 | 0.841±0.037 | remains conservative default |

Decision: P6 validates structural-backbone direction on UEFB/real PSNR, but synthetic PSNR remains below P3c mean, so do not promote to 3-seed yet. Recommended next branch: P6b synthetic-protection structural route, not scalar output-level distillation.


## P6b-MS-SYNPROTECT update — 2026-07-01

Executed P6b synthetic-protection sweep: multiscale P6 backbone, no teacher distillation, synthetic paired reconstruction weights 1.05/1.10/1.25/1.50 via `rec_by_dataset`.

| Variant | synthetic rec weight | UEFB PSNR | Real PSNR | Synthetic PSNR | Decision |
|---|---:|---:|---:|---:|---|
| P6B_MS_SYN105 | 1.05 | 17.347 | 19.996 | 18.708 | no promotion |
| P6B_MS_SYN110 | 1.10 | 17.220 | 19.569 | 18.948 | no promotion |
| P6B_MS_SYN125 | 1.25 | 16.894 | 19.482 | 19.115 | no promotion |
| P6B_MS_SYN150 | 1.50 | 16.452 | 19.337 | 19.097 | no promotion |
| P3c default mean | — | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | remains default |

Decision: P6b rescues synthetic PSNR but harms UEFB/real; no 3-seed promotion. Do not continue scalar synthetic-rec weights. Next route should protect synthetic through gate/restoration aggressiveness or capacity control, not full reconstruction upweighting.


## P6c-MS-GATEPROTECT update — 2026-07-01

Executed P6c gate-protection sweep: multiscale P6 backbone, no teacher/output distillation, no `rec_by_dataset`, and gate-restoration controls via `gate_identity`, lower supervised gate loss, or `backbone_blocks=2`.

| Variant | blocks | gate | gate_identity | UEFB PSNR | Real PSNR | Synthetic PSNR | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| P6C_MS_GATEID005 | 3 | 0.020 | 0.005 | 17.604 | 20.456 | 17.550 | no promotion |
| P6C_MS_GATEID010 | 3 | 0.020 | 0.010 | 17.651 | 19.818 | 17.890 | no promotion |
| P6C_MS_GATELOW005 | 3 | 0.005 | 0.000 | 18.046 | 19.831 | 17.422 | no promotion |
| P6C_MS_B2_GATEID005 | 2 | 0.020 | 0.005 | 17.561 | 19.314 | 17.103 | no promotion |
| P3c default mean | — | — | — | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | remains default |

Decision: no 3-seed promotion. P6C_MS_GATEID005 passes real only; P6C_MS_GATEID010 passes synthetic only; P6C_MS_GATELOW005 passes UEFB only; blocks=2 hurts all primary PSNR metrics. Simple scalar A-gate protection shifts the trade-off but does not solve it. Next route should be domain-conditioned calibration/adapters rather than more global scalar knobs.


## P7-MS-DOMAINHEAD update — 2026-07-01

Executed P7 domain-conditioned calibration sweep: multiscale P6 trunk, M4J_ES losses, no teacher/output distillation, no `rec_by_dataset`; model receives canonical domains (`uefb`, `real`, `synthetic`) and learns tiny domain adapters.

| Variant | Adapter | UEFB PSNR | Real PSNR | Synthetic PSNR | Decision |
|---|---|---:|---:|---:|---|
| P7_MS_DGATE | `gate_bias` | 17.602 | 19.309 | 17.025 | no promotion |
| P7_MS_DHEAD | `head_bias` | 18.232 | 19.209 | 17.913 | no promotion |
| P7_MS_DAFFINE_GATE | `feature_affine+gate_bias` | 17.533 | 17.021 | 18.124 | no promotion |
| P3c default mean | — | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | remains default |

Decision: no 3-seed promotion. P7_MS_DHEAD is promising because UEFB and synthetic pass P3c mean without synthetic-rec upweighting, but real PSNR fails. Next route should start from DHEAD and add real-domain anchoring/regularization, not reintroduce global scalar knobs.


## P7b-MS-DHEAD-REALANCHOR update — 2026-07-02

Executed P7b real-domain head-bias anchoring: starts from P7_MS_DHEAD, keeps multiscale trunk and `domain_adapter=head_bias`, uses no teacher/output distillation and no `rec_by_dataset`; adds `domain_head_anchor_by_dataset={real: w, uefb: 0, synthetic: 0}`.

| Variant | real_anchor | UEFB PSNR | Real PSNR | Synthetic PSNR | Decision |
|---|---:|---:|---:|---:|---|
| P7B_DHEAD_RA001 | 0.001 | 18.190 | 19.127 | 17.517 | no promotion |
| P7B_DHEAD_RA005 | 0.005 | 17.936 | 19.058 | 18.008 | no promotion |
| P7B_DHEAD_RA010 | 0.010 | 18.085 | 19.847 | 17.965 | no promotion |
| P3c default mean | — | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | remains default |

Decision: no 3-seed promotion. P7B_DHEAD_RA010 is the best/closest candidate: UEFB and synthetic pass P3c mean, real improves by +0.638 over P7_MS_DHEAD but still misses P3c real mean by 0.174 dB. Next recommended route: P7c fine sweep around stronger real anchors (0.015/0.020/0.030) or warmup+anchor.


## P7c-MS-DHEAD-REALANCHOR-FINE update — 2026-07-02

Executed P7c stronger real-domain head-bias anchor fine sweep: starts from the P7B_DHEAD_RA010 design, keeps `domain_adapter=head_bias`, no teacher/output distillation, no `rec_by_dataset`, and sweeps real anchors 0.015/0.020/0.030.

| Variant | real_anchor | UEFB PSNR | Real PSNR | Synthetic PSNR | Decision |
|---|---:|---:|---:|---:|---|
| P7C_DHEAD_RA015 | 0.015 | 17.954 | 19.082 | 17.591 | no promotion |
| P7C_DHEAD_RA020 | 0.020 | 17.441 | 19.517 | 17.820 | no promotion |
| P7C_DHEAD_RA030 | 0.030 | 18.062 | 18.641 | 17.682 | no promotion |
| P3c default mean | — | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | remains default |

Decision: no 3-seed promotion. Stronger anchors did not improve on P7B_DHEAD_RA010; best P7-family candidate remains P7B_DHEAD_RA010. Next route, if continuing, should test scheduling (`P7d-MS-DHEAD-WARMANCHOR`) rather than larger scalar anchor coefficients.


## DGB branch stop/consolidation update — 2026-07-04

Executed and consolidated the full DGB diagnostic ladder: Phase2 full DGB, P2B controlled isolation, P2C UEFB recovery, P2D real-preservation, and P2E image-stat gate prior. No GitHub/local git commit was made.

| Best split-wise DGB result | Variant | UEFB PSNR | Real PSNR | Synthetic PSNR | Final status |
|---|---|---:|---:|---:|---|
| UEFB | P2D_ESHAPE010_STRUCT005 | 17.863 | 19.541 | 18.399 | no promotion |
| Real | P2B_DGB_GAUGE_ONLY | 17.719 | 20.432 | 18.402 | diagnostic only |
| Synthetic | P2E_ESHAPE010_GATEPRI002 | 17.850 | 19.487 | 18.683 | no promotion |
| P3c default mean | M4J_ES e_shape=0.05 | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | conservative default |
| P7B near-miss | P7B_DHEAD_RA010 | 18.085 | 19.847 | 17.965 | reference only |

Decision: **stop DGB branch and consolidate**. DGB generated useful diagnostics, but no Phase2/P2B/P2C/P2D/P2E variant passed the joint UEFB + real + synthetic promotion gate against P3c/P7B references. Do not continue DGB scalar/gate/router sweeps or multiseed promotion. Keep P3c M4J_ES e_shape=0.05 as the conservative validated default; keep P2B Gauge-only and P2E gp002 as diagnostic references only.

Artifacts: `results/hermes_audit/reports/DGB_BRANCH_STOP_CONSOLIDATION_REPORT.md`, `results/tables/dgb_branch_consolidation_summary.{csv,json}`, and compact bundle `results/hermes_audit/checkpoints/dgb_branch_stopped_20260704.tar.gz`.
