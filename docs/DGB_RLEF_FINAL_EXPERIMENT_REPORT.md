# DGB-RLEF Final Experiment Report

Generated: 2026-07-05T03:06:24.533523+00:00

## Executive decision

```text
DGB branch: stopped and consolidated.
Current default: P3c M4J_ES e_shape=0.05.
Paper framing: interpretable, gauge-aware exposure-field auxiliary calibration, not SOTA restoration.
```

## Background problem

The original research question asked whether local exposure-field modeling can improve low-light image enhancement. The executed evidence shows that the main bottleneck is not merely adding a physical branch, but separating gauge-invariant exposure shape from absolute exposure gauge and avoiding domain trade-offs across UEFB, LOL-v2-real, and LOL-v2-synthetic.

## Main results

| method | role | evidence_level | uefb_psnr | real_psnr | synthetic_psnr | uefb_E_corr | real_E_corr | synthetic_E_corr | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 restorer_only | minimal restorer baseline | single-seed ablation | 17.379 | 15.306 | 12.564 | 0.000 | — | — | mainline ablation / route evidence |
| M4 A-gate | strong UEFB mechanism | single-seed ablation | 18.653 | 17.411 | 13.963 | -0.139 | — | — | mainline ablation / route evidence |
| M4J joint | joint-training bridge | single-seed route evidence | 18.082 | 19.581 | 16.508 | 0.282 | 0.501 | 0.369 | mainline ablation / route evidence |
| M4J_ES seed3407 | E-shape route evidence | single-seed route evidence | 17.981 | 20.277 | 17.141 | 0.440 | 0.648 | 0.799 | mainline ablation / route evidence |
| P3c M4J_ES e_shape=0.05 | conservative default | 3-seed default | 17.915 | 20.021 | 17.678 | 0.436 | 0.707 | 0.841 | current conservative default |
| Retinexformer | official baseline | official-code eval | — | 22.794 | 25.669 | — | — | — | official baseline comparison |
| Zero-DCE++ | official baseline | official-code eval | — | 18.491 | 17.576 | — | — | — | official baseline comparison |
| KinD++ | official baseline | official-code eval | — | 22.211 | 19.259 | — | — | — | official baseline comparison |
| DGB branch | diagnostic branch | stopped consolidation | 17.863 | 20.432 | 18.683 | — | — | — | stopped; diagnostic only |

Key boundary: Retinexformer remains much stronger on paired fidelity (`real=22.794`, `synthetic=25.669`), while P3c remains the conservative RLEF default (`real=20.021`, `synthetic=17.678`). DGB did not pass the joint gate and is kept as diagnostic evidence only.

## Ablation and negative-route evidence

| group | variant | key_result | decision | paper_location |
| --- | --- | --- | --- | --- |
| Gauge / E-shape | P3c e_shape=0.05 | UEFB=17.915; real=20.021; synthetic=17.678 | default | main ablation |
| Gauge / E-shape | P3d e_shape=0.10 | UEFB=18.111; real=19.875; synthetic=17.626 | ablation only | main ablation |
| Distillation | P5_RD_T03 | UEFB=18.169; real=20.416; synthetic=16.996 | appendix only | appendix negative evidence |
| Structural backbone | P6_MS_M4J_ES / P6B_MS_SYN125 / P6C_MS_GATEID005 | P6 real=20.197; P6b synthetic=19.115; P6c real=20.456 | appendix only | appendix negative evidence |
| Domain heads | P7B_DHEAD_RA010 | UEFB=18.085; real=19.847; synthetic=17.965 | near-miss reference only | appendix negative evidence |
| DGB controlled isolation | P2B_DGB_GAUGE_ONLY | best real=20.432; best synthetic=18.683; joint_gate_any=False | stopped and consolidated | appendix negative evidence |

## No-reference supplementary audit

The no-reference supplement evaluated frozen P3c outputs on local unpaired real images. It is stored at `results/hermes_audit/reports/P2_NOREF_SUPPLEMENTARY_EVAL.md`. BRISQUE was unavailable locally and is not fabricated.

## Failure cases and limitations

- DGB/P2B-P2E changed route statistics but never solved joint UEFB/real/synthetic promotion.
- Retinexformer remains the fidelity ceiling in this project; any paper claim must avoid SOTA wording.
- Q maps in the frozen P3c visual panels are inactive/constant because the default checkpoint does not use the Q branch as a main claim.
- No-reference metrics are auxiliary and cannot override paired PSNR/SSIM evidence.

## Claim ledger

### Verified claims

1. P3c M4J_ES e_shape=0.05 is the validated conservative default with positive exposure-shape correlations.
2. Retinexformer is substantially stronger on paired real/synthetic fidelity.
3. DGB branch is stopped; its controlled-isolation results are diagnostic negative evidence.
4. S1-S4 paper evidence pipeline and final supplementary artifacts are reproducible from local files.

### Rejected claims

1. DGB-RLEF is a SOTA LLIE method.
2. DGB branch should enter 3-seed or full schedule.
3. Teacher distillation, dataset-weighted reconstruction, or larger real anchors are final main innovations.

## Publication value

The defensible paper idea is a mechanism/benchmark/analysis paper around gauge-aware exposure-field auxiliary calibration, UEFB-v2 diagnostics, and honest negative-route evidence. It is not yet a method/SOTA paper.

## Next strengthening route

Only after this evidence package is reviewed should a new non-DGB route be opened: either RLEF-as-auxiliary on a strong restoration backbone, or a Retinex-factorization redesign. Both require fresh gates and must not reuse stopped DGB scalar sweeps.
