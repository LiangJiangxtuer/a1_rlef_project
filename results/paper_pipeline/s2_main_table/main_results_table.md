# S2 Manuscript-Grade Main Table

This table intentionally keeps the main paper compact. Distillation, scalar controls, domain-head near-misses, and DGB diagnostics are moved to S4 appendix artifacts.

| method | role | evidence_level | uefb_psnr | real_psnr | synthetic_psnr | uefb_E_corr | real_E_corr | synthetic_E_corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 restorer_only | minimal restorer baseline | single-seed ablation | 17.379 | 15.306 | 12.564 | 0.000 | — | — |
| M4 A-gate | strong UEFB mechanism | single-seed ablation | 18.653 | 17.411 | 13.963 | -0.139 | — | — |
| M4J joint | joint-training bridge | single-seed route evidence | 18.082 | 19.581 | 16.508 | 0.282 | 0.501 | 0.369 |
| M4J_ES seed3407 | E-shape route evidence | single-seed route evidence | 17.981 | 20.277 | 17.141 | 0.440 | 0.648 | 0.799 |
| P3c M4J_ES e_shape=0.05 | conservative default | 3-seed default | 17.915 | 20.021 | 17.678 | 0.436 | 0.707 | 0.841 |
| Retinexformer | official baseline | official-code eval | — | 22.794 | 25.669 | — | — | — |
| Zero-DCE++ | official baseline | official-code eval | — | 18.491 | 17.576 | — | — | — |
| KinD++ | official baseline | official-code eval | — | 22.211 | 19.259 | — | — | — |

## Table-use rule

Use this as the main experimental table. Do not mix stopped DGB/P5/P6/P7 scalar-route rows into the main text unless discussing negative diagnostics.
