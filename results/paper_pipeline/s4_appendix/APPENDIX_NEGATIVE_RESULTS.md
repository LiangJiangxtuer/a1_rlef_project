# S4 Appendix: Negative Routes and Diagnostic Findings

Generated: 2026-07-04T16:22:51.270814+00:00

These rows should be used as appendix evidence / negative-route diagnostics, not as main-method claims.

| route | best_reference | best_snapshot | decision | reason | n_runs |
| --- | --- | --- | --- | --- | --- |
| P5/P5b distillation | P5_RD_T03 | UEFB=18.169; real=20.416; synthetic=16.996 | appendix only | output-level teacher distillation shifts real/synthetic trade-off and is not a stable deployable mainline | 3 |
| P6/P6b/P6c structural scalar controls | P6_MS_M4J_ES / P6B_MS_SYN125 / P6C_MS_GATEID005 | P6 real=20.197; P6b synthetic=19.115; P6c real=20.456 | appendix only | structural backbone is useful, but scalar synthetic/gate protection over-corrects and does not solve three-domain trade-off | 9 |
| P7/P7b/P7c domain heads | P7B_DHEAD_RA010 | UEFB=18.085; real=19.847; synthetic=17.965 | near-miss reference only | P7B_RA010 is closest but misses real P3c mean; stronger P7c anchors degrade | 9 |
| DGB branch | P2B_DGB_GAUGE_ONLY | best real=20.432; best synthetic=18.683; joint_gate_any=False | stopped and consolidated | joint gate failed across Phase2/P2B/P2C/P2D/P2E; routing changes did not recover real-domain quality | 11 |

## Appendix writing policy

- Present these as decisions that prevented overclaiming.
- Emphasize diagnostic learning, not failure.
- Do not revive scalar sweeps unless a new non-DGB hypothesis and validation gate are explicitly defined.
