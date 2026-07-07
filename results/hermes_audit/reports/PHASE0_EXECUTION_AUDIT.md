# PHASE0 Execution Audit

Generated: 2026-07-03T00:36:56.818658+00:00  
Audited files:

- `/home/user/a1_rlef_project/results/hermes_audit/reports/PHASE0_BASELINE_REPRODUCTION.md`
- `/home/user/a1_rlef_project/results/hermes_audit/tables/phase0_baselines_summary.csv`
- `/home/user/a1_rlef_project/results/hermes_audit/claim_ledgers/phase0_reproduction_ledger.md`
- `/home/user/a1_rlef_project/results/hermes_audit/logs/phase0_result.json`
- `/home/user/a1_rlef_project/results/hermes_audit/logs/phase0_discovery.json`

## 1. Audit verdict

**Verdict: Phase 0 is valid and can be used to guide Phase 1.**

Reasons:

1. Required Phase 0 outputs exist and are parseable.
2. `phase0_result.json` reports `strict_target_ok=true`, `all_prompt_ok=true`, `anomaly_count=0`, and `phase1_executed=false`.
3. `phase0_baselines_summary.csv` contains no explicit reproduction failures.
4. P3c is correctly judged by the 3-seed aggregate mean, not by individual seed rows.
5. P6, P7_MS_DHEAD, P7B_DHEAD_RA010, Retinexformer, and Zero-DCE++ reproduce within the configured thresholds.
6. Current full test suite still passes: `72 passed`.

## 2. Reproduction table

| Baseline | UEFB PSNR | UEFB E_corr | Real PSNR | Real E_corr | Synthetic PSNR | Synthetic E_corr | Repro |
|---|---:|---:|---:|---:|---:|---:|---|
| P3c M4J_ES e=0.05 mean | 17.915 | 0.436 | 20.021 | 0.707 | 17.678 | 0.841 | PASS |
| P6_MS_M4J_ES | 18.015 | 0.464 | 20.197 | 0.629 | 17.598 | 0.812 | PASS |
| P7_MS_DHEAD | 18.232 | 0.449 | 19.209 | 0.706 | 17.913 | 0.827 | PASS |
| P7B_DHEAD_RA010 | 18.085 | 0.438 | 19.847 | 0.729 | 17.965 | 0.841 | PASS |
| Retinexformer official blind | — | — | 22.794 | — | 25.669 | — | PASS |
| Zero-DCE++ official | — | — | 18.491 | — | 17.576 | — | PASS |

## 3. Artifact integrity checks

| Check | Result |
|---|---:|
| summary rows | 25 |
| explicit fail rows | 0 |
| P3c seed trace-only rows | 9 |
| RLEF per-image rows | 4200 |
| phase1_executed | False |
| anomaly_count | 0 |
| project_root | `/home/user/a1_rlef_project` |
| git_head | `b6d24a72347af86fca885bed9124be759552b5f8` |
| torch/cuda | `2.5.1+cu124` / `12.4` |
| GPU | `0, NVIDIA GeForce RTX 4090, 529 MiB, 23553 MiB; 1, NVIDIA GeForce RTX 4090, 18 MiB, 24064 MiB` |

## 4. Protocol audit

### P3c aggregate handling

The initial Phase 0 script re-evaluated all P3c seeds and retained seed rows. Individual P3c seeds vary widely, especially on synthetic, but the prompt/A1_QA expectation is explicitly `mean±std`. The corrected artifact therefore uses:

```text
P3c aggregate mean vs expected mean
```

This is the correct comparison. The individual seed rows are retained only as traceability rows with `pass_reproduction=seed_trace_only`.

### Official baselines

Retinexformer and Zero-DCE++ were not retrained. Existing official output folders were re-evaluated with the local evaluator. This is acceptable for Phase 0 baseline freezing because the goal is to verify that current outputs match recorded expectations.

KinD++ exists as context only and is high-assisted. It should not be used as a blind fair baseline in Phase 1 claim decisions.

### Boundary compliance

Phase 0 did **not** implement DGB-RLEF. No Phase 1 method modules were created. `configs/dgb_rlef/` exists as an empty/placeholder target directory only.

## 5. Minor caveats / cleanup notes

These are **not blockers** for Phase 1:

1. `results/hermes_audit/logs/__pycache__/` is an execution byproduct; ignore it in reports/commits.
2. Official-baseline per-image CSVs are still under `results/tables/p4_details/`, not merged into `phase0_per_image_metrics.csv`. For Phase 1 comparison plots, use both sources or create a later unified per-image table.
3. The working tree is intentionally dirty from the ongoing experiment chain. No commit/push was made. Before risky long-running Phase 2+ experiments, ask the user before creating a local checkpoint commit/tag.

## 6. Phase 1 entry decision

Phase 1 is allowed **as implementation-only work with strict TDD**, because Phase 0 is clean.

However, Phase 1 should be constrained by these gates:

- Do not start Phase 2 training during Phase 1.
- Do not modify frozen Phase 0 configs/checkpoints.
- Do not introduce teacher distillation, `rec_by_dataset`, or larger real anchors.
- Preserve P3c/P6/P7B reproducibility contracts.
- Add unit tests before production code.
