# PHASE0 BASELINE REPRODUCTION

Generated: 2026-07-02T16:45:01.235472+00:00  
Project root: `/home/user/a1_rlef_project`  
Master prompt: `/home/user/下载/prompt.txt`  
A1 QA guidance: `/home/user/下载/A1_QA_audit_DGB_RLEF_experiment_guidance_20260702.md`

## Scope

Executed **Phase 0 only**: reproduce and freeze the current baselines. No new DGB-RLEF method was implemented, no new training was launched, and no Phase 1 code was added.

User-targeted baselines: P3c, P6, P7B_DHEAD_RA010. Prompt/A1_QA Phase 0 also lists P7_MS_DHEAD and official Retinexformer/Zero-DCE++ baselines, so they were included as audit rows where existing outputs/checkpoints were available.

## Environment and project discovery

- Python env used: `/home/user/miniconda3/envs/cutler_dinov3/bin/python`
- Runtime Python: `3.11.15`
- PyTorch: `2.5.1+cu124` / CUDA `12.4` / cuda_available `True`
- Device: `cuda`
- Git HEAD: `b6d24a72347af86fca885bed9124be759552b5f8`
- Git status: see `results/hermes_audit/logs/phase0_discovery.json`
- GPU: `0, NVIDIA GeForce RTX 4090, 529 MiB, 23553 MiB; 1, NVIDIA GeForce RTX 4090, 18 MiB, 24064 MiB`

## Data paths

| Dataset key | Path | Exists | Image count |
|---|---|---:|---:|
| uefb | `/home/user/a1_rlef_project/data/UEFB-v2/test` | True | 2500 |
| real_low | `/home/user/a1_rlef_project/data/LOL-v2/Real_captured/Test/Low` | True | 100 |
| real_high | `/home/user/a1_rlef_project/data/LOL-v2/Real_captured/Test/Normal` | True | 100 |
| synthetic_low | `/home/user/a1_rlef_project/data/LOL-v2/Synthetic/Test/Low` | True | 100 |
| synthetic_high | `/home/user/a1_rlef_project/data/LOL-v2/Synthetic/Test/Normal` | True | 100 |
| lol_v1 | `/home/user/a1_rlef_project/data/LOL-v1` | True | 1000 |

## Configs and checkpoints found

| Baseline | Config/checkpoint evidence |
|---|---|
| P3c seed3407 | `/home/user/a1_rlef_project/configs/p3c_multiseed_sweep/p3c_m4j_es_seed3407_e0050.yml` + `/home/user/a1_rlef_project/experiments/p3b_m4j_es_m4_joint_eshape_seed3407/checkpoints/last.pth` |
| P3c seed2027 | `/home/user/a1_rlef_project/configs/p3c_multiseed_sweep/p3c_m4j_es_seed2027_e0050.yml` + `/home/user/a1_rlef_project/experiments/p3c_m4j_es_seed2027_e0050/checkpoints/last.pth` |
| P3c seed42 | `/home/user/a1_rlef_project/configs/p3c_multiseed_sweep/p3c_m4j_es_seed42_e0050.yml` + `/home/user/a1_rlef_project/experiments/p3c_m4j_es_seed42_e0050/checkpoints/last.pth` |
| P6 | `/home/user/a1_rlef_project/configs/p6_structural_backbone/p6_p6_ms_m4j_es_m4j_eshape_multiscale_backbone.yml` + `/home/user/a1_rlef_project/experiments/p6_p6_ms_m4j_es_m4j_eshape_multiscale_backbone_seed3407/checkpoints/last.pth` |
| P7_MS_DHEAD | `/home/user/a1_rlef_project/configs/p7_domainhead/p7_p7_ms_dhead_m4j_eshape_multiscale_domain_head_bias.yml` + `/home/user/a1_rlef_project/experiments/p7_p7_ms_dhead_m4j_eshape_multiscale_domain_head_bias_seed3407/checkpoints/last.pth` |
| P7B_DHEAD_RA010 | `/home/user/a1_rlef_project/configs/p7b_realanchor/p7b_p7b_dhead_ra010_m4j_eshape_multiscale_dhead_realanchor010.yml` + `/home/user/a1_rlef_project/experiments/p7b_p7b_dhead_ra010_m4j_eshape_multiscale_dhead_realanchor010_seed3407/checkpoints/last.pth` |

## Reproduction summary

| Baseline | UEFB PSNR | UEFB E_corr | Real PSNR | Real E_corr | Synthetic PSNR | Synthetic E_corr | Repro |
|---|---:|---:|---:|---:|---:|---:|---|
| P3c M4J_ES e=0.05 mean | 17.915 | 0.436 | 20.021 | 0.707 | 17.678 | 0.841 | PASS |
| P6_MS_M4J_ES | 18.015 | 0.464 | 20.197 | 0.629 | 17.598 | 0.812 | PASS |
| P7_MS_DHEAD | 18.232 | 0.449 | 19.209 | 0.706 | 17.913 | 0.827 | PASS |
| P7B_DHEAD_RA010 | 18.085 | 0.438 | 19.847 | 0.729 | 17.965 | 0.841 | PASS |
| Retinexformer official blind | — | — | 22.794 | — | 25.669 | — | PASS |
| Zero-DCE++ official | — | — | 18.491 | — | 17.576 | — | PASS |

## Pass/fail decision

- User-targeted Phase 0 baseline reproduction (P3c/P6/P7B): **PASS**
- Full prompt-listed Phase 0 rows including official baselines: **PASS**

## Threshold policy

- Deterministic PSNR absolute difference threshold: `<= 0.05 dB`.
- E_corr absolute difference threshold: `<= 0.01` where the prompt/A1_QA provided E_corr expectations.
- P3c is compared using the 3-seed aggregate mean because the guidance expectation is reported as mean±std.

## Anomaly audit

No reproduction row exceeded the Phase 0 threshold. P3c individual seed rows are retained in the CSV for traceability, but threshold comparison is correctly applied to the 3-seed aggregate mean reported by the prompt/A1_QA. No need to enter failure localization before Phase 1.

## Saved artifacts

- `/home/user/a1_rlef_project/results/hermes_audit/tables/phase0_baselines_summary.csv`
- `/home/user/a1_rlef_project/results/hermes_audit/tables/phase0_per_image_metrics.csv`
- `/home/user/a1_rlef_project/results/hermes_audit/logs/phase0_discovery.json`
- `/home/user/a1_rlef_project/results/hermes_audit/logs/phase0_anomalies.json`
- `/home/user/a1_rlef_project/results/hermes_audit/reports/PHASE0_BASELINE_REPRODUCTION.md`
- `/home/user/a1_rlef_project/results/hermes_audit/claim_ledgers/phase0_reproduction_ledger.md`

## Boundary

This report intentionally stops at Phase 0. No DGB-RLEF new module was implemented.
