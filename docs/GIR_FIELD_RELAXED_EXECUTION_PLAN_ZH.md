# GIR-Field relaxed pipeline 执行计划

来源指导：`/home/user/下载/A1_after_master_prompt_QA_experiment_audit_next_top_tier_innovations_20260706.md`

## 放松设置

- 不训练新的主干模型：`True`
- 不恢复 DGB/P2F：`True`
- 使用 frozen checkpoints 与已有 official-baseline outputs。
- 每个 split 最多评估 `64` 张图，用于 relaxed routing evidence。
- bootstrap samples: `1000`。
- LPIPS/BRISQUE 若本地包不可用则不报告，禁止伪造。

## Pipeline

1. N0: 证据清洗、source guidance 冻结、claim ledger。
2. N1: M4/M4J/M4J_ES/P3c/P3d per-image 统计复核。
3. N2: UEFB-G gauge perturbation，验证 gauge shift 与 shape distortion 可区分。
4. N3: 外部基线 registry，区分 black-box output-only 与 internal-field metrics。
5. N4: recoverability risk calibration probe；轻量 logistic risk probe，不改主干。
6. N5: 论文主表、图、报告、manifest。

## Claim 边界

这不是 SOTA 方法实验，也不是 DGB 复活实验。输出只支持 GIR-Field/UEFB-G 机制与 benchmark 的 relaxed evidence。
