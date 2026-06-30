# A1-RLEF v2 实验过程报告（已执行证据版）

生成时间：2026-07-01T01:15:58  
项目根目录：`/home/user/a1_rlef_project`  
源指导文件：`docs/source/A1_RLEF_v2_experiment_guidance_after_process_audit_20260701.md`

## 1. 已完成工作

本轮不是只写计划，已经完成了一个新的 RLEF v2 研究工程 scaffold，并执行了可验证的 P0/P1 轻量实验闭环。

已完成：

- 新项目目录：`/home/user/a1_rlef_project`
- 完整中文执行计划：`docs/EXECUTION_PLAN_RLEF_V2_ZH.md`
- 源指导文件归档：`docs/source/A1_RLEF_v2_experiment_guidance_after_process_audit_20260701.md`
- 数据软链接：`data -> /home/user/a1_local_exposure_field_project/data`
- TDD 单测：metrics、UEFB 生成、RLEF 输出契约、loss 标量、训练 smoke
- UEFB-v2 smoke 生成脚本：`scripts/make_uefb_v2.py`
- RLEF-Former MVP：`src/rlef/models/rlef_former.py`
- 训练/评估脚本：`scripts/train.py`, `scripts/eval_uefb.py`, `scripts/eval_paired.py`
- P0 smoke ablation：M0/M3/M4/M5，每个 30 steps
- P1 light replication：LOL-v2-real/synthetic，nogauge vs fixed0p02，每个 30 steps

## 2. 环境与数据核验

- GPU：RTX 4090 × 2
- Python：`/home/user/miniconda3/envs/cutler_dinov3/bin/python`
- PyTorch：`2.5.1+cu124`
- 数据就绪：LOL-v1、LOL-v2-real、LOL-v2-synthetic、unpaired_real 均可访问。
- 重要限制：`cv2` 在当前环境有 libpng 符号冲突，因此新代码不依赖 OpenCV。

## 3. 测试结果

```text
pytest tests -q
........ [100%]
8 passed
```

## 4. P0：UEFB-smoke + RLEF-M0/M3/M4/M5 轻量 ablation

UEFB-smoke：从 `LOL-v2/Real_captured/Train/Normal` 抽样生成 20 train / 20 test，image size 96。

结果表：`results/tables/p0_smoke_summary.csv`

| run | PSNR | SSIM | LEE↓ | NAI↓ | E_MAE↓ | E_aligned↓ | E_corr↑ | identity_drop↑ | Q_ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rlef_m0_restorer_smoke | 15.928 | 0.636 | 0.263 | 2.403 | 1.289 | 0.244 | 0.000 | 4.571 | 0.427 |
| rlef_m3_adaptive_gauge_smoke | 15.215 | 0.527 | 0.280 | 2.230 | 0.610 | 0.493 | -0.287 | 3.858 | 0.431 |
| rlef_m4_gate_smoke | 15.478 | 0.595 | 0.275 | 1.447 | 0.638 | 0.567 | -0.411 | 4.121 | 0.430 |
| rlef_m5_recoverability_smoke | 15.468 | 0.596 | 0.277 | 1.455 | 0.636 | 0.565 | -0.413 | 4.110 | 0.214 |

### P0 解析

- 当前 30-step smoke 中，图像质量最高的是 `rlef_m0_restorer_smoke`，PSNR=15.928。
- M3/M4/M5 的 exposure branch 明显降低 absolute `E_MAE`（约 0.61-0.64 vs M0 的 1.289），说明 gauge/Poisson 分支开始学习 E 的全局量级。
- 但 M3-M5 的 `E_corr` 仍为负，说明空间形状还没学好；这符合指导文件的警告：E 指标必须分解为 absolute/gauge/spatial，不可只看 MAE。
- M4/M5 显著降低 NAI（约 1.45 vs M0 的 2.40），说明 gate/recoverability 分支在 smoke 阶段已有降低噪声放大的迹象。
- M5 的 Q_ECE=0.214，低于 M0/M3/M4，说明 Q branch 在 smoke 条件下改善了 recoverability calibration；但这只是 smoke evidence，不能写成正式 claim。

## 5. P1：旧过程关键现象轻量复现

P1-light 只跑 30 steps、64 张训练图、12 张验证图，用于检查代码路径，不等价于旧项目 1000/3000-step 证据。

结果表：`results/tables/p1_light_replication_summary.csv`

| dataset | mode | PSNR | SSIM | LEE↓ | NAI↓ | input PSNR | identity_drop |
|---|---|---:|---:|---:|---:|---:|---:|
| real | nogauge | 15.602 | 0.688 | 0.284 | 3.750 | 7.015 | 8.587 |
| real | fixed0p02 | 17.176 | 0.699 | 0.231 | 3.782 | 7.015 | 10.161 |
| synthetic | nogauge | 14.318 | 0.533 | 0.338 | 2.134 | 11.565 | 2.753 |
| synthetic | fixed0p02 | 14.732 | 0.540 | 0.299 | 2.178 | 11.565 | 3.167 |

### P1-light 解析

- 在 real light subset 中，fixed0p02 比 nogauge 高 1.574 dB，方向与旧过程“fixed gauge 提升 real paired”一致。
- 在 synthetic light subset 中，fixed0p02 比 nogauge 高 0.414 dB；这**没有复现**旧过程中的 synthetic anchor hurt。
- 该差异不能解释为新结论，因为 P1-light 只有 30 steps/小子集/新 RLEF scaffold。正式 P1 必须按计划跑 1000-3000 steps/full test，再决定 domain tension 是否仍成立。

## 6. 当前 claim ledger

| Claim | 当前状态 | 证据 |
|---|---|---|
| 新项目可运行、可训练、可评估 | 已执行支持 | P0/P1-light、8 tests passed |
| UEFB-smoke 可生成 E/A/Q GT 并被 dataloader 消费 | 已执行支持 | `data/UEFB-v2-smoke`, tests |
| adaptive gauge / Poisson 分支能降低 absolute E_MAE | smoke 支持 | P0 M3/M4/M5 E_MAE 低于 M0 |
| E spatial shape 已解决 | 不支持 | P0 E_corr 为负 |
| recoverability Q 改善 Q_ECE | smoke 支持，需正式验证 | P0 M5 Q_ECE 最低 |
| fixed gauge real 提升 | light 支持，正式待验证 | P1 real +1.574 dB |
| fixed gauge synthetic 退化/domain tension | 本轮 light 未复现，旧过程支持 | 本轮 synthetic +0.414 dB；必须跑正式 P1 |
| RLEF-Former 接近/超过 Retinexformer | 未验证 | P4 pending |

## 7. 关键产物路径

```text
docs/EXECUTION_PLAN_RLEF_V2_ZH.md
src/rlef/models/rlef_former.py
src/rlef/losses/total_loss.py
scripts/make_uefb_v2.py
scripts/train.py
scripts/run_p0_smoke.py
scripts/run_p1_light_replication.py
results/tables/p0_smoke_summary.csv
results/tables/p1_light_replication_summary.csv
experiments/p0_rlef_m5_recoverability_smoke_seed3407/visuals/
```

## 8. 下一步必须执行的正式任务

1. P1 formal：对 real/synthetic nogauge/fixed/adaptive 跑至少 1000 steps，full test，不再用 12 张子集。
2. P2 formal：生成 UEFB-v2 formal train/test，至少 3000/500。
3. P3 formal：M0-M5 1000-step single seed，按 Go/No-Go 淘汰模块。
4. 保留模块后跑 3 seeds。
5. 接入 Retinexformer/Zero-DCE++/KinD++ 同协议。

## 9. 本轮结论

RLEF v2 已经从指导文档变成可运行项目，并完成最小真实闭环。当前结果只能作为 smoke evidence：支持继续推进 adaptive gauge / recoverability gate 方向，但不支持任何 SOTA 或正式论文主表 claim。下一步应优先跑 P1/P3 formal，而不是继续添加新模块。
