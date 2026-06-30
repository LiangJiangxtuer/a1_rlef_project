# A1-RLEF v2 完整执行计划（Linux 落地版）

> 源指导文件：`docs/source/A1_RLEF_v2_experiment_guidance_after_process_audit_20260701.md`  
> 新项目根目录：`/home/user/a1_rlef_project`  
> 旧证据项目：`/home/user/a1_local_exposure_field_project`  
> 当前执行原则：所有数值必须来自真实运行；长周期 benchmark 未完成前只能写为 pending protocol，不伪造结果。

## 0. 当前环境与边界

- 运行环境：Ubuntu/Linux，RTX 4090 × 2，CUDA 可用。
- Python 环境：复用 `/home/user/miniconda3/envs/cutler_dinov3/bin/python`，已具备 `torch 2.5.1+cu124`、`torchvision`、`PIL`、`skimage`、`yaml`。
- 注意：`cv2` 在该环境中有 libpng 符号冲突，因此 v2 项目不依赖 OpenCV，图像 I/O 全部用 PIL / torchvision / skimage。
- 数据策略：新项目通过软链接 `data -> /home/user/a1_local_exposure_field_project/data` 复用已验证 LOL-v1、LOL-v2、unpaired_real 数据，不复制大数据。

## 1. 研究主张收缩

本轮不再把 A1 写成“大而全物理成像系统”。主线改为：

> 强 restoration backbone + gauge-normalized local exposure field + recoverability-aware gate + UEFB-v2 专项评价。

可写 claim：

1. Poisson/梯度域 exposure-field 存在 additive gauge ambiguity，必须报告/修复 gauge。
2. adaptive gauge 比 fixed `e_mean` 更适合处理 real/synthetic domain tension。
3. recoverability-aware gate 用于保护 near-identity / 不可恢复区域，降低过增强。
4. exposure field 作为强 restoration backbone 的辅助解释/正则，而不是 tiny Retinex 主路径独自追 SOTA。

不可写 claim：

- 当前不声称 RLEF 已经 SOTA；
- 不把 CRF/noise/U 作为 v2 主贡献；
- 不把未跑完的多 seed / SOTA baseline 表写成结论。

## 2. 工程结构

```text
/home/user/a1_rlef_project/
  configs/
  src/rlef/{datasets,models,losses,metrics,utils}/
  scripts/
  tests/
  docs/
  results/tables/
  experiments/
```

## 3. 阶段计划与执行顺序

### P0：环境与管线 smoke（本轮必须真实执行）

1. 生成 UEFB-smoke：从 LOL-v2-real normal 中抽 20/20 张生成 low/high/E_gt/A_gt/Q_gt/meta。
2. TDD 单测：metrics、UEFB dataset、RLEF 输出契约、loss 标量、train/eval smoke。
3. 训练 M0/M3/M4/M5 smoke 每个 20-50 steps。
4. 评估 UEFB-smoke，保存 PSNR/SSIM/E_MAE/E_corr/identity_drop/Q_ECE。
5. 保存可视化：`input | output | gt | E | A | Q`。

验收：`pytest tests -q` 全绿；`results/tables/p0_smoke_summary.csv/json` 存在；训练 loss 非 NaN；输出图像非空；每个 run 保存 `run_meta.json`。

### P1：复现旧过程关键现象（轻量版先执行，可后续加长）

| ID | 设置 | 数据 | 轻量执行 | 正式执行 |
|---|---|---|---|---|
| P1-a | no-gauge | LOL-v2-real | 100-300 steps | 3000 steps × 3 seeds |
| P1-b | fixed gauge/e_mean | LOL-v2-real | 100-300 steps | 3000 steps × 3 seeds |
| P1-c | no-gauge | LOL-v2-synthetic | 100-300 steps | 3000 steps × 3 seeds |
| P1-d | fixed gauge/e_mean | LOL-v2-synthetic | 100-300 steps | 3000 steps × 3 seeds |

### P2：UEFB-v2 正式 benchmark

- smoke：20/20，本轮执行；formal：train 3000、val 300、test 500，固定 seed 3407；分 easy/hard/identity。
- 指标：E_MAE、E_MAE_aligned、E_corr、A_AUC、Q_ECE、identity_drop、SHI。

### P3：RLEF-Former MVP 消融

| ID | Variant | 目的 | Go/No-Go |
|---|---|---|---|
| M0 | Restorer only | 强主干 baseline | 所有模块比较基准 |
| M1 | + physics aux | 保持物理分支不退化 | PSNR 不明显降，E 可视化可用 |
| M2 | + Poisson | 检查 E spatial shape | E_aligned/E_corr 改善 |
| M3 | + adaptive gauge | 核心 gauge claim | real/synthetic 更稳 |
| M4 | + A gate | 防 near-identity 过增强 | identity_drop 改善 |
| M5 | + Q recoverability | failure awareness | Q_ECE/AURC 优于 rule mask |

### P4-P7：正式论文实验（长周期）

- P4：Retinexformer/Zero-DCE++/KinD++ 同协议 full-resolution 对比；
- P5：LIME/NPE/MEF/DICM/VV unpaired real 可视化和 no-ref 指标；
- P6：3 seeds (`3407, 2027, 42`) mean±std + 显著性检验；
- P7：CRF/noise/RAW 仅在主线稳定后作为扩展。

## 4. 文件与日志规范

每个 run 保存 `config.yml`、`run_meta.json`、`checkpoints/last.pth`、`metrics/train_log.csv`、`metrics/eval_metrics.json`、`visuals/*.png`。

## 5. 本轮真实执行目标

完成新项目 scaffold、完整计划文档、TDD 单测与核心代码、UEFB-smoke 生成、P0 smoke 训练/评估/可视化、P3 轻量 ablation、过程报告、claim ledger、Git commit/tag。若 GitHub auth 不可用，生成归档和 push 指南。
