# P1 Formal 1000-step Gauge Replication Report

生成时间：2026-07-01T08:21:38
项目根目录：`/home/user/a1_rlef_project`
执行脚本：`scripts/run_p1_formal.py`
结果表：`results/tables/p1_formal_1000_summary.csv` / `.json`

## 1. Protocol

本轮执行指导文件的 P1 formal：在新 RLEF v2 scaffold 中复核 gauge 相关关键现象，而不是直接做 SOTA claim。

- 数据：LOL-v2-real 与 LOL-v2-synthetic。
- 训练：full train，不再使用 P1-light 的 64 张子集。
- 测试：full test，评估时 `crop_size=None`，不再使用 12 张验证子集。
- 每个 run：1000 steps, seed=3407, train crop=128, batch size=8。
- 并行：2 GPUs，real/synthetic × nogauge/fixed0p02/adaptive 共 6 runs。
- 主指标：PSNR/SSIM/LEE/NAI/identity_drop/Q_ECE。
- 注意：这是 single-seed formal replication，不是 3-seed paper table。

## 2. Executed results

| Dataset | Mode | PSNR↑ | SSIM↑ | LEE↓ | NAI↓ | Input PSNR | Identity drop↑ | Q_ECE↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| real | adaptive | 19.050 | 0.7572 | 0.256 | 2.170 | 9.718 | 9.332 | 0.571 |
| real | fixed0p02 | 18.734 | 0.7617 | 0.270 | 2.321 | 9.718 | 9.016 | 0.598 |
| real | nogauge | 18.748 | 0.7647 | 0.268 | 2.474 | 9.718 | 9.030 | 0.586 |
| synthetic | adaptive | 20.269 | 0.8431 | 0.211 | 1.871 | 11.221 | 9.048 | 0.541 |
| synthetic | fixed0p02 | 19.906 | 0.8455 | 0.224 | 1.945 | 11.221 | 8.685 | 0.529 |
| synthetic | nogauge | 20.116 | 0.8523 | 0.205 | 1.972 | 11.221 | 8.895 | 0.541 |

## 3. Delta analysis

### LOL-v2-real

- `fixed0p02` vs `nogauge`: PSNR -0.015 dB, SSIM -0.0031, LEE +0.003, NAI -0.152.
- `adaptive` vs `nogauge`: PSNR +0.302 dB, SSIM -0.0075, LEE -0.012, NAI -0.304.

Interpretation:

- 旧过程中的“fixed e_mean 在 real 上显著提升”没有在本轮 1000-step 新 scaffold 中复现：fixed0p02 与 nogauge 基本持平且略低。
- adaptive gauge 是 real 中 PSNR 最优，且 LEE/NAI 更低；说明“可学习/输入自适应 gauge”比固定常数 anchor 更值得保留。
- nogauge 的 SSIM 略高，但 PSNR/LEE/NAI 三项支持 adaptive 作为下一轮候选。

### LOL-v2-synthetic

- `fixed0p02` vs `nogauge`: PSNR -0.210 dB, SSIM -0.0068, LEE +0.018, NAI -0.027.
- `adaptive` vs `nogauge`: PSNR +0.154 dB, SSIM -0.0092, LEE +0.006, NAI -0.100.

Interpretation:

- fixed0p02 在 synthetic 上相对 nogauge 下降 0.210 dB，正式复现了“固定 anchor 在 synthetic 上存在脆弱性/负迁移”的方向。
- adaptive gauge 在 synthetic 上 PSNR 最高，且 NAI 最低；但 SSIM/LEE 不是最优，说明它改善了整体 fidelity/噪声放大，但空间局部曝光误差仍需 P2/P3 用 UEFB 和 E-field 指标进一步拆解。

## 4. Claim decision after P1 formal

| Claim | P1 formal status | Decision |
|---|---|---|
| fixed `e_mean=0.02` 在 real 上显著提升 | 未复现 | 不再把 fixed anchor 写成主贡献；保留为脆弱 baseline |
| fixed anchor 在 synthetic 上有 domain tension | 支持 | 可写为固定 gauge 的 domain-dependent fragility |
| adaptive gauge 比 fixed gauge 更稳 | single-seed 支持 | 进入 P3/M3-M5 正式消融 |
| RLEF 已接近 Retinexformer / SOTA | 未验证 | 不能写，需 P4 official baseline |
| E-field spatial correctness 已验证 | 未验证 | 需 UEFB-v2 E_MAE/E_aligned/E_corr |

## 5. Next execution step

根据指导文件，下一步不是调更多固定 anchor，而是进入：

1. **P2 formal UEFB-v2**：生成 3000 train / 500 test，保存 `E_gt/A_gt/Q_gt/meta.json`。
2. **P3 formal M0-M5**：用 UEFB + LOL-v2-real/synthetic 跑 1000-step single-seed ablation，验证 adaptive gauge、A gate、Q branch 是否在 E 指标/identity_drop/Q_ECE 上成立。
3. 若 P3 支持，再跑 3 seeds；否则按 pivot 规则降级 claim。

## 6. Repro commands

```bash
cd /home/user/a1_rlef_project
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_p1_formal.py   --max_steps 1000 --train_crop 128 --batch_size 8 --parallel 2
```

Logs:

```text
logs/p1_formal/p1formal_real_nogauge.log
logs/p1_formal/p1formal_real_fixed0p02.log
logs/p1_formal/p1formal_real_adaptive.log
logs/p1_formal/p1formal_synthetic_nogauge.log
logs/p1_formal/p1formal_synthetic_fixed0p02.log
logs/p1_formal/p1formal_synthetic_adaptive.log
```
