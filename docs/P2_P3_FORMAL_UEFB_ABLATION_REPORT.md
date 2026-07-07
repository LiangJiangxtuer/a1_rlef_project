# P2/P3 Formal Report: UEFB-v2 generation and M0-M5 mechanism ablation

生成时间：2026-07-01T11:17:43  
项目根目录：`/home/user/a1_rlef_project`

> 注意：按用户要求，本轮没有主动提交或推送 GitHub。本报告和结果文件保留在本地工作区。

## 1. P2 formal: UEFB-v2 generation

执行命令：

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/make_uefb_v2.py   --source data/LOL-v2/Real_captured/Train/Normal   --output data/UEFB-v2   --num_train 3000 --num_test 500 --image_size 256 --seed 3407
```

生成目录：`data/UEFB-v2`  
生成后大小：约 `3.4G`

完整性校验：

| Split | low PNG | high PNG | E_gt PNG/NPY | A_gt PNG/NPY | Q_gt PNG/NPY | meta samples | near-identity | mean clip ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 3000 | 3000 | 3000/3000 | 3000/3000 | 3000/3000 | 3000 | 621 | 0.0178 |
| test | 500 | 500 | 500/500 | 500/500 | 500/500 | 500 | 117 | 0.0212 |

P2 结论：UEFB-v2 formal 数据集已生成并通过数量、meta、E/A/Q 双格式校验，可以作为 exposure-field / gate / recoverability 机制指标的专用 benchmark。

## 2. P3 formal: M0-M5 ablation protocol

执行脚本：`scripts/run_p3_formal.py`

协议：

- 训练数据：UEFB-v2 train 3000。
- 主验证：UEFB-v2 test 500，含 `E_gt/A_gt/Q_gt`。
- 额外诊断：LOL-v2-real test 与 LOL-v2-synthetic test full-resolution paired evaluation。
- 每个 variant：1000 steps，seed=3407，train crop=128，batch size=8。
- 变体：M0 restorer only, M1 phys aux, M2 Poisson, M3 adaptive gauge, M4 gate, M5 Q recoverability。
- 说明：首次运行时发现 `eval_paired.py/eval_uefb.py` 作为 CLI 执行时 import 路径缺陷，已用新增 CLI contract test 复现并修复；随后用 `--resume` 复用已完成的训练 checkpoint，补跑 paired evaluation 与 summary。

结果表：`results/tables/p3_formal_uefb_m0_m5_summary.csv`

| ID | Variant | UEFB PSNR↑ | UEFB SSIM↑ | E_MAE↓ | E_aligned↓ | E_corr↑ | identity_drop↑ | Q_ECE↓ | Real PSNR↑ | Synthetic PSNR↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | restorer_only | 17.379 | 0.7863 | 1.076 | 0.222 | 0.000 | 4.383 | 0.412 | 15.306 | 12.564 |
| M1 | phys_aux | 17.913 | 0.8084 | 1.039 | 0.611 | -0.366 | 4.917 | 0.407 | 16.396 | 13.178 |
| M2 | poisson | 17.891 | 0.8089 | 1.833 | 0.230 | -0.104 | 4.896 | 0.398 | 16.689 | 13.340 |
| M3 | adaptive_gauge | 17.670 | 0.7966 | 0.302 | 0.223 | -0.095 | 4.674 | 0.398 | 16.111 | 13.586 |
| M4 | gate | 18.653 | 0.8169 | 0.292 | 0.247 | -0.139 | 5.657 | 0.394 | 17.411 | 13.963 |
| M5 | recoverability_q | 18.535 | 0.8152 | 0.285 | 0.226 | -0.243 | 5.540 | 0.389 | 17.029 | 13.778 |

## 3. P3 findings

### 3.1 Image fidelity / paired diagnostics

- UEFB PSNR 最优：**M4 gate**, 18.653 dB。
- LOL-v2-real PSNR 最优：**M4 gate**, 17.411 dB。
- LOL-v2-synthetic PSNR 最优：**M4 gate**, 13.963 dB。
- M4 gate 相对 M0：UEFB +1.274 dB，real +2.105 dB，synthetic +1.399 dB。

解释：在 UEFB 训练路线上，**A gate 是当前最强的图像质量模块**。它同时提升 UEFB、real paired diagnostic、synthetic paired diagnostic，值得进入下一轮 3-seed 或联合训练验证。

### 3.2 Exposure-field metrics

- Absolute E_MAE 最优：**M5 recoverability_q**, E_MAE=0.285。
- M3 adaptive gauge 将 M0 的 E_MAE 1.076 降到 0.302。
- M4/M5 进一步保持低 E_MAE：M4=0.292, M5=0.285。
- 但所有 E-branch 的 `E_corr` 仍为负：M3=-0.095, M4=-0.139, M5=-0.243。

解释：adaptive gauge 明显解决了 exposure field 的**绝对量级/全局偏置**问题，但还没有解决空间形状方向问题。论文不能写“物理 E-field 空间正确”，只能写“gauge normalization improves absolute exposure-field calibration; spatial shape remains a failure mode”。

### 3.3 Recoverability / Q

- Q_ECE 最优：**M5 recoverability_q**, Q_ECE=0.389。
- M5 相比 M4：Q_ECE 0.394 → 0.389，略有改善；但 UEFB PSNR 18.653 → 18.535 下降 0.118 dB。

解释：Q branch 有校准改善迹象，但牺牲了一点图像质量。当前应把 Q 写成辅助分析/校准模块，不应写成无条件性能增益贡献。

## 4. Go/No-Go decisions

| Module | Decision | Evidence |
|---|---|---|
| Phys aux (M1) | Keep only as baseline | 改善 PSNR，但 E_corr 明显负，不能单独支撑物理 claim |
| Poisson (M2) | Keep as diagnostic | real/synthetic PSNR 高于 M1，但 absolute E_MAE 很差；说明梯度约束不能单独解决 gauge |
| Adaptive gauge (M3) | Keep | E_MAE 大幅下降，支持 gauge-normalization 方向，但 spatial E_corr 未过关 |
| A gate (M4) | Strong keep | UEFB/real/synthetic PSNR 最优，是当前最稳定正向模块 |
| Q recoverability (M5) | Conditional keep | Q_ECE 最优，但 PSNR 低于 M4；作为 auxiliary calibration 而非主性能模块 |

## 5. Next recommended experiment

不要继续调 fixed anchor。下一步建议：

1. **P3b joint-training**：以 M4 为主，训练数据从 UEFB-only 扩展到 UEFB + LOL-v2-real + LOL-v2-synthetic，检查是否能保留 UEFB E/A/Q 优势同时提高 real/synthetic paired PSNR。
2. **M4 vs M5 3-seed mini**：如果要保留 Q claim，必须确认 Q_ECE 改善在 3 seed 稳定且不会明显牺牲 PSNR。
3. **E-shape repair diagnostic**：针对 `E_corr < 0`，单独加入 low-pass/multi-scale E supervision 或 sign/gradient consistency diagnostic，不能继续只调 gauge scalar。

## 6. Repro commands

```bash
cd /home/user/a1_rlef_project
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_p3_formal.py   --max_steps 1000 --train_crop 128 --batch_size 8 --parallel 2 --resume
```

若从零训练则去掉 `--resume`。
