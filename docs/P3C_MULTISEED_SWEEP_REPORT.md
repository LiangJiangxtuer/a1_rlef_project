# P3c Formal Report: M4J_ES 3-seed validation and e_shape sweep

生成时间：2026-07-01T12:10:48  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有 GitHub 提交/推送，也没有本地 git commit。所有结果保留在工作区。

## 1. Objective

P3b 发现 `M4J_ES = M4 gate joint-training + low-pass E-shape consistency` 在 single-seed 下同时提升 paired PSNR 与 E_corr。P3c 执行两个验证：

1. **3-seed stability**：固定 `e_shape=0.05`，seeds = 3407, 2027, 42。
2. **e_shape weight sweep**：固定 seed=3407，`e_shape ∈ {0.02, 0.05, 0.10}`。

其中 seed=3407 / e_shape=0.05 复用 P3b 已完成 checkpoint 和 metrics，未重复训练。

## 2. Protocol

执行脚本：`scripts/run_p3c_multiseed_sweep.py`

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_p3c_multiseed_sweep.py   --max_steps 1000 --train_crop 128 --batch_size 8 --parallel 2
```

训练数据：

```text
UEFB-v2 train + LOL-v2-real train + LOL-v2-synthetic train
```

评估：

```text
UEFB-v2 test
LOL-v2-real full test
LOL-v2-synthetic full test
```

结果文件：

```text
results/tables/p3c_multiseed_sweep_summary.csv
results/tables/p3c_multiseed_sweep_summary.json
results/tables/p3c_multiseed_sweep_aggregate.json
```

## 3. 3-seed results at e_shape=0.05

| Seed | UEFB PSNR↑ | UEFB E_corr↑ | Real PSNR↑ | Real E_corr↑ | Synthetic PSNR↑ | Synthetic E_corr↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 17.613 | 0.434 | 19.870 | 0.749 | 18.515 | 0.868 |
| 2027 | 18.150 | 0.435 | 19.914 | 0.726 | 17.379 | 0.855 |
| 3407 | 17.981 | 0.440 | 20.277 | 0.648 | 17.141 | 0.799 |

Mean ± std:

| Metric | Mean±Std |
|---|---:|
| UEFB PSNR | 17.915±0.275 |
| UEFB E_MAE | 0.276±0.015 |
| UEFB E_corr | 0.436±0.003 |
| Real PSNR | 20.021±0.223 |
| Real E_corr | 0.707±0.053 |
| Synthetic PSNR | 17.678±0.735 |
| Synthetic E_corr | 0.841±0.037 |

Interpretation:

- UEFB E_corr 稳定为正：`0.436±0.003`，std 仅 0.003。
- Real PSNR 稳定在 20 dB 左右：`20.021±0.223`。
- Synthetic PSNR 均高于 P3b non-e_shape joint 的 16.508，mean=`17.678±0.735`，但 std 较大，说明 synthetic 对 seed 敏感。

## 4. e_shape sweep at seed=3407

| e_shape | UEFB PSNR↑ | UEFB E_MAE↓ | UEFB E_corr↑ | Real PSNR↑ | Real E_corr↑ | Synthetic PSNR↑ | Synthetic E_corr↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 18.273 | 0.271 | 0.415 | 20.219 | 0.621 | 17.475 | 0.817 |
| 0.05 | 17.981 | 0.292 | 0.440 | 20.277 | 0.648 | 17.141 | 0.799 |
| 0.10 | 18.101 | 0.267 | 0.457 | 20.179 | 0.630 | 17.821 | 0.801 |

Sweep interpretation:

- Best UEFB PSNR: `e_shape=0.02` with 18.273 dB.
- Best Real PSNR: `e_shape=0.05` with 20.277 dB.
- Best Synthetic PSNR: `e_shape=0.10` with 17.821 dB.
- Best UEFB E_corr: `e_shape=0.10` with 0.457.

`e_shape=0.10` is the strongest single-seed balance: it gives the best UEFB E_corr and synthetic PSNR, with real PSNR only 0.098 dB below the best real setting.

## 5. Comparison to earlier stages

| Route | UEFB PSNR | UEFB E_corr | Real PSNR | Synthetic PSNR |
|---|---:|---:|---:|---:|
| P3 M4 UEFB-only | 18.653 | -0.139 | 17.411 | 13.963 |
| P3b M4J joint | 18.082 | 0.282 | 19.581 | 16.508 |
| P3c M4J_ES e=0.05 mean | 17.915 | 0.436 | 20.021 | 17.678 |
| P3c M4J_ES e=0.10 seed3407 | 18.101 | 0.457 | 20.179 | 17.821 |

## 6. Claim calibration

Now supported at P3c level:

```text
M4J_ES at e_shape=0.05 has stable positive UEFB E_corr across three seeds and improves paired LOL-v2 fidelity over UEFB-only M4. A small e_shape sweep suggests stronger E-shape weighting (0.10) may further improve synthetic fidelity and E_corr, but this weight still needs multi-seed validation.
```

Still not supported:

- SOTA / Retinexformer-level claims.
- Real-image physical E-field correctness. Paired LOL-v2 E metrics are oracle diagnostics, not real physical ground truth.
- e_shape=0.10 as the final default, because it is only single-seed so far.

## 7. Decision

| Candidate | Decision | Reason |
|---|---|---|
| M4J_ES e=0.05 | Keep as validated default | 3-seed positive E_corr and strong paired PSNR |
| M4J_ES e=0.10 | Promote to next 3-seed | Best single-seed balance for E_corr + synthetic PSNR |
| M4J without e_shape | Demote | Lower E_corr and lower paired PSNR than M4J_ES |
| M5/Q branch | Keep paused | Current bottleneck is fidelity + E-shape, not Q calibration |

## 8. Next recommended step

Before P4 official baselines, run **e_shape=0.10 3-seed validation**:

```text
M4J_ES, e_shape=0.10, seeds = 3407, 2027, 42
```

If it keeps real PSNR near 20 dB, synthetic PSNR > e=0.05 mean, and E_corr positive, promote e=0.10 as default and then proceed to P4 official baselines.
