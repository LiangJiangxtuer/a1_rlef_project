# P3d Formal Report: M4J_ES e_shape=0.10 3-seed validation

生成时间：2026-07-01T12:57:19  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有 GitHub 提交/推送，也没有本地 git commit。

## 1. Objective

P3c seed3407 sweep 显示 `e_shape=0.10` 在单 seed 下获得最高 synthetic PSNR 与 UEFB E_corr，因此 P3d 对它执行 3-seed 验证，作为进入 P4 official baselines 前的默认配置决策。

## 2. Protocol

执行脚本：`scripts/run_p3d_e010_multiseed.py`

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_p3d_e010_multiseed.py   --max_steps 1000 --train_crop 128 --batch_size 8 --parallel 2
```

设置：

```text
M4J_ES, e_shape=0.10, seeds = 3407, 2027, 42
```

seed3407/e=0.10 复用 P3c 已完成 checkpoint 与 paired eval。

## 3. Results

| Seed | UEFB PSNR↑ | UEFB E_corr↑ | Real PSNR↑ | Real E_corr↑ | Synthetic PSNR↑ | Synthetic E_corr↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 18.317 | 0.444 | 20.059 | 0.729 | 18.018 | 0.839 |
| 2027 | 17.915 | 0.446 | 19.387 | 0.697 | 17.040 | 0.864 |
| 3407 | 18.101 | 0.457 | 20.179 | 0.630 | 17.821 | 0.801 |

Mean ± std:

| Metric | e=0.10 Mean±Std | e=0.05 Mean±Std from P3c |
|---|---:|---:|
| UEFB PSNR | 18.111±0.201 | 17.915±0.275 |
| UEFB E_MAE | 0.273±0.007 | 0.276±0.015 |
| UEFB E_corr | 0.449±0.007 | 0.436±0.003 |
| Real PSNR | 19.875±0.427 | 20.021±0.223 |
| Real E_corr | 0.686±0.050 | 0.707±0.053 |
| Synthetic PSNR | 17.626±0.517 | 17.678±0.735 |
| Synthetic E_corr | 0.835±0.032 | 0.841±0.037 |

## 4. Decision

P3d go/no-go rule from P3c was:

```text
Proceed/promote e=0.10 if real PSNR remains near 20 dB,
synthetic PSNR > e=0.05 mean, and E_corr stays positive.
```

Observed:

- Real PSNR: 19.875±0.427; acceptable but slightly below e=0.05 mean 20.021±0.223.
- Synthetic PSNR: 17.626±0.517; **does not exceed** e=0.05 mean 17.678±0.735.
- UEFB E_corr: 0.449±0.007; stable positive and slightly higher than e=0.05.
- Synthetic E_corr: 0.835±0.032; stable positive, comparable to e=0.05.

Decision:

```text
e_shape=0.10 is not promoted as final default.
e_shape=0.05 remains the validated default for P4 official baseline comparison.
```

Rationale: e=0.10 improves UEFB PSNR/E_corr slightly, but does not improve synthetic PSNR mean over e=0.05 and has lower real PSNR mean. Therefore the safer paper route is to proceed to P4 with **M4J_ES e=0.05** as the multi-seed validated default, while recording e=0.10 as a robustness/ablation variant.

## 5. P4 action

Because a default has now been selected (`M4J_ES e=0.05`), P4 official baselines may proceed, but claims must compare against baselines under exact protocol and must distinguish:

- fully executed official code/checkpoint baselines;
- official released result images, if used;
- blockers for baselines whose checkpoints/dependencies are unavailable.
