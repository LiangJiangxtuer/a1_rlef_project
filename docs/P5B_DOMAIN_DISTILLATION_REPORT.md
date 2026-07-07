# P5b Domain-Conditioned Distillation Report

生成时间：2026-07-01T17:51:08  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有 GitHub 提交/推送，也没有本地 git commit。

## 1. Purpose

P5 证明 simple output-level Retinexformer distillation 存在 real/synthetic trade-off：`distill=0.30` 提升 real 但伤害 synthetic，`distill=0.10` 较平衡但 real 下降。因此 P5b 测试 domain-conditioned distillation：

```text
real paired train:      distill = 0.30
synthetic paired train: distill = 0.05
UEFB train:             distill = 0.00
```

目标是保留 P5_RD_T03 的 real gain，同时避免 synthetic 被强 teacher 拉低。

## 2. Implementation

新增契约与代码：

```text
weights['distill_by_dataset']
scripts/run_p5b_domain_distill.py
tests/test_p5b_domain_distill_runner_contract.py
```

`compute_total_loss` 现在支持按 batch 中的 `dataset` 字段分组施加不同 teacher-distill 权重，并为所有配置域输出稳定的 log scalar，避免 mixed `ConcatDataset` batch 的 CSV fieldnames 不一致。

## 3. Result

| Variant | real distill | synthetic distill | UEFB PSNR↑ | UEFB E_corr↑ | Real PSNR↑ | Real E_corr↑ | Synthetic PSNR↑ | Synthetic E_corr↑ | Promote to 3-seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P5B_DW_R03_S005 | 0.3 | 0.05 | 17.574 | 0.431 | 19.366 | 0.728 | 17.473 | 0.865 | No |

Reference P3c default (`M4J_ES e_shape=0.05`, 3-seed mean):

| Metric | P3c default |
|---|---:|
| UEFB PSNR | 17.915±0.275 |
| UEFB E_corr | 0.436±0.003 |
| Real PSNR | 20.021±0.223 |
| Real E_corr | 0.707±0.053 |
| Synthetic PSNR | 17.678±0.735 |
| Synthetic E_corr | 0.841±0.037 |

Delta vs P3c mean:

| Variant | ΔUEFB PSNR | ΔReal PSNR | ΔSynthetic PSNR | ΔUEFB E_corr | ΔReal E_corr | ΔSynthetic E_corr |
|---|---:|---:|---:|---:|---:|---:|
| P5B_DW_R03_S005 | -0.340 | -0.655 | -0.205 | -0.005 | +0.021 | +0.025 |

## 4. Interpretation

P5b-DW improves E-field correlation on paired diagnostics:

```text
Real E_corr:      0.728 vs P3c 0.707±0.053
Synthetic E_corr: 0.865 vs P3c 0.841±0.037
```

But it does **not** pass the PSNR go/no-go gate:

```text
Real PSNR:      19.366 vs P3c 20.021±0.223
Synthetic PSNR: 17.473 vs P3c 17.678±0.735
UEFB PSNR:      17.574 vs P3c 17.915±0.275
```

Therefore P5b-DW should be treated as a diagnostic that teacher/domain weighting can strengthen E-shape alignment, not as a method improvement.

## 5. Relation to P5 and P4

| Method | Real PSNR | Synthetic PSNR | Real E_corr | Synthetic E_corr | Decision |
|---|---:|---:|---:|---:|---|
| P3c M4J_ES e=0.05 | 20.021±0.223 | 17.678±0.735 | 0.707±0.053 | 0.841±0.037 | conservative default |
| P5_RD_T01 | 19.588 | 17.960 | 0.610 | 0.830 | not default |
| P5_RD_T03 | 20.416 | 16.996 | 0.621 | 0.804 | not default |
| P5B_DW_R03_S005 | 19.366 | 17.473 | 0.728 | 0.865 | no 3-seed promotion |
| Retinexformer official | 22.794 | 25.669 | — | — | strong baseline gap remains |

## 6. Go / no-go

```text
No-go for 3-seed P5b-DW.
```

Reason: promotion criterion required both real and synthetic PSNR to exceed P3c mean. P5b-DW failed both PSNR gates, despite improving paired E_corr.

## 7. Claim decision

Safe claim:

```text
Domain-conditioned distillation improves paired E-field diagnostic correlation, but does not improve restoration PSNR over the P3c default. It is evidence that teacher/domain weighting affects interpretability signals, not evidence of a better restoration method.
```

Current default remains:

```text
P3c M4J_ES e_shape=0.05
```

## 8. Next recommendation

Stop spending on scalar output-level distillation sweeps. The next branch should be structural:

```text
P6 / P5c: stronger-backbone RLEF integration
- keep E/A-gate/E-shape heads as auxiliary outputs
- replace the tiny restoration trunk with a stronger restoration module, or
- attach RLEF heads to Retinexformer-like features / intermediate supervision
```

If a smaller diagnostic is needed before P6, use feature-level or intermediate-output distillation, not another scalar output-loss sweep.
