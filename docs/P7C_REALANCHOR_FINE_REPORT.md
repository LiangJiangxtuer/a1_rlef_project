# P7c-MS-DHEAD-REALANCHOR-FINE Report

生成时间：2026-07-02T12:26:06  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有 GitHub 提交/推送，也没有本地 git commit。

## 1. Purpose

P7b 的最佳 near-miss 是 `P7B_DHEAD_RA010`：

```text
UEFB PSNR:      18.085 > P3c mean
Real PSNR:      19.847 < P3c mean by 0.174 dB
Synthetic PSNR: 17.965 > P3c mean
```

P7c 因此执行更强 real-domain anchor fine sweep：

```text
start from P7B_DHEAD_RA010 design
keep multiscale trunk
keep domain_adapter=head_bias
no Retinexformer teacher
no output-level distillation
no rec_by_dataset
sweep real-domain head-bias anchor = 0.015 / 0.020 / 0.030
```

## 2. Implementation

新增：

```text
scripts/run_p7c_realanchor_fine.py
  - P7c fine-sweep runner
  - outputs configs/p7c_realanchor_fine/
  - outputs logs/p7c_realanchor_fine/
  - outputs results/tables/p7c_realanchor_fine_summary.csv/json

tests/test_p7c_realanchor_fine_runner_contract.py
  - route/config/promotion/GPU assignment contract
```

P7c 不需要新增模型/loss 代码，因为沿用 P7b 已验证的：

```text
RLEFFormer(..., domain_adapter='head_bias') -> out['domain_head_bias_l2']
compute_total_loss(..., weights['domain_head_anchor_by_dataset'])
```

## 3. Variants

| Variant | Real anchor weight |
|---|---:|
| P7C_DHEAD_RA015 | 0.015 |
| P7C_DHEAD_RA020 | 0.020 |
| P7C_DHEAD_RA030 | 0.030 |

All runs are single-seed, 1000 steps, `base_channels=24`, `backbone=multiscale`, `backbone_blocks=3`, `domain_adapter=head_bias`.

## 4. Results

| Variant | real_anchor | UEFB PSNR↑ | Real PSNR↑ | Synthetic PSNR↑ | UEFB E_corr↑ | Real E_corr↑ | Synthetic E_corr↑ | Promote? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P7C_DHEAD_RA015 | 0.015 | 17.954 | 19.082 | 17.591 | 0.447 | 0.733 | 0.809 | No |
| P7C_DHEAD_RA020 | 0.020 | 17.441 | 19.517 | 17.820 | 0.422 | 0.728 | 0.838 | No |
| P7C_DHEAD_RA030 | 0.030 | 18.062 | 18.641 | 17.682 | 0.447 | 0.739 | 0.818 | No |

Reference P3c default (`M4J_ES e_shape=0.05`, 3-seed mean):

| Metric | P3c default |
|---|---:|
| UEFB PSNR | 17.915±0.275 |
| Real PSNR | 20.021±0.223 |
| Synthetic PSNR | 17.678±0.735 |
| UEFB E_corr | 0.436±0.003 |
| Real E_corr | 0.707±0.053 |
| Synthetic E_corr | 0.841±0.037 |

Delta vs P3c mean:

| Variant | ΔUEFB PSNR | ΔReal PSNR | ΔSynthetic PSNR | ΔUEFB E_corr | ΔReal E_corr | ΔSynthetic E_corr |
|---|---:|---:|---:|---:|---:|---:|
| P7C_DHEAD_RA015 | +0.039 | -0.938 | -0.087 | +0.011 | +0.026 | -0.031 |
| P7C_DHEAD_RA020 | -0.473 | -0.504 | +0.141 | -0.014 | +0.021 | -0.002 |
| P7C_DHEAD_RA030 | +0.147 | -1.380 | +0.004 | +0.011 | +0.032 | -0.023 |

## 5. Effect relative to P7B_DHEAD_RA010

P7B_DHEAD_RA010 baseline:

```text
UEFB PSNR:      18.085
Real PSNR:      19.847
Synthetic PSNR: 17.965
Real E_corr:    0.729
Synthetic E_corr:0.841
```

Delta vs P7B_DHEAD_RA010:

| Variant | real_anchor | ΔUEFB PSNR | ΔReal PSNR | ΔSynthetic PSNR | ΔReal E_corr | ΔSynthetic E_corr |
|---|---:|---:|---:|---:|---:|---:|
| P7C_DHEAD_RA015 | 0.015 | -0.131 | -0.765 | -0.374 | +0.004 | -0.032 |
| P7C_DHEAD_RA020 | 0.020 | -0.644 | -0.330 | -0.146 | -0.001 | -0.002 |
| P7C_DHEAD_RA030 | 0.030 | -0.023 | -1.206 | -0.284 | +0.010 | -0.023 |

Key result:

```text
Stronger real anchors did not improve over P7B_DHEAD_RA010.
Best P7c real PSNR is 19.517 (P7C_DHEAD_RA020),
which is -0.330 dB below P7B_DHEAD_RA010.
```

## 6. Anchor scalar verification

Final train-log scalar values:

| Variant | domain_head_anchor | real | uefb | synthetic |
|---|---:|---:|---:|---:|
| P7C_DHEAD_RA015 | 0.0008882512338459492 | 0.0008882512338459492 | 0.0 | 0.0 |
| P7C_DHEAD_RA020 | 0.0008136872784234583 | 0.0008136872784234583 | 0.0 | 0.0 |
| P7C_DHEAD_RA030 | 0.0008277441374957561 | 0.0008277441374957561 | 0.0 | 0.0 |

This verifies the anchor remains real-domain-only.

## 7. Relation to previous routes and official baselines

| Method | UEFB PSNR | Real PSNR | Synthetic PSNR | Real E_corr | Synthetic E_corr | Decision |
|---|---:|---:|---:|---:|---:|---|
| P3c default mean | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | 0.707±0.053 | 0.841±0.037 | conservative default |
| P6_MS_M4J_ES | 18.015 | 20.197 | 17.598 | 0.629 | 0.812 | structural evidence, synthetic miss |
| P7_MS_DHEAD | 18.232 | 19.209 | 17.913 | 0.706 | 0.827 | UEFB/synthetic pass, real fails |
| P7B_DHEAD_RA010 | 18.085 | 19.847 | 17.965 | 0.729 | 0.841 | best P7-family near-miss |
| P7c best real (P7C_DHEAD_RA020) | 17.441 | 19.517 | 17.820 | 0.728 | 0.838 | no promotion |
| Retinexformer official | — | 22.794 | 25.669 | — | — | strong baseline gap remains |

## 8. Go / no-go

```text
No-go for P7c 3-seed promotion.
```

Promotion criterion requires UEFB, real, and synthetic PSNR all exceed P3c mean. No P7c variant satisfies this:

```text
P7C_DHEAD_RA015: UEFB passes; real and synthetic fail.
P7C_DHEAD_RA020: synthetic passes; UEFB and real fail.
P7C_DHEAD_RA030: UEFB and synthetic pass narrowly; real fails badly.
```

P7c also does not beat the P7b near-miss:

```text
P7-family best remains P7B_DHEAD_RA010.
```

## 9. Claim decision

Safe claim:

```text
A stronger real-domain head-bias anchor fine sweep does not improve on the P7B_DHEAD_RA010 near-miss; the useful anchoring effect saturates around 0.010, and larger anchors trade off UEFB/synthetic/real without producing a balanced default.
```

Current default remains:

```text
P3c M4J_ES e_shape=0.05
```

## 10. Next recommendation

Do **not** continue scalar-strength anchor sweeps. The next meaningful option is a schedule change, not a stronger coefficient:

```text
P7d-MS-DHEAD-WARMANCHOR
- keep head_bias and real anchor around 0.010
- train 300 steps without domain-head anchor
- enable RA010 or RA020 for the remaining 700 steps
- still no teacher, no distill, no rec_by_dataset
- if warm-anchor does not beat P7B_DHEAD_RA010, stop P7 family and report P7b as near-miss diagnostic evidence
```

If the goal is to stabilize a current default rather than continue search, keep:

```text
P3c M4J_ES e_shape=0.05
```
