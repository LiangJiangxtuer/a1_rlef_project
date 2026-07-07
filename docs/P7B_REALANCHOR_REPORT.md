# P7b-MS-DHEAD-REALANCHOR Report

生成时间：2026-07-02T08:33:45  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有 GitHub 提交/推送，也没有本地 git commit。

## 1. Purpose

P7_MS_DHEAD was the strongest P7 route:

```text
UEFB PSNR passes P3c mean
Synthetic PSNR passes P3c mean
Real PSNR fails P3c mean
```

P7b therefore keeps the exact P7_MS_DHEAD route and adds a small real-domain anchor:

```text
keep P6 multiscale trunk
keep M4J_ES E/A-gate/E-shape losses
keep domain_adapter=head_bias
no Retinexformer teacher
no output-level distillation
no rec_by_dataset synthetic/real upweight
anchor only real-domain domain-head bias L2 toward zero
```

This is not a global scalar rec/gate knob: it directly regularizes the domain-conditioned head-bias parameters used on real-domain samples.

## 2. Implementation

新增/修改：

```text
src/rlef/models/rlef_former.py
  - head_bias adapters now expose per-sample `domain_head_bias_l2`
  - includes restoration-head bias and exposure-head bias terms

src/rlef/losses/total_loss.py
  - new `domain_head_anchor_by_dataset`
  - canonical domains: uefb / real / synthetic
  - stable scalar logging:
      domain_head_anchor
      domain_head_anchor_real
      domain_head_anchor_uefb
      domain_head_anchor_synthetic

scripts/run_p7b_realanchor.py
  - P7b single-seed routing runner

tests/test_p7b_realanchor_runner_contract.py
  - route/config/promotion/GPU assignment contract
```

Verified train logs contain active real-only anchor scalars and zero UEFB/synthetic anchor scalars.

## 3. Variants

| Variant | Real anchor weight |
|---|---:|
| P7B_DHEAD_RA001 | 0.001 |
| P7B_DHEAD_RA005 | 0.005 |
| P7B_DHEAD_RA010 | 0.010 |

All runs are single-seed, 1000 steps, `base_channels=24`, `backbone=multiscale`, `backbone_blocks=3`, `domain_adapter=head_bias`.

## 4. Results

| Variant | real_anchor | UEFB PSNR↑ | Real PSNR↑ | Synthetic PSNR↑ | UEFB E_corr↑ | Real E_corr↑ | Synthetic E_corr↑ | Promote? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P7B_DHEAD_RA001 | 0.001 | 18.190 | 19.127 | 17.517 | 0.455 | 0.719 | 0.823 | No |
| P7B_DHEAD_RA005 | 0.005 | 17.936 | 19.058 | 18.008 | 0.440 | 0.719 | 0.839 | No |
| P7B_DHEAD_RA010 | 0.010 | 18.085 | 19.847 | 17.965 | 0.438 | 0.729 | 0.841 | No |

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
| P7B_DHEAD_RA001 | +0.276 | -0.894 | -0.161 | +0.019 | +0.012 | -0.017 |
| P7B_DHEAD_RA005 | +0.022 | -0.962 | +0.330 | +0.004 | +0.011 | -0.001 |
| P7B_DHEAD_RA010 | +0.171 | -0.174 | +0.287 | +0.002 | +0.022 | +0.000 |

## 5. Effect relative to P7_MS_DHEAD

P7_MS_DHEAD baseline:

```text
UEFB PSNR:      18.232
Real PSNR:      19.209
Synthetic PSNR: 17.913
Real E_corr:    0.706
Synthetic E_corr:0.827
```

Delta vs P7_MS_DHEAD:

| Variant | real_anchor | ΔUEFB PSNR | ΔReal PSNR | ΔSynthetic PSNR | ΔReal E_corr | ΔSynthetic E_corr |
|---|---:|---:|---:|---:|---:|---:|
| P7B_DHEAD_RA001 | 0.001 | -0.042 | -0.082 | -0.396 | +0.013 | -0.003 |
| P7B_DHEAD_RA005 | 0.005 | -0.296 | -0.150 | +0.095 | +0.013 | +0.013 |
| P7B_DHEAD_RA010 | 0.010 | -0.147 | +0.638 | +0.052 | +0.023 | +0.014 |

Key diagnostic:

```text
P7B_DHEAD_RA010 improves real PSNR by +0.638 over P7_MS_DHEAD
while retaining UEFB and synthetic PSNR above P3c mean.
```

But it still misses P3c real PSNR by:

```text
-0.174 dB
```

## 6. Relation to previous routes and official baselines

| Method | UEFB PSNR | Real PSNR | Synthetic PSNR | Real E_corr | Synthetic E_corr | Decision |
|---|---:|---:|---:|---:|---:|---|
| P3c default mean | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | 0.707±0.053 | 0.841±0.037 | conservative default |
| P6_MS_M4J_ES | 18.015 | 20.197 | 17.598 | 0.629 | 0.812 | structural evidence, synthetic miss |
| P6C best balanced (P6C_MS_GATELOW005) | 18.046 | 19.831 | 17.422 | 0.582 | 0.779 | no promotion |
| P7_MS_DHEAD | 18.232 | 19.209 | 17.913 | 0.706 | 0.827 | UEFB/synthetic pass, real fails |
| P7b best (P7B_DHEAD_RA010) | 18.085 | 19.847 | 17.965 | 0.729 | 0.841 | closest to balanced, real still fails |
| Retinexformer official | — | 22.794 | 25.669 | — | — | strong baseline gap remains |

## 7. Go / no-go

```text
No-go for P7b 3-seed promotion.
```

Promotion criterion requires UEFB, real, and synthetic PSNR all exceed P3c mean. No P7b variant satisfies this.

Most important variant:

```text
P7B_DHEAD_RA010
UEFB PSNR:      18.085 vs P3c 17.915±0.275  PASS
Real PSNR:      19.847 vs P3c 20.021±0.223  FAIL by 0.174 dB
Synthetic PSNR: 17.965 vs P3c 17.678±0.735  PASS
```

## 8. Claim decision

Safe claim:

```text
A real-domain head-bias anchor substantially recovers the real-domain weakness of P7_MS_DHEAD while preserving UEFB and synthetic gains, but the best single-seed P7b variant remains below the P3c real-PSNR mean and is not yet a balanced default.
```

Current default remains:

```text
P3c M4J_ES e_shape=0.05
```

## 9. Next recommendation

P7b is constructive: the anchor works in the intended direction. The next route should not abandon P7b; it should refine the real anchor around the current best.

Suggested next branch:

```text
P7c-MS-DHEAD-REALANCHOR-FINE
- start from P7B_DHEAD_RA010
- fine sweep stronger real anchor: 0.015 / 0.020 / 0.030
- optionally include one warmup variant: train base 300 steps without anchor then enable RA010/RA020
- still no teacher, no distill, no rec_by_dataset
- promotion rule unchanged: UEFB, real, synthetic PSNR all exceed P3c mean before 3-seed
```

Stopping criterion for search:

```text
If stronger anchors push real over P3c but collapse synthetic/UEFB, stop P7-family and report P7b as near-miss diagnostic evidence.
```
