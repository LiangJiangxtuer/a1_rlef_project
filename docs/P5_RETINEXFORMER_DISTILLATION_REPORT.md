# P5 Retinexformer-Teacher Distillation Report

生成时间：2026-07-01T17:28:27  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有 GitHub 提交/推送，也没有本地 git commit。

## 1. Purpose

P4 证明当前 tiny RLEF backbone 不是 Retinexformer-level，但 RLEF 的 E/A-gate/E-shape heads 有稳定的解释性诊断价值。P5 因此测试一个低风险 pivot：保留 RLEF M4J_ES heads，把 Retinexformer 官方输出作为 train-time teacher，对 paired LOL-v2 real/synthetic train samples 做 output-level distillation。

## 2. Teacher generation

Teacher 来源：P4 已执行的 Retinexformer 官方 repo + 官方 LOL-v2 pretrained weights。

| Dataset | Train low count | Teacher outputs | Status |
|---|---:|---:|---|
| LOL-v2-real train | 689 | 689 | already_present |
| LOL-v2-synthetic train | 900 | 900 | already_present |

Teacher output directories:

```text
experiments/p5_retinexformer_train_teacher/real
experiments/p5_retinexformer_train_teacher/synthetic
```

## 3. P5 variants

Both P5 variants use:

```text
base = M4J_ES
train = UEFB-v2 train + LOL-v2-real train + LOL-v2-synthetic train
RLEF heads = exposure branch + adaptive gauge + A-gate + E-shape loss
steps = 1000
seed = 3407
```

Distillation loss:

```text
loss = rec_to_GT + distill_weight * rec_to_Retinexformer_teacher + physical/E/A auxiliary losses
```

## 4. Results

| Variant | distill weight | UEFB PSNR↑ | UEFB E_corr↑ | Real PSNR↑ | Real E_corr↑ | Synthetic PSNR↑ | Synthetic E_corr↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| P5_RD_T01 | 0.10 | 17.948 | 0.438 | 19.588 | 0.610 | 17.960 | 0.830 |
| P5_RD_T03 | 0.30 | 18.169 | 0.442 | 20.416 | 0.621 | 16.996 | 0.804 |

Reference P3c default:

| Variant | UEFB PSNR | UEFB E_corr | Real PSNR | Real E_corr | Synthetic PSNR | Synthetic E_corr |
|---|---:|---:|---:|---:|---:|---:|
| M4J_ES e=0.05, 3-seed mean | 17.915±0.275 | 0.436±0.003 | 20.021±0.223 | 0.707±0.053 | 17.678±0.735 | 0.841±0.037 |

Delta vs P3c e=0.05 mean:

| Variant | ΔUEFB PSNR | ΔReal PSNR | ΔSynthetic PSNR | ΔUEFB E_corr | ΔReal E_corr | ΔSynthetic E_corr |
|---|---:|---:|---:|---:|---:|---:|
| P5_RD_T01 | +0.034 | -0.433 | +0.281 | +0.002 | -0.097 | -0.011 |
| P5_RD_T03 | +0.255 | +0.395 | -0.683 | +0.005 | -0.087 | -0.037 |

## 5. Interpretation

### P5_RD_T01: weak distillation

Weak teacher distillation is the more balanced P5 outcome:

- Synthetic PSNR improves over P3c mean: `17.960` vs `17.678±0.735`.
- UEFB PSNR is slightly above P3c mean: `17.948` vs `17.915±0.275`.
- Real PSNR drops below P3c mean: `19.588` vs `20.021±0.223`.
- E_corr remains positive but is below P3c on real/synthetic.

### P5_RD_T03: stronger distillation

Stronger teacher distillation creates domain tension:

- Real PSNR improves: `20.416` vs P3c `20.021±0.223`.
- Synthetic PSNR drops substantially: `16.996` vs P3c `17.678±0.735`.
- E_corr remains positive, but synthetic E_corr is below P3c.

## 6. Relation to P4 official baselines

P5 still does not close the Retinexformer gap:

| Method | Real PSNR | Synthetic PSNR | Note |
|---|---:|---:|---|
| Retinexformer official | 22.794 | 25.669 | P4 official blind baseline |
| RLEF P5_RD_T03 | 20.416 | 16.996 | better real, worse synthetic |
| RLEF P5_RD_T01 | 19.588 | 17.960 | more balanced weak distill |
| Zero-DCE++ official | 18.491 | 17.576 | P4 official blind baseline |

## 7. Claim decision

Safe P5 claim:

```text
Simple output-level Retinexformer distillation transfers some domain-specific fidelity to the RLEF prototype, but it introduces a real/synthetic trade-off and does not close the strong-baseline gap. The current evidence supports RLEF heads as auxiliary calibration/explainability components, not as a standalone SOTA restoration backbone.
```

Current default decision:

```text
Do not promote P5_RD_T03 as default because it hurts synthetic.
Do not promote P5_RD_T01 as final default yet because it is single-seed and lowers real PSNR/E_corr versus P3c.
Keep P3c M4J_ES e=0.05 as the conservative default until a stronger-backbone or domain-conditioned distillation branch is validated.
```

## 8. Next recommended P5 branch

The next scientifically clean step is not to increase output distillation weight. Instead:

```text
P5b: domain-conditioned / backbone-level integration
1. real/synthetic domain-conditioned distill weights, or
2. feature-level distillation from Retinexformer intermediate restoration features, or
3. replace only the tiny restoration trunk while keeping RLEF E/A/Q heads as auxiliary outputs.
```

Minimal next experiment:

```text
P5b-DW: distill=0.30 for real, distill=0.05 or 0.00 for synthetic,
then run 3-seed only if single-seed improves both real and synthetic over P3c mean.
```
