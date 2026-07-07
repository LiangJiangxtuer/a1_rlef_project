# P6 Stronger-Backbone RLEF Integration Report

生成时间：2026-07-01T18:43:49  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有 GitHub 提交/推送，也没有本地 git commit。

## 1. Purpose

P5/P5b 表明 scalar output-level Retinexformer distillation 不适合作为主线：它会引入 real/synthetic trade-off，或只改善 E-field diagnostics 而不改善 PSNR。P6 因此执行结构性路线：

```text
keep RLEF M4J_ES heads and losses
replace tiny one-level trunk with a stronger two-level multiscale restoration backbone
no teacher/output-level distillation
```

## 2. Implementation

新增/修改：

```text
src/rlef/models/rlef_former.py
  - MultiScaleRestorationBackbone
  - RLEFFormer(backbone='tiny'|'multiscale', backbone_blocks=N)

scripts/train.py
  - build_model forwards backbone/backbone_blocks from config

scripts/run_p6_structural_backbone.py
  - P6 single-seed structural runner

tests/test_p6_structural_backbone_runner_contract.py
  - runner/config/promotion rule contract
```

P6 variant:

```yaml
variant_id: P6_MS_M4J_ES
seed: 3407
model:
  base_channels: 24
  backbone: multiscale
  backbone_blocks: 3
  exposure_branch: true
  adaptive_gauge: true
  gate_branch: true
loss:
  rec: 1.0
  phys: 0.15
  poisson: 0.05
  gauge: 0.10
  id: 0.02
  gate: 0.02
  wtv: 0.02
  e_shape: 0.05
```

## 3. Result

| Variant | Backbone | UEFB PSNR↑ | UEFB E_corr↑ | Real PSNR↑ | Real E_corr↑ | Synthetic PSNR↑ | Synthetic E_corr↑ | Promote to 3-seed? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| P6_MS_M4J_ES | multiscale x3 | 18.015 | 0.464 | 20.197 | 0.629 | 17.598 | 0.812 | No |

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
| P6_MS_M4J_ES | +0.101 | +0.176 | -0.081 | +0.028 | -0.079 | -0.029 |

## 4. Interpretation

P6 is the first post-P5 branch that improves restoration PSNR on UEFB and real without teacher loss:

```text
UEFB PSNR: 18.015 vs P3c 17.915±0.275
Real PSNR: 20.197 vs P3c 20.021±0.223
```

But it still does not satisfy the balanced promotion gate because synthetic PSNR is slightly below the P3c mean:

```text
Synthetic PSNR: 17.598 vs P3c 17.678±0.735
```

E-field diagnostics are mixed:

```text
UEFB E_corr improves:      0.464 vs 0.436±0.003
Real E_corr decreases:     0.629 vs 0.707±0.053
Synthetic E_corr decreases:0.812 vs 0.841±0.037
```

Therefore P6 validates the structural hypothesis partially: stronger restoration capacity helps real/UEFB PSNR, but the current multiscale trunk does not preserve synthetic generalization or paired E-field alignment well enough to become the default.

## 5. Relation to P5/P5b/P4

| Method | Real PSNR | Synthetic PSNR | UEFB PSNR | Real E_corr | Synthetic E_corr | Decision |
|---|---:|---:|---:|---:|---:|---|
| P3c M4J_ES e=0.05 | 20.021±0.223 | 17.678±0.735 | 17.915±0.275 | 0.707±0.053 | 0.841±0.037 | conservative default |
| P5_RD_T01 | 19.588 | 17.960 | 17.948 | 0.610 | 0.830 | not default |
| P5_RD_T03 | 20.416 | 16.996 | 18.169 | 0.621 | 0.804 | not default |
| P5B_DW_R03_S005 | 19.366 | 17.473 | 17.574 | 0.728 | 0.865 | diagnostic only |
| P6_MS_M4J_ES | 20.197 | 17.598 | 18.015 | 0.629 | 0.812 | promising but no 3-seed |
| Retinexformer official | 22.794 | 25.669 | — | — | — | strong baseline gap remains |

## 6. Go / no-go

```text
No-go for immediate 3-seed P6 promotion.
```

Promotion criterion required UEFB, real, and synthetic PSNR all to exceed P3c mean. P6 passes UEFB and real but misses synthetic by `0.081` dB.

## 7. Claim decision

Safe claim:

```text
A stronger multiscale restoration trunk improves UEFB and real PSNR over the P3c mean without teacher distillation, supporting the structural-backbone direction. However, it does not yet improve synthetic PSNR or paired E-field correlations enough for default promotion.
```

Current default remains:

```text
P3c M4J_ES e_shape=0.05
```

P6 is now the best evidence for the next route, not the final method.

## 8. Next recommendation

Run a targeted P6b synthetic-protection structural route rather than returning to output-level distillation:

```text
P6b-MS-SYNPROTECT:
- keep multiscale backbone
- preserve no-teacher setting
- add synthetic-protection control, e.g.
  (a) slightly lower restoration capacity/gate aggressiveness, or
  (b) synthetic-balanced sampler/loss weighting, or
  (c) E-shape/gate regularization tuned only if it does not hurt real PSNR
- promote only if UEFB, real, and synthetic PSNR all exceed P3c mean
```
