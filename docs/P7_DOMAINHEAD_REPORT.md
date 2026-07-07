# P7-MS-DOMAINHEAD Report

生成时间：2026-07-01T23:03:44  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有 GitHub 提交/推送，也没有本地 git commit。

## 1. Purpose

P6/P6b/P6c 已证明：

```text
P6 multiscale trunk 有结构价值；
P6b scalar synthetic-rec 能救 synthetic，但伤 UEFB/real；
P6c scalar gate controls 能移动 trade-off，但不能同时过线。
```

P7-MS-DOMAINHEAD 因此测试下一条非 scalar-only 路线：

```text
keep P6 multiscale trunk
keep M4J_ES E/A-gate/E-shape losses
no Retinexformer teacher
no output-level distillation
no rec_by_dataset synthetic upweight
learn tiny domain-conditioned calibration modules
```

Domain canonicalization:

```text
UEFB / uefb -> uefb
LOL-v2-real-* / real -> real
LOL-v2-synthetic-* / synthetic -> synthetic
```

## 2. Implementation

新增/修改：

```text
src/rlef/models/rlef_former.py
  - RLEFFormer(domain_conditioning=True)
  - domain_names=['uefb', 'real', 'synthetic']
  - domain embedding
  - domain adapters:
      gate_bias
      head_bias        # restoration/exposure head biases
      feature_affine+gate_bias
  - forward(low, domain=...)

scripts/train.py
  - passes batch['dataset'] into model(..., domain=...)

scripts/eval_paired.py
  - new --domain argument
  - paired eval now tags real/synthetic test data with canonical domain

scripts/run_p7_domainhead.py
  - P7 runner, same promotion rule as P6/P6b/P6c

tests/test_p7_domainhead_runner_contract.py
  - P7 route/config/promotion/GPU assignment contract
```

## 3. P7 variants

| Variant | Adapter | Description |
|---|---|---|
| P7_MS_DGATE | `gate_bias` | domain-conditioned A-gate bias only |
| P7_MS_DHEAD | `head_bias` | domain-conditioned restoration/exposure head biases |
| P7_MS_DAFFINE_GATE | `feature_affine+gate_bias` | domain-conditioned feature affine adapter plus A-gate bias |

All runs are single-seed, 1000 steps, `base_channels=24`, `backbone=multiscale`, `backbone_blocks=3`.

## 4. Results

| Variant | Adapter | UEFB PSNR↑ | Real PSNR↑ | Synthetic PSNR↑ | UEFB E_corr↑ | Real E_corr↑ | Synthetic E_corr↑ | Promote? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| P7_MS_DGATE | `gate_bias` | 17.602 | 19.309 | 17.025 | 0.485 | 0.695 | 0.839 | No |
| P7_MS_DHEAD | `head_bias` | 18.232 | 19.209 | 17.913 | 0.449 | 0.706 | 0.827 | No |
| P7_MS_DAFFINE_GATE | `feature_affine+gate_bias` | 17.533 | 17.021 | 18.124 | 0.486 | 0.640 | 0.783 | No |

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
| P7_MS_DGATE | -0.313 | -0.712 | -0.654 | +0.049 | -0.012 | -0.002 |
| P7_MS_DHEAD | +0.318 | -0.812 | +0.235 | +0.013 | -0.001 | -0.014 |
| P7_MS_DAFFINE_GATE | -0.382 | -3.000 | +0.446 | +0.050 | -0.068 | -0.058 |

## 5. Interpretation

P7 gives a clearer architectural diagnostic than P6b/P6c, but still no balanced winner.

Best UEFB and best balanced P7 variant:

```text
P7_MS_DHEAD / head_bias
UEFB PSNR:      18.232 vs P3c 17.915±0.275
Real PSNR:      19.209 vs P3c 20.021±0.223
Synthetic PSNR: 17.913 vs P3c 17.678±0.735
```

Best synthetic P7 variant:

```text
P7_MS_DAFFINE_GATE / feature_affine+gate_bias
Synthetic PSNR: 18.124
Real PSNR:      17.021
```

Observed pattern:

```text
- domain gate bias alone hurts all primary PSNR metrics relative to P3c;
- domain head bias is the strongest P7 route: UEFB and synthetic pass P3c mean, real fails;
- feature_affine+gate_bias over-specializes synthetic and severely hurts real;
- P7 improves over P6b/P6c in interpretability: the head-bias path is useful, but needs real-domain protection.
```

## 6. Relation to P6/P6b/P6c and official baselines

| Method | UEFB PSNR | Real PSNR | Synthetic PSNR | Real E_corr | Synthetic E_corr | Decision |
|---|---:|---:|---:|---:|---:|---|
| P3c default mean | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | 0.707±0.053 | 0.841±0.037 | conservative default |
| P6_MS_M4J_ES | 18.015 | 20.197 | 17.598 | 0.629 | 0.812 | structural evidence, synthetic miss |
| P6B best synthetic (P6B_MS_SYN125) | 16.894 | 19.482 | 19.115 | 0.679 | 0.793 | synthetic rescued, UEFB/real hurt |
| P6C best balanced (P6C_MS_GATELOW005) | 18.046 | 19.831 | 17.422 | 0.582 | 0.779 | no promotion |
| P7 best balanced (P7_MS_DHEAD) | 18.232 | 19.209 | 17.913 | 0.706 | 0.827 | UEFB/synthetic pass, real fails |
| Retinexformer official | — | 22.794 | 25.669 | — | — | strong baseline gap remains |

## 7. Go / no-go

```text
No-go for P7 3-seed promotion.
```

Promotion criterion requires UEFB, real, and synthetic PSNR all exceed P3c mean. No P7 variant satisfies this:

```text
P7_MS_DGATE: all primary PSNR metrics fail.
P7_MS_DHEAD: UEFB/synthetic pass, real fails.
P7_MS_DAFFINE_GATE: synthetic passes, UEFB/real fail badly.
```

## 8. Claim decision

Safe claim:

```text
Tiny domain-conditioned head biases improve the multiscale RLEF route on UEFB and synthetic test sets without scalar synthetic reconstruction upweighting, but they currently reduce real PSNR; domain-conditioned calibration is promising but not yet a balanced default.
```

Current default remains:

```text
P3c M4J_ES e_shape=0.05
```

## 9. Next recommendation

P7 shows the most constructive next target: protect real while preserving P7_MS_DHEAD's UEFB/synthetic gains.

Suggested next branch:

```text
P7b-MS-DHEAD-REALANCHOR
- start from P7_MS_DHEAD
- keep domain-conditioned head bias only
- add a small real-domain anchor, not a global scalar knob:
  1. real-domain residual/head-bias L2 prior toward zero, or
  2. initialize real-domain bias to zero and regularize only real bias, or
  3. freeze trunk longer / warm up domain heads after base convergence
- promotion rule unchanged: UEFB, real, and synthetic PSNR all exceed P3c mean before 3-seed
```

Alternative: if the goal is paper narrative rather than more search, P7 already supports a strong diagnostic story: scalar knobs fail; domain-conditioned head bias partly solves synthetic/UEFB but exposes real-domain calibration as the remaining bottleneck.
