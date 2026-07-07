# P6c-MS-GATEPROTECT Report

生成时间：2026-07-01T22:27:46  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有 GitHub 提交/推送，也没有本地 git commit。

## 1. Purpose

P6b 证明 full synthetic reconstruction upweighting 能 rescue synthetic PSNR，但会明显牺牲 UEFB/real。因此 P6c-MS-GATEPROTECT 改走更细的 gate/restoration aggressiveness 控制：

```text
keep P6 multiscale structural route
keep M4J_ES E/A-gate/E-shape heads
no Retinexformer teacher
no output-level distillation
no rec_by_dataset / no synthetic rec upweight
protect synthetic by modifying A-gate behavior or restoration capacity
```

## 2. Implementation

新增/修改：

```text
src/rlef/losses/total_loss.py
  - weights['gate_identity']
  - identity-weighted A-gate penalty: exp(-|high-low|/tau) * A

scripts/run_p6c_gateprotect.py
  - P6c gate-protection sweep runner
  - --gpu_start support, reused because GPU 0 remained occupied

tests/test_p6c_gateprotect_runner_contract.py
  - P6c route/config/promotion/GPU assignment contract
```

P6c tested four variants:

```text
P6C_MS_GATEID005:   blocks=3, gate=0.020, gate_identity=0.005
P6C_MS_GATEID010:   blocks=3, gate=0.020, gate_identity=0.010
P6C_MS_GATELOW005:  blocks=3, gate=0.005, gate_identity=0.000
P6C_MS_B2_GATEID005:blocks=2, gate=0.020, gate_identity=0.005
```

All were run single-seed, 1000 steps, GPU 1.

## 3. Results

| Variant | blocks | gate | gate_identity | UEFB PSNR↑ | Real PSNR↑ | Synthetic PSNR↑ | UEFB E_corr↑ | Real E_corr↑ | Synthetic E_corr↑ | Promote? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P6C_MS_GATEID005 | 3 | 0.020 | 0.005 | 17.604 | 20.456 | 17.550 | 0.455 | 0.598 | 0.834 | No |
| P6C_MS_GATEID010 | 3 | 0.020 | 0.010 | 17.651 | 19.818 | 17.890 | 0.495 | 0.671 | 0.855 | No |
| P6C_MS_GATELOW005 | 3 | 0.005 | 0.000 | 18.046 | 19.831 | 17.422 | 0.489 | 0.582 | 0.779 | No |
| P6C_MS_B2_GATEID005 | 2 | 0.020 | 0.005 | 17.561 | 19.314 | 17.103 | 0.407 | 0.600 | 0.764 | No |

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
| P6C_MS_GATEID005 | -0.310 | +0.436 | -0.128 | +0.018 | -0.110 | -0.006 |
| P6C_MS_GATEID010 | -0.264 | -0.203 | +0.211 | +0.059 | -0.036 | +0.014 |
| P6C_MS_GATELOW005 | +0.131 | -0.189 | -0.256 | +0.053 | -0.125 | -0.062 |
| P6C_MS_B2_GATEID005 | -0.354 | -0.707 | -0.575 | -0.029 | -0.107 | -0.077 |

## 4. Interpretation

P6c gives useful diagnostics but still no balanced winner.

Best real PSNR:

```text
P6C_MS_GATEID005: real PSNR 20.456, synthetic PSNR 17.550, UEFB PSNR 17.604
```

Best synthetic PSNR and synthetic E_corr:

```text
P6C_MS_GATEID010: synthetic PSNR 17.890, synthetic E_corr 0.855
```

Best UEFB PSNR:

```text
P6C_MS_GATELOW005: UEFB PSNR 18.046
```

The key pattern:

```text
- gate_identity=0.005 improves real strongly but does not rescue synthetic or UEFB.
- gate_identity=0.010 rescues synthetic and improves synthetic E_corr, but real/UEFB remain below P3c.
- lowering gate supervised weight to 0.005 improves UEFB, but real/synthetic drop.
- reducing backbone_blocks to 2 hurts all primary PSNR metrics.
```

So the P6/P6c trade-off is not solved by simple A-gate scalar penalties.

## 5. Relation to P6, P6b, and official baselines

| Method | UEFB PSNR | Real PSNR | Synthetic PSNR | Real E_corr | Synthetic E_corr | Decision |
|---|---:|---:|---:|---:|---:|---|
| P3c default mean | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | 0.707±0.053 | 0.841±0.037 | conservative default |
| P6_MS_M4J_ES | 18.015 | 20.197 | 17.598 | 0.629 | 0.812 | best structural PSNR evidence, synthetic miss |
| P6B best synthetic (P6B_MS_SYN125) | 16.894 | 19.482 | 19.115 | 0.679 | 0.793 | synthetic rescued, UEFB/real harmed |
| P6C best balanced (P6C_MS_GATELOW005) | 18.046 | 19.831 | 17.422 | 0.582 | 0.779 | no promotion |
| Retinexformer official | — | 22.794 | 25.669 | — | — | strong baseline gap remains |

## 6. Go / no-go

```text
No-go for P6c 3-seed promotion.
```

Promotion criterion required UEFB, real, and synthetic PSNR all to exceed P3c mean. No P6c variant satisfies this:

```text
P6C_MS_GATEID005: real passes, UEFB/synthetic fail.
P6C_MS_GATEID010: synthetic passes, UEFB/real fail.
P6C_MS_GATELOW005: UEFB passes, real/synthetic fail.
P6C_MS_B2_GATEID005: all primary PSNR metrics fail.
```

## 7. Claim decision

Safe claim:

```text
A-gate identity protection can shift the P6 trade-off between real fidelity, synthetic fidelity, and exposure-field correlation, but simple scalar gate penalties do not produce a balanced improvement over the P3c default.
```

Current default remains:

```text
P3c M4J_ES e_shape=0.05
```

P6 remains the most useful structural evidence; P6b/P6c show that the synthetic issue is controllable but not solved by simple scalar reweighting or scalar gate regularization.

## 8. Next recommendation

Stop scalar-only controls for this branch. Suggested next route:

```text
P7-MS-DOMAINHEAD
- keep multiscale trunk evidence from P6
- avoid global scalar rec/gate knobs
- add tiny domain-conditioned adapter/head or domain-conditioned gate bias
- objective: learn domain-specific calibration without forcing one global gate/reconstruction trade-off
- run same promotion rule: UEFB, real, and synthetic PSNR must all exceed P3c mean before 3-seed
```

A lower-risk alternative is to pause architecture tinkering and produce the current paper-quality negative/diagnostic report around P3c/P6/P6b/P6c.
