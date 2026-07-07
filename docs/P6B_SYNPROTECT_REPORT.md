# P6b-MS-SYNPROTECT Report

生成时间：2026-07-01T21:47:14  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有 GitHub 提交/推送，也没有本地 git commit。

## 1. Purpose

P6 的 multiscale backbone 提升了 UEFB/real PSNR，但 synthetic PSNR 比 P3c default mean 低 `0.081 dB`。P6b-MS-SYNPROTECT 因此测试一个最小 synthetic-protection 控制：

```text
keep P6 multiscale backbone
keep M4J_ES E/A-gate/E-shape heads
no Retinexformer teacher / no output-level distillation
replace scalar rec with rec_by_dataset
upweight only LOL-v2-synthetic-train reconstruction
```

实现动机：如果 P6 的 synthetic miss 只是 paired synthetic supervision 不够强，轻量 synthetic reconstruction upweighting 应能拉回 synthetic 且不显著牺牲 UEFB/real。

## 2. Implementation

新增/修改：

```text
src/rlef/losses/total_loss.py
  - weights['rec_by_dataset']
  - stable rec_<dataset> scalar logging for all configured domains

scripts/run_p6b_synprotect.py
  - P6b synthetic-protection sweep runner
  - --gpu_start to skip busy GPU 0 when another long job occupies it

tests/test_p6b_synprotect_runner_contract.py
  - P6b route/config/promotion/GPU assignment contract
```

本轮运行时 GPU 0 被另一个长期 CutLER export 进程占用，第一次 P6b 子进程真实 OOM；已记录在 log 中。随后通过 `--gpu_start 1` 在 GPU 1 上成功完成所有 P6b runs。

## 3. Results

| Variant | synthetic rec weight | UEFB PSNR↑ | Real PSNR↑ | Synthetic PSNR↑ | UEFB E_corr↑ | Real E_corr↑ | Synthetic E_corr↑ | Promote? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P6B_MS_SYN105 | 1.05 | 17.347 | 19.996 | 18.708 | 0.486 | 0.644 | 0.785 | No |
| P6B_MS_SYN110 | 1.10 | 17.220 | 19.569 | 18.948 | 0.459 | 0.680 | 0.818 | No |
| P6B_MS_SYN125 | 1.25 | 16.894 | 19.482 | 19.115 | 0.475 | 0.679 | 0.793 | No |
| P6B_MS_SYN150 | 1.50 | 16.452 | 19.337 | 19.097 | 0.468 | 0.682 | 0.785 | No |

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
| P6B_MS_SYN105 | -0.567 | -0.025 | +1.029 | +0.050 | -0.064 | -0.056 |
| P6B_MS_SYN110 | -0.694 | -0.452 | +1.270 | +0.023 | -0.027 | -0.023 |
| P6B_MS_SYN125 | -1.020 | -0.538 | +1.437 | +0.039 | -0.029 | -0.047 |
| P6B_MS_SYN150 | -1.462 | -0.684 | +1.419 | +0.032 | -0.025 | -0.055 |

## 4. Interpretation

P6b succeeded at increasing synthetic PSNR, but it over-corrected the training objective:

```text
Best synthetic PSNR: P6B_MS_SYN125 = 19.115
P3c synthetic mean: 17.678±0.735
P6 synthetic:       17.598
```

However, every synthetic-protection variant lost UEFB and real PSNR relative to P3c:

```text
Best-balanced P6b candidate: P6B_MS_SYN105
UEFB PSNR: 17.347 vs P3c 17.915±0.275
Real PSNR: 19.996 vs P3c 20.021±0.223
Synthetic PSNR: 18.708 vs P3c 17.678±0.735
```

Thus dataset-weighted reconstruction is too blunt: it shifts capacity toward synthetic restoration and away from UEFB/real balance. Even the very light `1.05` synthetic weight improves synthetic strongly but drops UEFB by `0.567 dB` and real by `0.025 dB`.

## 5. Relation to P6 and official baselines

| Method | UEFB PSNR | Real PSNR | Synthetic PSNR | Real E_corr | Synthetic E_corr | Decision |
|---|---:|---:|---:|---:|---:|---|
| P3c default mean | 17.915±0.275 | 20.021±0.223 | 17.678±0.735 | 0.707±0.053 | 0.841±0.037 | conservative default |
| P6_MS_M4J_ES | 18.015 | 20.197 | 17.598 | 0.629 | 0.812 | best structural evidence, not promoted |
| P6B_MS_SYN105 | 17.347 | 19.996 | 18.708 | 0.644 | 0.785 | synthetic rescued, UEFB/real hurt |
| P6B_MS_SYN125 | 16.894 | 19.482 | 19.115 | 0.679 | 0.793 | over-protection |
| Retinexformer official | — | 22.794 | 25.669 | — | — | strong baseline gap remains |

## 6. Go / no-go

```text
No-go for P6b 3-seed promotion.
```

Promotion criterion required UEFB, real, and synthetic PSNR to all beat P3c mean. P6b variants beat synthetic but fail UEFB and real. Therefore the correct conclusion is not “increase synthetic weight more”, but “dataset-weighted rec is an over-correction mechanism.”

## 7. Claim decision

Safe claim:

```text
Synthetic-specific reconstruction upweighting can rescue synthetic PSNR under the multiscale RLEF backbone, but it substantially harms UEFB and real performance. This confirms that the P6 synthetic issue is controllable, but naive dataset-weighted reconstruction is not a balanced solution.
```

Current default remains:

```text
P3c M4J_ES e_shape=0.05
```

P6 remains the better structural direction than P6b, but still needs a less blunt synthetic-protection mechanism.

## 8. Next recommendation

Do **not** run more scalar synthetic-rec weights. The 1.05 result already shows this axis is too sensitive.

Recommended next branch:

```text
P6c-MS-GATEPROTECT
- keep multiscale backbone
- no teacher/output-level distillation
- do not upweight full synthetic reconstruction
- instead constrain the gate/restoration aggressiveness:
  1. lower gate loss weight or add A sparsity/identity protection,
  2. optionally reduce backbone_blocks from 3 to 2,
  3. keep synthetic objective scalar unchanged.
- promote only if UEFB, real, and synthetic PSNR all exceed P3c mean.
```
