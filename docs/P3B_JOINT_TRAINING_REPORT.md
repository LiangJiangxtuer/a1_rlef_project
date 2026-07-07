# P3b Formal Report: M4 joint-training and E-shape repair diagnostic

生成时间：2026-07-01T11:53:22  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有主动 GitHub 提交/推送，也没有本地 git commit。当前结果保留在工作区，等待用户明确要求再做提交/上传。

## 1. 目标

P2/P3 结论显示：M4 gate 是 UEFB-only 训练下最强图像质量模块，但 LOL-v2 paired PSNR 偏低，且 E_corr 为负。P3b 因此执行两个下一步：

1. **M4 joint-training**：训练集从 UEFB-only 扩展到 `UEFB-v2 + LOL-v2-real + LOL-v2-synthetic`，检查是否能保留 UEFB 机制指标同时提升真实/合成 paired fidelity。
2. **E-shape repair diagnostic**：在 M4 joint-training 上加入 low-pass centered correlation shape loss：`e_shape=0.05, kernel=7`，检查是否能修复 `E_corr < 0`。

## 2. Protocol

执行脚本：`scripts/run_p3b_joint.py`

```bash
/home/user/miniconda3/envs/cutler_dinov3/bin/python scripts/run_p3b_joint.py   --max_steps 1000 --train_crop 128 --batch_size 8 --parallel 2
```

训练设置：

- seed: 3407
- steps: 1000
- train crop: 128
- batch size: 8
- train datasets:
  - `data/UEFB-v2/train`，3000 samples
  - `data/LOL-v2/Real_captured/Train`，689 pairs
  - `data/LOL-v2/Synthetic/Train`，900 pairs
- primary eval: `data/UEFB-v2/test`，500 samples
- paired diagnostics:
  - LOL-v2-real full test
  - LOL-v2-synthetic full test

实现变化：

- `PairedRGBDataset` 现在为 paired RGB 样本动态提供 oracle diagnostic maps：`E_gt/A_gt/Q_gt`，使 joint batches 与 UEFB batches 拥有一致 key，并允许 paired diagnostics 输出 E 指标。
- `total_loss.py` 新增 `e_shape`：low-pass centered correlation loss，用于空间 E-shape 修复诊断。

## 3. Results

结果表：`results/tables/p3b_joint_m4_summary.csv`

| Variant | Name | UEFB PSNR↑ | E_MAE↓ | E_aligned↓ | E_corr↑ | Real PSNR↑ | Real E_corr↑ | Synthetic PSNR↑ | Synthetic E_corr↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M4J | m4_joint | 18.082 | 0.278 | 0.215 | 0.282 | 19.581 | 0.501 | 16.508 | 0.369 |
| M4J_ES | m4_joint_eshape | 17.981 | 0.292 | 0.200 | 0.440 | 20.277 | 0.648 | 17.141 | 0.799 |

参考 P3 UEFB-only M4：

| Variant | UEFB PSNR | E_MAE | E_aligned | E_corr | Real PSNR | Synthetic PSNR |
|---|---:|---:|---:|---:|---:|---:|
| P3 M4 UEFB-only | 18.653 | 0.292 | 0.247 | -0.139 | 17.411 | 13.963 |

## 4. Findings

### 4.1 Joint training solves the paired-fidelity bottleneck directionally

M4 joint-training 相对 P3 UEFB-only M4：

- UEFB PSNR: 18.653 → 18.082 (-0.571 dB)
- Real PSNR: 17.411 → 19.581 (+2.169 dB)
- Synthetic PSNR: 13.963 → 16.508 (+2.545 dB)
- UEFB E_corr: -0.139 → 0.282

解释：joint-training 牺牲了约 0.571 dB UEFB PSNR，但显著提升 real/synthetic paired PSNR，并且把 UEFB E_corr 从负值拉到正值。说明 M4 的主路线应从 UEFB-only 转向 joint-training。

### 4.2 E-shape repair diagnostic 有强正信号

M4J_ES 相对 M4J：

- UEFB E_corr: 0.282 → 0.440 (+0.158)
- Real E_corr: 0.501 → 0.648 (+0.148)
- Synthetic E_corr: 0.369 → 0.799 (+0.430)
- Real PSNR: 19.581 → 20.277 (+0.696 dB)
- Synthetic PSNR: 16.508 → 17.141 (+0.632 dB)
- UEFB PSNR: 18.082 → 17.981 (-0.101 dB)

解释：`e_shape` 明显提升 E_corr，并且提升 real/synthetic PSNR；代价是 UEFB PSNR 小幅下降 0.101 dB、absolute E_MAE 小幅变差。该结果支持把 E-shape repair 从 diagnostic 升级为下一轮正式候选，但仍需 3-seed 验证。

### 4.3 Claim boundary

当前可以写：

```text
Joint training with UEFB and paired LOL-v2 data substantially improves paired fidelity, while a low-pass centered E-shape consistency diagnostic turns previously negative E-field correlations positive on UEFB and paired diagnostics.
```

当前不能写：

- SOTA 或 Retinexformer-level，因为 P4 official baseline 还没跑；
- 真实图像物理 E-field 完全正确，因为 LOL-v2 的 E 指标是 oracle diagnostic，不是真实物理 GT；
- 多 seed 稳定性，因为本轮仍是 single-seed routing。

## 5. Go/No-Go

| Route | Decision | Reason |
|---|---|---|
| M4 UEFB-only | 降级为机制 baseline | UEFB 强，但 paired PSNR 明显低 |
| M4 joint | 保留 | paired PSNR 大幅提升，E_corr 转正 |
| M4 joint + e_shape | 强保留，进入 3-seed | E_corr 与 paired PSNR 同时提升，UEFB PSNR 仅小幅下降 |
| M5/Q | 暂不继续扩展 | P3 已显示 Q_ECE 小幅改善但牺牲 PSNR，优先先验证 M4J_ES |

## 6. Next recommended step

1. 对 **M4J_ES** 跑 3 seeds：3407, 2027, 42，验证 E_corr/PSNR 是否稳定。
2. 若 3 seeds 成立，再接入 P4 official baselines（Retinexformer / Zero-DCE++ / KinD++）同协议比较。
3. 同时做 small sweep：`e_shape = 0.02, 0.05, 0.10`，确认 UEFB PSNR 与 E_corr 的 trade-off。
