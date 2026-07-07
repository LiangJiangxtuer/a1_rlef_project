# P4 Official Baseline Comparison Report

生成时间：2026-07-01T16:46:14  
项目根目录：`/home/user/a1_rlef_project`

> 按用户偏好：本轮没有 GitHub 提交/推送，也没有本地 git commit。

## 1. Purpose

P4 的目标是把当前默认路线 `RLEF M4J_ES e_shape=0.05` 放到官方低光增强 baseline 语境中校准 claim。P3d 已验证 `e_shape=0.10` 不应升为默认，因此 P4 默认仍为 P3c 的三 seed `e_shape=0.05`。

## 2. Executed official baselines

| Method | Source | Checkpoint/runtime status |
|---|---|---|
| Retinexformer | `caiyuanhao1998/Retinexformer` official repo | official `LOL_v2_real.pth` / `LOL_v2_synthetic.pth` downloaded and executed |
| Zero-DCE++ | `Li-Chongyi/Zero-DCE_extension` official repo | official `Epoch99.pth` already present and executed with official model definition |
| KinD++ | `zhangyhuaee/KinD_plus` official repo | official checkpoint zip downloaded/extracted; TensorFlow CPU venv created; executed |

Important protocol note: KinD++ official `evaluate_LOLdataset.py` uses high-image information to compute illumination adjustment ratio (`decom_i_high` / `decom_i_low` and `decom_r_high` / restoration). Therefore KinD++ numbers are reported as official-code results but are **high-assisted / non-blind** under this script. Retinexformer and Zero-DCE++ runs here are blind at test time.

## 3. Uniform evaluator outputs

All method outputs were re-evaluated by `scripts/run_p4_official_baselines.py` using the same project PSNR/SSIM/LEE/NAI functions. This avoids mixing printed metrics from different repositories.

Summary files:

```text
results/tables/p4_official_baselines_summary.csv
results/tables/p4_official_baselines_summary.json
results/tables/p4_details/*_per_image.csv
```

### LOL-v2-real test

| Method | PSNR↑ | SSIM↑ | LEE↓ | NAI↓ | Protocol note |
|---|---:|---:|---:|---:|---|
| Retinexformer | 22.794 | 0.842 | 0.186 | 3.215 | official blind |
| KinD++ | 22.211 | 0.841 | 0.148 | 4.610 | official KinD++ high-assisted ratio (uses high image during eval) |
| Zero-DCE++ | 18.491 | 0.593 | 0.445 | 4.984 | official blind |
| RLEF M4J_ES e=0.05 (ours, 3-seed mean) | 20.021±0.223 | — | — | — | blind, trained 1000-step research prototype |
| RLEF M4J_ES e=0.10 (ours, 3-seed mean) | 19.875±0.427 | — | — | — | blind, ablation; not default |

### LOL-v2-synthetic test

| Method | PSNR↑ | SSIM↑ | LEE↓ | NAI↓ | Protocol note |
|---|---:|---:|---:|---:|---|
| Retinexformer | 25.669 | 0.929 | 0.125 | 2.160 | official blind |
| KinD++ | 19.259 | 0.805 | 0.215 | 2.587 | official KinD++ high-assisted ratio (uses high image during eval) |
| Zero-DCE++ | 17.576 | 0.811 | 0.413 | 2.334 | official blind |
| RLEF M4J_ES e=0.05 (ours, 3-seed mean) | 17.678±0.735 | — | — | — | blind, trained 1000-step research prototype |
| RLEF M4J_ES e=0.10 (ours, 3-seed mean) | 17.626±0.517 | — | — | — | blind, ablation; not default |

## 4. Interpretation

### What survives

- RLEF M4J_ES remains an interpretable research prototype with stable positive E-field diagnostic correlations from P3c/P3d.
- RLEF M4J_ES clearly beats Zero-DCE++ in this local full-test protocol:
  - Real: RLEF e=0.05 mean 20.021±0.223 vs Zero-DCE++ 18.491
  - Synthetic: RLEF e=0.05 mean 17.678±0.735 vs Zero-DCE++ 17.576
- P3d confirms `e_shape=0.10` is a useful E-correlation ablation but not the default.

### What does not survive

- RLEF M4J_ES is not yet competitive with Retinexformer:
  - Real: Retinexformer 22.794 vs RLEF e=0.05 20.021±0.223
  - Synthetic: Retinexformer 25.669 vs RLEF e=0.05 17.678±0.735
- RLEF is also below KinD++ official-code numbers, but KinD++ uses high-assisted evaluation in this script, so this is not a strictly blind-to-blind comparison.

## 5. Claim decision

Safe claim after P4:

```text
RLEF M4J_ES is not SOTA against strong supervised restoration baselines such as Retinexformer, but it provides a compact interpretable exposure-field route with stable positive E-field diagnostics and better local full-test PSNR than Zero-DCE++ under the executed protocol.
```

Do not claim:

```text
SOTA / Retinexformer-level performance
Blind superiority over KinD++
Real physical E-field correctness on real images
```

Recommended next research branch:

```text
P5: keep the RLEF exposure/A-gate/E-shape heads as auxiliary explainability/calibration modules, but attach them to a stronger restoration backbone or distill from Retinexformer outputs. The paper story should be “interpretable exposure-field auxiliary supervision/calibration”, not raw restoration SOTA from the tiny current backbone.
```
