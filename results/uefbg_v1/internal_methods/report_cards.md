# UEFB-G v1 report cards

Validation status: `PASS`

Rows: 6300 | Variants: 9 | Datasets: 3

| variant_id | dataset | n | reporting_mode | psnr_mean | ssim_mean | S_corr_mean | Gauge_MAE_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M4 | real | 100 | field_aware | 17.4113 | 0.6996 | -0.3314 | 0.7004 |
| M4 | synthetic | 100 | field_aware | 13.9632 | 0.6284 | -0.3372 | 0.9916 |
| M4 | uefb | 500 | field_aware | 18.6530 | 0.8169 | -0.1393 | 0.1564 |
| M4J | real | 100 | field_aware | 19.5807 | 0.7827 | 0.5007 | 0.5253 |
| M4J | synthetic | 100 | field_aware | 16.5085 | 0.7343 | 0.3690 | 0.6533 |
| M4J | uefb | 500 | field_aware | 18.0818 | 0.8063 | 0.2818 | 0.1760 |
| M4J_ES | real | 100 | field_aware | 20.2770 | 0.7919 | 0.6483 | 0.5369 |
| M4J_ES | synthetic | 100 | field_aware | 17.1405 | 0.7431 | 0.7992 | 0.6989 |
| M4J_ES | uefb | 500 | field_aware | 17.9811 | 0.8039 | 0.4399 | 0.2222 |
| P3C_E0050_S2027 | real | 100 | field_aware | 19.9145 | 0.7748 | 0.7256 | 0.2258 |
| P3C_E0050_S2027 | synthetic | 100 | field_aware | 17.3792 | 0.7453 | 0.8551 | 0.5188 |
| P3C_E0050_S2027 | uefb | 500 | field_aware | 18.1498 | 0.8094 | 0.4346 | 0.1778 |
| P3C_E0050_S3407 | real | 100 | field_aware | 20.2770 | 0.7919 | 0.6483 | 0.5369 |
| P3C_E0050_S3407 | synthetic | 100 | field_aware | 17.1405 | 0.7431 | 0.7992 | 0.6989 |
| P3C_E0050_S3407 | uefb | 500 | field_aware | 17.9811 | 0.8039 | 0.4399 | 0.2222 |
| P3C_E0050_S42 | real | 100 | field_aware | 19.8703 | 0.7788 | 0.7485 | 0.2217 |
| P3C_E0050_S42 | synthetic | 100 | field_aware | 18.5151 | 0.7725 | 0.8679 | 0.3011 |
| P3C_E0050_S42 | uefb | 500 | field_aware | 17.6127 | 0.8080 | 0.4340 | 0.1763 |
| P3D_E0100_S2027 | real | 100 | field_aware | 19.3869 | 0.7674 | 0.6970 | 0.2468 |
| P3D_E0100_S2027 | synthetic | 100 | field_aware | 17.0398 | 0.7408 | 0.8644 | 0.4134 |
| P3D_E0100_S2027 | uefb | 500 | field_aware | 17.9150 | 0.8045 | 0.4464 | 0.1925 |
| P3D_E0100_S3407 | real | 100 | field_aware | 20.1792 | 0.7847 | 0.6305 | 0.4879 |
| P3D_E0100_S3407 | synthetic | 100 | field_aware | 17.8209 | 0.7589 | 0.8007 | 0.5211 |
| P3D_E0100_S3407 | uefb | 500 | field_aware | 18.1006 | 0.8067 | 0.4570 | 0.1984 |
| P3D_E0100_S42 | real | 100 | field_aware | 20.0587 | 0.7808 | 0.7292 | 0.2546 |
| P3D_E0100_S42 | synthetic | 100 | field_aware | 18.0181 | 0.7648 | 0.8388 | 0.3451 |
| P3D_E0100_S42 | uefb | 500 | field_aware | 18.3167 | 0.8126 | 0.4438 | 0.1811 |

## Reporting rules

- `output_only` rows are valid for black-box enhancers and report only RGB/output metrics.
- `field_aware` rows must provide all core field metrics: `E_MAE`, `S_MAE`, `Gauge_MAE`, and `S_corr`.
- Partial field reports are rejected to avoid mixing output-only and internal-field claims.
