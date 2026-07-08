# UEFB-G v1 report cards

Validation status: `PASS`

Rows: 1200 | Variants: 6 | Datasets: 2

| variant_id | dataset | n | reporting_mode | psnr_mean | ssim_mean | S_corr_mean | Gauge_MAE_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KinD++__log_ratio_lowpass_r8 | real | 100 | field_aware | 22.2109 | 0.8411 | 0.9445 | 0.1161 |
| KinD++__log_ratio_lowpass_r8 | synthetic | 100 | field_aware | 19.2594 | 0.8047 | 0.9228 | 0.1944 |
| KinD++__log_ratio_raw | real | 100 | field_aware | 22.2109 | 0.8411 | 0.8921 | 0.1163 |
| KinD++__log_ratio_raw | synthetic | 100 | field_aware | 19.2594 | 0.8047 | 0.8674 | 0.1957 |
| Retinexformer__log_ratio_lowpass_r8 | real | 100 | field_aware | 22.7939 | 0.8421 | 0.9547 | 0.1827 |
| Retinexformer__log_ratio_lowpass_r8 | synthetic | 100 | field_aware | 25.6690 | 0.9285 | 0.9893 | 0.1265 |
| Retinexformer__log_ratio_raw | real | 100 | field_aware | 22.7939 | 0.8421 | 0.9382 | 0.1828 |
| Retinexformer__log_ratio_raw | synthetic | 100 | field_aware | 25.6690 | 0.9285 | 0.9814 | 0.1265 |
| Zero-DCE++__log_ratio_lowpass_r8 | real | 100 | field_aware | 18.4907 | 0.5931 | 0.6590 | 0.4859 |
| Zero-DCE++__log_ratio_lowpass_r8 | synthetic | 100 | field_aware | 17.5764 | 0.8106 | 0.7998 | 0.4812 |
| Zero-DCE++__log_ratio_raw | real | 100 | field_aware | 18.4907 | 0.5931 | 0.3983 | 0.4859 |
| Zero-DCE++__log_ratio_raw | synthetic | 100 | field_aware | 17.5764 | 0.8106 | 0.7285 | 0.4810 |

## Reporting rules

- `output_only` rows are valid for black-box enhancers and report only RGB/output metrics.
- `field_aware` rows must provide all core field metrics: `E_MAE`, `S_MAE`, `Gauge_MAE`, and `S_corr`.
- Partial field reports are rejected to avoid mixing output-only and internal-field claims.
