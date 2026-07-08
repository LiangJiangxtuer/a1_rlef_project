# I2 External field-aware adapter study

Date: 2026-07-08

## Scope

This study evaluates whether frozen P4 external baseline outputs can be mapped into UEFB-G field-aware reporting through a defensible luminance-gain adapter.

Adapter definition:

```text
E_hat = log(Y_pred + eps) - log(Y_low + eps)
E_gt  = log(Y_high + eps) - log(Y_low + eps)
```

Two adapter variants are reported:

- `log_ratio_raw`: raw per-pixel luminance-gain proxy.
- `log_ratio_lowpass_r8`: dependency-light box low-pass illumination-scale proxy, radius 8 px.

Important claim guardrail: these are adapter-derived proxies. They do not prove that Retinexformer, Zero-DCE++, or KinD++ internally predicts this exact field.

## Execution summary

- Per-image rows: 1200
- UEFB-G validation status: `PASS`
- Methods: KinD++, Retinexformer, Zero-DCE++
- Datasets: real, synthetic

## Summary table

| Method | Dataset | Adapter | n | PSNR↑ | S_corr↑ | S_MAE↓ | Gauge_MAE↓ | E_MAE↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| KinD++ | real | log_ratio_lowpass_r8 | 100 | 22.2109 | 0.9445 | 0.0833 | 0.1161 | 0.1519 |
| KinD++ | real | log_ratio_raw | 100 | 22.2109 | 0.8921 | 0.1230 | 0.1163 | 0.1860 |
| KinD++ | synthetic | log_ratio_lowpass_r8 | 100 | 19.2594 | 0.9228 | 0.1683 | 0.1944 | 0.2458 |
| KinD++ | synthetic | log_ratio_raw | 100 | 19.2594 | 0.8674 | 0.2378 | 0.1957 | 0.2821 |
| Retinexformer | real | log_ratio_lowpass_r8 | 100 | 22.7939 | 0.9547 | 0.0591 | 0.1827 | 0.1904 |
| Retinexformer | real | log_ratio_raw | 100 | 22.7939 | 0.9382 | 0.0899 | 0.1828 | 0.2009 |
| Retinexformer | synthetic | log_ratio_lowpass_r8 | 100 | 25.6690 | 0.9893 | 0.0657 | 0.1265 | 0.1362 |
| Retinexformer | synthetic | log_ratio_raw | 100 | 25.6690 | 0.9814 | 0.0878 | 0.1265 | 0.1461 |
| Zero-DCE++ | real | log_ratio_lowpass_r8 | 100 | 18.4907 | 0.6590 | 0.1744 | 0.4859 | 0.4965 |
| Zero-DCE++ | real | log_ratio_raw | 100 | 18.4907 | 0.3983 | 0.2519 | 0.4859 | 0.5100 |
| Zero-DCE++ | synthetic | log_ratio_lowpass_r8 | 100 | 17.5764 | 0.7998 | 0.2912 | 0.4812 | 0.4983 |
| Zero-DCE++ | synthetic | log_ratio_raw | 100 | 17.5764 | 0.7285 | 0.3436 | 0.4810 | 0.5068 |

## Interpretation

Blind external methods with lowpass adapter `S_corr >= 0.5`: ['Retinexformer', 'Zero-DCE++'].

**GO.** At least two blind external methods produce nontrivial adapter-field shape signal, so UEFB-G can be framed as applicable beyond the internal RLEF family when adapter scope is clearly stated.

## Artifact list

```text
external_adapter_per_image.csv
external_adapter_summary.csv
uefbg_v1_bundle/
EXTERNAL_FIELD_ADAPTER_STUDY_REPORT.md
```
