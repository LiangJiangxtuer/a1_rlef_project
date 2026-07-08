# UEFB-G Baseline Submission Format

A UEFB-G v1 submission is a CSV file with one row per method / dataset / image.

## Minimal output-only row

```csv
variant_id,dataset,name,psnr,ssim
Retinexformer,real,0001,22.79,0.85
```

This is appropriate for black-box enhancers that do not expose an internal exposure field.

## Field-aware row

```csv
variant_id,dataset,name,psnr,ssim,E_MAE,S_MAE,Gauge_MAE,S_corr
M4J_ES,uefb,test_00001,17.98,0.80,0.29,0.18,0.22,0.44
```

A field-aware submission must provide all four field metrics:

```text
E_MAE
S_MAE
Gauge_MAE
S_corr
```

Providing only `S_corr` or only `Gauge_MAE` is invalid because it prevents gauge-vs-shape failure separation.

## Public evaluator command

```bash
python scripts/run_uefbg_benchmark_v1.py \
  --input-metrics your_submission.csv \
  --protocol configs/uefbg/protocol_v1.yaml \
  --out results/uefbg_v1/your_submission
```

## Expected output

```text
summary.csv
summary.json
validation.json
report_cards.md
manifest.json
```

`validation.json` is the first file to inspect. If it reports `FAIL`, the report cards should not be used for paper claims until the schema errors are fixed.
