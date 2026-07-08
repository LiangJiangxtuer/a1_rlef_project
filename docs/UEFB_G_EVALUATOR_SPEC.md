# UEFB-G Evaluator Specification

Version: v1

## Entrypoints

Library:

```python
from scripts.uefbg_eval import compute_gauge_metrics_np, write_evaluation_bundle
```

CLI:

```bash
python scripts/run_uefbg_benchmark_v1.py \
  --input-metrics results/girfield_formal/N1_statistics/per_image_metrics.csv \
  --protocol configs/uefbg/protocol_v1.yaml \
  --out results/uefbg_v1/internal_methods
```

## Submission table schema

Required identity columns:

```text
variant_id,dataset,name
```

Recommended metadata columns:

```text
display,role,seed,e_shape,sample_index
```

Output-only metrics:

```text
psnr,ssim,lee,nai,input_psnr,identity_drop,q_ece,over,under
```

Field-aware metrics:

```text
E_MAE,S_MAE,Gauge_MAE,S_corr
```

Optional auxiliary field columns:

```text
E_MAE_aligned,E_corr,mu_pred,mu_gt,A_mean,Q_mean,mu_E
```

## Validation rules

1. Every row must have `variant_id`, `dataset`, and `name`.
2. Every row must contain at least one output metric.
3. A row with none of the field metric quartet is `output_only`.
4. A row with all of the field metric quartet is `field_aware`.
5. A row with only part of the field metric quartet is rejected as `partial_field_metrics`.

## Outputs

The evaluator writes:

```text
input_metrics.csv
summary.csv
summary.json
validation.json
report_cards.md
manifest.json
```

## Metric formulas

For exposure prediction `E_pred` and target `E_gt`:

```text
μ_pred = mean(E_pred)
μ_gt   = mean(E_gt)
S_pred = E_pred - μ_pred
S_gt   = E_gt - μ_gt

E_MAE     = mean(abs(E_pred - E_gt))
S_MAE     = mean(abs(S_pred - S_gt))
Gauge_MAE = abs(μ_pred - μ_gt)
S_corr    = dot(S_pred, S_gt) / (||S_pred|| ||S_gt|| + eps)
```

## Interpretation

- High PSNR with negative or low `S_corr` indicates output/field mismatch.
- Good `S_corr` with poor `Gauge_MAE` indicates shape is right but global gauge is wrong.
- Output-only black-box methods are valid baselines but are not assigned field-aware scores.
