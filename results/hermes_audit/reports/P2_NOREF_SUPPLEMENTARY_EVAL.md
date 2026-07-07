# P2 No-Reference Supplementary Evaluation

Generated: 2026-07-05T03:06:24.265656+00:00

## Scope

Frozen-artifact supplementary evaluation on local `data/unpaired_real` images. This stage does **not** train a new model and does **not** create a SOTA claim.

```text
n_unpaired_images = 128
methods = input_low, P3c_RLEF_seed3407
NIQE = available: Retinexformer basicsr NIQE implementation
BRISQUE = unavailable locally; not reported/fabricated
```

## Summary

| dataset | method | n | niqe | mean_luma | over | under | dark_ratio | contrast | sharpness_proxy | noise_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DICM | P3c_RLEF_seed3407 | 69.000 | 4.445 | 0.419 | 0.008 | 0.000 | 0.122 | 0.228 | 331.489 | 0.010 |
| DICM | input_low | 69.000 | 4.406 | 0.325 | 0.027 | 0.152 | 0.326 | 0.219 | 530.984 | 0.014 |
| LIME | P3c_RLEF_seed3407 | 10.000 | 4.995 | 0.318 | 0.000 | 0.000 | 0.123 | 0.184 | 351.539 | 0.008 |
| LIME | input_low | 10.000 | 4.648 | 0.144 | 0.003 | 0.179 | 0.572 | 0.153 | 569.541 | 0.015 |
| MEF | P3c_RLEF_seed3407 | 17.000 | 4.637 | 0.310 | 0.000 | 0.000 | 0.165 | 0.219 | 217.590 | 0.006 |
| MEF | input_low | 17.000 | 4.815 | 0.163 | 0.006 | 0.232 | 0.606 | 0.191 | 371.427 | 0.010 |
| NPE | P3c_RLEF_seed3407 | 8.000 | 4.342 | 0.434 | 0.000 | 0.000 | 0.085 | 0.251 | 425.496 | 0.019 |
| NPE | input_low | 8.000 | 4.521 | 0.341 | 0.004 | 0.066 | 0.299 | 0.256 | 602.635 | 0.021 |
| VV | P3c_RLEF_seed3407 | 24.000 | 4.064 | 0.354 | 0.000 | 0.000 | 0.228 | 0.271 | 153.832 | 0.004 |
| VV | input_low | 24.000 | 3.955 | 0.260 | 0.017 | 0.201 | 0.488 | 0.274 | 263.865 | 0.005 |

## Interpretation rule

No-reference metrics are auxiliary only. If they support the visual story, cite them as supplementary evidence; if they conflict with paired PSNR/visual audit, use them as a limitation.

## Saved artifacts

- `results/hermes_audit/tables/noref_supplementary_summary.csv`
- `results/hermes_audit/tables/noref_supplementary_summary.json`
- `results/hermes_audit/tables/noref_supplementary_per_image.csv`
- `results/hermes_audit/figures/noref_unpaired_grid.png`
