# Figure integration notes

Camera-ready integration package created in:

```text
docs/paper/gir_field/figures/
```

For each formal-protocol PNG, both the original PNG copy and a 300-DPI PDF wrapper were written. The paper skeleton now uses:

```tex
\graphicspath{{figures/}}
```

and `sections/figures.tex` references the PDF versions:

- `fig_psnr_vs_scorr.pdf`
- `fig_gauge_perturbation_bar.pdf`
- `fig_psnr_misranking_failure_grid.pdf`
- `fig_risk_reliability.pdf`

Asset manifest:

```text
docs/paper/gir_field/figures/FIGURE_ASSET_MANIFEST.json
```

Note: the PDFs are camera-ready wrappers around the generated raster figures, not newly redrawn vector graphics. If a conference requires vector-only plots, regenerate `fig_psnr_vs_scorr` and `fig_gauge_perturbation_bar` directly from CSV using Matplotlib PDF/SVG backends.
