# GIR-Field continuation completion report

Continuation request addressed the previously listed unfinished tasks:

1. Add compact theorem/proposition box for gauge identifiability.
2. Add official citations / BibTeX after metadata verification.
3. Convert diagnostic figures into camera-ready figure panels.
4. Add appendix tables for DGB/P5/P6/P7 negative-route audit.

## New / updated files

### Theory integration

- Updated: `sections/method.tex`
  - Added Proposition: `Gauge-identifiable shape/gauge decomposition`.
  - Added proof that the zero-mean shape/gauge decomposition is unique and invariant to additive gauge shift.
- Updated wrappers:
  - `gir_field_full_paper_skeleton.tex`
  - `gir_field_method_experiments_draft.tex`
  - Added `amsthm` and `\newtheorem{proposition}{Proposition}`.

### Citations

- Added: `references.bib`
- Added: `CITATION_VERIFICATION_LEDGER.md`
- Updated citations in:
  - `sections/introduction.tex`
  - `sections/related_work.tex`
  - `sections/experiments.tex`
- Added bibliography to wrappers.

Verified citation keys used by the draft:

```text
benjamini1995fdr
cai2023retinexformer
chen2018sid
guo2020zerodce
land1971retinex
li2021zerodcepp
wei2018deepretinex
wilcoxon1945ranking
zhang2019kind
```

No unused BibTeX keys remain.

### Figure package

- Added figure package directory: `figures/`
- Added both `.png` and `.pdf` versions for:
  - `fig_psnr_vs_scorr`
  - `fig_gauge_perturbation_bar`
  - `fig_psnr_misranking_failure_grid`
  - `fig_risk_reliability`
- Added: `figures/FIGURE_ASSET_MANIFEST.json`
- Added: `FIGURE_INTEGRATION_NOTES.md`
- Updated: `sections/figures.tex` to use PDF figure assets.
- Updated: `gir_field_full_paper_skeleton.tex` to use `\graphicspath{{figures/}}`.

### Appendix

- Added: `sections/appendix.tex`
- Added: `appendix/negative_route_audit_table.csv`
- Added: `appendix/negative_route_audit_table.tex`
- Added: `appendix/negative_route_audit_manifest.json`
- Updated full skeleton to include:

```tex
\input{sections/appendix}
```

Appendix audit table:

```text
CSV rows: 32
TeX selected rows: 17
```

Sources included:

```text
p5_retinex_distill_summary.csv
p5b_domain_distill_summary.csv
p6_structural_backbone_summary.csv
p6b_synprotect_summary.csv
p6c_gateprotect_summary.csv
p7_domainhead_summary.csv
p7b_realanchor_summary.csv
p7c_realanchor_fine_summary.csv
dgb_branch_consolidation_summary.csv
```

## Validation

Checks passed:

- all section files exist;
- full skeleton includes abstract, introduction, related work, method, experiments, figures, limitations/conclusion, appendix;
- theorem/proposition environment is defined and used;
- equation, align, table, table*, figure, proposition, proof, and itemize environments are balanced;
- brace counts are balanced;
- all `\cite{...}` keys exist in `references.bib`;
- no unused BibTeX keys;
- all figure PNG/PDF assets exist and PNGs verify with PIL;
- appendix table has 32 CSV rows;
- claim boundary strings remain present: `not a SOTA`, `Retinexformer remains`, `DGB/P2F`.

## Compilation boundary

`pdflatex` is still not installed in this environment, so no PDF was compiled locally. The TeX structure and assets were validated, but final camera-ready compilation should be done in a TeX-enabled environment.
