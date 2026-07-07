# GIR-Field TeX/template migration and final package report

Date: 2026-07-07

## 1. TeX compile environment adaptation

A local user-space TeX compiler was installed without system package changes:

```text
tools/tectonic/tectonic
Tectonic 0.16.9
```

Source archive:

```text
tools/tectonic/tectonic-0.16.9-x86_64-unknown-linux-musl.tar.gz
```

Reason: no `pdflatex`, `latexmk`, `xelatex`, or `bibtex` executable was initially available in PATH, and conda search was unreliable in this environment.

## 2. Conference template migration

Added IEEE-style conference wrapper:

```text
docs/paper/gir_field/gir_field_ieee_conference.tex
```

The previous article wrapper remains available:

```text
docs/paper/gir_field/gir_field_full_paper_skeleton.tex
```

Both wrappers compile with Tectonic.

## 3. 2024--2026 literature refresh

Added raw and curated literature-refresh artifacts:

```text
docs/paper/gir_field/literature_refresh_arxiv_raw.json
docs/paper/gir_field/literature_refresh_arxiv_candidates_2024_2026.json
docs/paper/gir_field/LITERATURE_REFRESH_2024_2026.md
docs/paper/gir_field/LITERATURE_REFRESH_VERIFICATION_LEDGER.md
```

Updated:

```text
docs/paper/gir_field/references.bib
docs/paper/gir_field/sections/related_work.tex
```

The refresh added 16 arXiv-verified 2024--2026 LLIE/exposure-correction/evaluation papers. It changes the paper positioning, not the experimental claim boundary: no new SOTA claim is made.

## 4. Appendix negative-route audit expansion

Added:

```text
docs/paper/gir_field/sections/appendix_negative_routes_expanded.tex
docs/paper/gir_field/appendix/negative_route_family_summary.csv
docs/paper/gir_field/appendix/negative_route_family_summary.tex
docs/paper/gir_field/appendix/negative_route_expansion_manifest.json
```

Expanded appendix evidence:

```text
variant-level rows: 32
family-level rows: 9
```

## 5. Compiled PDFs

Article-style draft:

```text
docs/paper/gir_field/build_article_v3/gir_field_full_paper_skeleton.pdf
pages: 12
bytes: 327570
sha256: 98286fd02085028d0c605277f061bfbd0a247d98837d000d01f775eeca685270
```

IEEE-style conference draft:

```text
docs/paper/gir_field/build_ieee_v3/gir_field_ieee_conference.pdf
pages: 8
bytes: 313677
sha256: e79b896d1b8306c03af21e6bc354030c9bd49bf0818813774747b8435ed8c34f
```

Compile logs:

```text
docs/paper/gir_field/build_article_v3/tectonic_article_v3_compile.log
sha256: af2b6cf2de040968396d66259a39b294e91e569a73cbe41a4748991c853c86cd

docs/paper/gir_field/build_ieee_v3/tectonic_ieee_v3_compile.log
sha256: d9b35012ba74e8188f2654252565f6d953f18627793ba2644bcd7d1326c8e348
```

Warnings remain only layout warnings: small overfull/underfull boxes, mainly from dense appendix tables and bibliography line wrapping. There are no fatal errors.

## 6. Validation

Validation artifact:

```text
docs/paper/gir_field/FINAL_PACKAGE_VALIDATION.json
```

Validation status:

```text
PASS
```

Checks passed:

- all wrapper includes exist;
- TeX environments balanced;
- braces balanced;
- all cited keys exist in `references.bib`;
- no unused BibTeX keys;
- all figure PNG/PDF assets exist and PNGs verify;
- appendix rows match expected counts;
- both article and IEEE PDFs exist;
- compile logs contain no fatal errors.

## 7. Deliverable package

A source/PDF package was created at:

```text
/home/user/下载/GIR_FIELD_PAPER_PACKAGE_20260707.tar.gz
bytes: 1535598
sha256: a45bdb8858aad02b6ebb1797ecdba654c9adacec622f8f113ce147efa05f00fa
```

Package checksum was also written to:

```text
docs/paper/gir_field/PACKAGE_SHA256.txt
```

## 8. Claim boundary preserved

The draft still explicitly avoids unsupported claims:

```text
not a SOTA LLIE paper
Retinexformer remains stronger on RGB fidelity
DGB/P2F not revived
risk probe is diagnostic, not solved recoverability
```
