# S1-S4 Pipeline Validation Report

Generated: 2026-07-04T16:25:34.482808+00:00

## Validation status

```text
PASS
```

## Checks performed

- Manifest parsed and contains exactly `S1 -> S2 -> S3 -> S4`.
- All manifest file records exist, are non-empty, and match recorded byte counts / SHA256 hashes.
- JSON artifacts parsed successfully: manifest, S1 claim matrix, S2 main table, S4 negative-route summary.
- CSV artifacts have expected minimum row counts.
- Markdown artifacts are non-empty and no longer contain stale Python `None` rendering.
- S3 generated exactly 8 PNG qualitative panels: 4 real + 4 synthetic.
- Each PNG was opened/verified by PIL and has expected contact-sheet dimensions.
- New pipeline contract tests passed.
- Full repository test suite passed.

## Test evidence

```text
110 passed in 19.58s
```

## Pipeline manifest evidence

```text
stages: ['S1', 'S2', 'S3', 'S4']
manifest_files: 20
visual_panels: 8
```

## Primary artifacts

- `results/paper_pipeline/S1_S4_PIPELINE_REPORT.md`
- `results/paper_pipeline/S1_S4_PIPELINE_MANIFEST.json`
- `results/paper_pipeline/s1_evidence_freeze/EVIDENCE_FREEZE.md`
- `results/paper_pipeline/s2_main_table/main_results_table.md`
- `results/paper_pipeline/s2_main_table/main_results_table.tex`
- `results/paper_pipeline/s3_visualizations/VISUALIZATION_INDEX.md`
- `results/paper_pipeline/s4_appendix/APPENDIX_NEGATIVE_RESULTS.md`

## Visual panels

- `results/paper_pipeline/s3_visualizations/real/real_000_low00690_panel.png`
- `results/paper_pipeline/s3_visualizations/real/real_001_low00691_panel.png`
- `results/paper_pipeline/s3_visualizations/real/real_002_low00692_panel.png`
- `results/paper_pipeline/s3_visualizations/real/real_003_low00693_panel.png`
- `results/paper_pipeline/s3_visualizations/synthetic/synthetic_000_r00816405t_panel.png`
- `results/paper_pipeline/s3_visualizations/synthetic/synthetic_001_r00869422t_panel.png`
- `results/paper_pipeline/s3_visualizations/synthetic/synthetic_002_r00879054t_panel.png`
- `results/paper_pipeline/s3_visualizations/synthetic/synthetic_003_r01058910t_panel.png`
