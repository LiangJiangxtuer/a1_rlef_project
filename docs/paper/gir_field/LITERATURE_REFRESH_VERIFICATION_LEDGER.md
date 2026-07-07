# 2024--2026 literature refresh verification ledger

Search date: 2026-07-07.

## Search protocol

Sources actually queried:

- arXiv API:
  - `all:"low-light image enhancement"`
  - `all:"low light image enhancement"`
  - `all:"exposure correction" AND cat:cs.CV`
  - `all:"illumination degradation" AND cat:cs.CV`
  - `all:"low-light" AND diffusion AND cat:cs.CV`
- Semantic Scholar API was attempted, but returned HTTP 429 for the query batch, so it was not used for verified metadata in this refresh.

Raw machine-readable candidates:

```text
docs/paper/gir_field/literature_refresh_arxiv_raw.json
docs/paper/gir_field/literature_refresh_arxiv_candidates_2024_2026.json
```

Human summary:

```text
docs/paper/gir_field/LITERATURE_REFRESH_2024_2026.md
```

## Entries added to `references.bib`

| Bib key | arXiv ID | Title | Positioning in draft |
|---|---|---|---|
| `weng2024mamballie` | 2405.16105v1 | MambaLLIE: Implicit Retinex-Aware Low Light Enhancement with Global-then-Local State Space | Recent state-space LLIE. |
| `perez-zarate2024alen` | 2407.19708v5 | ALEN: A Dual-Approach for Uniform and Non-Uniform Low-Light Image Enhancement | Recent non-uniform LLIE context. |
| `dong2024ecmamba` | 2410.21535v1 | ECMamba: Consolidating Selective State Space Model with Retinex Guidance for Efficient Multiple Exposure Correction | Recent exposure-correction/state-space context. |
| `li2024osmamba` | 2411.15255v2 | OSMamba: Omnidirectional Spectral Mamba with Dual-Domain Prior Generator for Exposure Correction | Recent exposure-correction/state-space context. |
| `he2025unfoldir` | 2505.06683v1 | UnfoldIR: Rethinking Deep Unfolding Network in Illumination Degradation Image Restoration | Recent illumination-degradation restoration. |
| `liu2025ntire2025challenge` | 2510.13670v1 | NTIRE 2025 Challenge on Low Light Image Enhancement: Methods and Results | Benchmark/challenge context. |
| `xu2025boostingfidelityfo` | 2510.17105v1 | Boosting Fidelity for Pre-Trained-Diffusion-Based Low-Light Image Enhancement via Condition Refinement | Recent diffusion LLIE context. |
| `wei2025illumflow` | 2511.02411v1 | IllumFlow: Illumination-Adaptive Low-Light Enhancement via Conditional Rectified Flow and Retinex Decomposition | Recent flow/Retinex LLIE context. |
| `pilligua2025evaluatinglowlight` | 2511.15496v2 | Evaluating Low-Light Image Enhancement Across Multiple Intensity Levels | Evaluation context. |
| `xu2025consistretinex` | 2512.08982v3 | Consist-Retinex: One-Step Noise-Emphasized Consistency Training Accelerates High-Quality Retinex Enhancement | Recent Retinex consistency training. |
| `li2026rethinkingexposure` | 2604.04136v1 | Rethinking Exposure Correction for Spatially Non-uniform Degradation | Directly relevant non-uniform exposure correction. |
| `yan2026ntire2026challenge` | 2605.02212v1 | NTIRE 2026 Challenge on Efficient Low Light Image Enhancement: Methods and Results | Benchmark/challenge context. |
| `bai2026rethinkinglowlight` | 2605.02627v1 | Rethinking Low-Light Image Enhancement: A Log-Domain Intensity--Chromaticity Decoupling Perspective | Recent representation-level decomposition context. |
| `chen2026aigsnet` | 2606.17998v1 | AIGS-Net: Compact Illumination Field Modeling via 2D Gaussian Splatting for Fast Low-Light Image Enhancement | Adjacent illumination-field modeling. |
| `chen2026gaussianlightfield` | 2606.17985v1 | Gaussian Light Field Splatting: A Physical Prior-Driven Vision Transformer for Unsupervised Low-Light Image Enhancement | Adjacent light-field/physical-prior modeling. |
| `sun2026leiqassessor` | 2606.29752v1 | LEIQ-Assessor: Multi-dimensional Quality Assessment of Low-light Enhanced Images via Multi-task Learning | Recent multidimensional low-light quality assessment. |

## Boundary

The refresh updates the draft's positioning, not its experimental claim. These works were not re-run as baselines. Therefore the paper must still say that GIR-Field is a mechanism/evaluation contribution and not a SOTA LLIE result.
