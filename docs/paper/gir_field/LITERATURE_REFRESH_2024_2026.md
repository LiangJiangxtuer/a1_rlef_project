# 2024--2026 LLIE / exposure-field literature refresh

Search date: 2026-07-07. Sources: arXiv API queries for low-light image enhancement, exposure correction, illumination degradation, Retinex, Mamba, diffusion, and quality assessment. Semantic Scholar was unavailable due HTTP 429 rate limiting, so entries below are treated as arXiv-verified preprints unless separately accepted by venue metadata.

## Selected papers added to draft bibliography

| Key | Year | arXiv | Title | Why it matters for GIR-Field |
|---|---:|---|---|---|
| `weng2024mamballie` | 2024 | `2405.16105v1` | MambaLLIE: Implicit Retinex-Aware Low Light Enhancement with Global-then-Local State Space | State-space / Retinex-aware LLIE trend; useful as recent efficient-restoration context. |
| `perez-zarate2024alen` | 2024 | `2407.19708v5` | ALEN: A Dual-Approach for Uniform and Non-Uniform Low-Light Image Enhancement | Explicitly separates uniform and non-uniform low-light enhancement, close to GIR-Field non-uniform motivation. |
| `dong2024ecmamba` | 2024 | `2410.21535v1` | ECMamba: Consolidating Selective State Space Model with Retinex Guidance for Efficient Multiple Exposure Correction | Exposure-correction Mamba direction; positions GIR-Field against exposure correction rather than only LLIE. |
| `li2024osmamba` | 2024 | `2411.15255v2` | OSMamba: Omnidirectional Spectral Mamba with Dual-Domain Prior Generator for Exposure Correction | Spectral Mamba + exposure correction; another recent state-space exposure-correction baseline family. |
| `he2025unfoldir` | 2025 | `2505.06683v1` | UnfoldIR: Rethinking Deep Unfolding Network in Illumination Degradation Image Restoration | Illumination-degradation restoration/unfolding direction; reinforces need to separate degradation modeling from field identifiability. |
| `liu2025ntire2025challenge` | 2025 | `2510.13670v1` | NTIRE 2025 Challenge on Low Light Image Enhancement: Methods and Results | NTIRE 2025 LLIE challenge; benchmark context and evidence that the field remains restoration-score centered. |
| `xu2025boostingfidelityfo` | 2025 | `2510.17105v1` | Boosting Fidelity for Pre-Trained-Diffusion-Based Low-Light Image Enhancement via Condition Refinement | Diffusion-based LLIE refinement; positions GIR-Field against generative/restoration advances without claiming SOTA. |
| `wei2025illumflow` | 2025 | `2511.02411v1` | IllumFlow: Illumination-Adaptive Low-Light Enhancement via Conditional Rectified Flow and Retinex Decomposition | Rectified-flow + Retinex; recent generative-Retinex hybrid. |
| `pilligua2025evaluatinglowlight` | 2025 | `2511.15496v2` | Evaluating Low-Light Image Enhancement Across Multiple Intensity Levels | Evaluation across multiple intensity levels; closest recent benchmark/evaluation paper in the refresh. |
| `xu2025consistretinex` | 2025 | `2512.08982v3` | Consist-Retinex: One-Step Noise-Emphasized Consistency Training Accelerates High-Quality Retinex Enhancement | One-step Retinex consistency training; recent Retinex acceleration/quality direction. |
| `li2026rethinkingexposure` | 2026 | `2604.04136v1` | Rethinking Exposure Correction for Spatially Non-uniform Degradation | Spatially non-uniform exposure degradation; directly supports the need for non-uniform/gauge-aware exposure evaluation. |
| `yan2026ntire2026challenge` | 2026 | `2605.02212v1` | NTIRE 2026 Challenge on Efficient Low Light Image Enhancement: Methods and Results | NTIRE 2026 efficient LLIE challenge; most recent benchmark context. |
| `bai2026rethinkinglowlight` | 2026 | `2605.02627v1` | Rethinking Low-Light Image Enhancement: A Log-Domain Intensity--Chromaticity Decoupling Perspective | Log-domain intensity/chromaticity decoupling; recent representation-level critique aligned with decomposition framing. |
| `chen2026aigsnet` | 2026 | `2606.17998v1` | AIGS-Net: Compact Illumination Field Modeling via 2D Gaussian Splatting for Fast Low-Light Image Enhancement | Illumination-field modeling via 2D Gaussian splatting; directly adjacent to field modeling. |
| `chen2026gaussianlightfield` | 2026 | `2606.17985v1` | Gaussian Light Field Splatting: A Physical Prior-Driven Vision Transformer for Unsupervised Low-Light Image Enhancement | Gaussian light-field splatting and physical priors; adjacent to field/physical prior direction. |
| `sun2026leiqassessor` | 2026 | `2606.29752v1` | LEIQ-Assessor: Multi-dimensional Quality Assessment of Low-light Enhanced Images via Multi-task Learning | Low-light enhanced-image quality assessment; reinforces artifact/multi-dimensional assessment beyond PSNR. |

## Claim boundary after refresh

- Recent papers strengthen the need to discuss state-space models, diffusion/flow models, Gaussian/field representations, controllability, and quality assessment.
- They do **not** change the GIR-Field claim boundary: the current paper should remain a gauge-identifiability / benchmark / failure-audit paper, not a SOTA LLIE paper.
- Because these are arXiv-verified entries from a rapid refresh, the final submission should still perform official-code/result verification for any baseline claim.

