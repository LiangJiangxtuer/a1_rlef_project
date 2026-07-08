# GIR-Field 全面审核与下一轮迭代方案

Date: 2026-07-07  
Repo HEAD audited: `9ac323672b57e866eecadecb99958a144d1e2c70`  
Primary paper draft: `docs/paper/gir_field/build_ieee_v3/gir_field_ieee_conference.pdf`  
Scope: innovation, main claims, experiment evidence, top-conference/top-journal constraints, and next iteration design.

---

## 0. Executive verdict

**总体判断：GIR-Field 已经从“LLIE 方法调参路线”成功转成了一个有清晰论文主张的“gauge-identifiable exposure-field learning + benchmark/evaluation + failure audit”方向。** 这是正确转向，也是当前工作最有价值的创新点。

但以顶会/顶刊标准看，当前版本仍是：

```text
strong arXiv / internal draft / workshop-level evidence
NOT yet a fully competitive CVPR/ICCV/ECCV/TIP submission
```

关键原因不是主张不对，而是顶级审稿会追问：

1. UEFB-G 是否是可公开复现、可被他人使用的 benchmark/protocol？
2. internal-field metrics 是否只服务于自己的模型，还是能公平审查一类 exposure-field 方法？
3. 相比 2024--2026 LLIE / exposure-correction / quality-assessment 工作，GIR-Field 的不可替代性是什么？
4. 当前输出质量明显落后 Retinexformer，为什么这个 paper 仍值得顶会接收？
5. risk/recoverability 目前校准弱，是否应从主贡献中完全降级？

因此下一轮不应继续小幅 scalar sweep，也不应复活 DGB。推荐主线是：

```text
Benchmark-first + strong-baseline-compatible field probe
```

即：把 GIR-Field 做成一个真正可复现、可审查、可供其他方法接入的 UEFB-G benchmark/evaluator，同时证明 gauge-identifiability 可以作为强 restoration backbone 的 auxiliary/explainability layer，而不是一个弱输出模型的附属指标。

---

## 1. 创新性审核

### 1.1 当前真正的创新点

| 创新点 | 当前强度 | 审核判断 |
|---|---:|---|
| Gauge ambiguity formalization: `E = S + μ` | A- | 清晰、可解释、与 exposure/Retinex field 的物理语义有关；需要更强地连接到已有 Retinex/illumination ambiguity 文献。 |
| Centered E-shape calibration | B+ | 方法简单但有效；作为单独 method contribution 偏弱，作为 gauge-identifiability 的 operationalization 合理。 |
| UEFB-G gauge-controlled evaluation | A- / B+ | 最有潜力的贡献。若公开 evaluator/dataset/protocol，并覆盖更多方法，可成为顶会级 benchmark/evaluation contribution。 |
| PSNR-only misranking failure | A- | 论证张力强：PSNR 更高但 exposure shape 方向错误。是论文故事的核心惊讶点。 |
| Negative-route audit | B+ | 工程和学术诚实度强；可转化为 appendix/failure analysis，但不是主创新。 |
| Recoverability risk probe | C+ | 有 AUC signal，但 ECE 太高，只能 diagnostic，不能当主贡献。 |

### 1.2 创新性风险

当前最大风险是：**审稿人可能认为 centered correlation / mean-centering 是常识性处理，不足以支撑方法创新。**

对治方式不是把它夸大，而是把主贡献从“一个新 loss”改成：

```text
我们指出 exposure-field learning 存在 gauge-identifiability failure；
我们构建 benchmark/evaluator 暴露这个 failure；
centered E-shape 是最小可行校正机制，用来证明这个问题不是纯 metric artifact。
```

### 1.3 创新性评分

```text
Conceptual novelty:        A-
Method novelty:            B
Benchmark/evaluation value: A- if public/protocol-ready; B currently
Empirical novelty:          B
Top-tier narrative:         B+ currently, A- after next iteration
```

---

## 2. 主要主张审核

### 2.1 当前可支持的主张

#### Claim 1: centered E-shape improves gauge-free field identifiability

Supported.

Evidence:

```text
M4J_ES vs M4J, UEFB S_corr:
delta = +0.1581
95% CI = [0.1369, 0.1791]
FDR q = 2.18e-43
win_rate = 0.826
```

同时 paired fidelity 没有牺牲，反而改善：

```text
Real PSNR:      +0.6963 dB, q=1.57e-05
Synthetic PSNR: +0.6321 dB, q=6.04e-08
```

审核结论：这是当前最稳主张。

Recommended wording:

```text
Centered exposure-shape calibration significantly improves gauge-free exposure-field identifiability over the joint baseline under full local evaluation, while preserving or improving paired fidelity.
```

不要写：

```text
It solves exposure-field learning.
It is universally better for LLIE.
```

---

#### Claim 2: PSNR-only ranking can select physically incoherent fields

Supported and rhetorically strong.

Evidence:

```text
M4 A-gate:
UEFB PSNR = 18.6530
UEFB S_corr = -0.1393

M4J_ES:
UEFB PSNR = 17.9811
UEFB S_corr = 0.4399
```

M4J_ES vs M4:

```text
UEFB S_corr delta = +0.5792
95% CI = [0.5361, 0.6230]
FDR q = 3.52e-64
```

审核结论：这是最适合作为 Introduction 开头的“惊讶”。

---

#### Claim 3: UEFB-G separates global gauge error from local shape error

Supported as sanity-check / protocol claim.

Evidence from formal report:

```text
global shift +0.5:
E_MAE=0.5000, Gauge_MAE=0.5000, S_MAE=0.0000, S_corr=1.0000

local shape distortion:
S_MAE=0.2026, S_corr=0.6027, Gauge_MAE=0.0000
```

审核结论：指标分解成立。但顶会需要进一步证明：这个 protocol 对真实模型选择有实际影响，而不仅是 synthetic perturbation sanity check。当前 PSNR misranking 已提供初步真实模型证据。

---

#### Claim 4: recoverability risk is predictable but not calibrated

Supported as diagnostic-only.

Evidence:

```text
M4J_ES risk probe:
AUC = 0.7658
ECE = 0.2808

P3C_E0050_S3407 risk probe:
AUC = 0.7655
ECE = 0.2809
```

审核结论：只能写成 future direction / diagnostic probe。

不要写：

```text
risk head solves enhancement
recoverability is calibrated
selective enhancement is solved
```

---

### 2.2 当前不支持的主张

| 主张 | 状态 | 原因 |
|---|---|---|
| GIR-Field / RLEF is SOTA LLIE | 不支持 | Retinexformer real/synthetic PSNR 明显更高。 |
| M4J_ES/P3c beats Retinexformer | 不支持 | P3c real=20.0206, synthetic=17.6783；Retinexformer real=22.7939, synthetic=25.6690。 |
| DGB solves tri-domain trade-off | 不支持 | DGB/P2B/P2C/P2D/P2E 无 promoted row。 |
| Recoverability risk is solved | 不支持 | AUC 有信号但 ECE≈0.281。 |
| UEFB-G is already a public benchmark | 尚不支持 | 当前更多是 local protocol/artifact；需要 public evaluator/package/dataset card。 |
| Generalization across 2024--2026 methods | 尚不支持 | 只刷新了文献，没有跑新方法。 |

---

## 3. 实验结果审核

### 3.1 优点

1. **证据链比普通实验强**：有 per-image rows、bootstrap CI、Wilcoxon、BH-FDR correction。
2. **有正式 full protocol**：500 UEFB + 100 real + 100 synthetic，6300 per-image records。
3. **没有把 negative results 藏起来**：P5/P6/P7/DGB negative-route audit 完整。
4. **有外部强 baseline 边界**：Retinexformer / Zero-DCE++ / KinD++ output-only 表明确防止 SOTA overclaim。
5. **论文已可编译**：IEEE-style PDF 8 pages，validation PASS。

### 3.2 主要弱点

#### Weakness A: benchmark publicness 不够

当前 UEFB-G 是 local formal protocol，但顶会会问：

```text
Can another group run UEFB-G on their own exposure-field model?
Where is the official evaluator?
What is the dataset card / generation protocol / license boundary?
```

必须补：

```text
uefbg_eval.py
UEFB-G dataset card
metric definitions with exact formulas
input/output JSON schema
black-box vs field-aware method reporting rules
```

---

#### Weakness B: method strength 不够

作为 LLIE 方法，P3c/M4J_ES output quality 不够强：

```text
P3c synthetic PSNR = 17.6783
Retinexformer synthetic PSNR = 25.6690
```

所以不能走“new restoration method”路线。若要进入顶会，必须变成：

```text
benchmark/evaluation paper
or
strong-backbone-compatible explainability/field probe paper
```

---

#### Weakness C: internal-field metrics only compare internal variants

外部黑盒方法没有 `E/S/Gauge`。这是合理的，但审稿人会质疑：

```text
Is UEFB-G only useful for your own architecture?
```

下一轮需要至少接入 2 类外部/半外部 field-producing methods：

```text
Retinex/illumination-map methods
exposure-correction methods with estimated illumination/exposure maps
```

即便它们不是原生 `E`，也可以定义 adapter protocol：把 illumination/curve/gain map 转成 comparable log-exposure field。

---

#### Weakness D: risk calibration 现在拉低主线

Risk probe AUC≈0.766 但 ECE≈0.281。若放太前，会被攻击。建议主文只保留 1 段，appendix 详述。

---

#### Weakness E: literature refresh 是检索级，不是 baseline 级

当前 2024--2026 文献已进 BibTeX，但没有 re-run。必须在论文中写清：

```text
We discuss recent methods for positioning; we do not claim to outperform them.
```

若目标顶会，至少要挑 4--6 个近年代表方法做 output-only baseline 或 protocol table。

---

## 4. 顶会/顶刊约束审核

### 4.1 CVPR/ICCV/ECCV 约束

想投 CVPR/ICCV/ECCV，当前最可能被接受的定位不是 method，而是：

```text
A benchmark/evaluation + mechanism paper for gauge-identifiable exposure-field learning.
```

需要满足：

1. 公开 benchmark/evaluator；
2. 多方法覆盖，不只是自家模型；
3. 与最新 LLIE/exposure-correction 文献有实证连接；
4. 图表证明 PSNR misranking 是系统性现象；
5. 主贡献简洁且不可替代。

当前缺口：

```text
public benchmark package: missing
external field-aware adapters: missing
newer baselines execution: missing
cross-dataset generalization: weak
```

### 4.2 TPAMI / IJCV / TIP 约束

期刊路线要求更系统：

1. 数学定义完整；
2. benchmark protocol 清晰；
3. 更多数据集和方法；
4. 更完整 ablation；
5. limitation / negative-route audit 充分；
6. 代码数据可复现。

当前更接近 TIP/IJCV 的“evaluation + mechanism”初稿，但还需要明显扩容。

### 4.3 NeurIPS Datasets & Benchmarks / CVPR Datasets Track 约束

如果把 UEFB-G 做成 benchmark paper，需要：

1. Dataset/protocol release；
2. Datasheet / license / intended use；
3. evaluator package；
4. baseline zoo；
5. maintenance plan；
6. broad community relevance。

当前潜力大，但还未完成公开 benchmark 形态。

### 4.4 当前投稿成熟度评分

```text
arXiv technical report: ready
workshop paper: likely ready after minor cleanup
CVPR/ICCV/ECCV main: not ready
TIP/IJCV: not ready
NeurIPS D&B: not ready until public benchmark package exists
```

---

## 5. 新迭代方案

### Iteration North Star

下一轮目标不是继续调小 loss，而是把工作升级为：

```text
A public, reproducible, field-aware evaluation benchmark plus a strong-backbone-compatible proof that gauge-identifiable exposure-field learning matters.
```

---

## 6. 推荐路线：I1--I5

### I1. UEFB-G Benchmark v1 public package

**目标**：把当前 local protocol 变成可发布 benchmark。

Deliverables:

```text
scripts/uefbg_eval.py
configs/uefbg/protocol_v1.yaml
docs/UEFB_G_DATASET_CARD.md
docs/UEFB_G_EVALUATOR_SPEC.md
docs/UEFB_G_BASELINE_SUBMISSION_FORMAT.md
results/uefbg_v1_baselines/
tests/test_uefbg_evaluator_contract.py
```

Evaluator API:

```text
Input:
- enhanced image path or result JSON
- optional predicted exposure field E_pred
- optional A/Q/internal maps

Output:
- output-only metrics: PSNR, SSIM, LEE, over/under, identity_drop
- field-aware metrics: E_MAE, S_corr, S_MAE, Gauge_MAE
- perturbation sanity metrics
- report card JSON/CSV/Markdown
```

Go/no-go:

```text
GO if another method can be evaluated from a documented JSON/NPY/image folder without touching training code.
NO-GO if evaluator still assumes RLEF internals.
```

Priority: P0, must do first.

---

### I2. External field-aware adapter study

**目标**：证明 UEFB-G 不是只审查自家模型。

Methods to adapt:

```text
Retinex/illumination-map methods
Zero-DCE-like curve/gain maps
exposure-correction methods with estimated illumination fields
Retinexformer-derived illumination proxy if accessible
```

Adapter examples:

```text
illumination map L -> E = log(L + eps)
curve/gain map G -> E = log(G + eps)
Retinex factor I = R * L -> E = log(L)
```

Deliverables:

```text
scripts/run_uefbg_external_adapters.py
src/rlef/adapters/exposure_field_adapters.py
docs/EXTERNAL_FIELD_ADAPTER_PROTOCOL.md
results/uefbg_external_adapters/
```

Key question:

```text
Do strong/restoration methods also show PSNR-vs-field inconsistency when adapted into exposure-field space?
```

Go/no-go:

```text
GO if at least 2 external/semiexternal methods can produce comparable field proxies and reveal nontrivial S/Gauge behavior.
NO-GO if adapters are too arbitrary; then keep UEFB-G for explicit-field methods only and state the boundary.
```

---

### I3. Strong-backbone-compatible GIR probe

**目标**：解决“输出质量弱”问题，但不转回 SOTA claim。

Instead of building another weak RLEF restorer, attach GIR-Field as auxiliary/probe to a stronger restoration backbone:

```text
GIR-Retinexformer-Probe
or
Retinexformer + exposure-field auxiliary head
or
frozen Retinexformer output + learned exposure-field explanation head
```

Two safe variants:

#### I3-A: Post-hoc field explainer

```text
Input: low image + enhanced output from strong backbone
Output: E_pred explaining log-luminance transformation
Loss: centered E-shape + gauge decomposition + RGB consistency diagnostic
Backbone frozen
```

Pros: safe, avoids training huge model.  
Cons: explanation may be post-hoc, less method contribution.

#### I3-B: Auxiliary field head during fine-tuning

```text
Backbone: Retinexformer-like or lightweight strong U-Net
Heads: enhanced RGB + E/S/μ field
Loss: RGB restoration + centered E-shape + gauge consistency
```

Pros: stronger method claim.  
Cons: expensive and riskier.

Recommended start: I3-A.

Go/no-go thresholds:

```text
Output PSNR should stay within <=0.3 dB of the frozen strong backbone.
S_corr should improve meaningfully over naive illumination proxy.
No degradation in visual artifacts / no-reference metrics.
```

---

### I4. Recoverability calibration upgrade

**目标**：只有在主线稳定后再做；不要让它拖累 paper。

Methods:

```text
held-out calibration split
temperature scaling / isotonic regression
selective risk-coverage curves
Brier score + ECE + AUC
per-patch reliability diagrams
```

Deliverables:

```text
scripts/run_recoverability_calibration.py
results/recoverability_calibration_v1/
docs/RECOVERABILITY_CALIBRATION_AUDIT.md
```

Go/no-go:

```text
GO if ECE <= 0.08 and AUC >= 0.78 on held-out split.
Strong GO if ECE <= 0.05.
Otherwise keep as appendix diagnostic only.
```

---

### I5. Paper v2 rewrite for top-tier framing

**目标**：从 current draft 升级为 submission-style paper。

Rewrite priorities:

1. Introduction starts with PSNR misranking; current version already good but can be sharper.
2. Related Work must become deeper dialogue, not only expanded citation list.
3. Methods should split:
   - gauge formalism;
   - evaluator/protocol;
   - calibration objective;
   - optional strong-backbone probe.
4. Experiments should be reorganized around reviewer questions:
   - Does PSNR misrank exposure-field models?
   - Does S/Gauge decomposition isolate failure modes?
   - Does centered E-shape fix the field without harming output?
   - Does this generalize beyond RLEF internals?
   - Can strong backbones be made field-identifiable?
5. Appendix should contain negative-route audit and full tables.

Deliverables:

```text
docs/paper/gir_field_v2/
```

Go/no-go:

```text
Proceed to top-tier submission only after I1 + at least one of I2/I3 succeeds.
```

---

## 7. Prioritized execution plan

### Week/Phase 1: Benchmark hardening

```text
I1.1 Freeze UEFB-G protocol v1
I1.2 Implement standalone evaluator
I1.3 Write dataset/protocol card
I1.4 Add contract tests
I1.5 Re-run current internal methods through public evaluator wrapper
```

Expected output:

```text
UEFB-G is no longer a local script; it is a benchmark artifact.
```

### Week/Phase 2: External adapters

```text
I2.1 Choose 2--3 field-proxy external methods
I2.2 Implement map-to-E adapters
I2.3 Run UEFB-G on adapters
I2.4 Audit whether PSNR-vs-S_corr mismatch appears beyond RLEF
```

Expected output:

```text
Evidence that UEFB-G evaluates a class of methods, not only ours.
```

### Week/Phase 3: Strong-backbone probe

```text
I3.1 Implement post-hoc field explainer for strong backbone outputs
I3.2 Run on LOL-v2 real/synthetic + UEFB-G
I3.3 Compare output retention + S_corr improvement
```

Expected output:

```text
A path to combine Retinexformer-level RGB quality with field identifiability.
```

### Week/Phase 4: Risk calibration if still useful

```text
I4.1 Held-out calibration split
I4.2 Temperature/isotonic calibration
I4.3 Risk-coverage curves
I4.4 Decide whether risk enters main paper or appendix
```

Expected output:

```text
Risk either becomes calibrated contribution or is cleanly downgraded.
```

### Week/Phase 5: Paper v2

```text
I5.1 Rewrite contribution structure
I5.2 Replace generic related-work refresh with real literature dialogue
I5.3 Add benchmark package section
I5.4 Add external adapter/strong-backbone experiments
I5.5 Compile camera-ready draft
```

---

## 8. New main claim hierarchy after iteration

If I1 + I2 succeed:

```text
Primary claim:
UEFB-G is a public gauge-controlled benchmark that exposes PSNR-only misranking in exposure-field methods.

Secondary claim:
Centered E-shape calibration is a simple effective correction for gauge-free exposure-shape identifiability.
```

If I1 + I3 succeed:

```text
Primary claim:
Strong restoration and field identifiability can be decoupled: a strong backbone can retain output quality while exposing a gauge-identifiable local exposure field.
```

If I4 also succeeds:

```text
Tertiary claim:
Recoverability risk can be calibrated for selective enhancement.
```

If I4 fails:

```text
Keep risk as appendix diagnostic.
```

---

## 9. Recommended next concrete action

The immediate next executable step should be:

```text
Build UEFB-G public evaluator v1 and rerun current internal methods through it.
```

Why this first:

1. It attacks the biggest top-tier weakness: benchmark publicness.
2. It does not require new expensive training.
3. It strengthens the core paper identity.
4. It creates infrastructure needed by I2/I3.

Proposed command target after implementation:

```bash
python scripts/run_uefbg_benchmark_v1.py \
  --protocol configs/uefbg/protocol_v1.yaml \
  --methods manifests/uefbg/internal_methods_v1.json \
  --out results/uefbg_v1/internal_methods
```

Success criteria:

```text
- standalone evaluator can run without importing RLEF model internals;
- exact metric outputs match current formal protocol within tolerance;
- output contains JSON/CSV/Markdown report cards;
- tests cover missing-field, black-box-only, and field-aware submissions;
- docs include metric formulas and schema.
```

---

## 10. Final audit statement

Current GIR-Field is **scientifically promising and correctly claim-calibrated**, but not yet top-tier-ready. The strongest publishable story is not “we beat LLIE baselines”; it is:

```text
Modern low-light evaluation can select models that reconstruct RGB well but learn physically incoherent exposure fields. GIR-Field formalizes the gauge ambiguity behind this failure, UEFB-G evaluates it, and centered E-shape calibration demonstrates that the failure is correctable without sacrificing paired fidelity.
```

To reach top-tier standard, the next iteration must make UEFB-G public/protocol-grade and demonstrate either cross-method generality or compatibility with strong restoration backbones.
