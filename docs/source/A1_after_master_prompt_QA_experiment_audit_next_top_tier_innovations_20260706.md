# A1 after-master-prompt 实验链审核与顶会/顶刊级深化方向

## Material Passport

- Origin Task: 审核 `image_enhancement_after_master_prompt_sessions_merged_chronological_20260705_162742_QA_chronological.md` 中围绕 A1/DGB-RLEF 的图像增强实验过程，提炼顶会/顶刊级下一步创新路线。
- Source 1: `G:\9chapterwork\workspace\image_enhancement_research_20260629\A1_QA_audit_DGB_RLEF_experiment_guidance_20260702.md`
- Source 2: `G:\9chapterwork\workspace\json_dialog_exports_20260705\image_enhancement_after_master_prompt_sessions_merged_chronological_20260705_162742_QA_chronological.md`
- Skills Applied: `using-superpowers`; `academic-research-suite`; `academic-paper-reviewer` 的 methodology/domain/devil's-advocate 审核视角；`experiment-agent` 的统计解释、复现实验与实验计划规范。
- Verification Status: ANALYZED_FROM_LOCAL_FILES
- What Was Not Done: 本轮没有重新训练模型、没有重新跑评估脚本、没有联网更新 2026 最新 SOTA。所有数值结论来自上述两个本地 Markdown 文件；正式投稿前必须联网核验最新方法和官方实现。
- Date: 2026-07-06

## 0. 结论先行

当前实验链没有实现原始 DGB-RLEF 的顶会方法型核心主张。

如果原始主张是：DGB-RLEF 通过 domain-gauge balance 与 recoverability route 同时解决 UEFB、LOL-v2-real、LOL-v2-synthetic 三域 trade-off，并作为新方法进入 3-seed/full schedule 或冲击 SOTA，那么证据是不成立的。DGB 分支中 best-real、best-synthetic、best-UEFB 是分裂的，没有任何一个 DGB variant 同时超过 P3c 三域均值和 P7B near-miss gate。

但实验链已经产生了一个更值得继续深化的顶会/顶刊级方向：不要继续把 DGB 包装成 SOTA enhancement method，而应转向 **Gauge-Identifiable Exposure Field Learning and Evaluation**，即把论文核心从“我提出一个更强增强网络”调整为“低光/曝光不均匀增强中，局部曝光场存在 gauge 不可辨识、可恢复性条件风险和三域 Pareto trade-off；本文提出可识别的 exposure-shape/gauge 分解、UEFB-v2/E-A-Q 解释性评价协议，以及 claim-calibrated failure audit”。

推荐下一步主线：

**GIR-Field: Gauge-Identifiable and Recoverability-Calibrated Exposure Field Learning for Uneven Low-Light Enhancement**

这条线的 claim 不是超过 Retinexformer 的 PSNR，而是建立一个更基础的问题定义、评估协议和机制验证框架。它可以保留 P3c/M4J_ES 的正证据，吸收 DGB/P5/P6/P7 的负结果，并把这些负结果变成论文价值，而不是继续堆一个没有通过 joint gate 的新方法。

## 1. 过程数据质量审核

Q&A 文件共有 17 组问答。整体实验链是连续的，但有两类过程噪声需要在论文整理时明确剔除：

| Q&A | 用户请求 | Assistant 回答状态 | 审核处理 |
|---|---|---|---|
| Q009 | 停止 DGB branch 并 consolidate | 回答重复 P2E 内容 | 视为错配/重复，不作为独立实验 |
| Q012 | 执行 S1-S4 pipeline | 回答重复 P2E 内容 | 视为错配/重复，不作为 pipeline 证据 |
| Q010/Q013 | 停止 DGB branch 并 consolidate | 给出正式 consolidation | 采用为 DGB stop 证据 |
| Q011/Q014 | 总结阶段实验并调整安排 | 给出同类总结 | 合并为一类总结证据 |
| Q015 | S1-S4 pipeline | 给出正式 pipeline 输出 | 采用为 pipeline 证据 |

这意味着后续论文证据链不能按“17 个回答 = 17 个独立实验”写。真实可用证据应按阶段合并：Phase 0、Phase 1、Phase 2、P2B、P2C、P2D、P2E、DGB stop/consolidation、stage summary、S1-S4 pipeline、no-reference supplementary、final claim audit。

## 2. 全实验链审计表

| 阶段 | 实验目的 | 关键结果 | Go/No-Go | 顶会级审核意见 |
|---|---|---|---|---|
| Phase 0 baseline reproduction | 复现并冻结 P3c/P6/P7B/official baselines，防止后续建立在错基线上 | P3c: UEFB 17.915, Real 20.021, Synthetic 17.678; P6: 18.015/20.197/17.598; P7B_RA010: 18.085/19.847/17.965; Retinexformer: Real 22.794, Synthetic 25.669 | PASS | 这是后续所有 claim 的最低证据底座。Retinexformer gap 明确限制了方法型 SOTA claim。 |
| Phase 0 audit -> Phase 1 guidance | 审计 Phase 0 并限制 Phase 1 只做最小实现 | Phase 0 valid; phase1_executed=false; 明确不跑 Phase 2 training | PASS | 审计纪律正确，防止 scaffold 与训练结果混淆。 |
| Phase 1 minimal DGB implementation | 实现 zero-mean decomposition、shape loss、warm scheduler、gauge head、router、metrics ledger | 单元测试/full suite/compile check 通过；未训练 | PASS as scaffold | 只证明代码 contract 存在，不证明方法有效。论文中不能把 Phase 1 写成性能贡献。 |
| Phase 2 DGB minimal | 检查 DGB_RLEF_MINIMAL_S3407 是否有三域潜力 | UEFB 17.754; Real 19.939; Synthetic 18.479; UEFB 低于 P3c/P7B，Real 低于 P3c；safe_fraction 约 0.57-0.64 | NO-GO | Synthetic gain 明显，但三域不平衡。safe router 可能过保守。 |
| P2B controlled isolation | 分离 gauge-only 与 route-only，判断瓶颈 | Gauge-only: UEFB 17.719, Real 20.432, Synthetic 18.402; Route-only: UEFB 17.713, Real 19.967, Synthetic 17.945 | NO-GO | Gauge-only 证明 image-stat gauge 对 real/synthetic 有价值，但 UEFB 崩；safe router 是 real-domain bottleneck，但不是全部瓶颈。 |
| P2C UEFB recovery | 从 gauge-only 出发，用 gate floor/e_shape 降低恢复 UEFB | e_shape010: UEFB 17.792, Real 19.604, Synthetic 18.588; floor015/025 没救 UEFB | NO-GO | 简单增加 restoration/gate floor 无效；降低 E-shape 可提升 UEFB/synthetic，但牺牲 real。 |
| P2D balanced real preservation | 加 structure_preserve 尝试保留 e_shape010 收益并修 real | struct005: UEFB 17.863, Real 19.541, Synthetic 18.399; struct002: Real 19.781, Synthetic 18.542 | NO-GO | structure_preserve 有局部作用，但不能稳定恢复 real 和 UEFB。说明瓶颈不是一个简单结构保持 loss。 |
| P2E gate prior balance | 检查 e_shape010 下 route/gate calibration 是否可修 | gp002: UEFB 17.850, Real 19.487, Synthetic 18.683; gp005: UEFB 17.660, Real 19.789, Synthetic 18.248 | NO-GO | gate_prior 改变 routing，但更多 real routing 没转化为 real PSNR。routing usage 与 output quality 脱钩。 |
| DGB stop/consolidation | 汇总 11 个 DGB runs，决定是否继续 P2F/3-seed | best UEFB: 17.863/19.541/18.399; best real: 17.719/20.432/18.402; best synthetic: 17.850/19.487/18.683; joint gate false | STOP | 决策正确。继续 DGB sweep 会变成调参堆砌，不符合顶会审稿偏好。 |
| Stage summary/replanning | 将 P5/P6/P7/DGB 归入 appendix/negative evidence，并收缩 claim | P3c 3-seed 是 default；Retinexformer gap 大；DGB/P5/P6/P7 为 negative-route diagnostics | PASS | 论文定位从 SOTA method 转为 mechanism/evaluation/claim-calibrated audit 是正确调整。 |
| S1-S4 pipeline | 建立 evidence freeze、主表、可视化、appendix negative diagnostics | P5/P6/P7/DGB 移出主表；P3c/Retinexformer/Zero-DCE++/KinD++ 进入主表；验证通过 | PASS as artifact pipeline | 有论文整理价值，但不是新实验性能证据。 |
| No-reference supplementary | 用 unpaired real 128 张图补充评估 P3c 输出 | P3c 输出降低 under/dark_ratio；no-reference mixed evidence | SUPPORT ONLY | 只能辅助说明真实图泛化，不能作为主 claim 或 SOTA 证据。 |
| Final claim audit | 判断是否实现核心创新、贡献与顶会潜力 | DGB 方法主张未实现；M4J->M4J_ES 是最大亮点；机制/benchmark paper 有潜力 | PASS as audit | 后续必须围绕 gauge-identifiability、UEFB-v2/E-A-Q、负结果审计组织论文。 |

## 3. 核心数据判断

### 3.1 原始 DGB-RLEF 方法主张未成立

DGB 分支最强证据是 split-wise 强，而不是 joint 强：

| Method / Branch | UEFB PSNR | Real PSNR | Synthetic PSNR | 结论 |
|---|---:|---:|---:|---|
| P3c default | 17.915 | 20.021 | 17.678 | 3-seed default |
| DGB best UEFB | 17.863 | 19.541 | 18.399 | UEFB 仍低于 P3c |
| DGB best real | 17.719 | 20.432 | 18.402 | real 强，但 UEFB 低 |
| DGB best synthetic | 17.850 | 19.487 | 18.683 | synthetic 强，但 real 低 |
| DGB joint gate | - | - | - | False |

审稿级解释：DGB 的各个组件并非完全没有用，但它们只是把模型沿 Pareto front 移动，没有把 Pareto front 整体外推。作为顶会方法论文，最容易被攻击的问题是：作者用大量局部 sweep 找到 split-wise 亮点，却没有一个统一模型同时改善所有核心域。

### 3.2 最强正证据来自 centered / gauge-invariant E-shape

当前最强、最可写成论文动机的结果是 M4J -> M4J_ES：

| Variant | UEFB E_corr | Real E_corr | Synthetic E_corr | Real PSNR | Synthetic PSNR |
|---|---:|---:|---:|---:|---:|
| M4J joint | 0.282 | 0.501 | 0.369 | 19.581 | 16.508 |
| M4J_ES | 0.440 | 0.648 | 0.799 | 20.277 | 17.141 |
| P3c e=0.05 3-seed | 0.436 | 0.707 | 0.841 | 20.021 | 17.678 |

M4J_ES 相比 M4J 的提升：UEFB E_corr +0.158，Real E_corr +0.148，Synthetic E_corr +0.430，Real PSNR +0.696 dB，Synthetic PSNR +0.632 dB。这不是单纯解释性指标变好，而是 paired fidelity 也同步变好。因此它比 DGB 的 split-wise gain 更适合作为论文核心证据。

这组结果支持一个更本质的 claim：曝光场预测失败的根源不是“网络不够复杂”，而是 exposure field 的绝对 gauge 和空间 shape 混在一起，导致模型可以在 PSNR 上局部变好但 E field 方向错误。centered E-shape loss 把问题投影到 gauge-invariant shape 空间，才让 E_corr 稳定转正。

### 3.3 UEFB-v2/E-A-Q 的价值是揭示 PSNR 看不到的错

早期 A-gate 的结果说明只看 UEFB PSNR 会误判：

| Variant | UEFB PSNR | UEFB E_corr | 说明 |
|---|---:|---:|---|
| M0 restorer_only | 17.379 | 0.000 | 没有 exposure field 解释力 |
| M4 A-gate | 18.653 | -0.139 | UEFB PSNR 高，但 exposure shape 方向错 |
| M4J_ES | 17.981 | 0.440 | PSNR 略低，但解释性正确 |
| P3c 3-seed | 17.915 | 0.436 | 稳定 default |

这正是 benchmark/evaluation paper 的价值：标准 paired PSNR 会鼓励模型走向 M4 这种“图像指标局部好、物理解释错误”的解。UEFB-v2/E-A-Q 可以把这类错误暴露出来。

### 3.4 与 Retinexformer 的 gap 决定了投稿定位

| Method | Real PSNR | Synthetic PSNR |
|---|---:|---:|
| P3c RLEF default | 20.021 | 17.678 |
| Retinexformer | 22.794 | 25.669 |
| Gap | -2.773 dB | -7.991 dB |

这个 gap 对方法型顶会论文是硬伤。除非后续新方法能把 real gap 缩小到约 1 dB 内，并明显缩小 synthetic gap，同时保留解释性优势，否则不应写成 SOTA LLIE method paper。

## 4. 顶会/顶刊审稿视角审核

### 4.1 Methodology reviewer: 主要方法学缺口

1. **DGB 后续都是 single-seed / 1000-step 探索，不能支撑主 claim**  
   Phase 2/P2B/P2C/P2D/P2E 对定位瓶颈有价值，但没有任何 candidate 进入 3-seed 和 full schedule。顶会审稿人会认为 DGB 只完成了探索性消融，不能作为最终方法结论。

2. **缺少 per-image 显著性和 effect-size 报告**  
   P3c 有 3-seed 均值/方差，但还需要 per-image paired tests、bootstrap 95% CI、Wilcoxon signed-rank、Cliff's delta 或 paired effect size，尤其要证明 E-shape 不只是平均值偶然提升。

3. **no-reference 真实图评估只能辅助**  
   unpaired real 128 张图显示 under/dark_ratio 降低，但 no-reference 指标混合且容易与视觉偏好冲突。它不能替代 paired benchmark，也不能证明方法更好。

4. **benchmark 公信力还不足**  
   UEFB-v2 是当前最有价值资产，但必须公开生成协议、随机种子、E/A/Q ground truth、分层统计、baseline output 和评价脚本，否则会被认为是自建指标服务自家方法。

5. **结果选择风险需要显式控制**  
   过程里存在大量 sweep：teacher、rec weight、scalar gate、domain head、anchor、DGB variants。论文必须用 claim ledger 和 promotion rule 解释为什么某些结果进主表、某些只进 appendix，避免 cherry-picking 指控。

### 4.2 Domain reviewer: 领域贡献是否足够

作为传统 LLIE/图像增强方法论文，目前贡献不足，因为 fidelity gap 太大。作为机制/benchmark/analysis paper，有潜力，理由是：

- 发现并量化了 exposure field 的 gauge ambiguity 与 shape/gauge 混淆问题。
- 用 UEFB-v2/E-A-Q 指标证明标准 PSNR 会隐藏解释性错误。
- 系统性负结果覆盖 teacher distillation、dataset-weighted rec、scalar gate、domain head、static anchor、safe router，能形成设计原则。
- P3c/M4J_ES 提供了一个可复现实证：centered shape calibration 能同时改善 E_corr 与 paired fidelity。

但领域贡献仍需要补强：

- 必须把理论从“工程解释”写成 formal problem：曝光场等价类、gauge 不可辨识、shape/gauge 分离、recoverability 条件风险。
- 必须补齐强基线和最新基线。源文件只确认 Retinexformer/Zero-DCE++/KinD++；正式论文前需联网核验 2024-2026 低光增强/曝光校正最新方法和官方协议。
- 必须证明该评价协议不是只适用于 RLEF。至少要让黑箱方法也能在 UEFB-v2 上评价输出质量、failure cases、过增强/欠增强、exposure consistency proxy。

### 4.3 Devil's advocate: 最强反驳

最强反驳是：这不是一篇新方法论文，而是一组围绕弱基线的失败调参记录。作者尝试了多个模块，最终没有一个 DGB variant 通过三域 joint gate，也远落后 Retinexformer。因此如果论文标题或摘要声称提出了强新方法，审稿人会直接质疑贡献是否成立。

要化解这个反驳，必须主动收缩定位：

- 不声称 DGB-RLEF resolves tri-domain trade-off。
- 不声称 DGB-RLEF is SOTA。
- 不声称 E_corr improvement guarantees better enhancement。
- 不把 DGB 失败隐藏起来，而是把它作为“为什么需要 gauge-identifiable evaluation and claim-calibrated audit”的证据。
- 把贡献从“性能超越”改为“问题定义 + 理论可识别性 + benchmark + 机制验证 + 负结果审计”。

这个转向如果做得扎实，反而比继续修 DGB 更接近顶会/顶刊要求。

## 5. 必须停止和保留的路线

### 5.1 必须停止

| 路线 | 停止原因 |
|---|---|
| 继续 P2F/DGB route-quality coupling sweep | 前面 11 个 DGB runs 已证明 split-wise gain 不能转化为 joint gate；继续扫参数像 cherry-picking。 |
| 更大 real_anchor / static anchor | P7c 已显示 RA015/020/030 不优于 RA010；fixed anchor 有域脆弱性。 |
| teacher/output distillation 作为主线 | P5/P5b 显示 distillation 主要移动 real/synthetic trade-off，不是统一提升。 |
| rec_by_dataset / synthetic rec weight 作为主线 | P6b 能拉 synthetic，但牺牲 UEFB/real，是 dataset-specific overcorrection。 |
| scalar gate / identity scalar 作为主线 | P6c 说明 scalar gate 只能移动 trade-off，不能解决像素/区域级 recoverability。 |
| 只用 E_corr 证明方法更好 | oracle-E 和 P5b 类结果已说明 E 更准不必然 RGB 更好。 |

### 5.2 必须保留

| 证据 | 保留方式 |
|---|---|
| P3c 3-seed default | 作为所有新机制的最低 baseline。 |
| M4J -> M4J_ES | 作为最大正证据，支撑 centered/gauge-invariant E-shape。 |
| UEFB-v2/E-A-Q | 作为论文最有独立价值的 benchmark/evaluation protocol。 |
| P5/P6/P7/DGB negative routes | 作为 appendix 和 design rationale，证明问题不是简单加 teacher、加结构或加 scalar 可解决。 |
| Retinexformer gap | 作为 limitation 主动披露，防止 SOTA overclaim。 |

## 6. 推荐的顶会/顶刊级创新点

### A1. GIR-Field: Gauge-Identifiable and Recoverability-Calibrated Exposure Field Learning

**推荐等级：最高。** 这是最适合从现有证据继续深化的主线。

**要解决的问题本质**  
局部曝光场增强不是单纯预测一个 E-map。E-map 在 Poisson/gradient/log-gain 约束下存在 additive gauge ambiguity：`E` 的绝对均值与空间形状混合在一起，不同数据域的最佳 gauge 不一致。模型若只优化 RGB PSNR，很容易得到“图像看起来局部变好，但 E shape 错”的解；若只优化 E，又可能 RGB PSNR 下降。真正的问题是要在 exposure-field 的 quotient space 中学习 gauge-free shape，并把 absolute gauge 作为可校准变量。

**核心 claim**  
A robust exposure-field enhancer should decompose exposure into a gauge-invariant spatial shape and a calibrated image-level gauge; centered shape consistency improves exposure-field identifiability and reveals failure modes hidden by PSNR-only evaluation.

**可用数学工具**

- Gauge equivalence class: `E ~ E + c`，把曝光场分解为 `S = E - mean(E)` 与 `mu = mean(E)`。
- Identifiability proof: 在约束 `mean(S)=0` 下，`E=S+mu` 的 shape/gauge 分解唯一。
- Gauge-invariant loss: centered correlation / centered low-pass consistency，只惩罚 shape 错误，不惩罚全局 offset。
- Calibration risk: 对 `mu` 做 image-stat calibration，报告 Gauge_MAE 与 domain-wise calibration error。
- Pareto analysis: 用 UEFB/real/synthetic 三域指标构造 Pareto frontier，而不是单域 best checkpoint。

**为什么有顶会/顶刊潜力**

它不是继续堆网络模块，而是重新定义 LLIE 中 exposure field 的可识别性问题。已有结果能支持这个方向：M4J_ES 同时提升 E_corr 与 real/synthetic PSNR；M4 A-gate 证明 PSNR 可以误导；DGB/P7/P6 证明简单 domain/scalar 修补不能解决三域冲突。

**下一步验证路线**

1. 复现并冻结 P3c、M4J、M4J_ES、P6、P7B、Retinexformer、Zero-DCE++、KinD++ 已有结果。
2. 对 M4J/M4J_ES/P3c 做 per-image 统计，报告 E_corr、S_corr、Gauge_MAE、PSNR 的 paired bootstrap CI。
3. 构造 gauge perturbation set：对 GT exposure field 加全局 offset、局部 shape distortion、mixed offset+shape distortion，检验哪些指标能区分错误类型。
4. 把黑箱方法纳入 UEFB-v2 输出评价：即便没有内部 E-map，也可评价 RGB fidelity、over/under exposure、failure-region consistency；内部 E 指标标注为 N/A，不强迫黑箱方法输出 E。
5. 形成 Figure: PSNR-only misranking vs gauge-identifiable ranking，展示 M4、M4J_ES、P3c 的排序差异。

**Go/No-Go**

- Go: M4J_ES/P3c 的 S_corr/E_corr 提升在 per-image bootstrap 下稳定，且 real/synthetic PSNR 不下降或有明确提升。
- Go: UEFB-v2 可以揭示至少 3 类 PSNR 隐藏失败，如 over-enhancement、wrong exposure shape、near-identity damage。
- No-Go: 如果 E-shape 指标只对 RLEF 内部模型有效，对外部方法没有任何解释价值，则 benchmark 贡献不足。

### A2. UEFB-G: Gauge-Controlled Uneven Exposure Benchmark and Claim-Calibrated Evaluation Protocol

**推荐等级：高。** 可与 A1 合并为 benchmark + mechanism 论文。

**要解决的问题本质**  
现有 paired LLIE benchmark 偏向整体 fidelity，无法系统考察曝光不均匀、局部过增强、可恢复性差异和 near-identity 保护。UEFB-v2 已经暴露了 M4 这种“PSNR 高但 E_corr 错”的现象，但还需要升级为可公开、可复现、可被外部方法使用的 benchmark。

**核心 claim**  
Standard LLIE metrics can mis-rank exposure-field methods under uneven illumination. A gauge-controlled benchmark with E/A/Q annotations exposes shape, gauge, recoverability and over-enhancement failures that paired PSNR misses.

**可用数学/统计工具**

- Stratified benchmark design: 按不均匀曝光强度、局部曝光比例、near-identity 区域、噪声强度分层。
- Multi-objective ranking: PSNR/SSIM、E/S/gauge、A/Q calibration、over-enhancement risk 分开报告，不做单一分数掩盖冲突。
- Bootstrap confidence interval 和 paired Wilcoxon，报告每类 failure stratum 上的稳定性。
- Claim ledger: 每个 claim 必须绑定到表格、图、统计检验和反例。

**实验路线**

1. 固定 UEFB-v2 生成脚本、seed、训练/验证/测试划分。
2. 保存每张图的 E_gt、A_gt、Q_gt、exposure-stratum label、near-identity mask、noise/structure metadata。
3. 为所有方法生成统一 per-image CSV：PSNR、SSIM、LPIPS、LEE、NAI、under/over ratio、E_corr/S_corr/Gauge_MAE、Q_ECE、identity_drop。
4. 至少纳入源文件已确认的 Retinexformer、Zero-DCE++、KinD++、P3c、P6、P7B、DGB diagnostics；正式投稿前联网核验并补充近三年代表性方法。
5. 论文主图展示：标准 PSNR 排名、UEFB-G stratified ranking、failure cases、claim ledger。

**顶会门槛**

- benchmark 必须公开协议和脚本；如果数据版权限制不能公开原图，至少公开生成代码、metadata、指标脚本和可复现实验 split。
- benchmark 不能只服务 RLEF；必须能评价黑箱增强方法。
- 必须展示 benchmark 改变了研究判断，而不是只是多几个指标。

### A3. RACE: Recoverability-Aware Calibrated Enhancement under Conditional Risk

**推荐等级：中高。** 适合作为 A1 的方法扩展，不建议单独立即主投。

**要解决的问题本质**  
DGB safe-router 失败说明“可恢复性路由”方向是对的，但当前 router 把更多样本送到 safe path 或 restoration path 并不必然提升 output quality。问题不是 route usage，而是 route 的条件风险没有被校准：某个区域应该增强、保持、还是保守输出，取决于该区域的噪声、结构、曝光、near-identity 状态和预测不确定性。

**核心 claim**  
Enhancement should be treated as selective restoration under recoverability risk: the model must predict not only an enhanced image, but also where enhancement is beneficial, harmful, or uncertain.

**可用数学工具**

- Selective prediction / abstention risk: 在高风险区域选择 safe output 或 lower enhancement。
- Calibration: 用 Q_ECE、A_AUC、risk reliability diagram 检查 recoverability map 是否可信。
- Conformal risk control: 对 unsafe over-enhancement 设定可控错误率。
- Conditional treatment effect: 把 enhancement 看成 treatment，估计某区域增强相对 safe output 的 expected benefit。

**实验路线**

1. 从 DGB 不继续 sweep，而是重建 route label：定义 beneficial/harmful/uncertain 区域。
2. 用 P3c/M4J_ES 输出生成 pseudo-risk targets：增强后 PSNR/SSIM/over/under 是否改善，是否损伤 near-identity 区域。
3. 训练轻量 risk head，不直接改主干；先验证 risk head 对 failure cases 的可解释性。
4. 报告 harmful over-enhance rate、identity_drop、risk ECE、selective PSNR/SSIM，而不是只报全图 PSNR。
5. 若 risk head 有稳定价值，再作为 A1/GIR-Field 的扩展模块。

**No-Go 条件**

- 如果 risk map 与真实 harmful enhancement 无相关性，不能继续包装为 recoverability innovation。
- 如果 selective route 只降低错误但大幅降低 PSNR，需要定位为 safety/diagnostic，不应写成 enhancement 主方法。

### A4. Physics-Regularized Exposure Transport for Uneven Illumination

**推荐等级：探索。** 可作为后续理论增强点，但不建议立即替代 A1。

**要解决的问题本质**  
曝光不均匀本质上具有空间传播和边缘保持结构。仅用局部 E-map 或 CNN head 容易产生不连续、过度平滑或错误跨边缘传播。可以把 exposure correction 看作 log-domain illumination transport，在图像结构引导下求解。

**可用数学工具**

- Weighted Poisson equation / anisotropic diffusion。
- Edge-aware total variation / weighted Laplacian。
- Optimal transport 或 Wasserstein barycenter，用于曝光分布校正。
- PDE-constrained learning，把网络输出作为边界条件或残差项。

**实验优先级**

只有在 A1/A2 证据链写稳后再做。否则容易变成又一个工程模块，重复 DGB 失败。

## 7. 创新点迭代记录 V0-V6

| 版本 | 想法 | 审核结果 | 是否保留 |
|---|---|---|---|
| V0 | DGB-RLEF as SOTA method | DGB joint gate false，Retinexformer gap 大 | 否 |
| V1 | DGB-RLEF as balanced RLEF method | Split-wise gain 明显，但没有统一模型；容易被审稿人认为失败 sweep | 否 |
| V2 | Centered E-shape as method module | M4J_ES 数据强，但单独作为模块论文偏窄 | 部分保留 |
| V3 | UEFB-v2/E-A-Q benchmark paper | 有独立价值，但需要更强理论和外部方法覆盖 | 保留 |
| V4 | Gauge-identifiable exposure field theory + benchmark | 能解释 M4/M4J/M4J_ES/P3c 和 DGB 失败；理论空间清晰 | 强保留 |
| V5 | 加入 recoverability conditional risk | 能吸收 DGB safe-router 失败，但要先证明 risk calibration | 作为扩展 |
| V6 | GIR-Field: gauge-identifiable + recoverability-calibrated evaluation and learning | 同时利用正证据、负结果、benchmark 和数学问题定义 | 最终推荐 |

最终不应把论文写成“DGB-RLEF: a new SOTA LLIE method”。更合理题目方向是：

- `Gauge-Identifiable Exposure Field Learning for Uneven Low-Light Enhancement`
- `When PSNR Misleads Exposure Enhancement: A Gauge-Controlled Benchmark and Failure Audit`
- `Recoverability-Calibrated Exposure Fields for Uneven Illumination Enhancement`

## 8. 下一步可直接执行的实验路线

### Phase N0: 证据清洗与结果冻结

**目的**  
把 Q&A 过程里的重复/错配回答剔除，形成可投稿的 evidence ledger。

**任务**

1. 从原始实验目录导出每次 run 的 config、seed、checkpoint、metrics JSON、per-image CSV。
2. 给每个实验分配 ID：P3c、M4J、M4J_ES、P6、P7B、Phase2 DGB、P2B、P2C、P2D、P2E。
3. 生成 `claim_ledger.csv`：每个 claim 对应 source table、metric、stat test、是否进入主文。
4. 标记 Q009/Q012 为 process-noise，不进入实验数。

**成功标准**

- 每个数值都能从原始 JSON/CSV 回溯，不只来自对话总结。
- 所有 “best” 都注明选择规则，不允许 post-hoc best split 混进主表。

### Phase N1: P3c/M4J/M4J_ES 统计复核

**目的**  
把当前最强正证据从“平均数好看”升级为“统计上可支撑”。

**必跑对象**

| Variant | 目的 |
|---|---|
| M4 | PSNR 高但 E_corr 错的反例 |
| M4J | joint training without E-shape |
| M4J_ES | centered E-shape 正证据 |
| P3c e=0.05 3-seed | 稳定 default |
| P3d e=0.10 | 说明 E-shape 权重过强的 trade-off |

**指标**

- Fidelity: PSNR、SSIM、LPIPS。
- Exposure: E_corr、S_corr、E_MAE、E_aligned_MAE、Gauge_MAE。
- Recoverability: A_AUC、Q_ECE、identity_drop、unsafe_overenhance。
- Real/no-reference: NIQE/BRISQUE 只作为附录，不能进入主 claim。

**统计检验**

- 对每个 dataset/split 做 paired bootstrap 95% CI。
- M4J vs M4J_ES 做 paired Wilcoxon signed-rank。
- 报告 effect size，不只报 p-value。
- 多指标比较用 Benjamini-Hochberg FDR 控制。

**Go/No-Go**

- Go: M4J_ES 的 E_corr/S_corr 在三域均稳定高于 M4J，且 real/synthetic PSNR 不降或提升。
- No-Go: 如果 per-image 显示提升只来自少数 outlier，A1 主线需要降级为探索性观察。

### Phase N2: UEFB-G / gauge perturbation benchmark 构建

**目的**  
证明 benchmark 不只是另一个 test set，而是能区分 gauge error、shape error 和 RGB fidelity error。

**数据构造**

1. 从 UEFB-v2 生成基础 low/high/E_gt/A_gt/Q_gt。
2. 对 E_gt 构造三类扰动：
   - Global gauge shift: `E' = E + c`。
   - Local shape distortion: `E' = E + delta(x)` 且 `mean(delta)=0`。
   - Mixed distortion: `E' = E + c + delta(x)`。
3. 每张图保存 stratum label：全局欠曝、局部曝光不均匀、高噪声、near-identity、过曝风险、纹理弱区域。

**验证问题**

- E_MAE 是否会把 gauge shift 误判为 shape error？
- S_corr 是否能正确忽略 global gauge shift？
- PSNR 是否会偏好 exposure-shape 错误但 RGB 近似的输出？
- 不同方法在不同 stratum 上是否出现系统 failure？

**输出图表**

- Figure 1: gauge/shape decomposition schematic。
- Figure 2: PSNR ranking vs gauge-identifiable ranking。
- Figure 3: stratum-wise failure heatmap。
- Table 1: benchmark metadata and split。
- Table 2: method comparison with fidelity + interpretability + calibration。

### Phase N3: 外部方法与强基线补齐

**目的**  
防止 benchmark 只对 RLEF 系列有效。

**最低要求**

- 源文件已经使用或引用的 Retinexformer、Zero-DCE++、KinD++ 必须保留。
- P3c、P6、P7B、DGB diagnostics 作为 internal methods。
- 正式论文前必须联网核验 2024-2026 最新 low-light / exposure correction 方法，优先选择官方代码、官方权重、公开推理协议。

**注意**

- 黑箱方法没有内部 E/A/Q map 时，不要强行比较 internal E_corr。应区分 `internal-field metrics` 与 `output-only metrics`。
- 对黑箱方法可增加 output proxy：local exposure consistency、over/under ratio、identity-region damage、failure-stratum quality。

### Phase N4: Recoverability risk calibration 探索

**目的**  
不是复活 DGB router，而是检验 recoverability 是否可以作为可信风险估计。

**设计**

1. 固定 P3c/M4J_ES 输出，不先改主干。
2. 构造 per-region benefit label：增强后相对 input/safe output 的局部 PSNR/SSIM/曝光偏离是否改善。
3. 训练或评估一个轻量 risk predictor，输出 beneficial/harmful/uncertain。
4. 用 reliability diagram、ECE、AUC、selective risk-coverage curve 评价。

**Go/No-Go**

- Go: risk predictor 能稳定识别 harmful over-enhancement 和 near-identity damage。
- No-Go: risk map 只能解释训练集、不能跨 real/synthetic/UEFB 泛化。

### Phase N5: 论文级统计与图表

**主表建议**

| 方法 | UEFB PSNR | UEFB S_corr | Real PSNR | Real S_corr/Gauge | Synthetic PSNR | Synthetic S_corr/Gauge | Harmful rate | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M4 |  |  |  |  |  |  |  | PSNR high but shape wrong |
| M4J |  |  |  |  |  |  |  | joint baseline |
| M4J_ES |  |  |  |  |  |  |  | centered shape |
| P3c |  |  |  |  |  |  |  | 3-seed default |
| Retinexformer |  | N/A |  | N/A |  | N/A |  | black-box strong baseline |
| Zero-DCE++ |  | N/A |  | N/A |  | N/A |  | zero-reference baseline |

**附录表**

- P5/P5b distillation no-go。
- P6/P6b/P6c structure/scalar no-go。
- P7/P7B/P7C domain/anchor no-go。
- DGB Phase2/P2B/P2C/P2D/P2E no-go。

**图例要求**

- 每张 failure case 图必须包含 input、GT、M4、M4J_ES/P3c、Retinexformer、error heatmap、E/S/gauge 或 output-proxy 指标。
- 不只展示成功图，必须展示 DGB/P6/P7 的失败图。
- 每个图例都要回答“PSNR 看不到什么？”

## 9. 论文 claim 边界

### 可以写

1. Centered/gauge-invariant exposure-shape calibration is necessary for stable local exposure-field interpretation.
2. UEFB-v2/E-A-Q exposes exposure-shape and recoverability failures that PSNR-only evaluation misses.
3. Teacher distillation, dataset-weighted reconstruction, scalar gates, domain heads and DGB routing move the tri-domain Pareto trade-off but do not resolve it under current evidence.
4. P3c/M4J_ES is a robust RLEF default/mechanism baseline, not a SOTA restoration method.

### 不能写

1. DGB-RLEF is SOTA。
2. DGB-RLEF outperforms Retinexformer。
3. DGB-RLEF resolves tri-domain trade-off。
4. E_corr improvement guarantees better visual restoration。
5. No-reference unpaired real metrics prove method superiority。

## 10. 最终建议

下一步不要继续围绕 DGB 做 P2F 或更多 router/gate scalar sweep。最值得继续深化的是：

**主线：GIR-Field = Gauge-identifiable exposure field theory + UEFB-G benchmark + centered E-shape mechanism + recoverability risk audit。**

这条线的顶会/顶刊约束是：

- 以问题定义和机制验证为核心，而不是以 PSNR SOTA 为核心。
- 用 P3c/M4J_ES 的稳定正证据作为 main evidence。
- 用 DGB/P5/P6/P7 的负结果作为 design rationale，不隐藏失败。
- 用 UEFB-G/E-A-Q 把 benchmark 从自用测试集升级为公开评价协议。
- 用统计检验、per-image CSV、claim ledger 和 failure figures 防止 cherry-picking。

一句话版本：

> 当前实验最有价值的发现不是 DGB 成功，而是 DGB 和多条路线失败后揭示的本质问题：局部曝光场增强需要在 gauge-free shape、absolute gauge calibration 与 recoverability risk 三者之间建立可识别、可评价、可统计检验的框架。围绕这个问题继续深化，才有顶会/顶刊潜力。
