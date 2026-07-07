# GIR-Field：方法与实验章节草稿（中文论证版）

> 证据来源：`docs/GIR_FIELD_FORMAL_PROTOCOL_REPORT.md`、`results/girfield_formal/`。  
> 证据边界：本章节只声称 **GIR-Field / UEFB-G 的机制与评价协议证据**，不声称 Retinexformer 级 SOTA，也不复活 DGB/P2F。

---

## 1. 方法章节：Gauge-identifiable exposure field learning

### 1.1 从 PSNR 反例出发的问题定义

标准低光增强通常以 RGB fidelity 为中心，例如 PSNR/SSIM。然而本研究的正式实验显示，单一 RGB 指标会误导 exposure-field 方法的模型选择：M4 A-gate 在 UEFB-v2 上取得更高 PSNR（18.6530），但其 gauge-free exposure-shape correlation 为负（S_corr = -0.1393）；相反，M4J_ES 的 UEFB PSNR 略低（17.9811），但 S_corr 明显为正（0.4399）。这说明增强输出的像素接近性和曝光场解释的物理一致性并不等价。

因此，本文不把目标定义为“再提出一个更强的 low-light enhancer”，而是定义一个更基础的问题：在局部曝光不均匀增强中，如何让模型学习到 **可识别的 exposure-field shape**，并用评价协议区分 RGB fidelity、absolute gauge calibration 和 spatial shape correctness。

### 1.2 Exposure field 的 gauge ambiguity

设低光图像为 \(I_l\)，目标正常曝光图像为 \(I_h\)，局部曝光场 \(E(x)\) 可以理解为 log-luminance gain：

\[
E^*(x) = \log(Y_h(x)+\epsilon)-\log(Y_l(x)+\epsilon),
\]

其中 \(Y_l,Y_h\) 为 luminance。问题在于：在 gradient-domain / Poisson-like 约束下，\(E\) 与 \(E+c\) 具有相同的局部梯度结构。也就是说，absolute exposure gauge 和 spatial exposure shape 被混在一起。若直接优化 \(E\) 的绝对误差，模型可能被全局 offset 主导；若只优化 RGB PSNR，模型可能得到 RGB 上局部更好但 exposure shape 错误的解。

为此，本文把 exposure field 分解为：

\[
\mu(E) = \frac{1}{|\Omega|}\sum_{x\in\Omega} E(x), \qquad
S(E)(x) = E(x)-\mu(E),
\]

并写作：

\[
E(x)=S(x)+\mu, \qquad \sum_{x\in\Omega}S(x)=0.
\]

在约束 \(\mathrm{mean}(S)=0\) 下，\(S\) 表示 gauge-free spatial shape，\(\mu\) 表示 absolute image-level gauge。这样，\(E\sim E+c\) 的等价类被拆解为一个 shape component 与一个 gauge component。

### 1.3 Gauge-invariant E-shape calibration

本文采用 centered low-pass exposure-shape consistency 来约束 spatial shape，而不是强制绝对 \(E\) 完全一致。令 \(\mathcal{L}_{LP}\) 表示低通滤波，\(C(A)=A-\mathrm{mean}(A)\) 表示中心化，则 E-shape loss 为：

\[
\mathcal{L}_{shape} = 1 - \mathrm{corr}\big(C(\mathcal{L}_{LP}(\hat{E})), C(\mathcal{L}_{LP}(E^*))\big)
+ \beta \|\nabla \hat{E} - \nabla E^*\|_1.
\]

该损失有两个作用：第一，中心化操作让损失对 additive gauge shift 不敏感；第二，低通项避免把局部噪声当作 exposure shape；第三，梯度项保留边缘和不均匀曝光结构。实际实现中使用 kernel size 7，默认 \(\beta=0.1\)，P3c 默认权重为 \(e\_shape=0.05\)。

### 1.4 UEFB-G：gauge-controlled evaluation protocol

为避免“PSNR 看起来更好，但 exposure field 更错”的误判，本文提出 UEFB-G 风格的评价协议。其核心不是强迫所有方法输出内部 E-map，而是区分两类指标：

1. **output-only metrics**：所有 black-box enhancer 都可评价，包括 PSNR、SSIM、over/under exposure ratio、identity_drop、local exposure error 等。
2. **internal-field metrics**：仅对显式预测 exposure field 的方法评价，包括 E_MAE、S_MAE、Gauge_MAE、S_corr 等。

核心指标定义为：

\[
\mathrm{S\_MAE}=\|S(\hat{E})-S(E^*)\|_1,
\]

\[
\mathrm{Gauge\_MAE}=|\mu(\hat{E})-\mu(E^*)|,
\]

\[
\mathrm{S\_corr}=\mathrm{corr}(S(\hat{E}),S(E^*)).
\]

为了验证这些指标确实区分 gauge error 与 shape error，本文构造三类 perturbation：

- global gauge shift：\(E'=E+c\)；
- local shape distortion：\(E'=E+\delta(x),\;\mathrm{mean}(\delta)=0\)；
- mixed distortion：\(E'=E+c+\delta(x)\)。

正式实验中，global shift +0.5 的 E_MAE 为 0.5000、Gauge_MAE 为 0.5000，但 S_MAE 为 0.0000、S_corr 为 1.0000；local shape distortion 的 Gauge_MAE 为 0.0000，但 S_MAE 为 0.2026、S_corr 降至 0.6027。这验证了 UEFB-G 指标可以把 global gauge 与 local shape error 分开。

### 1.5 Recoverability risk audit

DGB route/safe-router 的失败说明：routing usage 本身不等于 output quality。本文因此不复活 DGB router，而是把 recoverability 作为一个诊断问题：能否预测某个局部区域的增强是 beneficial、harmful 还是 uncertain？

在 formal protocol 中，我们固定 P3c/M4J_ES 输出，不改主干，构造 patch-level benefit label：若增强输出相对 input 在局部 MSE 上变差，则标为 harmful；若变好则为 beneficial。使用低图亮度、输出亮度、梯度、A/Q、E/S magnitude、saturation proxy 等特征训练轻量 logistic probe。该 probe 在 full protocol 中达到 AUC 约 0.766，但 ECE 约 0.281，说明 recoverability risk 有信号但校准不足。因此它只能作为诊断证据，不能写成已解决 recoverability 的方法贡献。

---

## 2. 实验章节：Full frozen-evidence protocol

### 2.1 实验设置

正式协议使用 frozen checkpoints，不进行新主干训练，不恢复 DGB/P2F。数据规模为：UEFB-v2 test 500 张，LOL-v2-real test 100 张，LOL-v2-synthetic test 100 张。9 个内部 variant 每个评估 700 张图，共 6300 条 per-image records。统计检验使用 5000 次 bootstrap、paired Wilcoxon signed-rank test，并用 Benjamini-Hochberg FDR 控制多重比较。

内部方法包括：M4、M4J、M4J_ES、P3c e=0.05 三种 seed、P3d e=0.10 三种 seed。外部 black-box baseline 包括 Retinexformer、Zero-DCE++、KinD++。由于 black-box baseline 不输出内部 exposure field，因此只报告 output-only metrics，不报告 S_corr/Gauge_MAE。

### 2.2 主表结果

| Method | UEFB PSNR | UEFB S_corr | Real PSNR | Real S_corr | Synthetic PSNR | Synthetic S_corr |
|---|---:|---:|---:|---:|---:|---:|
| M4 A-gate | 18.6530 | -0.1393 | 17.4113 | -0.3314 | 13.9632 | -0.3372 |
| M4J joint | 18.0818 | 0.2818 | 19.5807 | 0.5007 | 16.5085 | 0.3690 |
| M4J_ES seed3407 | 17.9811 | 0.4399 | 20.2770 | 0.6483 | 17.1405 | 0.7992 |
| P3c e=0.05 3-seed | 17.9145 | 0.4361 | 20.0206 | 0.7075 | 17.6783 | 0.8407 |
| P3d e=0.10 3-seed | 18.1108 | 0.4491 | 19.8749 | 0.6856 | 17.6263 | 0.8347 |
| Retinexformer | — | N/A | 22.7939 | N/A | 25.6690 | N/A |
| Zero-DCE++ | — | N/A | 18.4907 | N/A | 17.5764 | N/A |
| KinD++ | — | N/A | 22.2109 | N/A | 19.2594 | N/A |

该表给出两个核心判断。第一，M4 是典型 PSNR-only misranking：它有最高 UEFB PSNR，但 S_corr 为负。第二，M4J_ES/P3c 的贡献不是 SOTA restoration，而是 exposure-field interpretability 与 paired fidelity 的共同改善。

### 2.3 Centered E-shape 的统计证据

M4J_ES 相比 M4J 在 full protocol 中表现稳定：

| Comparison | Dataset | Metric | Delta | 95% CI | FDR q | Win rate |
|---|---|---:|---:|---:|---:|---:|
| M4J_ES - M4J | UEFB | S_corr | +0.1581 | [0.1369, 0.1791] | 2.18e-43 | 0.826 |
| M4J_ES - M4J | Real | PSNR | +0.6963 dB | [0.4141, 0.9664] | 1.57e-05 | 0.660 |
| M4J_ES - M4J | Synthetic | PSNR | +0.6321 dB | [0.4182, 0.8559] | 6.04e-08 | 0.760 |
| M4J_ES - M4J | Synthetic | S_corr | +0.4301 | [0.3499, 0.5160] | 1.72e-17 | 0.990 |

这些结果支持本文的核心机制 claim：centered E-shape calibration 显著改善 gauge-free exposure shape，并且这种改善没有以 real/synthetic PSNR 下降为代价。

### 2.4 PSNR-only misranking

M4J_ES 与 M4 的比较显示，如果只按 UEFB PSNR 排序，M4 会被错误地认为优于 M4J_ES；但在 exposure-shape 上，M4J_ES 相比 M4 的 UEFB S_corr 提升为 +0.5792，95% CI [0.5361, 0.6230]，FDR q = 3.52e-64，win rate = 0.858。该结果说明 UEFB-G 的价值不是“多报几个指标”，而是改变了研究判断：它暴露了 PSNR 无法发现的 exposure-shape failure。

### 2.5 E-shape 权重的 trade-off

P3c e=0.05 与 P3d e=0.10 的比较显示，更强 E-shape 权重在 UEFB PSNR/S_corr 上略有收益，但 real/synthetic 上不如 e=0.05 稳定。P3d 的 UEFB PSNR 为 18.1108、S_corr 为 0.4491；P3c 的 UEFB PSNR 为 17.9145、S_corr 为 0.4361。但 P3c 在 real S_corr（0.7075 vs 0.6856）和 synthetic S_corr（0.8407 vs 0.8347）上更稳，synthetic PSNR 也略高（17.6783 vs 17.6263）。因此 e=0.05 仍应作为当前 default，而 e=0.10 作为 stronger-shape control。

### 2.6 与外部强基线的边界

Retinexformer 在 paired fidelity 上仍明显更强：real PSNR 22.7939、synthetic PSNR 25.6690，显著高于 P3c 的 20.0206 / 17.6783。因此本文不能声称提出了 SOTA LLIE 方法。相反，Retinexformer 的存在恰好界定了本文定位：GIR-Field 贡献在于可识别 exposure-field learning 与 gauge-controlled benchmark，而不是 RGB fidelity 领先。

### 2.7 Recoverability risk 的诊断结果

Formal risk probe 的结果为：M4J_ES AUC = 0.7658、ECE = 0.2808；P3c seed3407 AUC = 0.7655、ECE = 0.2809。AUC 表明 harmful enhancement 有可预测信号，但 ECE 表明概率校准仍不可靠。因此 recoverability risk 目前应写作 diagnostic audit，而不是主方法模块。

---

## 3. 本章可写与不可写的结论

可以写：

1. Centered/gauge-invariant exposure-shape calibration 在 full local eval 上显著提升 S_corr。
2. UEFB-G 能区分 global gauge shift 与 local shape error。
3. PSNR-only 会误判 M4 这类 RGB 上较优但 exposure shape 错误的模型。
4. Recoverability risk 有预测信号，但 calibration 不足。

不能写：

1. GIR-Field/P3c 超过 Retinexformer。
2. DGB-RLEF 解决三域 trade-off。
3. E_corr 提升必然等于视觉增强更好。
4. Risk head 已经解决 recoverability。

---

## 4. 下一步写作建议

这两个章节应作为论文主体的 Method 与 Experiments 初稿。后续如果要进一步投稿化，应补：Related Work、formal theorem/proposition、更多 2024-2026 外部方法、以及 failure-case figure caption 的精修。
