# A1-RLEF v2 实验指导文件：基于过程数据审计后的局部曝光场研究路线

## Material Passport

- Origin Skill: using-superpowers + academic-research-suite / academic-paper-reviewer + experiment-agent
- Origin Mode: review + experiment plan
- Origin Date: 2026-07-01
- Verification Status: ANALYZED（基于本地 A1 方案与实验过程 JSON；未在本轮重新训练模型）
- Version Label: A1_RLEF_v2_after_process_audit_20260701
- Input Plan: `G:\9chapterwork\workspace\image_enhancement_research_20260629\A1_local_exposure_field_experiment_plan.md`
- Process Data: `G:\9chapterwork\workspace\image_enhancement_research_20260629\image_enhancement_sessions_merged_20260630.json`
- Output Purpose: 作为下一轮实验落地文档，指导环境、数据、算法、公式、消融、实验执行和论文级 claim 收敛。

---

## 0. 结论先行

A1 原方案的方向是有价值的，但原始表述过大：`局部曝光场 + CRF + 异方差噪声 + 不确定性 + RAW/ISP` 不能全部作为当前主贡献。过程数据中的真实实验更支持一个收缩后的顶会路线：

> Recoverability/Gauge-aware Local Exposure Field Regularization for Low-Light and Uneven-Exposure Enhancement。

中文可表述为：

> 面向低光和曝光不均图像增强的规范化局部曝光场建模：通过 gauge 约束、可解释曝光场评估和恢复性/近恒等样本识别，提升真实 paired RGB 场景的增强稳定性，并揭示 real/synthetic domain tension。

当前证据最强的创新点不是“堆物理模块”，而是：

1. Poisson/梯度域曝光场存在 additive gauge ambiguity，必须显式锚定或规范化。
2. `e_mean` gauge anchor 在 LOL-v2-real 上显著提升 paired fidelity，但在 synthetic split 上会产生 domain tension。
3. 只把预测曝光场拉近 oracle log-gain 可以改善 E-field 指标，却不一定改善最终 RGB 图像，说明曝光场和图像重建之间需要更强的耦合设计。
4. tiny U-Net 风格 LEPI 主干不能独自追上 Retinexformer；更合理路线是把局部曝光场作为强 restoration backbone 的可解释辅助分支/正则，而不是用弱主干硬追 SOTA。

因此，本文件推荐将 A1 迭代为 **RLEF-Net / RLEF-Former**：Recoverability-aware Local Exposure Field Network。主论文不要声称全物理闭环已经解决，应聚焦“规范化曝光场 + domain tension + 可解释评估协议 + 强主干融合”。

---

## 1. 过程数据审计摘要

### 1.1 JSON 基本结构

本轮审计读取到的过程数据为一个合并会话集合：

| 项目 | 数值 |
|---|---:|
| sessions | 6 |
| messages | 1685 |
| tool messages | 998 |
| assistant messages | 654 |
| user messages | 33 |
| terminal calls | 276 |
| read_file calls | 259 |
| patch calls | 107 |
| skill_view calls | 91 |
| search_files calls | 85 |
| tests 最高记录 | 23 passed |

过程数据不是最终实验表，而是连续压缩会话的实验探索轨迹。可用证据主要来自 assistant stop 总结、工具输出、训练/评估结果和阶段报告路径。

### 1.2 已完成的关键实验链条

| 阶段 | 目的 | 关键结果 | 对论文路线的影响 |
|---|---|---|---|
| 初始 smoke | 跑通 LEPI-Net compact 原型与 synthetic uneven-exposure | M0/M1 PSNR 约 22.7，Poisson/full 的 aligned E_MAE 更低但 PSNR 不占优 | 证明代码闭环可跑，但不能证明 SOTA |
| gauge anchor sweep | 修复 Poisson E-field additive ambiguity | LOL-v2-real PSNR `19.463 -> 22.111`，LEE `0.216 -> 0.158`；LOL-v2-synthetic 退化 | gauge 约束是当前最强机制点 |
| SOTA closure | 接入 Retinexformer/Zero-DCE++ official | LOL-v2-real: LEPI 21.921, Retinexformer 22.790；LOL-v2-synthetic: LEPI 14.232, Retinexformer 25.648 | 不能写 outperform SOTA；要写 mechanism + boundary |
| synthetic failure audit | 分析 synthetic 失败原因 | anchor hurts `79/100`；GT-mean scaling 只能把 14.232 提到 17.405；E_corr 为负 | 失败不是简单亮度偏移，是空间形状/域分布/容量耦合问题 |
| synthetic-specific training | 排查是否模型完全不适配 synthetic | synthetic train/test 后 no-anchor 达 20.428，超过 Zero-DCE++ 17.545 | 主要是 domain mismatch，不是模型完全无效 |
| oracle-E diagnostic | 监督 E 接近 oracle log-gain | E_MAE `0.686 -> 0.387`，E_corr `0.686 -> 0.833`，但 PSNR `20.402 -> 20.004` | E-field 正确不等于 RGB 正确，需耦合重构 |
| capacity check | 加宽 c24/c48/c64 | c48 最好 20.802，c64 无收益；仍落后 Retinexformer 4.847 dB | width-only 不是主瓶颈 |
| residual refine | 小 RGB residual head | c48 20.802 -> 21.163，提升 +0.361 dB，未达 +0.5 dB go/no-go | 有用但不是主解法 |
| stronger decoder | residual branch blocks=1/2 | blocks=1/2 反而下降 | 不应继续堆 post-hoc decoder |
| formulation loss | base_rec + residual_l1 | 压小 residual 但 PSNR 下降 | 不能靠 residual 正则解决 synthetic gap |
| direct RGB head | direct branch + identity loss | 最好 20.796，低于 shallow residual 21.163；base path 退化 | direct dual-head 不能作为主路线 |

### 1.3 顶会/顶刊审稿视角结论

当前可以保留的 claim：

- Gauge-aware local exposure-field modeling 对 real paired RGB low-light enhancement 有实验证据。
- Poisson/梯度域曝光场需要 gauge fixing；否则绝对曝光场不可识别。
- Synthetic split 暴露了真实低光和合成低光之间的 domain tension，这可以成为论文贡献的一部分，而不是回避的失败。
- E-field 解释性必须和重建质量联合评价，单独改善 E_MAE 不能证明增强更好。

当前必须删除或降级的 claim：

- 不能写 LEPI-Net outperforms SOTA。
- 不能把 CRF/noise/U 作为主贡献，过程数据没有足够正向证据。
- 不能声称 RAW/ISP physical inversion 已被验证。
- 不能把 direct RGB branch 当作“性能+可解释性双赢”，实验显示它会破坏 Retinex base path。

推荐论文定位：

> A rigorous exposure-field regularization and evaluation framework, integrated with a competitive restoration backbone, that improves real-paired low-light enhancement while revealing and addressing domain-dependent exposure gauge failures.

---

## 2. 迭代后的创新点设计

### 2.1 最终主创新点 C1：Gauge-normalized Local Exposure Field

#### 要解决的问题本质

Poisson/梯度域约束只约束 `nabla E`，不约束 `E` 的全局常数项。因此 E-field 有 additive gauge ambiguity：

```math
E'(x) = E(x) + c
```

其梯度相同，但曝光倍率整体不同，最终增强结果可能完全不同。这是 A1 原方案里最关键的理论缺口。

#### 主要 claim

> Local exposure-field inversion is ill-posed under gradient-domain Poisson regularization; explicit gauge normalization or adaptive anchoring is necessary for physically interpretable and stable low-light enhancement.

#### 工程落点

- 输出 `E(x)` 后加入 gauge head 或 adaptive anchor。
- 不再固定单一 `e_mean=0.02`，改成输入自适应 `mu_E(I)`。
- 在 synthetic benchmark 上同时报告 `E_MAE`、`E_MAE_aligned`、`E_corr`，区分 offset 错误和空间形状错误。

#### 预期效果

- real paired RGB 上 PSNR/LEE 更稳定。
- 避免某些图像整体过增强或欠增强。
- 能把 Poisson 约束从“隐式正则”变为可审稿的数学贡献。

---

### 2.2 最终主创新点 C2：Recoverability-aware Exposure Control

#### 要解决的问题本质

低光增强里有两类输入：

1. 本来已经接近正常曝光的 near-identity 样本，过度增强会退化。
2. 极暗/过曝/饱和区域，本质上信息不可恢复，强行生成会产生幻觉。

过程数据里的 synthetic bucket analysis 显示：`input already close >=18dB` 时，LEPI 会把原本接近 GT 的输入从 23.954 PSNR 降到 16.343，说明缺少 recoverability/identity-aware 控制。

#### 主要 claim

> Enhancement should be conditioned on local recoverability and near-identity likelihood; a model should learn where to enhance, where to preserve, and where to report uncertainty.

#### 工程落点

- 输出 `A(x)`：local adjustment gate，控制 `I_base` 与增强输出的混合。
- 输出 `Q(x)`：recoverability/confidence map，不直接作为未验证主 claim，而作为训练权重与评估对象。
- 加入 identity-aware loss，避免 near-identity 样本被强行增强。

#### 预期效果

- synthetic near-identity bucket 的退化减少。
- unpaired real 的过增强、噪声放大和饱和率下降。
- 给 failure awareness 提供比原 U-head 更清晰的定义。

---

### 2.3 最终主创新点 C3：Exposure-field Auxiliary Branch on Strong Restoration Backbone

#### 要解决的问题本质

过程数据显示 tiny LEPI backbone 即使加宽到 c48/c64、加 residual refine、加 direct RGB head，也无法接近 Retinexformer。瓶颈不是单个 head，而是整体 restoration capacity 和特征表达。

#### 主要 claim

> Local exposure fields are most useful as interpretable regularizers and auxiliary supervision for a strong restoration backbone, rather than as the sole reconstruction pathway of a small Retinex network.

#### 工程落点

- 主干采用 Restormer-lite / Retinexformer-like encoder-decoder。
- 主输出 `I_hat` 由强 restoration branch 负责。
- 辅助输出 `E(x), A(x), Q(x)` 进入损失、可视化和 failure analysis。
- 不再让 `R * L * exp(E)` 独自承担所有重建任务；它保留为 `I_phys` 监督与解释分支。

#### 预期效果

- 缩小与 Retinexformer 的 paired benchmark gap。
- 保留曝光场解释性，避免 direct RGB branch 破坏 base path。
- 更符合顶会审稿对“性能 + 机制”的要求。

---

## 3. 最终算法方案：RLEF-Net / RLEF-Former

### 3.1 模型输入与输出

输入：低光或曝光不均 RGB 图像

```math
I_l \in [0,1]^{H \times W \times 3}
```

输出：

| 输出 | 符号 | 用途 | 是否主输出 |
|---|---|---|---|
| 增强图像 | `I_hat` | 论文主图像结果与 PSNR/SSIM/LPIPS 评价 | 是 |
| 物理分支增强图 | `I_phys` | Retinex/exposure-field 解释分支 | 否，辅助 |
| 局部曝光场 | `E(x)` | log-exposure correction | 核心解释变量 |
| 自适应 gauge | `mu_E` | 修复全局 additive ambiguity | 核心机制 |
| 调整门控 | `A(x)` | 控制增强/保留，保护 near-identity 区域 | 核心机制 |
| 恢复性图 | `Q(x)` | 表示局部是否可可靠恢复 | 辅助机制 |
| 可选噪声图 | `sigma(x)` | 仅在后续噪声实验中启用 | 扩展 |

### 3.2 推荐结构

```text
I_l
 ├─ shallow features: conv + gradient/luminance embedding
 ├─ strong restoration backbone: Restormer-lite / Retinexformer-style blocks
 │   └─ restoration head -> I_rest
 ├─ exposure auxiliary branch
 │   ├─ E_head -> raw E(x)
 │   ├─ gauge_head -> mu_E(I)
 │   ├─ A_head -> adjustment gate A(x)
 │   └─ Q_head -> recoverability Q(x)
 ├─ physics branch
 │   ├─ L_head -> L_l(x)
 │   ├─ R_head -> R(x)
 │   └─ I_phys = R(x) * L_l(x) * exp(E_norm(x))
 └─ final fusion
     I_hat = A(x) * I_rest + (1 - A(x)) * I_l
```

关键变化：

- `I_rest` 是主输出能力来源，避免弱 Retinex 主干拖累指标。
- `E_norm` 是规范化后的曝光场，不再使用未约束 `E`。
- `I_phys` 不再强行作为 final output，而是提供曝光场解释、损失约束和可视化。
- `A(x)` 保护 near-identity 样本，解决 synthetic bucket 中“本来接近 GT 却被增强坏”的问题。

### 3.3 曝光场规范化

原始曝光场：

```math
E_{raw}(x) = h_E(F(I_l))(x)
```

输入自适应 gauge：

```math
\mu_E(I_l) = h_\mu(\operatorname{GAP}(F(I_l)))
```

规范化曝光场：

```math
E(x) = E_{raw}(x) - \operatorname{mean}(E_{raw}) + \mu_E(I_l)
```

其中 `mu_E` 不应固定为 `0.02`，而应由图像内容预测。训练时可以用目标曝光均值或 oracle log-gain 均值作弱监督：

```math
\mu_E^{gt} = \operatorname{mean}(\log(Y_h+\epsilon)-\log(Y_l+\epsilon))
```

```math
\mathcal{L}_{gauge} = |\mu_E(I_l) - \operatorname{sg}(\mu_E^{gt})|
```

如果没有 GT，则用 well-exposed prior 或无监督亮度范围约束，只作为 fine-tuning 辅助。

### 3.4 Poisson exposure-field 约束

对 paired 数据，定义亮度线性近似：

```math
Y_l = \operatorname{lum}(I_l), \quad Y_h = \operatorname{lum}(I_h)
```

目标 log-gain 梯度：

```math
b = \nabla \log(Y_h+\epsilon) - \nabla \log(Y_l+\epsilon)
```

变分目标：

```math
\min_E \int_\Omega \|\nabla E - b\|_2^2 dx
```

Euler-Lagrange 推导：

```math
\delta \mathcal{J}
= 2\int_\Omega (\nabla E-b)\cdot\nabla\delta E dx
= -2\int_\Omega \nabla\cdot(\nabla E-b)\delta E dx
```

令任意 `delta E` 下变分为零：

```math
\nabla\cdot(\nabla E-b)=0
```

得到 Poisson 方程：

```math
\Delta E = \nabla\cdot b
```

训练损失：

```math
\mathcal{L}_{poisson}=\|\Delta E - \nabla\cdot b\|_1
```

注意：该损失只约束空间形状，不约束全局常数。因此必须和 `L_gauge` 或 mean-normalized evaluation 配合使用。

### 3.5 Recoverability / identity-aware 控制

定义 near-identity 权重：

```math
W_{id}(x)=\exp\left(-\frac{|I_h(x)-I_l(x)|}{\tau}\right)
```

增强门控输出：

```math
A(x)=\sigma(h_A(F(I_l))(x))
```

最终输出：

```math
\hat{I}(x)=A(x)\hat{I}_{rest}(x)+(1-A(x))I_l(x)
```

identity protection：

```math
\mathcal{L}_{id}=\| W_{id}\odot(\hat{I}-I_l)\|_1
```

为了防止模型把 `A(x)` 全部压到 0，需要加入任务重建损失和门控均值约束：

```math
\mathcal{L}_{gate}=\left|\operatorname{mean}(A)-\operatorname{mean}(A^{gt})\right|
```

其中 `A_gt` 可以由 oracle exposure magnitude 与 residual 构造：

```math
A^{gt}(x)=\operatorname{clip}\left(\frac{|\log(Y_h+\epsilon)-\log(Y_l+\epsilon)|}{q_{95}},0,1\right)
```

恢复性图 `Q(x)` 不直接决定 final output，而用于训练加权和校准评价：

```math
Q^{gt}(x)=1-\operatorname{clip}\left(\frac{|\hat{I}(x)-I_h(x)|}{q_{95}},0,1\right)
```

```math
\mathcal{L}_{Q}=\|Q-\operatorname{sg}(Q^{gt})\|_1 + \operatorname{BCE}(1-Q, M_{bad})
```

### 3.6 总损失

推荐 v2 总损失：

```math
\mathcal{L}
=\lambda_{rec}\mathcal{L}_{rec}
+\lambda_{phys}\mathcal{L}_{phys}
+\lambda_{poi}\mathcal{L}_{poisson}
+\lambda_{gauge}\mathcal{L}_{gauge}
+\lambda_{id}\mathcal{L}_{id}
+\lambda_{gate}\mathcal{L}_{gate}
+\lambda_Q\mathcal{L}_Q
+\lambda_{tv}\mathcal{L}_{wTV}
```

各项定义：

```math
\mathcal{L}_{rec}=\|\hat{I}-I_h\|_1 + 0.2(1-SSIM(\hat{I},I_h)) + 0.05 LPIPS(\hat{I},I_h)
```

```math
\mathcal{L}_{phys}=\|I_{phys}-I_h\|_1
```

```math
\mathcal{L}_{wTV}=\sum_x \exp(-\alpha|\nabla Y_l(x)|)|\nabla E(x)|
```

初始权重建议：

| loss | weight | 说明 |
|---|---:|---|
| `rec` | 1.0 | 主图像质量 |
| `phys` | 0.15 | 保持物理分支不退化 |
| `poisson` | 0.05 | 曝光场空间约束 |
| `gauge` | 0.10 | 修复 additive ambiguity |
| `id` | 0.02 | 防 near-identity 过增强 |
| `gate` | 0.02 | 防 A collapse |
| `Q` | 0.02 | 恢复性校准 |
| `wTV` | 0.02 | 抑制曝光场噪声 |

不要一开始加入 CRF/noise NLL。它们在过程数据中没有形成正证据，建议放到第二阶段单独验证。

### 3.7 可选 CRF/noise 扩展的准入门槛

只有满足以下条件后，才恢复 CRF/noise/U 扩展：

1. RLEF-Former 在 LOL-v2-real 接近或超过 Retinexformer official，同协议差距小于 `0.3 dB`。
2. synthetic train/test 的 PSNR 超过 `22.0 dB`，且 E_corr 高于 `0.75`。
3. near-identity bucket 不再明显退化，PSNR 不低于 input baseline。
4. 3 个 seed 的主指标方差可控。

CRF 恢复时只作为 monotonic calibration module，不作为主创新点。noise NLL 只在 SID/RAW 或合成噪声专项中评估。

---

## 4. 环境要求

### 4.1 推荐硬件

| 级别 | GPU | 显存 | 用途 |
|---|---:|---:|---|
| smoke | RTX 3060/4060 | 8-12 GB | 256 crop、小 batch、单 seed |
| 标准 | RTX 3090/4090/A5000 | 24 GB | 384 crop、完整 ablation、3 seeds |
| 强实验 | 2-4 张 24 GB GPU | 48-96 GB | 多数据集、多 backbone、RAW/SID 扩展 |

### 4.2 推荐系统与 Python 环境

优先使用 Ubuntu 22.04 / WSL2 Ubuntu 22.04。Windows 原生可以做数据整理和少量 smoke，但训练、baseline 复现和 CUDA 兼容建议在 Linux/WSL2 中完成。

```bash
conda create -n rlef python=3.10 -y
conda activate rlef

# 按 CUDA 版本选择，这里以 cu121 为例
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install opencv-python pillow imageio scikit-image tqdm pyyaml einops timm kornia lpips
pip install pandas numpy scipy matplotlib seaborn tensorboard torchmetrics piq thop natsort
pip install rawpy colour-science
```

可选：如果复用 BasicSR/Retinexformer 生态，需要单独隔离环境，避免依赖污染主实验环境。

```bash
conda create -n rlef_baselines python=3.8 -y
conda activate rlef_baselines
# 根据 baseline 官方仓库安装对应版本
```

### 4.3 固定随机性

每次训练必须保存 seed 和环境信息。代码中固定：

```python
seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

正式表格至少跑：

```text
seed = 3407, 2027, 42
```

---

## 5. 数据集要求

### 5.1 第一阶段必须数据

| 数据集 | 用途 | 当前过程数据状态 | 必须性 |
|---|---|---|---|
| LOL-v1 | 经典 paired RGB benchmark | 已曾记录就绪，train 485/test 15 | 必须 |
| LOL-v2-real | 真实 captured paired benchmark | 已曾记录手动放置，train 689/test 100 | 必须 |
| LOL-v2-synthetic | 合成 paired benchmark，暴露 domain tension | 已曾记录手动放置，train 900/test 100 | 必须 |
| LIME/NPE/MEF/DICM/VV | unpaired real 泛化 | 已曾记录共 128 张 | 必须，用于真实泛化 |
| UEFB-v2 自建 benchmark | 有 GT exposure field 的专项测试 | 需要新生成 | 必须，支撑核心 claim |

### 5.2 第二阶段扩展数据

| 数据集 | 用途 | 是否进入主论文 |
|---|---|---|
| SICE | 多曝光/曝光不均 | 推荐，如果能稳定获取 |
| MIT-Adobe FiveK | 高质量正常曝光图，用于合成 UEFB-v2 | 推荐 |
| SID Sony/Fuji | RAW 噪声与物理扩展 | 不进入 v2 主线，作为扩展 |
| SMID/SDSD/NTIRE | 高分辨率与视频/挑战扩展 | 可选 |

### 5.3 推荐目录结构

所有项目和数据建议放在固定 workspace：

```text
G:\9chapterwork\workspace\a1_rlef_project\
  data\
    LOL-v1\
      train\low\
      train\high\
      test\low\
      test\high\
    LOL-v2\
      Real_captured\Train\Low\
      Real_captured\Train\Normal\
      Real_captured\Test\Low\
      Real_captured\Test\Normal\
      Synthetic\Train\Low\
      Synthetic\Train\Normal\
      Synthetic\Test\Low\
      Synthetic\Test\Normal\
    unpaired_real\
      LIME\
      NPE\
      MEF\
      DICM\
      VV\
    UEFB-v2\
      train\low\
      train\high\
      train\E_gt\
      train\A_gt\
      train\Q_gt\
      test\low\
      test\high\
      test\E_gt\
      test\A_gt\
      test\Q_gt\
  configs\
  src\rlef\
  scripts\
  experiments\
  results\
  docs\
  paper\
```

### 5.4 数据预处理规范

paired RGB 数据：

1. 统一读为 RGB，范围 `[0,1]`。
2. 训练 crop 先用 `256x256`，正式训练用 `384x384`。
3. 随机水平翻转、垂直翻转、90 度旋转。
4. 不在主训练中随机 gamma/brightness，否则会污染 exposure field 学习；这些只能作为鲁棒性实验。
5. 评估不使用 GT mean trick；任何 oracle scaling 只能作为 diagnostic。

unpaired real 数据：

1. 保留原分辨率评估。
2. 使用 tile inference 防止显存溢出。
3. 指标只使用 no-reference 和 proxy，不伪装成 paired 结果。

### 5.5 UEFB-v2：必须新增的专项 benchmark

UEFB-v2 是本路线能否达到顶会标准的关键。它用于直接验证 `E(x)`、`A(x)` 和 `Q(x)`，解决普通 LOL benchmark 无法评价曝光场是否正确的问题。

从高质量正常曝光图 `I_h` 合成低光/曝光不均图：

```math
Y_l = \operatorname{clip}\left(Y_h \cdot \exp(-E_{syn}(x)) + \epsilon(x), 0, 1\right)
```

```math
I_l = \operatorname{sRGB}(Y_l)
```

其中 `E_syn(x)` 由以下分量随机组合：

| 分量 | 公式/生成方式 | 目的 |
|---|---|---|
| global exposure | 常数 `c ~ U(0.2, 2.2)` | 普通低光 |
| linear gradient | `a x + b y` | 左右/上下曝光不均 |
| vignette | 中心到边缘二次函数 | 暗角 |
| spotlight | 1-3 个 Gaussian blob | 局部灯光/阴影 |
| occlusion shadow | smooth random field | 大面积阴影 |
| near-identity mask | 部分图像 E 接近 0 | 防过增强测试 |
| saturation region | 局部 clip 到 1 | 不可恢复高光 |
| underexposed region | 局部 clip 到 0 | 不可恢复暗区 |

合成时保存：

```text
I_l       输入低光图
I_h       GT 正常曝光图
E_gt      合成 log-exposure field
A_gt      应增强区域门控
Q_gt      recoverability map
meta.json 每张图的合成参数
```

推荐规模：

| split | 数量 | 来源 |
|---|---:|---|
| train | 3000-10000 | FiveK/SICE/LOL normal images |
| val | 300 | 与 train 不重叠 |
| test | 500 | 固定 seed，论文表格使用 |

必须固定合成 seed 并发布生成脚本：

```text
scripts/make_uefb_v2.py --seed 3407 --source data/FiveK --output data/UEFB-v2
```

### 5.6 UEFB-v2 生成伪代码

```python
def synthesize_uefb_sample(I_high, rng):
    Y = srgb_to_linear(luminance_or_rgb(I_high))
    E_global = rng.uniform(0.2, 2.2)
    E_field = E_global
    E_field += sample_linear_gradient(rng)
    E_field += sample_vignette(rng)
    E_field += sample_gaussian_spotlights(rng)
    E_field += sample_smooth_shadow_field(rng)

    near_identity = rng.random() < 0.20
    if near_identity:
        E_field *= rng.uniform(0.0, 0.15)

    Y_low = Y * torch.exp(-E_field)
    Y_low = add_poisson_gaussian_noise(Y_low, rng)
    Y_low, M_over, M_under = apply_random_clipping(Y_low, rng)

    A_gt = normalize_abs(E_field)
    Q_gt = 1.0 - soft_union(M_over, M_under, severe_noise_mask(Y_low))
    I_low = linear_to_srgb(Y_low)
    return I_low, I_high, E_field, A_gt, Q_gt
```

---

## 6. 代码落地结构

建议直接创建新项目，不在旧 `/home/user/...` 路径上继续。固定位置：

```text
G:\9chapterwork\workspace\a1_rlef_project
```

核心文件：

```text
src/rlef/
  datasets/
    paired_rgb_dataset.py
    unpaired_rgb_dataset.py
    uefb_dataset.py
  models/
    rlef_former.py
    exposure_heads.py
    restoration_backbone.py
    physics_branch.py
  losses/
    reconstruction.py
    exposure_poisson.py
    gauge_losses.py
    recoverability_losses.py
  metrics/
    full_reference.py
    exposure_field.py
    no_reference.py
    calibration.py
  utils/
    seed.py
    image_io.py
    env.py
    visualization.py
scripts/
  make_uefb_v2.py
  train.py
  eval_paired.py
  eval_unpaired.py
  eval_uefb.py
  run_baselines.py
  make_tables.py
  export_visuals.py
configs/
  rlef_m0_restorer.yml
  rlef_m1_phys_aux.yml
  rlef_m2_gauge.yml
  rlef_m3_gate.yml
  rlef_m4_full.yml
  rlef_uefb_pretrain.yml
```

### 6.1 训练命令模板

```bash
python scripts/train.py ^
  --config configs/rlef_m4_full.yml ^
  --data_root G:\9chapterwork\workspace\a1_rlef_project\data ^
  --output_dir G:\9chapterwork\workspace\a1_rlef_project\experiments\rlef_m4_full_lolv2_seed3407 ^
  --seed 3407
```

### 6.2 paired 评估命令模板

```bash
python scripts/eval_paired.py ^
  --config configs/rlef_m4_full.yml ^
  --checkpoint G:\9chapterwork\workspace\a1_rlef_project\experiments\rlef_m4_full_lolv2_seed3407\checkpoints\best_psnr.pth ^
  --dataset LOL-v2-real ^
  --data_root G:\9chapterwork\workspace\a1_rlef_project\data ^
  --output_dir G:\9chapterwork\workspace\a1_rlef_project\results\paired\rlef_m4_full\LOL-v2-real
```

### 6.3 UEFB 评估命令模板

```bash
python scripts/eval_uefb.py ^
  --checkpoint experiments\rlef_m4_full_lolv2_seed3407\checkpoints\best_psnr.pth ^
  --data_root data\UEFB-v2 ^
  --output_dir results\uefb\rlef_m4_full
```

### 6.4 unpaired 评估命令模板

```bash
python scripts/eval_unpaired.py ^
  --checkpoint experiments\rlef_m4_full_lolv2_seed3407\checkpoints\best_psnr.pth ^
  --input_root data\unpaired_real ^
  --output_dir results\unpaired\rlef_m4_full ^
  --tile 512 --tile_overlap 64
```

---

## 7. 实验路线设计

### 7.1 总体阶段

| 阶段 | 目标 | 数据 | 产物 | Go/No-Go |
|---|---|---|---|---|
| P0 | 环境和数据 smoke | LOL-v1 20 张 + UEFB 20 张 | dataloader、指标、可视化 | 输出非空，loss 可降 |
| P1 | 复现当前证据 | LOL-v2-real/synthetic | no-anchor/e_mean 对照 | 复现 real 提升和 synthetic tension |
| P2 | UEFB-v2 生成与验证 | FiveK/SICE/LOL normal | 固定 UEFB-v2 | E_gt/A_gt/Q_gt 可视化合理 |
| P3 | RLEF-Former MVP | LOL-v2-real + UEFB | M0-M4 消融 | 相比 LEPI c48 明显提升 |
| P4 | SOTA 同协议对比 | LOL-v1/v2/UEFB | Retinexformer/Zero-DCE++/KinD++ 同表 | 不写 SOTA，除非真实超过 |
| P5 | real unpaired 泛化 | LIME/NPE/MEF/DICM/VV | no-ref 指标 + 视觉图 | 过增强减少，噪声不过度放大 |
| P6 | 多 seed 正式实验 | all core datasets | mean±std, significance | 3 seed 稳定 |
| P7 | 可选 RAW/CRF/noise | SID/SICE | 扩展表 | 只有主线稳定后执行 |

### 7.2 P0：环境 smoke

目标：确认项目能跑，不做论文结论。

执行：

```bash
python scripts/make_uefb_v2.py --source data\LOL-v2\Real_captured\Train\Normal --output data\UEFB-v2-smoke --num_train 20 --num_test 20 --seed 3407
python scripts/train.py --config configs/rlef_m0_restorer.yml --max_steps 100 --output_dir experiments\smoke_m0_seed3407
python scripts/eval_paired.py --checkpoint experiments\smoke_m0_seed3407\checkpoints\last.pth --dataset LOL-v1 --max_images 15 --output_dir results\smoke\LOL-v1
python scripts/eval_uefb.py --checkpoint experiments\smoke_m0_seed3407\checkpoints\last.pth --data_root data\UEFB-v2-smoke --output_dir results\smoke\UEFB
pytest tests -q
```

验收：

- `pytest` 全部通过。
- `I_hat/E/A/Q` 图像可打开且非空。
- `PSNR/SSIM/E_MAE/E_corr` 能写入 JSON/CSV。
- train loss 不出现 NaN。

### 7.3 P1：复现过程数据中的关键现象

目的：确保新项目能复现旧过程数据的核心结论，而不是在新代码中直接跳到新方法。

实验：

| ID | 模型 | 数据 | 目标 |
|---|---|---|---|
| P1-a | LEPI-like no-anchor | LOL-v2-real | 复现 no-anchor baseline |
| P1-b | LEPI-like fixed e_mean=0.02 | LOL-v2-real | 复现 real 上 anchor 提升 |
| P1-c | LEPI-like no-anchor | LOL-v2-synthetic | 复现 synthetic no-anchor 较强 |
| P1-d | LEPI-like fixed e_mean=0.02 | LOL-v2-synthetic | 复现 anchor hurts |

判断：

- 如果 LOL-v2-real 上 anchor 仍提升，则保留 gauge 研究线。
- 如果 synthetic 上不再退化，说明新实现/数据处理改变了分布，需要重新审计。
- 如果两者都复现不了，先不要做新模型，检查数据、resize、评估脚本和 checkpoint 选择。

### 7.4 P2：UEFB-v2 合成 benchmark

必须做 3 类测试集：

| 测试集 | 描述 | 用途 |
|---|---|---|
| UEFB-easy | global + mild gradient | 检查曝光场基本能力 |
| UEFB-hard | vignette + spotlight + clipping + noise | 检查困难局部曝光 |
| UEFB-identity | 20%-30% near-identity | 检查防过增强 |

UEFB-v2 指标必须包括：

| 指标 | 定义 | 目的 |
|---|---|---|
| E_MAE | `mean(abs(E_pred - E_gt))` | 绝对曝光场误差 |
| E_MAE_aligned | 去均值后 MAE | 空间形状误差 |
| E_corr | 预测与 GT E 的 Pearson/Spearman | 空间方向是否一致 |
| A_AUC | `A(x)` 对应应增强区域的 AUC | 门控是否找到要增强区域 |
| Q_ECE | 恢复性图校准误差 | 不确定性是否可信 |
| identity_drop | near-identity subset 中 `PSNR(output)-PSNR(input)` | 是否过增强 |
| saturation_hallucination | clipped 区域结构伪造比例 | 是否在不可恢复区域造纹理 |

### 7.5 P3：RLEF-Former MVP 消融

核心模型矩阵：

| ID | Variant | 主干 | E | gauge | A/Q | phys branch | 目的 |
|---|---|---|---|---|---|---|---|
| M0 | Restorer only | Restormer-lite | - | - | - | - | 强主干 baseline |
| M1 | + phys aux | Restormer-lite | yes | no | no | yes | 看物理分支是否有帮助 |
| M2 | + Poisson | Restormer-lite | yes | no | no | yes | 看梯度域约束 |
| M3 | + adaptive gauge | Restormer-lite | yes | yes | no | yes | 核心 gauge claim |
| M4 | + A gate | Restormer-lite | yes | yes | A | yes | 防过增强 |
| M5 | + Q recoverability | Restormer-lite | yes | yes | A+Q | yes | failure awareness |
| M6 | + UEFB pretrain | Restormer-lite | yes | yes | A+Q | yes | 专项 exposure-field 学习 |
| M7 | full RLEF-Former | stronger backbone | yes | yes | A+Q | yes | 最终模型 |

Go/No-Go：

- `M3` 必须在 LOL-v2-real 或 UEFB 上相对 `M2` 有稳定提升，否则 gauge head 只能作为分析而非贡献。
- `M4` 必须让 UEFB-identity 的 `identity_drop` 明显改善，否则 A gate 不成立。
- `M5` 必须让 Q_ECE/AURC 改善，否则 Q 不能写成有效不确定性。
- `M7` 至少要接近 Retinexformer，同协议差距小于 `0.3-0.5 dB`，否则论文定位仍是 mechanism paper，不是 SOTA method paper。

### 7.6 P4：SOTA 同协议比较

必须统一 evaluator，不能混用论文表格数值和自己评估数值。

优先 baseline：

| 方法 | 类型 | 状态建议 |
|---|---|---|
| Retinexformer | 强 transformer baseline | 必须，official weights 同协议 |
| Zero-DCE++ | zero-reference baseline | 必须，已在过程数据中接入过 |
| KinD++ | Retinex baseline | 推荐，隔离 TF1 环境 |
| SNR-Aware LLIE | 噪声/低光相关 | 推荐 |
| Diff-Retinex | diffusion/生成路线 | 推荐，若成本太高可引用官方但需标注 |
| HVI/颜色空间方法 | 颜色增强相关 | 推荐，视实现可用性 |

主表：

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | LEE↓ | NAI↓ | Params | FLOPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LOL-v1 | Retinexformer | | | | | | | |
| LOL-v1 | RLEF-Former | | | | | | | |
| LOL-v2-real | Retinexformer | | | | | | | |
| LOL-v2-real | RLEF-Former | | | | | | | |
| LOL-v2-synthetic | Retinexformer | | | | | | | |
| LOL-v2-synthetic | RLEF-Former | | | | | | | |

UEFB 表：

| Method | PSNR↑ | SSIM↑ | E_MAE↓ | E_aligned↓ | E_corr↑ | A_AUC↑ | identity_drop↑ | Q_ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Retinexformer | | | N/A | N/A | N/A | N/A | | N/A |
| M2 + Poisson | | | | | | | | |
| M3 + gauge | | | | | | | | |
| M4 + gate | | | | | | | | |
| M7 full | | | | | | | | |

### 7.7 P5：真实无 GT 泛化

数据：LIME/NPE/MEF/DICM/VV。

指标：

| 指标 | 解释 |
|---|---|
| NIQE/BRISQUE | 无参考自然度，只作辅助 |
| exposure deviation | 与合理亮度区间的偏离 |
| saturation rate | 输出过曝比例 |
| underexposure rate | 输出仍过暗比例 |
| NAI | 暗区噪声放大 |
| color cast index | 简单颜色偏移 |

必须保存可视化：

```text
results/visuals/unpaired_grid/
  DICM_*.png
  LIME_*.png
  MEF_*.png
  NPE_*.png
  VV_*.png
```

每张图展示：

```text
input | Retinexformer | RLEF-Former | E_map | A_map | Q_map
```

### 7.8 P6：统计与复现实验

正式表格要求：

- 3 seeds：`3407, 2027, 42`。
- 报告 mean ± std。
- 对 paired benchmark 用 paired t-test 或 Wilcoxon signed-rank test 比较每图 PSNR/SSIM/LPIPS。
- 对多指标报告 effect size，不只报告 p-value。
- 对多个数据集/多个指标做 FDR 或明确标注 exploratory。

结果格式：

```text
results/tables/main_paired_mean_std.csv
results/tables/uefb_mean_std.csv
results/tables/unpaired_no_ref.csv
results/tables/significance_tests.csv
results/tables/complexity.csv
```

---

## 8. 指标定义

### 8.1 Local Exposure Error

将图像分为 `K x K` patch，例如 `16 x 16`：

```math
LEE = \frac{1}{K^2}\sum_p |\log(\mu_p(\hat{I})+\epsilon)-\log(\mu_p(I_h)+\epsilon)|
```

### 8.2 Noise Amplification Index

暗区 mask：

```math
D=1[\operatorname{mean}_c I_l^c < 0.1]
```

高频残差：

```math
HF(I)=I-GaussianBlur(I)
```

```math
NAI=\frac{std(HF(\hat{I})|D)}{std(HF(I_l)|D)+\epsilon}
```

注意排除强边缘区域，否则真实纹理会被误判成噪声。

### 8.3 identity_drop

对 near-identity subset：

```math
identity\_drop = PSNR(\hat{I},I_h)-PSNR(I_l,I_h)
```

目标：`identity_drop >= 0`。如果小于 0，说明模型把已经接近 GT 的输入增强坏了。

### 8.4 Recoverability Calibration

定义归一化误差：

```math
err(x)=clip(\|\hat{I}(x)-I_h(x)\|_1/q_{95},0,1)
```

若 `Q(x)` 表示可恢复性，则期望：

```math
1-Q(x) \approx err(x)
```

ECE：

```math
ECE_Q=\sum_b \frac{|B_b|}{N}|mean(1-Q|B_b)-mean(err|B_b)|
```

### 8.5 Saturation Hallucination Index

对 GT 或 synthetic 标记的 clipped region：

```math
M_{clip}=M_{over}\cup M_{under}
```

计算输出在不可恢复区域的高频能量与周围上下文不一致程度：

```math
SHI = mean(|HF(\hat{I})| \mid M_{clip}) / (mean(|HF(I_h)| \mid dilate(M_{clip})\setminus M_{clip})+\epsilon)
```

该指标只作为辅助，必须配合可视化。

---

## 9. 推荐配置模板

### 9.1 M0 strong restorer baseline

```yaml
experiment_name: rlef_m0_restorer_lolv2
seed: 3407
model:
  name: rlef_former
  backbone: restormer_lite
  base_channels: 48
  exposure_branch: false
  physics_branch: false
  gate_branch: false
training:
  datasets: [LOL-v2-real]
  crop_size: 256
  batch_size: 8
  epochs: 300
  optimizer: adamw
  lr: 0.0002
  weight_decay: 0.0001
  scheduler: cosine
  amp: true
loss:
  rec: 1.0
  lpips: 0.05
  poisson: 0.0
  gauge: 0.0
  id: 0.0
  gate: 0.0
  q: 0.0
```

### 9.2 M3 adaptive gauge

```yaml
experiment_name: rlef_m3_adaptive_gauge_lolv2_uefb
seed: 3407
model:
  name: rlef_former
  backbone: restormer_lite
  base_channels: 48
  exposure_branch: true
  adaptive_gauge: true
  physics_branch: true
  gate_branch: false
  q_branch: false
  e_max: 3.5
training:
  datasets: [LOL-v2-real, UEFB-v2]
  sampling_weights: [0.6, 0.4]
  crop_size: 256
  batch_size: 8
  epochs: 400
  optimizer: adamw
  lr: 0.0002
  min_lr: 0.000001
  weight_decay: 0.0001
  scheduler: cosine
  amp: true
loss:
  rec: 1.0
  lpips: 0.05
  phys: 0.15
  poisson: 0.05
  gauge: 0.10
  wtv: 0.02
  id: 0.0
  gate: 0.0
  q: 0.0
```

### 9.3 M5 full recoverability

```yaml
experiment_name: rlef_m5_recoverability_lolv2_uefb
seed: 3407
model:
  name: rlef_former
  backbone: restormer_lite
  base_channels: 48
  exposure_branch: true
  adaptive_gauge: true
  physics_branch: true
  gate_branch: true
  q_branch: true
  e_max: 3.5
training:
  datasets: [LOL-v2-real, LOL-v2-synthetic, UEFB-v2]
  sampling_weights: [0.45, 0.25, 0.30]
  crop_size: 256
  batch_size: 8
  epochs: 500
  optimizer: adamw
  lr: 0.0002
  min_lr: 0.000001
  weight_decay: 0.0001
  scheduler: cosine
  amp: true
loss:
  rec: 1.0
  lpips: 0.05
  phys: 0.15
  poisson: 0.05
  gauge: 0.10
  id: 0.02
  gate: 0.02
  q: 0.02
  wtv: 0.02
mask:
  tau_low: 0.0157
  tau_high: 0.9804
  soft_k: 40
```

---

## 10. 消融实验清单

### 10.1 必做模块消融

| ID | 对照 | 要回答的问题 | 通过标准 |
|---|---|---|---|
| A0 | M0 restorer only | 强主干本身有多强 | 作为所有提升基准 |
| A1 | M1 phys aux | Retinex/physical branch 是否有帮助 | 不明显降低 PSNR，改善 E 可视化 |
| A2 | M2 + Poisson | Poisson 是否改善 E spatial shape | E_aligned/E_corr 改善 |
| A3 | M3 + adaptive gauge | gauge 是否修复 real/synthetic tension | LOL-v2-real 提升且 synthetic 不大幅退化 |
| A4 | M4 + A gate | gate 是否防过增强 | identity_drop 改善 |
| A5 | M5 + Q | Q 是否校准 failure | Q_ECE/AURC 改善 |
| A6 | M5 no UEFB | UEFB 是否必要 | 无 UEFB 时 E 指标下降则说明 benchmark 有价值 |
| A7 | fixed e_mean vs adaptive gauge | 固定 anchor 是否脆弱 | adaptive 在 real/synthetic 均更稳 |

### 10.2 损失权重消融

| 实验 | 取值 |
|---|---|
| `lambda_poisson` | `0, 0.02, 0.05, 0.10` |
| `lambda_gauge` | `0, 0.05, 0.10, 0.20` |
| `lambda_id` | `0, 0.01, 0.02, 0.05` |
| `lambda_phys` | `0, 0.05, 0.15, 0.30` |
| UEFB sampling weight | `0, 0.2, 0.3, 0.5` |

每次只扫一个维度，其他固定为 M5 默认值。不要同时改多个参数，否则无法解释。

### 10.3 结构消融

| 实验 | 设置 |
|---|---|
| backbone | U-Net, Restormer-lite, Retinexformer-like |
| final fusion | `I_hat=I_rest`, `A*I_rest+(1-A)*I_l`, `A*I_rest+(1-A)*I_phys` |
| gauge | fixed e_mean, predicted scalar, predicted low-res map |
| E output | full-res E, low-res upsample E, multi-scale E |
| Q target | residual-based, clipping-based, combined |

---

## 11. 失败判定和 pivot 规则

### 11.1 必须停止的路线

根据过程数据，以下路线已经不值得继续作为主线：

| 路线 | 原因 |
|---|---|
| tiny LEPI 继续加宽到 c64/c96 | c64 已无收益，width-only 不是瓶颈 |
| 更深 post-hoc residual branch | blocks=1/2 下降 |
| direct RGB dual-head final output | 不超过 shallow residual，破坏 base path |
| base_rec + residual_l1 继续加权 | 压小 residual 但性能下降 |
| CRF/noise/U 作为当前主贡献 | 过程数据没有正证据 |

### 11.2 需要 pivot 的触发条件

如果出现以下情况，立即停止当前分支：

1. `M3 adaptive gauge` 相比 `M2` 在 3 seed 上无稳定提升。
2. `M4 gate` 不能改善 UEFB-identity 的 `identity_drop`。
3. `M5 Q` 的 Q_ECE 不优于简单 rule-based mask。
4. RLEF-Former 相比 Retinexformer 在 LOL-v2-real 差距大于 `1.0 dB`，且 UEFB 指标也无优势。
5. 加 UEFB 训练导致真实 unpaired 视觉明显变差。

pivot 方向：

- 若 E 指标好但 RGB 差：加强 restoration/fusion，不再调 E loss。
- 若 RGB 好但 E 差：把 E 降级为解释辅助，不能写物理准确。
- 若 real 好 synthetic 差：将 domain tension 写成分析贡献，使用 adaptive gauge/domain-conditioned anchor。
- 若 synthetic 好 real 差：检查 synthetic 退化模型是否过窄，增加真实无 GT fine-tune。

---

## 12. 论文 claim 写法

### 12.1 当前可以写的强 claim

```text
We identify and address the gauge ambiguity of Poisson-regularized local exposure fields in low-light enhancement. By normalizing the exposure field with an adaptive gauge and evaluating it on a synthetic exposure-field benchmark, the proposed framework provides interpretable exposure control beyond conventional image-level fidelity metrics.
```

中文：

```text
本文指出并处理低光增强中 Poisson 正则化局部曝光场的 gauge 歧义。通过自适应曝光场规范化和带 GT 曝光场的合成评估协议，模型不仅输出增强图像，也能解释增强发生的位置和强度。
```

### 12.2 可以写的中等 claim

```text
Recoverability-aware gating reduces over-enhancement on near-identity and unrecoverable regions, revealing a practical way to connect physical exposure modeling with robust enhancement behavior.
```

### 12.3 不能写的 claim

除非后续实验真实支持，否则不要写：

- `Our method achieves state-of-the-art performance on all low-light benchmarks.`
- `The predicted exposure field is physically accurate in real images.`
- `CRF and heteroscedastic noise modeling are proven effective.`
- `The method solves RAW low-light enhancement.`
- `Uncertainty reliably detects unrecoverable content.`

### 12.4 推荐论文标题方向

| 风格 | 标题 |
|---|---|
| 顶会方法 | Gauge-Aware Local Exposure Fields for Robust Low-Light Image Enhancement |
| 偏机制分析 | Revisiting Local Exposure Fields: Gauge Ambiguity and Recoverability in Low-Light Enhancement |
| 偏 benchmark | UEFB: A Synthetic Benchmark for Evaluating Local Exposure Fields in Uneven-Exposure Enhancement |
| 强主干融合 | RLEF-Former: Recoverability-Aware Exposure Field Regularization for Low-Light Enhancement |

---

## 13. 每次实验必须记录的信息

每个 run 保存一个 `run_meta.yml`：

```yaml
run_id: rlef_m5_lolv2_uefb_seed3407_YYYYMMDD_HHMM
objective: "test adaptive gauge + recoverability gate"
command: >
  python scripts/train.py --config configs/rlef_m5_recoverability_lolv2_uefb.yml --seed 3407
working_directory: G:\9chapterwork\workspace\a1_rlef_project
environment:
  os:
  python:
  torch:
  cuda:
  cudnn:
  gpu:
  driver:
  conda_env:
data:
  train_sets:
  val_sets:
  data_root:
  split_files:
model:
  name:
  backbone:
  params:
  flops_256:
seed: 3407
checkpoints:
  best_psnr:
  best_uefb:
  last:
metrics:
  best_epoch:
  lolv2_real_psnr:
  lolv2_real_ssim:
  uefb_e_mae:
  uefb_e_corr:
  identity_drop:
  q_ece:
anomalies:
  - none
notes:
  - no GT mean trick
  - same evaluator for all methods
```

训练日志必须包含：

```text
loss_rec, loss_phys, loss_poisson, loss_gauge, loss_id, loss_gate, loss_q
val_psnr, val_ssim, val_lpips, val_e_mae, val_e_corr, val_identity_drop
learning_rate, gpu_mem, time_per_iter
```

---

## 14. 最小执行顺序

按下面顺序执行，不要跳步：

1. 创建 `G:\9chapterwork\workspace\a1_rlef_project`。
2. 迁移或软链接 LOL-v1、LOL-v2、unpaired_real 数据。
3. 实现 `make_uefb_v2.py` 并生成 smoke 版 UEFB。
4. 实现 metrics：PSNR/SSIM/LPIPS/LEE/NAI/E_MAE/E_corr/identity_drop/Q_ECE。
5. 跑 P0 smoke，确保所有脚本和测试通过。
6. 跑 P1 复现旧过程数据关键现象。
7. 生成正式 UEFB-v2。
8. 实现 RLEF-Former M0-M5。
9. 先单 seed 跑 M0-M5，淘汰无效模块。
10. 对保留模型跑 3 seeds。
11. 接入 Retinexformer/Zero-DCE++/KinD++ 同协议评估。
12. 跑 unpaired real 可视化和 no-reference 指标。
13. 生成主表、UEFB 表、消融表、复杂度表。
14. 根据真实结果决定论文 claim 强度。

---

## 15. 最终验收门槛

### 15.1 机制论文最低门槛

满足以下全部：

- adaptive gauge 相比 fixed/no-anchor 在 real/synthetic 之间更稳。
- UEFB 上 E_MAE/E_corr 明显优于无 Poisson/无 gauge 版本。
- A gate 改善 identity_drop。
- 至少在 LOL-v2-real 上接近 Retinexformer，或在 UEFB 解释指标上明显优于 Retinexformer 类黑箱方法。
- 所有主实验 3 seed mean±std。

### 15.2 顶会方法论文门槛

满足以下大部分：

- LOL-v2-real 与 Retinexformer 差距小于 `0.3 dB` 或超过。
- LOL-v2-synthetic 不再灾难性落后，至少超过 Zero-DCE++ 且接近强 baseline。
- UEFB-v2 上有明确领先，且可视化直观。
- unpaired real 中过增强和噪声放大低于强 baseline。
- 有清晰失败案例，不回避 synthetic/real tension。

### 15.3 顶刊扩展门槛

在顶会方法基础上增加：

- RAW/SID 噪声与相机响应实验。
- 多相机/多 ISP 验证。
- 更完整的可恢复性理论和校准分析。
- 主观评价或下游任务评价。
- 数据集/代码/模型公开协议。

---

## 16. 一句话路线

不要继续把 A1 做成“大而全物理头堆叠”。下一轮应把主线收敛为：**强 restoration backbone + gauge-normalized local exposure field + recoverability-aware gate + UEFB-v2 专项评价**。先证明曝光场规范化和恢复性控制能在真实 paired、synthetic exposure-field 和 unpaired real 三个场景中同时带来稳定收益，再决定是否恢复 CRF、noise 和 RAW 扩展。
