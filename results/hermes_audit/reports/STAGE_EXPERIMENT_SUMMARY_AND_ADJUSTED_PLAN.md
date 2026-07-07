# 阶段实验总结合后续实验调整计划

生成时间：2026-07-04T14:32:22.635666+00:00  
项目根目录：`/home/user/a1_rlef_project`

## 0. 当前总判断

```text
当前不应继续 DGB / scalar knob 路线。
主线应从“追求恢复 SOTA”调整为：
interpretable exposure-field auxiliary calibration / physically-inspired diagnostic module。
```

当前保守默认模型：

```text
P3c M4J_ES e_shape=0.05
UEFB PSNR = 17.915±0.275
Real PSNR = 20.021±0.223
Synthetic PSNR = 17.678±0.735
```

保留为参考而非默认：

- `P7B_DHEAD_RA010`：P7-family near-miss。
- `P2B_DGB_GAUGE_ONLY`：DGB real-domain 诊断参考。
- `P2E_ESHAPE010_GATEPRI002`：DGB synthetic-domain 诊断参考。
- Retinexformer official：强 restoration baseline，用于界定性能上限/差距。

---

## 1. 阶段实验总结

| 阶段 | 已执行内容 | 关键结果 | 结论 / 对后续的影响 |
|---|---|---|---|
| P0/P1 smoke + P1 formal | Scaffold、UEFB-smoke、LOL-v2 real/synthetic gauge replication | P1 formal adaptive: real 19.050 / synthetic 20.269；fixed0p02 synthetic 19.906 | 固定 gauge 不是主贡献；adaptive gauge 可保留为基础机制。 |
| P2/P3 UEFB formal | UEFB-v2 3000/500，M0-M5 机制消融 | M4 gate UEFB 18.653 最强；M5 Q_ECE 更好但 PSNR 代价；E_corr 仍负 | A-gate 是有效机制；Q branch 只能作为质量校准，不应作为 PSNR 主路径。 |
| P3b joint + E-shape | UEFB+LOL-v2 joint training，加入 low-pass centered E-shape | M4J_ES: real 20.277、synthetic 17.141、E_corr 全转正/大幅提高 | E-shape 是核心机制；把故事从“绝对物理 E”调整为“gauge-invariant spatial exposure structure”。 |
| P3c/P3d 多种子 | e_shape=0.05 与 0.10 3-seed | e=0.05: UEFB 17.915±0.275, real 20.021±0.223, synthetic 17.678±0.735; e=0.10 UEFB 更高 18.111 但 real/syn 更低 | P3c e=0.05 成为 conservative default；e=0.10 只保留为 robustness/ablation。 |
| P4 official baselines | Retinexformer / Zero-DCE++ / KinD++ official-code eval | Retinexformer real 22.794, synthetic 25.669; RLEF P3c real 20.021, synthetic 17.678 | 不能走 SOTA/restoration 主线；可主张 interpretable exposure-field auxiliary calibration。 |
| P5/P5b distillation | Retinexformer output-level + domain-weighted distillation | P5_T03 real 20.416 但 synthetic 16.996；P5b real 19.366, synthetic 17.473 | 输出级 distillation 是 trade-off，不继续 scalar distill sweep。 |
| P6/P6b/P6c structural/scalar controls | multiscale backbone、synthetic rec protection、gate protection | P6 real 20.197/UEFB 18.015 但 synthetic 17.598 低；P6b synthetic 最高 19.115 但 UEFB/real 崩；P6c real 最高 20.456 但其他不稳 | 结构增强有方向性，但 scalar protection 不能解决三域 trade-off。 |
| P7/P7b/P7c domain-head | domain adapters、real anchor、stronger anchor sweep | P7_MS_DHEAD UEFB/syn 过 P3c 但 real 19.209；P7B_RA010 near-miss: UEFB 18.085, real 19.847, syn 17.965; P7c 更强 anchor 退化 | P7B_RA010 作为 near-miss reference；停止更大 anchor。 |
| DGB branch | Phase2/P2B/P2C/P2D/P2E 全 ladder | best real P2B Gauge-only 20.432; best synthetic P2E gp002 18.683; joint_gate_any=False | DGB 正式停止并封存；只保留诊断结论。 |

---

## 2. 对整体工作的影响

| 影响维度 | 调整前可能想法 | 实验后结论 | 对整体工作的影响 |
|---|---|---|---|
| 论文主张 | 做一个可与 Retinexformer 竞争的 LLIE/restoration 方法 | P4 显示 Retinexformer real/synthetic 大幅领先 | 主张改为：可解释 exposure-field auxiliary / calibration，不做 SOTA restoration claim。 |
| 物理解释 | 绝对 exposure field correctness | Gauge ambiguity + 早期 E_corr 失败，e_shape 后才稳定正相关 | 只主张 gauge-invariant spatial structure；必须报告 absolute/mean-aligned MAE/E_corr。 |
| 方法核心 | 多头越多越好：phys/gauge/A/Q/DGB | A-gate 与 e_shape 是核心；Q、DGB、distill、scalar controls 多为诊断/辅助 | 收缩贡献：M4J_ES + e_shape + interpretability diagnostics。 |
| 实验策略 | 继续调 scalar knob | 多轮 P5-P7/DGB 表明 scalar sweep 只是转移 trade-off | 后续禁止低价值 scalar sweep，改为 high-leverage 新结构或论文固化。 |
| 默认 checkpoint | 继续追 near-miss | P3c e=0.05 唯一有 3-seed 稳定性和清晰解释 | P3c e=0.05 作为保守默认；P7B/P2B/P2E 只作参考。 |

### 2.1 研究问题被收缩，但更清晰

这些实验把问题从“做一个低光增强 SOTA”收缩为：

```text
如何在不牺牲太多恢复性能的前提下，
让模型学习一个稳定、可诊断、gauge-invariant 的 exposure-field structure，
并用它解释/校准恢复过程。
```

这比原来的泛化 restoration claim 更可防守，因为已有证据支持：

- adaptive gauge 能降低 absolute E_MAE；
- e_shape consistency 让 E_corr 由负/弱变为稳定正相关；
- P3c 3-seed 有稳定性；
- P4 已清楚标定与 Retinexformer 的差距，避免过 claim。

### 2.2 负结果改变了后续实验价值排序

P5-P7 和 DGB 的价值主要不是“找到最终模型”，而是证明：

```text
继续调权重 / 调 gate / 调 route / 调 anchor 的边际价值很低。
```

因此后续实验必须从“低成本小 knob”切换为：

1. 论文证据固化与可视化；或
2. 另开非 DGB 的 high-leverage 结构路线。

---

## 3. 立即停止的实验方向

| 停止项 | 原因 |
|---|---|
| DGB / P2F continuation | 已封存；路由统计可变但 real quality 不恢复，joint gate 失败。 |
| 更大 real anchor / domain_head_anchor sweep | P7c 比 P7B_RA010 更差，说明强度不是瓶颈。 |
| output-level Retinexformer scalar distillation | P5/P5b 产生 real/synthetic trade-off，且偏离可解释主线。 |
| rec_by_dataset / synthetic scalar protection | P6b 明确 over-correction，破坏 UEFB/real。 |
| gate_identity / route_floor / structure_preserve scalar sweeps | P6c/P2C/P2D/P2E 均显示只改变 trade-off。 |

---

## 4. 后续阶段实时调整后的安排

| 优先级 | 后续阶段 | 目的 | 执行安排 | Go/No-Go Gate |
|---|---|---|---|---|
| P0 | Evidence freeze + paper framing | 把已完成结果转成稳定论文证据 | 0.5-1 天；整理总表、claim ledger、method story、negative-result appendix | 不需要训练；所有 claim 必须能追到表/报告。 |
| P1 | Qualitative + diagnostic visualization | 支撑“可解释辅助校准”而不是 SOTA | 1-2 天；P3c default vs input/Zero-DCE++/Retinexformer；展示 E/A/Q maps、失败样例、domain trade-off | 图像必须显示 RLEF 的解释价值；若只显示画质落后，则转为 appendix。 |
| P2 | Official/no-reference supplementary eval | 补齐 unpaired real / perceptual quality 证据 | 1 天；只对 frozen checkpoints 做 NIQE/BRISQUE/LOE/visual audit，不新训练 | 若指标不支持，不硬写主 claim；只保留为 limitation。 |
| P3 | Manuscript-grade ablation table | 固化主线贡献 | 1 天；核心表仅保留 M0/M4/M4J/M4J_ES/P3c/P4 baseline；DGB/P5-P7 放 appendix | 主表必须避免“实验太散”。 |
| P4 optional new route: RLEF-as-aux on strong backbone | 如果还要追性能，需要脱离 DGB/scalar knobs | 先做 200-step smoke，再 1000-step single-seed；可用 stronger restoration backbone + E/A/Q auxiliary heads | 只有当 real/synthetic PSNR 接近强 backbone baseline（drop ≤0.2-0.3 dB）且 E_corr 保持正，才进入 3-seed。 |
| P5 optional new route: Retinex-factorization redesign | 如果论文更偏方法创新而非性能 | 先做最小 factorization prototype；严格比较 P3c，不碰 DGB | 1000-step single-seed 必须同时超过 P3c real/synthetic 或显著提升 E diagnostics 且 PSNR 不降。 |
| P6 final paper route | 论文写作与答辩材料 | 在 P0-P3 完成后推进；不等待 optional route | 若 optional route 未过 gate，论文以 P3c/P4/P7/DGB diagnostics 为完整故事。 |

---

## 5. 推荐执行顺序

### 先做：论文证据固化，而不是继续训练

```text
S1. 固化主线故事：P0/P1 → P3 → P3c → P4。
S2. 制作主表：M0/M4/M4J/M4J_ES/P3c/Retinexformer/Zero-DCE++。
S3. 制作可视化：input / RLEF / Retinexformer / E-map / A-map / Q-map / failure cases。
S4. 把 P5-P7/DGB 全部放 appendix/negative-results/diagnostic findings。
```

### 再决定是否开新路线

只有当论文主线证据固化后，才建议开新实验路线。新路线不能叫 P2F/DGB continuation，而应是新分支：

```text
RLEF-as-auxiliary-on-strong-backbone
或
Retinex-factorization redesign
```

新路线准入 gate：

```text
单 seed 1000-step 必须同时满足：
1. real/synthetic PSNR 不明显低于对应 strong baseline 或 P3c default；
2. E_corr 保持正且有解释价值；
3. 没有引入 teacher-only / dataset-label-only 的不可部署机制；
4. 不再依赖 scalar sweep 作为主要创新。
```

---

## 6. 论文/汇报 framing 调整

### 应该主张

```text
RLEF provides a physically-inspired, gauge-aware exposure-field auxiliary mechanism
that improves interpretability and stabilizes exposure-structure diagnostics under joint training.
```

中文表述：

```text
RLEF 不是为了直接击败 Retinexformer，
而是提供一种可解释的 exposure-field 辅助校准机制，
在 joint training 中通过 gauge-invariant E-shape 约束获得稳定的空间曝光结构诊断。
```

### 不应该主张

```text
- SOTA / Retinexformer-level LLIE performance
- absolute physical exposure-field correctness
- DGB 是最终有效模块
- teacher distillation 带来稳健提升
- scalar domain/gate/route tuning 解决三域 trade-off
```

---

## 7. 当前 artifact 索引

- Claim ledger: `docs/CLAIM_LEDGER.md`
- DGB stop report: `results/hermes_audit/reports/DGB_BRANCH_STOP_CONSOLIDATION_REPORT.md`
- 本报告: `results/hermes_audit/reports/STAGE_EXPERIMENT_SUMMARY_AND_ADJUSTED_PLAN.md`
- JSON plan: `results/tables/adjusted_stage_plan_20260704.json`
- Core default aggregate: `results/tables/p3c_multiseed_sweep_aggregate.json`
- Official baselines: `results/tables/p4_official_baselines_summary.json`
