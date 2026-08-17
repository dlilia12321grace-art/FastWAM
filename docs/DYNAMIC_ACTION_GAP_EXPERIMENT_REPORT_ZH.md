# Dynamic ActionGap：基于中间隐藏状态的动态 Internal/Full 路由

> 阶段性研究报告与论文实验路线图
> 更新日期：2026-08-15
> 当前状态：核心方案已实现，四个 LIBERO Suite 的初步跨任务验证已完成；关键消融与公平基线仍在补充。

## 1. 报告目的

老师提出的核心方向是：不必让每个 action denoising step 都运行到 Action DiT 的最后一层，可以利用中间层 hidden states 预测当前步骤的重要性，并动态决定该步继续走完整分支（full）还是提前走浅层分支（internal）。

本报告用于持续回答以下问题：

1. 不同去噪步骤的重要性是否确实不同？
2. 中间 hidden states 能否预测 internal 相对 full 的动作误差风险？
3. 动态路由能否在基本不降低闭环成功率的前提下提高速度？
4. 收益究竟来自状态相关的 hidden features，还是 MLP 只学到了一张固定 timestep 调度表？
5. 动态路由是否引入动作抖动、重规划跳变或夹爪异常？

因此，本文档既记录已有结果，也作为后续实验、可视化和论文写作的统一框架。

## 2. 当前核心结论

截至目前，能够较稳妥地表述的结论是：

- Dynamic ActionGap 已形成从数据采集、MLP 风险预测、在线路由、安全约束到闭环评测的完整流程。
- 在 `libero_goal`、`libero_spatial`、`libero_object` 和 `libero_10` 共 380 个 fixed/dynamic 配对 episode 上，两种方法均成功 `366/380`。
- 相对 fixed gap=4，dynamic 的加权平均 Action denoise 加速约 `1.043x`，`infer_action` 加速约 `1.029x`，internal ratio 从固定 `60.0%` 提高到约 `64.9%`。
- 相对不使用 internal 分支的 full baseline，当前 dynamic 在已统计实验中的 Action denoise 约为 `2.03x`，完整 `infer_action` 约为 `1.70x`。
- 路由比例会随 Suite 变化：Goal/LIBERO-10 更积极，Object/Spatial 更保守。这与“简单状态多复用、风险状态多纠偏”的设计目标一致。
- Meta-only 在 Goal/Spatial task1–9 上取得 `173/180`，不低于 hidden+meta 的 `172/180`；其 internal ratio 在所有任务上均固定为 `70.0%`。当前结果说明 timestep/meta 已经能够形成很强的调度，尚无证据证明 hidden states 带来额外闭环收益。
- 在相同训练任务下，meta-only 的验证 MAE 为 `0.10240`，约为 hidden+meta（`0.01465`）的 7 倍；Smooth L1 也从 `0.0002024` 增至 `0.008336`。这提供了明确的离线证据：hidden state 对预测真实 internal/full action gap 很重要，但该预测优势尚未转化为当前单点路由配置下的闭环优势。
- task9 Goal 的小样本抖动分析中，dynamic 的 jerk 和 replan jump 高于 fixed，且成功率为 `8/10`。这是需要进一步分析的风险信号，暂不能下普遍性结论。

## 3. 方法

### 3.1 问题定义

原始 fixed gap=4 对所有任务和状态使用固定 full/internal 调度，internal ratio 恒为 `60%`。这种调度无法区分：

- 当前视觉与动作状态是否简单；
- 当前 diffusion step 是否敏感；
- internal 输出是否已经足够接近 full 输出；
- 当前是否处于接触、抓取、放置等高风险阶段。

Dynamic ActionGap 将每个 action denoising step 看作一次路由决策：

- 预测风险较低：使用 internal 分支；
- 预测风险较高：运行完整 Action DiT；
- 安全规则强制部分步骤运行 full，防止长时间累积误差。

### 3.2 MLP 输入与训练目标

当前 hidden+meta gate 的输入包括：

- Action DiT 共享浅层 hidden state 的池化特征；
- diffusion timestep；
- 当前 step index；
- 距离上一次 full step 的步数等路由元信息。

监督目标为：

```text
log1p(normalized internal/full action gap)
```

该 gap 衡量同一步骤中 internal action 与 full action 的归一化差异。预测值不高于阈值时走 internal，否则走 full。

hidden+meta gate 当前验证集结果：

| 指标 | 数值 |
|---|---:|
| Smooth L1 | 0.0002024 |
| MAE | 0.01465 |
| 初始验证阈值 | 0.24291 |

### 3.3 最终候选配置

当前主候选 `v4_t036`：

```text
threshold = 0.36
max_internal_run = 7
anchor_action_gap = 8
first_step = full
last_step = full
```

其中 first/last step、周期 anchor 和最大连续 internal 次数是安全约束。阈值和约束在 task0 smoke 阶段选定，随后冻结用于 held-out task。

### 3.4 对照方法

| 方法 | 含义 | 当前状态 |
|---|---|---|
| full | 每一步均运行完整 Action DiT | 已测主要速度基线 |
| fixed_gap4 | 固定 ActionGap=4，internal ratio 为 60% | 已完成 |
| v4_t036 | hidden+meta MLP 动态路由 | 已完成主要跨任务评测 |
| meta_only_t036 | 不读取 hidden state，只使用 timestep/meta | smoke 已完成，跨任务评测中 |
| hidden_only | 只读取 hidden state，不使用 timestep/meta | 待补充 |
| compute-matched fixed | 与 dynamic 使用相近 internal ratio 的固定调度 | 待补充 |
| compute-matched random | 与 dynamic 使用相近 internal ratio 的随机路由 | 待补充 |

## 4. 实验协议

- Benchmark：LIBERO Goal、Spatial、Object、LIBERO-10。
- 主模型与 internal LoRA checkpoint 在对比中保持不变。
- fixed 与 dynamic 使用相同 task、initial states 和 episode 数。
- task0 用于 smoke 与候选选择；其余 task 用于冻结参数后的跨任务验证。
- 主要指标：闭环成功率、Action denoise 时间、完整 `infer_action` 时间、internal ratio。
- 安全指标：action delta、delta p95、jerk、replan jump、gripper flips。
- 路由指标：predicted gap、逐 timestep internal rate、route reason、不同 task/Suite 的 internal ratio。

注意：`infer_action` 包括 video prefill 与 Action DiT 等动作推理过程，但不等于完整机器人系统从传感器到执行器的端到端墙钟时间。

## 5. 已有实验结果

### 5.1 task0：40-episode 扩展验证

| Suite | 方法 | 成功率 | Denoise ms | Infer ms | Internal ratio |
|---|---|---:|---:|---:|---:|
| libero_goal | fixed_gap4 | 20/20 | 147.19 | 214.74 | 60.0% |
| libero_goal | v4_t036 | 20/20 | 130.36 | 197.68 | 70.0% |
| libero_spatial | fixed_gap4 | 19/20 | 148.87 | 217.74 | 60.0% |
| libero_spatial | v4_t036 | 19/20 | 151.01 | 219.58 | 61.6% |

汇总：

| 指标 | fixed_gap4 | v4_t036 | 变化 |
|---|---:|---:|---:|
| Success | 39/40 | 39/40 | 相同 |
| Denoise ms | 148.03 | 140.69 | `1.052x` |
| Infer ms | 216.24 | 208.63 | `1.036x` |
| Internal ratio | 60.0% | 65.8% | +5.8 个百分点 |

这一结果验证了方案可运行，但 task0 参与过阈值选择，不能作为最终泛化证据。

### 5.2 Goal + Spatial：held-out task1–9

每个 task、Suite、方法运行 10 episodes，共 180 个 episode/方法。

| 方法 | 成功率 | Denoise ms | Infer ms | Internal ratio |
|---|---:|---:|---:|---:|
| full | 174/180 | 294.03 | 364.15 | n/a |
| fixed_gap4 | 174/180 | 151.76 | 222.29 | 60.0% |
| v4_t036 | 172/180 | 144.57 | 214.92 | 64.9% |

速度比较：

| 比较 | Denoise speedup | Infer speedup |
|---|---:|---:|
| fixed vs full | `1.938x` | `1.638x` |
| dynamic vs full | `2.034x` | `1.694x` |
| dynamic vs fixed | `1.050x` | `1.034x` |

解释：dynamic 相对 full 的主要收益来自已有 internal 分支，而 MLP 动态路由在 fixed gap=4 之上进一步贡献约 3%–5% 的增量加速。这里必须同时报告两组数字，不能把 `1.88x/1.69x` 与 `1.03x` 当作矛盾结果。

### 5.3 Object + LIBERO-10：task0–9

每个 task、Suite、方法运行 10 episodes，共 200 个 episode/方法。

| 方法 | 成功率 | Denoise ms | Infer ms | Internal ratio |
|---|---:|---:|---:|---:|
| fixed_gap4 | 192/200 | 149.51 | 217.49 | 60.0% |
| v4_t036 | 194/200 | 144.25 | 212.36 | 64.86% |

dynamic 相对 fixed：

- 成功率变化：`+1.0` 个百分点；
- Action denoise：`1.036x`；
- `infer_action`：`1.024x`；
- internal ratio：增加约 `4.86` 个百分点。

该组 full baseline 的速度比较为：

- fixed vs full：denoise `1.955x`，infer `1.658x`；
- dynamic vs full：denoise `2.026x`，infer `1.698x`。

【待补充】从 full-cross 汇总文件补入 full 的精确成功数和绝对耗时，形成完整三方法表格。

### 5.4 四个 Suite 的 fixed/dynamic 总汇总

合并 Goal、Spatial、Object 和 LIBERO-10：

| 指标 | fixed_gap4 | v4_t036 | 变化 |
|---|---:|---:|---:|
| Success | 366/380 | 366/380 | 相同（96.3%） |
| Denoise ms（加权） | 150.58 | 144.40 | `1.043x` |
| Infer ms（加权） | 219.77 | 213.57 | `1.029x` |
| Internal ratio | 60.0% | 约 64.9% | 约 +4.9 个百分点 |

该结果支持“动态路由在目前样本上基本保持总体成功率，并带来稳定的小幅增量加速”。但它还不是严格的成功率非劣性证明。

### 5.5 配对成功结果

Goal + Spatial 的主要差异：

- fixed-only：3 个 episode；
- dynamic-only：1 个 episode；
- dynamic 净减少 2 个成功 episode。

Object + LIBERO-10 的主要差异：

- fixed-only：2 个 episode；
- dynamic-only：4 个 episode；
- dynamic 净增加 2 个成功 episode。

两组抵消后，四个 Suite 的总成功数相同。现象更像少量闭环随机差异，而不是明确的单向退化或提升；最终需要置信区间、McNemar 检验及更大样本支持。

## 6. 动态路由可视化与指标

### 6.1 已生成可视化

- `importance_by_step.png`：不同 diffusion step 的 predicted gap/importance。
- `internal_rate_by_step.png`：不同 step 的 internal 使用率。
- `route_reason_distribution.png`：predicted safe/risky、anchor、first/last step 等路由原因。
- `task_internal_ratio_heatmap.png`：不同 Suite/task 的 internal ratio。
- `failure_heatmaps/`：失败 episode 的逐步预测与路由热图。
- `dynamic_route_steps.csv`：可进一步分析的逐步路由明细。

已有图片：

![不同去噪步的重要性](../dynamic_action_gap_story_visuals/importance_by_step.png)

![不同去噪步的 internal 使用率](../dynamic_action_gap_story_visuals/internal_rate_by_step.png)

![路由原因分布](../dynamic_action_gap_story_visuals/route_reason_distribution.png)

![任务级 internal ratio](../dynamic_action_gap_story_visuals/task_internal_ratio_heatmap.png)

### 6.2 Suite 级路由统计

| Suite | 路由步数 | Predicted gap mean | Internal ratio | Predicted risky 数 |
|---|---:|---:|---:|---:|
| libero_10 | 26,650 | 0.2437 | 66.34% | 976 |
| libero_goal | 11,610 | 0.2434 | 66.30% | 430 |
| libero_object | 13,280 | 0.2807 | 61.91% | 1,075 |
| libero_spatial | 10,210 | 0.2659 | 63.34% | 680 |

初步观察：Object 与 Spatial 的平均预测风险更高、internal ratio 更低；Goal 与 LIBERO-10 更积极地使用 internal。这说明最终执行结果并非所有 Suite 共用完全相同的固定比例。

但该现象仍不能单独证明 hidden state 有效，因为 task 长度、timestep 分布和安全约束也可能造成比例差异，必须结合 meta-only 和 compute-matched 消融。

### 6.3 仍需增强的可视化

【待补充】

1. 同一 episode 中，视频帧、predicted importance、真实 action gap 和 full/internal route 的时间轴对齐图。
2. 成功与失败 episode 的 importance/route 分布对比。
3. hidden+meta 与 meta-only 在同一 episode 上的路由差异图。
4. predicted gap 与真实 gap 的散点图、相关系数、校准曲线和 AUC/PR-AUC。
5. 按任务阶段划分的路由比例：接近目标、接触、抓取、搬运、放置。
6. speed–success Pareto 曲线：横轴为耗时/internal ratio，纵轴为闭环成功率。

## 7. 动作抖动与失败分析

目前在 `libero_goal task9` 上进行了 10 episodes/方法的小样本评测：

| 方法 | 成功率 | Delta mean | Delta p95 | Jerk mean | Replan jump | Gripper flips |
|---|---:|---:|---:|---:|---:|---:|
| full | 9/10 | 0.05190 | 0.12596 | 0.04022 | 0.10547 | 1.50 |
| fixed_gap4 | 10/10 | 0.05057 | 0.12126 | 0.03983 | 0.08668 | 2.00 |
| v4_t036 | 8/10 | 0.05231 | 0.12899 | 0.04560 | 0.11176 | 3.30 |

相对 fixed，dynamic 在该小样本中的 jerk 约高 `14.5%`，replan jump 约高 `28.9%`，gripper flips 也更多。这提示路由切换可能引入动作不连续，但存在以下混杂因素：

- dynamic 有两个失败 episode，运行时长可能不同；
- 仅测试了一个 task 和 10 个 episode；
- gripper flips 需要按 episode 长度或有效动作步数归一化；
- 需要将抖动峰值与 route 切换时刻、真实接触阶段和失败视频对齐。

因此当前只能把它视为风险信号，而不能下结论说动态方案普遍导致抖动。

【待补充】扩大到代表性的 Goal/Spatial/Object/LIBERO-10 任务；分别统计 full→internal、internal→full 切换附近的 jerk；检查失败视频；之后再决定是否加入 hysteresis、最短驻留时间或预测值平滑。

## 8. Meta-only 消融：当前进展

为了验证 MLP 是否真正使用 hidden state，已实现三种输入模式：

- `hidden_meta`：hidden state + meta，当前主方法；
- `meta_only`：仅 timestep/step index/路由元信息；
- `hidden_only`：仅 hidden state。

### 8.1 task0 smoke

| Suite | 方法 | 成功率 | Denoise ms | Infer ms | Internal ratio |
|---|---|---:|---:|---:|---:|
| libero_goal | v4_t036 | 5/5 | 130.88 | 204.06 | 69.8% |
| libero_goal | meta_only_calibrated（0.24649） | 5/5 | 152.92 | 225.68 | 60.0% |
| libero_goal | meta_only_t036 | 5/5 | 134.24 | 209.11 | 70.0% |
| libero_spatial | v4_t036 | 5/5 | 151.67 | 230.16 | 61.8% |
| libero_spatial | meta_only_calibrated（0.24649） | 5/5 | 157.08 | 239.18 | 60.0% |
| libero_spatial | meta_only_t036 | 5/5 | 133.34 | 211.59 | 70.0% |

解释：meta-only 使用训练得到的默认阈值 `0.24649` 时，两个 Suite 均恰好为 `60.0%` internal，速度与 fixed gap=4 基本相当且略慢；阈值提高到 `0.30` 或 `0.36` 时，两个 Suite 又同时达到结构上限 `70.0%`。这进一步说明 meta-only 的行为主要是一张与任务状态无关且高度离散的 timestep/meta 调度表。它在高阈值下更快，主要因为跳过比例更高，不能据此认为 meta-only 更优。由于每个 action chunk 只有有限个 denoising steps，单一阈值未必能产生约 `65%` 的中间预算；公平对照需要跨 chunk 交替 60%/70% 调度，或构造显式 compute-matched fixed/random baseline。

### 8.2 Goal + Spatial task1–9

| 方法 | 成功率 | Denoise ms | Infer ms | Internal ratio |
|---|---:|---:|---:|---:|
| full | 174/180 | 294.03 | 364.15 | n/a |
| fixed_gap4 | 174/180 | 151.76 | 222.29 | 60.0% |
| hidden+meta（v4_t036） | 172/180 | 144.57 | 214.92 | 64.9% |
| meta_only_t036 | 173/180 | 131.66 | 202.13 | 70.0% |

meta-only 相对 fixed 的 Action denoise 加速为 `1.153x`，`infer_action` 加速为 `1.100x`，成功数仅少 1 个；相对 full 分别为 `2.233x` 和 `1.802x`。配对结果中，meta-only 相对 fixed 有 2 个独有成功、3 个独有失败。

该结果对当前故事提出了直接挑战：

- meta-only 成功数比 hidden+meta 多 1 个，并没有显示 hidden features 的闭环优势；
- meta-only 在所有 task/Suite 上均为 `70.0%` internal，基本可视为 MLP 学到了一张 timestep/meta 调度表；
- meta-only 比 hidden+meta 更快，主要因为两者计算预算不同（`70.0%` 对 `64.9%`），不能直接据此判断哪种路由更优；
- hidden+meta 的主要潜在价值变成了“会根据任务风险主动降低 internal ratio”，但必须证明这种保守行为在相同计算量或速度—成功率曲线上能够换来更高可靠性。

因此，下一步的首要任务不是继续单点比较，而是让 hidden+meta、meta-only、fixed/random 在相同 internal ratio 下比较，或分别扫描阈值形成 speed–success Pareto curve。

### 8.3 离线预测能力对比

两种 gate 在相同类型的训练/验证流程下得到：

| Gate 输入 | Default threshold | Validation Smooth L1 | Validation MAE |
|---|---:|---:|---:|
| hidden+meta | 0.24291 | 0.0002024 | 0.01465 |
| meta-only | 0.24649 | 0.008336 | 0.10240 |

meta-only 的 MAE 约为 hidden+meta 的 `7.0x`，Smooth L1 约为 `41x`。这说明 hidden state 明显提升了对真实 internal/full action gap 的回归能力，MLP 并非完全只依赖 timestep/meta。

这一结果与闭环结果并不矛盾：当前 LIBERO 任务可能允许一张较激进的固定时间表仍保持较高成功率，而 hidden+meta 对风险的更准确预测在 `threshold=0.36`、安全约束和有限 episode 下没有转化为更高成功数。论文需要分别报告“预测是否准确”和“准确预测是否改善速度—成功率前沿”。

【待补充】在相同验证样本上增加 Pearson/Spearman 相关系数、风险分类 ROC-AUC/PR-AUC、校准曲线，并绘制两种 gate 的 prediction-vs-target 散点图。

### 8.4 判定标准

meta-only 跨任务评测完成后分三种情况解释：

1. **hidden+meta 在相同 internal ratio 下成功率更高**：最支持老师提出的 hidden-state importance 故事。
2. **两者成功率相近，但 hidden+meta 会随任务调整 internal ratio**：hidden state 主要提供风险自适应和安全性，仍有可讲价值。
3. **meta-only 在相同成功率/算力下不弱于 hidden+meta**：当前证据不足以证明 hidden state 有用，应把主线改为可学习动态 schedule，或重新设计标签、特征和训练方法。

## 8.5 Compute-matched baseline：task0 smoke

为排除“dynamic 只是使用了更多 internal step”的混杂因素，实现了两个约 65% internal 的对照：

- `fixed65`：相邻 action chunk 交替使用 6/10 和 7/10 个 internal step；
- `random65`：保持相同 full/internal 数量与安全约束，但随机选择额外 full step 的位置。

| Suite | 方法 | 成功率 | Denoise ms | Infer ms | Internal ratio |
|---|---|---:|---:|---:|---:|
| libero_goal | v4_t036 | 5/5 | 130.88 | 204.06 | 69.8% |
| libero_goal | fixed65 | 5/5 | 138.85 | 211.62 | 64.8% |
| libero_goal | random65 | 5/5 | 143.72 | 219.63 | 64.8% |
| libero_spatial | v4_t036 | 5/5 | 151.67 | 230.16 | 61.8% |
| libero_spatial | fixed65 | 5/5 | 138.61 | 217.34 | 64.9% |
| libero_spatial | random65 | 5/5 | 139.28 | 226.48 | 64.9% |

两个 baseline 均正确达到目标计算预算，且所有 smoke episode 成功。按两个 Suite 简单平均，v4 的 internal ratio 约为 `65.8%`，与 compute-matched baseline 接近；但 fixed65 的平均 denoise/infer 时间略低于 v4。该结果尚未显示 MLP 在 task0 上优于固定 65% 调度。由于每组只有 5 episodes，下一步需在冻结的 task1–9 上进行正式配对比较。

## 9. 必须补充的实验

### P0：形成可信核心结论

| 实验 | 目的 | 完成标准 | 状态 |
|---|---|---|---|
| Meta-only task1–9 | 判断是否只学到 timestep schedule | 与 v4 使用相同任务/seeds，报告成功率、速度、internal ratio | 已完成：173/180，70.0% internal |
| Meta-only Object/10 | 验证四个 Suite 的泛化 | task0–9，至少 200 episodes | 待运行 |
| Compute-matched fixed | 排除“只是跳得更多” | internal ratio 与 dynamic 约 65% 对齐 | 已实现，待闭环评测 |
| Compute-matched random | 验证 importance routing 优于随机分配 | 使用相同 full/internal 数量和安全约束 | 已实现，待闭环评测 |
| Full 精确总表 | 回答最终总推理加速 | 四个 Suite 均报告成功率和绝对 timing | 部分完成 |

### P1：把故事讲完整

| 实验 | 目的 | 建议输出 |
|---|---|---|
| Threshold sweep | 展示速度—成功率权衡 | Pareto curve |
| hidden-only | 分离 hidden 与 meta 的贡献 | 三输入模式消融表 |
| 标签质量/校准 | 证明 MLP 能预测真实风险 | MAE、相关系数、AUC、calibration curve |
| Route-switch jitter | 检查路由切换的副作用 | 切换点附近 jerk/replan jump 曲线 |
| Failure case study | 解释独有失败 | 视频帧 + importance + route 时间轴 |
| 多 seeds/置信区间 | 提高统计可信度 | bootstrap CI、paired McNemar |

### P2：根据结果决定的改进

- 若切换导致抖动：加入 hysteresis、EMA 平滑、最短驻留步数或切换惩罚。
- 若 Spatial/Object 过度保守：改善训练数据的跨 Suite 平衡或进行风险校准，而不是在 held-out task 上单独调阈值。
- 若 meta-only 很强：重新设计更依赖状态的标签，增加阶段变化、接触风险或 uncertainty/entropy 特征。
- 若阈值跨任务不稳定：研究分位数阈值、校准层或带预算约束的路由。

## 10. 计划中的论文图表

### 主表

1. 四个 LIBERO Suite 上 full、fixed、dynamic 的成功率与速度。
2. hidden+meta、meta-only、hidden-only、compute-matched fixed/random 的消融。
3. 动作平滑性与失败指标。

### 主图

1. 方法图：浅层 hidden → MLP importance predictor → internal/full route。
2. 单个 rollout 的视频帧—importance—真实 gap—路由对齐图。
3. 各 Suite/task 的 internal ratio heatmap。
4. speed–success Pareto curve。
5. 成功/失败案例及路由切换分析。

## 11. 可写入论文的叙事框架

### Motivation

固定 ActionGap 假设所有状态和 denoising steps 具有相同计算需求，但机器人动作生成中，不同步骤和任务阶段的风险并不均匀。

### Method

利用已经计算出的中间 hidden state，通过轻量 MLP 估计 internal 分支相对 full 分支的动作误差，并在安全约束下按步骤动态分配计算。

### Evidence chain

1. 真实 internal/full action gap 随 step、task 和 episode 变化。
2. MLP 对该 gap 具有预测能力。
3. 路由结果随任务风险变化，而非固定比例。
4. 相同算力下，预测路由优于 fixed/random/meta-only。
5. 闭环成功率基本保持，推理时间降低。
6. 路由切换不会造成不可接受的动作抖动；若有，则通过安全机制控制。

### 当前可用表述

> 在四个 LIBERO Suite 的 380 个 fixed/dynamic 配对 episode 上，hidden+meta Dynamic ActionGap 与 fixed gap=4 均取得 366/380 的成功数，同时将 internal ratio 从 60.0% 提高到约 64.9%，获得约 1.043x Action denoise 和 1.029x `infer_action` 增量加速。相对 full inference，当前动态方案的 Action denoise 和 `infer_action` 加速约为 2.03x 和 1.70x。另一方面，meta-only 在 Goal/Spatial task1–9 上取得 173/180，高于 hidden+meta 的 172/180，并以固定 70.0% internal ratio 获得更高速度。因此，目前能够证明可学习调度有效，但尚不能证明 hidden state 比 timestep/meta 调度更优；该结论需要 compute-matched 和 Pareto 消融确认。

### 当前禁止过度声称

在关键消融完成前，不应写成：

- “已经证明 hidden state 能准确判断每一步的重要性”；
- “动态方案严格不掉成功率”；
- “动态方案在所有任务上都比 fixed 更快”；
- “已经实现整个机器人系统 1.70x 端到端加速”；
- “动态路由不会引入动作抖动”。

## 12. 后续执行顺序

1. 对 hidden+meta 与 meta-only 做阈值扫描，至少覆盖约 60%、65%、70% internal ratio。
2. 实现并运行 compute-matched fixed/random baseline。
3. 汇总 Goal/Spatial 的 speed–success Pareto curve，判断 hidden 的预测优势能否转化为闭环收益。
4. 增加预测相关性、风险分类和校准可视化，建立 hidden state importance 的直接证据。
5. 若 hidden 在 Pareto 或高风险样本上有价值，再完成 Object/LIBERO-10 的 meta-only 扩展评测；若没有，则先修改路由目标或安全策略。
6. 对 route switching、jitter 和独有失败进行视频对齐分析。
7. 根据消融结果决定是否加入平滑/hysteresis，以及最终论文主线。
8. 更新本报告并提炼为 5–8 页组会 PPT 或论文的 Method/Experiments 初稿。

## 13. 当前完成度清单

- [x] full teacher 数据采集与 internal/full gap 标签
- [x] hidden+meta MLP 训练与验证
- [x] 在线 dynamic routing 与安全约束
- [x] task0 smoke/阈值筛选
- [x] Goal/Spatial task1–9
- [x] Object/LIBERO-10 task0–9
- [x] full/fixed/dynamic timing breakdown
- [x] route heatmap、route reason 和 task internal ratio 可视化
- [x] 初步 rollout 视频与 jitter 指标
- [x] meta-only/hidden-only 代码与单元测试
- [x] meta-only task0 smoke
- [x] meta-only Goal/Spatial task1–9
- [x] hidden+meta 与 meta-only 验证误差对比
- [ ] meta-only Object/LIBERO-10 task0–9
- [x] compute-matched fixed/random baseline 代码与单元测试
- [ ] compute-matched fixed/random 闭环评测
- [ ] threshold Pareto curve
- [ ] 完整 jitter 与 failure case study
- [ ] 统计检验和置信区间
- [ ] 最终论文图表与文字
