# FastWAM Internal Early Exit 当前阶段快照

> 冻结日期：2026-08-19  
> 用途：论文写作、项目交接和后续实验的唯一当前入口。  
> 状态：最低实验闭环已完成，停止新增实验并进入论文写作。历史 fork4/v4 结果仅用于说明研究演进，不代表当前最终方案。

## 1. 当前论文主线

论文以 **FastWAM** 为唯一主线，研究 Action DiT 的 internal early exit：

1. **Where / how much**：从哪个 fork layer 提前退出、需要多少 internal blocks；
2. **When**：利用中间 hidden state 和轻量 MLP 判断当前 denoising step 走 internal 还是 full；
3. **Evaluation**：闭环成功率、Action denoise、`infer_action`、参数量、训练成本、internal ratio 和控制稳定性。

LingBot-VA、RoboTwin 2.0 和真机不是当前论文成立的前提。LingBot-VA 已完成初步迁移 smoke，可作为可选泛化证据；RoboTwin 2.0 由老师侧运行，真机仍属于未来扩展。

## 2. 已完成工作

- FastWAM/LIBERO 复现与 full/fixed ActionGap 基线；
- 旧 fork4_b1 Dynamic ActionGap 全链路、meta-only、matched fixed/random、oracle、可视化和 jitter 诊断；
- `fork layer={2,4,8,12,16} × internal blocks={0,1,2,4}` 分级结构筛选；
- 五个 Pareto 候选的 100-update 正式训练与 held-out pilot；
- 最终三个结构在 task1–9 Goal/Spatial 上的正式配对验证；
- fork2_b0 与 fork4_b1 在 Object/LIBERO-10 上的正式泛化；
- fork2_b0 hidden+meta gate 重新采集、训练和 task1–9 阈值扫描；
- 所有当前 runner 均支持显式 fork/source 配置；主要长评测支持结果级断点续跑；
- 本地测试：`38 passed`。

## 3. 最终 static architecture 结果

最终选择：`fork2_b0`，即从 Action DiT layer 2 fork，hidden 直接接 action head，不复制额外 internal transformer block。

### Goal/Spatial task1–9（180 episodes/结构）

| Method | Success | Denoise ms | Infer ms | Internal | Trainable |
|---|---:|---:|---:|---:|---:|
| fork2_b0 | 172/180 | 134.01 | 203.27 | 60.0% | 7,175 |
| fork2_b4 | 171/180 | 158.22 | 229.12 | 60.0% | 1,383,431 |
| fork4_b1 reference | 171/180 | 152.53 | 222.86 | 60.0% | 351,239 |

### Object/LIBERO-10 task0–9（200 episodes/结构）

| Method | Success | Denoise ms | Infer ms | Internal | Trainable |
|---|---:|---:|---:|---:|---:|
| fork2_b0 | 196/200 | 133.59 | 201.05 | 60.0% | 7,175 |
| fork4_b1 reference | 193/200 | 151.02 | 219.14 | 60.0% | 351,239 |

### 四 Suite 合计

| Method | Success | Weighted denoise ms | Weighted infer ms | Internal | Trainable |
|---|---:|---:|---:|---:|---:|
| fork2_b0 | **368/380** | **133.79** | **202.10** | 60.0% | **7,175** |
| fork4_b1 reference | 364/380 | 151.74 | 220.90 | 60.0% | 351,239 |

相对 reference，fork2_b0 观察到成功数 `+4`（`+1.05 pp`，不声称统计显著），denoise/infer 分别加速约 `1.134x/1.093x`，可训练参数减少约 `49x`。这是当前最稳固的论文主结果。

架构筛选还表明：更晚 fork、更多 blocks 可以显著降低离线蒸馏 loss，但没有转化为更好的闭环成功率；不能用 loss 单独选择 early-exit 结构。

## 4. fork2_b0 Dynamic MLP 结果

新 hidden+meta gate：

| Metric | Value |
|---|---:|
| Default threshold | 0.256825 |
| Validation MAE | 0.011909 |
| Validation Smooth L1 | 0.0001378 |

### task1–9 Goal/Spatial（180 episodes/method）

| Method | Success | Denoise ms | Infer ms | Internal | 定位 |
|---|---:|---:|---:|---:|---|
| fixed gap4 | 172/180 | 133.40 | 202.75 | 60.0% | static reference |
| default threshold | 176/180 | 190.84 | 260.60 | 40.7% | conservative / quality-first |
| threshold 0.35 | 174/180 | 135.43 | 204.95 | 61.1% | balanced candidate |
| threshold 0.40 | 172/180 | **114.28** | **183.40** | 68.5% | speed-first candidate |

解释边界：

- default threshold 多成功 4 次但使用更多 full computation，且明显更慢；它只是 quality-first 点。
- t0.35 多成功 2 次但 wall-clock 略慢，不能称为加速方案。
- t0.40 与 fixed 成功总数相同，配对 dynamic-only/fixed-only 均为 2；相对 fixed 的 denoise/infer 为 `1.167x/1.106x`。
- t0.40 的 68.5% compute-matched fixed/random 已完成：learned route 没有优于同算力 state-independent schedule。

### t0.40 compute-matched（task1–9 Goal/Spatial，180 episodes/method）

| Method | Success | Denoise ms | Infer ms | Internal |
|---|---:|---:|---:|---:|
| fork2_b0 t0.40 | 172/180 | 114.28 | 183.40 | 68.5% |
| fixed685 | 171/180 | **112.70** | 182.44 | 68.1% |
| random685 | 172/180 | **112.69** | **182.12** | 68.1% |

t0.40 与 random685 成功数相同，但 denoise/infer 分别约慢 `1.4%/0.7%`；相对 fixed685 多成功 1 次，但同样略慢。当前 MLP 没有改善 compute-matched 闭环 frontier。

## 5. 当前可写与不可写的结论

### 可以写

1. FastWAM 的浅层 hidden state 可以直接支持轻量 action head；额外 internal block 并非必要。
2. fork2_b0 在四个 LIBERO Suite 的现有 380-episode 证据中，同时改善参数量、延迟和成功数观察值。
3. fork/block 存在明确的质量—延迟—参数权衡，离线 loss 与闭环成功率不完全一致。
4. hidden+meta MLP 能准确预测 internal/full action gap，并产生从 quality-first 到 speed-first 的阈值前沿。
5. t0.40 在 Goal/Spatial 的 180 episodes 上观察到与 fixed 相同成功数和约 10.6% `infer_action` 增量加速。
6. LingBot-VA 已跑通 layer-2 direct head：60.8% internal timing smoke 中，internal step、Action DiT 和 infer wall 分别约为 `10.68x/2.27x/1.66x`；独立的 40% internal 小样本闭环对照与 full 均为 `9/10`。

### 不能写

1. 不能声称成功率有统计显著提升或已经证明非劣。
2. 不能声称 t0.40 的收益来自 learned importance；matched 结果显示它未优于 random685。
3. 不能把相对 full 的全部收益归因于 MLP；主要收益来自 internal branch。
4. 不能把旧 fork4/v4 的 matched 负结果当成 fork2_b0 的最终 matched 结论。
5. 不能声称已完成 LingBot 正式泛化、RoboTwin 或真机验证；LingBot 当前只有 timing smoke 和训练 trial 重叠的小样本闭环结果。
6. 不能把 LingBot 的 60% timing 与 40% success 合并成同一配置的 speed-success 结论。

## 6. 实验停止点

最低必需实验已经全部完成，当前没有必须继续的训练或测试。论文主线应以 fork2_b0 architecture Pareto 为正结果，以 MLP matched 负结果作为诊断和边界。LingBot-VA 初步迁移已完成但不阻塞写作；其未见 trial 配对、t0.35 matched、Object/LIBERO-10 dynamic、jitter、多 seed、RoboTwin/真机均为可选扩展。

## 7. 关键产物

### 本地代码与报告

```text
F:\codexprogramms\fastWAM\FastWAM-official
F:\codexprogramms\fastWAM\FastWAM-official\docs\DYNAMIC_ACTION_GAP_EXPERIMENT_REPORT_ZH.md
F:\codexprogramms\fastWAM\FastWAM-official\docs\LINGBOT_VA_TRANSFER_REPORT_ZH.md
F:\codexprogramms\fastWAM\HANDOFF_FASTWAM.md
```

### 远端结果

```text
/root/autodl-tmp/evaluate_results/internal_architecture_fork_screen
/root/autodl-tmp/evaluate_results/internal_architecture_pareto
/root/autodl-tmp/evaluate_results/dynamic_action_gap_fork2_b0
```

关键 JSON：

```text
internal_architecture_pareto/cross_v1/summary.json
internal_architecture_pareto/cross_object10_v1/summary.json
dynamic_action_gap_fork2_b0/cross_task_summary_default_threshold.json
dynamic_action_gap_fork2_b0/cross_task_summary_f2b0_t035.json
dynamic_action_gap_fork2_b0/cross_task_summary_f2b0_t040.json
dynamic_action_gap_fork2_b0/cross_task_summary_fixed685.json
dynamic_action_gap_fork2_b0/cross_task_summary_random685.json
```

## 8. 文档阅读顺序

1. 本快照；
2. `DYNAMIC_ACTION_GAP_EXPERIMENT_REPORT_ZH.md`：完整实验演进与细节；
3. `LINGBOT_VA_TRANSFER_REPORT_ZH.md`：LingBot-VA 迁移、timing、闭环 smoke 与证据边界；
4. `HANDOFF_FASTWAM.md`：环境、代码、远端和历史操作；
5. `paper/`：论文 Agent 维护的稿件，若与本快照数字冲突，以本快照和原始 JSON 为准。
