# Dynamic ActionGap MLP 实施方案

更新日期：2026-08-14

## 1. 结论先行

在现有固定 `action_gap` 周期调度上增加一个轻量 MLP，预测当前去噪步使用 internal 分支的风险。推理时根据预测风险动态选择：

- 风险低：走浅层 internal 分支，节省计算；
- 风险高：走完整 Action DiT，及时纠偏；
- 第一步、最后一步以及超过最大连续 internal 步数时，强制走 full，作为安全边界。

建议把方法暂命名为 **Dynamic ActionGap（DAG）** 或 **Learned Internal Routing（LIR）**。

核心公式：

```text
z_t = Pool(h_t^fork) + timestep/statistics/history
r_hat_t = MLP(z_t)

route_t = internal,  if r_hat_t <= tau
          full,      otherwise
```

其中 `r_hat_t` 表示 MLP 预测的 internal/full 输出差异，而不是直接预测一个含义不清楚的整数 gap。

## 2. 为什么这是当前最合适的方案

现有实现已经具备：

1. full Action DiT 路径；
2. 从第 4 层分叉的浅层 Internal LoRA 路径；
3. 固定周期的 `build_action_gap_schedule()`；
4. 每一步记录 `full/internal` 路由与耗时的评测链路；
5. Internal 分支与 full 分支做蒸馏所需的中间层和 teacher 输出。

当前 `action_gap=4` 的路由固定为：

```text
full, internal, internal, full, internal, internal, internal, full, internal, full
```

已有 30 次评测结果表明固定调度具备可用性：

| 任务 | 成功次数 | Action 去噪平均耗时 |
|---|---:|---:|
| LIBERO Goal task 0 | 30/30 | 154.31 ms/chunk |
| LIBERO Spatial task 0 | 29/30 | 156.50 ms/chunk |

但是固定 gap 不看当前观测、动作状态或去噪难度。简单任务和稳定阶段可能不需要频繁 full，困难状态或高波动阶段则可能需要提前 full。MLP 动态路由正好补上这一点。

## 3. 预测目标：优先预测“风险”，不要直接预测 route

### 3.1 Teacher 标签

离线采集时，同一个去噪步同时计算 full 和 internal 两个动作噪声预测：

```text
p_full_t = full branch(h_t^fork)
p_int_t  = internal branch(h_t^fork)
```

定义连续风险标签：

```text
gap_t = mean((p_int_t - p_full_t)^2) /
        (mean(p_full_t^2) + eps)
```

训练 MLP 回归：

```text
target_t = log(1 + gap_t)
loss_gate = SmoothL1(r_hat_t, target_t)
```

推理时：

```text
use_internal = r_hat_t <= tau
```

选择连续风险回归的原因：

- 阈值 `tau` 可以在验证集上扫描，直接得到速度—精度曲线；
- 不必为每种算力预算重新训练分类器；
- 可以画出预测风险随去噪步变化的曲线，论文“动态故事”更直观；
- 后续可以用风险校准或不确定性估计继续扩展。

### 3.2 MVP 也可以先做二分类

如果需要最快验证代码链路，可先用：

```text
y_t = 1[gap_t <= tau_label]
loss_gate = BCEWithLogitsLoss(logit_t, y_t)
```

但正式实验建议回到连续风险回归。

## 4. MLP 输入特征

### 4.1 推荐 MVP 输入

路由点设在 `fork_layer=4`。前 4 层是 full/internal 两条路径共同需要的，因此先得到 `h_t^fork` 再决策，不会浪费 internal 路径的计算。

```text
pooled_h = mean(h_t^fork, dim=action_token)
meta = [
    normalized_timestep,
    normalized_step_index,
    latent_l2_norm,
    latent_delta_l2_norm,
    last_observed_gap,
    steps_since_last_full,
]
z_t = concat(LayerNorm(pooled_h), meta)
```

第一版至少保留：

- `pooled_h`：包含当前动作 token、观测 K/V 交互和指令信息；
- `normalized_timestep`：不同去噪阶段难度不同；
- `steps_since_last_full`：防止误差长时间累积。

`last_observed_gap` 只在 full 校正步上更新。full 步可额外运行一次很便宜的 internal 后缀，从而得到真实 gap；其余步骤沿用上一次观测值。

### 4.2 MLP 结构

```python
class DynamicActionGapGate(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int = 256):
        super().__init__()
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(bottleneck + 6, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, pooled_hidden, meta):
        return self.head(torch.cat([self.hidden_proj(pooled_hidden), meta], dim=-1))
```

MLP 只输出一个标量，计算开销相对于一个 Transformer block 可以忽略。

## 5. 推理路由规则

纯 MLP 决策需要加三条硬约束，避免一次误判造成连续漂移：

```python
force_full = (
    step_idx == 0
    or step_idx == num_steps - 1
    or steps_since_last_full >= max_internal_run
)

if force_full:
    route = "full"
elif predicted_gap <= threshold:
    route = "internal"
else:
    route = "full"
```

推荐初始值：

```text
max_internal_run = 3
```

它对应现有 `action_gap=4` 的最坏安全边界，但允许 MLP 在风险升高时提前 full。等验证稳定后，再尝试 4、5 或关闭该约束。

## 6. 代码落点

### 6.1 新增文件

建议新增：

```text
src/fastwam/models/wan22/dynamic_action_gap.py
scripts/collect_dynamic_action_gap_data.py
scripts/train_dynamic_action_gap.py
tests/test_dynamic_action_gap.py
```

职责：

- `dynamic_action_gap.py`：MLP、特征构造、阈值路由和 checkpoint I/O；
- `collect_dynamic_action_gap_data.py`：采集 `h_fork/full_pred/internal_pred/meta`；
- `train_dynamic_action_gap.py`：训练、验证、阈值扫描；
- `test_dynamic_action_gap.py`：边界条件、强制 full、状态 reset、checkpoint 测试。

### 6.2 修改现有文件

#### `src/fastwam/models/wan22/mot.py`

当前 `forward_action_with_video_cache()` 可以在第 4 层停止，但 full 路径不能从第 4 层继续。需要增加可复用的 suffix forward，使一次前缀计算后可以二选一：

```text
h0 -> Action blocks 1..4 -> h4
                         |-> full blocks 5..30 -> full head
                         `-> copied internal block -> internal head
```

建议新增类似：

```python
forward_action_suffix_with_video_cache(
    action_tokens,
    start_layer,
    ...
)
```

不要在选择 full 后重新从第 1 层计算，否则 gate 使用 `h4` 会让 full 步重复计算前四层。

#### `src/fastwam/models/wan22/fastwam.py`

需要：

- 在模型状态中注册 gate；
- 增加 episode 级 `reset_dynamic_action_gap_state()`；
- 在 `infer_action()` 中加入动态路由开关；
- 记录每步预测 gap、真实 gap、route、强制 full 原因；
- 支持 gate checkpoint 加载；
- 保留固定 ActionGap 作为 baseline 和回退路径。

#### `experiments/libero/eval_libero_single.py`

增加配置透传和结果汇总：

```text
enable_dynamic_action_gap
dynamic_action_gap_checkpoint
dynamic_action_gap_threshold
dynamic_action_gap_max_internal_run
collect_dynamic_action_gap_data
```

episode 开始前必须 reset gate 历史状态，不能把上一条轨迹的 `last_gap` 带入下一条轨迹。

#### `configs/sim_libero.yaml`

加入默认关闭的配置：

```yaml
enable_dynamic_action_gap: false
dynamic_action_gap_checkpoint: null
dynamic_action_gap_threshold: 0.0
dynamic_action_gap_max_internal_run: 3
collect_dynamic_action_gap_data: false
```

## 7. 数据采集与训练流程

### 阶段 A：只采数据，不改变实际动作

1. 加载已经训练好的浅层 Internal LoRA checkpoint；
2. 实际 rollout 始终使用 full 输出，避免采集策略改变状态分布；
3. 每个去噪步从共同的 `h4` 同时计算 full/internal 输出；
4. 保存 pooled feature、meta 和 gap 标签；
5. 采集多个 task/suite，不能只采 `goal0` 和 `spatial0`。

建议保存已经池化后的特征，不保存完整 token：

```python
{
    "pooled_hidden": ...,        # bf16/float16
    "meta": ...,
    "gap": ...,
    "suite": ...,
    "task_id": ...,
    "episode_id": ...,
    "chunk_index": ...,
    "step_index": ...,
}
```

### 阶段 B：离线训练 MLP

- 冻结 FastWAM 和 Internal LoRA，只训练 gate；
- 按 task 或 episode 划分 train/validation，不能随机打散同一轨迹的 step；
- 先用 `SmoothL1Loss`；
- 早停指标使用 validation MAE/Spearman，同时看 route 分类的 precision/recall；
- 在 validation 上扫描阈值，产生速度—风险 Pareto 曲线。

### 阶段 C：闭环评测

加载 gate 后做真实动态路由。闭环数据分布可能不同于 full-policy 采集数据，因此第一轮结果出来后，可以再采一轮动态策略数据进行微调，但测试任务必须保持隔离。

## 8. 实验矩阵

### 8.1 必做对照

| 编号 | 方法 | 目的 |
|---|---|---|
| A | Full Action DiT | 精度和耗时基线 |
| B | Fixed ActionGap = 2 | 保守固定调度 |
| C | Fixed ActionGap = 4 | 当前主要基线 |
| D | Fixed ActionGap = 5/10 | 更激进的固定调度 |
| E | Dynamic gate，无 max-run | 验证纯学习路由 |
| F | Dynamic gate，max-run=3 | 推荐方案 |

公平比较必须同时报告：

- LIBERO success rate；
- `action_denoise_total_ms_mean`；
- `infer_action_total_ms_mean`；
- full/internal 步数与比例；
- 预测 gap MAE、相关系数；
- 高风险步漏判率：本应 full 却走 internal 的比例。

### 8.2 消融实验

1. 只用 timestep/step index；
2. 加 `pooled_h4`；
3. 加历史波动特征；
4. 不同 `fork_layer`；
5. 不同阈值与 `max_internal_run`；
6. 回归 gap 与二分类 route 对比。

最关键的消融是第 1 和第 2 项：它能证明动态收益来自状态特征，而不是 MLP 仅仅学会了另一张固定时间表。

## 9. 最小可行版本（MVP）

第一版不要一次做太复杂，按下面顺序推进：

1. 重构为共享 1..4 层前缀，确保 full/internal 从同一个 `h4` 分流；
2. 实现双分支数据采集，验证保存的 gap 数值正确；
3. 用 `pooled_h4 + timestep + steps_since_last_full` 训练两层 MLP；
4. 接入 `threshold + max_internal_run=3` 动态路由；
5. 先跑 `goal0/spatial0` 各 5 次 smoke test；
6. 没有明显退化后，各跑 30 次；
7. 再扩到完整 LIBERO 任务集。

第一阶段的验收条件：

- 关闭新开关时，原始输出和固定 ActionGap 行为不变；
- gate checkpoint 可保存、加载并复现相同 route；
- 第一步和最后一步始终 full；
- 连续 internal 不超过 `max_internal_run`；
- 动态路由确实随样本/状态变化，而不只是随 step index 固定变化；
- 5 次 smoke test 不掉点后再进行大规模评测。

## 10. 需要避免的坑

1. **标签泄漏**：部署时不能用当前步的 full 输出计算 route，否则虽然叫动态路由，但没有省掉 full 计算。
2. **重复前缀计算**：基于 `h4` 决策后，full 路径必须从 h4 继续，不能从第 1 层重跑。
3. **只在两个简单任务训练**：MLP 可能学到任务偏置，必须跨任务采集并按任务划分验证集。
4. **只报平均成功率**：要同时报告最差任务、高风险漏判率与 bootstrap 置信区间。
5. **阈值在测试集调参**：阈值只能用 validation 选择，测试集只做最终一次报告。
6. **忽略 gate 状态 reset**：每个 episode 开始必须清空 `last_gap`、历史 latent 和连续 internal 计数。
7. **声称整机获得同等倍数加速**：Action 分支加速不等于端到端 policy 同倍数加速，必须分别报告两者。

## 11. 可以向老师汇报的一句话

> 我准备把当前固定周期的 ActionGap 改成一个基于内部特征的动态路由器：在共享浅层特征上用轻量 MLP 预测 internal 与 full 分支的误差风险，低风险时走 internal，高风险时自动调用完整网络纠偏，并用最大连续浅层步数保证稳定性。这样可以在相同成功率约束下学习每一步是否值得走 full，而不是人工固定间隔。

## 12. 当前推荐决策

先实现 **“预测 normalized internal/full gap + threshold 路由 + max-run 安全约束”**，不要第一版就预测可变整数 `action_gap`。前者标签明确、容易训练、容易做阈值扫描，也最符合“预测这一步要不要走 internal”的要求。

## 13. 2026-08-14 首版落地状态

已实现：

- `dynamic_action_gap.py`：风险 MLP、metadata、硬安全路由和 checkpoint；
- `fastwam.py`：共享前缀后的 full/internal 动态分流；
- full teacher rollout 下的 `h_fork + normalized gap` 数据采集；
- `train_dynamic_action_gap.py`：按 chunk 划分验证集并训练 gate；
- LIBERO 配置透传、episode reset、逐步路由日志和数据集保存；
- 首尾强制 full、最大连续 internal、checkpoint round-trip 单元测试。

采集数据时需要打开 Internal LoRA 和采集开关，并关闭固定/动态调度：

```bash
python experiments/libero/eval_libero_single.py \
  EVALUATION.enable_internal_lora_branch=true \
  EVALUATION.internal_lora_checkpoint=/path/to/internal_lora.pt \
  EVALUATION.internal_lora_fork_layer=4 \
  EVALUATION.internal_lora_source_start_layer=30 \
  EVALUATION.internal_lora_source_end_layer=30 \
  EVALUATION.collect_dynamic_action_gap_data=true \
  EVALUATION.enable_action_gap_schedule=false \
  EVALUATION.enable_dynamic_action_gap=false
```

离线训练：

```bash
PYTHONPATH=src python scripts/train_dynamic_action_gap.py \
  evaluate_results/**/gpu*_dynamic_action_gap.pt \
  --output checkpoints/dynamic_action_gap_gate.pt \
  --max-internal-run 3 \
  --target-internal-rate 0.6
```

动态闭环评测：

```bash
python experiments/libero/eval_libero_single.py \
  EVALUATION.enable_internal_lora_branch=true \
  EVALUATION.internal_lora_checkpoint=/path/to/internal_lora.pt \
  EVALUATION.internal_lora_fork_layer=4 \
  EVALUATION.internal_lora_source_start_layer=30 \
  EVALUATION.internal_lora_source_end_layer=30 \
  EVALUATION.enable_dynamic_action_gap=true \
  EVALUATION.dynamic_action_gap_checkpoint=/path/to/dynamic_action_gap_gate.pt \
  EVALUATION.dynamic_action_gap_threshold=null \
  EVALUATION.dynamic_action_gap_max_internal_run=3
```

`dynamic_action_gap_threshold=null` 表示使用训练 checkpoint 中按验证集选出的默认阈值。

本地已通过 16 个 CPU 单元测试与 Python 编译检查。尚未在当前 Windows 环境执行真实 LIBERO/GPU rollout，因此成功率、路由比例和端到端时延仍属于待验证项。

## 14. 一键 GPU smoke 流程

仓库新增：

```text
scripts/run_dynamic_action_gap_smoke.sh
scripts/summarize_dynamic_action_gap_smoke.py
```

在 AutoDL 仓库根目录运行：

```bash
source /root/autodl-tmp/fastwam_env.sh
cd /root/autodl-tmp/workspace/FastWAM

INTERNAL_LORA_CKPT=/path/to/shallow_fork4_checkpoint.pt \
  bash scripts/run_dynamic_action_gap_smoke.sh all
```

也可以拆开运行和断点续接：

```bash
bash scripts/run_dynamic_action_gap_smoke.sh collect
bash scripts/run_dynamic_action_gap_smoke.sh train
bash scripts/run_dynamic_action_gap_smoke.sh eval
```

默认执行：

1. `libero_goal task0` 和 `libero_spatial task0` 各 5 次 full teacher 数据采集；
2. 训练 Dynamic ActionGap gate；
3. 在相同两个任务上分别运行 fixed `action_gap=4` 与 dynamic gate；
4. 输出成功率、Action 去噪耗时、整体推理耗时、internal 比例和速度比。

结果汇总文件：

```text
/root/autodl-tmp/evaluate_results/dynamic_action_gap_smoke/comparison_summary.json
```

本地合成端到端 smoke 已通过：80 条样本、5 个 epoch 下，validation MAE 从约 `0.267` 降到 `0.146`；训练 checkpoint 成功保存、重新加载并产生有效的 `full/internal` 风险路由。当前完整测试数为 `16 passed`。
