# FastWAM Internal Early Exit / Dynamic ActionGap

本分支在官方 FastWAM Action DiT 上实现了中间层提前退出与动态计算分配，用于研究：是否可以利用浅层 hidden state 判断当前 action denoising step 的重要性，并在基本保持闭环成功率的情况下跳过后续 Action DiT blocks。

## 1. 实现组成

### Internal Head

- 从指定 Action DiT 中间层收集 hidden states 与 full model 输出。
- 离线蒸馏轻量 `InternalActionHead`。
- 在指定 denoising steps 从中间层直接解码 action noise，实现 early exit。

入口：

- `src/fastwam/models/wan22/fastwam.py`
- `scripts/train_internal_action_heads.py`
- `experiments/libero/eval_libero_single.py`

### Internal LoRA branch

- 从共享浅层 fork 出轻量 Action DiT 分支。
- 支持训练、保存、加载、merge 与指定 step 推理。
- 固定 ActionGap 与动态 MLP 均复用该 early-exit 分支。

入口：

- `src/fastwam/models/wan22/internal_action_branch.py`
- `src/fastwam/models/wan22/mot.py`
- `src/fastwam/models/wan22/fastwam.py`

### Dynamic ActionGap MLP

- 输入浅层 pooled hidden state、diffusion timestep、step index 和距离上次 full step 的间隔。
- 回归 `log1p(normalized internal/full action gap)`。
- 低风险 step 走 internal，高风险 step 走 full。
- first/last step、周期 anchor 和最大连续 internal 次数提供安全约束。

入口：

- `src/fastwam/models/wan22/dynamic_action_gap.py`
- `scripts/train_dynamic_action_gap.py`
- `scripts/run_dynamic_action_gap_smoke.sh`

### 公平对照与分析

- `fixed65`：交替使用 6/10 与 7/10 internal steps。
- `random65`：保持相同计算预算，随机选择额外 full step。
- Meta-only / hidden-only 输入消融。
- 路由 heatmap、route reason、任务级 internal ratio、动作 jitter 与 paired failure 分析。

入口：

- `scripts/summarize_dynamic_action_gap_smoke.py`
- `scripts/summarize_dynamic_action_gap_cross_task.py`
- `scripts/visualize_dynamic_action_gap.py`
- `scripts/summarize_action_jitter.py`

## 2. 当前主配置

```text
method            = v4_t036
threshold         = 0.36
max_internal_run  = 7
anchor_action_gap = 8
first/last step   = full
```

## 3. 当前结果摘要

在 LIBERO Goal、Spatial、Object 和 LIBERO-10 共 380 个 fixed/dynamic 配对 episode 上：

| Method | Success | Denoise ms | Infer ms | Internal ratio |
|---|---:|---:|---:|---:|
| fixed gap=4 | 366/380 | 150.58 | 219.77 | 60.0% |
| dynamic | 366/380 | 144.40 | 213.57 | 约 64.9% |

Dynamic 相对 fixed 获得约 `1.043x` Action denoise 和 `1.029x` `infer_action` 增量加速；相对 full inference 约为 `2.03x` 和 `1.70x`。

Hidden+meta gate 的验证 MAE 为 `0.01465`，meta-only 为 `0.10240`，说明 hidden state 明显提升了 internal/full gap 的预测精度。该离线预测优势能否转化为优于相同算力 fixed/random 路由的闭环优势，仍在通过 compute-matched 实验验证。

完整实验与限制见：

- `docs/DYNAMIC_ACTION_GAP_EXPERIMENT_REPORT_ZH.md`
- `docs/DYNAMIC_ACTION_GAP_MLP_PLAN_ZH.md`

## 4. 测试

```bash
export PYTHONPATH="$PWD/src:$PWD"
python -m pytest tests -q
```

最近本地结果：`33 passed`。

## 5. 结果表述边界

当前可以说明 dynamic routing 在已有样本上保持总体成功数并获得增量加速，也可以说明中间 hidden state 对 action-gap 标签具有更强预测能力。

在 compute-matched 跨任务结果完成前，不应声称 hidden routing 已经严格优于 fixed、random 或 timestep-only 调度。
