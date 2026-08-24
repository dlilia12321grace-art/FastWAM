# FastWAM VideoGap 实验计划

## 目标

在保持 Action 分支配置不变的前提下，测试相邻 action/replanning chunks 间复用
Video DiT K/V cache 是否能够减少 `infer_action` 延迟并保持 LIBERO 闭环成功率。

## 当前实现

- `video_gap=1`：每个 action chunk 都根据当前观测刷新图像编码、Video pre-DiT
  和 Video KV prefill，等价于原始基线。
- `video_gap=N`：第一个 chunk 刷新，之后每隔 N 个 chunks 刷新一次；其余
  chunks 复用最近一次 Video KV cache。
- cache 在每个 episode 开始时强制清空，禁止跨轨迹复用。
- action horizon 或输入图像 shape 改变时强制刷新。
- 日志记录 `refresh_cache`、`cache_age`、刷新率，以及 image encode、Video
  pre-DiT、Video KV prefill、Action denoise 和 `infer_action` 分项耗时。

注意：当前实现复用完整视觉条件，因此 cache chunk 同时跳过图像 VAE encode、
Video pre-DiT 和 Video KV prefill。它不改变 ActionGap 路由。

## 第一阶段：task0 smoke

固定 `fork2_b0 + ActionGap=4`，在 Goal/Spatial task0 各运行5个 episode：

```bash
source /root/autodl-tmp/fastwam_env.sh
cd /root/autodl-tmp/workspace/FastWAM

VIDEO_GAPS="1 2 4" \
NUM_TRIALS=5 \
SUITES="libero_goal libero_spatial" \
  bash scripts/run_video_gap_smoke.sh all
```

若 fork2_b0 checkpoint 不在默认位置，显式指定：

```bash
INTERNAL_LORA_CKPT=/path/to/fork2_b0.best.pt \
  bash scripts/run_video_gap_smoke.sh all
```

默认结果目录：

```text
/root/autodl-tmp/evaluate_results/video_gap_smoke
```

## 决策标准

1. 先确认 gap=1 与既有 fork2_b0 fixed-gap4 baseline 的成功率和计时一致。
2. gap=2 若10/10且有稳定 wall-clock 收益，再扩大到 held-out tasks。
3. gap=4 若 task0 已明显掉点，不扩大评测；保留为速度上界。
4. 第一阶段不与 Dynamic ActionGap 联合，避免无法区分 VideoGap 与 Action 路由贡献。

## task0 smoke 结果（2026-08-20）

| Method | Success | Image encode ms | Video pre ms | Video KV ms | Action ms | Infer ms | Refresh |
|---|---:|---:|---:|---:|---:|---:|---:|
| VideoGap1 | 10/10 | 12.17 | 1.06 | 30.30 | 135.26 | 210.43 | 100.0% |
| VideoGap2 | 6/10 | 5.31 | 0.55 | 15.48 | 137.44 | 189.27 | 50.4% |
| VideoGap4 | 4/10 | 3.44 | 0.33 | 8.14 | 139.90 | 182.70 | 25.6% |

VideoGap2/4 分别获得 `1.112x/1.152x` `infer_action` 加速，但成功数下降
`4/10` 和 `6/10`。固定跨 chunk 视觉复用未通过 smoke，不扩大相同配置；它仅用于
证明视觉计算的速度潜力和 stale-visual 风险。

## 后续候选

若固定 VideoGap 可用，再比较：

- 基于像素/latent变化量的动态刷新；
- 强制最大 cache age；
- 接触、夹爪切换或高动作波动时强制刷新；
- VideoGap 与 Action early exit 的组合速度—成功率前沿。

## 动态图像变化基线

固定 VideoGap 失败后，下一步不直接训练 MLP，而是先验证动态刷新的可行上界。
当前实现使用当前输入图像相对上次 Video cache 刷新图像的归一化输入空间 MSE：

- `image_mse <= threshold`：图像稳定，复用 Video KV；
- `image_mse > threshold`：图像变化明显，刷新完整视觉条件；
- cache age 超过上限时无条件刷新；第一阶段上限为1，因此不会连续复用两个 chunks。

先全刷新采集 task0 Goal/Spatial 的相邻图像 MSE 分布：

```bash
bash scripts/run_dynamic_video_gap_smoke.sh collect
```

脚本输出 q20/q35 两个保守候选阈值。随后运行动态 smoke：

```bash
bash scripts/run_dynamic_video_gap_smoke.sh dynamic
```

若需要手动指定阈值：

```bash
DYNAMIC_THRESHOLDS="0.001 0.002" MAX_CACHE_AGE=1 \
  bash scripts/run_dynamic_video_gap_smoke.sh dynamic
```

只有在成功率接近 gap1 且产生非零复用率时，才值得进一步加入 latent、proprio、
gripper/contact 或 MLP 特征。
