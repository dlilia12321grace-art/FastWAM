# FastWAM Action VDE 实验计划

## 目的

在同一 FastWAM checkpoint、LIBERO 初始状态和 10-step Action DiT 设置下，比较 Full 与 training-free Action VDE，验证 VDE 是否能在不训练新参数的情况下减少完整 Action DiT 调用。

FastWAM 的在线 `infer_action()` 只对 Action latent 进行多步去噪；Video 侧为每个 action chunk 的一次观测编码和 KV Prefill。因此本实验只实现 Action VDE，VideoGap 作为跨 chunk KV 复用方法单独评测。

## 第一阶段配置

- `enable_action_vde=true`
- `action_vde_warmup_steps=4`
- `action_vde_anchor_interval=2`
- 10 个去噪步对应路由：`full, full, full, full, estimate, full, estimate, full, estimate, full`
- 每个 chunk 预计 7 次 Full、3 次 estimate。
- 首轮强制关闭 C3ache、Internal Head、Internal LoRA、ActionGap 和 VideoGap，避免归因混淆。

## 运行顺序

```bash
cd /root/autodl-tmp/workspace/FastWAM
source /root/autodl-tmp/envs/fastwam/bin/activate

bash scripts/run_action_vde_smoke.sh both
```

通过 1-trial smoke 后，将 `NUM_TRIALS=1`、`TASK_ID=0..9` 跑 LIBERO-10 cross10；确认没有系统性掉点后再跑每任务 3 次的 cross30。

## 记录指标

- 成功数与失败任务；
- `action_vde.step_modes`、Full/estimate/fallback 数；
- `action_denoise_step_ms` 与 `infer_action_total_ms`；
- 相对 Full 的 Action DiT 与端到端加速；
- 是否出现 NaN、shape 变化或 scheduler 异常。

## 表述边界

1-trial smoke 只验证工程闭环。cross30 仍属于 pilot，不足以单独声明严格的成功率 non-inferiority；正确表述应同时报告成功数、失败任务和加速比。
