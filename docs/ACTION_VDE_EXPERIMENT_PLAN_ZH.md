# FastWAM Training-free Action VDE 实验报告

> 更新日期：2026-08-29
> 状态：FastWAM/LIBERO cross30 与 RoboTwin 初步跨任务验证已完成；停止继续拆分 smoke，进入结果整理。

## 目的

在同一 FastWAM checkpoint、LIBERO 初始状态和 10-step Action DiT 设置下，比较 Full 与 training-free Action VDE，验证 VDE 是否能在不训练新参数的情况下减少完整 Action DiT 调用。

FastWAM 的在线 `infer_action()` 只对 Action latent 进行多步去噪；Video 侧为每个 action chunk 的一次观测编码和 KV Prefill。因此本实验只实现 Action VDE，VideoGap 作为跨 chunk KV 复用方法单独评测。

## 实验配置

- `enable_action_vde=true`
- `action_vde_warmup_steps=4`
- `action_vde_anchor_interval=2`
- 10 个去噪步对应路由：`full, full, full, full, estimate, full, estimate, full, estimate, full`
- 每个 chunk 预计 7 次 Full、3 次 estimate。
- 首轮强制关闭 C3ache、Internal Head、Internal LoRA、ActionGap 和 VideoGap，避免归因混淆。

## 实现与运行入口

```bash
cd /root/autodl-tmp/workspace/FastWAM
source /root/autodl-tmp/envs/fastwam/bin/activate

bash scripts/run_action_vde_smoke.sh both
```

RoboTwin 通过 `configs/sim_robotwin.yaml`、`experiments/robotwin/eval_robotwin_single.py`
和 `experiments/robotwin/fastwam_policy/deploy_policy.py` 转发同一组 VDE 参数。两条评测链路
都默认关闭 VDE，只有显式设置 `enable_action_vde=true` 才启用，避免污染既有 Full 基线。

## LIBERO-10 结果

### cross10

| 方法 | 成功 | records | Action mean/median ms | Infer mean/median ms | fallback |
|---|---:|---:|---:|---:|---:|
| Full | 10/10 | 248 | 295.881 / 295.338 | 379.787 / 362.036 | - |
| Action VDE | 10/10 | 247 | 211.847 / 210.194 | 296.667 / 276.343 | 0 |

Action mean/median 加速为 `1.397x/1.405x`，`infer_action` mean/median 加速为
`1.280x/1.310x`。

### cross30 主 pilot

| 方法 | 成功 | records | Action mean/median ms | Infer mean/median ms | fallback |
|---|---:|---:|---:|---:|---:|
| Full | 29/30 | 761 | 298.823 / 298.891 | 371.472 / 365.704 | - |
| Action VDE | 30/30 | 730 | 213.544 / 212.185 | 286.834 / 278.972 | 0 |

Action mean/median 加速为 `1.399x/1.409x`，`infer_action` mean/median 加速为
`1.295x/1.311x`。Action 与端到端耗时分别降低约 `28.5%` 和 `22.8%`。730 个
action chunks 全部采用固定 `7 Full + 3 estimate` 路由，累计约 2190 个 estimate，
没有 fallback。Full 的一次失败发生在 moka pots 任务；VDE 的 30/30 不能解释为成功率提升。

## RoboTwin 结果

环境为 RTX 5090、PyTorch 2.7.1+cu128、RoboTwin `demo_clean`、ALOHA AgileX，使用
`robotwin_uncond_3cam_384.pt`。SAPIEN/OIDN 在 Blackwell GPU 上会输出 `invalid handle`
警告，但本轮评测能够正常完成；该警告不是模型或 VDE 错误。

### beat_block_hammer 配对 3 seeds

| 方法 | 成功 | records | Action mean/median ms | Infer mean/median ms |
|---|---:|---:|---:|---:|
| Full | 3/3 | 17 | 325.011 / 324.614 | 538.296 / 404.740 |
| Action VDE | 3/3 | 15 | 215.068 / 214.992 | 310.965 / 287.832 |

Action mean/median 加速均约 `1.51x`；`infer_action` mean/median 加速为
`1.731x/1.406x`。均值受首次推理和编译开销影响较大，因此跨环境表述优先采用中位数。

### 四任务跨任务 smoke

任务覆盖 `click_alarmclock`、`open_microwave`、`handover_block` 和
`stack_blocks_two`。Full 与 Action VDE 在相同 seed 上均为 `1/1`，合计各 `4/4`；
所有 VDE chunks 均为 `7 Full + 3 estimate + 0 fallback`。

汇总 timing（Full 48 records，VDE 55 records）：Action mean/median 加速为
`1.385x/1.425x`，`infer_action` mean/median 加速为 `1.267x/1.293x`。由于不同方法
达到终止状态所需的 replanning 次数不同，pooled timing 只作为工程速度参考，不能代替
逐任务配对统计。

## 核心结论

Action VDE 在不训练新参数的前提下，将 10-step Action DiT 的完整模型调用从 10 次
降为 7 次。FastWAM/LIBERO cross30 观察到约 `1.40x` Action DiT 和 `1.30x`
端到端动作推理加速，且未观察到 aggregate success 下降；RoboTwin 的 5 个任务初步
验证同样保持成功，并观察到约 `1.3x--1.5x` 的 Action 加速。结果支持 Action VDE
可以跨 FastWAM 的 LIBERO 与 RoboTwin 评测链路工作。

## 记录指标

- 成功数与失败任务；
- `action_vde.step_modes`、Full/estimate/fallback 数；
- `action_denoise_step_ms` 与 `infer_action_total_ms`；
- 相对 Full 的 Action DiT 与端到端加速；
- 是否出现 NaN、shape 变化或 scheduler 异常。

## 表述边界

1. LIBERO cross30 仍属于 pilot，不足以声明统计显著提升或严格 non-inferiority。
2. RoboTwin 仅有一个任务 3 seeds 和四任务单 seed，属于跨任务工程验证，不是完整 benchmark。
3. 不把 VDE 的 30/30 相对 Full 29/30 表述为成功率提升。
4. 不将不同模型、GPU 或运行路径下的绝对毫秒数直接横向比较。
5. 正确表述应同时报告成功数、路由、fallback 和加速比。
