# LingBot-VA Internal Early Exit 迁移实验记录

> 更新日期：2026-08-19  
> 定位：FastWAM 论文的可选跨模型泛化实验，不替代 FastWAM 四 Suite 主结果。  
> 当前状态：代码链路、teacher 蒸馏、固定比例 early exit、加速和小样本闭环 smoke 已跑通；正式未见 trial 配对评测尚未完成。

## 1. 迁移目标与方法

将 FastWAM 最终 static architecture 的核心结论迁移到 LingBot-VA：在 Action DiT 第 2 层后 fork，直接使用轻量 action head 解码，不增加额外 internal transformer block（`fork2_b0`）。

当前实现支持：

- full teacher 特征/动作目标采集；
- 直接 action head 蒸馏训练；
- 固定 internal ratio 的 full/internal 调度；
- CUDA event 级 Video DiT、Action DiT、KV cache 和总推理计时；
- 每个 action denoising step 的 route 与耗时记录。

LingBot-VA 的 direct head 为 `Linear(3072, 30)`，共 `92,190` 个可训练参数。该数字不可与 FastWAM 的 `7,175` 直接比较，因为两者 action 输出维度和模型实现不同。

## 2. Full baseline profiling

在 `libero_10 task0` 上运行 5 个闭环 episode，成功 `5/5`。共记录 90 个 infer chunk、85 次 KV cache update。

| Metric | Mean |
|---|---:|
| Video DiT | 1713.52 ms |
| Video loop | 1724.75 ms |
| Action DiT | 4159.42 ms |
| Action loop | 4191.65 ms |
| Infer wall | 5927.01 ms |
| Action DiT step | 81.56 ms |
| Video DiT step | 81.60 ms |
| Video KV | 83.06 ms |
| Action KV | 81.64 ms |
| KV wall | 343.91 ms |

Action DiT 约占单次 `infer wall` 的 70%，因此是 early exit 的主要加速目标。

## 3. Teacher 数据与 head 训练

### 3.1 链路 smoke

- 1 episode；
- 18 个 teacher chunk；
- checkpoint：`lingbot_fork2_b0_head_smoke.pt`；
- validation MSE：0.32517；
- validation MAE：0.34443；
- 训练时间：2.81 秒。

该 checkpoint 只用于验证训练和推理链路，不能用于评价成功率。

### 3.2 扩展训练

在 LIBERO-10 的 10 个任务上按每任务 2 episode 发起 teacher collection。当前确认保存：

- 393 个 teacher chunk；
- `visualization/real` 总目录 4.9GB（包含 timing、视频及 teacher 文件）；
- checkpoint：`lingbot_fork2_b0_head_collect20.pt`。

训练结果：

| Metric | Value |
|---|---:|
| Validation MSE | 0.22597 |
| Validation MAE | 0.28923 |
| Validation values | 3,867,840 |
| Training time | 122.53 s |
| Trainable parameters | 92,190 |

相对单 episode smoke，MSE 下降约 30.5%，MAE 下降约 16.0%。当前没有保存到报告中的完整 collection 最终成功数，因此不补写该数字。

## 4. 60% internal：纯加速链路验证

使用单 episode smoke head、目标 internal ratio 60% 运行 task0。该 episode 失败并跑满 796 帧；失败符合训练数据极少、验证误差较高的预期，不能用于判断最终模型质量。

但 50 个 infer chunk 的计算链路结果有效：

| Metric | Value |
|---|---:|
| Actual internal ratio | 60.78%（31/51 calls） |
| Full action step | 80.13 ms |
| Internal action step | 7.50 ms |
| Internal step speedup | 10.68x |
| Action DiT | 1835.19 ms |
| Infer wall | 3568.41 ms |
| Action DiT speedup vs full | 2.266x |
| Infer speedup vs full | 1.661x |

这一结果证明 LingBot-VA 的浅层 direct head 确实跳过了大部分 Action DiT 计算，并产生实际 wall-clock 收益。它不证明 60% 配置已经保持成功率。

## 5. 40% internal：闭环 smoke 与小样本对照

使用扩展训练 checkpoint，将目标 internal ratio 降至 40%。

### 5.1 task0

- 首次 smoke：`1/1`；
- 随后 5-trial：`5/5`；
- 5-trial 总运行时间约 482 秒；
- rollout 长度为 264–297 帧，均未跑满。

首个 `1/1` 与 5-trial 中的 trial0 可能使用相同初始状态，因此应写作“两轮均成功”，不合并声称为 6 个独立 trial。

### 5.2 LIBERO-10 task0–9，各 1 trial

| Method | Success | Failed task |
|---|---:|---|
| Full | 9/10 | task3 |
| fork2_b0, target 40% internal | 9/10 | task9 |

配对结果：

- 两者结果一致：8/10；
- t040-only success：task3；
- full-only success：task9。

当前仅能表述为：在这组 10-task smoke 中，没有观察到 aggregate success 下降。不能声称严格非劣，也不能将不同失败任务解释为确定的方法差异。

## 6. 证据边界

### 当前可以写

1. LingBot-VA 的 Action DiT 是主要推理开销，约占 full infer wall 的 70%。
2. fork2 direct head 在 RTX 5090 上将 internal step 从约 80.1 ms 降到约 7.5 ms。
3. 在 60.8% internal ratio 的 timing smoke 中，Action DiT 和 infer wall 分别获得约 2.27x 和 1.66x 加速。
4. 扩展 teacher 数据训练后，40% internal 在 task0 取得 5/5，并在 10-task smoke 中与 full 同为 9/10。
5. FastWAM 的“浅层 hidden 直接接 action head”机制可以迁移并运行于第二种 VLA/VA 实现。

### 当前不能写

1. 不能声称 LingBot-VA 已完成正式泛化验证。
2. 不能把 60% timing 与 40% success 拼成同一个配置的 speed-success 结果。
3. 不能声称 40% 配置已有 1.66x infer 加速；其独立 timing 尚待汇总。
4. 不能声称不掉点或统计非劣；当前每任务只有 1 个对照 trial。
5. teacher collection 使用了 task0–9 的前两个 trial，当前 cross10 的 trial0 与训练数据重叠。
6. 不能将 LingBot 结果与 FastWAM 的毫秒数直接横向比较；模型规模、实现和运行路径不同。

## 7. 待补实验（非当前立即必跑）

### P0：形成论文可用 LingBot 表格

1. 独立汇总 t040 的 Action DiT、infer wall 和实际 internal ratio；
2. 在未参与训练的 trial2–4 上做 Full/t040 相同初始状态配对；
3. 至少报告 30 个未见 episode，或说明其为 pilot；
4. 给成功率差异提供 paired disagreement 与置信区间。

### P1：完整 speed-success curve

- full；
- 40% internal；
- 50% internal；
- 60% internal；
- 每个点使用同一组 trial，并分别报告实际 ratio、Action DiT、infer wall 和 success。

## 8. 关键远端产物

```text
/root/autodl-tmp/checkpoints/lingbot-va-posttrain-libero-long
/root/autodl-tmp/checkpoints/lingbot_fork2_b0_head_smoke.pt
/root/autodl-tmp/checkpoints/lingbot_fork2_b0_head_collect20.pt
/root/autodl-tmp/evaluate_results/lingbot_va_profile_full
/root/autodl-tmp/evaluate_results/lingbot_va_fork2_b0_exit_smoke
/root/autodl-tmp/evaluate_results/lingbot_va_fork2_b0_t040_task0_5
/root/autodl-tmp/evaluate_results/lingbot_va_fork2_b0_t040_cross10
/root/autodl-tmp/evaluate_results/lingbot_va_full_cross10
/root/autodl-tmp/workspace/lingbot-va/visualization/real
```

## 9. 当前一句话结论

> LingBot-VA 初步迁移验证表明，Action DiT layer-2 hidden 通过约 9.2 万参数的 direct head 可将 internal action step 加速约 10.7x；在 60.8% internal 的 timing smoke 中获得约 2.27x Action DiT 和 1.66x infer wall 加速，而更保守的 40% internal 在 10-task 小样本对照中与 full 同为 9/10。该结果支持方法具有跨实现迁移潜力，但正式未见 trial 配对评测仍待补充。
