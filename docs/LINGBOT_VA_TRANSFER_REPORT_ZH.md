# LingBot-VA Internal Early Exit 迁移实验记录

> 更新日期：2026-08-29
> 定位：FastWAM 论文的可选跨模型泛化实验，不替代 FastWAM 四 Suite 主结果。  
> 当前状态：代码链路、teacher 蒸馏、固定比例 early exit、加速和小样本闭环 smoke 已跑通；Pure Dual-stream VDE 与 VDE+Internal 的 30 对 interleaved paired pilot 已完成。Internal-vs-Full 的正式未见 trial 扩展评测仍未完成。

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

### 5.3 Full 扩展到每任务 3 trial

Full 随后运行 task0–9、每任务 trial0–2。该次完整调用的结果为 `29/30`，其中未参与 teacher collection 的 trial2 为 `9/10`，失败项为 task9 trial2。

需要保留一个 runner 缺陷说明：旧版 `client.py` 将 `video_save_root_dict` 固定为 `None`，因此再次调用并不会真正 resume，而会从 trial0 开始重跑。task3 trial0 第一次为 False，后一次为 True，目录中因此同时存在两份视频，原始文件总数为 31。这里的 `29/30` 指后一次完整 30-episode 调用；历史全部实际执行为 `29/31`。该随机翻转也说明 10-episode 单次失败不能被过度解释。runner 已增加按 task/trial id 识别结果的续跑修复，后续正式配对应使用新输出目录和修复版本。

### 5.4 t040 clean 30-episode 对照

修复 runner 后，t040 使用全新目录运行 task0–9、每任务 trial0–2，共 30 个无重复 episode：

| Method | Overall | Unseen trial2 |
|---|---:|---:|
| Full | 29/30 | 9/10 |
| fork2_b0, target 40% internal | 27/30 | 9/10 |

t040 的 30 个 episode 共记录 643 个 infer chunk；同任务分布的 Full 30-episode 日志包含 553 个 infer chunk。最终 matched timing 汇总如下：

| Metric | Full | t040 | Speedup |
|---|---:|---:|---:|
| Success | 29/30 | 27/30 | — |
| Actual internal ratio | 0% | 39.22% | — |
| Full action step | — | 81.08 ms | — |
| Internal action step | — | 7.60 ms | 10.66x（vs t040 full step） |
| Video DiT | 1673.62 ms | 1705.27 ms | 0.981x |
| Action DiT | 4071.26 ms | 2665.49 ms | **1.527x** |
| Infer wall | 5796.82 ms | 4421.78 ms | **1.311x** |

相对 Full，t040 将平均 Action DiT 和 infer wall 分别降低约 34.5% 和 23.7%。Video DiT 基本不变，符合 early exit 只修改 Action DiT 的预期。由于失败 episode 会产生更多 infer chunk，两种方法的 chunk 总数不同；表中 latency 是每个 infer chunk 的均值，而不是整段 rollout 总时长。

t040 的三个失败为：task3 trial2、task4 trial1、task9 trial0。Full 后一次完整调用的失败为 task9 trial2。因此 30 个配对结果中：full-only=3、t040-only=1、both-fail=0、both-success=26；观察成功率差为 `-6.67 pp`，样本很小且 exact McNemar 双侧检验约为 `p=0.625`，不能声称显著退化或严格非劣。

更重要的未见初始状态 trial2 上，两者均为 `9/10`：Full 失败 task9，t040 失败 task3，配对上各有一个独有成功。这一子集没有观察到 aggregate success 下降，但只有 10 episodes，只能作为 held-out pilot。

### 5.5 Pure Dual-stream VDE 与 VDE+Internal 严格配对验证

在 LIBERO-10 task0–9、每任务 3 个固定 seed 上完成了 30 对、共 60 个 episode 的 interleaved paired 实验。Pure VDE 成功 `28/30`，VDE+Internal 成功 `29/30`；配对 outcome 为两者均成功 27 对、仅 Pure 成功 1 对、仅 Internal 成功 2 对、两者均失败 0 对。

VDE+Internal 相比 Pure VDE 的 Action DiT 延迟平均降低 `84.87 ms（4.90%）`，`infer_action` 平均降低 `84.16 ms（2.95%）`。两组均无 fallback、路由异常或基础设施重跑；Pure VDE 的 Internal 调用严格为 0，组合方案实际调用 6061 个 Internal step。该结果支持 Internal 在 Pure VDE 基础上提供约 `3%–5%` 的额外推理收益，本次小样本中未观察到成功率下降；但尚不能声称统计显著或统计非劣。完整结果见 [`LINGBOT_PAIRED_VDE_INTERNAL_RESULT_ZH.md`](../../lingbot-va-upstream/docs/LINGBOT_PAIRED_VDE_INTERNAL_RESULT_ZH.md)。

## 6. 证据边界

### 当前可以写

1. LingBot-VA 的 Action DiT 是主要推理开销，约占 full infer wall 的 70%。
2. fork2 direct head 在 RTX 5090 上将 internal step 从约 80.1 ms 降到约 7.5 ms。
3. 在 60.8% internal ratio 的 timing smoke 中，Action DiT 和 infer wall 分别获得约 2.27x 和 1.66x 加速。
4. 扩展 teacher 数据训练后，40% internal 在 task0 取得 5/5；30-episode pilot 为 27/30，对照 Full 为 29/30，而未见 trial2 子集两者均为 9/10。
5. FastWAM 的“浅层 hidden 直接接 action head”机制可以迁移并运行于第二种 VLA/VA 实现。
6. 在 Pure Dual-stream VDE 与 VDE+Internal 的 30 对 interleaved paired pilot 中，组合方案成功 29/30（Pure VDE 为 28/30），并额外降低约 4.90% Action DiT 延迟和 2.95% `infer_action` 延迟；两组均未出现 fallback 或路由异常。

### 当前不能写

1. 不能声称 LingBot-VA 已完成正式泛化验证。
2. 不能把 60% timing 与 40% success 拼成同一个配置的 speed-success 结果。
3. 不能把 60% 配置的 1.66x infer 加速归给 40% 配置；t040 的独立结果为约 1.31x。
4. 不能把 29/30 与 28/30 表述为成功率提升，也不能声称 Internal 不掉点、统计显著或统计非劣；现有 Internal-vs-Full 和 VDE 组合实验均只有 30 个 paired pilot 样本。
5. teacher collection 使用了 task0–9 的前两个 trial，当前 cross10 的 trial0 与训练数据重叠；现有 paired VDE 结果报告未证明其 seeds 与 teacher states 完全隔离，因此不得将该组结果描述为 unseen evaluation。
6. 不能将 LingBot 结果与 FastWAM 的毫秒数直接横向比较；模型规模、实现和运行路径不同。

## 7. 待补实验（非当前立即必跑）

### P0：增强论文中的跨模型证据

1. Internal-vs-Full 已完成同任务分布的 30-episode matched timing；若进入正式投稿，将未见配对从 trial2 扩展到 trial3–4，使未见子集达到 30 episodes/method；
2. Pure VDE 与 VDE+Internal 的 30 对 interleaved paired pilot 已完成；若将其升级为主结果，再扩大固定 seed 数量；
3. 对扩展实验报告 paired disagreement、成功率差置信区间和配对检验；在样本量与预设界限不足时仍不使用统计非劣表述；
4. 当前 trial2、Internal-vs-Full 30-episode 结果以及 VDE 组合 30 对结果均应明确标为 pilot。

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

> LingBot-VA 初步迁移验证表明，Action DiT layer-2 hidden 通过约 9.2 万参数的 direct head 可将 internal action step 加速约 10.7x。实际 39.2% internal 的 matched 30-episode pilot 获得约 1.53x Action DiT 和 1.31x infer wall 加速，成功率为 27/30（Full 29/30），而未见 trial2 子集两者均为 9/10；在另一组 30 对 interleaved pilot 中，VDE+Internal 相比 Pure Dual-stream VDE 额外降低约 4.90% Action DiT 和 2.95% `infer_action` 延迟，成功数为 29/30 与 28/30。结果支持 Internal 的跨实现迁移及其与 VDE 的可组合性，但尚未构成统计显著或正式非劣性证明。
