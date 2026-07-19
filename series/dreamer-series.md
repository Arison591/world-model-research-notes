---
type: series
title: "Dreamer V1/V2/V3：从 latent imagination 到统一跨域训练"
papers: [dreamer-v1, dreamer-v2, dreamer-v3]
status: stable
updated: 2026-07-19
---

# Dreamer V1/V2/V3：从 latent imagination 到统一跨域训练

## 一句话结论

三代 Dreamer 的主骨架没有改变：从真实经验训练 RSSM，在 latent imagination 中训练 actor/critic，再回到真实环境收集数据。演化重点依次是“如何从模型学行为”“如何适配离散且突变的 Atari”“如何统一不同任务的数值尺度和超参数”。

## 核心差异

| 维度 | DreamerV1 | DreamerV2 | DreamerV3 |
|---|---|---|---|
| 主要目标 | 用 imagination 训练策略 | 在 Atari 上掌握离散任务 | 一套配置覆盖多类 domain |
| stochastic latent | Gaussian | categorical | categorical + unimix |
| actor gradient | dynamics / reinforce | 两者混合 | 简化、归一化后的 actor objective |
| KL 处理 | 常规 KL | KL balancing | KL balancing + free bits |
| reward/value 表示 | 常规回归 | 常规建模 | symlog + twohot + symexp |
| return 尺度 | 任务相关 | 任务相关 | percentile normalization |
| 核心贡献 | latent imagination | 离散表示与 prior 优化 | 跨域稳定训练 recipe |

## 保持不变的骨架

1. replay 中的真实 observation/action 训练 world model；
2. posterior state 作为 imagination 起点；
3. prior 在 latent space 展开未来轨迹；
4. reward、continuation 和 critic 构成 value target；
5. actor 只在真实环境执行，训练主要发生在 imagined trajectories。

这意味着 V2/V3 不是替换 Dreamer 范式，而是在同一范式中调整表示、梯度分配和数值稳定性。

## V1 → V2：表示与梯度

V1 已经解决“如何在世界模型里学习行为”，但 Atari 的状态变化和动作空间更离散，prior 质量也更容易成为瓶颈。V2 的 categorical latent、straight-through estimator、KL balancing 和混合 actor gradient 共同处理这些问题。

最关键的启发不是“离散 latent 永远更好”，而是 world-model prior、representation 和 policy gradient 必须共同服务 imagination。

## V2 → V3：统一数值尺度

V2 已能处理 Atari，但 reward scale、observation scale、entropy coefficient 和 value target 仍需随任务调整。V3 保留 categorical RSSM，增加：

- symlog 压缩有符号大数；
- twohot 将连续 target 转为分布预测；
- free bits 避免过度 KL regularization；
- unimix 避免 categorical 过度确定；
- percentile return normalization 统一 actor 更新尺度。

V3 的贡献更接近系统化的 robust optimization，而不是新的世界模型结构。

## 对当前研究的启发

- Dreamer 只需 latent dynamics 支持控制，不要求生成可观看的视频；不要把它与视频世界模型的目标混为一谈。
- reconstruction 提供密集监督，但可能忽略对决策关键的小目标。
- actor 会利用 world model 漏洞，因此短 imagination、posterior 起点和真实数据闭环只能缓解、不能消除 model exploitation。
- Genie 的无标签 latent action 与 Dreamer 的 action-conditioned RSSM 可能在“无动作视频预训练 + 少量动作 grounding”处汇合。

## 单篇与专题

- [DreamerV1](../papers/model-based-rl/dreamer/dreamer-v1.md)
- [DreamerV2](../papers/model-based-rl/dreamer/dreamer-v2.md)
- [DreamerV3](../papers/model-based-rl/dreamer/dreamer-v3.md)
- [RSSM](../topics/rssm.md)
- [Latent imagination](../topics/latent-imagination.md)
