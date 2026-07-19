---
type: topic
title: "Latent Imagination：在世界模型中训练行为"
source_papers: [dreamer-v1, dreamer-v2, dreamer-v3]
status: stable
updated: 2026-07-19
---

# Latent Imagination：在世界模型中训练行为

## 它解决什么问题

学习到 world model 后，agent 仍需决定怎样利用预测未来。PlaNet 类方法在每个真实 step 在线搜索动作序列；Dreamer 则把搜索替换为 actor，在 latent rollout 中反复训练 actor 与 critic，真实交互时直接执行 actor。

## 标准流程

```text
真实 replay
  → posterior 编码真实状态
  → 从 posterior state 选 imagination 起点
  → prior + actor 展开 latent trajectory
  → reward/continuation model 预测回报
  → critic 构造 λ-return
  → 更新 actor 与 critic
  → actor 回到真实环境收集新数据
```

训练 actor 时通常不解码像素，只 rollout latent state、reward 与 continuation。observation decoder 主要负责 representation learning 与可视化。

## 为什么需要 critic

有限 imagination horizon 只能显式计算近期 reward。critic 用 bootstrap 补上 horizon 之外的价值，$\lambda$-return 在多步真实预测和 value bootstrap 之间折中：

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}G_t^{(n)}
+\lambda^{H-1}G_t^{(H)}.
$$

## Actor 的梯度来源

- dynamics gradient：梯度穿过可微 world model，方差低，但会直接继承模型偏差；
- REINFORCE：不要求 transition 可微，适合离散 action，但方差更高；
- mixed estimator：DreamerV2 混合两者；DreamerV3 进一步简化和稳定化。

## 主要风险

1. actor 会寻找预测回报高的模型漏洞；
2. prior rollout 越长，compounding error 越严重；
3. reconstruction representation 可能忽略控制相关的小目标；
4. critic target 本身有偏时，会把误差传播给 actor；
5. 统一 return scale 对跨任务训练非常重要。

## 关联笔记

- [RSSM](rssm.md)
- [Dreamer 系列比较](../series/dreamer-series.md)
- [强化学习基础](reinforcement-learning-foundations.md)
