---
type: topic
title: "Latent Action：从无动作标签视频中发现可控变化"
source_papers: [genie, lingbot-va-v2]
status: stable
updated: 2026-07-19
---

# Latent Action：从无动作标签视频中发现可控变化

## 定义

latent action 是从相邻状态变化中推断出的低维转移变量。它让模型可以利用没有键盘、手柄或机器人控制标注的视频，但它首先描述“画面如何变化”，不保证与真实可执行动作一一对应。

## 两种代表路线

### Genie

Genie 的 Latent Action Model 从相邻视频帧推断离散 code，Dynamics Model 再用 code 条件化下一帧生成。用户推理时选择 code，像使用少量虚拟按键控制环境。

优点是可以扩展到大规模互联网视频；限制是 code 可能编码相机运动、物体变化或数据偏差，而不是稳定的真实动作语义。

### LingBot-VA 2.0

IDM 从相邻 semantic visual latent 推断 transition variable，FDM 以 transport map 与 residual 重建下一状态。瓶颈与 backward consistency 试图阻止 latent action 复制整幅画面。

这些 latent action 可用于视频预训练，但机器人部署仍需带真实 action 的数据和 embodiment-specific action head。

## 不可辨识性

只观察 $(x_t,x_{t+1})$ 通常无法唯一恢复真实动作：

- 多个动作可能产生相似视觉结果；
- 相同动作在不同状态中效果不同；
- 相机运动和主体运动可能混淆；
- 背景变化可能比动作对象更容易预测；
- 低维 bottleneck 的 code 可以任意置换，缺少天然语义。

因此 latent action 的价值不在于“恢复真实动作名称”，而在于形成可重复、可区分、可用于预测的控制接口。

## 应如何评价

- action distinctness：不同 code 是否产生可区分结果；
- consistency：同一 code 跨场景是否保持相似作用；
- state dependence：效果是否符合当前状态约束；
- real-action alignment：少量标注能否映射到真实动作；
- long-horizon controllability：作用是否随 rollout 衰减；
- causal localization：变化是否发生在合理对象与区域。

## 关联笔记

- [Genie](../papers/interactive-world-models/genie/genie-1.md)
- [LingBot-VA 2.0](../papers/embodied-world-models/lingbot-va/lingbot-va-v2.md)
- [Action controllability](action-controllability.md)
- [反事实动作评估设计](../research/action-controllability/evaluation-design.md)
