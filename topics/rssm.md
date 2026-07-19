---
type: topic
title: "RSSM：确定性记忆与随机状态的潜在动力学"
source_papers: [dreamer-v1, dreamer-v2, dreamer-v3]
status: stable
updated: 2026-07-19
---

# RSSM：确定性记忆与随机状态的潜在动力学

## 核心定义

Recurrent State-Space Model 同时维护：

- deterministic state $h_t$：由上一状态和动作递归更新，负责长期记忆；
- stochastic state $z_t$：表示当前不确定性和多模态状态；
- posterior $q(z_t\mid h_t,x_t)$：训练时看得到当前 observation；
- prior $p(z_t\mid h_t)$：imagination 时只能依赖历史。

典型更新可以概括为：

$$
h_t=f(h_{t-1},z_{t-1},a_{t-1}),\qquad
z_t\sim q(z_t\mid h_t,x_t).
$$

训练时 posterior 为 latent state 提供 observation grounding；真正想象未来时必须从 prior 采样，因此 prior 与 posterior 的对齐质量决定 rollout 是否可靠。

## 训练目标

RSSM 通常同时优化 observation reconstruction、reward prediction、continuation prediction 与 KL：

$$
\mathcal L_{\text{model}}
=\mathcal L_{\text{obs}}
+\mathcal L_{\text{reward}}
+\mathcal L_{\text{continue}}
+\beta D_{\mathrm{KL}}(q\Vert p).
$$

KL 同时承担两件事：训练 prior 预测 posterior，并限制 posterior 不要编码无法从历史预测的信息。若优化失衡，posterior 可能退化为容易预测但信息不足的表示，或 prior 长期追不上 posterior。

## Dreamer 三代中的变化

| 版本 | stochastic latent | KL/优化重点 |
|---|---|---|
| DreamerV1 | Gaussian | 常规变分状态空间模型 |
| DreamerV2 | categorical | KL balancing，增强 prior 学习；straight-through 传梯度 |
| DreamerV3 | categorical | KL balance + free bits + unimix，提高跨任务稳定性 |

V2 的经验结果支持 categorical latent 在 Atari 上更有效，但不能据此推出所有世界模型都应使用离散状态。环境多模态性、容量、优化方式和任务目标都可能是混杂因素。

## 为什么 deterministic 与 stochastic 都需要

- 只有 deterministic recurrent state：难以表示环境随机性与多种可能未来。
- 只有 stochastic state：缺少稳定的长期记忆和高效递归汇总。
- 两者结合：$h_t$ 汇总可预测历史，$z_t$ 表示当前随机因素与观测更新。

## 常见失败模式

- **Prior drift**：imagination 完全依赖 prior，误差会随 horizon 累积。
- **Posterior collapse**：随机状态没有承载有效信息。
- **过度重建**：latent 容量被背景纹理占据，忽略控制关键的小目标。
- **Model exploitation**：actor 主动寻找 world model 预测错误但回报高的状态。
- **表示错配**：对视觉重建友好的 latent 不一定最适合控制。

## 关联笔记

- [DreamerV1](../papers/model-based-rl/dreamer/dreamer-v1.md)
- [DreamerV2](../papers/model-based-rl/dreamer/dreamer-v2.md)
- [DreamerV3](../papers/model-based-rl/dreamer/dreamer-v3.md)
- [Dreamer 系列比较](../comparisons/dreamer-v1-v2-v3.md)
- [Latent imagination](latent-imagination.md)
