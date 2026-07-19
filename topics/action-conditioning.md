---
type: topic
title: "Action Conditioning：动作如何进入世界模型"
source_papers: [dreamer-v1, genie, lingbot-world-v1, lingbot-world-v2, lingbot-va-v1, lingbot-va-v2]
status: stable
updated: 2026-07-19
---

# Action Conditioning：动作如何进入世界模型

## 三种不同问题

“action-conditioned world model”至少覆盖三类不同设定：

1. **已知真实 action**：Dreamer 根据环境动作预测 latent transition；
2. **从视频发现 latent action**：Genie 先推断控制 code，再预测下一帧；
3. **生成未来并反推动作**：LingBot-VA 先想象任务期望的未来，再由 inverse dynamics 输出机器人动作。

不能因为都出现 action token，就假设它们解决的是同一个控制问题。

## 常见注入方式

| 模型 | 动作表示 | 注入位置 | 作用 |
|---|---|---|---|
| Dreamer | 环境 action vector | RSSM recurrent transition | 决定下一 latent state |
| Genie | 离散 latent action code | Dynamics Transformer | 条件化下一帧 token |
| LingBot-World | Plücker + multi-hot / prompt | AdaLN 与 DiT conditioning | 控制相机、键鼠与语义事件 |
| LingBot-VA | action token stream | video-action shared attention | 生成 action chunk 并维护历史 |
| EnerVerse-AC | action vector、trajectory map、ray map | cross-attention 与 latent concat | 为扩散 UNet 提供多路动作/几何条件 |

## 设计取舍

- **直接拼接**实现简单，但高维空间条件可能难以压缩；
- **AdaLN/FiLM**适合全局或逐层调制；
- **cross-attention**适合让视觉 token 按需读取动作 token；
- **空间 condition map**能保留 ray、trajectory 等像素对齐结构；
- **双流 Transformer**保留视频与动作各自容量，但系统更复杂。

## 关键风险

- 模型可能忽略动作，只根据历史视觉预测；
- 动作数据较少时，全量微调会破坏生成先验；
- 多路条件可能互相替代，难以知道性能来自哪一路；
- 训练中看到真实历史，推理中看到生成历史，导致动作作用衰减；
- 视觉变化与物理动作之间可能不可辨识。

## 关联笔记

- [Latent action](latent-action.md)
- [Action controllability](action-controllability.md)
- [EnerVerse-AC 实现概览](../implementations/enerverse-ac/overview.md)
- [LingBot 系列比较](../comparisons/lingbot-series.md)
