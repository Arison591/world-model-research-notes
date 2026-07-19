---
type: topic
title: "Action Controllability：世界模型是否真的听动作"
source_papers: [genie, lingbot-world-v1, lingbot-world-v2, lingbot-va-v1, lingbot-va-v2]
status: stable
updated: 2026-07-19
---

# Action Controllability：世界模型是否真的听动作

## 问题定义

动作被送入模型，不代表动作真正决定了未来。action controllability 关心：固定其他条件时，改变 action 是否会稳定、方向正确、幅度合理地改变生成结果。

需要同时区分：

- sensitivity：动作变化是否让输出变化；
- correctness：变化方向是否符合动作语义；
- distinctness：不同动作是否可区分；
- consistency：同一动作能否跨状态保持稳定含义；
- sufficiency：模型是否主要依据动作，而非偷偷依赖其他条件；
- fidelity：提高动作敏感性时是否牺牲整体预测质量。

## 为什么常见指标不够

- 图像 PSNR 变化只能说明输出不同，不能说明变化有意义；
- 光流幅度可能来自背景抖动，而不是动作对象；
- 任务成功率同时受 planner、视觉预测和 action decoder 影响；
- 定性视频容易挑选成功案例；
- 画质指标主要评价生成质量，不评价动作遵从。

## 建议的评估维度

| 维度 | 例子 |
|---|---|
| 局部因果作用 | 受动作影响的对象/区域是否发生变化 |
| 方向与幅度 | reverse/scale action 是否产生相反/单调变化 |
| 跨场景一致性 | 相同 action 在不同初始状态中的作用是否可解释 |
| 长时保持 | 连续 rollout 后 action effect 是否衰减 |
| 真实动作对齐 | latent action 能否以少量标注映射到真实 action |
| 质量—控制权衡 | controllability 提升是否伴随预测误差或画质恶化 |

## 与因果性的关系

causal attention mask 只限制模型不能看未来；它不保证动作干预具有正确的反事实结果。检验 action causality 需要主动改变动作并观察输出，而不是只在观测数据上计算相关性。

## 关联笔记

- [反事实动作评估设计](../research/action-controllability-evaluation.md)
- [Latent action](latent-action.md)
- [Action conditioning](action-conditioning.md)
- [LingBot 系列比较](../series/lingbot-series.md)
