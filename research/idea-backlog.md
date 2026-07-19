---
type: research
title: "Research Idea Backlog"
status: active
updated: 2026-07-19
---

# Research Idea Backlog

这里保存从论文阅读中抽出的、尚未进入正式实验的研究问题。状态使用 `seed / investigating / experiment-ready / parked`。

## IDEA-001：反事实动作可控性评估

- **状态：**experiment-ready
- **来源：**Genie、LingBot-World、LingBot-VA、EnerVerse-AC
- **问题：**固定初始状态和随机性后，改变 action 是否稳定、方向正确地改变未来？
- **下一步：**执行 original / zero / reversed / scaled / shuffled action 干预。
- **详细设计：**[Action Controllability Evaluation](action-controllability/evaluation-design.md)

## IDEA-002：视频生成中的多层级信用分配

- **状态：**investigating
- **来源：**CoCA、DDPO、时序信用分配综述
- **问题：**终局视频 reward 应如何分配到 diffusion timestep、视频帧、空间区域和动作片段？
- **下一步：**先验证相邻帧 reward correlation 与关键 denoising stage。
- **详细设计：**[Credit Assignment for Video Generation](rl-for-video-generation/credit-assignment.md)

## IDEA-003：无动作视频预训练与少量真实动作 grounding

- **状态：**seed
- **来源：**Genie、DreamerV3、LingBot-VA 2.0
- **问题：**能否先用无动作视频学习 dynamics/latent action，再以少量标注动作建立可执行控制接口？
- **风险：**latent action 的不可辨识性可能让映射依赖场景或 embodiment。

## IDEA-004：显式空间记忆约束生成式 world model

- **状态：**seed
- **来源：**LingBot-Map 与 LingBot-World
- **问题：**能否用 anchor、trajectory memory 或显式地图约束长时视频 rollout 中的地点身份和尺度？
- **风险：**几何状态与生成 latent 的接口、动态物体更新和 loop closure 尚不明确。

## IDEA-005：控制相关的 representation objective

- **状态：**seed
- **来源：**DreamerV2 的 Video Pinball failure、LingBot-VA 2.0 tokenizer
- **问题：**如何避免 pixel reconstruction 被大面积背景支配，并让 latent 更关注决定动作与回报的小目标？
- **候选方向：**object-centric weighting、task-aware prediction、long-horizon auxiliary objective。
