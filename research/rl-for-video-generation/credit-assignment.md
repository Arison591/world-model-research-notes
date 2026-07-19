---
type: research
title: "视频生成中的多层级信用分配"
status: investigating
updated: 2026-07-19
---

# 视频生成中的多层级信用分配

## 动机

图像 diffusion RL 已经面临“最终 reward 如何分给 denoising timestep”的问题。视频额外存在内容时间轴与动作轴，因此终局 reward 可能需要同时分配到：

1. diffusion timestep；
2. 视频帧或 temporal chunk；
3. 空间对象/区域；
4. action segment。

## 当前假设

单纯把 CoCA 的 latent similarity 沿 diffusion timestep 搬到视频，只处理生成时间，不处理哪一帧、哪个对象和哪个动作真正决定最终 reward。更合理的设计可能需要可分解或层级 credit：

$$
w_{d,t,r,a}\approx
w^{\text{diffusion}}_d
\cdot w^{\text{frame}}_t
\cdot w^{\text{region}}_r
\cdot w^{\text{action}}_a.
$$

这只是结构假设，未证明四个维度可以独立分解。

## 第一阶段可验证问题

- 相邻帧 reward 或 reward feature 是否高度相关；
- 哪些 diffusion stage 最影响运动结构，哪些主要影响纹理；
- 视觉 feature change 能否预测终局 reward change；
- action intervention 后，reward 变化集中在哪些帧与区域；
- 重新分配后的逐步 reward 是否保持原始总量；
- 权重是否跨 prompt、视频长度和 reward model 泛化。

## 候选实验

1. 在固定视频生成模型上保存所有 denoising intermediate；
2. 对 diffusion timestep、frame chunk 和 action segment 分别做遮蔽/替换；
3. 计算终局 reward、语义 feature、光流与对象轨迹变化；
4. 比较 uniform、CoCA-style、frame-aware 和 action-aware weighting；
5. 检查 reward query/sample efficiency，同时报告最终质量与多样性。

## 主要风险

- 中间 latent similarity 与人类偏好不一致；
- 多维权重联合后方差过大；
- reward model 对单帧敏感但忽略时序质量；
- 重新加权改变优化尺度，被误判为更好的 credit assignment；
- action 数据不足以识别动作的因果贡献。

## 关联笔记

- [CoCA](../../papers/generative-model-rl/coca/step-level-reward-for-free.md)
- [时序信用分配专题](../../topics/temporal-credit-assignment.md)
- [Diffusion 模型中的强化学习](../../topics/rl-for-diffusion-models.md)
