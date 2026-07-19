---
type: research
title: "Action Controllability Evaluation Design"
status: experiment-ready
updated: 2026-07-19
---

# Action Controllability Evaluation Design

## 研究问题

给定同一个初始 observation、文本条件和生成随机性，action 的改变是否会稳定、单调、语义正确地改变 world model 的未来？

## 干预矩阵

对每条真实 action sequence $a_{1:T}$ 构造：

| 干预 | 定义 | 检查目标 |
|---|---|---|
| original | 原动作 | 预测质量基线 |
| zero | 全零或 no-op | 模型是否忽略 action |
| reversed | 方向或时间反转 | 运动方向是否反转 |
| scaled | $0.5a, 1.5a$ 等 | 效果是否近似单调 |
| shuffled | 打乱时间顺序 | 模型是否利用 action timing |
| random | 同分布随机动作 | action distinctness |

所有对照必须固定初始 observation、文本、采样噪声、scheduler 和其他 condition。

## 指标

### 动作响应

- optical-flow direction 与 action direction 的一致率；
- 相机轨迹、物体质心和关键点位移；
- action scale 与运动幅度的 rank correlation；
- 不同 action 结果之间的 feature distance；
- 受影响区域与目标对象 mask 的 overlap。

### 长时稳定性

- action effect 随 rollout horizon 的衰减；
- 连续干预后 drift 与 failure time；
- 从错误 action 切回真实 action 后的恢复能力。

### 质量—控制权衡

同时记录真实 action 下的 reconstruction/prediction error、感知质量和时序一致性，避免模型通过制造噪声获得更高的 action sensitivity。

## 不同模型的附加检查

- **Genie/latent action：**同一 code 跨场景的语义一致性、少量真实 action 对齐。
- **LingBot-World/EnerVerse-AC：**相机、机器人动作和文本事件分别消融，检查多路 condition 是否互相替代。
- **LingBot-VA：**干预 predicted future latent，检查变化是否传给 action decoder，而不是 action branch 绕过 future prediction。

## 最小实验

1. 选择 20–50 个短序列 case；
2. 每个 case 运行六类 action 干预；
3. 每种条件使用至少三个固定随机种子；
4. 报告方向一致率、scale correlation、target-region overlap 和预测质量；
5. 保存完整配置、action 文件和生成视频，允许复现。

## 失败判据

- zero/reversed action 与 original 几乎无差别：模型忽略 action；
- action 变化只造成纹理或背景噪声：伪 controllability；
- 短期响应正确但很快衰减：long-horizon conditioning failure；
- 控制更强但真实 action 预测质量大幅下降：质量—控制失衡。

## 关联笔记

- [Action controllability 专题](../../topics/action-controllability.md)
- [Action conditioning 专题](../../topics/action-conditioning.md)
- [LingBot 系列比较](../../comparisons/lingbot-series.md)
- [EnerVerse-AC 实现概览](../../implementations/enerverse-ac/overview.md)
