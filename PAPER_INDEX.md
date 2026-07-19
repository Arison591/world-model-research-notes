# Paper Index

> 此文件由 `python scripts/kb.py build-index` 根据论文 YAML 自动生成，请勿手工编辑。

当前共有 **11** 篇论文笔记。

## Model-based Reinforcement Learning

| Paper | Year | Venue | Main idea | Status | Confidence |
|---|---:|---|---|---|---|
| [DreamerV1](papers/model-based-rl/dreamer/dreamer-v1.md) | 2019 | ICLR 2020 | 在学习到的 latent dynamics 中展开想象轨迹，并直接训练 actor 与 critic。 | `finished` | medium |
| [DreamerV2](papers/model-based-rl/dreamer/dreamer-v2.md) | 2020 | ICLR 2021 | 用 categorical latent、KL balancing 与混合梯度估计把 Dreamer 推进到 Atari。 | `finished` | medium |
| [DreamerV3](papers/model-based-rl/dreamer/dreamer-v3.md) | 2023 | — | 以 symlog、twohot、free bits 和 return normalization 统一跨领域训练尺度。 | `finished` | medium |

## Interactive World Models

| Paper | Year | Venue | Main idea | Status | Confidence |
|---|---:|---|---|---|---|
| [Genie](papers/interactive-world-models/genie/genie-1.md) | 2024 | — | 从无动作标签视频中学习离散 latent action，并据此逐帧生成可交互环境。 | `finished` | medium |
| [LingBot-World 1.0](papers/interactive-world-models/lingbot-world/lingbot-world-v1.md) | 2026 | — | 把开放域视频生成器通过长视频训练、动作适配、因果改造和蒸馏变成实时可控世界模型。 | `finished` | medium |
| [LingBot-World 2.0](papers/interactive-world-models/lingbot-world/lingbot-world-v2.md) | 2026 | — | 以原生 causal teacher、MoBA、少步蒸馏和 Director–Pilot 接口提升长时生成与交互能力。 | `finished` | medium |

## Embodied World Models

| Paper | Year | Venue | Main idea | Status | Confidence |
|---|---:|---|---|---|---|
| [LingBot-Map](papers/embodied-world-models/lingbot-map/lingbot-map.md) | 2026 | — | 用 anchor、局部窗口和压缩 trajectory memory 分工保存流式三维重建所需的历史。 | `finished` | medium |
| [LingBot-VA 1.0](papers/embodied-world-models/lingbot-va/lingbot-va-v1.md) | 2026 | — | 先预测任务期望的未来视觉 latent，再以 inverse dynamics 解出动作，并用真实观察持续校正。 | `finished` | medium |
| [LingBot-VA 2.0](papers/embodied-world-models/lingbot-va/lingbot-va-v2.md) | 2026 | — | 围绕语义 visual-action 表示、原生 causal 预训练、MCP 与闭环部署重做机器人世界模型。 | `finished` | medium |

## RL for Generative Models

| Paper | Year | Venue | Main idea | Status | Confidence |
|---|---:|---|---|---|---|
| [CoCA](papers/generative-model-rl/coca/step-level-reward-for-free.md) | 2025 | — | 用相邻去噪状态对最终图像的相似度变化，为 diffusion trajectory 分配逐步训练权重。 | `revisit` | medium |

## Surveys & Benchmarks

| Paper | Year | Venue | Main idea | Status | Confidence |
|---|---:|---|---|---|---|
| [TCA Survey](papers/surveys-and-benchmarks/temporal-credit-assignment-survey.md) | 2023 | — | 用深度、密度和广度刻画时序信用分配，并统一比较时间邻近与回报分解方法。 | `finished` | medium |
