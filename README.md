# World Model Research Notes

一个记录 World Model 领域论文阅读、系列演化、核心知识、代码复现与研究思考的中文仓库。目前重点关注生成式 world model、action-conditioned world model、embodied world model 和 world model evaluation。

这里首先保存能够独立阅读的单篇论文笔记；`topics/` 与 `series/` 在确有跨论文价值时进一步串联知识，不会为了目录结构把原笔记拆空。

## 阅读路线

### 1. Model-based RL 与控制

- [DreamerV1：Latent Imagination](papers/model-based-rl/dreamer/dreamer-v1.md)
- [DreamerV2：Discrete World Models](papers/model-based-rl/dreamer/dreamer-v2.md)
- [DreamerV3：Cross-domain Stability](papers/model-based-rl/dreamer/dreamer-v3.md)
- [Dreamer 三代比较](series/dreamer-series.md)
- [RSSM](topics/rssm.md)
- [Latent Imagination](topics/latent-imagination.md)

### 2. 生成式与交互式 World Model

- [Genie：从无动作视频发现 latent action](papers/generative-world-models/genie.md)
- [LingBot-World 1.0](papers/generative-world-models/lingbot-world-v1.md)
- [LingBot-World 2.0](papers/generative-world-models/lingbot-world-v2.md)
- [Latent Action](topics/latent-action.md)
- [Action Conditioning](topics/action-conditioning.md)

### 3. 具身与空间 World Model

- [LingBot-VA 1.0](papers/embodied-world-models/lingbot-va-v1.md)
- [LingBot-VA 2.0](papers/embodied-world-models/lingbot-va-v2.md)
- [LingBot-Map](papers/spatial-world-models/lingbot-map.md)
- [LingBot 系列总结](series/lingbot-series.md)

### 4. RL、Diffusion 与相关方法

- [强化学习基础](topics/reinforcement-learning-foundations.md)
- [时序信用分配综述](papers/surveys/temporal-credit-assignment-survey.md)
- [时序信用分配专题](topics/temporal-credit-assignment.md)
- [Diffusion 模型中的强化学习](topics/rl-for-diffusion-models.md)
- [CoCA：Step-level Reward](papers/related-methods/step-level-reward-for-free.md)

## 论文系列

- [Dreamer V1/V2/V3](series/dreamer-series.md)：RSSM、latent imagination、离散表征与跨任务稳定性的连续演化。
- [LingBot World / VA / Map](series/lingbot-series.md)：生成式交互、机器人控制和流式三维重建三条互补路线。

完整书目、状态和一句话贡献见自动生成的 [Paper Index](PAPER_INDEX.md)。后续计划阅读 PlaNet、DIAMOND、OASIS、DDPO、TDPO 等工作，详见 [Roadmap](ROADMAP.md)。

## 专题知识

- 动力学与行为学习：[RSSM](topics/rssm.md)、[Latent Imagination](topics/latent-imagination.md)
- 动作表示与控制：[Latent Action](topics/latent-action.md)、[Action Conditioning](topics/action-conditioning.md)、[Action Controllability](topics/action-controllability.md)
- 强化学习与生成模型：[RL Foundations](topics/reinforcement-learning-foundations.md)、[Temporal Credit Assignment](topics/temporal-credit-assignment.md)、[RL for Diffusion](topics/rl-for-diffusion-models.md)

专题允许与单篇笔记存在必要重叠：单篇笔记负责完整解释论文，专题负责比较多篇工作的共性、差异和未解决问题。

## 代码阅读与复现

- [EnerVerse-AC 仓库概览](implementations/enerverse-ac/overview.md)
- [EnerVerse-AC 源码深读](implementations/enerverse-ac/code-walkthrough.md)

实现文档独立记录测试范围、上游仓库与版本边界，不作为论文笔记或复现成功声明。

## 当前研究思考

- [想法池](research/ideas.md)
- [Action Controllability Evaluation](research/action-controllability-evaluation.md)
- [视频生成中的多层级信用分配](research/video-generation-credit-assignment.md)

Action controllability 是当前重点研究问题之一，但不是仓库唯一的组织轴。研究文档可以在同一文件中包含问题、假设、方法、指标和实验计划，只有想法足够具体后才独立成文。

## 仓库结构

```text
papers/           单篇论文笔记，按少量稳定类别组织
series/           论文系列演化与横向比较
topics/           跨论文长期专题
research/         想法、研究问题和实验设计
implementations/  外部代码仓库阅读与复现记录
assets/           与正文对应的图片、矢量文件和视频
templates/        论文、系列、专题和实现模板
config/           分类与标签词表
scripts/          索引生成与仓库校验
tests/            维护脚本测试
```

仓库不按飞书、Notion 或本地 Markdown 等来源分类。原始资料链接直接记录在对应文档的 `source_urls` 中。

## 新增与维护

新增论文时从 [论文模板](templates/paper-note.md) 开始，优先保证笔记自身完整；只有形成跨多篇论文的长期认识时，再更新专题或系列文档。详细规则见 [Contributing](CONTRIBUTING.md)。

```powershell
python -m pip install -r requirements-dev.txt
python scripts/kb.py build-index
python scripts/kb.py validate
python -m unittest discover -s tests
```

## 授权与素材

原创笔记采用 [CC BY-NC 4.0](LICENSE)，维护脚本采用 [MIT](LICENSE-CODE)。`assets/` 中来自论文、项目或第三方的图片、表格、PDF 和视频不包含在上述原创内容授权中，其权利归原作者；仓库保留来源说明，仅用于研究记录与评论。
