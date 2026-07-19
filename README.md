# World Model Research Notes

这是一个面向公开研究积累的中文知识库，记录 world model、model-based reinforcement learning、interactive generation、embodied world model，以及生成模型中的强化学习与信用分配。

仓库不把论文笔记当成彼此孤立的摘要，而是沿着下面的路径沉淀：

```text
单篇论文事实
  → 专题知识
  → 横向比较
  → 研究问题
  → 可执行实验
```

## 当前研究重点

1. Action-conditioned world models
2. Action controllability evaluation
3. Latent dynamics 与 latent action
4. 长时、实时的视频与具身世界模型
5. RL 与生成模型中的多层级信用分配

## 阅读地图

### Model-based RL 基础线

- [DreamerV1：Latent Imagination](papers/model-based-rl/dreamer/dreamer-v1.md)
- [DreamerV2：Discrete World Models](papers/model-based-rl/dreamer/dreamer-v2.md)
- [DreamerV3：Cross-domain Stability](papers/model-based-rl/dreamer/dreamer-v3.md)
- [Dreamer 三代比较](comparisons/dreamer-v1-v2-v3.md)
- [RSSM 专题](topics/rssm.md)
- [Latent Imagination 专题](topics/latent-imagination.md)

### Interactive world models

- [Genie：从无动作视频发现 latent action](papers/interactive-world-models/genie/genie-1.md)
- [LingBot-World 1.0](papers/interactive-world-models/lingbot-world/lingbot-world-v1.md)
- [LingBot-World 2.0](papers/interactive-world-models/lingbot-world/lingbot-world-v2.md)
- [Latent Action 专题](topics/latent-action.md)

### Embodied world models

- [LingBot-VA 1.0](papers/embodied-world-models/lingbot-va/lingbot-va-v1.md)
- [LingBot-VA 2.0](papers/embodied-world-models/lingbot-va/lingbot-va-v2.md)
- [LingBot-Map](papers/embodied-world-models/lingbot-map/lingbot-map.md)
- [LingBot 系列的三条演化路线](comparisons/lingbot-series.md)

### RL 与信用分配

- [强化学习基础](topics/reinforcement-learning-foundations.md)
- [时序信用分配综述](papers/surveys-and-benchmarks/temporal-credit-assignment-survey.md)
- [时序信用分配专题](topics/temporal-credit-assignment.md)
- [Diffusion 模型中的强化学习](topics/rl-for-diffusion-models.md)
- [CoCA：Step-level Reward](papers/generative-model-rl/coca/step-level-reward-for-free.md)

## 从阅读到研究

- [Research Idea Backlog](research/idea-backlog.md)
- [Action Controllability Evaluation](research/action-controllability/evaluation-design.md)
- [视频生成中的多层级信用分配](research/rl-for-video-generation/credit-assignment.md)
- [EnerVerse-AC 实现概览](implementations/enerverse-ac/overview.md)
- [EnerVerse-AC 源码深读](implementations/enerverse-ac/code-walkthrough.md)

## 仓库导航

- [论文索引](PAPER_INDEX.md)：由论文 YAML 元数据自动生成。
- [阅读与研究路线图](ROADMAP.md)：记录下一批论文和知识缺口。
- [维护规范](CONTRIBUTING.md)：新增笔记、图片和标签前请先阅读。
- `papers/`：一篇论文一个事实单元。
- `topics/`：跨论文沉淀知识。
- `comparisons/`：版本与方法关系。
- `research/`：可继续验证的问题与实验设计。
- `implementations/`：外部代码仓库的复现和源码阅读。

## 阅读状态

| 状态 | 含义 |
|---|---|
| `todo` | 尚未开始 |
| `reading` | 正在阅读 |
| `skimmed` | 已掌握摘要、方法和主要结果 |
| `finished` | 关键方法、公式和实验已整理 |
| `revisit` | 已读，但仍有重要问题待核对 |
| `implemented` | 已运行代码或完成相关实验 |

## 本地维护

```powershell
python -m pip install -r requirements-dev.txt
python scripts/kb.py build-index
python scripts/kb.py validate
python -m unittest discover -s tests
```

## 授权与素材

原创笔记采用 [CC BY-NC 4.0](LICENSE)，维护脚本采用 [MIT](LICENSE-CODE)。`assets/` 中来自论文、项目或第三方的图片、表格、PDF 和视频不包含在上述原创内容授权中，其权利归原作者；仓库保留来源说明，仅用于研究记录与评论。
