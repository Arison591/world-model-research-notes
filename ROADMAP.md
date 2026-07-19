# Reading & Research Roadmap

路线图只记录真实的阅读候选与知识缺口，不为尚未阅读的论文创建空笔记。

## 下一批基础论文

- [ ] **PlaNet**：补齐 RSSM、latent planning 与 DreamerV1 的前置关系。
- [ ] **DIAMOND**：比较 diffusion world model 与 RSSM 在 Atari 环境建模中的差异。
- [ ] **OASIS**：补充开放世界交互式视频生成路线，并与 Genie、LingBot-World 比较。
- [ ] **DDPO**：补完整 diffusion-as-MDP 和 policy-gradient 推导。
- [ ] **TDPO**：理解带 timestep critic 的 diffusion credit assignment。

## 待补专题

- [ ] World model evaluation：prediction、control、planning 与 generative quality 的指标边界。
- [ ] Discrete vs. continuous latent：表示容量、优化、multimodality 与任务依赖。
- [ ] Long-horizon memory：KV cache、显式地图、对象身份和 loop closure。
- [ ] Video tokenization：重建质量、语义表示和 dynamics prediction 的取舍。

## 比较计划

- [ ] Dreamer vs. TD-MPC：policy learning in imagination 与 latent planning。
- [ ] RSSM vs. Transformer dynamics：递归状态与长上下文 token memory。
- [ ] Genie vs. action-labeled world models：latent action 的数据规模与 grounding 成本。
- [ ] Diffusion vs. autoregressive video world models：速度、漂移和 action conditioning。

## 研究推进顺序

1. 在 EnerVerse-AC 或可运行的小型 ACWM 上完成 [反事实动作评估](research/action-controllability-evaluation.md)。
2. 将评估扩展到 long-horizon rollout，检查 action effect 衰减。
3. 复现 CoCA 或等价 timestep reweighting，确认奖励总量问题。
4. 在小型视频生成任务上验证 frame-aware 与 action-aware credit signal。
5. 只有实验开始后再建立 `projects/<project-slug>/`。
