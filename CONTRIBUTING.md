# 维护规范

## 新增一篇论文

1. 从 `templates/paper-note.md` 复制模板。
2. 将文件放入 `papers/<category>/<paper-slug>.md`；系列较多时可以增加一级系列目录，例如 `papers/model-based-rl/dreamer/`。
3. 填写 YAML；未知日期或链接使用 `null`，不要猜测。
4. 从 `config/taxonomy.yml` 选择稳定的 `category`、`status`、`confidence` 和 tags。
5. 图片放入 `assets/papers/<paper-slug>/`，使用能说明内容的英文文件名。
6. 保证单篇笔记能够独立读懂。只有存在真正的跨论文长期主题时，才新增或更新 `topics/`、`series/`。
7. 生成索引并运行完整校验。

## 内容组织原则

- `papers/` 保存完整论文阅读，包含方法、公式、实验、局限和个人判断。
- `series/` 比较同一系列或明确相关的方法演化。
- `topics/` 汇总跨多篇论文反复出现的概念，允许与单篇笔记存在必要重叠。
- `research/` 集中保存想法和已具体化的研究设计，不再拆分为 questions、protocols、proposals。
- `implementations/` 保存代码阅读和复现边界；是否运行成功在对应实现文档中说明。

飞书、Notion、ChatGPT 和本地 Markdown 只是来源，不是分类。原始链接写入对应文档的 `source_urls`，不要新建来源目录或来源映射表。

## 论文 YAML

```yaml
---
type: paper
title: ""
short_name: ""
authors: []
year: null
venue: null

paper_url: ""
code_url: null
project_url: null
source_urls: []

category: ""
series: null
tags: []

status: reading
confidence: low
read_date: null
updated: null

main_idea: ""
---
```

论文状态仅使用 `todo / reading / skimmed / finished / revisit / implemented`。跨领域关系使用扁平 tags 表达；新增标签前先确认现有词表无法覆盖，再修改 `config/taxonomy.yml`。

## 正文与证据边界

不为统一格式重写已有分析。新增笔记建议明确区分：

- `论文明确内容`：作者报告的方法、公式、实验、结果与局限；
- `我的理解`：对贡献、取舍和适用范围的解释；
- `推测与问题`：尚未被论文证明、需要核验或实验验证的判断。

不要把 related work 的数字写成本论文结果，也不要把 causal attention 等同于因果机制。

## 图片与附件

- 使用 `overall-architecture.png`、`atari-results.png` 等语义化名称。
- Markdown 提供可理解的 alt text，并尽量注明 Figure/Table 编号与来源。
- 原始矢量文件可以放入该论文资产目录的 `source/`，但必须从 Markdown 链接。
- 移动附件时核对二进制哈希，不修改或丢弃历史图片、PDF、SVG 和视频。
- 第三方素材不属于仓库原创内容授权。

## 维护命令

```powershell
python scripts/kb.py build-index
python scripts/kb.py build-index --check
python scripts/kb.py validate
python -m unittest discover -s tests
```

`PAPER_INDEX.md` 是生成文件，不手工编辑。CI 不联网探测外部 URL，避免网络波动造成误报。
