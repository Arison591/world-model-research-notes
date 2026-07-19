# 维护规范

## 新增一篇论文

1. 从 `templates/paper-note.md` 复制模板。
2. 将文件放入 `papers/<category>/<series>/<paper-slug>.md`；路径只使用小写英文、数字和连字符。
3. 填写完整 YAML；未知字段使用 `null`，不要猜测。
4. 从 `config/taxonomy.yml` 选择 category、status、confidence 和三层标签。
5. 图片放入 `assets/papers/<paper-slug>/`，使用描述内容的文件名。
6. 写完后更新至少一个 topic、comparison 或 research 链接。
7. 运行生成、校验和测试命令。

## 内容边界

单篇论文笔记回答“作者做了什么”，专题回答“这个知识点如何跨论文演化”，比较文档回答“方法之间为什么不同”，研究文档保存可证伪的问题和实验设计。

论文笔记必须明确包含：

- `论文明确内容`：方法、公式、实验、作者声明的结果与局限；
- `我的理解`：对贡献、取舍和适用范围的解释；
- `推测与问题`：未被论文证明、需要核验或实验验证的判断。

不要把 related work 的数字写成本论文结果，也不要把 causal attention 等同于因果机制。

## YAML 约定

论文必填字段由 `scripts/kb.py` 检查。`year` 表示论文首次公开年份；正式 venue 若跨年写在 `venue`。`read_date` 只在日期确定时填写。`main_idea` 必须是一句话，供索引使用。

标签只分三层：

- `task`：任务与应用域；
- `method`：方法、架构或训练技术；
- `problem`：论文处理的研究难点。

新增标签前先确认现有词表无法表达，再修改 `config/taxonomy.yml`。

## 图片与附件

- 使用 `overall-architecture.png`、`atari-results.png` 等语义化名称。
- Markdown 必须提供可理解的 alt text。
- 图片下说明来源论文及 Figure/Table 编号；无法确认编号时明确写“待核验”。
- 裁剪、重绘或拼接图要注明修改方式。
- 原始矢量文件放入该论文资产目录的 `source/`，并从正文或资产索引链接。
- 不把第三方素材视为仓库原创授权的一部分。

## 实现与复现记录

实现文档必须记录外部仓库 URL、测试 commit、运行环境和复现边界。若历史记录缺少 commit，使用 `null` 并明确提示，不能伪造版本。

## 维护命令

```powershell
python scripts/kb.py build-index
python scripts/kb.py build-index --check
python scripts/kb.py validate
python -m unittest discover -s tests
```

`PAPER_INDEX.md` 是生成文件，不手工编辑。外部 HTTP 链接不在 CI 中联网探测，以避免临时网络故障造成误报。
