import tempfile
import unittest
from pathlib import Path

import yaml

from scripts import kb


TAXONOMY = {
    "categories": ["model-based-rl"],
    "paper_statuses": ["finished"],
    "document_statuses": ["stable"],
    "confidence_levels": ["medium"],
    "tags": ["world-model", "rssm"],
}


def paper_text(extra_body="", tag="rssm", include_main_idea=True):
    main_idea = 'main_idea: "Test idea"\n' if include_main_idea else ""
    return f"""---
type: paper
title: "Test Paper"
short_name: Test
authors: [A. Author]
year: 2026
venue: null
paper_url: "https://example.com/paper"
code_url: null
project_url: null
source_urls: []
category: model-based-rl
series: dreamer
tags: [world-model, {tag}]
status: finished
confidence: medium
read_date: null
updated: null
{main_idea}---

# Test Paper

{extra_body}
"""


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "config").mkdir()
        (self.root / "config" / "taxonomy.yml").write_text(
            yaml.safe_dump(TAXONOMY, sort_keys=False), encoding="utf-8"
        )
        self.paper = self.root / "papers" / "model-based-rl" / "dreamer" / "test-paper.md"
        self.paper.parent.mkdir(parents=True)
        self.paper.write_text(paper_text(), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_repository_and_nullable_dates(self):
        self.assertEqual(kb.validate_repository(self.root), [])

    def test_index_contains_paper_link(self):
        index = kb.render_index(self.root)
        self.assertIn("[Test](papers/model-based-rl/dreamer/test-paper.md)", index)
        self.assertIn("当前共有 **1** 篇论文笔记", index)

    def test_missing_required_metadata_is_reported(self):
        self.paper.write_text(paper_text(include_main_idea=False), encoding="utf-8")
        errors = kb.validate_repository(self.root)
        self.assertTrue(any("缺少 metadata 字段 main_idea" in error for error in errors))

    def test_unknown_flat_tag_is_reported(self):
        self.paper.write_text(paper_text(tag="unknown-method"), encoding="utf-8")
        errors = kb.validate_repository(self.root)
        self.assertTrue(any("未登记标签" in error for error in errors))

    def test_grouped_tags_are_rejected(self):
        text = paper_text().replace(
            "tags: [world-model, rssm]", "tags:\n  method: [rssm]"
        )
        self.paper.write_text(text, encoding="utf-8")
        errors = kb.validate_repository(self.root)
        self.assertTrue(any("tags 必须是扁平列表" in error for error in errors))

    def test_broken_relative_link_is_reported(self):
        self.paper.write_text(paper_text("[missing](missing.md)"), encoding="utf-8")
        errors = kb.validate_repository(self.root)
        self.assertTrue(any("断开的相对链接" in error for error in errors))

    def test_orphan_asset_is_reported(self):
        asset = self.root / "assets" / "papers" / "test" / "figure.png"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"not-a-real-png")
        errors = kb.validate_repository(self.root)
        self.assertTrue(any("孤立资产" in error for error in errors))

    def test_overdeep_content_path_is_reported(self):
        deep = self.root / "papers" / "model-based-rl" / "dreamer" / "extra" / "deep.md"
        deep.parent.mkdir(parents=True)
        deep.write_text(paper_text().replace("short_name: Test", "short_name: Deep"), encoding="utf-8")
        errors = kb.validate_repository(self.root)
        self.assertTrue(any("正文目录层级超过三层" in error for error in errors))

    def test_source_mapping_directory_is_rejected(self):
        (self.root / "sources").mkdir()
        errors = kb.validate_repository(self.root)
        self.assertTrue(any("不应保留来源分类" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
