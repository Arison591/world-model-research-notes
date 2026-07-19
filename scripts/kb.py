#!/usr/bin/env python3
"""Build and validate the World Model Research Notes knowledge base."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIRS = ("papers", "topics", "comparisons", "research", "implementations")
PAPER_REQUIRED = (
    "type",
    "title",
    "short_name",
    "authors",
    "year",
    "venue",
    "arxiv_id",
    "paper_url",
    "code_url",
    "project_url",
    "category",
    "tags",
    "status",
    "read_date",
    "updated",
    "confidence",
    "main_idea",
)
OTHER_REQUIRED = {
    "topic": ("type", "title", "source_papers", "status", "updated"),
    "comparison": ("type", "title", "papers", "status", "updated"),
    "research": ("type", "title", "status", "updated"),
    "implementation": (
        "type",
        "title",
        "repository_url",
        "tested_commit",
        "status",
        "updated",
    ),
}
KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class Document:
    path: Path
    metadata: Dict[str, Any]
    body: str


def load_taxonomy(root: Path) -> Dict[str, Any]:
    path = root / "config" / "taxonomy.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def split_frontmatter(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("frontmatter 缺少结束分隔符 ---") from exc
    metadata = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(metadata, dict):
        raise ValueError("frontmatter 必须是 YAML mapping")
    return metadata, "\n".join(lines[end + 1 :])


def read_document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    metadata, body = split_frontmatter(text)
    return Document(path=path, metadata=metadata or {}, body=body)


def paper_documents(root: Path) -> List[Document]:
    return [read_document(path) for path in sorted((root / "papers").rglob("*.md"))]


def markdown_documents(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.md")):
        if ".git" not in path.parts:
            yield path


def strip_fenced_code(text: str) -> str:
    result: List[str] = []
    in_fence = False
    fence = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence = stripped[:3]
            continue
        if in_fence and stripped.startswith(fence):
            in_fence = False
            continue
        if not in_fence:
            result.append(line)
    return "\n".join(result)


def scalar_date(value: Any) -> bool:
    return value is not None and bool(DATE.fullmatch(str(value)))


def is_url_or_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.startswith(("https://", "http://")))


def validate_metadata(doc: Document, root: Path, taxonomy: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    rel = doc.path.relative_to(root).as_posix()
    metadata = doc.metadata
    doc_type = metadata.get("type")
    if doc_type == "paper":
        required = PAPER_REQUIRED
    else:
        required = OTHER_REQUIRED.get(str(doc_type), ())
        if not required:
            return [f"{rel}: 未知或缺失的 type: {doc_type!r}"]
    for key in required:
        if key not in metadata:
            errors.append(f"{rel}: 缺少 metadata 字段 {key}")

    if doc_type == "paper":
        category = metadata.get("category")
        if category not in taxonomy["categories"]:
            errors.append(f"{rel}: 非法 category {category!r}")
        parts = doc.path.relative_to(root / "papers").parts
        if parts and category != parts[0]:
            errors.append(f"{rel}: category 与目录 {parts[0]!r} 不一致")
        if metadata.get("status") not in taxonomy["paper_statuses"]:
            errors.append(f"{rel}: 非法论文 status {metadata.get('status')!r}")
        if metadata.get("confidence") not in taxonomy["confidence_levels"]:
            errors.append(f"{rel}: 非法 confidence {metadata.get('confidence')!r}")
        if not isinstance(metadata.get("authors"), list) or not metadata.get("authors"):
            errors.append(f"{rel}: authors 必须是非空列表")
        if not isinstance(metadata.get("year"), int):
            errors.append(f"{rel}: year 必须是整数")
        arxiv_id = metadata.get("arxiv_id")
        if arxiv_id is not None and not ARXIV_ID.fullmatch(str(arxiv_id)):
            errors.append(f"{rel}: arxiv_id 格式错误 {arxiv_id!r}")
        for field in ("paper_url", "code_url", "project_url"):
            if not is_url_or_null(metadata.get(field)):
                errors.append(f"{rel}: {field} 必须是 http(s) URL 或 null")
        if metadata.get("read_date") is not None and not scalar_date(metadata.get("read_date")):
            errors.append(f"{rel}: read_date 必须是 YYYY-MM-DD 或 null")
        if not isinstance(metadata.get("main_idea"), str) or not metadata.get("main_idea", "").strip():
            errors.append(f"{rel}: main_idea 不能为空")
        tags = metadata.get("tags")
        if not isinstance(tags, dict):
            errors.append(f"{rel}: tags 必须按 task/method/problem 分组")
        else:
            for group in ("task", "method", "problem"):
                values = tags.get(group)
                if not isinstance(values, list):
                    errors.append(f"{rel}: tags.{group} 必须是列表")
                    continue
                allowed = set(taxonomy["tags"][group])
                invalid = sorted(set(values) - allowed)
                if invalid:
                    errors.append(f"{rel}: tags.{group} 包含未登记标签 {invalid}")
    elif metadata.get("status") not in taxonomy["document_statuses"]:
        errors.append(f"{rel}: 非法文档 status {metadata.get('status')!r}")

    if not scalar_date(metadata.get("updated")):
        errors.append(f"{rel}: updated 必须是 YYYY-MM-DD")
    return errors


def validate_path(path: Path, root: Path) -> List[str]:
    rel = path.relative_to(root)
    if not rel.parts or rel.parts[0] not in CONTENT_DIRS:
        return []
    errors: List[str] = []
    for part in rel.parts[1:-1]:
        if not KEBAB.fullmatch(part):
            errors.append(f"{rel.as_posix()}: 目录名不是 kebab-case: {part}")
    if not KEBAB.fullmatch(path.stem):
        errors.append(f"{rel.as_posix()}: 文件名不是 kebab-case")
    return errors


def resolve_link(source: Path, raw_target: str) -> Optional[Path]:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    if not target or target.startswith(("#", "http://", "https://", "mailto:", "data:")):
        return None
    target = unquote(target.split("#", 1)[0])
    if not target:
        return source
    return (source.parent / target).resolve()


def validate_repository(root: Path = ROOT) -> List[str]:
    errors: List[str] = []
    try:
        taxonomy = load_taxonomy(root)
    except Exception as exc:  # pragma: no cover - fatal configuration path
        return [f"无法读取 config/taxonomy.yml: {exc}"]

    linked_assets: Set[Path] = set()
    short_names: Dict[str, Path] = {}
    for path in markdown_documents(root):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"{rel}: 不是有效 UTF-8: {exc}")
            continue
        clean = strip_fenced_code(text)
        h1_count = sum(1 for line in clean.splitlines() if re.match(r"^# [^#]", line))
        if h1_count != 1:
            errors.append(f"{rel}: 应有且仅有一个 H1，当前为 {h1_count}")
        errors.extend(validate_path(path, root))

        if path.parts[len(root.parts)] in CONTENT_DIRS:
            try:
                doc = read_document(path)
            except (UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
                errors.append(f"{rel}: metadata 解析失败: {exc}")
                continue
            errors.extend(validate_metadata(doc, root, taxonomy))
            if doc.metadata.get("type") == "paper":
                short_name = str(doc.metadata.get("short_name", "")).casefold()
                if short_name in short_names:
                    errors.append(
                        f"{rel}: short_name 与 {short_names[short_name].relative_to(root).as_posix()} 重复"
                    )
                short_names[short_name] = path

        for match in LINK.finditer(clean):
            resolved = resolve_link(path, match.group(1))
            if resolved is None:
                continue
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                errors.append(f"{rel}: 相对链接越出仓库 {match.group(1)!r}")
                continue
            if not resolved.exists():
                errors.append(f"{rel}: 断开的相对链接 {match.group(1)!r}")
            else:
                try:
                    resolved.relative_to((root / "assets").resolve())
                except ValueError:
                    pass
                else:
                    if resolved.is_file() and resolved.suffix.lower() != ".md":
                        linked_assets.add(resolved)

    assets_root = root / "assets"
    if assets_root.exists():
        for asset in sorted(path.resolve() for path in assets_root.rglob("*") if path.is_file()):
            if asset.suffix.lower() == ".md":
                continue
            if asset not in linked_assets:
                errors.append(f"{asset.relative_to(root.resolve()).as_posix()}: 孤立资产，未被 Markdown 引用")
    return sorted(set(errors))


def escape_cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_index(root: Path = ROOT) -> str:
    taxonomy = load_taxonomy(root)
    grouped: Dict[str, List[Document]] = {category: [] for category in taxonomy["categories"]}
    for doc in paper_documents(root):
        category = doc.metadata.get("category")
        if category in grouped:
            grouped[category].append(doc)

    category_titles = {
        "model-based-rl": "Model-based Reinforcement Learning",
        "interactive-world-models": "Interactive World Models",
        "embodied-world-models": "Embodied World Models",
        "generative-model-rl": "RL for Generative Models",
        "surveys-and-benchmarks": "Surveys & Benchmarks",
    }
    lines = [
        "# Paper Index",
        "",
        "> 此文件由 `python scripts/kb.py build-index` 根据论文 YAML 自动生成，请勿手工编辑。",
        "",
        f"当前共有 **{sum(len(items) for items in grouped.values())}** 篇论文笔记。",
        "",
    ]
    for category in taxonomy["categories"]:
        docs = sorted(
            grouped[category],
            key=lambda doc: (doc.metadata.get("year", 9999), str(doc.metadata.get("short_name", "")).casefold()),
        )
        if not docs:
            continue
        lines.extend(
            [
                f"## {category_titles.get(category, category)}",
                "",
                "| Paper | Year | Venue | Main idea | Status | Confidence |",
                "|---|---:|---|---|---|---|",
            ]
        )
        for doc in docs:
            rel = doc.path.relative_to(root).as_posix()
            meta = doc.metadata
            paper = f"[{escape_cell(meta.get('short_name'))}]({rel})"
            lines.append(
                "| "
                + " | ".join(
                    [
                        paper,
                        escape_cell(meta.get("year")),
                        escape_cell(meta.get("venue")),
                        escape_cell(meta.get("main_idea")),
                        f"`{escape_cell(meta.get('status'))}`",
                        escape_cell(meta.get("confidence")),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_index(root: Path, check: bool) -> int:
    path = root / "PAPER_INDEX.md"
    expected = render_index(root)
    if check:
        actual = path.read_text(encoding="utf-8") if path.exists() else ""
        if actual != expected:
            print("PAPER_INDEX.md 与论文元数据不同步；请运行 build-index。", file=sys.stderr)
            return 1
        print("PAPER_INDEX.md 已同步。")
        return 0
    path.write_text(expected, encoding="utf-8")
    print(f"已生成 {path.relative_to(root)}。")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-index", help="从论文 metadata 生成 PAPER_INDEX.md")
    build.add_argument("--check", action="store_true", help="只检查索引是否同步")
    subparsers.add_parser("validate", help="验证 metadata、链接、命名和资产")
    args = parser.parse_args(argv)

    if args.command == "build-index":
        return build_index(ROOT, args.check)
    errors = validate_repository(ROOT)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"验证失败：{len(errors)} 个问题。", file=sys.stderr)
        return 1
    print("知识库验证通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
