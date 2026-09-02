"""Load the portfolio markdown files (resume + 5 project write-ups) into memory.

Each file has a small YAML frontmatter block (id/title/tags/...) followed by the body.
Frontmatter is optional metadata Chroma will let us filter/display by later (e.g. "only
show me chunks from project write-ups, not the resume profile" or "cite this as coming
from smart-hydro-alert").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SourceDoc:
    doc_id: str
    title: str
    tags: list[str]
    metadata: dict = field(default_factory=dict)
    content: str = ""
    source_path: str = ""


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    """Split a '---\\nyaml\\n---\\nbody' file into (metadata_dict, body)."""
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    _, fm_text, body = parts
    metadata = yaml.safe_load(fm_text) or {}
    return metadata, body.strip()


def load_portfolio_docs(portfolio_dir: Path) -> list[SourceDoc]:
    """Read every *.md file in portfolio_dir into a SourceDoc."""
    docs: list[SourceDoc] = []
    for path in sorted(portfolio_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, body = _split_frontmatter(raw)
        docs.append(
            SourceDoc(
                doc_id=metadata.get("id", path.stem),
                title=metadata.get("title", path.stem),
                tags=metadata.get("tags", []),
                metadata=metadata,
                content=body,
                source_path=str(path),
            )
        )
    return docs
