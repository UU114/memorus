"""YAML frontmatter parsing / serialization for vault markdown files.

Format mirrors the Obsidian / Jekyll convention:

    ---
    key: value
    ---
    body text
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FrontmatterDoc:
    meta: dict[str, Any]
    body: str


def _yaml():
    import yaml  # PyYAML is a transitive dep via memorus.team.config
    return yaml


def parse(text: str) -> FrontmatterDoc:
    """Parse a string with optional YAML frontmatter into (meta, body)."""
    if not text.startswith("---"):
        return FrontmatterDoc(meta={}, body=text)
    lines = text.splitlines()
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return FrontmatterDoc(meta={}, body=text)
    fm_text = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    try:
        meta = _yaml().safe_load(fm_text) or {}
    except Exception:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return FrontmatterDoc(meta=meta, body=body)


def dump(meta: dict[str, Any], body: str) -> str:
    """Serialize meta+body back to a frontmatter document."""
    yaml = _yaml()
    fm_text = yaml.safe_dump(meta, sort_keys=True, allow_unicode=True).rstrip()
    return f"---\n{fm_text}\n---\n{body}\n"
