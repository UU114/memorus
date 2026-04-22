#!/usr/bin/env python3
"""Migrate OpenClaw Markdown memories into Memorus.

Reads MEMORY.md (long-term) and memory/*.md (daily journals),
splits them into chunks, and imports via the Python Memory API.

Usage:
    python migrate.py --memory-dir ~/.openclaw
    python migrate.py --memory-dir ~/.openclaw --dry-run
    python migrate.py --memory-dir ~/.openclaw --config ~/memorus-config.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def parse_markdown_sections(text: str, source_file: str) -> list[dict]:
    """Split markdown text into chunks by headings or double-newline paragraphs."""
    chunks = []
    # Split by headings (##, ###, etc.)
    sections = re.split(r"(?=^#{1,4}\s)", text, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Further split long sections by double-newline paragraphs
        paragraphs = re.split(r"\n\n+", section)
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 10:
                continue
            # Skip pure heading lines with no content
            lines = para.split("\n")
            content_lines = [l for l in lines if not re.match(r"^#{1,4}\s*$", l)]
            if not content_lines:
                continue
            chunks.append({
                "content": para,
                "metadata": {
                    "source": "openclaw_migration",
                    "original_file": source_file,
                    "source_type": "import",
                },
            })

    return chunks


def parse_daily_journal(text: str, filename: str) -> list[dict]:
    """Parse a daily journal file (memory/YYYY-MM-DD.md) into chunks."""
    # Extract date from filename
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    date_str = date_match.group(1) if date_match else None

    chunks = []
    # Split by list items (- or *) or headings
    entries = re.split(r"(?=^[-*]\s|^#{1,4}\s)", text, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry or len(entry) < 10:
            continue
        meta = {
            "source": "openclaw_migration",
            "original_file": f"memory/{filename}",
            "source_type": "import",
        }
        if date_str:
            meta["original_date"] = date_str
        chunks.append({"content": entry, "metadata": meta})

    return chunks


def collect_chunks(memory_dir: str) -> list[dict]:
    """Collect all memory chunks from an OpenClaw memory directory."""
    base = Path(memory_dir)
    all_chunks = []

    # 1. MEMORY.md (long-term memory)
    memory_md = base / "MEMORY.md"
    if memory_md.exists():
        text = memory_md.read_text(encoding="utf-8")
        chunks = parse_markdown_sections(text, "MEMORY.md")
        all_chunks.extend(chunks)
        print(f"  MEMORY.md: {len(chunks)} chunks")

    # 2. memory/*.md (daily journals)
    memory_subdir = base / "memory"
    if memory_subdir.is_dir():
        for md_file in sorted(memory_subdir.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            chunks = parse_daily_journal(text, md_file.name)
            all_chunks.extend(chunks)
            if chunks:
                print(f"  memory/{md_file.name}: {len(chunks)} chunks")

    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate OpenClaw Markdown memories into Memorus"
    )
    parser.add_argument(
        "--memory-dir",
        required=True,
        help="Path to OpenClaw memory directory (contains MEMORY.md and/or memory/)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to Memorus config JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and display chunks without importing",
    )
    args = parser.parse_args()

    memory_dir = os.path.expanduser(args.memory_dir)
    if not os.path.isdir(memory_dir):
        print(f"ERROR: Directory not found: {memory_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {memory_dir} ...")
    chunks = collect_chunks(memory_dir)

    if not chunks:
        print("No memory chunks found. Nothing to migrate.")
        return

    print(f"\nTotal: {len(chunks)} chunks collected")

    if args.dry_run:
        print("\n--- DRY RUN (no data will be imported) ---\n")
        for i, chunk in enumerate(chunks, 1):
            content_preview = chunk["content"][:120].replace("\n", " ")
            source = chunk["metadata"]["original_file"]
            print(f"  [{i}] ({source}) {content_preview}...")
        print(f"\n{len(chunks)} chunks would be imported.")
        return

    # Import into Memorus
    config = None
    if args.config:
        config_path = os.path.expanduser(args.config)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    from memorus.core.memory import Memory

    mem = Memory(config=config)

    imported = 0
    errors = 0
    for i, chunk in enumerate(chunks, 1):
        try:
            mem.add(chunk["content"], metadata=chunk["metadata"])
            imported += 1
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(chunks)}")
        except Exception as exc:
            errors += 1
            print(f"  ERROR importing chunk {i}: {exc}", file=sys.stderr)

    print(f"\nMigration complete: {imported} imported, {errors} errors")


if __name__ == "__main__":
    main()
