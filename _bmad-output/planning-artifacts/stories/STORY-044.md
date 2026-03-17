# STORY-044: 实现导入/导出功能

**Epic:** EPIC-008 (用户界面与发布)
**Priority:** Could Have
**Story Points:** 4
**Status:** Done
**Assigned To:** Developer
**Created:** 2026-02-27
**Sprint:** 4

---

## User Story

As a Memorus user
I want to backup and transfer my memories
So that my knowledge is portable across machines and projects

---

## Description

### Background
`Memory.export()` 和 `Memory.import_data()` 当前作为方法存根存在于 `memorus/memory.py`，调用时抛出 `NotImplementedError`。用户需要：
1. 导出全部记忆（或按 scope 过滤）为 JSON 或 Markdown 格式，用于备份或迁移
2. 从 JSON 文件导入记忆，经过 Curator 去重后入库，避免重复

### Scope
**In scope:**
- `Memory.export(format="json", scope=None)` — 完整 Bullet 元数据 JSON 导出
- `Memory.export(format="markdown", scope=None)` — 人类可读 Markdown 列表
- `Memory.import_data(data, format="json")` — JSON 导入 + Curator 去重
- CLI `memorus export` / `memorus import` 命令
- scope 过滤导出

**Out of scope:**
- CSV/YAML 格式
- 增量导出（仅导出变更）
- 跨版本迁移工具
- 导入时的冲突解决 UI

---

## Acceptance Criteria

- [ ] `Memory.export(format="json")` 返回包含所有记忆的 JSON 结构，每条记忆含 `id`, `content`, `metadata`（含所有 `memorus_` 前缀字段）
- [ ] `Memory.export(format="json", scope="project:myapp")` 仅导出指定 scope 的记忆
- [ ] JSON 导出包含版本头 `{"version": "1.0", "exported_at": "...", "total": N, "memories": [...]}`
- [ ] `Memory.export(format="markdown")` 返回人类可读 Markdown 字符串，按 section 分组
- [ ] `Memory.import_data(data, format="json")` 解析 JSON 并逐条经过 `CuratorEngine.curate()` 去重
- [ ] 导入时重复记忆（similarity ≥ threshold）被跳过，返回统计 `{"imported": N, "skipped": M, "merged": K}`
- [ ] 导入时 `BulletMetadata` 字段完整恢复（scope, decay_weight, tags 等）
- [ ] `memorus export [--format json|markdown] [--scope SCOPE] [--output FILE]` CLI 命令
- [ ] `memorus import FILE [--json]` CLI 命令
- [ ] 导出文件 > 0 条时可成功导回（round-trip 测试）

---

## Technical Notes

### Components
- `memorus/memory.py` — 实现 `export()` 和 `import_data()`（替换 NotImplementedError 存根）
- `memorus/cli/main.py` — 新增 `export` 和 `import` 命令
- `memorus/utils/bullet_factory.py` — 可能需要 `from_export_dict()` 反序列化辅助

### API Design

```python
# Memory.export()
def export(self, format: str = "json", scope: str | None = None) -> str | dict:
    all_memories = self.get_all()
    memories = all_memories.get("results", [])

    if scope:
        memories = [m for m in memories
                    if m.get("metadata", {}).get("memorus_scope", "global") == scope]

    if format == "json":
        return {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "total": len(memories),
            "memories": memories,
        }
    elif format == "markdown":
        return self._export_markdown(memories)
    else:
        raise ValueError(f"Unsupported format: {format}")

# Memory.import_data()
def import_data(self, data: dict | str, format: str = "json") -> dict:
    if format == "json":
        if isinstance(data, str):
            data = json.loads(data)
        memories = data.get("memories", [])
    else:
        raise ValueError(f"Unsupported import format: {format}")

    imported = 0
    skipped = 0
    merged = 0

    for mem in memories:
        content = mem.get("memory", "")
        metadata = mem.get("metadata", {})
        # Reconstruct CandidateBullet from metadata
        candidate = BulletFactory.from_export_payload(content, metadata)
        # Run through Curator
        existing = self._load_existing_for_curator()
        result = self._curator.curate([candidate], existing)

        if result.to_add:
            self._mem0.add([{"role": "user", "content": content}],
                           metadata=BulletFactory.to_mem0_metadata(candidate))
            imported += 1
        elif result.to_merge:
            # Apply merge
            merged += 1
        else:
            skipped += 1

    return {"imported": imported, "skipped": skipped, "merged": merged}
```

### Markdown Export Format

```markdown
# Memorus Knowledge Export
> Exported: 2026-04-20T10:30:00Z | Total: 42 memories

## Commands (5)
- [0.92] **tool_pattern** | Use pytest -v for verbose output `abc123`
- [0.85] **method** | git rebase --onto for branch surgery `def456`

## Debugging (8)
- [0.78] **pitfall** | Check stderr before stdout in subprocess `ghi789`
...

## Preferences (3)
...
```

### CLI Commands

```python
@cli.command()
@click.option("--format", "-f", type=click.Choice(["json", "markdown"]), default="json")
@click.option("--scope", default=None, help="Filter by scope")
@click.option("--output", "-o", default=None, help="Output file path (default: stdout)")
@click.pass_context
def export(ctx, format, scope, output):
    """Export memories to file."""
    memory = ctx.obj["memory"]
    result = memory.export(format=format, scope=scope)

    content = json.dumps(result, indent=2, ensure_ascii=False) if format == "json" else result

    if output:
        Path(output).write_text(content, encoding="utf-8")
        click.echo(f"Exported to {output}")
    else:
        click.echo(content)

@cli.command("import")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output result as JSON")
@click.pass_context
def import_memories(ctx, file_path, as_json):
    """Import memories from a JSON file."""
    memory = ctx.obj["memory"]
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    result = memory.import_data(data, format="json")

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Import complete:")
        click.echo(f"  Imported: {result['imported']}")
        click.echo(f"  Merged:   {result['merged']}")
        click.echo(f"  Skipped:  {result['skipped']}")
```

### Dependencies on Existing Code
- `memorus/memory.py:Memory` — `get_all()`, `export()` / `import_data()` 存根
- `memorus/engines/curator/engine.py:CuratorEngine` — 去重检查
- `memorus/utils/bullet_factory.py:BulletFactory` — 序列化/反序列化
- `memorus/cli/main.py` — Click group

### Edge Cases
- 空数据库导出 → `{"version": "1.0", "total": 0, "memories": []}`
- 导入空文件 → `{"imported": 0, "skipped": 0, "merged": 0}`
- 导入非法 JSON → 抛出明确 `ValueError`
- 导入缺少 `version` 字段 → 默认按 v1.0 处理
- 导入旧版本格式（无 memorus_ 前缀）→ 使用默认元数据
- 大文件导入（>10000 条）→ 分批处理，避免内存爆炸

---

## Dependencies

**Prerequisite Stories:**
- STORY-004: MemorusMemory Decorator（已完成）
- STORY-017: Curator 核心去重（已完成）

**Blocked Stories:**
- None

---

## Definition of Done

- [ ] `Memory.export(format="json")` 实现并测试
- [ ] `Memory.export(format="markdown")` 实现并测试
- [ ] `Memory.import_data()` 经过 Curator 去重并测试
- [ ] Round-trip 测试（export → import → 无重复）
- [ ] CLI `memorus export` / `memorus import` 命令实现并测试
- [ ] scope 过滤导出测试
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **export JSON + Markdown:** 1 point
- **import + Curator 去重:** 1.5 points
- **CLI 命令:** 0.5 points
- **测试:** 1 point
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
