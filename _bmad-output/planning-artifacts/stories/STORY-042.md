# STORY-042: 实现 CLI 管理命令（learn + list + forget）

**Epic:** EPIC-008 (用户界面与发布)
**Priority:** Should Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 3

---

## User Story

As a Memorus user
I want to manually manage my memories
So that I have full control over my Playbook

---

## Description

### Background
除了自动学习，用户也需要手动管理知识库的能力。`memorus learn` 让用户直接教 Memorus 新知识（经过 Reflector + Curator 处理），`memorus list` 列出已有记忆（支持过滤），`memorus forget` 删除指定记忆，`memorus sweep` 手动触发衰退扫描。

### Scope
**In scope:**
- `memorus learn <content>` — 手动添加知识
- `memorus list` — 列出记忆
- `memorus forget <id>` — 删除记忆
- `memorus sweep` — 手动衰退扫描
- 各命令支持 `--json` 输出

**Out of scope:**
- `memorus export/import`（STORY-044）
- 批量操作
- 交互式编辑

---

## Acceptance Criteria

- [ ] `memorus learn <content>` 将内容经过 IngestPipeline（Reflector + Curator）处理后存储
- [ ] learn 成功后显示蒸馏结果（knowledge_type、distilled_rule、是否被去重）
- [ ] `memorus learn --raw <content>` 跳过 Reflector，直接存储（用于精确控制内容）
- [ ] `memorus list` 列出所有记忆（id、content 摘要、type、decay_weight、tags）
- [ ] `memorus list --scope project:<name>` 按 scope 过滤
- [ ] `memorus list --type <knowledge_type>` 按类型过滤
- [ ] `memorus list --limit N` 限制显示数量（默认 20）
- [ ] `memorus forget <id>` 删除指定 id 的记忆，显示确认信息
- [ ] `memorus forget <id> --yes` 跳过确认直接删除
- [ ] `memorus sweep` 手动执行 DecayEngine.sweep()，显示结果（archived/permanent/updated 数量）
- [ ] 所有命令支持 `--json` 输出格式

---

## Technical Notes

### Components
- `memorus/cli/main.py` — 追加命令到已有 Click group

### API Design

```python
@cli.command()
@click.argument("content")
@click.option("--raw", is_flag=True, help="Skip Reflector, store as-is")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def learn(ctx, content: str, raw: bool, as_json: bool):
    """Teach Memorus new knowledge."""
    memory = ctx.obj["memory"]
    if raw:
        result = memory.add(
            [{"role": "user", "content": content}],
            user_id="manual",
            metadata={"source_type": "manual"},
        )
    else:
        result = memory.add(
            [{"role": "user", "content": content}],
            user_id="manual",
        )
    # Display result
    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_learn_result(result)

@cli.command("list")
@click.option("--scope", default=None, help="Filter by scope (e.g., project:myapp)")
@click.option("--type", "knowledge_type", default=None, help="Filter by knowledge type")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def list_memories(ctx, scope, knowledge_type, limit, as_json):
    """List all memories in the knowledge base."""
    memory = ctx.obj["memory"]
    all_memories = memory.get_all()

    # Apply filters
    filtered = _apply_filters(all_memories, scope=scope, knowledge_type=knowledge_type)
    filtered = filtered[:limit]

    if as_json:
        click.echo(json.dumps(filtered, indent=2, ensure_ascii=False))
    else:
        _print_memory_list(filtered)

@cli.command()
@click.argument("memory_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def forget(ctx, memory_id: str, yes: bool, as_json: bool):
    """Delete a specific memory by ID."""
    memory = ctx.obj["memory"]

    if not yes:
        # Show memory content for confirmation
        mem = memory.get(memory_id)
        if not mem:
            click.echo(f"Memory not found: {memory_id}", err=True)
            raise SystemExit(1)
        click.echo(f"Will delete: {mem.get('memory', '')[:100]}")
        if not click.confirm("Are you sure?"):
            click.echo("Cancelled.")
            return

    memory.delete(memory_id)
    result = {"deleted": memory_id}
    if as_json:
        click.echo(json.dumps(result))
    else:
        click.echo(f"Deleted memory: {memory_id}")

@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def sweep(ctx, as_json: bool):
    """Run manual decay sweep on all memories."""
    memory = ctx.obj["memory"]
    # Access decay engine through memory
    result = memory.run_decay_sweep()

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Decay sweep complete:")
        click.echo(f"  Updated:   {result.get('updated', 0)}")
        click.echo(f"  Archived:  {result.get('archived', 0)}")
        click.echo(f"  Permanent: {result.get('permanent', 0)}")
```

### Output Format

**learn (human):**
```
Learned new knowledge:
  Type:      tool_pattern
  Rule:      "When using pytest, always use -v for verbose output"
  Tags:      pytest, testing
  ID:        abc123
```

**list (human):**
```
Memories (42 total, showing 20)
═══════════════════════════════

abc123  [0.92] preference    | User prefers dark mode
def456  [0.85] tool_pattern  | pytest -v flag usage
ghi789  [0.71] error_fix     | subprocess stderr check
...

Use --scope or --type to filter, --limit to show more.
```

**sweep (human):**
```
Decay sweep complete:
  Updated:   35
  Archived:  3
  Permanent: 8
  Unchanged: 12
```

### Dependencies on Existing Code
- `memorus/cli/main.py` — STORY-041 的 Click group
- `memorus/memory.py:Memory` — add(), get_all(), get(), delete()
- `memorus/engines/decay/engine.py:DecayEngine` — sweep()
- `memorus/pipeline/ingest.py:IngestPipeline` — learn 路径

### Edge Cases
- `memorus learn ""` → 空内容，显示错误提示
- `memorus forget` 无 ID → Click 自动报错
- `memorus forget <不存在的ID>` → "Memory not found" 错误
- `memorus list` 无记忆 → "No memories yet"
- `memorus sweep` 无记忆 → 显示全零结果
- content 包含 shell 特殊字符 → Click 参数自动处理
- 非常长的 content → IngestPipeline 内部截断

---

## Dependencies

**Prerequisite Stories:**
- STORY-041: CLI 基础命令（status + search）

**Blocked Stories:**
- None

---

## Definition of Done

- [ ] `memorus/cli/main.py` 新增 learn, list, forget, sweep 命令
- [ ] 各命令 `--json` 输出测试
- [ ] 各命令错误路径测试
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **learn 命令:** 1 point
- **list 命令 + 过滤:** 1 point
- **forget + sweep 命令:** 1 point
- **测试:** 1 point
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
