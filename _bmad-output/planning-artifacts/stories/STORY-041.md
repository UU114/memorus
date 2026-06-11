# STORY-041: 实现 CLI 基础命令（status + search）

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
I want CLI commands to inspect my knowledge base
So that I can see what Memorus has learned and search my memories

---

## Description

### Background
CLI 是 Memorus 面向用户的主要交互界面。`memorus status` 提供知识库概览（总数、分布、健康状态），`memorus search` 提供交互式混合检索。两者是最基础的用户可见功能。使用 Click 框架实现。

### Scope
**In scope:**
- `memorus status` — 知识库统计命令
- `memorus search <query>` — 混合检索命令
- Click 框架搭建
- `--json` 输出格式支持
- 入口点注册（pyproject.toml `[project.scripts]`）

**Out of scope:**
- 管理命令（learn/list/forget → STORY-042）
- Daemon 管理命令
- 配置命令

---

## Acceptance Criteria

- [ ] `memorus status` 输出：记忆总数、section 分布、knowledge_type 分布、平均 decay_weight、ACE 模式（on/off）
- [ ] `memorus status --json` 输出 JSON 格式
- [ ] `memorus search <query>` 使用 RetrievalPipeline 混合检索
- [ ] 搜索结果格式化输出：score、content 摘要、knowledge_type、tags
- [ ] `memorus search <query> --json` 输出 JSON 格式
- [ ] `memorus search <query> --limit N` 限制结果数量（默认 5）
- [ ] 无记忆时 `memorus status` 显示友好提示
- [ ] 搜索无结果时显示友好提示
- [ ] `memorus --version` 显示版本号
- [ ] `memorus --help` 显示帮助信息

---

## Technical Notes

### Components
- `memorus/cli/__init__.py` — 包入口
- `memorus/cli/main.py` — Click CLI app

### API Design

```python
import click
from memorus.memory import Memory

@click.group()
@click.version_option()
@click.pass_context
def cli(ctx):
    """Memorus - Intelligent Memory Engine for AI Tools."""
    ctx.ensure_object(dict)
    ctx.obj["memory"] = Memory(config={"ace_enabled": True})

@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def status(ctx, as_json: bool):
    """Show knowledge base statistics."""
    memory = ctx.obj["memory"]
    all_memories = memory.get_all()

    stats = {
        "total": len(all_memories),
        "ace_enabled": True,
        "sections": _count_by(all_memories, "memorus_section"),
        "knowledge_types": _count_by(all_memories, "memorus_knowledge_type"),
        "avg_decay_weight": _avg_field(all_memories, "memorus_decay_weight"),
    }

    if as_json:
        click.echo(json.dumps(stats, indent=2))
    else:
        _print_status(stats)

@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Max results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def search(ctx, query: str, limit: int, as_json: bool):
    """Search knowledge base with hybrid retrieval."""
    memory = ctx.obj["memory"]
    results = memory.search(query, limit=limit)

    if as_json:
        click.echo(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        _print_results(results)
```

### Output Format

**status (human):**
```
Memorus Knowledge Base Status
══════════════════════════
Total memories:    42
ACE mode:          ON

Section distribution:
  playbook:        28 (66.7%)
  tool_pattern:    10 (23.8%)
  error_fix:        4 (9.5%)

Knowledge types:
  preference:      15 (35.7%)
  tool_pattern:    12 (28.6%)
  workflow:         8 (19.0%)
  error_fix:        4 (9.5%)
  fact:             3 (7.1%)

Avg decay weight:  0.78
```

**search (human):**
```
Search: "async error handling"  (3 results)
────────────────────────────────

[0.92] When using asyncio, always wrap with try/except for CancelledError
       type: tool_pattern | tags: python, asyncio
       id: abc123

[0.85] Prefer structured error handling with Result type over bare exceptions
       type: preference | tags: error-handling
       id: def456

[0.71] If subprocess fails, check stderr before retrying
       type: error_fix | tags: subprocess
       id: ghi789
```

### Entry Point

```toml
# pyproject.toml
[project.scripts]
memorus = "memorus.cli.main:cli"
```

### Dependencies on Existing Code
- `memorus/memory.py:Memory` — get_all(), search() API
- `memorus/config.py:MemorusConfig` — 配置加载

### Edge Cases
- 空知识库 → status 显示 "No memories yet. Use `memorus learn` to add knowledge."
- 搜索无结果 → "No results found for: <query>"
- Memory 初始化失败 → 显示友好错误信息并退出
- Unicode query → 正常处理（ensure_ascii=False）

---

## Dependencies

**Prerequisite Stories:**
- STORY-004: MemorusMemory Decorator ✓（已完成）
- STORY-030: RetrievalPipeline ✓（已完成）

**Blocked Stories:**
- STORY-042: CLI 管理命令（learn + list + forget）

---

## Definition of Done

- [ ] `memorus/cli/main.py` 实现 status + search 命令
- [ ] pyproject.toml 注册 entry point
- [ ] `memorus --version` 和 `memorus --help` 正常工作
- [ ] `--json` 输出格式测试
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **Click 框架搭建 + 入口点:** 0.5 points
- **status 命令:** 1.5 points
- **search 命令:** 1.5 points
- **测试:** 0.5 points
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
