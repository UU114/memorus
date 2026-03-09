# STORY-066: 实现 Team CLI 命令

**Epic:** EPIC-011 (Federation Mode MVP)
**Priority:** Should Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 6

---

## User Story

As a developer
I want CLI commands to manage team features
So that I can check status, sync, and nominate from terminal

---

## Description

### Background
Team 功能需要用户可操作的 CLI 接口。本 Story 在现有 memx CLI 基础上添加 team 子命令组，提供状态查看、手动同步和提名管理功能。

### Scope
**In scope:**
- `ace team status` — 状态概览
- `ace team sync` — 手动同步
- `ace nominate list` — 列出候选
- `ace nominate submit <id>` — 手动提名
- Team 未启用时友好提示
- `--json` 输出格式

**Out of scope:**
- `ace upvote/downvote`（STORY-069）
- Team 配置修改 CLI
- Server 管理命令

### User Flow
```
$ ace team status
Mode: federation
Team ID: my-team
Server: https://ace.example.com
Cached Bullets: 1,234 / 2,000
Last Sync: 2 hours ago (success)
Subscribed Tags: #frontend #react
Pending Nominations: 3

$ ace team sync
Syncing... pulled 12 new bullets, 2 tombstones.
Cache: 1,244 / 2,000

$ ace nominate list
ID          | Score | Content Preview
bullet-abc  | 0.92  | "Use React.memo for expensive renders..."
bullet-def  | 0.88  | "Always validate JWT expiry before..."
bullet-ghi  | 0.85  | "SQLAlchemy session scope should..."

$ ace nominate submit bullet-abc
Redacting... done.
Review sanitized content:
  "Use [FRAMEWORK].memo for expensive renders when props are stable"
Confirm nomination? [y/N]: y
Nominated successfully!
```

---

## Acceptance Criteria

- [ ] `ace team status` — 显示模式、缓存数量、上次同步时间、订阅标签
- [ ] `ace team sync` — 强制增量同步
- [ ] `ace nominate list` — 列出待提名候选
- [ ] `ace nominate submit <id>` — 手动提名（经过 Redactor）
- [ ] Team 未启用时友好提示（"Team features not enabled. Configure team settings..."）
- [ ] 支持 `--json` 输出格式
- [ ] `ace team sync --full` — 强制全量同步

---

## Technical Notes

### Components
- **File:** `memx/team/cli.py`
- **Framework:** Click（与现有 CLI 一致）

### CLI Structure
```python
@click.group()
def team():
    """Team memory management commands"""
    pass

@team.command()
@click.option("--json", "as_json", is_flag=True)
def status(as_json):
    ...

@team.command()
@click.option("--full", is_flag=True, help="Force full sync")
def sync(full):
    ...

@click.group()
def nominate():
    """Knowledge nomination commands"""
    pass

@nominate.command("list")
@click.option("--json", "as_json", is_flag=True)
def list_candidates(as_json):
    ...

@nominate.command()
@click.argument("bullet_id")
def submit(bullet_id):
    ...
```

### Edge Cases
- Team 未配置 → 友好错误信息 + 配置引导
- Server 不可达（sync 时）→ 错误信息 + 建议检查网络
- 无候选 Bullet（nominate list）→ "No candidates found"
- 无效 bullet_id（nominate submit）→ "Bullet not found"

---

## Dependencies

**Prerequisite Stories:**
- STORY-059: TeamCacheStorage
- STORY-064: Nominator 提名流水线

**Blocked Stories:** None

**External Dependencies:**
- Click（CLI 框架，已有）

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] team status 输出测试
  - [ ] team sync 测试
  - [ ] nominate list 测试
  - [ ] nominate submit 流程测试
  - [ ] --json 输出格式测试
  - [ ] Team 未启用错误提示测试
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **CLI 命令实现:** 2 points
- **测试:** 1 point
- **Total:** 3 points

**Rationale:** CLI 命令主要是调用现有组件，逻辑简单，格式化输出占主要工作量。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
