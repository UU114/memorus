# STORY-043: 实现层级 Scope 管理

**Epic:** EPIC-008 (用户界面与发布)
**Priority:** Could Have
**Story Points:** 5
**Status:** Done
**Assigned To:** Developer
**Created:** 2026-02-27
**Sprint:** 4

---

## User Story

As a multi-project developer
I want separate knowledge scopes per project
So that project-specific knowledge doesn't leak across projects

---

## Description

### Background
当前 `BulletMetadata.scope` 和 `CandidateBullet.scope` 字段已存在（默认值 `"global"`），且通过 `BulletFactory` 序列化为 `memorus_scope` 存入 mem0 payload。但 scope 在以下关键路径中完全不生效：

1. **Search 路径**：`BulletForSearch` 无 scope 字段，`Memory._load_bullets_for_search()` 不加载 scope，`GeneratorEngine.search()` 无 scope 过滤
2. **Add 路径**：`IngestPipeline.process()` 不接受 scope 参数，scope 始终为 `"global"`
3. **Curator**：跨 scope 去重（不同项目的记忆可能被错误合并）
4. **CLI**：`memorus search` 无 `--scope` 选项

本 Story 需要将 scope 贯穿整个数据流，实现 project + global 双层检索。

### Scope
**In scope:**
- `BulletForSearch` 新增 `scope` 字段
- `Memory._load_bullets_for_search()` 加载 scope
- `Memory.add()` / `Memory.search()` 新增可选 `scope` 参数
- `GeneratorEngine.search()` 按 scope 过滤（project + global 合并）
- `IngestPipeline.process()` 传递 scope
- `CuratorEngine.curate()` 同 scope 内去重
- CLI `memorus search --scope` 支持
- 自动 scope 检测（基于 cwd 或 `--scope` 参数）

**Out of scope:**
- scope 层级嵌套（如 org > project > module）
- scope 间记忆迁移
- scope 权限控制

---

## Acceptance Criteria

- [ ] `Memory.add(messages, scope="project:myapp")` 将 scope 传递到 `CandidateBullet.scope` → `BulletMetadata.scope` → `memorus_scope`
- [ ] `Memory.add()` 不传 scope 时默认 `"global"`
- [ ] `Memory.search(query, scope="project:myapp")` 返回 project:myapp + global 两个 scope 的合并结果
- [ ] project scope 记忆在 `ScoreMerger` 中获得 scope 加权（默认 ×1.3，可配置）
- [ ] `Memory.search()` 不传 scope 时仅搜索 `"global"`（向后兼容）
- [ ] `BulletForSearch` 新增 `scope: str = "global"` 字段
- [ ] `Memory._load_bullets_for_search()` 从 mem0 payload 中加载 `memorus_scope` 到 `BulletForSearch.scope`
- [ ] `GeneratorEngine.search()` 接受 `scope` 参数，过滤 bullets 为 `scope == target` 或 `scope == "global"`
- [ ] `CuratorEngine.curate()` 仅在同 scope 内比较去重（不跨 scope 合并）
- [ ] `memorus search <query> --scope project:myapp` CLI 命令支持
- [ ] 保留 `user_id` / `agent_id` 正交维度（scope 是独立过滤维度）
- [ ] 全部改动单元测试覆盖

---

## Technical Notes

### Components
- `memorus/types.py` — `BulletForSearch` 新增 scope 字段
- `memorus/memory.py` — `add()`, `search()`, `_load_bullets_for_search()` 增加 scope 参数
- `memorus/engines/generator/engine.py` — `GeneratorEngine.search()` scope 过滤
- `memorus/engines/generator/score_merger.py` — scope 加权
- `memorus/engines/curator/engine.py` — `CuratorEngine.curate()` scope 感知
- `memorus/pipeline/ingest.py` — `IngestPipeline.process()` scope 透传
- `memorus/pipeline/retrieval.py` — `RetrievalPipeline.search()` scope 透传
- `memorus/config.py` — `RetrievalConfig` 新增 `scope_boost: float = 1.3`
- `memorus/cli/main.py` — `search` 命令新增 `--scope` 选项

### API Changes

```python
# Memory.add() — 新增 scope 参数
def add(self, messages, user_id=None, agent_id=None, run_id=None,
        metadata=None, filters=None, prompt=None, scope=None, **kwargs):
    # scope=None → "global" (backward compatible)
    effective_scope = scope or "global"
    # Pass to IngestPipeline
    ...

# Memory.search() — 新增 scope 参数
def search(self, query, user_id=None, agent_id=None, run_id=None,
           limit=100, filters=None, scope=None, **kwargs):
    # scope=None → only "global" (backward compatible)
    # scope="project:myapp" → "project:myapp" + "global" merged
    ...

# BulletForSearch — 新增字段
@dataclass
class BulletForSearch:
    bullet_id: str
    content: str = ""
    metadata: MetadataInfo = ...
    created_at: datetime | None = None
    decay_weight: float = 1.0
    extra: dict[str, Any] = ...
    scope: str = "global"          # NEW

# GeneratorEngine.search() — 新增 scope 参数
def search(self, query, bullets, limit=20, filters=None, scope=None):
    if scope:
        bullets = [b for b in bullets if b.scope == scope or b.scope == "global"]
    ...

# ScoreMerger — scope 加权
# 当 bullet.scope == target_scope (非 global) 时，final_score *= scope_boost
```

### Scope Boost 逻辑
```
FinalScore = BaseScore × DecayWeight × RecencyBoost × ScopeBoost
where ScopeBoost = 1.3 if bullet.scope == target_scope else 1.0
```

### Curator Scope 感知
```python
# CuratorEngine.curate() 改动
def curate(self, candidates, existing):
    for candidate in candidates:
        # Only compare against same-scope existing bullets
        same_scope = [e for e in existing if e.metadata.get("memorus_scope", "global") == candidate.scope]
        ...
```

### Edge Cases
- `scope=""` → 视为 `"global"`
- `scope="project:"` （无名称）→ 抛出 `ValueError`
- `scope="project:myapp"` 无记忆 → 仅返回 global 结果
- 旧记忆无 `memorus_scope` 字段 → 默认 `"global"`（向后兼容）

---

## Dependencies

**Prerequisite Stories:**
- STORY-004: MemorusMemory Decorator（已完成）
- STORY-028: GeneratorEngine + 降级（已完成）

**Blocked Stories:**
- None

---

## Definition of Done

- [ ] `BulletForSearch.scope` 字段添加
- [ ] `Memory.add(scope=...)` 端到端生效
- [ ] `Memory.search(scope=...)` 双层合并检索测试
- [ ] `CuratorEngine` 同 scope 去重测试
- [ ] `GeneratorEngine` scope 过滤测试
- [ ] `ScoreMerger` scope 加权测试
- [ ] CLI `memorus search --scope` 测试
- [ ] 向后兼容测试（无 scope 参数时行为不变）
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **BulletForSearch + _load_bullets_for_search 改动:** 1 point
- **Memory.add/search scope 参数 + IngestPipeline 透传:** 1 point
- **GeneratorEngine scope 过滤 + ScoreMerger 加权:** 1 point
- **CuratorEngine scope 感知:** 1 point
- **CLI + 测试:** 1 point
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
