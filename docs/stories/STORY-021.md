# STORY-021: 实现 Decay sweep 和召回强化

**Epic:** EPIC-004 (Decay 衰退引擎)
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As a MemX system
I want batch decay updates and recall reinforcement
So that memory lifecycle is automatically managed

---

## Description

### Background
DecayEngine 的 compute_weight() 只计算单条记忆的权重。系统需要 sweep() 方法批量更新所有记忆的衰退权重，并需要 reinforce() 方法在记忆被召回时强化其权重。sweep 通常在 session 结束时执行，reinforce 在每次 search 命中时异步执行。

### Scope
**In scope:**
- `sweep()` 批量衰退扫描
- `reinforce(bullet_ids)` 召回强化
- DecaySweepResult 结果数据结构
- 归档标记（不物理删除）
- 异步 reinforce 支持

**Out of scope:**
- 物理删除归档记忆
- 自动调度（Daemon 触发 sweep 在 Sprint 3）
- Memory.run_decay_sweep() 对接（在本 story 中实现引擎层，对接在 STORY-030）

---

## Acceptance Criteria

- [ ] `sweep(bullets)` 批量计算所有记忆的 decay_weight 并返回更新结果
- [ ] 返回 `DecaySweepResult(updated, archived, permanent, unchanged)` 统计
- [ ] `reinforce(bullet_ids, recall_fn)` 更新 recall_count + 1 和 last_recall = now()
- [ ] reinforce 通过回调函数更新（不直接依赖 mem0 存储层）
- [ ] 归档标记不物理删除记忆（仅将 should_archive 标记返回）
- [ ] sweep 处理空列表不报错
- [ ] 单条记忆 sweep 失败不影响其他记忆的处理

---

## Technical Notes

### API Design

```python
@dataclass
class BulletDecayInfo:
    """Input for sweep: minimal info needed for decay calculation."""
    bullet_id: str
    created_at: datetime
    recall_count: int = 0
    last_recall: datetime | None = None
    current_weight: float = 1.0

@dataclass
class DecaySweepResult:
    updated: int = 0        # weight changed
    archived: int = 0       # marked for archive
    permanent: int = 0      # recall_count >= threshold
    unchanged: int = 0      # still in protection or same weight
    errors: list[str] = field(default_factory=list)

class DecayEngine:
    # ... (from STORY-020)

    def sweep(
        self,
        bullets: list[BulletDecayInfo],
        now: datetime | None = None,
    ) -> DecaySweepResult: ...

    def reinforce(
        self,
        bullet_ids: list[str],
        update_fn: Callable[[str, dict], None],  # callback(id, {recall_count, last_recall})
    ) -> int:  # number of reinforced
        ...
```

### Components
- `memx/engines/decay/engine.py` — 扩展 DecayEngine，添加 sweep() 和 reinforce()

### Edge Cases
- sweep 1000+ 条记忆性能需可接受
- reinforce 空列表不报错
- update_fn 回调失败 → 记录 WARNING，不中断其他条目
- sweep 中某条记忆 created_at 为 None → 跳过该条，记入 errors

---

## Dependencies

**Prerequisite Stories:**
- STORY-020: DecayEngine 核心衰退（compute_weight）

**Blocked Stories:**
- STORY-022: Decay 测试
- STORY-030: RetrievalPipeline（需要 reinforce）
- STORY-034: SessionEndHook（需要 sweep）

---

## Definition of Done

- [ ] `sweep()` 方法实现并通过测试
- [ ] `reinforce()` 方法实现并通过测试
- [ ] DecaySweepResult 数据结构定义
- [ ] 错误隔离：单条失败不影响批量处理
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **sweep 逻辑:** 2 points
- **reinforce 逻辑:** 1.5 points
- **数据结构 + 错误处理:** 1.5 points
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
