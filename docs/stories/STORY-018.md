# STORY-018: 实现 MergeStrategy

**Epic:** EPIC-003 (Curator 语义去重引擎)
**Priority:** Must Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As a Curator
I want a merge strategy to combine similar memories
So that merged Bullets retain the best information

---

## Description

### Background
当 CuratorEngine 判定候选 Bullet 与现有记忆高度相似时，需要通过 MergeStrategy 合并两者。支持两种策略：`keep_best`（保留更优的那条）和 `merge_content`（合并内容）。策略通过 CuratorConfig.merge_strategy 配置。

### Scope
**In scope:**
- `keep_best` 策略：保留 instructivity_score 更高的一条
- `merge_content` 策略：合并 content，取字段并集
- MergeResult 数据结构
- 策略选择通过 CuratorConfig 配置

**Out of scope:**
- LLM-based 智能合并（未来增强）
- 冲突检测（STORY-047）

---

## Acceptance Criteria

- [ ] `keep_best` 策略：保留 instructivity_score 更高的一条；相同时保留更长的 content
- [ ] `merge_content` 策略：合并两条的 content（去重拼接），保留较高 recall_count 和 instructivity_score
- [ ] related_tools 和 key_entities 取并集
- [ ] tags 取并集
- [ ] updated_at 更新为当前时间
- [ ] 策略通过 `CuratorConfig.merge_strategy` 选择
- [ ] 支持自定义策略扩展（Strategy pattern）

---

## Technical Notes

### API Design

```python
@dataclass
class MergeResult:
    """Result of merging a candidate with an existing bullet."""
    merged_content: str
    merged_metadata: dict[str, Any]
    source_id: str              # existing bullet ID to update
    strategy_used: str          # "keep_best" | "merge_content"

class MergeStrategy(ABC):
    """Abstract base for merge strategies."""
    @abstractmethod
    def merge(
        self,
        candidate: CandidateBullet,
        existing: ExistingBullet,
    ) -> MergeResult: ...

class KeepBestStrategy(MergeStrategy):
    def merge(self, candidate, existing) -> MergeResult: ...

class MergeContentStrategy(MergeStrategy):
    def merge(self, candidate, existing) -> MergeResult: ...

def get_merge_strategy(name: str) -> MergeStrategy:
    """Factory function for merge strategies."""
```

### Components
- `memx/engines/curator/merger.py`

### Edge Cases
- 两条 content 完全相同 → keep_best 保留现有
- 一条 content 为空 → 保留非空的
- 两条 related_tools 有重叠 → 并集去重
- 未知策略名 → 回退到 keep_best + WARNING 日志

---

## Dependencies

**Prerequisite Stories:**
- STORY-017: CuratorEngine 核心去重

**Blocked Stories:**
- STORY-019: Curator 测试

---

## Definition of Done

- [ ] `memx/engines/curator/merger.py` 实现 KeepBestStrategy 和 MergeContentStrategy
- [ ] Strategy pattern 可扩展
- [ ] 单元测试覆盖两种策略的核心路径
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **KeepBestStrategy:** 1.5 points
- **MergeContentStrategy:** 1.5 points
- **Strategy pattern + factory:** 1 point
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
