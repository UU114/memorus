# STORY-020: 实现 DecayEngine 核心衰退逻辑

**Epic:** EPIC-004 (Decay 衰退引擎)
**Priority:** Must Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As a long-term user
I want old unused memories to naturally fade
So that my context stays relevant and the Playbook doesn't grow unbounded

---

## Description

### Background
MemX 的记忆需要随时间自然衰退，类似人类遗忘曲线。长期不被召回的记忆应逐渐降低权重，最终归档。同时，高频召回的记忆应被永久保留。DecayEngine 是实现这一生命周期管理的核心模块。

### Scope
**In scope:**
- `compute_weight()` 衰退公式实现
- 保护期逻辑（新记忆不衰退）
- 永久保留逻辑（高频召回记忆锁定 weight=1.0）
- 归档标记逻辑（weight 低于阈值）
- 所有参数通过 `DecayConfig` 可配置
- 衰退公式独立模块 `formulas.py`

**Out of scope:**
- sweep 批量扫描（STORY-021）
- reinforce 召回强化（STORY-021）
- 物理删除归档记忆

---

## User Flow

1. 系统接收一条记忆的元数据（created_at, recall_count, last_recall）
2. DecayEngine.compute_weight() 计算当前衰退权重
3. 根据保护期、永久保留、归档阈值判断记忆状态
4. 返回新的 decay_weight 和是否应归档的标记

---

## Acceptance Criteria

- [ ] `compute_weight()` 实现公式：`2^(-age_days / half_life) × (1 + boost × recall_count)`
- [ ] 保护期内（默认 7 天）weight 锁定 1.0，不受公式影响
- [ ] `recall_count >= permanent_threshold`（默认 15）→ weight = 1.0（永久保留）
- [ ] weight < archive_threshold（默认 0.02）→ 返回 `should_archive=True`
- [ ] 所有参数通过 `DecayConfig` 可配置（half_life_days, boost_factor, protection_days, permanent_threshold, archive_threshold）
- [ ] weight 值始终 clamp 在 [0.0, 1.0] 范围内
- [ ] 传入 `None` 或缺失字段时使用合理默认值，不报错

---

## Technical Notes

### Components
- `memx/engines/decay/formulas.py` — 纯函数衰退公式
- `memx/engines/decay/engine.py` — DecayEngine 类

### API Design

```python
# formulas.py
def exponential_decay(age_days: float, half_life: float) -> float:
    """Compute base decay: 2^(-age_days / half_life)"""

def boosted_weight(base: float, boost_factor: float, recall_count: int) -> float:
    """Apply recall boost: base × (1 + boost × recall_count)"""

# engine.py
@dataclass
class DecayResult:
    weight: float          # new decay_weight [0.0, 1.0]
    should_archive: bool   # True if below archive_threshold
    is_permanent: bool     # True if recall_count >= permanent_threshold
    is_protected: bool     # True if within protection period

class DecayEngine:
    def __init__(self, config: DecayConfig = None): ...
    def compute_weight(
        self,
        created_at: datetime,
        recall_count: int = 0,
        last_recall: datetime | None = None,
        now: datetime | None = None,  # for testing
    ) -> DecayResult: ...
```

### Dependencies on Existing Code
- `memx/config.py:DecayConfig` — 已定义，含 half_life_days, boost_factor, protection_days, permanent_threshold, archive_threshold
- `memx/types.py:BulletMetadata` — decay_weight, recall_count, last_recall, created_at 字段已定义

### Edge Cases
- age_days = 0（刚创建）→ weight = 1.0
- age_days 非常大（365天+）→ weight 接近 0
- recall_count 远超 permanent_threshold → 仍 weight = 1.0
- half_life_days 配置为极小值（如 0.1）→ 快速衰退但不报错
- created_at 在未来（时钟偏移）→ 视为 age=0，保护期逻辑生效

---

## Dependencies

**Prerequisite Stories:**
- STORY-001: BulletMetadata 模型 ✓（已完成）
- STORY-003: MemXConfig / DecayConfig ✓（已完成）

**Blocked Stories:**
- STORY-021: Decay sweep + reinforce（依赖本 story 的 compute_weight）
- STORY-022: Decay 测试（依赖本 story + STORY-021）

---

## Definition of Done

- [ ] `memx/engines/decay/formulas.py` 实现纯函数衰退公式
- [ ] `memx/engines/decay/engine.py` 实现 DecayEngine.compute_weight()
- [ ] 单元测试覆盖全部 acceptance criteria
- [ ] mypy --strict 通过
- [ ] ruff check 通过
- [ ] 覆盖率 > 90%

---

## Story Points Breakdown

- **衰退公式:** 1 point
- **DecayEngine 逻辑:** 2 points
- **测试:** 1 point（基础测试，完整测试在 STORY-022）
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
