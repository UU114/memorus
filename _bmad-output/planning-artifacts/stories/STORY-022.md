# STORY-022: Decay 单元测试

**Epic:** EPIC-004 (Decay 衰退引擎)
**Priority:** Must Have
**Story Points:** 5
**Status:** Done
**Assigned To:** Claude
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As a QA engineer
I want Decay engine thoroughly tested
So that forgetting and reinforcement are predictable

---

## Description

### Background
Decay 引擎的衰退公式和 sweep/reinforce 逻辑直接影响记忆的生命周期管理。需要全面的测试确保公式精度、边界条件、批量处理和异步 reinforce 行为正确。

### Scope
**In scope:**
- 衰退公式数值精度测试
- 保护期边界测试
- 永久保留阈值测试
- 归档阈值测试
- sweep 批量测试
- reinforce 测试
- 配置变更对衰退行为的影响测试

**Out of scope:**
- 性能基准测试（STORY-045）
- 与 RetrievalPipeline 的集成测试（STORY-030）

---

## Acceptance Criteria

- [x] 衰退公式数值精度测试：已知输入 → 已知输出（至少 5 组数据点）
- [x] 保护期边界测试：第 6 天 vs 第 8 天（默认 7 天保护期）
- [x] 永久保留阈值测试：recall_count=14 衰退 vs recall_count=15 永久
- [x] 归档阈值测试：weight=0.021 保留 vs weight=0.019 归档
- [x] sweep 批量测试（100+ 条混合状态记忆）
- [x] sweep 单条失败不影响整体测试
- [x] reinforce 回调成功/失败测试
- [x] 配置自定义测试（自定义 half_life, boost_factor 等）
- [x] 覆盖率 > 90%

---

## Technical Notes

### Test File
- `tests/unit/test_decay.py`

### Test Structure

```python
class TestExponentialDecay:
    """Test formulas.exponential_decay()"""
    # age=0 → 1.0
    # age=half_life → 0.5
    # age=2*half_life → 0.25

class TestBoostedWeight:
    """Test formulas.boosted_weight()"""
    # recall=0 → base unchanged
    # recall=10, boost=0.1 → base * 2.0

class TestDecayEngineComputeWeight:
    """Test DecayEngine.compute_weight()"""
    # protection period
    # permanent retention
    # archive threshold
    # normal decay curve

class TestDecayEngineSweep:
    """Test DecayEngine.sweep()"""
    # empty list
    # mixed states (protected, permanent, decaying, archived)
    # error isolation

class TestDecayEngineReinforce:
    """Test DecayEngine.reinforce()"""
    # successful reinforce
    # callback failure
    # empty list
```

### Key Test Data Points
- half_life=30, age=0 → weight=1.0
- half_life=30, age=30 → weight≈0.5
- half_life=30, age=60 → weight≈0.25
- half_life=30, age=90 → weight≈0.125
- half_life=30, age=30, recall=10, boost=0.1 → weight≈0.5 × 2.0 = 1.0 (clamped)

---

## Dependencies

**Prerequisite Stories:**
- STORY-020: DecayEngine 核心
- STORY-021: Decay sweep + reinforce

---

## Definition of Done

- [x] `tests/unit/test_decay.py` 全部通过
- [x] 覆盖率 > 90%
- [x] 所有 acceptance criteria 对应至少 1 个测试用例
- [x] ruff check 通过

---

## Story Points Breakdown

- **公式测试:** 1 point
- **compute_weight 测试:** 1.5 points
- **sweep 测试:** 1.5 points
- **reinforce 测试:** 1 point
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
