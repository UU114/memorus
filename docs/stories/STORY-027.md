# STORY-027: 实现 ScoreMerger 综合评分

**Epic:** EPIC-005 (Generator 混合检索引擎)
**Priority:** Must Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As a search engine
I want a unified scoring formula
So that keyword and semantic results are properly blended

---

## Description

### Background
ScoreMerger 将 L1-L3 的 keyword 分数和 L4 的 semantic 分数混合，再乘以衰退权重和时效性加分，得到最终排序分数。支持降级模式（无 semantic 分数时仅用 keyword 分数）。

### Scope
**In scope:**
- 综合评分公式实现
- keyword/semantic 权重配置
- RecencyBoost 时效性加分
- DecayWeight 衰退权重加权
- 降级模式（无 semantic 分数）

**Out of scope:**
- Reranker（可能未来增强）

---

## Acceptance Criteria

- [ ] 公式：`FinalScore = (KeywordScore × kw_weight + SemanticScore × sem_weight) × DecayWeight × RecencyBoost`
- [ ] kw_weight / sem_weight 通过 RetrievalConfig 配置（默认 0.6 / 0.4）
- [ ] 权重不等于 1.0 时自动归一化（如 0.6+0.4=1.0 已归一化）
- [ ] RecencyBoost：7 天内 ×1.2（天数和倍率可配置）
- [ ] 降级模式：无 SemanticScore 时，KeywordScore 权重自动变为 1.0
- [ ] FinalScore 按降序排列返回

---

## Technical Notes

### API Design

```python
@dataclass
class ScoredBullet:
    """Final scored result combining all layers."""
    bullet_id: str
    content: str
    final_score: float
    keyword_score: float        # L1+L2+L3 combined
    semantic_score: float       # L4
    decay_weight: float
    recency_boost: float
    metadata: dict = field(default_factory=dict)

class ScoreMerger:
    def __init__(self, config: RetrievalConfig = None): ...

    def merge(
        self,
        keyword_results: dict[str, float],     # bullet_id → keyword score
        semantic_results: dict[str, float],     # bullet_id → semantic score
        bullet_infos: dict[str, BulletInfo],   # bullet_id → metadata for decay/recency
    ) -> list[ScoredBullet]: ...

    def compute_recency_boost(
        self, created_at: datetime, now: datetime | None = None
    ) -> float: ...
```

### Components
- `memx/engines/generator/score_merger.py`

### Scoring Formula Detail
```
KeywordScore = ExactMatcher.score + FuzzyMatcher.score + MetadataMatcher.score  (0-35)
SemanticScore = VectorSearcher.score  (0-1, normalized)

# Normalize to same scale
NormKeyword = KeywordScore / 35.0  (0-1)
NormSemantic = SemanticScore        (0-1)

# Weighted blend
BlendedScore = NormKeyword × kw_weight + NormSemantic × sem_weight

# Apply modifiers
FinalScore = BlendedScore × DecayWeight × RecencyBoost
```

---

## Dependencies

**Prerequisite Stories:**
- STORY-023~026: 四个 Matcher
- STORY-020: Decay（用于 decay_weight）

**Blocked Stories:**
- STORY-028: GeneratorEngine

---

## Definition of Done

- [ ] `memx/engines/generator/score_merger.py` 实现
- [ ] 全模式评分测试（full + degraded）
- [ ] RecencyBoost 计算测试
- [ ] 权重归一化测试
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **评分公式:** 1.5 points
- **RecencyBoost:** 1 point
- **降级模式:** 1 point
- **测试:** 0.5 points
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
