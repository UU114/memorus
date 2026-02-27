# STORY-010: 实现 KnowledgeScorer（Stage 2）

**Epic:** EPIC-002 — Reflector 知识蒸馏引擎
**Priority:** Must Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a Reflector engine
I want candidates scored and classified
So that only valuable knowledge enters the Playbook

---

## Description

### Background
KnowledgeScorer 是 Reflector 的 Stage 2，接收 PatternDetector 产出的 DetectedPattern，为其分配 knowledge_type（5 种分类）和 section（8 种分区），并计算 instructivity_score（教学价值评分）。低于阈值的候选被过滤，不进入后续阶段。这是"质量守门员"，确保只有有价值的知识才被记住。

### Scope

**In scope:**
- knowledge_type 分类逻辑（rules-based）
- section 分区分配逻辑
- instructivity_score 计算公式
- 低分过滤（< min_score 阈值）
- 所有参数通过 ReflectorConfig 可配置

**Out of scope:**
- LLM 辅助评分（Sprint 4）
- 自定义评分函数

### Scoring Formula

```
instructivity_score = base_score × density_penalty + distill_bonus

Where:
  base_score = pattern_confidence × 100      (from DetectedPattern)
  density_penalty = min(1.0, word_count / 20) (penalize very short content)
  distill_bonus = +10 if content contains actionable keywords
                  +5 if content contains specific tool/file names
```

### Classification Rules

**knowledge_type mapping:**
```
error_fix      → "pitfall" (陷阱/坑)
retry_success  → "method" (方法)
config_change  → "preference" (偏好)
new_tool       → "trick" (技巧)
repetitive_op  → "method" (方法)
unknown        → "knowledge" (通用知识)
```

**section assignment (keyword-based):**
```
contains "debug", "error", "fix", "crash"  → "debugging"
contains "install", "config", "env", "setup" → "workflow"
contains "git", "docker", "npm", "pip"     → "tools"
contains "architect", "design", "pattern"  → "architecture"
contains command-like patterns             → "commands"
contains "prefer", "always", "never"       → "preferences"
default                                    → "general"
```

---

## Acceptance Criteria

- [ ] `KnowledgeScorer.score(pattern: DetectedPattern) -> ScoredCandidate`
- [ ] ScoredCandidate 包含：knowledge_type, section, instructivity_score, raw_content, context
- [ ] knowledge_type 正确映射（error_fix→pitfall, retry→method 等）
- [ ] section 基于关键词正确分配（≥ 6 种分区有对应关键词）
- [ ] instructivity_score 计算：base × density_penalty + distill_bonus
- [ ] score < min_score（默认 30）的候选返回 None（被过滤）
- [ ] min_score 阈值通过 ReflectorConfig.min_score 可配置
- [ ] 测试：每种 knowledge_type 分类 ≥ 1 个用例
- [ ] 测试：评分边界（刚好过阈值 / 刚好低于阈值）

---

## Technical Notes

### File Location
`memx/engines/reflector/scorer.py`

### Implementation Sketch

```python
from typing import Optional
from memx.types import (
    DetectedPattern, BulletSection, KnowledgeType,
)
from memx.config import ReflectorConfig

class ScoredCandidate:
    knowledge_type: KnowledgeType
    section: BulletSection
    instructivity_score: float
    raw_content: str
    context: dict

class KnowledgeScorer:
    # Pattern type → knowledge type
    TYPE_MAP = {
        "error_fix": KnowledgeType.PITFALL,
        "retry_success": KnowledgeType.METHOD,
        "config_change": KnowledgeType.PREFERENCE,
        "new_tool": KnowledgeType.TRICK,
        "repetitive_op": KnowledgeType.METHOD,
    }

    # Section keywords
    SECTION_KEYWORDS = {
        BulletSection.DEBUGGING: {"debug", "error", "fix", "crash", "traceback", "exception"},
        BulletSection.WORKFLOW: {"install", "config", "env", "setup", "deploy", "build"},
        BulletSection.TOOLS: {"git", "docker", "npm", "pip", "cargo", "brew"},
        BulletSection.ARCHITECTURE: {"architect", "design", "pattern", "refactor", "structure"},
        BulletSection.COMMANDS: {"run", "exec", "command", "cli", "terminal", "shell"},
        BulletSection.PREFERENCES: {"prefer", "always", "never", "default", "convention"},
    }

    ACTIONABLE_KEYWORDS = {"use", "run", "try", "avoid", "instead", "should", "must", "tip"}

    def __init__(self, config: ReflectorConfig):
        self._min_score = config.min_score

    def score(self, pattern: DetectedPattern) -> Optional[ScoredCandidate]:
        knowledge_type = self._classify_type(pattern)
        section = self._assign_section(pattern.raw_content)
        score = self._compute_score(pattern)

        if score < self._min_score:
            return None

        return ScoredCandidate(
            knowledge_type=knowledge_type,
            section=section,
            instructivity_score=score,
            raw_content=pattern.raw_content,
            context=pattern.context,
        )

    def _compute_score(self, pattern: DetectedPattern) -> float:
        base = pattern.confidence * 100
        words = len(pattern.raw_content.split())
        density_penalty = min(1.0, words / 20)
        distill_bonus = 0
        content_lower = pattern.raw_content.lower()
        if any(kw in content_lower for kw in self.ACTIONABLE_KEYWORDS):
            distill_bonus += 10
        # Check for specific tool/file names
        if any(c in content_lower for c in {".", "/", "\\"}):
            distill_bonus += 5
        return base * density_penalty + distill_bonus
```

### Edge Cases
- pattern.confidence = 0 → score = 0 + bonus → 可能仍被过滤
- 极短内容（1-2 词）→ density_penalty 很低 → score 低 → 大概率被过滤（期望行为）
- 多个 section 关键词匹配 → 取第一个匹配的（优先级：debugging > workflow > tools > ...）
- pattern_type 不在 TYPE_MAP 中 → 默认 "knowledge"

---

## Dependencies

**Prerequisite Stories:**
- STORY-008: PatternDetector（输出 DetectedPattern）
- STORY-001: 枚举类型定义

**Blocked Stories:**
- STORY-012: BulletDistiller（接收 ScoredCandidate）
- STORY-013: ReflectorEngine（调用 KnowledgeScorer）

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented in `memx/engines/reflector/scorer.py`
- [ ] Unit tests in `tests/unit/test_reflector.py` (scorer section)
- [ ] ≥ 10 个测试用例
- [ ] All tests passing
- [ ] `ruff check` + `mypy` 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Classification logic:** 1 point
- **Scoring formula:** 1.5 points
- **Config integration:** 0.5 points
- **Tests:** 1 point
- **Total:** 4 points

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
