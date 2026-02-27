# STORY-013: 组装 ReflectorEngine 完整流水线

**Epic:** EPIC-002 — Reflector 知识蒸馏引擎
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As an IngestPipeline
I want a single ReflectorEngine entry point
So that I can call reflect() and get candidate Bullets

---

## Description

### Background
ReflectorEngine 是 Reflector 引擎的顶层编排器，将 Stage 1-4（PatternDetector → KnowledgeScorer → PrivacySanitizer → BulletDistiller）组装为一个完整流水线。它提供单一入口 `reflect(event)` 方法，内部编排四个阶段的顺序执行。关键架构要求：每个 Stage 有独立 try-catch 边界——任何一个 Stage 故障只跳过该 Stage，不会导致整条管线崩溃。

### Scope

**In scope:**
- ReflectorEngine 类：编排 4 个 Stage
- reflect() 方法：接收 InteractionEvent，返回 CandidateBullet 列表
- 三种运行模式配置："rules"（默认）/ "llm" / "hybrid"
- rules 模式零 LLM 调用验证
- 每个 Stage 独立 try-catch（NFR-006 优雅降级）
- 集成测试：完整 4-Stage 流水线

**Out of scope:**
- LLM 模式和 Hybrid 模式的实现（Sprint 4 buffer）
- Pipeline 集成（STORY-014）

### Stage Flow

```
InteractionEvent
    │
    ▼
[Stage 1: PatternDetector]
    │ DetectedPattern list
    │ (failure → return [])
    ▼
[Stage 2: KnowledgeScorer]
    │ ScoredCandidate list (filtered)
    │ (failure → skip scoring, pass raw patterns)
    ▼
[Stage 3: PrivacySanitizer]
    │ Sanitized content
    │ (failure → use original content with WARNING)
    ▼
[Stage 4: BulletDistiller]
    │ CandidateBullet list
    │ (failure → create minimal bullet from raw)
    ▼
Return CandidateBullet[]
```

---

## Acceptance Criteria

- [ ] `ReflectorEngine.reflect(event: InteractionEvent) -> list[CandidateBullet]`
- [ ] 编排 Stage 1→2→3→4 顺序执行
- [ ] 支持 mode 配置："rules"（默认）/ "llm" / "hybrid"
- [ ] rules 模式下零 LLM 调用（可通过 mock 验证）
- [ ] llm/hybrid 模式暂时 fallback 到 rules（log WARNING）
- [ ] Stage 1 异常 → 返回空列表（log WARNING）
- [ ] Stage 2 异常 → 跳过评分，使用默认分数和分类
- [ ] Stage 3 异常 → 使用原始内容（log WARNING "privacy sanitizer failed"）
- [ ] Stage 4 异常 → 创建最小化 Bullet（仅含 content 和默认元数据）
- [ ] 集成测试：完整流水线，从 InteractionEvent 到 CandidateBullet
- [ ] 集成测试：单个 Stage 故障时的降级行为

---

## Technical Notes

### File Location
`memx/engines/reflector/engine.py`

### Implementation Sketch

```python
import logging
from typing import Optional
from memx.types import InteractionEvent, CandidateBullet, DetectedPattern
from memx.config import ReflectorConfig
from memx.engines.reflector.detector import PatternDetector
from memx.engines.reflector.scorer import KnowledgeScorer
from memx.privacy.sanitizer import PrivacySanitizer
from memx.engines.reflector.distiller import BulletDistiller

logger = logging.getLogger(__name__)

class ReflectorEngine:
    """Orchestrates the 4-stage knowledge distillation pipeline."""

    def __init__(self, config: ReflectorConfig,
                 sanitizer: PrivacySanitizer,
                 llm=None):
        self._config = config
        self._detector = PatternDetector()
        self._scorer = KnowledgeScorer(config)
        self._sanitizer = sanitizer
        self._distiller = BulletDistiller(config)
        self._llm = llm
        self._mode = config.mode

        if self._mode in ("llm", "hybrid") and llm is None:
            logger.warning("LLM not available, falling back to rules mode")
            self._mode = "rules"

    def reflect(self, event: InteractionEvent) -> list[CandidateBullet]:
        """Run 4-stage distillation. Each stage has independent failure boundary."""

        # Stage 1: Pattern Detection
        patterns = self._run_stage1(event)
        if not patterns:
            return []

        # Stage 2: Knowledge Scoring
        scored = self._run_stage2(patterns)
        if not scored:
            return []

        # Stage 3: Privacy Sanitization
        sanitized = self._run_stage3(scored)

        # Stage 4: Bullet Distillation
        bullets = self._run_stage4(sanitized)

        return bullets

    def _run_stage1(self, event: InteractionEvent) -> list[DetectedPattern]:
        try:
            return self._detector.detect(event)
        except Exception as e:
            logger.warning(f"Stage 1 (PatternDetector) failed: {e}")
            return []

    def _run_stage2(self, patterns: list[DetectedPattern]) -> list:
        try:
            scored = []
            for p in patterns:
                if s := self._scorer.score(p):
                    scored.append(s)
            return scored
        except Exception as e:
            logger.warning(f"Stage 2 (KnowledgeScorer) failed: {e}")
            # Fallback: pass patterns with default scores
            return self._fallback_scoring(patterns)

    def _run_stage3(self, candidates: list) -> list:
        try:
            for c in candidates:
                result = self._sanitizer.sanitize(c.raw_content)
                c.raw_content = result.clean_content
            return candidates
        except Exception as e:
            logger.warning(f"Stage 3 (PrivacySanitizer) failed: {e}")
            return candidates  # Use unsanitized (risky but non-blocking)

    def _run_stage4(self, candidates: list) -> list[CandidateBullet]:
        try:
            return [self._distiller.distill(c) for c in candidates]
        except Exception as e:
            logger.warning(f"Stage 4 (BulletDistiller) failed: {e}")
            return self._fallback_distill(candidates)
```

### Key Design Decisions
1. **Independent try-catch per stage** — 最重要的 NFR-006 实现
2. **Stage 3 失败时不阻塞** — 虽然有隐私风险，但 Sanitizer 本身有内部 try-catch，只有极端情况才会到这里
3. **Fallback 方法** — 每个 Stage 的 fallback 都保证返回合法的下游输入格式

### Edge Cases
- event 为 None → reflect() 应处理，返回空列表
- LLM 在 reflect 过程中超时 → 只影响 LLM 模式，rules 模式不受影响
- 一个 event 产生多个 Pattern → 全部进入 Stage 2 评分
- 所有 Pattern 评分低于阈值 → Stage 2 返回空列表 → 最终返回空

---

## Dependencies

**Prerequisite Stories:**
- STORY-008: PatternDetector
- STORY-009: PatternDetector 扩展
- STORY-010: KnowledgeScorer
- STORY-011: PrivacySanitizer
- STORY-012: BulletDistiller

**Blocked Stories:**
- STORY-014: IngestPipeline（调用 ReflectorEngine）
- STORY-016: Reflector 测试全覆盖（需要完整引擎）

**External Dependencies:** None

---

## Definition of Done

- [ ] Code in `memx/engines/reflector/engine.py`
- [ ] Integration tests in `tests/unit/test_reflector.py` (engine section)
- [ ] 完整 4-Stage 流水线测试
- [ ] 每个 Stage 独立故障降级测试（4 个）
- [ ] rules 模式零 LLM 调用验证
- [ ] `ruff check` + `mypy` 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Engine orchestration:** 2 points
- **Fallback logic (4 stages):** 1.5 points
- **Integration tests:** 1.5 points
- **Total:** 5 points

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
