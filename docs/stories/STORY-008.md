# STORY-008: 实现 PatternDetector（Stage 1）— 基础框架

**Epic:** EPIC-002 — Reflector 知识蒸馏引擎
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a MemX engine
I want to detect learnable patterns from AI interactions
So that valuable knowledge can be captured automatically

---

## Description

### Background
PatternDetector 是 Reflector 引擎的 Stage 1（第一阶段），负责从 AI 交互中自动识别可学习的模式。它采用纯规则方式（零 LLM 调用），通过分析工具调用的输入/输出/成败来判断是否包含有价值的知识。这是"学到了"流程的入口——只有被 PatternDetector 识别的交互才会进入后续的评分、脱敏和蒸馏阶段。

### Scope

**In scope:**
- InteractionEvent 数据结构（定义在 STORY-001 的 types.py 中）
- DetectedPattern 数据结构
- PatternDetector.detect() 方法框架
- 2 种初始模式规则：错误修复模式、命令失败后成功模式
- 代码占比检测（> 60% 代码内容 → 拒绝）
- 单元测试

**Out of scope:**
- 更多模式规则（STORY-009）
- KnowledgeScorer 评分（STORY-010）
- LLM 增强检测（Sprint 4 buffer）

### Pattern Detection Rules

**Rule 1: Error Fix Pattern (错误修复模式)**
```
条件：
  - event.success == True
  - event.error_msg is not None (之前有错误)
  - 或者：上一次相同 tool 调用失败，这次成功
逻辑：
  用户遇到错误 → 找到解决方案 → 值得记住
```

**Rule 2: Command Retry Success (命令重试成功)**
```
条件：
  - 同一 tool_name 在短时间内（< 5 分钟）被调用 ≥ 2 次
  - 最后一次 success == True
  - 中间有 success == False
逻辑：
  命令失败后调整参数/方法重试成功 → 值得记住
```

**Code Content Filter (代码占比过滤器)**
```
条件：
  - 内容中代码行数 / 总行数 > 0.6
逻辑：
  主要是代码粘贴，不是可学习的知识 → 拒绝
检测方法：
  - 统计缩进行（以空格/tab 开头）
  - 统计包含特殊符号的行（{, }, =, ;, //）
  - 统计短行（< 3 个单词）
```

---

## Acceptance Criteria

- [ ] InteractionEvent 数据结构已在 `memx/types.py` 中定义（tool_name, input, output, success, error_msg, timestamp）
- [ ] DetectedPattern 数据结构定义完成（pattern_type, raw_content, context, confidence）
- [ ] PatternDetector 类实现 `detect(event: InteractionEvent) -> list[DetectedPattern]`
- [ ] 支持错误修复模式检测：成功的事件且有关联错误 → 返回 DetectedPattern
- [ ] 支持命令重试成功检测：需要维护最近 N 次事件上下文
- [ ] 代码占比 > 60% 的内容被拒绝（返回空列表）
- [ ] detect() 不抛异常（内部 try-catch，失败返回空列表）
- [ ] 单元测试：每种模式至少 3 个用例（正例 + 反例 + 边界）
- [ ] 代码占比检测测试

---

## Technical Notes

### File Location
`memx/engines/reflector/detector.py`

### Implementation Sketch

```python
import logging
from typing import Optional
from collections import deque
from memx.types import InteractionEvent, DetectedPattern

logger = logging.getLogger(__name__)

class PatternDetector:
    """Stage 1 of Reflector - detect learnable patterns from interactions."""

    def __init__(self, max_history: int = 20):
        self._history: deque[InteractionEvent] = deque(maxlen=max_history)

    def detect(self, event: InteractionEvent) -> list[DetectedPattern]:
        """Detect patterns from interaction event. Never raises."""
        try:
            self._history.append(event)
            patterns = []

            # Filter: reject code-heavy content
            if self._is_code_heavy(event.output or ""):
                return []

            # Rule 1: Error fix pattern
            if fix := self._detect_error_fix(event):
                patterns.append(fix)

            # Rule 2: Command retry success
            if retry := self._detect_retry_success(event):
                patterns.append(retry)

            return patterns
        except Exception as e:
            logger.warning(f"PatternDetector.detect failed: {e}")
            return []

    def _detect_error_fix(self, event: InteractionEvent) -> Optional[DetectedPattern]:
        """Detect: success with prior error context."""
        if not event.success:
            return None
        if event.error_msg:
            return DetectedPattern(
                pattern_type="error_fix",
                raw_content=f"Error: {event.error_msg}\nFix: {event.output}",
                context={"tool": event.tool_name, "input": event.input},
                confidence=0.8,
            )
        return None

    def _detect_retry_success(self, event: InteractionEvent) -> Optional[DetectedPattern]:
        """Detect: same tool failed then succeeded."""
        if not event.success:
            return None
        # Look back in history for same tool failures
        ...

    @staticmethod
    def _is_code_heavy(content: str, threshold: float = 0.6) -> bool:
        """Reject content that is mostly code."""
        ...
```

### Event History Design
- 使用 `collections.deque(maxlen=20)` 维护最近事件
- PatternDetector 是有状态的（需要跨事件检测模式）
- 清理历史：SessionEnd 时清空

### Edge Cases
- event.output 为 None → 跳过代码占比检测
- event.tool_name 为 None → 跳过重试检测
- 空 history 列表 → 无法做重试检测，跳过 Rule 2
- 所有检测失败 → 返回空 list（不是 None）

---

## Dependencies

**Prerequisite Stories:**
- STORY-001: BulletMetadata + InteractionEvent 类型
- STORY-006: 项目骨架

**Blocked Stories:**
- STORY-009: PatternDetector 扩展（在此基础上添加更多规则）
- STORY-010: KnowledgeScorer（处理 DetectedPattern）
- STORY-013: ReflectorEngine（调用 PatternDetector）

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented in `memx/engines/reflector/detector.py`
- [ ] Unit tests in `tests/unit/test_reflector.py` (detector section)
- [ ] ≥ 9 个测试用例（2 规则 × 3 + 3 个代码过滤测试）
- [ ] All tests passing
- [ ] `ruff check` 通过
- [ ] `mypy` 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Data structures:** 0.5 points
- **PatternDetector framework:** 1.5 points
- **2 detection rules:** 1.5 points
- **Code filter:** 0.5 points
- **Tests:** 1 point
- **Total:** 5 points

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
