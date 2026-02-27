# STORY-009: 扩展 PatternDetector — 更多模式规则

**Epic:** EPIC-002 — Reflector 知识蒸馏引擎
**Priority:** Must Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a MemX user
I want more interaction patterns detected
So that more useful knowledge is captured

---

## Description

### Background
STORY-008 建立了 PatternDetector 框架和 2 种基础模式。本 Story 在此基础上扩展 3 种新模式，使总计达到 ≥ 5 种，并设计可扩展的模式规则注册机制。

### Scope

**In scope:**
- 3 种新模式规则
- 模式规则的可扩展设计（子类/注册机制）
- 每种模式独立测试

**Out of scope:**
- LLM 增强检测（Sprint 4）
- 自定义用户规则（未来功能）

### New Pattern Rules

**Rule 3: Configuration Change Pattern (配置变更模式)**
```
条件：
  - tool_name 包含 "config", "setting", "env" 等关键词
  - 或 input 中包含配置文件路径 (.env, .yaml, .json, .toml 等)
  - success == True
逻辑：
  用户修改了配置 → 可能是环境偏好 → 记住
```

**Rule 4: New Tool Discovery (新工具发现模式)**
```
条件：
  - tool_name 首次出现在历史中
  - success == True
  - output 内容长度 > 一定阈值（说明有实质输出）
逻辑：
  用户使用了新工具/命令并成功 → 值得记住
```

**Rule 5: Repetitive Operation (重复操作模式)**
```
条件：
  - 相同 tool_name + 相似 input 出现 ≥ 3 次
  - 大部分 success == True
逻辑：
  用户反复做相同操作 → 可能是常用工作流 → 记住
相似判定：
  - input 去除变量部分后的模板相同
  - 或 Levenshtein distance < 阈值
```

---

## Acceptance Criteria

- [ ] 配置变更模式：检测含配置关键词的成功工具调用
- [ ] 新工具发现模式：检测历史中首次出现的工具名称
- [ ] 重复操作模式：检测 ≥ 3 次相同工具相似输入的模式
- [ ] 总计 ≥ 5 种模式（含 STORY-008 的 2 种）
- [ ] 模式规则可通过子类扩展（基类 `PatternRule` 或类似机制）
- [ ] 每种新模式 ≥ 3 个独立测试用例（正例 + 反例 + 边界）
- [ ] 新规则不破坏已有 STORY-008 的测试

---

## Technical Notes

### File Location
`memx/engines/reflector/patterns.py` (新文件，规则定义)
`memx/engines/reflector/detector.py` (修改，集成新规则)

### Extensible Pattern Architecture

```python
from abc import ABC, abstractmethod

class PatternRule(ABC):
    """Base class for all pattern detection rules."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def check(self, event: InteractionEvent,
              history: list[InteractionEvent]) -> Optional[DetectedPattern]: ...

class ErrorFixRule(PatternRule):
    name = "error_fix"
    def check(self, event, history): ...

class ConfigChangeRule(PatternRule):
    name = "config_change"
    CONFIG_KEYWORDS = {"config", "setting", "env", "configure", "setup"}
    CONFIG_EXTENSIONS = {".env", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}
    def check(self, event, history): ...

class NewToolRule(PatternRule):
    name = "new_tool"
    def check(self, event, history): ...

class RepetitiveOpRule(PatternRule):
    name = "repetitive_op"
    MIN_OCCURRENCES = 3
    def check(self, event, history): ...
```

### Detector Integration

```python
class PatternDetector:
    def __init__(self, rules: list[PatternRule] = None):
        self._rules = rules or self._default_rules()
        self._history = deque(maxlen=20)

    @staticmethod
    def _default_rules() -> list[PatternRule]:
        return [
            ErrorFixRule(),
            RetrySuccessRule(),
            ConfigChangeRule(),
            NewToolRule(),
            RepetitiveOpRule(),
        ]

    def detect(self, event):
        patterns = []
        for rule in self._rules:
            if not self._is_code_heavy(event.output or ""):
                if p := rule.check(event, list(self._history)):
                    patterns.append(p)
        ...
```

### Edge Cases
- 工具名称 None 或空字符串 → 跳过 new_tool 和 repetitive_op
- 配置文件路径在 Windows/Linux 格式差异 → 统一使用 Path 判断
- 重复操作的"相似"判定 → 简单实现先用精确匹配，后续可升级

---

## Dependencies

**Prerequisite Stories:**
- STORY-008: PatternDetector 基础框架

**Blocked Stories:**
- STORY-013: ReflectorEngine（需要完整的 PatternDetector）

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented in `memx/engines/reflector/patterns.py`
- [ ] PatternDetector 更新为使用 PatternRule 可扩展机制
- [ ] 3 种新规则实现
- [ ] ≥ 9 个新测试用例
- [ ] 所有旧测试仍通过
- [ ] `ruff check` + `mypy` 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **PatternRule 抽象:** 1 point
- **3 new rules:** 1.5 points
- **Detector refactor:** 0.5 points
- **Tests:** 1 point
- **Total:** 4 points

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
