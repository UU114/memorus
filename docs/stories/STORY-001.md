# STORY-001: 定义 BulletMetadata Pydantic 模型

**Epic:** EPIC-001 — Bullet 数据模型与配置基础
**Priority:** Must Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a MemX developer
I want a well-defined BulletMetadata data model
So that all ACE engines have a consistent knowledge unit structure

---

## Description

### Background
Bullet 是 ACE 系统的最小知识单元。每一条 MemX 记忆都包含一个 BulletMetadata 对象，附加在 mem0 的 payload metadata 中（使用 `memx_` 前缀）。所有四大引擎（Reflector, Curator, Decay, Generator）都围绕 Bullet 的字段进行操作。因此，数据模型的精确定义是整个系统的基石。

### Scope

**In scope:**
- BulletMetadata Pydantic 模型（所有 ACE 字段）
- 枚举类型：BulletSection, KnowledgeType, SourceType
- 辅助数据类型：InteractionEvent, DetectedPattern, ScoredCandidate, CandidateBullet
- 所有字段的合理默认值
- 单元测试：模型校验、序列化、反序列化

**Out of scope:**
- BulletFactory（STORY-002）
- mem0 payload 转换逻辑（STORY-002）

### Data Model Design

参考 ACE Universal Solution 和架构文档，BulletMetadata 包含以下字段：

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| section | BulletSection | "general" | 知识分区 |
| knowledge_type | KnowledgeType | "knowledge" | 知识类型 |
| instructivity_score | float | 50.0 | 教学价值评分 (0-100) |
| recall_count | int | 0 | 被召回次数 |
| last_recall | datetime | None | 最近一次召回时间 |
| decay_weight | float | 1.0 | 衰退权重 (0-1) |
| related_tools | list[str] | [] | 关联工具名称 |
| related_files | list[str] | [] | 关联文件路径 |
| key_entities | list[str] | [] | 关键实体 |
| tags | list[str] | [] | 用户标签 |
| distilled_rule | str | None | 蒸馏后的规则文本 |
| source_type | SourceType | "interaction" | 来源类型 |
| scope | str | "global" | 作用域 |
| created_at | datetime | now() | 创建时间 |
| updated_at | datetime | now() | 更新时间 |

---

## Acceptance Criteria

- [ ] BulletMetadata Pydantic 模型包含上述所有字段
- [ ] BulletSection 枚举包含 ≥ 8 种分区：`commands`, `debugging`, `architecture`, `workflow`, `tools`, `patterns`, `preferences`, `general`
- [ ] KnowledgeType 枚举包含 5 种类型：`method`, `trick`, `pitfall`, `preference`, `knowledge`
- [ ] SourceType 枚举包含 3 种来源：`interaction`, `manual`, `import`
- [ ] 所有字段有合理默认值（无必填字段，可零参构造 `BulletMetadata()`）
- [ ] instructivity_score 校验范围 0-100（超范围抛 ValidationError）
- [ ] decay_weight 校验范围 0-1
- [ ] recall_count 校验 ≥ 0
- [ ] `model_dump()` 和 `model_validate()` 工作正常
- [ ] datetime 字段序列化为 ISO 格式
- [ ] 辅助类型 InteractionEvent, DetectedPattern 等数据类定义完成
- [ ] 单元测试覆盖：创建、校验、序列化、反序列化、非法值拒绝

---

## Technical Notes

### File Location
`memx/types.py`

### Implementation Sketch

```python
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class BulletSection(str, Enum):
    COMMANDS = "commands"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    WORKFLOW = "workflow"
    TOOLS = "tools"
    PATTERNS = "patterns"
    PREFERENCES = "preferences"
    GENERAL = "general"

class KnowledgeType(str, Enum):
    METHOD = "method"
    TRICK = "trick"
    PITFALL = "pitfall"
    PREFERENCE = "preference"
    KNOWLEDGE = "knowledge"

class SourceType(str, Enum):
    INTERACTION = "interaction"
    MANUAL = "manual"
    IMPORT = "import"

class BulletMetadata(BaseModel):
    section: BulletSection = BulletSection.GENERAL
    knowledge_type: KnowledgeType = KnowledgeType.KNOWLEDGE
    instructivity_score: float = Field(default=50.0, ge=0.0, le=100.0)
    recall_count: int = Field(default=0, ge=0)
    last_recall: datetime | None = None
    decay_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    related_tools: list[str] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    distilled_rule: str | None = None
    source_type: SourceType = SourceType.INTERACTION
    scope: str = "global"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
```

### Python 3.9 Compatibility
- 使用 `Optional[datetime]` 而非 `datetime | None`（union syntax 需要 3.10+）
- 使用 `from __future__ import annotations` 或 `typing.Optional`

### Edge Cases
- `BulletMetadata()` 无参构造必须成功（所有字段有默认值）
- `instructivity_score=101` 或 `-1` 必须抛 ValidationError
- 空字符串的 `scope` 应该允许还是拒绝？→ 允许，由上层逻辑判断

---

## Dependencies

**Prerequisite Stories:**
- STORY-006: 项目骨架（需要 `memx/types.py` 文件路径存在）

**Blocked Stories:**
- STORY-002: BulletFactory（需要 BulletMetadata）
- STORY-003: MemXConfig（需要了解 Bullet 字段用于配置设计）
- STORY-008: PatternDetector（需要 InteractionEvent, DetectedPattern）
- STORY-010: KnowledgeScorer（需要 KnowledgeType, BulletSection）
- STORY-012: BulletDistiller（需要 CandidateBullet）
- STORY-020: DecayEngine（需要 BulletMetadata 的 decay 字段）

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented in `memx/types.py`
- [ ] Unit tests in `tests/unit/test_types.py`
- [ ] All tests passing
- [ ] `ruff check memx/types.py` 通过
- [ ] `mypy memx/types.py` 通过
- [ ] Acceptance criteria all validated
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Enums + BulletMetadata:** 1.5 points
- **Auxiliary types:** 0.5 points
- **Unit tests:** 1 point
- **Total:** 3 points

**Rationale:** 纯数据模型定义，无复杂逻辑，但字段数量较多且需要全面测试。

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
