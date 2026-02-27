# STORY-002: 实现 BulletFactory 工厂类

**Epic:** EPIC-001 — Bullet 数据模型与配置基础
**Priority:** Must Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a MemX engine
I want a factory to create, serialize, and deserialize Bullets
So that Bullet creation is standardized across all modules

---

## Description

### Background
BulletFactory 是 Bullet 数据在 MemX 内部表示和 mem0 VectorStore payload 之间的桥梁。所有 ACE 引擎通过 Factory 创建标准化 Bullet；存储时通过 Factory 将 Bullet 字段转为 `memx_` 前缀的 dict 嵌入 mem0 payload；检索时通过 Factory 从 payload 解析还原 Bullet。向后兼容性至关重要——没有 `memx_` 字段的旧 mem0 payload 必须使用默认值而不报错。

### Scope

**In scope:**
- `BulletFactory.create()` — 创建带默认元数据的 Bullet
- `BulletFactory.to_mem0_metadata()` — Bullet → `memx_` 前缀 dict
- `BulletFactory.from_mem0_payload()` — mem0 payload → Bullet（优雅处理缺失字段）
- 向后兼容测试
- 单元测试

**Out of scope:**
- BulletMetadata 模型定义（STORY-001）
- Curator 合并逻辑（STORY-018）

### Key Design: `memx_` Prefix Convention

```python
# BulletMetadata field → mem0 payload key mapping
{
    "section": "general"           → {"memx_section": "general"}
    "knowledge_type": "method"     → {"memx_knowledge_type": "method"}
    "instructivity_score": 75.0    → {"memx_instructivity_score": 75.0}
    "recall_count": 3              → {"memx_recall_count": 3}
    "decay_weight": 0.85           → {"memx_decay_weight": 0.85}
    # ... all fields get memx_ prefix
}
```

---

## Acceptance Criteria

- [ ] `BulletFactory.create(content, **kwargs)` 创建 dict 含 `content` 和 `BulletMetadata`
- [ ] `BulletFactory.to_mem0_metadata(bullet_meta)` 将所有 BulletMetadata 字段转为 `memx_` 前缀 dict
- [ ] `BulletFactory.from_mem0_payload(payload)` 从 mem0 payload 的 `metadata` 字典中解析 `memx_*` 字段
- [ ] 旧 payload（无 `memx_` 字段）→ 返回带默认值的 BulletMetadata，不报错
- [ ] 部分字段存在的 payload → 已有字段正确解析，缺失字段使用默认值
- [ ] datetime 字段正确序列化/反序列化（ISO 格式字符串 ↔ datetime 对象）
- [ ] list 字段（related_tools, tags 等）序列化为 JSON 字符串存储
- [ ] 单元测试覆盖：创建、序列化、反序列化、向后兼容、部分字段

---

## Technical Notes

### File Location
`memx/utils/bullet_factory.py`

### Implementation Sketch

```python
from memx.types import BulletMetadata

MEMX_PREFIX = "memx_"

class BulletFactory:
    @staticmethod
    def create(content: str, **kwargs) -> dict:
        """Create a new memory dict with Bullet metadata."""
        meta = BulletMetadata(**kwargs)
        return {"content": content, "metadata": meta}

    @staticmethod
    def to_mem0_metadata(bullet_meta: BulletMetadata) -> dict:
        """Convert BulletMetadata to memx_-prefixed dict for mem0 payload."""
        data = bullet_meta.model_dump(mode="json")
        return {f"{MEMX_PREFIX}{k}": v for k, v in data.items()}

    @staticmethod
    def from_mem0_payload(payload: dict) -> BulletMetadata:
        """Extract BulletMetadata from mem0 payload. Missing fields use defaults."""
        metadata = payload.get("metadata", {})
        bullet_fields = {}
        for key, value in metadata.items():
            if key.startswith(MEMX_PREFIX):
                field_name = key[len(MEMX_PREFIX):]
                bullet_fields[field_name] = value
        return BulletMetadata.model_validate(bullet_fields)

    @staticmethod
    def merge_metadata(existing: BulletMetadata, update: dict) -> BulletMetadata:
        """Merge partial updates into existing metadata."""
        data = existing.model_dump()
        data.update(update)
        return BulletMetadata.model_validate(data)
```

### Edge Cases
- payload 为空 dict `{}` → 返回全默认 BulletMetadata
- payload 含非 `memx_` 前缀字段（mem0 自有字段）→ 忽略
- `memx_instructivity_score` 值为字符串 "75" → Pydantic coerce 为 float
- `memx_related_tools` 存储为 JSON 字符串 → 需要判断是 list 还是 string 后解析

---

## Dependencies

**Prerequisite Stories:**
- STORY-001: BulletMetadata 模型
- STORY-006: 项目骨架

**Blocked Stories:**
- STORY-012: BulletDistiller（调用 Factory 创建 Bullet）
- STORY-017: CuratorEngine（调用 Factory 解析 Bullet）
- STORY-014: IngestPipeline（调用 Factory 序列化 Bullet）

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented in `memx/utils/bullet_factory.py`
- [ ] Unit tests in `tests/unit/test_bullet_factory.py`
- [ ] All tests passing
- [ ] `ruff check` 通过
- [ ] `mypy` 通过
- [ ] 向后兼容测试通过（空 payload, 部分字段, 完整字段）
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Factory methods:** 1.5 points
- **Serialization/deserialization:** 0.5 points
- **Unit tests + compat tests:** 1 point
- **Total:** 3 points

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
