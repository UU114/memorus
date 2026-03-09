# STORY-050: 定义 TeamBullet 数据模型

**Epic:** EPIC-009 (Core/Team 解耦重构)
**Priority:** Must Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 5

---

## User Story

As a MemX developer
I want a TeamBullet model extending Bullet
So that team knowledge can carry governance metadata

---

## Description

### Background
Team Memory 中的知识条目需要携带额外的治理元数据（作者、执行级别、投票数、状态等）。`TeamBullet` 继承 Core 的 `Bullet`（或 `BulletMetadata`），扩展 Team 专属字段，同时保持向前/向后兼容。

### Scope
**In scope:**
- `TeamBullet` 模型定义，继承 `Bullet` 并扩展 Team 字段
- Schema version 自动设置（v2）
- v1 → v2 向后兼容（自动填充默认值）
- v2 → v1 向前兼容（保留未知字段）
- 单元测试

**Out of scope:**
- TeamBullet 的存储实现（后续 Story）
- 治理逻辑实现

### User Flow
1. Team Layer 从 Server/Git 加载 JSON 数据
2. 自动反序列化为 TeamBullet 实例
3. v1 格式数据自动升级，v2 字段使用默认值

---

## Acceptance Criteria

- [ ] `TeamBullet` 继承 `Bullet`（或 `BulletMetadata`），新增字段：
  - `author_id: str` — 假名标识
  - `enforcement: str = "suggestion"` — suggestion / recommended / mandatory
  - `upvotes: int = 0`
  - `downvotes: int = 0`
  - `status: str = "approved"` — staging / approved / deprecated / tombstone
  - `deleted_at: datetime | None = None`
  - `origin_id: str | None = None` — 被 Supersede 的原始 Bullet ID
  - `context_summary: str | None = None` — Redactor 生成的脱敏上下文
- [ ] `schema_version = 2` 自动设置
- [ ] v1 → v2 读取自动填充默认值（enforcement="suggestion", upvotes=0, status="approved"）
- [ ] v2 → v1 序列化时保留未知字段（Pydantic model_config extra="allow"）
- [ ] 单元测试覆盖正反向兼容
- [ ] TeamBullet 的 `effective_score` 计算：`instructivity_score + upvotes - downvotes`（bounded 0-100）
- [ ] mypy --strict 通过

---

## Technical Notes

### Components
- `memx/team/types.py` — TeamBullet 模型定义

### Data Structures

```python
# memx/team/types.py
from memx.core.types import BulletMetadata
from pydantic import Field
from datetime import datetime

class TeamBullet(BulletMetadata):
    """Extended Bullet with team governance metadata."""
    model_config = {"extra": "allow"}  # forward compatibility

    # Team-specific fields
    author_id: str = ""
    enforcement: str = "suggestion"  # suggestion | recommended | mandatory
    upvotes: int = 0
    downvotes: int = 0
    status: str = "approved"  # staging | approved | deprecated | tombstone
    deleted_at: datetime | None = None
    origin_id: str | None = None
    context_summary: str | None = None

    def __init__(self, **data):
        data.setdefault("schema_version", 2)
        super().__init__(**data)

    @property
    def effective_score(self) -> float:
        """Score adjusted by community votes."""
        base = self.instructivity_score + self.upvotes - self.downvotes
        return max(0.0, min(100.0, base))

    @property
    def is_active(self) -> bool:
        return self.status in ("staging", "approved")
```

### Compatibility Strategy
- **v1 → v2**：旧数据缺少 Team 字段 → Pydantic 默认值自动填充
- **v2 → v1**：`extra="allow"` 保留未知字段，v1 代码读取时忽略

---

## Dependencies

**Prerequisite Stories:**
- STORY-051: BulletMetadata +schema_version +incompatible_tags

**Blocked Stories:**
- STORY-056: MultiPoolRetriever + Shadow Merge

---

## Definition of Done

- [ ] `TeamBullet` 模型定义完成
- [ ] v1 ↔ v2 兼容性测试通过
- [ ] `effective_score` 计算逻辑正确
- [ ] 单元测试覆盖所有字段的默认值和边界情况
- [ ] mypy --strict 通过
- [ ] ruff check 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **TeamBullet 模型定义:** 1 point
- **兼容性逻辑（v1↔v2）:** 1 point
- **单元测试:** 1 point
- **Total:** 3 points

**Rationale:** 数据模型扩展 + 兼容性处理，逻辑清晰但需仔细处理序列化/反序列化。

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
