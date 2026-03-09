# STORY-051: 扩展 BulletMetadata — schema_version + incompatible_tags

**Epic:** EPIC-009 (Core/Team 解耦重构)
**Priority:** Must Have
**Story Points:** 2
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 5

---

## User Story

As a MemX developer
I want Bullet to have schema_version and incompatible_tags fields
So that Team conflict detection and schema evolution are supported

---

## Description

### Background
为了支持 Team Memory 的 Schema 版本控制和冲突检测（incompatible_tags），需要在 Core 的 `BulletMetadata` 中新增两个字段。这是 Core 代码为 Team Memory 做的唯一小幅修改，其余所有 Team 功能通过扩展层实现。

两个字段都有默认值，旧数据完全向后兼容。

### Scope
**In scope:**
- `BulletMetadata` 新增 `schema_version` 和 `incompatible_tags` 字段
- BulletFactory 序列化/反序列化更新
- mem0 payload 中使用 `memx_` 前缀
- 现有测试通过

**Out of scope:**
- 冲突检测逻辑（STORY-056 实现）
- Schema 迁移工具

### User Flow
1. 新创建的 Bullet 自动带有 `schema_version=1`
2. Team Memory 创建的 TeamBullet 使用 `schema_version=2`
3. `incompatible_tags` 默认为空列表，Team Layer 按需设置

---

## Acceptance Criteria

- [ ] `BulletMetadata` 新增 `schema_version: int = 1`
- [ ] `BulletMetadata` 新增 `incompatible_tags: list[str] = []`
- [ ] 两个字段有默认值，旧数据反序列化完全向后兼容
- [ ] mem0 payload 中使用 `memx_schema_version` 和 `memx_incompatible_tags` 前缀
- [ ] BulletFactory 正确处理新字段的序列化/反序列化
- [ ] 全部现有测试通过，无需修改
- [ ] mypy --strict 通过

---

## Technical Notes

### Components
- `memx/core/types.py`（重构后路径）— BulletMetadata 扩展
- `memx/core/utils/bullet_factory.py` — 序列化/反序列化更新

### Implementation

```python
# In BulletMetadata (memx/core/types.py)
class BulletMetadata(BaseModel):
    # ... existing fields ...
    schema_version: int = 1          # NEW
    incompatible_tags: list[str] = Field(default_factory=list)  # NEW
```

```python
# In BulletFactory — payload serialization
PAYLOAD_MAPPING = {
    # ... existing mappings ...
    "schema_version": "memx_schema_version",
    "incompatible_tags": "memx_incompatible_tags",
}
```

### Edge Cases
- 旧 payload 无 `memx_schema_version` → 反序列化时默认为 1
- 旧 payload 无 `memx_incompatible_tags` → 反序列化时默认为 []
- `incompatible_tags` 中包含不存在的标签 → 正常存储，由上层消费者处理

---

## Dependencies

**Prerequisite Stories:**
- None（独立于包重构，可先行完成）

**Blocked Stories:**
- STORY-050: TeamBullet 数据模型

---

## Definition of Done

- [ ] `BulletMetadata` 新增两个字段
- [ ] BulletFactory 序列化/反序列化更新
- [ ] 全部现有测试通过
- [ ] 新增单元测试覆盖新字段的默认值和序列化
- [ ] mypy --strict 通过
- [ ] ruff check 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **字段添加 + BulletFactory 更新:** 1 point
- **测试验证:** 1 point
- **Total:** 2 points

**Rationale:** 最小改动，仅新增两个有默认值的字段和对应的序列化映射。是 Sprint 5 中最简单的 Story，推荐第一个执行。

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
