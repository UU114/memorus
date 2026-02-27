# STORY-014: 实现 IngestPipeline（add 路径管线）

**Epic:** EPIC-002 — Reflector 知识蒸馏引擎
**Priority:** Must Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As MemXMemory.add()
I want an ingest pipeline to process new memories
So that Reflector and Curator are automatically invoked

---

## Description

### Background
IngestPipeline 编排 add() 操作的完整处理流程：Raw Input → Reflector → Curator → mem0.add。它是连接 MemXMemory 和底层引擎的桥梁。管线设计遵循 NFR-006 的故障隔离原则——Reflector 异常时 fallback 到 raw add，Curator 异常时跳过去重直接 Insert。Sprint 1 中 Curator 尚未实现，IngestPipeline 先对接 Reflector，Curator 步骤预留接口。

### Scope

**In scope:**
- IngestPipeline.process() 方法
- Reflector → mem0.add 流程
- Curator 接口预留（Sprint 2 实现）
- IngestResult 返回值
- Reflector 异常 fallback 到 raw add
- 与 MemXMemory.add() 的集成

**Out of scope:**
- CuratorEngine 实现（STORY-017, Sprint 2）
- RetrievalPipeline（STORY-030, Sprint 2）

### Pipeline Flow

```
IngestPipeline.process(messages, metadata)
    │
    ├── Step 1: Parse messages to InteractionEvent
    │
    ├── Step 2: Reflector.reflect(event)
    │     │ (failure → fallback to raw add)
    │     ▼
    │   CandidateBullet[]
    │
    ├── Step 3: Curator.curate(candidates, existing)  [Sprint 2]
    │     │ (failure → skip dedup, insert all)
    │     ▼
    │   CurateResult{to_add, to_merge}
    │
    ├── Step 4: Write to mem0
    │     ├── For each to_add: mem0.add(content, metadata=bullet_metadata)
    │     └── For each to_merge: mem0.update(id, merged_content)
    │
    └── Return IngestResult
```

---

## Acceptance Criteria

- [ ] `IngestPipeline.process(messages, metadata) -> IngestResult`
- [ ] IngestResult 包含：bullets_added, bullets_merged, bullets_skipped, errors, raw_fallback
- [ ] 正常路径：messages → Reflector → CandidateBullet → mem0.add (带 memx_ metadata)
- [ ] Reflector 异常 → fallback 到 `mem0.add(messages, metadata)`（无 ACE 处理）
- [ ] Curator 未初始化/异常 → 跳过去重，所有 CandidateBullet 直接 Insert
- [ ] Bullet 元数据通过 BulletFactory.to_mem0_metadata() 转为 memx_ 前缀 dict
- [ ] ace_enabled=True 时 MemXMemory.add() 调用此管线
- [ ] IngestPipeline 可独立测试（不依赖 MemXMemory）
- [ ] 集成测试：正常流程 + Reflector 失败 fallback

---

## Technical Notes

### File Location
`memx/pipeline/ingest.py`

### Implementation Sketch

```python
import logging
from dataclasses import dataclass, field
from typing import Optional
from memx.engines.reflector.engine import ReflectorEngine
from memx.utils.bullet_factory import BulletFactory
from memx.types import InteractionEvent

logger = logging.getLogger(__name__)

@dataclass
class IngestResult:
    bullets_added: int = 0
    bullets_merged: int = 0
    bullets_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    raw_fallback: bool = False

class IngestPipeline:
    def __init__(self, reflector: ReflectorEngine,
                 curator=None,  # Optional, Sprint 2
                 mem0_add_fn=None,
                 config=None):
        self._reflector = reflector
        self._curator = curator
        self._mem0_add = mem0_add_fn  # callable: mem0.Memory.add
        self._config = config

    def process(self, messages, metadata: dict = None,
                user_id=None, **kwargs) -> IngestResult:
        result = IngestResult()

        # Step 1: Parse to InteractionEvent
        event = self._parse_event(messages, metadata)

        # Step 2: Reflector
        try:
            candidates = self._reflector.reflect(event)
        except Exception as e:
            logger.warning(f"Reflector failed, falling back to raw add: {e}")
            self._raw_add(messages, metadata, user_id, **kwargs)
            result.raw_fallback = True
            return result

        if not candidates:
            # No learnable patterns detected, do raw add
            self._raw_add(messages, metadata, user_id, **kwargs)
            result.raw_fallback = True
            return result

        # Step 3: Curator (skip if not available)
        if self._curator:
            try:
                curated = self._curator.curate(candidates, ...)
                # Process merges and additions
                ...
            except Exception as e:
                logger.warning(f"Curator failed, inserting all: {e}")
                # Fallthrough to direct insert

        # Step 4: Write candidates to mem0
        for bullet in candidates:
            try:
                bullet_meta = BulletFactory.to_mem0_metadata(bullet.to_metadata())
                merged_meta = {**(metadata or {}), **bullet_meta}
                self._mem0_add(bullet.content, user_id=user_id,
                              metadata=merged_meta, **kwargs)
                result.bullets_added += 1
            except Exception as e:
                result.errors.append(str(e))

        return result

    def _parse_event(self, messages, metadata) -> InteractionEvent:
        """Convert mem0 add() input to InteractionEvent."""
        ...

    def _raw_add(self, messages, metadata, user_id, **kwargs):
        """Fallback: direct mem0 add without ACE processing."""
        if self._mem0_add:
            self._mem0_add(messages, user_id=user_id,
                           metadata=metadata, **kwargs)
```

### Key Design Decisions
1. **mem0_add_fn 注入** — IngestPipeline 不直接持有 mem0.Memory 引用，而是接受 callable，方便测试和解耦
2. **raw_fallback 标记** — 结果中明确标记是否降级，方便调用方了解实际行为
3. **messages 格式** — mem0 的 add() 接受 string 或 list of messages，需要统一解析

### Edge Cases
- messages 为 None 或空 → 跳过处理，返回空 IngestResult
- Reflector 返回空列表（无模式检测到）→ 仍做 raw add
- mem0_add_fn 为 None（测试用途）→ 只运行 Reflector，不写入
- metadata 中已有 memx_ 前缀字段 → 不覆盖用户手动设置的值

---

## Dependencies

**Prerequisite Stories:**
- STORY-004: MemXMemory（调用方）
- STORY-013: ReflectorEngine
- STORY-002: BulletFactory（to_mem0_metadata）

**Blocked Stories:**
- STORY-015: Sanitizer safety net（在 IngestPipeline 中）

**External Dependencies:** None

---

## Definition of Done

- [ ] Code in `memx/pipeline/ingest.py`
- [ ] Unit tests + integration tests
- [ ] 正常路径测试
- [ ] Reflector 失败 fallback 测试
- [ ] IngestResult 字段验证
- [ ] MemXMemory.add(ace_enabled=True) 通过 IngestPipeline
- [ ] `ruff check` + `mypy` 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Pipeline orchestration:** 2 points
- **Event parsing:** 0.5 points
- **Fallback logic:** 0.5 points
- **Tests:** 1 point
- **Total:** 4 points

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
