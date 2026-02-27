# STORY-015: PrivacySanitizer Hardcoded Safety Net

**Epic:** EPIC-002 — Reflector 知识蒸馏引擎
**Priority:** Must Have
**Story Points:** 2
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a security architect
I want Sanitizer to run even if Reflector is disabled
So that privacy is guaranteed regardless of configuration

---

## Description

### Background
根据 NFR-003（隐私脱敏不可关闭），PrivacySanitizer 必须作为 IngestPipeline 的独立安全网存在。即使 ace_enabled=False，通过 add() 写入的数据也应该有选项经过 Sanitizer 脱敏。本 Story 确保 Sanitizer 在 IngestPipeline 中的位置是独立于 Reflector 的——它在 Reflector 之前运行，即使 Reflector 被禁用或故障，Sanitizer 仍然生效。

### Scope

**In scope:**
- IngestPipeline 中 Sanitizer 独立于 Reflector 执行
- ace_enabled=False 时的可选 sanitizer 开关
- 测试：关闭 Reflector 后 Sanitizer 仍生效
- 测试：Reflector 故障时 Sanitizer 仍生效

**Out of scope:**
- PrivacySanitizer 本身的实现（STORY-011）
- 全局强制 Sanitizer（ace_enabled=False 时默认不强制，但可配置开启）

### Architecture Change

```
IngestPipeline.process()
    │
    ├── Step 0: PrivacySanitizer (ALWAYS runs, before Reflector)
    │     ├── 对 messages content 进行脱敏
    │     └── 失败时 log WARNING，继续用原始内容
    │
    ├── Step 1: Reflector (may be disabled)
    │     ...
    ├── Step 2: Curator
    │     ...
    └── Step 3: Write to mem0
```

配置选项：
```python
class PrivacyConfig(BaseModel):
    # Force sanitization even when ace_enabled=False
    always_sanitize: bool = False  # Default: only sanitize in ACE mode
    custom_patterns: list[str] = []
    sanitize_paths: bool = True
```

---

## Acceptance Criteria

- [ ] IngestPipeline 中 Sanitizer 位于 Reflector 之前执行
- [ ] ace_enabled=True 时 Sanitizer 始终运行
- [ ] ace_enabled=False 且 privacy.always_sanitize=True 时 Sanitizer 仍运行
- [ ] ace_enabled=False 且 privacy.always_sanitize=False 时 Sanitizer 不运行（默认）
- [ ] Reflector 被禁用/故障时 Sanitizer 仍正常运行
- [ ] Sanitizer 自身故障时 log WARNING，不阻塞 add() 操作
- [ ] 测试：ace_enabled=True 关闭 Reflector → Sanitizer 生效
- [ ] 测试：ace_enabled=False + always_sanitize=True → Sanitizer 生效
- [ ] 测试：ace_enabled=False + always_sanitize=False → Sanitizer 不运行

---

## Technical Notes

### Changes to IngestPipeline

```python
class IngestPipeline:
    def process(self, messages, metadata=None, user_id=None, **kwargs):
        result = IngestResult()

        # Step 0: Privacy Sanitization (independent of Reflector)
        sanitized_messages = self._run_sanitizer(messages)

        # Step 1: Reflector (uses sanitized input)
        try:
            event = self._parse_event(sanitized_messages, metadata)
            candidates = self._reflector.reflect(event)
        except Exception:
            ...

    def _run_sanitizer(self, messages) -> str | list:
        """Run privacy sanitizer. Never raises."""
        if not self._sanitizer:
            return messages
        try:
            if isinstance(messages, str):
                result = self._sanitizer.sanitize(messages)
                return result.clean_content
            elif isinstance(messages, list):
                # Sanitize each message content
                ...
        except Exception as e:
            logger.warning(f"Privacy sanitizer failed: {e}")
            return messages  # Use original
```

### Changes to MemXMemory

```python
class Memory:
    def add(self, messages, ...):
        # If always_sanitize=True, run sanitizer even in proxy mode
        if (not self._config.ace_enabled
            and self._config.privacy.always_sanitize
            and self._sanitizer):
            messages = self._sanitizer.sanitize(messages).clean_content

        if not self._config.ace_enabled:
            return self._mem0.add(messages, ...)
        ...
```

### Edge Cases
- messages 是 list of dicts（OpenAI 格式）→ 只脱敏 content 字段
- 脱敏后内容为空 → 仍然 add（记录空内容）
- always_sanitize=True 但 Sanitizer 初始化失败 → log error，不阻塞

---

## Dependencies

**Prerequisite Stories:**
- STORY-011: PrivacySanitizer（必须先实现）
- STORY-014: IngestPipeline（需要在其中集成）

**Blocked Stories:** None

**External Dependencies:** None

---

## Definition of Done

- [ ] IngestPipeline 修改完成
- [ ] MemXMemory 修改完成（always_sanitize 支持）
- [ ] 3 个测试场景全部通过
- [ ] `ruff check` + `mypy` 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Pipeline modification:** 1 point
- **Memory class modification:** 0.5 points
- **Tests:** 0.5 points
- **Total:** 2 points

**Rationale:** 主要是在已有组件中加入独立执行逻辑，代码量不大但需要精确的执行顺序。

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
