# STORY-033: 实现 CLI PreInferenceHook

**Epic:** EPIC-006 (Integration Layer)
**Priority:** Should Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 3

---

## User Story

As a CLI AI tool user
I want relevant memories automatically recalled before each query
So that the AI remembers past interactions and provides contextual responses

---

## Description

### Background
PreInferenceHook 是 Memorus 最核心的集成点——在用户输入发送给 LLM 之前，自动从 Playbook 中检索相关记忆并注入为上下文。首要集成目标是 Claude Code 的 `UserPromptSubmit` Hook。本 Story 实现 CLI 场景下的具体 Hook。

### Scope
**In scope:**
- `CLIPreInferenceHook` 具体实现
- 读取用户输入 → 调用 RetrievalPipeline.search()
- 三种格式化模板（XML / Markdown / Plain）
- 空结果时返回 None（不注入）
- 可配置 token 预算和最大结果数

**Out of scope:**
- MCP Server 集成
- IDE Plugin 集成
- 自定义模板

---

## Acceptance Criteria

- [ ] `CLIPreInferenceHook` 继承 `PreInferenceHook`
- [ ] `on_user_input(input)` 调用 `memory.search(input)` 获取记忆
- [ ] 搜索结果为空时返回 `ContextInjection(memories=[], rendered="")`
- [ ] XML 模板格式输出：`<memorus-context><memory id="..." score="...">content</memory></memorus-context>`
- [ ] Markdown 模板格式输出：`## Memorus Context\n- **[score]** content`
- [ ] Plain 模板格式输出：`[Memorus] content1\n[Memorus] content2`
- [ ] 格式由 `IntegrationConfig.context_template` 配置（默认 "xml"）
- [ ] Hook 内异常被捕获，返回 None 并记录 WARNING

---

## Technical Notes

### Components
- `memorus/integration/cli_hooks.py` — CLIPreInferenceHook

### API Design

```python
class CLIPreInferenceHook(PreInferenceHook):
    def __init__(self, memory: Memory, config: IntegrationConfig):
        self._memory = memory
        self._config = config

    @property
    def name(self) -> str:
        return "cli_pre_inference"

    @property
    def enabled(self) -> bool:
        return self._config.auto_recall

    async def on_user_input(self, input: str) -> ContextInjection:
        results = self._memory.search(input)
        if not results:
            return ContextInjection(memories=[], format="", rendered="")
        rendered = self._format(results, self._config.context_template)
        return ContextInjection(
            memories=results,
            format=self._config.context_template,
            rendered=rendered,
        )

    def _format(self, results: list[dict], template: str) -> str: ...
```

### Template Examples

**XML (default):**
```xml
<memorus-context>
  <memory id="abc123" score="0.85" type="preference">
    User prefers dark mode in all applications.
  </memory>
  <memory id="def456" score="0.72" type="tool_pattern">
    When using pytest, always run with -v flag.
  </memory>
</memorus-context>
```

**Markdown:**
```markdown
## Memorus Context
- **[0.85]** User prefers dark mode in all applications.
- **[0.72]** When using pytest, always run with -v flag.
```

### Dependencies on Existing Code
- `memorus/integration/hooks.py:PreInferenceHook` — 抽象基类（STORY-032）
- `memorus/memory.py:Memory.search()` — 检索接口
- `memorus/config.py:IntegrationConfig` — context_template 字段

### Edge Cases
- 用户输入为空字符串 → 跳过搜索，返回空 ContextInjection
- 用户输入仅含标点 → 正常搜索（可能无结果）
- Memory.search() 抛异常 → 捕获，返回 None
- 搜索结果超长 → 由 TokenBudgetTrimmer 在 RetrievalPipeline 内处理

---

## Dependencies

**Prerequisite Stories:**
- STORY-032: IntegrationManager + BaseHook 抽象
- STORY-030: RetrievalPipeline ✓（已完成）

**Blocked Stories:**
- STORY-035: Integration 测试

---

## Definition of Done

- [ ] `memorus/integration/cli_hooks.py` 实现 CLIPreInferenceHook
- [ ] 三种模板格式单元测试
- [ ] 空结果、异常路径测试
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **Hook 逻辑:** 1.5 points
- **三种模板格式化:** 1.5 points
- **测试:** 1 point
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
