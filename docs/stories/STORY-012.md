# STORY-012: 实现 BulletDistiller（Stage 4）

**Epic:** EPIC-002 — Reflector 知识蒸馏引擎
**Priority:** Must Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a Reflector engine
I want scored candidates distilled into standard Bullets
So that knowledge is compact and structured

---

## Description

### Background
BulletDistiller 是 Reflector 的最终阶段（Stage 4）。它将 KnowledgeScorer 输出的 ScoredCandidate 转化为标准的 CandidateBullet 格式——包含截断后的内容、提取的元数据（related_tools, key_entities）、以及从 Scorer 继承的分类和评分。CandidateBullet 是交给 Curator 进行去重/合并判断的输入格式。

### Scope

**In scope:**
- content 截断（≤ 500 字符）
- code_content 截断（≤ 3 行）
- 自动提取 related_tools（从内容中识别工具/命令名称）
- 自动提取 key_entities（从内容中识别关键术语）
- 输出 CandidateBullet 格式

**Out of scope:**
- LLM 辅助蒸馏（Sprint 4）
- 语义压缩/改写（未来功能）

---

## Acceptance Criteria

- [ ] `BulletDistiller.distill(candidate: ScoredCandidate) -> CandidateBullet`
- [ ] content 截断到 ≤ max_content_length（默认 500 字符），以句子边界截断
- [ ] 代码内容（如有）截断到 ≤ max_code_lines（默认 3 行）
- [ ] 自动从内容中提取 related_tools（基于已知工具名列表 + 命令模式匹配）
- [ ] 自动从内容中提取 key_entities（基于大写词、引号包围的词、代码标识符）
- [ ] CandidateBullet 包含：content, knowledge_type, section, instructivity_score, related_tools, key_entities, source_type, context
- [ ] max_content_length 和 max_code_lines 通过 ReflectorConfig 可配置
- [ ] 单元测试：长度截断、实体提取、工具提取

---

## Technical Notes

### File Location
`memx/engines/reflector/distiller.py`

### Implementation Sketch

```python
import re
from memx.types import CandidateBullet, SourceType
from memx.config import ReflectorConfig

class BulletDistiller:
    # Common tool/command names to detect
    KNOWN_TOOLS = {
        "git", "docker", "npm", "pip", "cargo", "brew", "apt",
        "kubectl", "terraform", "ansible", "make", "cmake",
        "pytest", "ruff", "mypy", "black", "flake8",
        "curl", "wget", "ssh", "scp", "rsync",
        "python", "node", "java", "go", "rustc",
    }

    def __init__(self, config: ReflectorConfig):
        self._max_content = config.max_content_length
        self._max_code_lines = config.max_code_lines

    def distill(self, candidate) -> CandidateBullet:
        content = self._truncate_content(candidate.raw_content)
        tools = self._extract_tools(candidate.raw_content, candidate.context)
        entities = self._extract_entities(candidate.raw_content)

        return CandidateBullet(
            content=content,
            knowledge_type=candidate.knowledge_type,
            section=candidate.section,
            instructivity_score=candidate.instructivity_score,
            related_tools=tools,
            key_entities=entities,
            source_type=SourceType.INTERACTION,
            context=candidate.context,
        )

    def _truncate_content(self, content: str) -> str:
        """Truncate at sentence boundary, max length."""
        if len(content) <= self._max_content:
            return content
        # Find last sentence boundary before limit
        truncated = content[:self._max_content]
        for sep in [". ", "。", "\n", "; "]:
            idx = truncated.rfind(sep)
            if idx > self._max_content * 0.5:
                return truncated[:idx + len(sep)].strip()
        return truncated.strip() + "..."

    def _extract_tools(self, content: str, context: dict) -> list[str]:
        """Extract tool names from content and context."""
        tools = set()
        # From context (tool_name field)
        if tool := context.get("tool"):
            tools.add(tool.lower())
        # From content (known tools)
        words = set(re.findall(r'\b\w+\b', content.lower()))
        tools.update(words & self.KNOWN_TOOLS)
        return sorted(tools)

    def _extract_entities(self, content: str) -> list[str]:
        """Extract key entities: capitalized words, quoted terms, identifiers."""
        entities = set()
        # Quoted strings
        entities.update(re.findall(r'["`]([^"`]+)["`]', content))
        # CamelCase / PascalCase identifiers
        entities.update(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', content))
        # File paths / dotted names
        entities.update(re.findall(r'\b[\w]+\.[\w.]+\b', content))
        # Limit to top 10
        return sorted(entities)[:10]
```

### Edge Cases
- 极短内容（< 10 chars）→ 不截断，直接通过
- 内容全是代码 → 保留前 3 行作为 code_content
- 没有可识别的工具或实体 → 返回空列表（不报错）
- Unicode 内容（中文）→ 截断按字符数而非字节数
- context 字段为空 dict → 跳过 context 提取

---

## Dependencies

**Prerequisite Stories:**
- STORY-001: BulletMetadata, CandidateBullet 类型
- STORY-002: BulletFactory（可选，Distiller 直接构造 CandidateBullet）
- STORY-010: KnowledgeScorer（输出 ScoredCandidate）

**Blocked Stories:**
- STORY-013: ReflectorEngine（调用 BulletDistiller）

**External Dependencies:** None

---

## Definition of Done

- [ ] Code in `memx/engines/reflector/distiller.py`
- [ ] Unit tests in `tests/unit/test_reflector.py` (distiller section)
- [ ] 长度截断测试（超长、刚好、极短）
- [ ] 工具提取测试
- [ ] 实体提取测试
- [ ] `ruff check` + `mypy` 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Truncation logic:** 1 point
- **Tool extraction:** 0.5 points
- **Entity extraction:** 0.5 points
- **Tests:** 1 point
- **Total:** 3 points

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
