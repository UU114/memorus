# STORY-006: 创建 MemX 项目骨架和包结构

**Epic:** EPIC-001 — Bullet 数据模型与配置基础
**Priority:** Must Have
**Story Points:** 5
**Status:** Completed
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a developer
I want a well-organized project structure
So that all team members follow consistent patterns

---

## Description

### Background
MemX 需要一个清晰、标准的 Python 包结构作为所有后续开发的基础。项目采用独立 `memx/` 顶层包，不修改 mem0 内部代码。包结构需要预先创建所有子目录，并配置好工具链（ruff, mypy, pytest），使团队可以立即开始编码。

### Scope

**In scope:**
- `memx/` 顶层包及所有子目录创建
- `pyproject.toml` 完整配置（包名、版本、Python 版本、依赖声明、可选依赖分组）
- `__init__.py` 导出公开 API（Memory, AsyncMemory）
- ruff + mypy 配置文件
- pytest + conftest.py 基础配置
- `.gitignore` 更新

**Out of scope:**
- 具体业务逻辑实现（由后续 Story 完成）
- CI/CD 配置（Sprint 4 STORY-046）
- README 详细文档（Sprint 4）

### Expected File Structure

```
memx/
├── __init__.py              # exports: Memory, AsyncMemory
├── memory.py                # MemXMemory (STORY-004)
├── async_memory.py          # AsyncMemXMemory (STORY-005)
├── config.py                # MemXConfig (STORY-003)
├── types.py                 # BulletMetadata, enums (STORY-001)
├── exceptions.py            # MemX custom exceptions
├── engines/
│   ├── __init__.py
│   ├── reflector/
│   │   ├── __init__.py
│   │   ├── engine.py        # ReflectorEngine (STORY-013)
│   │   ├── detector.py      # PatternDetector (STORY-008)
│   │   ├── patterns.py      # Pattern rules (STORY-009)
│   │   ├── scorer.py        # KnowledgeScorer (STORY-010)
│   │   └── distiller.py     # BulletDistiller (STORY-012)
│   ├── curator/
│   │   ├── __init__.py
│   │   ├── engine.py        # CuratorEngine (STORY-017)
│   │   ├── merger.py        # MergeStrategy (STORY-018)
│   │   └── conflict.py      # ConflictDetector (STORY-047)
│   ├── decay/
│   │   ├── __init__.py
│   │   ├── engine.py        # DecayEngine (STORY-020)
│   │   └── formulas.py      # Decay math (STORY-020)
│   └── generator/
│       ├── __init__.py
│       ├── engine.py         # GeneratorEngine (STORY-028)
│       ├── exact_matcher.py  # L1 (STORY-023)
│       ├── fuzzy_matcher.py  # L2 (STORY-024)
│       ├── metadata_matcher.py # L3 (STORY-025)
│       ├── vector_searcher.py  # L4 (STORY-026)
│       └── score_merger.py   # ScoreMerger (STORY-027)
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py            # IngestPipeline (STORY-014)
│   └── retrieval.py         # RetrievalPipeline (STORY-030)
├── privacy/
│   ├── __init__.py
│   ├── sanitizer.py         # PrivacySanitizer (STORY-011)
│   └── patterns.py          # Regex patterns (STORY-011)
├── integration/
│   ├── __init__.py
│   ├── manager.py           # IntegrationManager (STORY-032)
│   ├── hooks.py             # BaseHook interfaces (STORY-032)
│   └── cli_hooks.py         # CLI hooks (STORY-033, 034)
├── embeddings/
│   ├── __init__.py
│   └── onnx.py              # ONNXEmbedder (STORY-036)
├── daemon/
│   ├── __init__.py
│   ├── server.py            # MemXDaemon (STORY-037)
│   ├── client.py            # DaemonClient (STORY-038)
│   └── ipc.py               # IPC transport (STORY-038)
├── cli/
│   ├── __init__.py
│   └── main.py              # CLI commands (STORY-041, 042)
└── utils/
    ├── __init__.py
    ├── bullet_factory.py    # BulletFactory (STORY-002)
    ├── token_counter.py     # TokenBudgetTrimmer (STORY-029)
    └── text_processing.py   # Text utils (STORY-024)

tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_types.py
│   ├── test_config.py
│   ├── test_reflector.py
│   ├── test_curator.py
│   ├── test_decay.py
│   └── test_generator.py
├── integration/
│   ├── __init__.py
│   ├── test_mem0_compat.py
│   ├── test_ingest_pipeline.py
│   ├── test_retrieval_pipeline.py
│   └── test_daemon.py
└── performance/
    ├── __init__.py
    └── test_benchmarks.py
```

---

## Acceptance Criteria

- [ ] `memx/` 顶层包创建，含所有子目录（engines/, pipeline/, privacy/, integration/, embeddings/, daemon/, cli/, utils/）
- [ ] 每个子目录含 `__init__.py`（空文件即可）
- [ ] `memx/__init__.py` 导出 `Memory`, `AsyncMemory`（占位 import，实际实现在后续 Story）
- [ ] `memx/exceptions.py` 定义基础异常类（MemXError, ConfigurationError, PipelineError, EngineError）
- [ ] `pyproject.toml` 配置完整：
  - 包名 `memx`
  - Python ≥ 3.9
  - 核心依赖：`mem0ai`, `pydantic>=2.0`
  - 可选依赖分组：`[onnx]` (onnxruntime), `[graph]` (neo4j等), `[all]`
  - 入口点 `memx` CLI（Click）
- [ ] ruff 配置：target Python 3.9, line-length 100, select rules
- [ ] mypy 配置：strict mode for `memx/` package
- [ ] pytest 配置：asyncio_mode=auto, test paths
- [ ] `tests/conftest.py` 含基础 fixture（mock mem0 Memory）
- [ ] `tests/` 目录结构创建（unit/, integration/, performance/）
- [ ] 运行 `ruff check memx/` 通过
- [ ] 运行 `mypy memx/` 通过（空模块阶段）
- [ ] 运行 `pytest tests/` 通过（无测试时 exit 0）

---

## Technical Notes

### pyproject.toml Key Sections

```toml
[project]
name = "memx"
version = "0.1.0"
description = "MemX: Adaptive Context Engine on top of mem0"
requires-python = ">=3.9"
dependencies = [
    "mem0ai>=0.1.0",
    "pydantic>=2.0",
    "click>=8.0",
]

[project.optional-dependencies]
onnx = ["onnxruntime>=1.16.0"]
graph = ["neo4j>=5.0"]
all = ["memx[onnx]", "memx[graph]"]

[project.scripts]
memx = "memx.cli.main:cli"

[tool.ruff]
target-version = "py39"
line-length = 100
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "SIM"]

[tool.mypy]
python_version = "3.9"
strict = true
packages = ["memx"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### exceptions.py Structure

```python
class MemXError(Exception):
    """Base exception for all MemX errors."""

class ConfigurationError(MemXError):
    """Invalid configuration."""

class PipelineError(MemXError):
    """Error in ingest/retrieval pipeline."""

class EngineError(MemXError):
    """Error in engine execution."""
```

### __init__.py Placeholder

```python
"""MemX: Adaptive Context Engine on top of mem0."""

__version__ = "0.1.0"

# Placeholder imports - will be implemented in STORY-004, STORY-005
# from memx.memory import Memory
# from memx.async_memory import AsyncMemory

__all__ = ["Memory", "AsyncMemory"]
```

### Edge Cases
- 确保所有 `__init__.py` 文件中不包含会导致 import 错误的代码（初始阶段为空或仅含 docstring）
- pyproject.toml 中 mem0ai 依赖版本需与当前 fork 版本一致

---

## Dependencies

**Prerequisite Stories:** None (this is the first story)

**Blocked Stories:**
- STORY-001: BulletMetadata 模型（需要 `memx/types.py` 存在）
- STORY-002: BulletFactory（需要 `memx/utils/` 存在）
- STORY-003: MemXConfig（需要 `memx/config.py` 存在）
- 所有后续 Story 都依赖项目骨架

**External Dependencies:** None

---

## Definition of Done

- [ ] 所有目录和文件创建完毕
- [ ] pyproject.toml 可通过 `pip install -e .` 安装
- [ ] `python -c "import memx"` 不报错
- [ ] `ruff check memx/` 通过
- [ ] `mypy memx/` 通过
- [ ] `pytest tests/` 通过（exit code 0）
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Directory structure:** 1 point
- **pyproject.toml:** 2 points
- **Tool configs (ruff/mypy/pytest):** 1 point
- **Exceptions + __init__.py:** 1 point
- **Total:** 5 points

**Rationale:** 虽然没有复杂业务逻辑，但需要精确创建大量文件和目录，配置多个工具链，确保一切从一开始就正确。

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master
- 2026-02-27: Started implementation
- 2026-02-27: Completed by Developer

**Actual Effort:** 5 points (matched estimate)

**Implementation Notes:**
- Created 48 files across memx/ and tests/ directories
- pyproject.toml with hatchling build, mem0ai/pydantic/click deps
- ruff check: All checks passed (49 files)
- mypy strict: no issues found (49 files)
- pytest: 2 smoke tests passing
- pip install -e . and import memx verified

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
