# STORY-003: 定义 MemXConfig 配置模型

**Epic:** EPIC-001 — Bullet 数据模型与配置基础
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a developer
I want all ACE configuration in a single Pydantic model
So that I can configure MemX with reasonable defaults

---

## Description

### Background
MemXConfig 是 MemX 的统一配置入口，继承/扩展 mem0 的 MemoryConfig。它管理所有 ACE 引擎的配置，同时保持与 mem0 原有配置的完全兼容。核心设计原则是"零配置即可用"——所有参数都有合理默认值，`Memory()` 无参构造即可正常工作。

### Scope

**In scope:**
- MemXConfig 主配置模型
- 子配置模型：ReflectorConfig, CuratorConfig, DecayConfig, RetrievalConfig, PrivacyConfig, IntegrationConfig, DaemonConfig
- 所有参数的合理默认值
- 配置校验（非法值抛 ConfigurationError）
- 从 dict/json 加载配置
- 单元测试

**Out of scope:**
- 配置持久化到文件（未来功能）
- CLI 配置命令（STORY-042）

### Configuration Hierarchy

```
MemXConfig (extends MemoryConfig)
├── ace_enabled: bool = False
├── reflector: ReflectorConfig
│   ├── mode: "rules" | "llm" | "hybrid" = "rules"
│   ├── min_score: float = 30.0
│   ├── max_content_length: int = 500
│   ├── max_code_lines: int = 3
│   └── score_weights: ScoreWeights
├── curator: CuratorConfig
│   ├── similarity_threshold: float = 0.8
│   ├── merge_strategy: "keep_best" | "merge_content" = "keep_best"
│   └── conflict_detection: bool = False
├── decay: DecayConfig
│   ├── half_life_days: float = 30.0
│   ├── boost_factor: float = 0.1
│   ├── protection_days: int = 7
│   ├── permanent_threshold: int = 15
│   ├── archive_threshold: float = 0.02
│   └── sweep_on_session_end: bool = True
├── retrieval: RetrievalConfig
│   ├── keyword_weight: float = 0.6
│   ├── semantic_weight: float = 0.4
│   ├── recency_boost_days: int = 7
│   ├── recency_boost_factor: float = 1.2
│   ├── max_results: int = 5
│   └── token_budget: int = 2000
├── privacy: PrivacyConfig
│   ├── custom_patterns: list[str] = []
│   └── sanitize_paths: bool = True
├── integration: IntegrationConfig
│   ├── auto_recall: bool = True
│   ├── auto_reflect: bool = True
│   └── sweep_on_exit: bool = True
├── daemon: DaemonConfig
│   ├── enabled: bool = False
│   ├── idle_timeout_seconds: int = 300
│   └── socket_path: str | None = None
└── [inherits all mem0 MemoryConfig fields]
```

---

## Acceptance Criteria

- [ ] MemXConfig 继承或组合 mem0 的 MemoryConfig
- [ ] 包含 `ace_enabled` 开关（默认 False）
- [ ] 7 个子配置模型全部定义完成（Reflector, Curator, Decay, Retrieval, Privacy, Integration, Daemon）
- [ ] 所有字段有完整默认值——`MemXConfig()` 无参构造成功
- [ ] 子配置校验：
  - `similarity_threshold` 范围 0-1
  - `half_life_days` > 0
  - `keyword_weight + semantic_weight` 允许不等于 1.0 但给出 warning
  - `min_score` 范围 0-100
  - `token_budget` > 0
- [ ] 非法值抛 `ConfigurationError`
- [ ] `from_dict()` 类方法支持从 dict 创建配置（兼容 mem0 config dict 格式）
- [ ] mem0 原有配置字段不受影响（version, custom_prompt 等仍可设置）
- [ ] 单元测试覆盖：默认值、校验、非法值、from_dict、mem0 兼容

---

## Technical Notes

### File Location
`memx/config.py`

### Implementation Strategy

```python
from pydantic import BaseModel, Field, model_validator
from memx.exceptions import ConfigurationError

class ReflectorConfig(BaseModel):
    mode: str = "rules"  # "rules" | "llm" | "hybrid"
    min_score: float = Field(default=30.0, ge=0.0, le=100.0)
    max_content_length: int = Field(default=500, gt=0)
    max_code_lines: int = Field(default=3, gt=0)

class DecayConfig(BaseModel):
    half_life_days: float = Field(default=30.0, gt=0)
    boost_factor: float = Field(default=0.1, ge=0)
    protection_days: int = Field(default=7, ge=0)
    permanent_threshold: int = Field(default=15, ge=1)
    archive_threshold: float = Field(default=0.02, ge=0, le=1)
    sweep_on_session_end: bool = True

# ... other sub-configs

class MemXConfig(BaseModel):
    """MemX unified configuration.

    Wraps mem0 config fields and adds ACE-specific configuration.
    """
    ace_enabled: bool = False
    reflector: ReflectorConfig = Field(default_factory=ReflectorConfig)
    curator: CuratorConfig = Field(default_factory=CuratorConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)

    # mem0 config passthrough
    mem0_config: dict = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MemXConfig":
        """Create config from dict, separating MemX and mem0 fields."""
        ...
```

### mem0 Config Compatibility
- mem0 的 `MemoryConfig` 使用 dataclass 而非 Pydantic
- 策略：MemXConfig 使用 Pydantic，但提供 `to_mem0_config()` 方法输出 mem0 格式
- `from_dict()` 智能分离 `ace_*` 字段和 mem0 字段

### Edge Cases
- `ace_enabled=False` 时，子配置仍可设置（不影响），但不会被使用
- mem0 config dict 中包含 MemX 不认识的字段 → 透传给 mem0，不报错
- `keyword_weight=0.8, semantic_weight=0.8` → 允许（归一化在 ScoreMerger 中处理），但 log warning

---

## Dependencies

**Prerequisite Stories:**
- STORY-001: BulletMetadata（了解字段定义设计配置）
- STORY-006: 项目骨架

**Blocked Stories:**
- STORY-004: MemXMemory（需要 MemXConfig）
- STORY-011: PrivacySanitizer（需要 PrivacyConfig）
- STORY-036: ONNXEmbedder（需要 DaemonConfig/EmbeddingConfig）

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented in `memx/config.py`
- [ ] All 7 sub-config models defined
- [ ] Unit tests in `tests/unit/test_config.py`
- [ ] All tests passing (≥85% coverage)
- [ ] `ruff check` 通过
- [ ] `mypy` 通过
- [ ] Acceptance criteria all validated
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Sub-config models (7):** 2 points
- **MemXConfig + from_dict:** 1.5 points
- **Unit tests:** 1.5 points
- **Total:** 5 points

**Rationale:** 7 个子配置 + 主配置，字段数量多且需要精确校验，from_dict 的 mem0 兼容逻辑有一定复杂度。

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
