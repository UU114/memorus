# STORY-049: 定义 TeamConfig 独立配置模型

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
I want a TeamConfig independent from MemXConfig
So that Team configuration can evolve without affecting Core

---

## Description

### Background
Team Memory 需要独立的配置模型，与 Core 的 `MemXConfig` 完全分离。这确保 Team 配置的演进不会影响 Core，也支持 Team Layer 的可剥离性（NFR-014）。

### Scope
**In scope:**
- `TeamConfig` Pydantic 模型定义
- `AutoNominateConfig`、`RedactorConfig`、`LayerBoostConfig`、`MandatoryOverride` 子模型
- 配置加载（文件 + 环境变量）
- 单元测试

**Out of scope:**
- 配置 UI / 交互式配置工具
- Server 端配置同步

### User Flow
1. 开发者创建 `team_config.yaml` 或设置环境变量
2. Team Layer 启动时自动加载 TeamConfig
3. 未配置时使用合理默认值

---

## Acceptance Criteria

- [ ] `TeamConfig` Pydantic 模型定义完成，包含 `enabled`, `server_url`, `team_id`, `subscribed_tags`, `cache_max_bullets`, `cache_ttl_minutes`
- [ ] `AutoNominateConfig` 子模型：`min_recall_count`, `min_score`, `max_prompts_per_session`, `silent`
- [ ] `RedactorConfig` 子模型：`llm_generalize`, `custom_patterns`
- [ ] `LayerBoostConfig` 子模型：`local_boost`, `team_boost`（默认 1.5 / 1.0）
- [ ] `MandatoryOverride` 子模型：`bullet_id`, `reason`, `expires`（reason 和 expires 必填）
- [ ] TeamConfig 与 MemXConfig 完全独立（不继承、不嵌套）
- [ ] 配置加载支持 YAML 文件和环境变量（`MEMX_TEAM_*` 前缀）
- [ ] 单元测试覆盖校验规则和默认值
- [ ] mypy --strict 通过

---

## Technical Notes

### Components
- `memx/team/config.py` — TeamConfig 及子模型定义

### Data Structures

```python
# memx/team/config.py
from pydantic import BaseModel, Field
from datetime import datetime

class LayerBoostConfig(BaseModel):
    local_boost: float = 1.5
    team_boost: float = 1.0

class AutoNominateConfig(BaseModel):
    min_recall_count: int = 3
    min_score: float = 70.0
    max_prompts_per_session: int = 1
    silent: bool = False

class RedactorConfig(BaseModel):
    llm_generalize: bool = False
    custom_patterns: list[str] = Field(default_factory=list)

class MandatoryOverride(BaseModel):
    bullet_id: str
    reason: str  # required
    expires: datetime  # required

class TeamConfig(BaseModel):
    enabled: bool = False
    server_url: str | None = None
    team_id: str | None = None
    subscribed_tags: list[str] = Field(default_factory=list)
    cache_max_bullets: int = 2000
    cache_ttl_minutes: int = 60
    layer_boost: LayerBoostConfig = Field(default_factory=LayerBoostConfig)
    auto_nominate: AutoNominateConfig = Field(default_factory=AutoNominateConfig)
    redactor: RedactorConfig = Field(default_factory=RedactorConfig)
    mandatory_overrides: list[MandatoryOverride] = Field(default_factory=list)
```

### Configuration Loading
- 文件路径优先级：`./team_config.yaml` > `~/.ace/team_config.yaml`
- 环境变量：`MEMX_TEAM_ENABLED=true`, `MEMX_TEAM_SERVER_URL=...` 等
- 环境变量覆盖文件配置

---

## Dependencies

**Prerequisite Stories:**
- STORY-048: 重构 memx/ → memx/core/

**Blocked Stories:**
- STORY-052: ext/team_bootstrap.py 条件注入

---

## Definition of Done

- [ ] `TeamConfig` 及所有子模型定义完成
- [ ] 配置加载逻辑实现（文件 + 环境变量）
- [ ] 单元测试覆盖：默认值、校验规则、环境变量覆盖
- [ ] mypy --strict 通过
- [ ] ruff check 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Pydantic 模型定义:** 1 point
- **配置加载逻辑:** 1 point
- **单元测试:** 1 point
- **Total:** 3 points

**Rationale:** 纯数据模型 + 配置加载，逻辑清晰，复杂度适中。

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
