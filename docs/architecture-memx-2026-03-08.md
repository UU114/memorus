# System Architecture: MemX

**Date:** 2026-03-08
**Architect:** TPY
**Version:** 2.0
**Project Type:** AI 记忆引擎（mem0 Fork + ACE 智能层 + Team Memory）
**Project Level:** Level 3（大型项目）
**Status:** Draft

---

## Document Overview

This document defines the system architecture for MemX. It provides the technical blueprint for implementation, addressing all 37 functional and 13 non-functional requirements from the PRD v2.0.

**Related Documents:**
- Product Requirements Document: `docs/prd-memx-2026-03-08.md`
- Product Brief: `docs/product-brief-memx-2026-02-27.md`
- ACE Analysis Report: `doc/ace-mem0-analysis-report.md`
- ACE Team Memory Architecture: `ace-team-memory-architecture.md` **(v2.0 新增)**

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-27 | TPY | Initial architecture — 12 components, 6 layers |
| 2.0 | 2026-03-08 | TPY | Team Memory expansion — Core/Team decoupling, Team Layer (7 new components), updated diagram, new NFR coverage (NFR-011~014), updated code organization and traceability |

---

## Executive Summary

MemX 在 mem0 之上叠加 ACE 智能层，形成"Layered Engine on Fork"架构。核心设计原则：新增模块完全独立于 mem0 现有代码，通过 Pipeline + Factory 模式插入；所有 ACE 组件有独立故障边界，任何故障不影响宿主。

**v2.0 新增**：Team Memory 扩展层，采用**充分解耦架构**。Team Layer 作为 Core 的纯可选扩展层，通过 `StorageBackend` Protocol 和组合模式注入，Core 代码零修改。依赖方向严格单向：`Core ← Team Layer ← Sync Server`。

架构分为 **7 层**：
1. Integration Layer（集成层）
2. Pipeline Layer（管线层）
3. Engine Layer（引擎层）
4. **Team Layer（团队扩展层）— v2.0 新增**
5. Storage Adapter Layer（存储适配层）
6. mem0 Infrastructure Layer（基础设施层）
7. Configuration Layer（配置层）

---

## Architectural Drivers

These requirements heavily influence architectural decisions:

### Core Drivers (v1.0)

1. **NFR-001: 检索延迟 < 50ms** → 混合检索管线高度优化，内存缓存，异步 I/O
2. **NFR-003: 隐私脱敏不可关闭** → Reflector 管线中 Sanitizer 是硬编码安全网
3. **NFR-005: mem0 API 完全兼容** → 所有改造通过"可选增强"模式叠加
4. **NFR-006: 优雅降级** → 每个 ACE 组件有独立 try-catch 边界
5. **NFR-010: 零配置启动** → 所有配置有合理默认值

### Team Memory Drivers (v2.0 新增)

6. **NFR-011: Team 检索延迟 < 100ms** → Team Cache 本地化，Pre-Inference 零远程请求，Shadow Merge < 5ms
7. **NFR-012: Team 数据隔离** → Local/Team 存储路径完全分离，Team 对 Local 只读不写
8. **NFR-013: Team 隐私保护** → 提名必须经过 Redactor + 用户审核，假名标识
9. **NFR-014: Team 可剥离性** → Core/Team 充分解耦，删除 Team 后 Core 行为不变

---

## System Overview

### High-Level Architecture

MemX 采用 **Layered Engine Architecture + Decoupled Team Extension**（分层引擎架构 + 解耦团队扩展）。

核心设计决策：
- **新增代码独立目录**：`memx/core/` 包含所有现有引擎代码，`memx/team/` 作为可选扩展
- **Pipeline 模式**：add/search 操作通过可组合的处理管线，每个 Stage 可独立启停
- **Factory + Strategy 模式**：引擎组件通过 Factory 创建，可按配置切换实现
- **Decorator 模式**：MemX 的 `Memory` 类包装 mem0 的 `Memory` 类，ACE 关闭时直接代理
- **组合模式（v2.0）**：MultiPoolRetriever 组合多个 StorageBackend，Generator 代码零改动

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    HOST AI PRODUCT                          │
│                                                             │
│  User Input ──→ [IntegrationManager]                        │
│                  ├── PreInferenceHook  (auto recall)        │
│                  ├── PostActionHook    (auto reflect)       │
│                  └── SessionEndHook   (sweep + flush)       │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    MEMX PUBLIC API                          │
│                                                             │
│  Memory(ace_enabled=True)                                   │
│  ├── .add(messages, user_id, **kwargs)                      │
│  ├── .search(query, user_id, **kwargs)                      │
│  ├── .get_all() / .get() / .update() / .delete()           │
│  └── .status() / .export() / .import()   [new]             │
│                                                             │
│  ACE OFF → direct proxy to mem0.Memory                      │
│  ACE ON  → pipeline processing below                        │
│                                                             │
└──────────────┬──────────────────┬───────────────────────────┘
               │ add() path       │ search() path
┌──────────────▼──────────┐  ┌────▼──────────────────────────┐
│    INGEST PIPELINE      │  │     RETRIEVAL PIPELINE        │
│                         │  │                               │
│  Raw Input              │  │  Query                        │
│    │                    │  │    │                          │
│    ▼                    │  │    ▼                          │
│  [Reflector]            │  │  [MultiPoolRetriever] ◄─ v2.0 │
│  ├─ Stage1: Detector    │  │    ├── Local Pool             │
│  ├─ Stage2: Scorer      │  │    │   ├── L1: ExactMatcher   │
│  ├─ Stage3: Sanitizer   │  │    │   ├── L2: FuzzyMatcher   │
│  └─ Stage4: Distiller   │  │    │   ├── L3: MetadataMatcher│
│    │                    │  │    │   └── L4: VectorSearcher  │
│    ▼                    │  │    │                          │
│  [Curator]              │  │    └── Team Pool (optional)   │
│  ├─ SimilarityCheck     │  │        ├── TeamCacheStorage   │
│  └─ MergeOrInsert       │  │        └── GitFallbackStorage │
│    │                    │  │    │                          │
│    ▼                    │  │    ▼                          │
│  mem0.add()             │  │  [ShadowMerger] ◄──── v2.0    │
│  (with Bullet metadata) │  │  (Local×1.5 + Team×1.0)      │
│                         │  │    │                          │
│                         │  │    ▼                          │
│                         │  │  DecayWeighter                │
│                         │  │    │                          │
│                         │  │    ▼                          │
│                         │  │  TokenBudgetTrimmer            │
│                         │  │    │                          │
│                         │  │    ▼                          │
│                         │  │  [Reranker] (optional)        │
│                         │  │    │                          │
│                         │  │    ▼                          │
│                         │  │  RecallReinforcer              │
│                         │  │  (async: update recall_count)  │
│                         │  │    │                          │
│                         │  │    ▼                          │
│                         │  │  Results                       │
└─────────────────────────┘  └───────────────────────────────┘
               │                          │
┌──────────────▼──────────────────────────▼───────────────────┐
│                    ENGINE LAYER (Core)                       │
│                                                             │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
│  │ Reflector  │ │ Curator  │ │ Decay    │ │ Generator   │  │
│  │ Engine     │ │ Engine   │ │ Engine   │ │ Engine      │  │
│  │            │ │          │ │          │ │             │  │
│  │ rules/     │ │ cosine   │ │ ebbhaus  │ │ L1-L4       │  │
│  │ llm/       │ │ merge    │ │ sweep    │ │ scoring     │  │
│  │ hybrid     │ │ conflict │ │ archive  │ │ degrade     │  │
│  └────────────┘ └──────────┘ └──────────┘ └─────────────┘  │
│                                                             │
│  ┌──────────────────┐  ┌─────────────────────────────────┐  │
│  │ PrivacySanitizer │  │ BulletFactory                   │  │
│  │ (hardcoded net)  │  │ (create/validate Bullet)        │  │
│  └──────────────────┘  └─────────────────────────────────┘  │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              TEAM LAYER (Optional Extension) ◄── v2.0       │
│                                                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐  │
│  │ TeamCacheStorage│ │ GitFallbackStore│ │ SyncClient   │  │
│  │ (Federation)    │ │ (JSONL reader)  │ │ (pull/push)  │  │
│  └─────────────────┘ └─────────────────┘ └──────────────┘  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐  │
│  │ TeamMerger      │ │ Redactor        │ │ Nominator    │  │
│  │ (Shadow Merge)  │ │ (L1+L2+L3)     │ │ (Promotion)  │  │
│  └─────────────────┘ └─────────────────┘ └──────────────┘  │
│  ┌─────────────────┐                                        │
│  │ TeamConfig      │  ← Independent from Core Config        │
│  └─────────────────┘                                        │
│                                                             │
│  Dependency: Core ←── Team Layer (never reverse)            │
│  Data: Team Layer reads Local Pool (never writes)           │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              MEM0 INFRASTRUCTURE LAYER (保留)                │
│                                                             │
│  ┌─────────────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐  │
│  │ VectorStore │ │ LLM      │ │ Embedder   │ │ Graph    │  │
│  │ (23 provs)  │ │ (13+)    │ │ (13+ONNX)  │ │ Store    │  │
│  └─────────────┘ └──────────┘ └────────────┘ └──────────┘  │
│  ┌─────────────┐ ┌──────────────────────────────────────┐   │
│  │ Reranker    │ │ SQLiteManager (history + audit)      │   │
│  │ (5 provs)   │ │                                      │   │
│  └─────────────┘ └──────────────────────────────────────┘   │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                 CONFIGURATION LAYER                          │
│                                                             │
│  MemXConfig(MemoryConfig)                                   │
│  ├── ace_enabled: bool = False                              │
│  ├── retrieval: RetrievalConfig                             │
│  ├── reflector: ReflectorConfig                             │
│  ├── curator: CuratorConfig                                 │
│  ├── decay: DecayConfig                                     │
│  ├── privacy: PrivacyConfig                                 │
│  ├── integration: IntegrationConfig                         │
│  ├── daemon: DaemonConfig                                   │
│  └── [继承 mem0 全部配置字段]                                  │
│                                                             │
│  TeamConfig (独立于 MemXConfig) ◄── v2.0                     │
│  ├── enabled: bool = False                                  │
│  ├── server_url: str = ""                                   │
│  ├── subscribed_tags: list[str] = []                        │
│  ├── cache_max_bullets: int = 2000                          │
│  ├── auto_nominate: AutoNominateConfig                      │
│  ├── redactor: RedactorConfig                               │
│  └── layer_boost: LayerBoostConfig                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Architectural Pattern

**Pattern:** Layered Engine Architecture + Decorator + Pipeline + Factory + **Composition (v2.0)**

**Rationale (v1.0):**

| 可选模式 | 优点 | 缺点 | 结论 |
|----------|------|------|------|
| 直接修改 mem0 源码 | 实现最快 | 上游同步噩梦，破坏 API 兼容 | 排除 |
| 微服务拆分 | 独立部署 | 过度工程化，单用户场景无意义 | 排除 |
| Middleware/Wrapper | 不侵入 mem0 | 无法深入改造检索管线 | 部分采用 |
| **Layered Engine + Decorator** | 独立模块 + 透明代理 + 深度集成 | 需要精心设计层间接口 | **采用** |

**v2.0 新增 — 组合模式用于 Team 扩展：**

| 可选模式 | 优点 | 缺点 | 结论 |
|----------|------|------|------|
| 修改 Generator 内部逻辑 | 实现直接 | 侵入 Core，违反解耦约束 | 排除 |
| 继承 Generator 类 | 面向对象 | 耦合紧，覆写风险 | 排除 |
| **组合模式 (MultiPoolRetriever)** | Core 零改动，Team 通过组合注入 | 初始化层需要胶水代码 | **采用** |

关键选择理由：
1. **Decorator 模式**用于 Memory 类包装 → 保证 mem0 API 兼容（NFR-005）
2. **Pipeline 模式**用于 add/search 管线 → 每个 Stage 可独立 try-catch（NFR-006）
3. **Factory 模式**用于引擎创建 → 按配置切换实现（NFR-010 零配置）
4. **独立目录**用于代码组织 → 最小化上游冲突（NFR-008）
5. **组合模式（v2.0）**用于 Team 扩展 → Core Generator 零改动（NFR-014 可剥离性）

---

## Technology Stack

### Backend

**Choice:** Python 3.9+ (与 mem0 一致)

| 组件 | 技术 | 理由 |
|------|------|------|
| 核心语言 | Python 3.9+ | mem0 要求，生态兼容 |
| 类型系统 | Pydantic v2 | 配置校验、数据模型，mem0 已依赖 |
| 异步框架 | asyncio + ThreadPoolExecutor | mem0 已有模式，AsyncMemory 支持 |
| CLI 框架 | Click | 轻量、标准、无额外依赖 |
| Daemon IPC | Named Pipe (Win) / Unix Socket (Linux/Mac) | 跨平台、低延迟 |
| ONNX Runtime | onnxruntime | 本地 Embedding 推理 |
| **HTTP Client (v2.0)** | **httpx** | **Team Sync Client，支持 async** |

### Database

**Choice:** SQLite (Core) + Team Cache 独立存储 (v2.0)

| 层 | 技术 | 用途 |
|----|------|------|
| 向量存储 | mem0 VectorStore (23 providers) | 记忆向量索引和检索 |
| 历史审计 | SQLiteManager (mem0 内置) | 变更历史追踪 |
| Bullet 元数据 | 嵌入向量存储 payload metadata | ACE 结构化字段 |
| Daemon 状态 | SQLite (独立数据库) | Session 注册、健康状态 |
| **Team Cache (v2.0)** | **独立文件存储** (`~/.ace/team_cache/{team_id}/`) | **Team Bullet 缓存 + 向量** |
| **Git Fallback (v2.0)** | **JSONL 文件** (`.ace/playbook.jsonl`) | **只读团队知识** |

### Third-Party Services

| 服务 | 用途 | 必需 | 说明 |
|------|------|------|------|
| PyPI | 包发布 | 是 | 发布渠道 |
| GitHub | 代码托管 | 是 | 开源社区 |
| LLM API (OpenAI等) | LLM 增强蒸馏 | 否 | 仅 LLM 模式需要 |
| Embedding API | 在线 Embedding | 否 | 本地 ONNX 可替代 |
| **ACE Sync Server (v2.0)** | **Federation Mode** | **否** | **仅 Federation 模式需要** |

### Development & Deployment

| 工具 | 选择 | 理由 |
|------|------|------|
| 包管理 | Poetry | mem0 已使用 |
| 测试框架 | pytest + pytest-asyncio | mem0 已使用 |
| 代码质量 | ruff (linter + formatter) | 快速、全面 |
| 类型检查 | mypy (strict) | ACE 新增模块强制类型 |
| CI/CD | GitHub Actions | 开源标准 |
| 文档 | MkDocs + mkdocstrings | Python 生态标准 |
| 版本管理 | SemVer + CalVer 混合 | 主版本跟随 mem0，patch 自增 |

---

## System Components

### Core Components (v1.0, unchanged)

以下 12 个组件的接口、职责、实现方案与 v1.0 完全一致，此处仅列出摘要。详细接口定义参见 v1.0 架构文档。

#### Component 1: MemXMemory (公开 API 层)

**Purpose:** MemX 的公开入口，包装 mem0 的 Memory 类

**FRs Addressed:** FR-013 (API 兼容), FR-012 (配置系统)

**v2.0 变更:** 无接口变更。Team 功能通过 `ext/team_bootstrap.py` 在初始化时注入，MemXMemory 本身不感知 Team Layer。

---

#### Component 2: IngestPipeline (入库管线)

**Purpose:** 处理 add() 操作：Raw Input → Reflect → Curate → Store

**FRs Addressed:** FR-002~FR-006

**v2.0 变更:** 无变更。Team 提名（Promotion）由独立的 Nominator 组件在 Ingest 完成后异步触发，不修改 IngestPipeline。

---

#### Component 3: RetrievalPipeline (检索管线)

**Purpose:** 处理 search() 操作：Query → Multi-layer Match → Score → Trim → Return

**FRs Addressed:** FR-009~FR-011

**v2.0 变更:** RetrievalPipeline 的 `storage_backend` 参数由初始化层注入。当 Team 功能启用时，`team_bootstrap.py` 注入 `MultiPoolRetriever`（组合 Local + Team Pool）；当 Team 功能禁用时，注入原始 Local StorageBackend。**RetrievalPipeline 代码本身零改动。**

```python
# ext/team_bootstrap.py (v2.0 新增)
def bootstrap_team(memory: MemXMemory, team_config: TeamConfig):
    """Inject Team Layer into Memory if team is configured."""
    if not team_config.enabled:
        return  # No-op, Core runs as-is

    # Determine Team storage backend
    if team_config.server_url:
        team_storage = TeamCacheStorage(team_config)
    elif _detect_git_fallback():
        team_storage = GitFallbackStorage()
    else:
        return  # No team source available

    # Inject MultiPoolRetriever into RetrievalPipeline
    local_storage = memory._retrieval.storage_backend
    multi_pool = MultiPoolRetriever(local=local_storage, team=team_storage)
    memory._retrieval.storage_backend = multi_pool
```

---

#### Component 4: ReflectorEngine (知识蒸馏引擎)

**FRs Addressed:** FR-002, FR-003, FR-004, FR-005, FR-019

**v2.0 变更:** 无变更。Reflector 生成 `incompatible_tags` 时对齐 Team Taxonomy（如果可用），但这是通过 Reflector 的可扩展 tag 生成接口实现，不修改 Reflector 核心代码。

---

#### Component 5: CuratorEngine (语义去重引擎)

**FRs Addressed:** FR-006, FR-020

**v2.0 变更:** 无变更。

---

#### Component 6: DecayEngine (衰退引擎)

**FRs Addressed:** FR-007, FR-008

**v2.0 变更:** 无变更。Team Cache 有独立的 TTL 机制（由 TeamCacheStorage 管理），不使用 DecayEngine。

---

#### Component 7: GeneratorEngine (混合检索引擎)

**FRs Addressed:** FR-009, FR-010, FR-011

**v2.0 变更:** 无代码变更。Generator 通过 `StorageBackend` Protocol 接口检索，MultiPoolRetriever 在外部组合后注入。Generator 对 Team Pool 完全无感知。

---

#### Component 8: IntegrationManager (集成管理器)

**FRs Addressed:** FR-014, FR-015, FR-016

**v2.0 变更:** 无变更。

---

#### Component 9: ONNXEmbedder (本地 Embedding)

**FRs Addressed:** FR-017

**v2.0 变更:** 无变更。Git Fallback 的向量缓存也使用 ONNXEmbedder 生成。

---

#### Component 10: MemXDaemon (常驻进程)

**FRs Addressed:** FR-018

**v2.0 变更:** 新增 `team_sync` IPC 命令（可选），触发 Team Cache 增量同步。

---

#### Component 11: BulletFactory + BulletMetadata (数据模型)

**FRs Addressed:** FR-001

**v2.0 变更:** `BulletMetadata` 新增两个字段：`schema_version: int = 1` 和 `incompatible_tags: list[str] = []`。两个字段均有默认值，向后完全兼容。

---

#### Component 12: MemXConfig (配置系统)

**FRs Addressed:** FR-012

**v2.0 变更:** 无变更。Team 配置（`TeamConfig`）完全独立于 `MemXConfig`，不在 MemXConfig 中添加任何字段。

---

### Team Layer Components (v2.0 新增)

以下 7 个组件全部位于 `memx/team/` 包中，Core 不 import 这些模块。

---

#### Component 13: MultiPoolRetriever (多路检索组合器)

**Purpose:** 组合 Local Pool 和 Team Pool 的检索结果，实现 Shadow Merge

**Responsibilities:**
- 持有多个 `StorageBackend` 实例（Local + Team）
- 并行查询各 Pool
- 执行 Shadow Merge：Local 加权 1.5、Team 加权 1.0
- 处理 `enforcement: "mandatory"` 的 TeamBullet（跳过加权直接优先）
- 基于 `incompatible_tags` 判定冲突 vs 互补
- Team Pool 不可用时自动降级为纯 Local

**Interfaces:**
```python
class MultiPoolRetriever:
    """Implements StorageBackend Protocol — Generator sees it as a single pool."""

    def __init__(self, local: StorageBackend, team: StorageBackend | None = None,
                 config: LayerBoostConfig = None):
        self.pools = [local]
        if team:
            self.pools.append(team)

    def search(self, query: str, top_k: int) -> list[Bullet]:
        results = []
        for pool in self.pools:
            try:
                results.extend(pool.search(query, top_k))
            except Exception:
                logger.warning(f"Pool {pool} failed, skipping")
        return self._shadow_merge(results)

    def _shadow_merge(self, results: list[Bullet]) -> list[Bullet]:
        """Apply layer boost, handle mandatory, resolve conflicts."""
        ...
```

**Dependencies:** StorageBackend (Core Protocol), LayerBoostConfig

**FRs Addressed:** FR-026, FR-029

---

#### Component 14: TeamCacheStorage (Federation 缓存)

**Purpose:** 本地 Team Cache 的 StorageBackend 实现

**Responsibilities:**
- 实现 `StorageBackend` Protocol，提供对 Team Cache 的检索
- 管理本地缓存文件（`~/.ace/team_cache/{team_id}/`）
- 缓存上限 2000 条，按 `effective_score` 保留 Top-N
- 缓存 TTL + 墓碑机制
- 支持按 `subscribed_tags` 过滤

**Interfaces:**
```python
class TeamCacheStorage:
    """Read-only StorageBackend backed by local team cache."""

    def __init__(self, config: TeamConfig):
        self.cache_dir = Path(f"~/.ace/team_cache/{config.team_id}/")

    def search(self, query: str, top_k: int) -> list[TeamBullet]:
        """Search local team cache using vectors + keywords."""
        ...

    def refresh(self, sync_client: AceSyncClient) -> SyncResult:
        """Pull incremental updates from server."""
        ...

    def cleanup_tombstones(self) -> int:
        """Remove expired tombstone entries (>90 days)."""
        ...
```

**Dependencies:** TeamConfig, AceSyncClient (for refresh)

**FRs Addressed:** FR-031

---

#### Component 15: GitFallbackStorage (Git 只读存储)

**Purpose:** 从仓库内 `.ace/playbook.jsonl` 只读加载团队知识

**Responsibilities:**
- 实现 `StorageBackend` Protocol
- 只读加载 `.ace/playbook.jsonl`（严格不写入）
- 自动生成向量缓存 `.ace/playbook.vec`（gitignored）
- 读时一次性去重，缓存到 `.ace/playbook.cache`
- 模型指纹不匹配时降级为纯关键词检索
- 支持项目级 `taxonomy.json`

**Interfaces:**
```python
class GitFallbackStorage:
    """Read-only StorageBackend backed by .ace/playbook.jsonl."""

    def __init__(self, playbook_path: Path = None):
        self.playbook_path = playbook_path or Path(".ace/playbook.jsonl")
        self._bullets: list[TeamBullet] = []
        self._vec_cache: VectorCache | None = None

    def search(self, query: str, top_k: int) -> list[TeamBullet]:
        """Search with vectors (if available) or keywords (degraded)."""
        ...

    def _load_playbook(self) -> None:
        """Load JSONL, verify model fingerprint, build/load vector cache."""
        ...

    def _dedup_on_load(self) -> None:
        """One-time semantic dedup, cache to .ace/playbook.cache."""
        ...
```

**Dependencies:** ONNXEmbedder (optional, for vector cache generation)

**FRs Addressed:** FR-027

---

#### Component 16: AceSyncClient (Federation 同步客户端)

**Purpose:** 与 ACE Sync Server 通信的客户端

**Responsibilities:**
- 增量拉取 TeamBullet（基于 `updated_at` 差分）
- 推送提名 Bullet 到 Staging
- 提交 Supersede Proposal
- 拉取 Tag Taxonomy
- 投票（upvote/downvote）

**Interfaces:**
```python
class AceSyncClient:
    def __init__(self, server_url: str, auth_token: str = None):
        self._client = httpx.AsyncClient(base_url=server_url)

    async def pull_index(self, since: datetime, tags: list[str]) -> list[BulletIndex]:
        """Pull incremental bullet index since timestamp."""
        ...

    async def fetch_bullets(self, ids: list[str]) -> list[TeamBullet]:
        """Fetch full bullet data by IDs."""
        ...

    async def nominate_bullet(self, sanitized_bullet: Bullet) -> str:
        """Upload sanitized bullet to staging pool."""
        ...

    async def propose_supersede(self, team_bullet_id: str, new_bullet: Bullet) -> str:
        """Submit supersede proposal."""
        ...

    async def cast_vote(self, team_bullet_id: str, vote: Literal["up", "down"]) -> None:
        ...

    async def pull_taxonomy(self) -> TagTaxonomy:
        ...
```

**Dependencies:** httpx, TeamConfig

**FRs Addressed:** FR-031, FR-035, FR-036

---

#### Component 17: Redactor (Team 级脱敏引擎)

**Purpose:** 提名 Bullet 到 Team Pool 前的三层脱敏

**Responsibilities:**
- L1：确定性规则（正则替换路径、凭证、IP + custom_patterns）
- L2：用户审核（展示脱敏后内容，不可跳过）
- L3：LLM 泛化（可选，将具体经验抽象为通用规则）

**Interfaces:**
```python
class Redactor:
    def __init__(self, config: RedactorConfig, sanitizer: PrivacySanitizer):
        """Reuse Core's PrivacySanitizer for L1, extend with Team patterns."""

    def redact(self, bullet: Bullet) -> RedactedBullet:
        """L1: deterministic sanitization."""
        ...

    def present_for_review(self, redacted: RedactedBullet) -> ReviewResult:
        """L2: present to user for confirmation (CANNOT be skipped)."""
        ...

    def generalize(self, redacted: RedactedBullet, llm: LLMBase) -> RedactedBullet:
        """L3 (optional): LLM generalization."""
        ...
```

**Dependencies:** PrivacySanitizer (Core, reused), LLMBase (optional)

**FRs Addressed:** FR-033

---

#### Component 18: Nominator (提名流水线)

**Purpose:** 自动检测和推荐高质量 Local Bullet 用于 Team 共享

**Responsibilities:**
- 检测提名候选：`recall_count > 10` 且 `instructivity_score > 80`
- 频率控制：每会话最多 1 次弹窗
- 静默模式支持
- 编排 Redactor 脱敏流程
- 通过 AceSyncClient 上传到 Staging

**Interfaces:**
```python
class Nominator:
    def __init__(self, config: AutoNominateConfig, redactor: Redactor,
                 sync_client: AceSyncClient):

    def check_candidates(self, bullets: list[Bullet]) -> list[Bullet]:
        """Filter bullets that qualify for nomination."""
        ...

    async def nominate(self, bullet: Bullet) -> NominationResult:
        """Full pipeline: redact → user review → upload."""
        ...

    def list_pending(self) -> list[Bullet]:
        """List candidates for silent mode (ace nominate list)."""
        ...
```

**Dependencies:** Redactor, AceSyncClient, AutoNominateConfig

**FRs Addressed:** FR-032

---

#### Component 19: TeamConfig (独立团队配置)

**Purpose:** Team Layer 的独立配置，不修改 Core Config

**Interfaces:**
```python
class TeamConfig(BaseModel):
    enabled: bool = False
    server_url: str = ""
    team_id: str = ""
    subscribed_tags: list[str] = []
    cache_max_bullets: int = 2000
    cache_ttl_minutes: int = 60
    auto_nominate: AutoNominateConfig = AutoNominateConfig()
    redactor: RedactorConfig = RedactorConfig()
    layer_boost: LayerBoostConfig = LayerBoostConfig()
    mandatory_overrides: list[MandatoryOverride] = []

class AutoNominateConfig(BaseModel):
    enabled: bool = True
    min_recall_count: int = 10
    min_instructivity_score: int = 80
    max_prompts_per_session: int = 1
    silent: bool = False

class RedactorConfig(BaseModel):
    llm_generalize: bool = False
    custom_patterns: list[str] = []

class LayerBoostConfig(BaseModel):
    local: float = 1.5
    team: float = 1.0

class MandatoryOverride(BaseModel):
    bullet_id: str
    reason: str
    expires: datetime
```

**Dependencies:** Pydantic

**FRs Addressed:** FR-025 (decoupling)

---

## Data Architecture

### Data Model

#### Bullet (v2.0 updated)

```
┌────────────────────────────────────────┐
│           Bullet (知识单元)              │
│                                        │
│ ┌─ Identity ─────────────────────────┐ │
│ │ id: UUID                           │ │
│ │ scope: "global"|"project:{name}"   │ │
│ │        |"team:{id}"  ◄── v2.0      │ │
│ │ schema_version: int = 1  ◄── v2.0  │ │
│ └────────────────────────────────────┘ │
│ ┌─ Content ──────────────────────────┐ │
│ │ content: str (≤500 chars)          │ │
│ │ distilled_rule: str? ("When X...") │ │
│ │ code_content: str? (≤3 lines)      │ │
│ └────────────────────────────────────┘ │
│ ┌─ Classification ───────────────────┐ │
│ │ section: BulletSection (8 types)   │ │
│ │ knowledge_type: KnowledgeType (5)  │ │
│ │ source_type: SourceType (3)        │ │
│ │ instructivity_score: int (0-100)   │ │
│ │ incompatible_tags: list[str] = []  │ │
│ │                       ◄── v2.0     │ │
│ └────────────────────────────────────┘ │
│ ┌─ Lifecycle ────────────────────────┐ │
│ │ recall_count: int                  │ │
│ │ last_recall: datetime?             │ │
│ │ decay_weight: float (0.0-1.0)     │ │
│ │ created_at: datetime               │ │
│ │ updated_at: datetime               │ │
│ └────────────────────────────────────┘ │
│ ┌─ Relations ────────────────────────┐ │
│ │ related_tools: list[str]           │ │
│ │ related_files: list[str]           │ │
│ │ key_entities: list[str]            │ │
│ │ tags: list[str]                    │ │
│ └────────────────────────────────────┘ │
└────────────────────────────────────────┘
```

#### TeamBullet (v2.0 新增)

```
┌────────────────────────────────────────┐
│     TeamBullet extends Bullet           │
│     (schema_version = 2)               │
│                                        │
│ ┌─ Team Identity ────────────────────┐ │
│ │ author_id: str (pseudonym, GDPR)   │ │
│ │ origin_id: str? (Supersede source) │ │
│ └────────────────────────────────────┘ │
│ ┌─ Team Governance ──────────────────┐ │
│ │ enforcement: "mandatory"|"suggest" │ │
│ │ upvotes: int = 0                   │ │
│ │ downvotes: int = 0                 │ │
│ │ status: pending|approved|archived  │ │
│ │         |tombstone                 │ │
│ │ deleted_at: datetime?              │ │
│ └────────────────────────────────────┘ │
│ ┌─ Context ──────────────────────────┐ │
│ │ context_summary: str?              │ │
│ └────────────────────────────────────┘ │
│                                        │
│ Schema Compatibility:                  │
│   v1 → v2: auto-fill defaults         │
│   v2 → v1: serde(flatten) keep fields │
└────────────────────────────────────────┘
```

### Storage Isolation (v2.0 新增)

| 维度 | Local Memory | Team Cache | Git Fallback |
|------|-------------|------------|--------------|
| **存储路径** | `~/.ace/{product}/` | `~/.ace/team_cache/{team_id}/` | `.ace/playbook.jsonl` (repo) |
| **数据格式** | 混合存储（元数据 + 向量索引） | 独立缓存格式 | JSONL + 生成的向量缓存 |
| **生命周期** | Decay 引擎管理 | TTL + 墓碑机制 | Git 版本控制 |
| **Schema** | Bullet (v1) | TeamBullet (v2) | TeamBullet (v2) |
| **写入权限** | 读写 | 只读（本地缓存）| 只读 |
| **谁管理** | Core Engine | Team Layer | Tech Lead / Curator |

**关键约束：Team Layer 对 Local Pool 只读不写。** Team 信息只通过 Shadow Merge 在检索结果层面合并，不会写入 Local Pool。

### Database Design Updates (v2.0)

**Core payload 新增字段（嵌入 mem0 VectorStore payload）：**

```json
{
  "memx_schema_version": 1,
  "memx_incompatible_tags": []
}
```

**Team Cache 存储结构（独立于 Core）：**

```
~/.ace/team_cache/{team_id}/
├── bullets.jsonl           # TeamBullet 序列化
├── vectors.bin             # 向量数据
├── index.json              # ID → offset 索引
├── taxonomy.json           # Tag Taxonomy 快照
└── sync_state.json         # last_sync_timestamp, tombstones
```

**Git Fallback 存储结构（仓库内）：**

```
.ace/
├── playbook.jsonl          # Git 追踪（只读，手动维护）
├── taxonomy.json           # 可选：标签归一化词表
├── .gitignore              # 包含: playbook.vec, playbook.cache
├── playbook.vec            # gitignored（本地生成的向量缓存）
└── playbook.cache          # gitignored（去重后的内存快照缓存）
```

### Data Flow Updates (v2.0)

**Retrieval Flow with Team (search)：**
```
query → MemXMemory.search()
  │
  └── ace_enabled=True → RetrievalPipeline.search()
        │
        ├── MultiPoolRetriever.search(query)  ◄── v2.0
        │   ├── LocalPool.search(query)       → local_results
        │   │   ├── L1-L4 as before
        │   │   └── ScoreMerger
        │   │
        │   ├── TeamPool.search(query)        → team_results  (if available)
        │   │   ├── TeamCacheStorage.search()  (Federation)
        │   │   └── GitFallbackStorage.search() (Git Fallback)
        │   │
        │   └── ShadowMerger.merge(local_results, team_results)
        │       ├── mandatory Bullets → direct inject
        │       ├── Local boost × 1.5
        │       ├── Team boost × 1.0
        │       ├── Incompatible Tags → conflict resolution
        │       └── Complementary (similarity ≥ 0.8) → keep both
        │
        ├── DecayWeighter
        ├── TokenBudgetTrimmer
        ├── Reranker (optional)
        ├── RecallReinforcer (async)
        └── Return results
```

**Promotion Flow (v2.0 新增)：**
```
IngestPipeline completes → new Bullet stored in Local Pool
  │
  └── [async] Nominator.check_candidates()
        │
        ├── recall_count > 10 AND score > 80 → candidate
        │
        ├── Frequency check: < max_prompts_per_session?
        │
        ├── Redactor.redact(candidate)
        │   ├── L1: regex sanitization
        │   └── L2: user review (CANNOT skip)
        │
        └── AceSyncClient.nominate_bullet(redacted)
            → Team Server Staging Pool
```

**Team Cache Sync Flow (v2.0 新增)：**
```
Session Start
  │
  └── [async] TeamCacheStorage.refresh(sync_client)
        │
        ├── pull_index(since=last_sync_timestamp, tags=subscribed_tags)
        │
        ├── fetch_bullets(new_ids)
        │
        ├── Apply tombstones (soft delete)
        │
        ├── Enforce cache_max_bullets (Top-N by effective_score)
        │
        └── Update sync_state.json
```

---

## API Design

### API Architecture

**类型:** Python Library API (非 REST)

核心 API 与 v1.0 完全一致（参见 v1.0 架构文档）。以下仅列出 v2.0 新增的 API。

### New API Endpoints (v2.0)

```python
# === Team CLI Commands (v2.0 新增) ===

# Team 状态
ace team status
# → { "mode": "federation|git_fallback|disabled",
#      "cache_count": 1523, "last_sync": "...",
#      "subscribed_tags": [...] }

# Team 初始化
ace team init --server-url https://ace.company.com
# → Configure team connection

# 手动提名
ace nominate list
# → List candidates for team sharing

ace nominate submit <bullet_id>
# → Redact + review + upload to staging

# 投票
ace upvote <team_bullet_id>
ace downvote <team_bullet_id>

# Supersede
ace supersede <team_bullet_id> --content "corrected knowledge"
# → Submit supersede proposal

# 手动同步
ace team sync
# → Force incremental sync

# 导入 Git Fallback 到 Federation
ace import --from .ace/playbook.jsonl --to team:{id}


# === Daemon IPC Extensions (v2.0) ===

REQUEST:  {"cmd": "team_sync"}
RESPONSE: {"status": "ok", "synced": 42, "tombstoned": 3}

REQUEST:  {"cmd": "team_status"}
RESPONSE: {"mode": "federation", "cache_count": 1523, "last_sync": "..."}
```

### Authentication & Authorization (v2.0 additions)

**Federation Mode:**
- **Client → Sync Server:** API Key 或 OIDC Token（由 TeamConfig 配置）
- **RBAC 角色：** Contributor（提名）、Reviewer（投票）、Curator（审核）、Admin（管理）
- **假名标识：** `author_id` 使用哈希化的用户 ID，GDPR 友好

**Git Fallback:**
- 无额外认证。复用 Git 仓库的现有权限控制。

---

## Non-Functional Requirements Coverage

### NFR-001 ~ NFR-010 (v1.0)

与 v1.0 架构文档完全一致。详细的 Architecture Solution、Implementation Notes、Validation 参见 v1.0。

以下仅列出 v2.0 新增的 NFR 覆盖方案。

---

### NFR-011: Performance — Team 检索延迟 < 100ms (v2.0)

**Requirement:** 含 Team Cache/Git Fallback 的端到端检索 < 100ms。Pre-Inference 零远程请求。

**Architecture Solution:**
- **Pre-Inference 纯本地化：** Team Cache 含完整向量数据，检索时零远程请求
- **Local Pool < 50ms：** 与 NFR-001 相同的优化策略
- **Team Cache 增量 < 40ms：** Team Cache 使用独立的内存向量索引，与 Local Pool 并行查询
- **Shadow Merge < 5ms：** 纯内存计算（排序 + 去重 + 加权），无 I/O
- **Git Fallback 向量缓存：** `.ace/playbook.vec` 首次加载后常驻内存，后续检索零磁盘 I/O
- **Post-Inference 异步补充（可选）：** 本次检索后异步向 Server 查询，结果缓存供下次使用

**Implementation Notes:**
- MultiPoolRetriever 可并行查询 Local 和 Team Pool（asyncio.gather）
- Team Cache 内存占用估算：2000 条 × 384 维 × 4 bytes = ~3MB
- Shadow Merge 使用 heapq.nlargest 避免全排序

**Validation:**
- pytest-benchmark: search with team cache < 100ms
- 断网测试：Team 不可达时延迟 < 50ms（与纯 Local 一致）

---

### NFR-012: Security — Team 数据隔离 (v2.0)

**Requirement:** Local Pool 和 Team Cache 数据完全隔离。

**Architecture Solution:**
- **物理路径隔离：** `~/.ace/{product}/` vs `~/.ace/team_cache/{team_id}/`
- **代码路径隔离：** `memx/core/` 中无任何 Team 写入代码路径
- **运行时隔离：** TeamCacheStorage 实例独立于 Core StorageBackend 实例
- **多 Team 隔离：** 不同 `team_id` 的缓存存储在不同子目录

**Implementation Notes:**
```python
# CI 静态检查：确保 Core 不 import Team
# .github/workflows/check-decoupling.yml
- name: Check Core does not import Team
  run: |
    ! grep -r "from memx.team" memx/core/
    ! grep -r "import memx.team" memx/core/
```

**Validation:**
- CI 静态检查：Core 不 import Team
- 集成测试：Team 功能禁用时 Core 测试 100% 通过

---

### NFR-013: Security — Team 隐私保护 (v2.0)

**Requirement:** 提名到 Team Pool 的知识必须经过完整脱敏。

**Architecture Solution:**
- **L1 确定性脱敏：** Redactor 复用 Core 的 PrivacySanitizer + Team 扩展规则
- **L2 用户审核：** Nominator 在上传前强制展示脱敏后内容给用户确认
- **L3 LLM 泛化（可选）：** 将 "项目 X 中发现的 Y" 泛化为 "当使用 Y 时..."
- **假名标识：** `author_id = sha256(user_id + team_salt)[:16]`
- **审计日志：** 所有提名和审核操作记录到 `~/.ace/team_cache/{team_id}/audit.log`

**Implementation Notes:**
```python
class Nominator:
    async def nominate(self, bullet: Bullet) -> NominationResult:
        # L1: deterministic redaction (CANNOT skip)
        redacted = self.redactor.redact(bullet)

        # L2: user review (CANNOT skip)
        review = self.redactor.present_for_review(redacted)
        if not review.approved:
            return NominationResult(status="rejected_by_user")

        # L3: optional LLM generalization
        if self.config.redactor.llm_generalize:
            redacted = self.redactor.generalize(redacted, self.llm)

        # Upload to staging
        return await self.sync_client.nominate_bullet(redacted.bullet)
```

---

### NFR-014: Reliability — Team 可剥离性 (v2.0)

**Requirement:** Team 功能可完整移除而不影响 Local Memory。

**Architecture Solution:**
- **包结构解耦：** `memx/team/` 作为独立可选包，Core 不 import
- **可选依赖：** `pip install memx[team]` 安装 Team 依赖（httpx 等）
- **初始化层胶水：** `memx/ext/team_bootstrap.py` 是唯一知道 Team 存在的文件
- **条件导入：** `team_bootstrap.py` 使用 `try: import memx.team` 保护

**Implementation Notes:**
```python
# memx/ext/team_bootstrap.py
def try_bootstrap_team(memory, config_path: str = None):
    """Attempt to load and initialize Team Layer. No-op if unavailable."""
    try:
        from memx.team import TeamConfig, TeamCacheStorage, GitFallbackStorage
        from memx.team import MultiPoolRetriever
    except ImportError:
        return  # Team package not installed, silently skip

    team_config = TeamConfig.load(config_path)
    if not team_config.enabled:
        return  # Team not configured, silently skip

    # ... inject team storage
```

**Validation:**
- CI 测试矩阵包含 `pip install memx`（无 team）场景
- 删除 `memx/team/` 后运行全部 Core 测试

---

### NFR-006 Update: 优雅降级 (v2.0 扩展)

**v2.0 新增降级链：**
- Team Server 不可达 → 使用上次 Team Cache 快照
- Team Cache 为空 → 纯 Local 检索（与 Team 未启用时一致）
- Git Fallback 文件不存在 → 纯 Local 检索
- Git Fallback 模型指纹不匹配 → 降级为纯关键词检索

**关键约束：** 断网测试需验证 Team 不可达时延迟、结果与纯 Local 无差异。

---

## Security Architecture (v2.0 additions)

### Team Authentication

| 模式 | 认证方式 | 说明 |
|------|----------|------|
| Git Fallback | 无（复用 Git 权限） | 零额外配置 |
| Federation (Lite) | API Key | 小团队 5 分钟部署 |
| Federation (Full) | OIDC (JWT) | 企业级 SSO 集成 |

### Team Authorization (RBAC)

| 角色 | 权限 |
|------|------|
| Contributor | 提名 Bullet 到 Staging |
| Reviewer | 投票（upvote/downvote） |
| Curator | 审核 Staging → Team Pool，管理 Taxonomy |
| Admin | 管理用户、角色、服务器配置 |

### Data Flow Security

```
Local Bullet → Redactor L1 (regex) → Redactor L2 (user review, CANNOT skip)
  → [optional L3 LLM generalize]
  → Upload to Team Server (TLS 1.2+)
  → Staging Pool → Three-tier review → Team Pool
  → Clients pull via incremental sync (TLS 1.2+)
```

---

## Development Architecture

### Code Organization (v2.0 updated)

```
memx/
├── core/                          # ◄── v2.0: 重命名为 core/（原 memx/ 根目录内容）
│   ├── __init__.py                # 公开 API: Memory, AsyncMemory
│   ├── memory.py                  # MemXMemory 类
│   ├── async_memory.py
│   ├── config.py                  # MemXConfig + 子配置 (不含 TeamConfig)
│   ├── types.py                   # BulletMetadata (含 schema_version, incompatible_tags)
│   ├── exceptions.py
│   ├── engines/
│   │   ├── reflector/             # ReflectorEngine + Stages
│   │   ├── curator/               # CuratorEngine + ConflictDetector
│   │   ├── decay/                 # DecayEngine + formulas
│   │   └── generator/             # GeneratorEngine + Matchers
│   ├── pipeline/
│   │   ├── ingest.py              # IngestPipeline
│   │   └── retrieval.py           # RetrievalPipeline
│   ├── privacy/
│   │   ├── sanitizer.py           # PrivacySanitizer
│   │   └── patterns.py
│   ├── integration/
│   │   ├── manager.py
│   │   ├── hooks.py
│   │   └── cli_hooks.py
│   ├── embeddings/
│   │   └── onnx.py
│   ├── daemon/
│   │   ├── server.py
│   │   ├── client.py
│   │   └── ipc.py
│   ├── cli/
│   │   └── main.py
│   └── utils/
│       ├── bullet_factory.py
│       ├── token_counter.py
│       └── text_processing.py
│
├── team/                          # ◄── v2.0 新增: 独立可选包
│   ├── __init__.py
│   ├── config.py                  # TeamConfig (独立于 core config)
│   ├── types.py                   # TeamBullet extends Bullet
│   ├── cache_storage.py           # TeamCacheStorage
│   ├── git_storage.py             # GitFallbackStorage
│   ├── sync_client.py             # AceSyncClient
│   ├── merger.py                  # MultiPoolRetriever + ShadowMerger
│   ├── redactor.py                # Redactor (L1+L2+L3)
│   ├── nominator.py               # Nominator (Promotion Pipeline)
│   └── cli.py                     # Team CLI commands (ace team ...)
│
├── ext/                           # ◄── v2.0 新增: 初始化 / 胶水层
│   └── team_bootstrap.py          # 检测 team 配置，注入 Team Layer
│
└── __init__.py                    # 顶层导出

mem0/                              # mem0 原始代码（Fork, 最小修改）
└── ... (保持不变)

tests/
├── unit/
│   ├── test_reflector.py
│   ├── test_curator.py
│   ├── test_decay.py
│   ├── test_generator.py
│   ├── test_sanitizer.py
│   ├── test_bullet.py
│   ├── test_config.py
│   ├── team/                      # ◄── v2.0 新增: Team 独立测试
│   │   ├── test_team_cache.py
│   │   ├── test_git_fallback.py
│   │   ├── test_shadow_merge.py
│   │   ├── test_redactor.py
│   │   ├── test_nominator.py
│   │   └── test_sync_client.py
│   └── test_decoupling.py         # ◄── v2.0 新增: 解耦验证
├── integration/
│   ├── test_ingest_pipeline.py
│   ├── test_retrieval_pipeline.py
│   ├── test_mem0_compat.py
│   ├── test_daemon.py
│   └── test_team_retrieval.py     # ◄── v2.0 新增
├── performance/
│   ├── test_search_latency.py
│   ├── test_reflect_latency.py
│   └── test_team_search_latency.py # ◄── v2.0 新增
└── conftest.py
```

### Module Dependency Direction (v2.0 updated)

```
cli → memory → pipeline → engines → types/config
                  │                      │
                  └── mem0 infrastructure ┘

privacy → (独立，被 pipeline 引用)
daemon → memory + engines

          ┌─────────────────────────────────────┐
          │            Core (不可修改)            │
          │  engines │ pipeline │ types │ config  │
          └──────────────┬──────────────────────┘
                         │ StorageBackend Protocol (扩展接口)
          ┌──────────────▼──────────────────────┐
          │         Team Layer (可选)             │
          │  cache_storage │ git_storage          │
          │  sync_client   │ merger │ redactor    │
          └──────────────┬──────────────────────┘
                         │ HTTP API
          ┌──────────────▼──────────────────────┐
          │       ACE Sync Server (独立项目)      │
          └─────────────────────────────────────┘

依赖方向严格单向：Core ← Team Layer ← Sync Server
ext/team_bootstrap.py 是唯一知道 Team 存在的胶水层
```

### Testing Strategy (v2.0 additions)

| 层级 | 工具 | 覆盖率目标 | 说明 |
|------|------|-----------|------|
| 单元测试 | pytest | > 80% | 每个 Engine + 每个 Matcher |
| 集成测试 | pytest | 关键路径 100% | add→reflect→curate→search→decay |
| mem0 兼容测试 | pytest | 100% | 运行 mem0 官方测试套件 |
| 性能测试 | pytest-benchmark | CI 门禁 | 检索 < 50ms, 蒸馏 < 20ms |
| 类型检查 | mypy (strict) | 100% | memx/ 包强制 |
| **Team 单元测试 (v2.0)** | pytest | > 80% | Team 独立 test suite |
| **解耦验证 (v2.0)** | pytest + CI | 100% | Team 禁用时 Core 测试 100% 通过 |
| **Team 性能测试 (v2.0)** | pytest-benchmark | CI 门禁 | Team 检索 < 100ms, Shadow Merge < 5ms |

### CI/CD Pipeline (v2.0 additions)

```
Push / PR
    │
    ├── ruff check (lint + format)
    ├── mypy --strict memx/
    │
    ├── [Core Tests — must pass without Team]
    │   ├── pytest tests/unit/ --ignore=tests/unit/team/ --cov --cov-fail-under=80
    │   ├── pytest tests/integration/
    │   ├── pytest tests/integration/test_mem0_compat.py
    │   └── pytest tests/performance/ --benchmark-compare
    │
    ├── [Decoupling Check] ◄── v2.0
    │   ├── grep -r "from memx.team" memx/core/ → MUST be empty
    │   └── pytest tests/unit/test_decoupling.py
    │
    ├── [Team Tests — separate suite] ◄── v2.0
    │   ├── pytest tests/unit/team/ --cov
    │   ├── pytest tests/integration/test_team_retrieval.py
    │   └── pytest tests/performance/test_team_search_latency.py
    │
    ▼
  All Green → Merge allowed

Release Tag
    │
    ├── poetry build
    ├── twine upload (PyPI) — memx + memx[team]
    └── GitHub Release
```

---

## Deployment Architecture (v2.0 additions)

### Package Distribution

```
pip install memx            # Core only: Rules-only, 纯关键词
pip install memx[onnx]      # + 本地 ONNX Embedding
pip install memx[team]      # + Team Memory (httpx, etc.)  ◄── v2.0
pip install memx[graph]     # + 图存储支持
pip install memx[all]       # 全部依赖
```

### ACE Sync Server (独立项目, v2.0)

ACE Sync Server 作为**独立项目**发布，不与 memx 核心库耦合。

| 部署模式 | 适用规模 | 技术栈 | 部署时间 |
|----------|----------|--------|----------|
| **Lite** | < 20 人 | 单容器, SQLite, API Key 认证 | ~5 分钟 |
| **Full** | 20+ 人 | PostgreSQL + Qdrant + OIDC | 按需 |

```dockerfile
# ACE Sync Server Lite (Docker Compose)
services:
  ace-server:
    image: ace-sync-server:lite
    ports: ["8080:8080"]
    volumes: ["./data:/data"]
    environment:
      ACE_AUTH_MODE: api_key
      ACE_API_KEY: ${ACE_API_KEY}
```

---

## Requirements Traceability

### Functional Requirements Coverage

| FR ID | FR Name | Components | Implementation Notes |
|-------|---------|------------|---------------------|
| FR-001 | Bullet 数据模型 | BulletFactory, BulletMetadata | `core/types.py` — v2.0 +schema_version, +incompatible_tags |
| FR-002 | Reflector Stage 1 | PatternDetector | `core/engines/reflector/detector.py` |
| FR-003 | Reflector Stage 2 | KnowledgeScorer | `core/engines/reflector/scorer.py` |
| FR-004 | Reflector Stage 3 | PrivacySanitizer | `core/privacy/sanitizer.py` |
| FR-005 | Reflector Stage 4 | BulletDistiller | `core/engines/reflector/distiller.py` |
| FR-006 | Curator 去重 | CuratorEngine | `core/engines/curator/engine.py` |
| FR-007 | Decay 衰退 | DecayEngine | `core/engines/decay/engine.py` |
| FR-008 | Decay 召回强化 | DecayEngine.reinforce | `core/engines/decay/engine.py` |
| FR-009 | Generator 混合检索 | GeneratorEngine, 4x Matcher | `core/engines/generator/` |
| FR-010 | Generator 降级模式 | GeneratorEngine.mode | `core/engines/generator/engine.py` |
| FR-011 | Token 预算 | TokenBudgetTrimmer | `core/utils/token_counter.py` |
| FR-012 | 配置系统 | MemXConfig + 子配置 | `core/config.py` |
| FR-013 | API 兼容 | MemXMemory (Decorator) | `core/memory.py` |
| FR-014 | Pre-Inference Hook | PreInferenceHook | `core/integration/hooks.py` |
| FR-015 | Post-Action Hook | PostActionHook | `core/integration/hooks.py` |
| FR-016 | Session-End Hook | SessionEndHook | `core/integration/hooks.py` |
| FR-017 | ONNX Embedding | ONNXEmbedder | `core/embeddings/onnx.py` |
| FR-018 | Daemon | MemXDaemon + DaemonClient | `core/daemon/` |
| FR-019 | LLM 增强蒸馏 | ReflectorEngine (llm mode) | `core/engines/reflector/engine.py` |
| FR-020 | 冲突检测 | ConflictDetector | `core/engines/curator/conflict.py` |
| FR-021 | 层级 Scope | MemXMemory + Generator | `core/memory.py` — v2.0 +team:{id} |
| FR-022 | 导入导出 | MemXMemory.export/import | `core/memory.py` — v2.0 +JSONL format |
| FR-023 | CLI 命令 | Click CLI app | `core/cli/main.py` |
| FR-024 | PyPI 发布 | pyproject.toml + CI/CD | v2.0 +memx[team] extra |
| **FR-025** | **Core/Team 解耦** | **core/, team/, ext/** | **`ext/team_bootstrap.py`** |
| **FR-026** | **StorageBackend 扩展** | **MultiPoolRetriever** | **`team/merger.py`** |
| **FR-027** | **Git Fallback** | **GitFallbackStorage** | **`team/git_storage.py`** |
| **FR-028** | **TeamBullet 模型** | **TeamBullet** | **`team/types.py`** |
| **FR-029** | **Shadow Merge** | **MultiPoolRetriever._shadow_merge** | **`team/merger.py`** |
| **FR-030** | **Mandatory 逃生舱** | **MultiPoolRetriever + TeamConfig** | **`team/merger.py`, `team/config.py`** |
| **FR-031** | **Team Cache 同步** | **TeamCacheStorage, AceSyncClient** | **`team/cache_storage.py`, `team/sync_client.py`** |
| **FR-032** | **提名流水线** | **Nominator** | **`team/nominator.py`** |
| **FR-033** | **Redactor 脱敏** | **Redactor** | **`team/redactor.py`** |
| **FR-034** | **三层审核治理** | **AceSyncClient + Server** | **Server-side (独立项目)** |
| **FR-035** | **Team Supersede** | **AceSyncClient** | **`team/sync_client.py`** |
| **FR-036** | **Tag Taxonomy** | **AceSyncClient + Reflector** | **`team/sync_client.py`** |
| **FR-037** | **订阅与分发** | **TeamCacheStorage** | **`team/cache_storage.py`** |

### Non-Functional Requirements Coverage

| NFR ID | NFR Name | Solution | Validation |
|--------|----------|----------|------------|
| NFR-001 | 检索 < 50ms | 内存索引 + 异步强化 | pytest-benchmark CI 门禁 |
| NFR-002 | 蒸馏 < 20ms | 纯规则 + 正则预编译 | pytest-benchmark |
| NFR-003 | 隐私不可关闭 | Sanitizer hardcoded in pipeline | 代码审计 + 测试 |
| NFR-004 | 数据本地化 | 默认零网络 + 提名需用户确认 | 代码审计 |
| NFR-005 | mem0 兼容 | Decorator 模式 | mem0 测试套件 100% |
| NFR-006 | 优雅降级 | Stage 级 try-catch + Team 降级链 | 故障注入 + 断网测试 |
| NFR-007 | 测试 > 80% | pytest-cov CI 门禁 + Team 独立 suite | 覆盖率报告 |
| NFR-008 | 上游同步 | 独立目录 + 月度 rebase | 文档化流程 |
| NFR-009 | 5K-50K 条 | 分级存储方案 | 压力测试 |
| NFR-010 | 零配置启动 | 默认值 + Git Fallback 自动检测 | 冒烟测试 |
| **NFR-011** | **Team 检索 < 100ms** | **本地 Cache + 并行查询 + Shadow Merge < 5ms** | **pytest-benchmark** |
| **NFR-012** | **Team 数据隔离** | **路径隔离 + 代码隔离 + CI 静态检查** | **CI grep + 集成测试** |
| **NFR-013** | **Team 隐私保护** | **Redactor L1+L2+假名+审计** | **L2 不可跳过 + 审计日志** |
| **NFR-014** | **Team 可剥离性** | **独立包 + 条件导入 + 胶水层** | **CI 无 team 测试矩阵** |

---

## Trade-offs & Decision Log

### Decisions 1-5 (v1.0)

与 v1.0 完全一致。参见：Decorator vs 继承、Bullet 嵌入 payload vs 独立表、memx/ 独立包、Rules-only 默认、同步管线。

### Decision 6: 组合模式 vs 修改 Generator (v2.0)

**选择：** 组合模式（MultiPoolRetriever 注入 Generator）

**Trade-off:**
- 得到：Core Generator 零改动，Team 功能可完整移除
- 失去：初始化层需要胶水代码（`team_bootstrap.py`），间接层增加调试复杂度

**Rationale:** 解耦是硬约束（NFR-014）。Generator 是 Core 最复杂的组件，任何修改都有回归风险。

---

### Decision 7: Team Cache 独立存储 vs 共享 VectorStore (v2.0)

**选择：** Team Cache 使用独立文件存储（非 mem0 VectorStore）

**Trade-off:**
- 得到：数据完全隔离（NFR-012），Team 功能移除时零残留
- 失去：无法复用 mem0 的 23 个 VectorStore Provider

**Rationale:** Team Cache 是只读缓存，不需要 VectorStore 的写入能力。独立存储更简单且隔离性更好。

---

### Decision 8: Git Fallback JSONL vs 数据库 (v2.0)

**选择：** JSONL 纯文本文件（引擎只读）

**Trade-off:**
- 得到：Git 可 diff/review、手动可编辑、零外部依赖、PR Review 友好
- 失去：无法高效查询（需要全量加载），大文件时性能差

**Rationale:** Git Fallback 定位是"零基础设施"方案。性能通过本地向量缓存（`.ace/playbook.vec`）补偿。

---

### Decision 9: TeamConfig 独立 vs 嵌入 MemXConfig (v2.0)

**选择：** TeamConfig 完全独立于 MemXConfig

**Trade-off:**
- 得到：Core Config 零修改，Team 可独立配置和升级
- 失去：用户需要管理两个配置文件/位置

**Rationale:** 解耦约束要求 Team 配置不污染 Core。用户通过 `ace team init` 命令初始化 Team 配置，体验可接受。

---

### Decision 10: Pre-Inference 纯本地 vs 允许远程 (v2.0)

**选择：** Pre-Inference 阶段严格纯本地，不做实时远程请求

**Trade-off:**
- 得到：检索延迟可预测（< 100ms），断网时无影响
- 失去：可能错过 Team Pool 中的相关知识（Cache 未命中）

**Rationale:** 延迟可预测性是用户体验底线。通过 Post-Inference 异步补充查询弥补覆盖率，结果缓存供下次使用。

---

## Open Issues & Risks

### v1.0 Issues (preserved)

1. **PyPI 包名 `memx` 可用性**
2. **mem0 v1.0.x → v2.x 升级风险**
3. **ONNX 模型中文质量**
4. **Windows Named Pipe Python 支持**
5. **L1-L3 内存索引内存占用**

### v2.0 Issues (新增)

6. **Team Cache 容量上限**：2000 条对大型团队（50+ 人）可能不足。缓解：按 `subscribed_tags` 分片 + Post-Inference 异步补充。
7. **Federation Server 落地复杂度**：即使 Lite 模式，仍需管理容器。缓解：提供 Docker Compose 一键部署。
8. **Taxonomy 冷启动**：第一批 tags 从哪来。缓解：预设模板 + 种子聚合。
9. **Supersede 时间窗口**：提交到全员同步可能有数天延迟。缓解：urgent 级别即时推送。
10. **Core/Team 包重构成本**：将 `memx/` 重命名为 `memx/core/` 需要更新所有 import path。缓解：P0 阶段一次性完成。

---

## Capacity Planning (v2.0 updated)

| 记忆规模 | Local 内存 | Local 磁盘 | Local 延迟 | + Team Cache | + Team 延迟 |
|----------|-----------|-----------|-----------|-------------|-------------|
| 100 条 | ~5MB | ~1MB | < 5ms | +3MB | < 10ms |
| 1,000 条 | ~15MB | ~5MB | < 15ms | +3MB | < 25ms |
| 5,000 条 | ~50MB | ~20MB | < 50ms | +3MB | < 80ms |
| 10,000 条 | ~100MB | ~40MB | < 80ms | +3MB | < 100ms |
| 50,000 条 | ~500MB | ~200MB | < 100ms | +3MB | < 120ms |

Team Cache 固定 ~3MB（2000 条 × 384 维 × 4 bytes）。

---

## Future Considerations

### v1.0 (preserved)
- MemX Cloud：基于 Daemon 扩展为远程服务
- 多模型 Embedding 切换
- Rust 核心引擎（L1-L3 Matcher via PyO3）
- MCP Server 模式
- 知识可视化

### v2.0 (新增)
- **ACE Sync Server 开源发布**：作为独立项目发布参考实现
- **Team Analytics Dashboard**：团队知识贡献统计、Cache Miss 指标、Top Contributors
- **Cross-Team Federation**：跨团队知识共享（企业级）
- **Real-time Push Notification**：WebSocket 推送 urgent Supersede
- **Embedding 模型一致性**：Team 成员使用不同 Embedding 模型时的兼容方案

---

## Implementation Priority (v2.0)

与 PRD Appendix C 和 `ace-team-memory-architecture.md` 第 12 节对齐：

| 阶段 | 对应 EPIC | 关键组件 | 核心产出 |
|------|----------|---------|---------|
| **P0: 解耦重构** | EPIC-009 | `core/`, `team/`, `ext/team_bootstrap.py` | 包结构分离，Core 测试 100% 通过 |
| **P1: Git Fallback** | EPIC-010 | GitFallbackStorage, MultiPoolRetriever | 仓库有 `.ace/playbook.jsonl` 即可用 |
| **P2: Federation MVP** | EPIC-011 | TeamCacheStorage, AceSyncClient, Nominator, Redactor | 最小可用 Federation Mode |
| **P3: 治理** | EPIC-012 | 三层审核, Supersede, Taxonomy | 完整治理流水线 |
| **P4: 运维** | (独立项目) | Docker Compose, 监控 | 生产可用 |

**Core EPICs (EPIC-001~008) 的实施顺序与 v1.0 一致：**
1. EPIC-001: Bullet + Config + API 兼容
2. EPIC-004: Decay 引擎
3. EPIC-002: Reflector 引擎
4. EPIC-003: Curator 引擎
5. EPIC-005: Generator 引擎
6. EPIC-006: Integration Layer
7. EPIC-007: ONNX + Daemon
8. EPIC-008: CLI + 发布

**P0 (EPIC-009) 可在 EPIC-001 之后、EPIC-002 之前执行**，确保后续 Core 开发已在正确的包结构中进行。

---

## Approval & Sign-off

**Review Status:**
- [ ] Technical Lead (TPY)
- [ ] Product Owner (TPY)
- [ ] Security Review
- [ ] DevOps Review

---

**This document was created using BMAD Method v6 - Phase 3 (Solutioning)**

*To continue: Run `/sprint-planning` to plan implementation sprints.*
