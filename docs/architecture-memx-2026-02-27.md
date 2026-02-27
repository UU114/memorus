# System Architecture: MemX

**Date:** 2026-02-27
**Architect:** TPY
**Version:** 1.0
**Project Type:** AI 记忆引擎（mem0 Fork + ACE 智能层）
**Project Level:** Level 3（大型项目）
**Status:** Draft

---

## Document Overview

This document defines the system architecture for MemX. It provides the technical blueprint for implementation, addressing all 24 functional and 10 non-functional requirements from the PRD.

**Related Documents:**
- Product Requirements Document: `docs/prd-memx-2026-02-27.md`
- Product Brief: `docs/product-brief-memx-2026-02-27.md`
- ACE Analysis Report: `doc/ace-mem0-analysis-report.md`

---

## Executive Summary

MemX 在 mem0 之上叠加 ACE 智能层，形成"Layered Engine on Fork"架构。核心设计原则：新增模块完全独立于 mem0 现有代码，通过 Pipeline + Factory 模式插入；所有 ACE 组件有独立故障边界，任何故障不影响宿主。架构分为 6 层：Integration Layer（集成层）→ Pipeline Layer（管线层）→ Engine Layer（引擎层）→ Storage Adapter Layer（存储适配层）→ mem0 Infrastructure Layer（基础设施层）→ Configuration Layer（配置层）。

---

## Architectural Drivers

These requirements heavily influence architectural decisions:

1. **NFR-001: 检索延迟 < 50ms** → 要求混合检索管线高度优化，内存缓存，异步 I/O
2. **NFR-003: 隐私脱敏不可关闭** → 要求 Reflector 管线中 Sanitizer 是硬编码安全网，无法通过配置绕过
3. **NFR-005: mem0 API 完全兼容** → 要求所有改造通过"可选增强"模式叠加，不修改 mem0 原有方法签名
4. **NFR-006: 优雅降级** → 要求每个 ACE 组件有独立 try-catch 边界，故障时降级而非崩溃
5. **NFR-010: 零配置启动** → 要求所有配置有合理默认值，`Memory()` 无参构造即可使用

---

## System Overview

### High-Level Architecture

MemX 采用 **Layered Engine Architecture**（分层引擎架构），在 mem0 基础设施之上叠加 ACE 智能层。

核心设计决策：
- **新增代码独立目录**：`memx/` 顶层包，不修改 `mem0/` 内部文件（仅扩展 config）
- **Pipeline 模式**：add/search 操作通过可组合的处理管线，每个 Stage 可独立启停
- **Factory + Strategy 模式**：引擎组件通过 Factory 创建，可按配置切换实现
- **Decorator 模式**：MemX 的 `Memory` 类包装 mem0 的 `Memory` 类，ACE 关闭时直接代理

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
│    ▼                    │  │    ├──→ L1: ExactMatcher      │
│  [Reflector]            │  │    ├──→ L2: FuzzyMatcher      │
│  ├─ Stage1: Detector    │  │    ├──→ L3: MetadataMatcher   │
│  ├─ Stage2: Scorer      │  │    └──→ L4: VectorSearcher   │
│  ├─ Stage3: Sanitizer   │  │    │                          │
│  └─ Stage4: Distiller   │  │    ▼                          │
│    │                    │  │  ScoreMerger                   │
│    ▼                    │  │  (keyword×0.6 + semantic×0.4)  │
│  [Curator]              │  │    │                          │
│  ├─ SimilarityCheck     │  │    ▼                          │
│  └─ MergeOrInsert       │  │  DecayWeighter                │
│    │                    │  │    │                          │
│    ▼                    │  │    ▼                          │
│  mem0.add()             │  │  TokenBudgetTrimmer            │
│  (with Bullet metadata) │  │    │                          │
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
│                    ENGINE LAYER                              │
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
└─────────────────────────────────────────────────────────────┘
```

### Architectural Pattern

**Pattern:** Layered Engine Architecture + Decorator + Pipeline + Factory

**Rationale:**

| 可选模式 | 优点 | 缺点 | 结论 |
|----------|------|------|------|
| 直接修改 mem0 源码 | 实现最快 | 上游同步噩梦，破坏 API 兼容 | 排除 |
| 微服务拆分 | 独立部署 | 过度工程化，单用户场景无意义 | 排除 |
| Middleware/Wrapper | 不侵入 mem0 | 无法深入改造检索管线 | 部分采用 |
| **Layered Engine + Decorator** | 独立模块 + 透明代理 + 深度集成 | 需要精心设计层间接口 | **采用** |

关键选择理由：
1. **Decorator 模式**用于 Memory 类包装 → 保证 mem0 API 兼容（NFR-005）
2. **Pipeline 模式**用于 add/search 管线 → 每个 Stage 可独立 try-catch（NFR-006）
3. **Factory 模式**用于引擎创建 → 按配置切换实现（NFR-010 零配置）
4. **独立目录**用于代码组织 → 最小化上游冲突（NFR-008）

---

## Technology Stack

### Frontend

**Choice:** N/A（MemX 是 Python 库，无独立前端）

保留 mem0 的 OpenMemory UI（Next.js）作为可选管理界面，不做修改。

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

**Trade-offs:**
- 得到：与 mem0 生态完全兼容，用户零学习成本
- 失去：Rust/Go 可能有更好的性能（但 Python 在 5000 条规模够用）

### Database

**Choice:** SQLite (与 mem0 一致) + 向量存储由 mem0 Provider 负责

| 层 | 技术 | 用途 |
|----|------|------|
| 向量存储 | mem0 VectorStore (23 providers) | 记忆向量索引和检索 |
| 历史审计 | SQLiteManager (mem0 内置) | 变更历史追踪 |
| Bullet 元数据 | 嵌入向量存储 payload metadata | ACE 结构化字段 |
| Daemon 状态 | SQLite (独立数据库) | Session 注册、健康状态 |

**Trade-offs:**
- 得到：零外部依赖，Local-First，单文件部署
- 失去：> 50,000 条时需要引入 sqlite-vec 或 LanceDB

### Infrastructure

**Choice:** Local-First (无云端依赖)

| 组件 | 方案 | 说明 |
|------|------|------|
| 运行环境 | 用户本地 Python 环境 | pip install 即用 |
| 数据存储 | `~/.memx/` 目录 | SQLite + ONNX 模型 |
| Daemon | 本地进程 | PID 文件管理 |
| 企业部署 | Docker (可选) | 企业版提供 Dockerfile |

### Third-Party Services

| 服务 | 用途 | 必需 | 说明 |
|------|------|------|------|
| PyPI | 包发布 | 是 | 发布渠道 |
| GitHub | 代码托管 | 是 | 开源社区 |
| LLM API (OpenAI等) | LLM 增强蒸馏 | 否 | 仅 LLM 模式需要 |
| Embedding API | 在线 Embedding | 否 | 本地 ONNX 可替代 |

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

### Component 1: MemXMemory (公开 API 层)

**Purpose:** MemX 的公开入口，包装 mem0 的 Memory 类

**Responsibilities:**
- 提供与 mem0 完全兼容的 API（add/search/get_all/get/update/delete/history/reset）
- ACE 功能开关（ace_enabled=True 时激活管线处理）
- ACE 关闭时直接代理到 mem0.Memory（零开销透传）
- 新增 API：status(), export(), import_data(), run_decay_sweep()

**Interfaces:**
```python
class Memory:  # 替代 mem0.Memory
    def __init__(self, config: dict = None):  # 兼容 mem0 签名
    def add(self, messages, user_id=None, agent_id=None, run_id=None,
            metadata=None, filters=None, prompt=None,
            scope=None, **kwargs) -> dict:  # scope 为新增可选参数
    def search(self, query, user_id=None, agent_id=None, run_id=None,
               limit=100, filters=None,
               scope=None, **kwargs) -> dict:
    # ... 其余方法签名与 mem0 完全一致

    # === ACE 新增方法 ===
    def status(self) -> PlaybookStats:
    def export(self, format="json", scope=None) -> str:
    def import_data(self, data, format="json") -> ImportResult:
    def run_decay_sweep(self) -> DecaySweepResult:
```

**Dependencies:** IngestPipeline, RetrievalPipeline, mem0.Memory, MemXConfig

**FRs Addressed:** FR-013 (API 兼容), FR-012 (配置系统)

---

### Component 2: IngestPipeline (入库管线)

**Purpose:** 处理 add() 操作的完整管线：Raw Input → Reflect → Curate → Store

**Responsibilities:**
- 编排 Reflector 四个 Stage 的顺序执行
- 编排 Curator 的去重/合并决策
- 每个 Stage 独立 try-catch，异常时跳过该 Stage（NFR-006）
- 将处理后的 Bullet 传递给 mem0 的底层 add() 写入存储

**Interfaces:**
```python
class IngestPipeline:
    def __init__(self, reflector: ReflectorEngine, curator: CuratorEngine,
                 config: MemXConfig):
    async def process(self, messages: list, metadata: dict) -> IngestResult:
    # IngestResult: {bullets_added, bullets_merged, bullets_skipped, errors}
```

**Dependencies:** ReflectorEngine, CuratorEngine, BulletFactory, PrivacySanitizer

**FRs Addressed:** FR-002~FR-006

---

### Component 3: RetrievalPipeline (检索管线)

**Purpose:** 处理 search() 操作的完整管线：Query → Multi-layer Match → Score → Trim → Return

**Responsibilities:**
- 编排 L1-L4 四层检索并合并评分
- 应用 DecayWeight 和 RecencyBoost 加权
- Token 预算裁剪
- 可选 Reranker 精排
- 异步更新被召回记忆的 recall_count（RecallReinforcer）

**Interfaces:**
```python
class RetrievalPipeline:
    def __init__(self, generator: GeneratorEngine, decay: DecayEngine,
                 reranker=None, config: MemXConfig):
    async def search(self, query: str, filters: dict, limit: int) -> SearchResult:
    # SearchResult: {results: list[ScoredBullet], mode: "full"|"degraded"}
```

**Dependencies:** GeneratorEngine, DecayEngine, Reranker (optional), mem0 VectorStore

**FRs Addressed:** FR-009~FR-011

---

### Component 4: ReflectorEngine (知识蒸馏引擎)

**Purpose:** ACE 四大引擎之一 — 负责"学到了"

**Responsibilities:**
- Stage 1 — PatternDetector：规则式交互模式检测
- Stage 2 — KnowledgeScorer：分类（knowledge_type）+ 评分（instructivity_score）
- Stage 3 — PrivacySanitizer：隐私脱敏（hardcoded safety net，不可关闭）
- Stage 4 — BulletDistiller：蒸馏为标准 Bullet 格式
- 支持三种运行模式：rules / llm / hybrid

**Interfaces:**
```python
class ReflectorEngine:
    def __init__(self, config: ReflectorConfig, sanitizer: PrivacySanitizer,
                 llm: LLMBase = None):
    def reflect(self, interaction: InteractionEvent) -> list[CandidateBullet]:

class PatternDetector:
    def detect(self, event: InteractionEvent) -> list[DetectedPattern]:

class KnowledgeScorer:
    def score(self, pattern: DetectedPattern) -> ScoredCandidate:

class PrivacySanitizer:  # 独立组件，非配置可关闭
    def sanitize(self, content: str) -> SanitizeResult:

class BulletDistiller:
    def distill(self, candidate: ScoredCandidate) -> CandidateBullet:
```

**Dependencies:** PrivacySanitizer, BulletFactory, LLMBase (optional)

**FRs Addressed:** FR-002, FR-003, FR-004, FR-005, FR-019

---

### Component 5: CuratorEngine (语义去重引擎)

**Purpose:** ACE 四大引擎之二 — 负责"不重复"

**Responsibilities:**
- 接收 Reflector 产出的候选 Bullet
- 与现有记忆计算 cosine similarity
- similarity ≥ 阈值 → Merge（合并内容和元数据）
- similarity < 阈值 → Insert（直接入库）
- 可选冲突检测（similarity 0.5-0.8 且内容矛盾）

**Interfaces:**
```python
class CuratorEngine:
    def __init__(self, config: CuratorConfig, embedder: EmbeddingBase,
                 vector_store: VectorStoreBase):
    def curate(self, candidates: list[CandidateBullet],
               existing: list[Memory]) -> CurateResult:
    # CurateResult: {to_add: list, to_merge: list[MergePair], to_skip: list}

    def detect_conflicts(self) -> list[Conflict]:  # FR-020
```

**Dependencies:** EmbeddingBase, VectorStoreBase

**FRs Addressed:** FR-006, FR-020

---

### Component 6: DecayEngine (衰退引擎)

**Purpose:** ACE 四大引擎之三 — 负责"该忘了"

**Responsibilities:**
- 计算单条记忆的 decay_weight（艾宾浩斯公式）
- 批量衰退扫描（run_decay_sweep）
- 召回强化（search 返回时更新 recall_count）
- 永久保留判定（recall_count ≥ 阈值）
- 归档判定（weight < 阈值）

**Interfaces:**
```python
class DecayEngine:
    def __init__(self, config: DecayConfig):

    def compute_weight(self, bullet_meta: BulletMetadata) -> float:
    def reinforce(self, bullet_ids: list[str]) -> None:  # 异步
    def sweep(self) -> DecaySweepResult:
    # DecaySweepResult: {updated: int, archived: int, permanent: int}
```

**Dependencies:** BulletMetadata, VectorStoreBase (for batch update)

**FRs Addressed:** FR-007, FR-008

---

### Component 7: GeneratorEngine (混合检索引擎)

**Purpose:** ACE 四大引擎之四 — 负责"想起来"

**Responsibilities:**
- L1 ExactMatcher：精确关键词全词匹配
- L2 FuzzyMatcher：中文 2-gram + 英文词干化模糊匹配
- L3 MetadataMatcher：related_tools / key_entities / tags 前缀匹配
- L4 VectorSearcher：调用 mem0 向量检索
- ScoreMerger：综合评分 = keyword×weight + semantic×weight
- 降级模式：L4 不可用时跳过语义，仅使用 L1-L3

**Interfaces:**
```python
class GeneratorEngine:
    def __init__(self, config: RetrievalConfig, embedder: EmbeddingBase = None,
                 vector_store: VectorStoreBase = None):

    def search(self, query: str, filters: dict, limit: int) -> list[ScoredBullet]:

    @property
    def mode(self) -> str:  # "full" | "degraded"

class ExactMatcher:
    def match(self, query: str, bullets: list) -> list[MatchScore]:

class FuzzyMatcher:
    def match(self, query: str, bullets: list) -> list[MatchScore]:

class MetadataMatcher:
    def match(self, query: str, bullets: list) -> list[MatchScore]:

class VectorSearcher:
    def search(self, query_embedding: list[float], limit: int) -> list[MatchScore]:

class ScoreMerger:
    def merge(self, keyword_scores: dict, semantic_scores: dict,
              decay_weights: dict, config: RetrievalConfig) -> list[ScoredBullet]:
```

**Dependencies:** mem0 VectorStore, mem0 Embedder, RetrievalConfig

**FRs Addressed:** FR-009, FR-010, FR-011

---

### Component 8: IntegrationManager (集成管理器)

**Purpose:** 管理 ACE 与宿主 AI 产品的三个集成点

**Responsibilities:**
- PreInferenceHook：用户输入后自动召回记忆
- PostActionHook：工具调用后自动蒸馏
- SessionEndHook：会话结束时兜底蒸馏 + Decay sweep
- 集成点注册/注销
- 为不同宿主提供具体实现（CLI Hook / MCP Server / API Middleware）

**Interfaces:**
```python
class IntegrationManager:
    def __init__(self, memory: Memory, config: IntegrationConfig):
    def register_hooks(self, hooks: list[BaseHook]) -> None:
    def unregister_all(self) -> None:

class PreInferenceHook(BaseHook):
    async def on_user_input(self, input: str) -> ContextInjection:

class PostActionHook(BaseHook):
    async def on_tool_result(self, event: ToolEvent) -> None:

class SessionEndHook(BaseHook):
    async def on_session_end(self, session_id: str) -> None:

# Concrete implementations
class CLIPreInferenceHook(PreInferenceHook): ...
class CLIPostActionHook(PostActionHook): ...
class CLISessionEndHook(SessionEndHook): ...
```

**Dependencies:** Memory, ReflectorEngine, DecayEngine

**FRs Addressed:** FR-014, FR-015, FR-016

---

### Component 9: ONNXEmbedder (本地 Embedding)

**Purpose:** 提供零网络依赖的本地 Embedding 能力

**Responsibilities:**
- 加载 ONNX 模型（all-MiniLM-L6-v2，384 维）
- 文本编码（tokenize + inference）
- 模型文件管理（自动下载到 `~/.memx/models/`）
- 注册到 mem0 的 EmbedderFactory

**Interfaces:**
```python
class ONNXEmbedder(EmbeddingBase):
    def __init__(self, config: ONNXEmbedderConfig):
    def embed(self, text: str, memory_action: str = None) -> list[float]:

class ONNXEmbedderConfig(BaseEmbedderConfig):
    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    model_path: str = "~/.memx/models/"
    auto_download: bool = True
```

**Dependencies:** onnxruntime, tokenizers

**FRs Addressed:** FR-017

---

### Component 10: MemXDaemon (常驻进程)

**Purpose:** 避免每次 Hook 调用冷启动 ONNX 模型

**Responsibilities:**
- ONNX 模型预加载和常驻
- IPC 服务（ping / recall / curate / session_register / session_unregister / shutdown）
- 多 Session 管理
- 空闲超时自动退出
- PID 文件管理 + 健康检查

**Interfaces:**
```python
class MemXDaemon:
    def __init__(self, config: DaemonConfig):
    async def start(self) -> None:
    async def stop(self) -> None:
    async def handle_request(self, request: DaemonRequest) -> DaemonResponse:

class DaemonClient:  # Hook 端使用
    def __init__(self, config: DaemonConfig):
    async def ping(self) -> bool:
    async def recall(self, query: str, **kwargs) -> list[dict]:
    async def curate(self, bullets: list[dict]) -> dict:
    async def register_session(self, session_id: str) -> None:
    async def unregister_session(self, session_id: str) -> None:
```

**Dependencies:** ONNXEmbedder, Memory, IPC transport

**FRs Addressed:** FR-018

---

### Component 11: BulletFactory + BulletMetadata (数据模型)

**Purpose:** 定义和管理 Bullet 结构化知识单元

**Responsibilities:**
- Bullet Pydantic 模型定义和校验
- 创建新 Bullet（自动填充默认元数据）
- 从 mem0 metadata 解析 Bullet 字段
- 向 mem0 metadata 注入 Bullet 字段
- 向后兼容处理（旧记忆无 Bullet 字段时使用默认值）

**Interfaces:**
```python
class BulletMetadata(BaseModel):
    section: BulletSection = "general"
    knowledge_type: KnowledgeType = "Knowledge"
    instructivity_score: int = 50
    recall_count: int = 0
    last_recall: Optional[datetime] = None
    decay_weight: float = 1.0
    related_tools: list[str] = []
    related_files: list[str] = []
    key_entities: list[str] = []
    tags: list[str] = []
    distilled_rule: Optional[str] = None
    source_type: SourceType = "auto_detected"
    scope: str = "global"

class BulletFactory:
    @staticmethod
    def create(content: str, **kwargs) -> BulletMetadata:
    @staticmethod
    def from_mem0_payload(payload: dict) -> BulletMetadata:
    @staticmethod
    def to_mem0_metadata(bullet: BulletMetadata) -> dict:
```

**Dependencies:** Pydantic

**FRs Addressed:** FR-001

---

### Component 12: MemXConfig (配置系统)

**Purpose:** 扩展 mem0 的配置系统

**Responsibilities:**
- 定义所有 ACE 配置子模型
- ace_enabled 总开关
- 合理默认值（零配置启动）
- 配置校验和错误提示

**Interfaces:**
```python
class MemXConfig(MemoryConfig):  # 继承 mem0
    ace_enabled: bool = False
    retrieval: RetrievalConfig = RetrievalConfig()
    reflector: ReflectorConfig = ReflectorConfig()
    curator: CuratorConfig = CuratorConfig()
    decay: DecayConfig = DecayConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    integration: IntegrationConfig = IntegrationConfig()
    daemon: DaemonConfig = DaemonConfig()

class RetrievalConfig(BaseModel):
    keyword_weight: float = 0.6
    semantic_weight: float = 0.4
    max_results: int = 5
    token_budget: int = 2000
    recency_boost_days: int = 7
    recency_boost_factor: float = 1.2

class ReflectorConfig(BaseModel):
    mode: Literal["rules", "llm", "hybrid"] = "rules"
    min_instructivity_score: int = 30
    max_content_length: int = 500
    max_code_lines: int = 3
    code_ratio_reject_threshold: float = 0.6

class CuratorConfig(BaseModel):
    dedup_similarity_threshold: float = 0.8
    merge_strategy: Literal["keep_best", "merge_content"] = "keep_best"
    conflict_detection: bool = False

class DecayConfig(BaseModel):
    half_life_days: int = 30
    protection_days: int = 7
    recall_boost_factor: float = 0.3
    permanent_recall_threshold: int = 15
    archive_weight_threshold: float = 0.02
    auto_reinforce: bool = True

class PrivacyConfig(BaseModel):
    filter_api_keys: bool = True
    filter_passwords: bool = True
    filter_user_paths: bool = True
    custom_patterns: list[str] = []
    # NOTE: 内置过滤器不可通过配置关闭

class IntegrationConfig(BaseModel):
    pre_inference: bool = False
    post_action: bool = False
    session_end: bool = False
    context_template: str = "xml"  # xml | markdown | plain

class DaemonConfig(BaseModel):
    enabled: bool = False
    idle_timeout_minutes: int = 5
    max_idle_minutes: int = 10
    health_check_interval_seconds: int = 60
    ipc: Literal["auto", "named_pipe", "unix_socket", "tcp"] = "auto"
```

**Dependencies:** Pydantic, mem0 MemoryConfig

**FRs Addressed:** FR-012

---

## Data Architecture

### Data Model

```
┌────────────────────────────────────────┐
│           Bullet (知识单元)              │
│                                        │
│ ┌─ Identity ─────────────────────────┐ │
│ │ id: UUID                           │ │
│ │ scope: "global" | "project:{name}" │ │
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
│ ┌─ Vector ───────────────────────────┐ │
│ │ embedding: float[384]              │ │
│ └────────────────────────────────────┘ │
│ ┌─ mem0 原有字段 ────────────────────┐ │
│ │ user_id: str?                      │ │
│ │ agent_id: str?                     │ │
│ │ run_id: str?                       │ │
│ │ hash: str (mem0 内部)              │ │
│ └────────────────────────────────────┘ │
└────────────────────────────────────────┘

枚举定义：
  BulletSection: coding | debugging | architecture | tooling
                 | preferences | domain | workflow | general
  KnowledgeType: Method | Trick | Pitfall | Preference | Knowledge
  SourceType:    auto_detected | user_manual | llm_evaluated
```

### Database Design

**策略：** Bullet 元数据嵌入 mem0 的 VectorStore payload，不新建独立表。

```
mem0 VectorStore payload 结构（改造后）：

{
  // === mem0 原有字段 ===
  "data": "When using async/await in Rust, always pin the future...",
  "hash": "abc123...",
  "user_id": "alice",
  "agent_id": null,
  "run_id": null,
  "created_at": "2026-03-01T10:00:00Z",
  "updated_at": "2026-03-01T10:00:00Z",

  // === ACE Bullet 元数据（新增） ===
  "memx_section": "coding",
  "memx_knowledge_type": "Pitfall",
  "memx_source_type": "auto_detected",
  "memx_instructivity_score": 75,
  "memx_recall_count": 3,
  "memx_last_recall": "2026-03-15T14:30:00Z",
  "memx_decay_weight": 0.87,
  "memx_scope": "project:my-rust-app",
  "memx_related_tools": ["cargo", "rustc"],
  "memx_key_entities": ["async", "await", "Future", "Pin"],
  "memx_tags": ["rust", "async"],
  "memx_distilled_rule": "When using async/await in Rust, pin the future before polling because unpinned futures can move in memory."
}
```

**关键设计决策：**
- 所有 ACE 字段使用 `memx_` 前缀，避免与 mem0 原有字段冲突
- 旧记忆（无 `memx_` 前缀）在读取时自动填充默认值
- 向量存储 payload 是 JSON，嵌入新字段无需 schema 迁移

**Daemon 状态数据库（独立 SQLite）：**

```sql
-- ~/.memx/daemon.db
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  registered_at TEXT NOT NULL,
  last_active TEXT NOT NULL,
  metadata TEXT  -- JSON
);

CREATE TABLE daemon_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
```

### Data Flow

**Ingest Flow (add)：**
```
messages → MemXMemory.add()
  │
  ├── ace_enabled=False → mem0.Memory.add() [直接代理]
  │
  └── ace_enabled=True → IngestPipeline.process()
        │
        ├── Reflector.reflect(messages)
        │   ├── Stage1: PatternDetector.detect() → patterns[]
        │   ├── Stage2: KnowledgeScorer.score() → scored_candidates[]
        │   ├── Stage3: PrivacySanitizer.sanitize() → clean_candidates[]
        │   └── Stage4: BulletDistiller.distill() → candidate_bullets[]
        │
        ├── Curator.curate(candidate_bullets, existing_memories)
        │   ├── similarity ≥ 0.8 → merge_pairs[]
        │   └── similarity < 0.8 → new_bullets[]
        │
        ├── For each new_bullet:
        │   └── BulletFactory.to_mem0_metadata() → metadata dict
        │       └── mem0.Memory._add_to_vector_store(content, metadata)
        │
        └── For each merge_pair:
            └── mem0.Memory.update(id, merged_content, merged_metadata)
```

**Retrieval Flow (search)：**
```
query → MemXMemory.search()
  │
  ├── ace_enabled=False → mem0.Memory.search() [直接代理]
  │
  └── ace_enabled=True → RetrievalPipeline.search()
        │
        ├── GeneratorEngine.search()
        │   ├── L1: ExactMatcher.match(query, bullets)     → exact_scores
        │   ├── L2: FuzzyMatcher.match(query, bullets)     → fuzzy_scores
        │   ├── L3: MetadataMatcher.match(query, bullets)  → meta_scores
        │   ├── L4: VectorSearcher.search(embedding)       → vector_scores
        │   │       (if embedding fails → skip L4, mode=degraded)
        │   │
        │   └── ScoreMerger.merge(keyword_scores, vector_scores)
        │       FinalScore = (Keyword×0.6 + Semantic×0.4) × DecayWeight × RecencyBoost
        │
        ├── TokenBudgetTrimmer.trim(scored_results, budget=2000, max=5)
        │
        ├── Reranker.rerank(query, trimmed_results) [optional]
        │
        ├── RecallReinforcer.reinforce(result_ids)  [async, non-blocking]
        │   └── DecayEngine.reinforce() → update recall_count + decay_weight
        │
        └── Return formatted results
```

---

## API Design

### API Architecture

**类型:** Python Library API (非 REST)

MemX 的核心交互方式是 Python 库 API，不是独立的 HTTP 服务。REST Server 由 mem0 的 `server/main.py` 提供（保留不改）。

**设计原则：**
- 公开 API 与 mem0 签名完全一致
- ACE 新增能力通过可选参数和新方法暴露
- 返回值格式与 mem0 兼容，ACE 元数据作为额外字段

### Endpoints (Python API)

```python
# === 与 mem0 兼容的核心 API ===

# 记忆入库 (ACE 模式下经过 Reflector + Curator)
m.add(messages, user_id="alice")
m.add(messages, user_id="alice", scope="project:myapp")  # ACE 新增: scope

# 记忆检索 (ACE 模式下使用混合检索 + 衰退加权)
m.search(query, user_id="alice")
m.search(query, user_id="alice", scope="project:myapp")  # ACE 新增: scope

# 记忆管理 (直接代理 mem0)
m.get_all(user_id="alice")
m.get(memory_id)
m.update(memory_id, data="updated content")
m.delete(memory_id)
m.delete_all(user_id="alice")
m.history(memory_id)
m.reset()

# === ACE 新增 API ===

# 知识库状态
m.status()
# → PlaybookStats {total, by_section, by_type, avg_decay, permanent_count, archived_count}

# 衰退扫描
m.run_decay_sweep()
# → DecaySweepResult {updated: 42, archived: 3, permanent: 5}

# 导入/导出
m.export(format="json", scope="global")
# → JSON string
m.import_data(data, format="json")
# → ImportResult {added: 10, merged: 3, skipped: 2}

# === CLI API ===

memx status                          # 知识库统计
memx search "async await error"      # 混合检索
memx learn "When using X, do Y"      # 手动记录
memx list --scope project:myapp      # 列出记忆
memx forget <memory-id>              # 删除记忆
memx sweep                           # 手动衰退扫描
memx export --format json > backup   # 导出
memx import < backup.json            # 导入

# === Daemon IPC Protocol ===

REQUEST:  {"cmd": "ping"}
RESPONSE: {"status": "ok", "version": "1.0.0", "sessions": 2}

REQUEST:  {"cmd": "recall", "query": "...", "user_id": "...", "limit": 5}
RESPONSE: {"results": [...], "mode": "full"}

REQUEST:  {"cmd": "curate", "bullets": [...]}
RESPONSE: {"added": 2, "merged": 1, "skipped": 0}

REQUEST:  {"cmd": "session_register", "session_id": "abc"}
RESPONSE: {"status": "ok"}

REQUEST:  {"cmd": "session_unregister", "session_id": "abc"}
RESPONSE: {"status": "ok", "active_sessions": 1}

REQUEST:  {"cmd": "shutdown"}
RESPONSE: {"status": "shutting_down"}
```

### Authentication & Authorization

**本地库模式（默认）：** 无需身份认证。数据存储在用户本地文件系统，由 OS 文件权限保护。

**Daemon IPC 模式：** Named Pipe / Unix Socket 由 OS 进程权限控制访问。不开放 TCP 端口（避免网络攻击面）。TCP 模式仅用于开发调试，默认关闭。

**mem0 Cloud Client：** 保留 mem0 原有的 API Key 认证（`Authorization: Token <api_key>`）。

---

## Non-Functional Requirements Coverage

### NFR-001: Performance — 检索延迟 < 50ms

**Requirement:** 5000 条记忆规模下端到端检索 < 50ms

**Architecture Solution:**
- L1-L3 关键词/元数据检索使用内存索引（Python dict/set），< 5ms
- L4 向量检索由 mem0 VectorStore 负责，Qdrant 本地模式 < 20ms
- ScoreMerger 纯内存计算，< 1ms
- RecallReinforcer 异步执行，不计入检索延迟
- Daemon 模式预加载 ONNX 模型，避免冷启动

**Implementation Notes:**
- 关键词索引在 Memory 初始化时一次性构建，后续 add 增量更新
- 内存中维护 `keyword_index: dict[str, set[str]]` 和 `metadata_index: dict[str, set[str]]`
- 5000 条内存占用估算：~50MB（含向量缓存）

**Validation:**
- pytest-benchmark 自动化性能测试
- CI 中设置 50ms 阈值门禁

---

### NFR-002: Performance — 蒸馏延迟 < 20ms

**Requirement:** Rules-only 模式单次蒸馏 < 20ms

**Architecture Solution:**
- 四个 Stage 全部为纯 Python 规则（无 I/O、无 LLM）
- 正则预编译（PrivacySanitizer 的所有 pattern 在初始化时编译）
- 蒸馏操作在后台线程执行（ThreadPoolExecutor）

**Validation:**
- 单次蒸馏基准测试 + CI 门禁

---

### NFR-003: Security — 隐私脱敏不可关闭

**Requirement:** 所有入库内容必须经过隐私脱敏

**Architecture Solution:**
- PrivacySanitizer 作为独立组件，在 Reflector Stage 3 **硬编码调用**
- 即使用户配置 `reflector.mode: "off"`，Sanitizer 仍在 IngestPipeline 中作为独立 Stage 执行
- 内置 pattern 列表为 hardcoded constant，不可通过配置删除
- 用户只能通过 `privacy.custom_patterns` **添加** 额外规则

**Implementation Notes:**
```python
# IngestPipeline.process() 中
# Sanitizer 独立于 Reflector，无法被绕过
sanitized = self.sanitizer.sanitize(content)  # ALWAYS runs
if self.config.ace_enabled:
    reflected = self.reflector.reflect(sanitized)  # Only if ACE on
```

---

### NFR-004: Security — 数据本地化

**Requirement:** 核心记忆数据存储在用户本地

**Architecture Solution:**
- 默认存储路径 `~/.memx/`，可配置
- Rules-only + ONNX 模式下零网络请求
- LLM 模式需用户显式配置 API Key 并确认数据发送
- 代码审计清单：确保无隐式 HTTP 调用

---

### NFR-005: Compatibility — mem0 API 兼容

**Requirement:** 公开 API 与 mem0 v1.0.x 完全兼容

**Architecture Solution:**
- MemXMemory 类使用 **Decorator 模式** 包装 mem0.Memory
- `ace_enabled=False` 时，所有方法调用直接代理到内部 `self._mem0_memory`
- 方法签名不变，ACE 参数通过 `**kwargs` 传入
- 运行 mem0 官方测试套件作为回归测试

**Implementation Notes:**
```python
class Memory:
    def __init__(self, config=None):
        self._config = MemXConfig(**(config or {}))
        self._mem0 = Mem0Memory(config=self._config.to_mem0_config())
        if self._config.ace_enabled:
            self._ingest = IngestPipeline(...)
            self._retrieval = RetrievalPipeline(...)

    def add(self, messages, **kwargs):
        if not self._config.ace_enabled:
            return self._mem0.add(messages, **kwargs)
        return self._ingest.process(messages, **kwargs)

    def search(self, query, **kwargs):
        if not self._config.ace_enabled:
            return self._mem0.search(query, **kwargs)
        return self._retrieval.search(query, **kwargs)
```

---

### NFR-006: Reliability — 优雅降级

**Requirement:** 任何 ACE 组件故障不影响宿主

**Architecture Solution:**
- **Pipeline Stage 级别的 try-catch**：每个 Stage 独立捕获异常
- **组件级降级链**：
  - Reflector 异常 → 跳过蒸馏，raw add 到 mem0
  - Curator 异常 → 跳过去重，直接 Insert
  - Generator L4 异常 → 降级为纯关键词（L1-L3）
  - Decay 异常 → 跳过衰退更新，下次重试
  - Daemon 异常 → 回退到直接读 SQLite
- **故障不传播**：ACE 异常绝不抛出到调用方

**Implementation Notes:**
```python
class IngestPipeline:
    async def process(self, messages, **kwargs):
        try:
            candidates = self.reflector.reflect(messages)
        except Exception as e:
            logger.warning(f"Reflector failed, skipping: {e}")
            candidates = self._fallback_raw_extract(messages)

        try:
            curated = self.curator.curate(candidates)
        except Exception as e:
            logger.warning(f"Curator failed, inserting directly: {e}")
            curated = CurateResult(to_add=candidates, to_merge=[], to_skip=[])

        # ... continue with mem0 add
```

---

### NFR-007: Maintainability — 测试覆盖 > 80%

**Architecture Solution:**
- 每个 ACE Engine 有独立的 `tests/` 目录
- 使用 dependency injection 便于 mock
- pytest-cov 集成到 CI

---

### NFR-008: Maintainability — 上游同步

**Architecture Solution:**
- MemX 新增代码全部在 `memx/` 顶层包
- 仅修改 mem0 的一个文件：`mem0/configs/base.py`（扩展 MemoryConfig）
- 其余通过 monkey-patch 或 wrapper 实现
- 月度 rebase 流程文档化

---

### NFR-009: Scalability — 5,000-50,000 条

**Architecture Solution:**
- < 5,000：内存 brute-force（L1-L3 全量扫描 + L4 向量 brute-force）
- 5,000-50,000：关键词索引升级为 inverted index + L4 使用 sqlite-vec
- > 50,000：推荐切换到 LanceDB 或远程 Qdrant

---

### NFR-010: Usability — 零配置启动

**Architecture Solution:**
- `Memory()` 无参构造：ace_enabled=False，行为与 mem0 完全一致
- `Memory(config={"ace_enabled": True})`：启用 ACE，所有引擎使用默认参数
- 无需预装模型、无需 API Key、无需外部服务
- 首次使用 ONNX 时自动下载模型

---

## Security Architecture

### Authentication

**本地库模式：** 无独立认证层。依赖操作系统文件权限。

**Daemon IPC：** Named Pipe / Unix Socket 基于进程 UID 权限控制。不暴露 TCP 端口。

### Authorization

**单用户 Local-First 模型：** 当前版本无多用户权限模型。所有操作对本地用户完全开放。

**未来扩展：** Enterprise 版本可增加 API Key + RBAC。

### Data Encryption

| 数据状态 | 方案 | 说明 |
|----------|------|------|
| At Rest | OS 级加密（BitLocker/FileVault） | 依赖用户 OS 配置 |
| In Transit (IPC) | N/A（本地进程间通信） | Named Pipe 不经过网络 |
| In Transit (LLM API) | TLS 1.2+ | 由 LLM Provider SDK 保证 |

### Security Best Practices

- **隐私脱敏（hardcoded）**：API Key/Token/密码/路径自动过滤
- **输入校验**：content 长度限制（500 字符），代码占比限制（60%）
- **无远程代码执行**：不执行用户提供的代码片段
- **依赖安全**：使用 `pip-audit` 检查依赖漏洞
- **最小权限**：ONNX 模型仅做 inference，不训练

---

## Scalability & Performance

### Scaling Strategy

MemX 是本地库，不涉及水平扩展。性能优化聚焦于：

| 维度 | 策略 |
|------|------|
| 计算 | 关键词索引内存化；正则预编译；异步 I/O |
| 存储 | 分级方案：brute-force → sqlite-vec → LanceDB |
| 内存 | 向量缓存懒加载；LRU 淘汰；ONNX 模型共享 |

### Performance Optimization

| 优化项 | 技术 | 效果 |
|--------|------|------|
| 关键词索引 | inverted index (dict[str, set[str]]) | L1-L3 查询 < 5ms |
| 正则预编译 | `re.compile()` 在 __init__ 时完成 | Sanitizer 提速 10x |
| 向量缓存 | 内存中维护 embedding 向量 | 避免重复 embed |
| 异步强化 | recall_count 更新在后台线程 | search 无阻塞 |
| ONNX batch | 支持 batch embedding | 批量入库提速 |

### Caching Strategy

| 缓存层 | 内容 | TTL | 失效策略 |
|--------|------|-----|----------|
| 关键词索引缓存 | keyword → bullet_ids 映射 | 永久（增量更新） | add/delete 时增量更新 |
| 元数据索引缓存 | tool/entity → bullet_ids | 永久（增量更新） | 同上 |
| 向量缓存 | bullet_id → embedding | 永久（懒加载） | delete 时移除 |
| Decay 缓存 | bullet_id → decay_weight | 5 min | sweep 后全量刷新 |

### Load Balancing

N/A（本地库，无分布式负载均衡需求）

---

## Reliability & Availability

### High Availability Design

MemX 是本地库，不涉及传统 HA。可靠性通过以下方式保证：

- **组件隔离**：每个 ACE 引擎有独立故障边界
- **降级链**：任何组件故障 → 降级到更简单的模式
- **数据持久化**：SQLite WAL 模式 + 定期 checkpoint

### Disaster Recovery

| 场景 | 恢复方案 |
|------|----------|
| 数据库损坏 | SQLite WAL 自动恢复；最坏情况从 export 恢复 |
| ONNX 模型损坏 | 自动重新下载 |
| 配置损坏 | 回退到默认配置 |

### Backup Strategy

- `memx export --format json > backup.json` 手动备份
- 企业版可增加自动定时备份

### Monitoring & Alerting

| 指标 | 方式 |
|------|------|
| 检索延迟 | `memx status` 显示 P50/P95 |
| 知识库大小 | `memx status` 显示条目数 |
| 衰退状态 | `memx status` 显示平均 decay_weight |
| 组件健康 | Daemon health check endpoint |

---

## Integration Architecture

### External Integrations

| 集成目标 | 方式 | 说明 |
|----------|------|------|
| Claude Code | Hooks (UserPromptSubmit / PostToolUse / Stop / SessionEnd) | 最优先集成目标 |
| OpenClaw | CLI Middleware / Pre-prompt Hook | 通用 CLI 集成 |
| IDE Plugin (Cursor/VS Code) | Extension API (onWillSendRequest / onDidExecuteCommand) | Phase 3 |
| Web/Agent | API Middleware / on_tool_result callback | Phase 3 |
| MCP | ACE 作为 MCP Server 提供 tools (ace_recall / ace_learn / ace_status) | 通用协议 |

### Internal Integrations

| 组件 | 接口 | 说明 |
|------|------|------|
| MemXMemory ↔ mem0.Memory | Decorator pattern | API 层代理 |
| IngestPipeline ↔ ReflectorEngine | Direct call | 管线内编排 |
| RetrievalPipeline ↔ GeneratorEngine | Direct call | 管线内编排 |
| DecayEngine ↔ VectorStore | Batch update API | 衰退扫描 |
| DaemonClient ↔ MemXDaemon | IPC (Named Pipe / Unix Socket) | 进程间通信 |

### Message/Event Architecture

MemX 使用简单的同步/异步调用模式，不引入消息队列：

```
同步路径: add() → IngestPipeline → mem0.add()
同步路径: search() → RetrievalPipeline → results
异步路径: search() → results + async RecallReinforcer
异步路径: PostActionHook → async Reflector
```

---

## Development Architecture

### Code Organization

```
memx/                          # MemX 顶层包（新增）
├── __init__.py                # 公开 API: Memory, AsyncMemory
├── memory.py                  # MemXMemory 类（Decorator on mem0.Memory）
├── async_memory.py            # AsyncMemXMemory 类
├── config.py                  # MemXConfig + 所有子配置
├── types.py                   # BulletMetadata, enums, type aliases
├── exceptions.py              # MemX 特有异常
│
├── engines/                   # ACE 四大引擎
│   ├── __init__.py
│   ├── reflector/             # 知识蒸馏引擎
│   │   ├── __init__.py
│   │   ├── engine.py          # ReflectorEngine
│   │   ├── detector.py        # PatternDetector (Stage 1)
│   │   ├── scorer.py          # KnowledgeScorer (Stage 2)
│   │   ├── distiller.py       # BulletDistiller (Stage 4)
│   │   └── patterns.py        # 内置模式规则定义
│   ├── curator/               # 语义去重引擎
│   │   ├── __init__.py
│   │   ├── engine.py          # CuratorEngine
│   │   ├── merger.py          # MergeStrategy
│   │   └── conflict.py        # ConflictDetector (FR-020)
│   ├── decay/                 # 衰退引擎
│   │   ├── __init__.py
│   │   ├── engine.py          # DecayEngine
│   │   └── formulas.py        # 衰退公式实现
│   └── generator/             # 混合检索引擎
│       ├── __init__.py
│       ├── engine.py          # GeneratorEngine
│       ├── exact_matcher.py   # L1 精确匹配
│       ├── fuzzy_matcher.py   # L2 模糊匹配
│       ├── metadata_matcher.py# L3 元数据匹配
│       ├── vector_searcher.py # L4 向量检索
│       └── score_merger.py    # 综合评分
│
├── pipeline/                  # 处理管线
│   ├── __init__.py
│   ├── ingest.py              # IngestPipeline (add 路径)
│   └── retrieval.py           # RetrievalPipeline (search 路径)
│
├── privacy/                   # 隐私脱敏（独立模块，不可绕过）
│   ├── __init__.py
│   ├── sanitizer.py           # PrivacySanitizer
│   └── patterns.py            # 内置敏感信息正则
│
├── integration/               # 宿主集成
│   ├── __init__.py
│   ├── manager.py             # IntegrationManager
│   ├── hooks.py               # BaseHook, PreInference, PostAction, SessionEnd
│   └── cli_hooks.py           # CLI 具体实现
│
├── embeddings/                # ONNX Embedding Provider
│   ├── __init__.py
│   └── onnx.py               # ONNXEmbedder
│
├── daemon/                    # 常驻进程
│   ├── __init__.py
│   ├── server.py              # MemXDaemon
│   ├── client.py              # DaemonClient
│   └── ipc.py                 # IPC transport abstraction
│
├── cli/                       # CLI 命令
│   ├── __init__.py
│   └── main.py                # Click CLI app
│
└── utils/                     # 工具函数
    ├── __init__.py
    ├── bullet_factory.py      # BulletFactory
    ├── token_counter.py       # Token 预算计算
    └── text_processing.py     # 分词、N-gram、词干化

mem0/                          # mem0 原始代码（Fork, 最小修改）
├── configs/
│   └── base.py                # 唯一修改：扩展 MemoryConfig（或通过继承）
└── ... (保持不变)

tests/
├── unit/
│   ├── test_reflector.py
│   ├── test_curator.py
│   ├── test_decay.py
│   ├── test_generator.py
│   ├── test_sanitizer.py
│   ├── test_bullet.py
│   └── test_config.py
├── integration/
│   ├── test_ingest_pipeline.py
│   ├── test_retrieval_pipeline.py
│   ├── test_mem0_compat.py    # mem0 API 兼容测试
│   └── test_daemon.py
├── performance/
│   ├── test_search_latency.py
│   └── test_reflect_latency.py
└── conftest.py
```

### Module Structure

**依赖方向（单向，无循环）：**
```
cli → memory → pipeline → engines → types/config
                  │                      │
                  └── mem0 infrastructure ┘

privacy → (独立，被 pipeline 引用)
daemon → memory + engines
integration → memory + engines
```

### Testing Strategy

| 层级 | 工具 | 覆盖率目标 | 说明 |
|------|------|-----------|------|
| 单元测试 | pytest | > 80% | 每个 Engine + 每个 Matcher |
| 集成测试 | pytest | 关键路径 100% | add→reflect→curate→search→decay |
| mem0 兼容测试 | pytest | 100% | 运行 mem0 官方测试套件 |
| 性能测试 | pytest-benchmark | CI 门禁 | 检索 < 50ms, 蒸馏 < 20ms |
| 类型检查 | mypy (strict) | 100% | memx/ 包强制 |

### CI/CD Pipeline

```
Push / PR
    │
    ├── ruff check (lint + format)
    ├── mypy --strict memx/
    ├── pytest tests/unit/ --cov --cov-fail-under=80
    ├── pytest tests/integration/
    ├── pytest tests/integration/test_mem0_compat.py  # 兼容性门禁
    ├── pytest tests/performance/ --benchmark-compare  # 性能门禁
    │
    ▼
  All Green → Merge allowed

Release Tag
    │
    ├── poetry build
    ├── twine upload (PyPI)
    └── GitHub Release
```

---

## Deployment Architecture

### Environments

| 环境 | 用途 | 说明 |
|------|------|------|
| 本地开发 | 开发者机器 | `pip install -e .` |
| CI | GitHub Actions | 自动化测试 |
| PyPI (prod) | 公开发布 | `pip install memx` |
| Docker (enterprise) | 企业私有部署 | 可选 Dockerfile |

### Deployment Strategy

**开源版本：** PyPI 发布，SemVer 版本管理。

```
pip install memx            # 最小安装（Rules-only, 纯关键词）
pip install memx[onnx]      # + 本地 ONNX Embedding
pip install memx[graph]     # + 图存储支持
pip install memx[all]       # 全部依赖
```

### Infrastructure as Code

**Dockerfile (企业版)：**
```dockerfile
FROM python:3.11-slim
RUN pip install memx[all]
COPY config.json /app/config.json
CMD ["memx", "daemon", "--config", "/app/config.json"]
```

---

## Requirements Traceability

### Functional Requirements Coverage

| FR ID | FR Name | Components | Implementation Notes |
|-------|---------|------------|---------------------|
| FR-001 | Bullet 数据模型 | BulletFactory, BulletMetadata | `memx/types.py`, `memx/utils/bullet_factory.py` |
| FR-002 | Reflector Stage 1 | PatternDetector | `memx/engines/reflector/detector.py` |
| FR-003 | Reflector Stage 2 | KnowledgeScorer | `memx/engines/reflector/scorer.py` |
| FR-004 | Reflector Stage 3 | PrivacySanitizer | `memx/privacy/sanitizer.py` |
| FR-005 | Reflector Stage 4 | BulletDistiller | `memx/engines/reflector/distiller.py` |
| FR-006 | Curator 去重 | CuratorEngine | `memx/engines/curator/engine.py` |
| FR-007 | Decay 衰退 | DecayEngine | `memx/engines/decay/engine.py` |
| FR-008 | Decay 召回强化 | DecayEngine.reinforce | `memx/engines/decay/engine.py` |
| FR-009 | Generator 混合检索 | GeneratorEngine, 4x Matcher | `memx/engines/generator/` |
| FR-010 | Generator 降级模式 | GeneratorEngine.mode | `memx/engines/generator/engine.py` |
| FR-011 | Token 预算 | TokenBudgetTrimmer | `memx/utils/token_counter.py` |
| FR-012 | 配置系统 | MemXConfig + 子配置 | `memx/config.py` |
| FR-013 | API 兼容 | MemXMemory (Decorator) | `memx/memory.py` |
| FR-014 | Pre-Inference Hook | PreInferenceHook | `memx/integration/hooks.py` |
| FR-015 | Post-Action Hook | PostActionHook | `memx/integration/hooks.py` |
| FR-016 | Session-End Hook | SessionEndHook | `memx/integration/hooks.py` |
| FR-017 | ONNX Embedding | ONNXEmbedder | `memx/embeddings/onnx.py` |
| FR-018 | Daemon | MemXDaemon + DaemonClient | `memx/daemon/` |
| FR-019 | LLM 增强蒸馏 | ReflectorEngine (llm mode) | `memx/engines/reflector/engine.py` |
| FR-020 | 冲突检测 | ConflictDetector | `memx/engines/curator/conflict.py` |
| FR-021 | 层级 Scope | MemXMemory + GeneratorEngine | `memx/memory.py`, `memx/engines/generator/` |
| FR-022 | 导入导出 | MemXMemory.export/import | `memx/memory.py` |
| FR-023 | CLI 命令 | Click CLI app | `memx/cli/main.py` |
| FR-024 | PyPI 发布 | pyproject.toml + CI/CD | 项目根目录 |

### Non-Functional Requirements Coverage

| NFR ID | NFR Name | Solution | Validation |
|--------|----------|----------|------------|
| NFR-001 | 检索 < 50ms | 内存索引 + 异步强化 | pytest-benchmark CI 门禁 |
| NFR-002 | 蒸馏 < 20ms | 纯规则 + 正则预编译 | pytest-benchmark |
| NFR-003 | 隐私不可关闭 | Sanitizer hardcoded in pipeline | 代码审计 + 测试 |
| NFR-004 | 数据本地化 | 默认零网络 + 本地 ONNX | 代码审计 |
| NFR-005 | mem0 兼容 | Decorator 模式 | mem0 测试套件 100% |
| NFR-006 | 优雅降级 | Stage 级 try-catch | 故障注入测试 |
| NFR-007 | 测试 > 80% | pytest-cov CI 门禁 | 覆盖率报告 |
| NFR-008 | 上游同步 | 独立目录 + 月度 rebase | 文档化流程 |
| NFR-009 | 5K-50K 条 | 分级存储方案 | 压力测试 |
| NFR-010 | 零配置启动 | 合理默认值 + Decorator | 冒烟测试 |

---

## Trade-offs & Decision Log

### Decision 1: Decorator vs. 直接继承 mem0.Memory

**选择：** Decorator 模式（包装而非继承）

**Trade-off:**
- 得到：完全控制代理行为，mem0 内部变更不影响 MemX 接口
- 失去：需要手动代理所有 mem0 方法，稍有维护成本

**Rationale:** 继承耦合太紧，mem0 的 Memory.__init__ 逻辑复杂（初始化 VectorStore/LLM/Embedder），直接继承会导致 ACE 初始化时机难以控制。

---

### Decision 2: Bullet 元数据嵌入 payload vs. 独立表

**选择：** 嵌入 mem0 VectorStore payload metadata

**Trade-off:**
- 得到：零 schema 迁移，与 mem0 存储完全兼容，无需额外数据库
- 失去：无法对 Bullet 字段建索引（L1-L3 需要内存索引补偿）

**Rationale:** 独立表需要维护两份数据的一致性，增加复杂度。内存索引在 5000 条规模内性能足够。

---

### Decision 3: memx/ 独立包 vs. 修改 mem0/ 内部

**选择：** `memx/` 独立顶层包

**Trade-off:**
- 得到：最小化上游冲突，rebase 几乎零冲突
- 失去：无法直接访问 mem0 私有方法，某些集成需要通过公开 API

**Rationale:** 上游同步是高概率风险（NFR-008），最小化冲突面是第一优先级。

---

### Decision 4: Rules-only 默认 vs. LLM 默认

**选择：** Rules-only 作为默认蒸馏模式

**Trade-off:**
- 得到：零成本、零延迟、零网络依赖、零配置
- 失去：蒸馏质量可能低于 LLM 辅助模式

**Rationale:** 零配置启动（NFR-010）和零 LLM 成本是核心竞争优势。LLM 模式保留为可选增强。

---

### Decision 5: 同步管线 vs. 事件驱动

**选择：** 同步管线 + 选择性异步（RecallReinforcer）

**Trade-off:**
- 得到：实现简单，调试容易，行为可预测
- 失去：无法轻松添加异步观察者

**Rationale:** MemX 是库而非服务，引入事件总线/消息队列过度工程化。

---

## Open Issues & Risks

1. **PyPI 包名 `memx` 可用性**：需提前确认是否被占用，备选：`memx-ai`、`ace-memory`
2. **mem0 v1.0.x → v2.x 升级风险**：如果 mem0 发布破坏性 API 变更，MemX 需要重大适配
3. **ONNX 模型中文质量**：all-MiniLM-L6-v2 的中文语义质量有待实测验证
4. **Windows Named Pipe Python 支持**：asyncio 在 Windows 上的 Named Pipe 支持不如 Linux 成熟
5. **L1-L3 内存索引内存占用**：50,000 条时 inverted index 可能占用 200-500MB

---

## Assumptions & Constraints

**约束（不可违反）：**
- 必须 Local-First
- 必须兼容 mem0 API
- 零 LLM 成本默认
- 隐私脱敏不可关闭
- mem0 Apache 2.0 许可证约束

**假设（需验证）：**
- SQLite + brute-force 在 5000 条内满足 50ms 目标
- all-MiniLM-L6-v2 中文语义质量可用
- mem0 上游保持稳定维护
- Python 3.9+ 覆盖目标用户群

---

## Future Considerations

- **MemX Cloud**：基于 Daemon 扩展为远程服务，支持多用户
- **多模型 Embedding 切换**：paraphrase-multilingual-MiniLM 用于中文增强
- **Rust 核心引擎**：将 L1-L3 Matcher 用 Rust 重写，通过 PyO3 绑定
- **MCP Server 模式**：ACE 作为 MCP Server 发布，适配更多 AI 工具
- **知识可视化**：基于 OpenMemory UI 扩展 Decay 曲线、Scope 拓扑可视化

---

## Approval & Sign-off

**Review Status:**
- [ ] Technical Lead (TPY)
- [ ] Product Owner (TPY)
- [ ] Security Review
- [ ] DevOps Review

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-27 | TPY | Initial architecture |

---

## Next Steps

### Phase 4: Sprint Planning & Implementation

Run `/sprint-planning` to:
- Break epics into detailed user stories
- Estimate story complexity
- Plan sprint iterations
- Begin implementation following this architectural blueprint

**Key Implementation Principles:**
1. Follow component boundaries defined in this document
2. Implement NFR solutions as specified
3. Use technology stack as defined
4. Follow API contracts exactly
5. Adhere to security and performance guidelines

**Implementation Order (recommended):**
1. EPIC-001: Bullet + Config + API 兼容 (地基)
2. EPIC-004: Decay 引擎 (最独立)
3. EPIC-002: Reflector 引擎 (核心价值)
4. EPIC-003: Curator 引擎 (依赖 Reflector)
5. EPIC-005: Generator 引擎 (检索改造)
6. EPIC-006: Integration Layer (集成)
7. EPIC-007: ONNX + Daemon (本地化)
8. EPIC-008: CLI + 发布 (收尾)

---

**This document was created using BMAD Method v6 - Phase 3 (Solutioning)**

*To continue: Run `/workflow-status` to see your progress and next recommended workflow.*

---

## Appendix A: Technology Evaluation Matrix

| 技术领域 | 选项 A | 选项 B | 选项 C | 选择 | 理由 |
|----------|--------|--------|--------|------|------|
| 包装模式 | Decorator | 继承 | Monkey-patch | Decorator | 最佳解耦 |
| 数据存储 | 嵌入 payload | 独立 SQLite 表 | 独立 PostgreSQL | 嵌入 payload | 零迁移成本 |
| 代码组织 | memx/ 独立包 | 修改 mem0/ | 混合 | memx/ 独立包 | 上游同步友好 |
| 蒸馏默认 | Rules-only | LLM | Hybrid | Rules-only | 零成本零依赖 |
| IPC | Named Pipe/Socket | TCP | gRPC | Named Pipe/Socket | 安全+低延迟 |
| CLI 框架 | Click | argparse | Typer | Click | 轻量、标准 |
| 文档工具 | MkDocs | Sphinx | Docusaurus | MkDocs | Python 生态标准 |

---

## Appendix B: Capacity Planning

| 记忆规模 | 内存占用 | 磁盘占用 | 检索延迟 | 推荐方案 |
|----------|----------|----------|----------|----------|
| 100 条 | ~5MB | ~1MB | < 5ms | brute-force |
| 1,000 条 | ~15MB | ~5MB | < 15ms | brute-force |
| 5,000 条 | ~50MB | ~20MB | < 50ms | brute-force |
| 10,000 条 | ~100MB | ~40MB | < 80ms | sqlite-vec |
| 50,000 条 | ~500MB | ~200MB | < 100ms | sqlite-vec / LanceDB |

---

## Appendix C: Cost Estimation

| 模式 | LLM API 成本 | Embedding 成本 | 存储成本 |
|------|-------------|---------------|----------|
| Rules-only + ONNX | $0 | $0 | $0 (本地) |
| Rules-only + OpenAI Embedding | $0 | ~$0.01/1000 条 | $0 |
| LLM-assisted + ONNX | ~$0.005/session | $0 | $0 |
| LLM-distill + OpenAI all | ~$0.01/session | ~$0.01/1000 条 | $0 |
| mem0 原版 (对比) | ~$0.05/session | ~$0.01/1000 条 | $0 |

**结论：** Rules-only + ONNX 模式实现完全零运营成本。
