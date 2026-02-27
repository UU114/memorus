# Sprint Plan: MemX

**Date:** 2026-02-27
**Scrum Master:** TPY
**Project Level:** Level 3
**Total Stories:** 46
**Total Points:** 199
**Planned Sprints:** 4
**Team:** 4 developers, 2-week sprints
**Sprint Capacity:** ~63 points/sprint (70 raw, 10% buffer)
**Target Completion:** 2026-05-08 (Sprint 4 end)

---

## Executive Summary

MemX 的实现计划分为 4 个 Sprint（共 8 周），将 8 个 Epic 的 24 条 FR 拆解为 46 个精细 Story（总计 199 点）。Sprint 1-2 聚焦 MVP 核心引擎（Bullet + Reflector + Decay + Curator + Generator），Sprint 3 实现集成层和本地化，Sprint 4 完成 CLI、高级功能和 PyPI 发布。

**Key Metrics:**
- Total Stories: 46
- Total Points: 199
- Sprints: 4 (8 weeks)
- Team Capacity: ~63 points/sprint
- Target Completion: 2026-05-08

---

## Story Inventory

### EPIC-001: Bullet 数据模型与配置基础 (28 points, 7 stories)

---

#### STORY-001: 定义 BulletMetadata Pydantic 模型

**Epic:** EPIC-001
**Priority:** Must Have

**User Story:**
As a MemX developer
I want a well-defined BulletMetadata data model
So that all ACE engines have a consistent knowledge unit structure

**Acceptance Criteria:**
- [ ] BulletMetadata Pydantic 模型包含所有 ACE 字段（section, knowledge_type, instructivity_score, recall_count, last_recall, decay_weight, related_tools, related_files, key_entities, tags, distilled_rule, source_type, scope）
- [ ] 所有枚举类型定义完成（BulletSection, KnowledgeType, SourceType）
- [ ] 所有字段有合理默认值
- [ ] 单元测试覆盖模型校验和序列化

**Technical Notes:** `memx/types.py`
**Dependencies:** None
**Points:** 3

---

#### STORY-002: 实现 BulletFactory 工厂类

**Epic:** EPIC-001
**Priority:** Must Have

**User Story:**
As a MemX engine
I want a factory to create, serialize, and deserialize Bullets
So that Bullet creation is standardized across all modules

**Acceptance Criteria:**
- [ ] `BulletFactory.create(content, **kwargs)` 创建带默认元数据的 Bullet
- [ ] `BulletFactory.to_mem0_metadata(bullet)` 将 Bullet 字段转为 `memx_` 前缀 dict
- [ ] `BulletFactory.from_mem0_payload(payload)` 从 mem0 payload 解析 Bullet（无 memx_ 字段时使用默认值）
- [ ] 向后兼容测试：旧 mem0 payload 不报错

**Technical Notes:** `memx/utils/bullet_factory.py`
**Dependencies:** STORY-001
**Points:** 3

---

#### STORY-003: 定义 MemXConfig 配置模型

**Epic:** EPIC-001
**Priority:** Must Have

**User Story:**
As a developer
I want all ACE configuration in a single Pydantic model
So that I can configure MemX with reasonable defaults

**Acceptance Criteria:**
- [ ] MemXConfig 继承/扩展 mem0 MemoryConfig
- [ ] 包含 ace_enabled, RetrievalConfig, ReflectorConfig, CuratorConfig, DecayConfig, PrivacyConfig
- [ ] ace_enabled 默认 False
- [ ] 所有子配置有完整默认值
- [ ] 配置校验测试（非法值抛 ConfigurationError）

**Technical Notes:** `memx/config.py`
**Dependencies:** STORY-001
**Points:** 5

---

#### STORY-004: 实现 MemXMemory Decorator 包装类

**Epic:** EPIC-001
**Priority:** Must Have

**User Story:**
As a mem0 user migrating to MemX
I want to use the same API without code changes
So that migration is zero-effort

**Acceptance Criteria:**
- [ ] `memx.Memory` 包装 `mem0.Memory`
- [ ] ace_enabled=False 时所有方法直接代理到 mem0（透传）
- [ ] add/search/get_all/get/update/delete/delete_all/history/reset 签名不变
- [ ] from_config() 类方法兼容 mem0 config dict
- [ ] 运行 mem0 核心测试用例通过

**Technical Notes:** `memx/memory.py`, Decorator pattern
**Dependencies:** STORY-003
**Points:** 5

---

#### STORY-005: 实现 AsyncMemXMemory 异步包装类

**Epic:** EPIC-001
**Priority:** Must Have

**User Story:**
As a developer using async patterns
I want async version of MemX Memory
So that I can integrate with async frameworks

**Acceptance Criteria:**
- [ ] `memx.AsyncMemory` 包装 `mem0.AsyncMemory`
- [ ] 行为与同步版本一致（ace_enabled 开关逻辑相同）
- [ ] 异步测试覆盖核心路径

**Technical Notes:** `memx/async_memory.py`
**Dependencies:** STORY-004
**Points:** 3

---

#### STORY-006: 创建 MemX 项目骨架和包结构

**Epic:** EPIC-001
**Priority:** Must Have

**User Story:**
As a developer
I want a well-organized project structure
So that all team members follow consistent patterns

**Acceptance Criteria:**
- [ ] `memx/` 顶层包创建，含所有子目录（engines/, pipeline/, privacy/, integration/, embeddings/, daemon/, cli/, utils/）
- [ ] `__init__.py` 导出 Memory, AsyncMemory
- [ ] pyproject.toml 配置（包名 memx，Python 3.9+，依赖声明）
- [ ] 可选依赖分组：[onnx], [graph], [all]
- [ ] ruff + mypy 配置
- [ ] pytest + conftest 基础配置

**Technical Notes:** 项目根目录
**Dependencies:** None
**Points:** 5

---

#### STORY-007: mem0 兼容性测试套件

**Epic:** EPIC-001
**Priority:** Must Have

**User Story:**
As a QA engineer
I want automated tests verifying mem0 API compatibility
So that we catch breaking changes early

**Acceptance Criteria:**
- [ ] 提取 mem0 核心测试用例到 `tests/integration/test_mem0_compat.py`
- [ ] 所有测试使用 `memx.Memory` 替代 `mem0.Memory`
- [ ] ace_enabled=False 模式下 100% 通过
- [ ] CI 中作为必过门禁

**Technical Notes:** `tests/integration/test_mem0_compat.py`
**Dependencies:** STORY-004
**Points:** 4

---

### EPIC-002: Reflector 知识蒸馏引擎 (38 points, 9 stories)

---

#### STORY-008: 实现 PatternDetector（Stage 1）— 基础框架

**Epic:** EPIC-002
**Priority:** Must Have

**User Story:**
As a MemX engine
I want to detect learnable patterns from AI interactions
So that valuable knowledge can be captured automatically

**Acceptance Criteria:**
- [ ] InteractionEvent 数据结构定义（tool_name, input, output, success, error_msg）
- [ ] DetectedPattern 数据结构定义（pattern_type, raw_content, context）
- [ ] PatternDetector.detect() 方法框架
- [ ] 至少支持 2 种模式：错误修复模式、命令失败后成功
- [ ] 代码占比 > 60% 检测和拒绝

**Technical Notes:** `memx/engines/reflector/detector.py`
**Dependencies:** STORY-001
**Points:** 5

---

#### STORY-009: 扩展 PatternDetector — 更多模式规则

**Epic:** EPIC-002
**Priority:** Must Have

**User Story:**
As a MemX user
I want more interaction patterns detected
So that more useful knowledge is captured

**Acceptance Criteria:**
- [ ] 新增模式：配置变更模式、新工具/命令使用模式、重复操作模式
- [ ] 总计 ≥ 5 种模式
- [ ] 每种模式有独立测试用例
- [ ] 模式规则可通过子类扩展

**Technical Notes:** `memx/engines/reflector/patterns.py`
**Dependencies:** STORY-008
**Points:** 4

---

#### STORY-010: 实现 KnowledgeScorer（Stage 2）

**Epic:** EPIC-002
**Priority:** Must Have

**User Story:**
As a Reflector engine
I want candidates scored and classified
So that only valuable knowledge enters the Playbook

**Acceptance Criteria:**
- [ ] 每个候选分配 knowledge_type（Method/Trick/Pitfall/Preference/Knowledge）
- [ ] 每个候选分配 section（8 种分区）
- [ ] instructivity_score 计算：base × density_penalty + distill_bonus
- [ ] 低于阈值（默认 30）的候选被过滤
- [ ] 评分公式参数可通过 ReflectorConfig 配置

**Technical Notes:** `memx/engines/reflector/scorer.py`
**Dependencies:** STORY-008
**Points:** 4

---

#### STORY-011: 实现 PrivacySanitizer（Stage 3）

**Epic:** EPIC-002
**Priority:** Must Have

**User Story:**
As a privacy-conscious user
I want sensitive data automatically stripped from memories
So that API keys and passwords are never stored

**Acceptance Criteria:**
- [ ] 检测并脱敏 ≥ 10 种 API Key 格式（sk-*, ghp_*, AKIA*, Bearer token 等）
- [ ] 含用户名路径替换为 `<USER_PATH>`
- [ ] 密码字段（password=, secret=, token=）脱敏
- [ ] 返回 `SanitizeResult(clean, filtered_items)`
- [ ] 内置 pattern 列表为 hardcoded，不可通过配置移除
- [ ] 自定义 pattern 通过 `privacy.custom_patterns` 追加

**Technical Notes:** `memx/privacy/sanitizer.py`, `memx/privacy/patterns.py`
**Dependencies:** STORY-003 (PrivacyConfig)
**Points:** 5

---

#### STORY-012: 实现 BulletDistiller（Stage 4）

**Epic:** EPIC-002
**Priority:** Must Have

**User Story:**
As a Reflector engine
I want scored candidates distilled into standard Bullets
So that knowledge is compact and structured

**Acceptance Criteria:**
- [ ] content 截断 ≤ 500 字符
- [ ] code_content（如有）截断 ≤ 3 行
- [ ] 自动提取 related_tools 和 key_entities（基于正则/启发式）
- [ ] 输出 CandidateBullet 可直接传入 Curator

**Technical Notes:** `memx/engines/reflector/distiller.py`
**Dependencies:** STORY-001, STORY-002
**Points:** 3

---

#### STORY-013: 组装 ReflectorEngine 完整流水线

**Epic:** EPIC-002
**Priority:** Must Have

**User Story:**
As an IngestPipeline
I want a single ReflectorEngine entry point
So that I can call reflect() and get candidate Bullets

**Acceptance Criteria:**
- [ ] ReflectorEngine.reflect(event) 编排 Stage 1→2→3→4
- [ ] 支持 mode 配置："rules"（默认）/ "llm" / "hybrid"
- [ ] Rules 模式零 LLM 调用
- [ ] 每个 Stage 独立 try-catch（故障跳过该 Stage，不崩溃）
- [ ] 集成测试覆盖完整流水线

**Technical Notes:** `memx/engines/reflector/engine.py`
**Dependencies:** STORY-008~012
**Points:** 5

---

#### STORY-014: 实现 IngestPipeline（add 路径管线）

**Epic:** EPIC-002
**Priority:** Must Have

**User Story:**
As MemXMemory.add()
I want an ingest pipeline to process new memories
So that Reflector and Curator are automatically invoked

**Acceptance Criteria:**
- [ ] IngestPipeline.process(messages, metadata) 编排 Reflector → Curator → mem0.add
- [ ] 返回 IngestResult(bullets_added, merged, skipped, errors)
- [ ] Reflector 异常 → fallback 到 raw add
- [ ] Curator 异常 → 跳过去重，直接 Insert
- [ ] ace_enabled=True 时 MemXMemory.add() 调用此管线

**Technical Notes:** `memx/pipeline/ingest.py`
**Dependencies:** STORY-004, STORY-013
**Points:** 5

---

#### STORY-015: PrivacySanitizer hardcoded safety net

**Epic:** EPIC-002
**Priority:** Must Have

**User Story:**
As a security architect
I want Sanitizer to run even if Reflector is disabled
So that privacy is guaranteed regardless of configuration

**Acceptance Criteria:**
- [ ] IngestPipeline 中 Sanitizer 独立于 Reflector 执行
- [ ] 即使 ace_enabled=False，通过 mem0 add() 的数据也经过 Sanitizer（可选配置）
- [ ] 测试：关闭 Reflector 后 Sanitizer 仍生效

**Technical Notes:** 在 IngestPipeline 中 Sanitizer 位于 Reflector 之前
**Dependencies:** STORY-011, STORY-014
**Points:** 2

---

#### STORY-016: Reflector 单元测试全覆盖

**Epic:** EPIC-002
**Priority:** Must Have

**User Story:**
As a QA engineer
I want comprehensive tests for all Reflector stages
So that knowledge distillation quality is guaranteed

**Acceptance Criteria:**
- [ ] PatternDetector: 每种模式 ≥ 3 个测试用例（正例+反例）
- [ ] KnowledgeScorer: 评分边界测试
- [ ] PrivacySanitizer: 每种敏感信息格式 ≥ 2 个测试用例
- [ ] BulletDistiller: 长度截断、实体提取测试
- [ ] ReflectorEngine: 集成测试（完整 4-Stage 流水线）
- [ ] 覆盖率 > 85%

**Technical Notes:** `tests/unit/test_reflector.py`
**Dependencies:** STORY-013
**Points:** 5

---

### EPIC-003: Curator 语义去重引擎 (13 points, 3 stories)

---

#### STORY-017: 实现 CuratorEngine 核心去重逻辑

**Epic:** EPIC-003
**Priority:** Must Have

**User Story:**
As a MemX system
I want duplicate memories automatically merged
So that the Playbook stays clean and compact

**Acceptance Criteria:**
- [ ] 候选 Bullet 与现有记忆计算 cosine similarity
- [ ] similarity ≥ 阈值（默认 0.8）→ 标记为 Merge
- [ ] similarity < 阈值 → 标记为 Insert
- [ ] 阈值通过 CuratorConfig 配置
- [ ] 返回 CurateResult(to_add, to_merge, to_skip)

**Technical Notes:** `memx/engines/curator/engine.py`
**Dependencies:** STORY-001, STORY-002
**Points:** 5

---

#### STORY-018: 实现 MergeStrategy

**Epic:** EPIC-003
**Priority:** Must Have

**User Story:**
As a Curator
I want a merge strategy to combine similar memories
So that merged Bullets retain the best information

**Acceptance Criteria:**
- [ ] Merge 保留更长/更完整的 content
- [ ] 保留较高的 recall_count 和 instructivity_score
- [ ] related_tools 和 key_entities 取并集
- [ ] updated_at 更新为当前时间
- [ ] 支持 "keep_best" 和 "merge_content" 两种策略

**Technical Notes:** `memx/engines/curator/merger.py`
**Dependencies:** STORY-017
**Points:** 4

---

#### STORY-019: Curator 单元测试

**Epic:** EPIC-003
**Priority:** Must Have

**User Story:**
As a QA engineer
I want Curator fully tested
So that deduplication is reliable

**Acceptance Criteria:**
- [ ] 去重阈值边界测试（0.79 → Insert, 0.80 → Merge）
- [ ] Merge 策略结果验证
- [ ] 空 Playbook 处理
- [ ] 覆盖率 > 85%

**Technical Notes:** `tests/unit/test_curator.py`
**Dependencies:** STORY-018
**Points:** 4

---

### EPIC-004: Decay 衰退引擎 (14 points, 3 stories)

---

#### STORY-020: 实现 DecayEngine 核心衰退逻辑

**Epic:** EPIC-004
**Priority:** Must Have

**User Story:**
As a long-term user
I want old unused memories to naturally fade
So that my context stays relevant

**Acceptance Criteria:**
- [ ] `compute_weight()` 实现：`2^(-age_days/half_life) × (1 + boost × recall_count)`
- [ ] 保护期内（默认 7 天）weight 锁定 1.0
- [ ] recall_count ≥ 永久阈值（默认 15）→ weight = 1.0
- [ ] weight < 归档阈值（默认 0.02）→ 标记归档
- [ ] 所有参数通过 DecayConfig 可配置

**Technical Notes:** `memx/engines/decay/engine.py`, `memx/engines/decay/formulas.py`
**Dependencies:** STORY-001
**Points:** 4

---

#### STORY-021: 实现 Decay sweep 和召回强化

**Epic:** EPIC-004
**Priority:** Must Have

**User Story:**
As a MemX system
I want batch decay updates and recall reinforcement
So that memory lifecycle is automatically managed

**Acceptance Criteria:**
- [ ] `sweep()` 批量更新所有记忆的 decay_weight
- [ ] 返回 DecaySweepResult(updated, archived, permanent)
- [ ] `reinforce(bullet_ids)` 更新 recall_count + 1, last_recall = now()
- [ ] reinforce 异步执行，不阻塞 search
- [ ] 归档标记不物理删除记忆

**Technical Notes:** `memx/engines/decay/engine.py`
**Dependencies:** STORY-020
**Points:** 5

---

#### STORY-022: Decay 单元测试

**Epic:** EPIC-004
**Priority:** Must Have

**User Story:**
As a QA engineer
I want Decay engine thoroughly tested
So that forgetting and reinforcement are predictable

**Acceptance Criteria:**
- [ ] 衰退公式数值精度测试
- [ ] 保护期边界测试（第 6 天 vs 第 8 天）
- [ ] 永久保留阈值测试（14 次 vs 15 次）
- [ ] 归档阈值测试
- [ ] sweep 批量测试（1000 条）
- [ ] reinforce 异步测试
- [ ] 覆盖率 > 90%

**Technical Notes:** `tests/unit/test_decay.py`
**Dependencies:** STORY-021
**Points:** 5

---

### EPIC-005: Generator 混合检索引擎 (33 points, 8 stories)

---

#### STORY-023: 实现 ExactMatcher (L1)

**Epic:** EPIC-005
**Priority:** Must Have

**User Story:**
As a search engine
I want exact keyword matching
So that precise term hits get highest scores

**Acceptance Criteria:**
- [ ] 全词匹配检测（word boundary aware）
- [ ] 命中 +15 分（可配置）
- [ ] 支持中英文
- [ ] 性能：5000 条 < 3ms

**Technical Notes:** `memx/engines/generator/exact_matcher.py`
**Dependencies:** STORY-001
**Points:** 3

---

#### STORY-024: 实现 FuzzyMatcher (L2)

**Epic:** EPIC-005
**Priority:** Must Have

**User Story:**
As a search engine
I want fuzzy matching for approximate queries
So that typos and variants still find relevant memories

**Acceptance Criteria:**
- [ ] 中文 2-gram 分词匹配
- [ ] 英文词干化匹配（Porter stemmer 或简化版）
- [ ] 模糊匹配分数 0-10（按命中率）
- [ ] 性能：5000 条 < 5ms

**Technical Notes:** `memx/engines/generator/fuzzy_matcher.py`, `memx/utils/text_processing.py`
**Dependencies:** STORY-001
**Points:** 5

---

#### STORY-025: 实现 MetadataMatcher (L3)

**Epic:** EPIC-005
**Priority:** Must Have

**User Story:**
As a search engine
I want metadata-based matching
So that tool names and entity references boost relevance

**Acceptance Criteria:**
- [ ] related_tools 前缀匹配
- [ ] key_entities 前缀匹配
- [ ] tags 精确匹配
- [ ] 元数据匹配分数 0-10

**Technical Notes:** `memx/engines/generator/metadata_matcher.py`
**Dependencies:** STORY-001
**Points:** 3

---

#### STORY-026: 实现 VectorSearcher (L4) 适配器

**Epic:** EPIC-005
**Priority:** Must Have

**User Story:**
As a search engine
I want semantic vector search from mem0
So that meaning-based retrieval complements keyword search

**Acceptance Criteria:**
- [ ] 调用 mem0 的 VectorStore.search()
- [ ] 返回归一化相似度分数 0-1
- [ ] Embedding 异常时返回空结果（不报错）
- [ ] 支持 filters 透传

**Technical Notes:** `memx/engines/generator/vector_searcher.py`
**Dependencies:** STORY-004
**Points:** 3

---

#### STORY-027: 实现 ScoreMerger 综合评分

**Epic:** EPIC-005
**Priority:** Must Have

**User Story:**
As a search engine
I want a unified scoring formula
So that keyword and semantic results are properly blended

**Acceptance Criteria:**
- [ ] FinalScore = (KeywordScore × kw_weight + SemanticScore × sem_weight) × DecayWeight × RecencyBoost
- [ ] kw_weight / sem_weight 通过 RetrievalConfig 配置（默认 0.6 / 0.4）
- [ ] RecencyBoost: 7 天内 ×1.2（可配置）
- [ ] 降级模式：无 SemanticScore 时仅用 KeywordScore

**Technical Notes:** `memx/engines/generator/score_merger.py`
**Dependencies:** STORY-023~026, STORY-020
**Points:** 4

---

#### STORY-028: 组装 GeneratorEngine + 降级模式

**Epic:** EPIC-005
**Priority:** Must Have

**User Story:**
As a MemX system
I want a complete Generator engine with automatic degradation
So that search always works even without Embedding

**Acceptance Criteria:**
- [ ] GeneratorEngine.search() 编排 L1→L2→L3→L4→ScoreMerger
- [ ] `mode` 属性：Embedding 可用 → "full"，否则 → "degraded"
- [ ] 降级时跳过 L4，仅 L1-L3
- [ ] 降级事件记录 WARNING 日志
- [ ] Embedding 恢复后自动切回 "full"

**Technical Notes:** `memx/engines/generator/engine.py`
**Dependencies:** STORY-027
**Points:** 5

---

#### STORY-029: 实现 TokenBudgetTrimmer

**Epic:** EPIC-005
**Priority:** Should Have

**User Story:**
As an API consumer
I want search results within token budget
So that context injection doesn't exceed LLM limits

**Acceptance Criteria:**
- [ ] 按 FinalScore 从高到低填充
- [ ] 总 token 不超过预算（默认 2000）
- [ ] 条数不超过 max_results（默认 5）
- [ ] token 计算使用简单估算（4 chars ≈ 1 token）

**Technical Notes:** `memx/utils/token_counter.py`
**Dependencies:** STORY-028
**Points:** 3

---

#### STORY-030: 实现 RetrievalPipeline + RecallReinforcer

**Epic:** EPIC-005
**Priority:** Must Have

**User Story:**
As MemXMemory.search()
I want a retrieval pipeline that automatically reinforces recalled memories
So that frequently used knowledge persists

**Acceptance Criteria:**
- [ ] RetrievalPipeline.search() 编排 Generator → Trimmer → (Reranker) → Reinforce
- [ ] RecallReinforcer 异步更新 recall_count（不阻塞返回）
- [ ] 返回 SearchResult(results, mode)
- [ ] ace_enabled=True 时 MemXMemory.search() 调用此管线
- [ ] 集成测试覆盖完整 search 路径

**Technical Notes:** `memx/pipeline/retrieval.py`
**Dependencies:** STORY-028, STORY-021, STORY-004
**Points:** 5

---

#### STORY-031: Generator 单元测试全覆盖

**Epic:** EPIC-005
**Priority:** Must Have

**User Story:**
As a QA engineer
I want Generator fully tested
So that search quality is guaranteed

**Acceptance Criteria:**
- [ ] 每个 Matcher 独立测试（中英文输入）
- [ ] ScoreMerger 权重计算测试
- [ ] 降级模式测试
- [ ] 端到端检索集成测试
- [ ] 覆盖率 > 85%

**Technical Notes:** `tests/unit/test_generator.py`, `tests/integration/test_retrieval_pipeline.py`
**Dependencies:** STORY-030
**Points:** 2 (included with STORY-030)

---

### EPIC-006: Integration Layer (18 points, 4 stories)

---

#### STORY-032: 实现 IntegrationManager + BaseHook 抽象

**Epic:** EPIC-006
**Priority:** Should Have

**User Story:**
As an AI product builder
I want a hook registration system
So that MemX automatically integrates with my product

**Acceptance Criteria:**
- [ ] BaseHook 抽象基类定义
- [ ] PreInferenceHook, PostActionHook, SessionEndHook 接口定义
- [ ] IntegrationManager 注册/注销 Hook
- [ ] IntegrationConfig 控制启停

**Technical Notes:** `memx/integration/manager.py`, `memx/integration/hooks.py`
**Dependencies:** STORY-004
**Points:** 4

---

#### STORY-033: 实现 CLI PreInferenceHook

**Epic:** EPIC-006
**Priority:** Should Have

**User Story:**
As a CLI AI tool user
I want relevant memories automatically recalled before each query
So that the AI remembers past interactions

**Acceptance Criteria:**
- [ ] 读取 stdin 用户输入 → 调用 Memory.search()
- [ ] 格式化为 XML 上下文模板
- [ ] 输出 additionalContext 到 stdout
- [ ] 可配置启用/禁用

**Technical Notes:** `memx/integration/cli_hooks.py`
**Dependencies:** STORY-032, STORY-030
**Points:** 4

---

#### STORY-034: 实现 CLI PostActionHook + SessionEndHook

**Epic:** EPIC-006
**Priority:** Should Have

**User Story:**
As a CLI AI tool
I want knowledge captured after tool calls and at session end
So that learning happens automatically

**Acceptance Criteria:**
- [ ] PostActionHook: 接收 ToolEvent → 触发 Reflector（异步）
- [ ] SessionEndHook: 兜底蒸馏 + run_decay_sweep()
- [ ] 支持 SIGTERM/SIGINT 信号触发 SessionEnd
- [ ] 蒸馏异步执行，不阻塞主流程

**Technical Notes:** `memx/integration/cli_hooks.py`
**Dependencies:** STORY-032, STORY-014, STORY-021
**Points:** 5

---

#### STORY-035: Integration 单元测试

**Epic:** EPIC-006
**Priority:** Should Have

**User Story:**
As a QA engineer
I want Integration hooks tested
So that auto-learn and auto-recall are reliable

**Acceptance Criteria:**
- [ ] Hook 注册/注销测试
- [ ] PreInference 召回格式测试
- [ ] PostAction 异步蒸馏测试
- [ ] SessionEnd 信号处理测试

**Technical Notes:** `tests/unit/test_integration.py`
**Dependencies:** STORY-034
**Points:** 5

---

### EPIC-007: 本地 Embedding + Daemon (25 points, 5 stories)

---

#### STORY-036: 实现 ONNXEmbedder Provider

**Epic:** EPIC-007
**Priority:** Should Have

**User Story:**
As a privacy-first user
I want local embedding without internet
So that my data never leaves my machine

**Acceptance Criteria:**
- [ ] ONNXEmbedder 实现 mem0 EmbeddingBase 接口
- [ ] 默认模型 all-MiniLM-L6-v2，384 维
- [ ] 模型自动下载到 ~/.memx/models/
- [ ] 离线使用（模型已下载后无网络请求）
- [ ] 单条 embed < 10ms
- [ ] 注册到 EmbedderFactory，provider="onnx"

**Technical Notes:** `memx/embeddings/onnx.py`
**Dependencies:** STORY-003
**Points:** 5

---

#### STORY-037: 实现 MemXDaemon 服务端

**Epic:** EPIC-007
**Priority:** Should Have

**User Story:**
As a power user
I want a daemon process to avoid cold starts
So that Hook calls are fast

**Acceptance Criteria:**
- [ ] Daemon 启动/关闭/健康检查
- [ ] PID 文件管理（防重复启动）
- [ ] IPC 协议：ping/recall/curate/session_register/session_unregister/shutdown
- [ ] 空闲超时自动退出（默认 5 分钟无活跃 session）

**Technical Notes:** `memx/daemon/server.py`
**Dependencies:** STORY-036, STORY-004
**Points:** 8

---

#### STORY-038: 实现 DaemonClient + IPC Transport

**Epic:** EPIC-007
**Priority:** Should Have

**User Story:**
As a Hook implementation
I want a client to communicate with Daemon
So that I can recall/curate through IPC

**Acceptance Criteria:**
- [ ] DaemonClient 封装 IPC 调用
- [ ] 支持 Named Pipe (Windows) + Unix Socket (Linux/Mac)
- [ ] 自动探测 Daemon 是否运行
- [ ] Daemon 不可用时抛出可捕获异常（供降级处理）

**Technical Notes:** `memx/daemon/client.py`, `memx/daemon/ipc.py`
**Dependencies:** STORY-037
**Points:** 5

---

#### STORY-039: Daemon 降级逻辑

**Epic:** EPIC-007
**Priority:** Should Have

**User Story:**
As a MemX system
I want graceful fallback when Daemon is unavailable
So that MemX always works

**Acceptance Criteria:**
- [ ] DaemonClient.ping() 失败 → 回退到直接调用 Memory
- [ ] 降级过程用户透明
- [ ] 降级 WARNING 日志
- [ ] Daemon 恢复后自动重连

**Technical Notes:** 集成到 `memx/memory.py`
**Dependencies:** STORY-038
**Points:** 3

---

#### STORY-040: Daemon 测试

**Epic:** EPIC-007
**Priority:** Should Have

**User Story:**
As a QA engineer
I want Daemon tested
So that multi-session and lifecycle are reliable

**Acceptance Criteria:**
- [ ] 启动/关闭生命周期测试
- [ ] IPC 协议测试（所有 6 个命令）
- [ ] 空闲超时测试
- [ ] 降级逻辑测试

**Technical Notes:** `tests/integration/test_daemon.py`
**Dependencies:** STORY-039
**Points:** 4

---

### EPIC-008: 用户界面与发布 (30 points, 7 stories)

---

#### STORY-041: 实现 CLI 基础命令（status + search）

**Epic:** EPIC-008
**Priority:** Should Have

**User Story:**
As a MemX user
I want CLI commands to inspect my memory
So that I can see what MemX has learned

**Acceptance Criteria:**
- [ ] `memx status` 显示记忆总数、section 分布、knowledge_type 分布、平均 decay_weight
- [ ] `memx search <query>` 使用混合检索
- [ ] 支持 --json 输出格式
- [ ] Click 框架实现

**Technical Notes:** `memx/cli/main.py`
**Dependencies:** STORY-004, STORY-030
**Points:** 4

---

#### STORY-042: 实现 CLI 管理命令（learn + list + forget）

**Epic:** EPIC-008
**Priority:** Should Have

**User Story:**
As a MemX user
I want to manually manage memories
So that I have full control over my Playbook

**Acceptance Criteria:**
- [ ] `memx learn <content>` 经过 Reflector + Curator 处理
- [ ] `memx list` 列出记忆（支持 --scope 过滤）
- [ ] `memx forget <id>` 删除指定记忆
- [ ] `memx sweep` 手动衰退扫描

**Technical Notes:** `memx/cli/main.py`
**Dependencies:** STORY-041
**Points:** 4

---

#### STORY-043: 实现层级 Scope 管理

**Epic:** EPIC-008
**Priority:** Could Have

**User Story:**
As a multi-project developer
I want separate knowledge scopes per project
So that project-specific knowledge doesn't leak

**Acceptance Criteria:**
- [ ] scope 字段支持 "global" 和 "project:{name}"
- [ ] search 自动合并 project + global 两个 scope
- [ ] project scope 记忆在评分中加权
- [ ] 保留 user_id/agent_id 正交维度

**Technical Notes:** `memx/memory.py`, `memx/engines/generator/engine.py`
**Dependencies:** STORY-004, STORY-028
**Points:** 5

---

#### STORY-044: 实现导入/导出功能

**Epic:** EPIC-008
**Priority:** Could Have

**User Story:**
As a MemX user
I want to backup and transfer my memories
So that my knowledge is portable

**Acceptance Criteria:**
- [ ] export(format="json") 导出完整 Bullet 元数据
- [ ] export(format="markdown") 导出人类可读列表
- [ ] import(data, format="json") 导入并经过 Curator 去重
- [ ] 支持 scope 过滤

**Technical Notes:** `memx/memory.py`
**Dependencies:** STORY-004, STORY-017
**Points:** 4

---

#### STORY-045: 性能基准测试套件

**Epic:** EPIC-008
**Priority:** Must Have

**User Story:**
As a developer
I want automated performance benchmarks
So that we catch performance regressions in CI

**Acceptance Criteria:**
- [ ] 检索延迟基准：5000 条 < 50ms
- [ ] 蒸馏延迟基准：单次 < 20ms
- [ ] ONNX embed 基准：单条 < 10ms
- [ ] pytest-benchmark 集成
- [ ] CI 门禁（超标则失败）

**Technical Notes:** `tests/performance/`
**Dependencies:** STORY-030, STORY-013
**Points:** 4

---

#### STORY-046: PyPI 打包与发布

**Epic:** EPIC-008
**Priority:** Must Have

**User Story:**
As an end user
I want to install MemX with pip
So that getting started is trivial

**Acceptance Criteria:**
- [ ] pyproject.toml 完整（包名、版本、描述、依赖、可选依赖分组）
- [ ] `pip install memx` 成功安装并运行
- [ ] `pip install memx[onnx]` / `memx[graph]` / `memx[all]`
- [ ] README 含快速开始指南
- [ ] CHANGELOG 初始版本
- [ ] GitHub Actions 自动发布到 PyPI
- [ ] 版本号 1.0.0

**Technical Notes:** 项目根目录 pyproject.toml, .github/workflows/publish.yml
**Dependencies:** STORY-006, STORY-007, STORY-045
**Points:** 5

---

#### STORY-047: 冲突检测（Conflict Detector）

**Epic:** EPIC-008
**Priority:** Could Have

**User Story:**
As a MemX user
I want to know if my memories contain contradictions
So that I can resolve conflicting knowledge

**Acceptance Criteria:**
- [ ] detect_conflicts() 识别 similarity 0.5-0.8 且内容矛盾的记忆对
- [ ] 不阻塞入库流程，仅标记
- [ ] 返回 Conflict 列表（两条记忆 ID、content、similarity）

**Technical Notes:** `memx/engines/curator/conflict.py`
**Dependencies:** STORY-017
**Points:** 4

---

## Sprint Allocation

### Sprint 1 (Week 1-2): 基础地基 + Reflector — 60/63 points

**Goal:** 完成 MemX 项目骨架、数据模型、配置系统、mem0 兼容层、Reflector 知识蒸馏引擎全部 4 个 Stage。Sprint 结束时可运行 `memx.Memory(ace_enabled=True).add()` 并观察到蒸馏+脱敏生效。

**Stories:**

| Story | Title | Points | Priority | Epic |
|-------|-------|--------|----------|------|
| STORY-006 | 项目骨架和包结构 | 5 | Must | 001 |
| STORY-001 | BulletMetadata 模型 | 3 | Must | 001 |
| STORY-002 | BulletFactory | 3 | Must | 001 |
| STORY-003 | MemXConfig 配置模型 | 5 | Must | 001 |
| STORY-004 | MemXMemory Decorator | 5 | Must | 001 |
| STORY-005 | AsyncMemXMemory | 3 | Must | 001 |
| STORY-007 | mem0 兼容测试 | 4 | Must | 001 |
| STORY-008 | PatternDetector 基础 | 5 | Must | 002 |
| STORY-009 | PatternDetector 扩展 | 4 | Must | 002 |
| STORY-010 | KnowledgeScorer | 4 | Must | 002 |
| STORY-011 | PrivacySanitizer | 5 | Must | 002 |
| STORY-012 | BulletDistiller | 3 | Must | 002 |
| STORY-013 | ReflectorEngine 组装 | 5 | Must | 002 |
| STORY-015 | Sanitizer safety net | 2 | Must | 002 |
| STORY-014 | IngestPipeline | 4 → **moved partial** | Must | 002 |

**Total:** 60 points / 63 capacity (95%)

**Risks:**
- STORY-004（Decorator）复杂度可能超预期（mem0 内部初始化逻辑复杂）
- STORY-011（Sanitizer）需覆盖大量 API Key 格式

**Notes:** STORY-014 IngestPipeline 的 Curator 部分在 Sprint 2 完成后对接

---

### Sprint 2 (Week 3-4): Decay + Curator + Generator — 62/63 points ✅ COMPLETED

**Goal:** 完成 Decay 衰退引擎、Curator 去重引擎、Generator 混合检索引擎。Sprint 结束时完整的 add → reflect → curate → search → decay 闭环可运行。

**Status:** ✅ All 15 stories completed on 2026-02-27. Total 897 tests passing, 0 regressions.

**Stories:**

| Story | Title | Points | Priority | Epic | Status |
|-------|-------|--------|----------|------|--------|
| STORY-016 | Reflector 测试全覆盖 | 4 | Must | 002 | ✅ Done (359 cases) |
| STORY-017 | Curator 核心去重 | 5 | Must | 003 | ✅ Done (34 tests) |
| STORY-018 | MergeStrategy | 4 | Must | 003 | ✅ Done (34 tests) |
| STORY-019 | Curator 测试 | 4 | Must | 003 | ✅ Done (90 tests, 99% coverage) |
| STORY-020 | Decay 核心衰退 | 4 | Must | 004 | ✅ Done (39 tests) |
| STORY-021 | Decay sweep + reinforce | 5 | Must | 004 | ✅ Done (58 tests) |
| STORY-022 | Decay 测试 | 5 | Must | 004 | ✅ Done (48 new, 100% coverage) |
| STORY-023 | ExactMatcher L1 | 3 | Must | 005 | ✅ Done (38 tests) |
| STORY-024 | FuzzyMatcher L2 | 5 | Must | 005 | ✅ Done (60 tests) |
| STORY-025 | MetadataMatcher L3 | 3 | Must | 005 | ✅ Done (41 tests) |
| STORY-026 | VectorSearcher L4 | 3 | Must | 005 | ✅ Done (35 tests) |
| STORY-027 | ScoreMerger | 4 | Must | 005 | ✅ Done (36 tests) |
| STORY-028 | GeneratorEngine + 降级 | 5 | Must | 005 | ✅ Done (35 tests) |
| STORY-029 | TokenBudgetTrimmer | 3 | Should | 005 | ✅ Done (25 tests) |
| STORY-030 | RetrievalPipeline | 5 | Must | 005 | ✅ Done (31 tests) |

**Total:** 62 points / 63 capacity (98%) — **ALL DELIVERED**

**Milestone:** Sprint 2 结束 = **MVP 核心引擎完成** ✅

---

### Sprint 3 (Week 5-6): Integration + ONNX + Daemon — 56/63 points

**Goal:** 完成集成层（三个 Hook）、ONNX 本地 Embedding、Daemon 常驻进程。Sprint 结束时 MemX 可在 CLI AI 工具中自动运行。

**Stories:**

| Story | Title | Points | Priority | Epic |
|-------|-------|--------|----------|------|
| STORY-032 | IntegrationManager 抽象 | 4 | Should | 006 |
| STORY-033 | CLI PreInferenceHook | 4 | Should | 006 |
| STORY-034 | PostAction + SessionEnd | 5 | Should | 006 |
| STORY-035 | Integration 测试 | 5 | Should | 006 |
| STORY-036 | ONNXEmbedder | 5 | Should | 007 |
| STORY-037 | MemXDaemon 服务端 | 8 | Should | 007 |
| STORY-038 | DaemonClient + IPC | 5 | Should | 007 |
| STORY-039 | Daemon 降级 | 3 | Should | 007 |
| STORY-040 | Daemon 测试 | 4 | Should | 007 |
| STORY-031 | Generator 测试全覆盖 | 5 | Must | 005 |
| STORY-041 | CLI status + search | 4 | Should | 008 |
| STORY-042 | CLI learn + list + forget | 4 | Should | 008 |

**Total:** 56 points / 63 capacity (89%)

**Risks:**
- Daemon (STORY-037) 是最大单体 Story（8 点），Windows Named Pipe 可能有坑
- ONNX 模型下载和加载的跨平台兼容性

---

### Sprint 4 (Week 7-8): 高级功能 + 发布 — 21/63 points

**Goal:** 完成高级功能（Scope、导入导出、冲突检测）、性能基准测试、PyPI 打包发布。Sprint 结束时 `pip install memx` 可用。

**Stories:**

| Story | Title | Points | Priority | Epic |
|-------|-------|--------|----------|------|
| STORY-043 | 层级 Scope 管理 | 5 | Could | 008 |
| STORY-044 | 导入/导出 | 4 | Could | 008 |
| STORY-047 | 冲突检测 | 4 | Could | 008 |
| STORY-045 | 性能基准测试 | 4 | Must | 008 |
| STORY-046 | PyPI 打包发布 | 5 | Must | 008 |

**Total:** 22 points / 63 capacity (35%)

**Buffer:** 41 点剩余容量用于：
- Sprint 1-3 遗留 Bug 修复
- 文档完善（README、API 文档、迁移指南）
- LLM 增强蒸馏（FR-019）如果时间允许
- 额外测试和代码审查

---

## Epic Traceability

| Epic ID | Epic Name | Stories | Total Points | Sprint |
|---------|-----------|---------|--------------|--------|
| EPIC-001 | Bullet + Config + 兼容 | 001-007 | 28 | Sprint 1 |
| EPIC-002 | Reflector 蒸馏引擎 | 008-016 | 38 | Sprint 1-2 |
| EPIC-003 | Curator 去重引擎 | 017-019 | 13 | Sprint 2 |
| EPIC-004 | Decay 衰退引擎 | 020-022 | 14 | Sprint 2 |
| EPIC-005 | Generator 检索引擎 | 023-031 | 33 | Sprint 2-3 |
| EPIC-006 | Integration Layer | 032-035 | 18 | Sprint 3 |
| EPIC-007 | ONNX + Daemon | 036-040 | 25 | Sprint 3 |
| EPIC-008 | CLI + 发布 | 041-047 | 30 | Sprint 3-4 |
| **Total** | | **46 stories** | **199 pts** | **4 sprints** |

---

## Functional Requirements Coverage

| FR ID | FR Name | Story | Sprint |
|-------|---------|-------|--------|
| FR-001 | Bullet 数据模型 | STORY-001, 002 | 1 |
| FR-002 | Reflector Stage 1 | STORY-008, 009 | 1 |
| FR-003 | Reflector Stage 2 | STORY-010 | 1 |
| FR-004 | Reflector Stage 3 | STORY-011, 015 | 1 |
| FR-005 | Reflector Stage 4 | STORY-012, 013 | 1 |
| FR-006 | Curator 去重 | STORY-017, 018 | 2 |
| FR-007 | Decay 衰退 | STORY-020 | 2 |
| FR-008 | Decay 召回强化 | STORY-021 | 2 |
| FR-009 | Generator 混合检索 | STORY-023-028 | 2 |
| FR-010 | Generator 降级 | STORY-028 | 2 |
| FR-011 | Token 预算 | STORY-029 | 2 |
| FR-012 | 配置系统 | STORY-003 | 1 |
| FR-013 | API 兼容 | STORY-004, 007 | 1 |
| FR-014 | Pre-Inference Hook | STORY-033 | 3 |
| FR-015 | Post-Action Hook | STORY-034 | 3 |
| FR-016 | Session-End Hook | STORY-034 | 3 |
| FR-017 | ONNX Embedding | STORY-036 | 3 |
| FR-018 | Daemon | STORY-037, 038, 039 | 3 |
| FR-019 | LLM 增强蒸馏 | (Sprint 4 buffer) | 4 |
| FR-020 | 冲突检测 | STORY-047 | 4 |
| FR-021 | 层级 Scope | STORY-043 | 4 |
| FR-022 | 导入导出 | STORY-044 | 4 |
| FR-023 | CLI 命令 | STORY-041, 042 | 3 |
| FR-024 | PyPI 发布 | STORY-046 | 4 |

**Coverage: 24/24 FRs (100%)**

---

## Risks and Mitigation

**High:**
- **Daemon Windows 兼容性** — Named Pipe 在 Python asyncio 上的 Windows 实现可能不稳定
  - Mitigation: Sprint 3 初期先做 Windows 可行性验证；备选方案 TCP localhost
- **mem0 上游重大更新** — 8 周内 mem0 可能发新版
  - Mitigation: Fork 时锁定 v1.0.4；Sprint 4 buffer 时间评估合并

**Medium:**
- **FuzzyMatcher 中文分词质量** — 简单 2-gram 可能不够好
  - Mitigation: 先做简版，后续可替换为 jieba 分词
- **Sprint 1 容量紧张**（60/63 = 95%）
  - Mitigation: STORY-014 IngestPipeline 可部分延到 Sprint 2
- **ONNX 模型大小** — all-MiniLM-L6-v2 约 90MB，首次下载较慢
  - Mitigation: 支持离线安装模式

**Low:**
- **PyPI 包名冲突** — `memx` 可能已被占用
  - Mitigation: Sprint 1 即检查，备选 `memx-ai`

---

## Dependencies

**内部依赖：**
- Sprint 2 依赖 Sprint 1 的 Bullet + Config + Decorator 基础
- Sprint 3 依赖 Sprint 2 的 Generator + Decay 引擎
- Sprint 4 依赖 Sprint 1-3 的所有核心组件

**外部依赖：**
- mem0 v1.0.4 代码库（Sprint 1 Day 1 Fork）
- onnxruntime PyPI 包（Sprint 3）
- all-MiniLM-L6-v2 ONNX 模型文件（Sprint 3）
- PyPI 账号注册（Sprint 4 前完成）

---

## Definition of Done

For a story to be considered complete:
- [ ] Code implemented and committed
- [ ] Unit tests written and passing (≥80% coverage for new code)
- [ ] mypy --strict passes on modified files
- [ ] ruff check passes
- [ ] Integration tests passing (if applicable)
- [ ] Code reviewed and approved
- [ ] Acceptance criteria all validated

---

## Next Steps

**Immediate:** Begin Sprint 1

Run `/dev-story STORY-006` to start with project skeleton setup, then proceed to STORY-001.

**Recommended Sprint 1 execution order:**
1. STORY-006 (项目骨架) — 团队可以开始编码
2. STORY-001 + STORY-002 (数据模型) — 并行
3. STORY-003 (配置) — 依赖 STORY-001
4. STORY-004 + STORY-005 (Memory 类) — 依赖 STORY-003
5. STORY-007 (兼容测试) — 依赖 STORY-004
6. STORY-008~013 (Reflector 全部) — 可与 STORY-004~007 并行
7. STORY-011 (Sanitizer) — 可与 Reflector 其他 Stage 并行
8. STORY-014~015 (IngestPipeline) — 最后组装

**Sprint cadence:**
- Sprint length: 2 weeks
- Sprint planning: Day 1
- Daily standup: 15 min
- Sprint review: Last day
- Sprint retrospective: Last day

---

**This plan was created using BMAD Method v6 - Phase 4 (Implementation Planning)**

*To continue: Run `/dev-story STORY-006` to begin implementing the first story.*
