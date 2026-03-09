# Sprint Plan: MemX

**Date:** 2026-03-08
**Scrum Master:** TPY
**Project Level:** Level 3
**Total Stories:** 72 (46 Core + 26 Team Memory)
**Total Points:** 310 (199 Core + 111 Team Memory)
**Planned Sprints:** 7 (4 Core + 3 Team Memory)
**Team:** 4 developers, 2-week sprints
**Sprint Capacity:** ~63 points/sprint (70 raw, 10% buffer)
**Target Completion:** 2026-06-19 (Sprint 7 end)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-27 | Initial plan — 46 stories, 199 pts, 4 sprints (Core EPIC-001~008) |
| 2.0 | 2026-03-08 | Team Memory extension — +26 stories, +111 pts, +3 sprints (EPIC-009~012) |

---

## Executive Summary

MemX 的实现计划从 4 个 Sprint（Core）扩展为 7 个 Sprint（Core + Team Memory），共 14 周。

- **Sprint 1-4**（Core，原计划不变）：MVP 核心引擎 + Integration + ONNX/Daemon + 发布
- **Sprint 5**（P0+P1）：Core/Team 解耦重构 + Git Fallback
- **Sprint 6**（P2）：Federation Mode MVP
- **Sprint 7**（P3）：Team 治理 + 高级功能

**关键原则**：P0（解耦）先行 → P1（Git Fallback）验证需求 → P2（Federation）再建 Server → 每个阶段独立可交付。

**Key Metrics:**
- Total Stories: 72 (46 Core + 26 Team)
- Total Points: 310 (199 + 111)
- Sprints: 7 (14 weeks)
- Team Capacity: ~63 points/sprint
- Target Completion: 2026-06-19

---

## Sprint 1-4 Summary (Core — unchanged)

Sprint 1-4 的 46 个 Story 与 v1.0 计划完全一致，不再重复列出。详见 `docs/sprint-plan-memx-2026-02-27.md`。

| Sprint | Goal | Points | Status |
|--------|------|--------|--------|
| Sprint 1 (W1-2) | 基础地基 + Reflector | 60/63 | Planned |
| Sprint 2 (W3-4) | Decay + Curator + Generator | 62/63 | ✅ Completed |
| Sprint 3 (W5-6) | Integration + ONNX + Daemon | 56/63 | Planned |
| Sprint 4 (W7-8) | 高级功能 + PyPI 发布 | 22/63 | Planned |

---

## Team Memory Story Inventory (v2.0 新增)

### EPIC-009: Core/Team 解耦重构 (21 points, 6 stories)

---

#### STORY-048: 重构 memx/ → memx/core/ 包结构

**Epic:** EPIC-009
**Priority:** Must Have

**User Story:**
As a MemX developer
I want the codebase restructured into core/ and team/ packages
So that Core and Team have clear boundaries and independent lifecycles

**Acceptance Criteria:**
- [ ] 所有现有代码从 `memx/` 移动到 `memx/core/`
- [ ] 所有 import path 更新完成
- [ ] `memx/__init__.py` 顶层导出保持不变（`from memx import Memory` 仍可用）
- [ ] 全部现有测试通过，零改动
- [ ] `memx/team/` 目录创建（空 `__init__.py`）

**Technical Notes:** 这是一次纯结构性重构，不修改任何逻辑。所有 import 通过 `memx/__init__.py` 的重新导出保持向后兼容。
**Dependencies:** Sprint 1-2 Core 代码完成
**Points:** 5

---

#### STORY-049: 定义 TeamConfig 独立配置模型

**Epic:** EPIC-009
**Priority:** Must Have

**User Story:**
As a MemX developer
I want a TeamConfig independent from MemXConfig
So that Team configuration can evolve without affecting Core

**Acceptance Criteria:**
- [ ] `TeamConfig` Pydantic 模型定义完成，包含 enabled/server_url/team_id/subscribed_tags/cache_max_bullets/cache_ttl_minutes
- [ ] `AutoNominateConfig`、`RedactorConfig`、`LayerBoostConfig`、`MandatoryOverride` 子模型完成
- [ ] TeamConfig 与 MemXConfig 完全独立（不继承、不嵌套）
- [ ] 配置加载支持文件和环境变量
- [ ] 单元测试覆盖校验和默认值

**Technical Notes:** `memx/team/config.py`
**Dependencies:** STORY-048
**Points:** 3

---

#### STORY-050: 定义 TeamBullet 数据模型

**Epic:** EPIC-009
**Priority:** Must Have

**User Story:**
As a MemX developer
I want a TeamBullet model extending Bullet
So that team knowledge can carry governance metadata

**Acceptance Criteria:**
- [ ] `TeamBullet` 继承 `Bullet`，新增 author_id/enforcement/upvotes/downvotes/status/deleted_at/origin_id/context_summary
- [ ] `schema_version=2` 自动设置
- [ ] v1 → v2 读取自动填充默认值（enforcement="suggestion", upvotes=0, status="approved"）
- [ ] v2 → v1 序列化时保留未知字段（serde flatten）
- [ ] 单元测试覆盖正反向兼容

**Technical Notes:** `memx/team/types.py`
**Dependencies:** STORY-051
**Points:** 3

---

#### STORY-051: 扩展 BulletMetadata — schema_version + incompatible_tags

**Epic:** EPIC-009
**Priority:** Must Have

**User Story:**
As a MemX developer
I want Bullet to have schema_version and incompatible_tags fields
So that Team conflict detection and schema evolution are supported

**Acceptance Criteria:**
- [ ] `BulletMetadata` 新增 `schema_version: int = 1`
- [ ] `BulletMetadata` 新增 `incompatible_tags: list[str] = []`
- [ ] 两个字段有默认值，旧数据完全向后兼容
- [ ] mem0 payload 中使用 `memx_schema_version` 和 `memx_incompatible_tags` 前缀
- [ ] BulletFactory 正确处理新字段的序列化/反序列化
- [ ] 全部现有测试通过

**Technical Notes:** `memx/core/types.py` — Core 的唯一小幅修改
**Dependencies:** None
**Points:** 2

---

#### STORY-052: 实现 ext/team_bootstrap.py 条件注入

**Epic:** EPIC-009
**Priority:** Must Have

**User Story:**
As a MemX developer
I want Team Layer to be injected conditionally at startup
So that Core code never directly depends on Team

**Acceptance Criteria:**
- [ ] `try_bootstrap_team(memory, config_path)` 实现条件导入
- [ ] Team 包未安装时静默跳过（ImportError 捕获）
- [ ] Team 未配置时静默跳过
- [ ] Team 启用时正确注入 MultiPoolRetriever 到 RetrievalPipeline
- [ ] Git Fallback 自动检测 `.ace/playbook.jsonl` 文件存在
- [ ] Memory 初始化流程调用 `try_bootstrap_team`

**Technical Notes:** `memx/ext/team_bootstrap.py` — 唯一知道 Team 存在的胶水层
**Dependencies:** STORY-048, STORY-049
**Points:** 5

---

#### STORY-053: 解耦验证测试套件

**Epic:** EPIC-009
**Priority:** Must Have

**User Story:**
As a MemX maintainer
I want automated tests that verify Core/Team decoupling
So that future changes don't accidentally introduce coupling

**Acceptance Criteria:**
- [ ] CI 静态检查：`memx/core/` 中无 `from memx.team` 或 `import memx.team`
- [ ] 测试：`pip install memx`（无 team extra）→ Core 功能正常
- [ ] 测试：Team 功能禁用时所有 Core 测试 100% 通过
- [ ] 测试：删除 `memx/team/` 后 Core 行为不变
- [ ] 集成到 CI Pipeline

**Technical Notes:** `tests/unit/test_decoupling.py` + CI workflow
**Dependencies:** STORY-048, STORY-052
**Points:** 3

---

### EPIC-010: Git Fallback 团队记忆 (25 points, 5 stories)

---

#### STORY-054: 实现 GitFallbackStorage — JSONL 只读加载

**Epic:** EPIC-010
**Priority:** Should Have

**User Story:**
As a team member
I want to `git clone` a repo and automatically get team knowledge
So that I benefit from team experience without any setup

**Acceptance Criteria:**
- [ ] `GitFallbackStorage` 实现 `StorageBackend` Protocol
- [ ] 正确解析 `.ace/playbook.jsonl`（每行一个 TeamBullet JSON）
- [ ] 首行 Header 解析 Embedding 模型指纹
- [ ] 模型指纹不匹配时降级为纯关键词检索（WARNING 日志）
- [ ] 严格只读：无任何写入 `.ace/playbook.jsonl` 的代码路径
- [ ] 文件不存在时返回空结果（不报错）

**Technical Notes:** `memx/team/git_storage.py`
**Dependencies:** STORY-048
**Points:** 5

---

#### STORY-055: 实现 Git Fallback 向量缓存

**Epic:** EPIC-010
**Priority:** Should Have

**User Story:**
As a developer
I want vector search on Git Fallback knowledge
So that semantic matching works for team knowledge too

**Acceptance Criteria:**
- [ ] 首次加载时自动生成 `.ace/playbook.vec`（gitignored）
- [ ] 向量缓存与 playbook.jsonl 的修改时间比较，过期时自动重建
- [ ] 使用 ONNXEmbedder（如可用）或跳过向量生成
- [ ] `.ace/.gitignore` 自动创建/更新，包含 `playbook.vec` 和 `playbook.cache`
- [ ] 向量缓存加载后常驻内存，后续检索零磁盘 I/O

**Technical Notes:** `memx/team/git_storage.py` 的向量缓存子模块
**Dependencies:** STORY-054, ONNXEmbedder (STORY-036)
**Points:** 5

---

#### STORY-056: 实现 MultiPoolRetriever + Shadow Merge

**Epic:** EPIC-010
**Priority:** Should Have

**User Story:**
As a developer
I want Local and Team search results merged intelligently
So that I get the best knowledge from both pools

**Acceptance Criteria:**
- [ ] `MultiPoolRetriever` 实现 `StorageBackend` Protocol
- [ ] 并行查询 Local + Team Pool
- [ ] Shadow Merge: Local boost ×1.5, Team boost ×1.0
- [ ] `enforcement: "mandatory"` 的 TeamBullet 跳过加权直接优先
- [ ] Incompatible Tags 冲突判定：标签互斥→保留高分，无互斥+相似度≥0.8→互补保留两条
- [ ] 兜底：无 incompatible_tags 的旧数据用相似度≥0.95 判定冲突
- [ ] Shadow Merge 延迟 < 5ms（纯内存计算）
- [ ] Team Pool 查询失败时静默降级，仅返回 Local 结果

**Technical Notes:** `memx/team/merger.py`
**Dependencies:** STORY-048, STORY-050
**Points:** 8

---

#### STORY-057: 实现读时去重 + playbook.cache

**Epic:** EPIC-010
**Priority:** Should Have

**User Story:**
As a developer
I want Git Fallback knowledge to be automatically deduplicated on load
So that redundant entries don't pollute search results

**Acceptance Criteria:**
- [ ] 首次加载 playbook.jsonl 时执行一次性语义去重
- [ ] 去重结果缓存到 `.ace/playbook.cache`（gitignored）
- [ ] 后续加载直接使用缓存，跳过去重计算
- [ ] 缓存过期检测（playbook.jsonl 修改时间变化时重建）
- [ ] 日常检索零开销

**Technical Notes:** `memx/team/git_storage.py`
**Dependencies:** STORY-054
**Points:** 3

---

#### STORY-058: Git Fallback 端到端集成测试

**Epic:** EPIC-010
**Priority:** Should Have

**User Story:**
As a QA engineer
I want comprehensive integration tests for Git Fallback
So that the feature works reliably end-to-end

**Acceptance Criteria:**
- [ ] 创建测试用 `.ace/playbook.jsonl` 夹具
- [ ] 测试：search 返回 Local + Git Fallback 合并结果
- [ ] 测试：Shadow Merge 正确应用 layer boost
- [ ] 测试：mandatory Bullet 正确优先
- [ ] 测试：playbook.jsonl 不存在时纯 Local 结果
- [ ] 测试：模型指纹不匹配时降级为关键词检索
- [ ] 性能测试：Team 检索增量 < 40ms

**Technical Notes:** `tests/integration/test_team_retrieval.py`, `tests/performance/test_team_search_latency.py`
**Dependencies:** STORY-054~057
**Points:** 4

---

### EPIC-011: Federation Mode MVP (39 points, 9 stories)

---

#### STORY-059: 实现 TeamCacheStorage

**Epic:** EPIC-011
**Priority:** Should Have

**User Story:**
As a team member with Federation Mode
I want team knowledge cached locally
So that search is fast without remote calls

**Acceptance Criteria:**
- [ ] `TeamCacheStorage` 实现 `StorageBackend` Protocol
- [ ] 缓存存储在 `~/.ace/team_cache/{team_id}/`
- [ ] 支持向量检索（内存索引）+ 关键词检索
- [ ] 缓存上限 `cache_max_bullets`（默认 2000），按 effective_score 保留 Top-N
- [ ] 缓存为空时返回空结果（不报错）

**Technical Notes:** `memx/team/cache_storage.py`
**Dependencies:** STORY-048, STORY-050
**Points:** 5

---

#### STORY-060: 实现 AceSyncClient — 拉取接口

**Epic:** EPIC-011
**Priority:** Should Have

**User Story:**
As a team member
I want to sync team knowledge from the server
So that I have up-to-date team knowledge locally

**Acceptance Criteria:**
- [ ] `pull_index(since, tags)` 增量拉取 Bullet 索引
- [ ] `fetch_bullets(ids)` 获取完整 Bullet 数据（含向量）
- [ ] `pull_taxonomy()` 拉取 Tag Taxonomy
- [ ] HTTP 客户端使用 httpx（支持 async）
- [ ] 支持 API Key 和 Bearer Token 认证
- [ ] 网络超时和重试配置
- [ ] Server 不可达时抛出可捕获异常（由调用方决定降级）

**Technical Notes:** `memx/team/sync_client.py`
**Dependencies:** STORY-049 (TeamConfig)
**Points:** 5

---

#### STORY-061: 实现 Team Cache 同步流程

**Epic:** EPIC-011
**Priority:** Should Have

**User Story:**
As a developer
I want team cache to sync automatically at session start
So that I always have recent team knowledge

**Acceptance Criteria:**
- [ ] Session Start 时后台异步拉取增量（`updated_at` 差分）
- [ ] 同步不阻塞用户操作（后台线程/asyncio.create_task）
- [ ] 定时刷新（默认每 1 小时，可配置）
- [ ] 同步状态持久化到 `sync_state.json`（last_sync_timestamp）
- [ ] 首次同步为全量拉取
- [ ] Server 不可达时使用上次缓存快照

**Technical Notes:** `memx/team/cache_storage.py` sync 方法
**Dependencies:** STORY-059, STORY-060
**Points:** 5

---

#### STORY-062: 实现墓碑机制 + Full Sync Check

**Epic:** EPIC-011
**Priority:** Should Have

**User Story:**
As a developer
I want deleted team knowledge to be properly cleaned up
So that my cache doesn't contain stale entries

**Acceptance Criteria:**
- [ ] 服务端删除 → 同步时接收 `status: tombstone` 记录
- [ ] 墓碑记录保留 90 天后清理
- [ ] `last_sync_timestamp` 早于墓碑清理时间 → 强制全量 ID 校验
- [ ] Full Sync Check 删除本地多余 Bullet
- [ ] 墓碑清理不影响缓存容量计算

**Technical Notes:** `memx/team/cache_storage.py`
**Dependencies:** STORY-061
**Points:** 3

---

#### STORY-063: 实现 Redactor 团队脱敏引擎

**Epic:** EPIC-011
**Priority:** Should Have

**User Story:**
As a privacy-conscious team member
I want my knowledge sanitized before sharing with the team
So that sensitive data never reaches the team pool

**Acceptance Criteria:**
- [ ] L1 确定性脱敏：复用 Core PrivacySanitizer + Team 扩展规则（custom_patterns）
- [ ] L2 用户审核：展示脱敏后内容给用户确认（**不可跳过**）
- [ ] L3 LLM 泛化（可选）：`redactor.llm_generalize = true` 时启用
- [ ] 脱敏结果支持附加 `context_summary`
- [ ] 单元测试覆盖各种敏感信息格式

**Technical Notes:** `memx/team/redactor.py`
**Dependencies:** PrivacySanitizer (STORY-011)
**Points:** 5

---

#### STORY-064: 实现 Nominator 提名流水线

**Epic:** EPIC-011
**Priority:** Should Have

**User Story:**
As a developer
I want my high-quality local knowledge automatically suggested for team sharing
So that the team benefits from my discoveries

**Acceptance Criteria:**
- [ ] 自动检测候选：`recall_count > min_recall_count` 且 `instructivity_score > min_score`（可配置）
- [ ] 频率控制：每会话最多 `max_prompts_per_session` 次弹窗（默认 1）
- [ ] 静默模式：`silent=true` 时不弹窗，`ace nominate list` 主动查看
- [ ] Session 结束时批量汇总待提名列表
- [ ] 编排 Redactor → 用户确认 → AceSyncClient.nominate_bullet 上传
- [ ] 用户可标记永久忽略特定 Bullet

**Technical Notes:** `memx/team/nominator.py`
**Dependencies:** STORY-063, STORY-060
**Points:** 5

---

#### STORY-065: 实现 subscribed_tags 订阅过滤

**Epic:** EPIC-011
**Priority:** Should Have

**User Story:**
As a frontend developer
I want to subscribe to #frontend #react tags
So that I only get relevant team knowledge

**Acceptance Criteria:**
- [ ] `subscribed_tags` 配置项支持标签列表
- [ ] 同步时按 tags 过滤请求
- [ ] 未订阅任何标签时拉取全量（受 cache_max_bullets 限制）
- [ ] 修改订阅后下次同步自动调整缓存
- [ ] 不同团队 (`team_id`) 的缓存路径隔离

**Technical Notes:** `memx/team/cache_storage.py`, `memx/team/sync_client.py`
**Dependencies:** STORY-061
**Points:** 3

---

#### STORY-066: 实现 Team CLI 命令

**Epic:** EPIC-011
**Priority:** Should Have

**User Story:**
As a developer
I want CLI commands to manage team features
So that I can check status, sync, and nominate from terminal

**Acceptance Criteria:**
- [ ] `ace team status` — 显示模式、缓存数量、上次同步时间、订阅标签
- [ ] `ace team sync` — 强制增量同步
- [ ] `ace nominate list` — 列出待提名候选
- [ ] `ace nominate submit <id>` — 手动提名（经过 Redactor）
- [ ] Team 未启用时友好提示
- [ ] 支持 `--json` 输出格式

**Technical Notes:** `memx/team/cli.py`
**Dependencies:** STORY-059, STORY-064
**Points:** 3

---

#### STORY-067: Federation MVP 端到端集成测试

**Epic:** EPIC-011
**Priority:** Should Have

**User Story:**
As a QA engineer
I want comprehensive tests for Federation Mode
So that sync, search, and nomination work reliably

**Acceptance Criteria:**
- [ ] Mock AceSyncServer 实现（pytest fixture）
- [ ] 测试：增量同步正确拉取新 Bullet
- [ ] 测试：墓碑机制正确清理删除的 Bullet
- [ ] 测试：search 返回 Local + Team Cache 合并结果
- [ ] 测试：提名流程端到端（detect → redact → upload）
- [ ] 测试：Server 不可达时降级为纯 Local
- [ ] 性能测试：含 Team Cache 检索 < 100ms

**Technical Notes:** `tests/integration/test_team_retrieval.py`, `tests/unit/team/`
**Dependencies:** STORY-059~066
**Points:** 5

---

### EPIC-012: Team 治理与高级功能 (26 points, 6 stories)

---

#### STORY-068: 实现 AceSyncClient — 推送接口

**Epic:** EPIC-012
**Priority:** Could Have

**User Story:**
As a team member
I want to vote on and correct team knowledge
So that the team knowledge base improves over time

**Acceptance Criteria:**
- [ ] `nominate_bullet(sanitized_bullet)` 上传到 Staging
- [ ] `cast_vote(bullet_id, "up"|"down")` 投票
- [ ] `propose_supersede(origin_id, new_bullet)` 提交纠正
- [ ] 支持 `priority: "urgent"` 字段
- [ ] 错误处理：网络失败时返回可重试错误

**Technical Notes:** `memx/team/sync_client.py` 扩展
**Dependencies:** STORY-060
**Points:** 5

---

#### STORY-069: 实现三层审核治理逻辑（客户端）

**Epic:** EPIC-012
**Priority:** Could Have

**User Story:**
As a team curator
I want different approval rules for different knowledge
So that sensitive knowledge gets proper review

**Acceptance Criteria:**
- [ ] 客户端标记：score ≥ 90 + 非敏感标签 → auto_approve 建议
- [ ] 敏感标签（security, architecture, mandatory）→ 标记为 curator_required
- [ ] `ace upvote/downvote <id>` CLI 命令
- [ ] 投票结果影响 TeamBullet 的 effective_score（本地缓存中）
- [ ] 不采纳 AI 执行结果作为投票信号

**Technical Notes:** `memx/team/nominator.py` 扩展, `memx/team/cli.py`
**Dependencies:** STORY-068
**Points:** 5

---

#### STORY-070: 实现 Supersede 知识纠正流程

**Epic:** EPIC-012
**Priority:** Could Have

**User Story:**
As a developer who found wrong team knowledge
I want to propose a correction
So that the entire team gets the updated version

**Acceptance Criteria:**
- [ ] Reflector 检测到 Local 纠正了 Team Pool 来源的知识
- [ ] 提示用户是否提交 Supersede Proposal
- [ ] 拒绝提交 → 仅 Local Pool 保留（Shadow Merge 覆盖）
- [ ] 同意提交 → 经 Redactor 脱敏 → `propose_supersede(origin_id, new_bullet)`
- [ ] 支持 `priority: "urgent"` 级别
- [ ] Team Bullet 更新后，检测到本地存在旧版覆盖 → 通知用户重新评估

**Technical Notes:** `memx/team/nominator.py`, Reflector 扩展接口
**Dependencies:** STORY-068, STORY-056 (Shadow Merge)
**Points:** 5

---

#### STORY-071: 实现 Tag Taxonomy 标签归一化

**Epic:** EPIC-012
**Priority:** Could Have

**User Story:**
As a team member
I want consistent tag naming across the team
So that knowledge is organized and discoverable

**Acceptance Criteria:**
- [ ] `pull_taxonomy()` 从 Server 下载最新 Taxonomy
- [ ] Taxonomy 缓存到 `~/.ace/team_cache/{team_id}/taxonomy.json`
- [ ] Reflector 蒸馏时标签对齐 Taxonomy（如可用）
- [ ] 提供预设 Taxonomy 模板（rust, python, react, security, architecture, testing 等）
- [ ] Git Fallback 支持项目级 `.ace/taxonomy.json`
- [ ] 兜底：无 Taxonomy 匹配时向量相似度 ≥ 0.9 视为同一标签

**Technical Notes:** `memx/team/sync_client.py`, Reflector 扩展
**Dependencies:** STORY-060
**Points:** 4

---

#### STORY-072: 实现 Mandatory 逃生舱

**Epic:** EPIC-012
**Priority:** Could Have

**User Story:**
As a developer on a legacy project
I want to override mandatory team rules locally
So that my project isn't blocked by rules that don't apply

**Acceptance Criteria:**
- [ ] `mandatory_overrides` 配置支持 `bullet_id`, `reason`, `expires`
- [ ] `reason` 和 `expires` 为必填字段
- [ ] 过期后自动恢复 mandatory 行为
- [ ] 偏离时 Generator 注入偏离提示到上下文
- [ ] Federation Mode 下偏离事件审计上报

**Technical Notes:** `memx/team/merger.py`, `memx/team/config.py`
**Dependencies:** STORY-056 (MultiPoolRetriever)
**Points:** 3

---

#### STORY-073: Team 治理集成测试

**Epic:** EPIC-012
**Priority:** Could Have

**User Story:**
As a QA engineer
I want governance features thoroughly tested
So that voting, supersede, and taxonomy work correctly

**Acceptance Criteria:**
- [ ] 测试：投票正确调整 effective_score
- [ ] 测试：Supersede 流程端到端
- [ ] 测试：Taxonomy 对齐正确应用
- [ ] 测试：Mandatory 逃生舱过期后自动恢复
- [ ] 测试：敏感标签强制 curator 审核标记

**Technical Notes:** `tests/unit/team/`, `tests/integration/`
**Dependencies:** STORY-068~072
**Points:** 4

---

## Sprint Allocation

### Sprint 1 (Week 1-2): 基础地基 + Reflector — 60/63 points

*(与 v1.0 计划完全一致，详见 sprint-plan-memx-2026-02-27.md)*

---

### Sprint 2 (Week 3-4): Decay + Curator + Generator — 62/63 points ✅ COMPLETED

*(与 v1.0 计划完全一致)*

---

### Sprint 3 (Week 5-6): Integration + ONNX + Daemon — 56/63 points

*(与 v1.0 计划完全一致)*

---

### Sprint 4 (Week 7-8): 高级功能 + PyPI 发布 — 22/63 points

*(与 v1.0 计划完全一致。Sprint 4 的 41 点 buffer 留给 Bug 修复和文档)*

---

### Sprint 5 (Week 9-10): P0 解耦重构 + P1 Git Fallback — 55/63 points

**Goal:** 完成 Core/Team 解耦重构（P0）和 Git Fallback 完整功能（P1）。Sprint 结束时，仓库中有 `.ace/playbook.jsonl` 的项目可自动加载团队知识，search 返回 Local + Team 合并结果。

**Stories:**

| Story | Title | Points | Priority | Epic |
|-------|-------|--------|----------|------|
| STORY-051 | BulletMetadata +schema_version +incompatible_tags | 2 | Must | 009 |
| STORY-048 | 重构 memx/ → memx/core/ | 5 | Must | 009 |
| STORY-049 | TeamConfig 独立配置 | 3 | Must | 009 |
| STORY-050 | TeamBullet 数据模型 | 3 | Must | 009 |
| STORY-052 | ext/team_bootstrap.py 条件注入 | 5 | Must | 009 |
| STORY-053 | 解耦验证测试 | 3 | Must | 009 |
| STORY-054 | GitFallbackStorage JSONL 加载 | 5 | Should | 010 |
| STORY-055 | Git Fallback 向量缓存 | 5 | Should | 010 |
| STORY-056 | MultiPoolRetriever + Shadow Merge | 8 | Should | 010 |
| STORY-057 | 读时去重 + playbook.cache | 3 | Should | 010 |
| STORY-058 | Git Fallback 集成测试 | 4 | Should | 010 |

**Total:** 46 points / 63 capacity (73%)

**Buffer:** 17 点用于：
- P0 重构可能的 import 修复
- Shadow Merge 性能优化
- Bug 修复

**Execution Order:**
1. STORY-051（Bullet 字段扩展）— 独立，可先行
2. STORY-048（包重构）— 最大影响面，尽早完成
3. STORY-049 + STORY-050（Team 数据模型）— 并行
4. STORY-052（team_bootstrap）— 依赖 048+049
5. STORY-053（解耦测试）— 依赖 048+052
6. STORY-054 + STORY-057（Git 存储 + 去重）— 并行
7. STORY-055（向量缓存）— 依赖 054
8. STORY-056（Shadow Merge）— 依赖 050
9. STORY-058（集成测试）— 最后

**Risks:**
- STORY-048（包重构）影响面大，需要仔细更新所有 import path
- STORY-056（Shadow Merge，8 点）是最大单体 Story，冲突判定逻辑复杂

**Milestone:** Sprint 5 结束 = **P0+P1 完成，Git Fallback 可用** → 可发布 `memx[team]` 预览版

---

### Sprint 6 (Week 11-12): P2 Federation Mode MVP — 44/63 points

**Goal:** 完成 Federation Mode 最小可用版本。Sprint 结束时，配置 `server_url` 后 Team Cache 自动同步，高质量 Local 知识可提名到 Team Pool。

**Stories:**

| Story | Title | Points | Priority | Epic |
|-------|-------|--------|----------|------|
| STORY-059 | TeamCacheStorage | 5 | Should | 011 |
| STORY-060 | AceSyncClient 拉取接口 | 5 | Should | 011 |
| STORY-061 | Team Cache 同步流程 | 5 | Should | 011 |
| STORY-062 | 墓碑机制 + Full Sync Check | 3 | Should | 011 |
| STORY-063 | Redactor 脱敏引擎 | 5 | Should | 011 |
| STORY-064 | Nominator 提名流水线 | 5 | Should | 011 |
| STORY-065 | subscribed_tags 过滤 | 3 | Should | 011 |
| STORY-066 | Team CLI 命令 | 3 | Should | 011 |
| STORY-067 | Federation MVP 集成测试 | 5 | Should | 011 |
| STORY-068 | AceSyncClient 推送接口 | 5 | Could | 012 |

**Total:** 44 points / 63 capacity (70%)

**Buffer:** 19 点用于：
- 同步机制调试
- Mock Server 维护
- Bug 修复

**Execution Order:**
1. STORY-060（SyncClient 拉取）— 网络层基础
2. STORY-059（TeamCacheStorage）— 本地缓存
3. STORY-061（同步流程）— 依赖 059+060
4. STORY-062（墓碑）— 依赖 061
5. STORY-065（tags 过滤）— 依赖 061
6. STORY-063（Redactor）— 可并行
7. STORY-064（Nominator）— 依赖 063+060
8. STORY-068（SyncClient 推送）— 依赖 060
9. STORY-066（CLI）— 依赖 059+064
10. STORY-067（集成测试）— 最后

**Risks:**
- 需要 Mock ACE Sync Server 进行测试（无真实 Server 可用）
- 异步同步流程的错误处理和竞态条件

**Milestone:** Sprint 6 结束 = **P2 Federation MVP 完成** → 可配合 ACE Sync Server 使用

---

### Sprint 7 (Week 13-14): P3 治理 + 收尾 — 21/63 points

**Goal:** 完成 Team 治理功能（投票、Supersede、Taxonomy、Mandatory 逃生舱），全部 Team Memory 功能完成。

**Stories:**

| Story | Title | Points | Priority | Epic |
|-------|-------|--------|----------|------|
| STORY-069 | 三层审核治理（客户端） | 5 | Could | 012 |
| STORY-070 | Supersede 知识纠正 | 5 | Could | 012 |
| STORY-071 | Tag Taxonomy 归一化 | 4 | Could | 012 |
| STORY-072 | Mandatory 逃生舱 | 3 | Could | 012 |
| STORY-073 | 治理集成测试 | 4 | Could | 012 |

**Total:** 21 points / 63 capacity (33%)

**Buffer:** 42 点用于：
- Sprint 5-6 遗留 Bug 修复
- 文档完善（Team Memory 使用指南）
- 性能调优（Team 检索 < 100ms 验证）
- ACE Sync Server Lite 参考实现原型（Docker Compose）
- `memx[team]` 正式发布到 PyPI

**Risks:**
- Supersede 流程依赖 Server 端支持，可能需要 Mock 或简化
- 治理功能为 Could Have，可按需裁剪

**Milestone:** Sprint 7 结束 = **Team Memory 完整版发布**

---

## Team Memory Epic Traceability

| Epic ID | Epic Name | Stories | Total Points | Sprint | Priority |
|---------|-----------|---------|--------------|--------|----------|
| EPIC-009 | Core/Team 解耦 (P0) | 048-053 | 21 | Sprint 5 | Must |
| EPIC-010 | Git Fallback (P1) | 054-058 | 25 | Sprint 5 | Should |
| EPIC-011 | Federation MVP (P2) | 059-067 | 39 | Sprint 6 | Should |
| EPIC-012 | 治理+高级 (P3) | 068-073 | 26 | Sprint 6-7 | Could |
| **Team Total** | | **26 stories** | **111 pts** | **3 sprints** | |

---

## Team Memory FR Coverage

| FR ID | FR Name | Story | Sprint |
|-------|---------|-------|--------|
| FR-025 | Core/Team 解耦 | STORY-048, 052, 053 | 5 |
| FR-026 | StorageBackend 扩展 | STORY-052, 056 | 5 |
| FR-027 | Git Fallback | STORY-054, 055, 057 | 5 |
| FR-028 | TeamBullet 模型 | STORY-050, 051 | 5 |
| FR-029 | Shadow Merge | STORY-056 | 5 |
| FR-030 | Mandatory 逃生舱 | STORY-072 | 7 |
| FR-031 | Team Cache 同步 | STORY-059, 061, 062 | 6 |
| FR-032 | 提名流水线 | STORY-064 | 6 |
| FR-033 | Redactor 脱敏 | STORY-063 | 6 |
| FR-034 | 三层审核治理 | STORY-069 | 7 |
| FR-035 | Team Supersede | STORY-070 | 7 |
| FR-036 | Tag Taxonomy | STORY-071 | 7 |
| FR-037 | 订阅与分发 | STORY-065 | 6 |

**Coverage: 13/13 Team FRs (100%)**

---

## Combined Project Summary

### Full Traceability

| Epic ID | Epic Name | Stories | Points | Sprint | Phase |
|---------|-----------|---------|--------|--------|-------|
| EPIC-001 | Bullet + Config + 兼容 | 001-007 | 28 | 1 | Core |
| EPIC-002 | Reflector 蒸馏引擎 | 008-016 | 38 | 1-2 | Core |
| EPIC-003 | Curator 去重引擎 | 017-019 | 13 | 2 | Core |
| EPIC-004 | Decay 衰退引擎 | 020-022 | 14 | 2 | Core |
| EPIC-005 | Generator 检索引擎 | 023-031 | 33 | 2-3 | Core |
| EPIC-006 | Integration Layer | 032-035 | 18 | 3 | Core |
| EPIC-007 | ONNX + Daemon | 036-040 | 25 | 3 | Core |
| EPIC-008 | CLI + 发布 | 041-047 | 30 | 3-4 | Core |
| EPIC-009 | Core/Team 解耦 (P0) | 048-053 | 21 | 5 | Team |
| EPIC-010 | Git Fallback (P1) | 054-058 | 25 | 5 | Team |
| EPIC-011 | Federation MVP (P2) | 059-067 | 39 | 6 | Team |
| EPIC-012 | 治理+高级 (P3) | 068-073 | 26 | 6-7 | Team |
| **Total** | | **72 stories** | **310 pts** | **7 sprints** | |

### MoSCoW Distribution

| Priority | Core Stories | Team Stories | Total Stories | Total Points |
|----------|------------|-------------|--------------|--------------|
| Must Have | 31 | 6 | 37 | 182 |
| Should Have | 10 | 15 | 25 | 99 |
| Could Have | 5 | 5 | 10 | 29 |
| **Total** | **46** | **26** | **72** | **310** |

---

## Risks and Mitigation (Team Memory additions)

**High:**
- **STORY-048 包重构影响面大** — 所有 import path 需更新
  - Mitigation: 使用 `memx/__init__.py` 重导出维持向后兼容；Sprint 5 初期专注完成
- **无 ACE Sync Server** — Federation Mode 测试依赖 Mock Server
  - Mitigation: 创建 pytest fixture Mock Server；Sprint 7 buffer 时间做 Lite 原型

**Medium:**
- **Shadow Merge 性能** — incompatible_tags 冲突判定可能有边界情况
  - Mitigation: 充分的单元测试 + 模糊测试
- **Team Cache 容量 2000 条** — 大团队可能不够
  - Mitigation: 按 subscribed_tags 分片；Post-Inference 异步补充

**Low:**
- **Git Fallback JSONL 大文件** — > 10000 条时加载慢
  - Mitigation: 向量缓存 + 读时去重缓存补偿

---

## Dependencies (Team Memory additions)

**内部依赖：**
- Sprint 5 依赖 Sprint 1-4 Core 代码完成
- Sprint 6 依赖 Sprint 5 的解耦结构 + StorageBackend Protocol
- Sprint 7 依赖 Sprint 6 的 AceSyncClient

**外部依赖：**
- httpx PyPI 包（Sprint 6）
- ACE Sync Server API 规范（Sprint 6 前定义）
- ACE Sync Server Lite 原型（Sprint 7，可选）

---

## Definition of Done

与 v1.0 一致，额外增加：
- [ ] **Team 解耦验证**：Team 功能禁用时 Core 测试 100% 通过
- [ ] **CI 静态检查**：Core 不 import Team
- [ ] **性能验证**：Team 检索增量 < 40ms，Shadow Merge < 5ms

---

## Next Steps

**Immediate:** 完成 Sprint 3-4（Core），然后进入 Sprint 5（Team Memory）

**Sprint 5 推荐执行顺序：**
1. STORY-051 (Bullet 字段扩展) — 最小改动，可先行
2. STORY-048 (包重构) — 最大影响面，尽早完成
3. STORY-049 + 050 (Team 模型) — 并行开发
4. STORY-052 (team_bootstrap) → STORY-053 (解耦测试)
5. STORY-054~058 (Git Fallback 全部)

**里程碑：**
- Sprint 5 结束 → `memx[team]` 预览版（Git Fallback 可用）
- Sprint 6 结束 → Federation Mode MVP 可用
- Sprint 7 结束 → Team Memory 完整版发布

**Sprint cadence:**
- Sprint length: 2 weeks
- Sprint planning: Day 1
- Sprint review: Last day
- Sprint retrospective: Last day

---

**This plan was created using BMAD Method v6 - Phase 4 (Implementation Planning)**

*To continue: Run `/dev-story STORY-048` to begin the Core/Team decoupling refactoring.*
