# ACE Team Memory Architecture: Collaborative Context Engine

**Date:** 2026-03-08
**Version:** 2.1
**Status:** Draft / Conceptual Architecture Design
**Scope:** Universal AI memory framework (Platform-agnostic)
**Changelog:**
- v2.1 - Added: Decoupling Architecture (Section 4), Risk Analysis & Mitigations (Section 11), Implementation Priority (Section 10). Team memory fully decoupled from local memory.
- v2.0 - Major restructure: removed phased rollout (Phase 1/2/3), unified as "Federation Mode + Git Fallback". Git demoted to offline fallback only. All v1.x review feedback preserved in design.
- v1.1~v1.4 - See Review Feedback Log at end of document.

---

## 1. Executive Summary

当前的 ACE (Adaptive Context Engine) 框架主要解决**单机单用户 (Single-Player, Local-First)** 的自适应记忆问题。本架构将其扩展为**团队级记忆池 (Team Memory)**。

**核心设计理念**：

1. **联邦式架构 (Federated Architecture)**：用户的本地环境永远是"主脑"，团队记忆池作为可选的"云端外脑"存在。用户完全掌握数据主权。
2. **充分解耦 (Full Decoupling)**：Team Memory 是 Local Memory 的**纯可选扩展层**。移除 Team 功能后，Local Memory 的所有代码路径、数据结构、性能特征必须与未引入 Team 方案前**完全一致**。详见第 4 节。

**两种运行模式**：

| 模式 | 基础设施 | 能力 | 适用场景 |
|------|----------|------|----------|
| **Federation Mode（主方案）** | ACE Sync Server | 双向贡献、治理、纠正、P2P 验证 | 有服务器的团队/企业 |
| **Git Fallback（退行备选）** | 仅 Git 仓库 | 只读共享项目记忆，无贡献/治理 | 无服务器、离线、个人开源项目 |

---

## 2. 记忆池模型 (Memory Pool Model)

### 2.1 三级记忆池

**与 Universal Solution 的兼容说明**：Universal 原有 `scope: "project:{name}"` 存储于本地混合存储（元数据文件 + 向量索引），这是**个人对某项目的私有记忆**。本方案新增的 Team Pool 是**团队共享记忆**，两者互补而非替代。现有用户的 Playbook 无需任何迁移。

| 层级 | scope 值 | 存储位置 | 性质 |
|------|----------|----------|------|
| Local (全局) | `global` | `~/.ace/{product}/` (元数据 + 向量索引) | 个人私有（Universal 原有） |
| Local (项目) | `project:{name}` | `~/.ace/{product}/` (元数据 + 向量索引) | 个人项目记忆（Universal 原有） |
| Team/Org | `team:{id}` | ACE Sync Server 或 Git Fallback | 团队共享记忆（**新增**） |

### 2.2 Schema 版本控制

```typescript
interface Bullet {
  schema_version: number;  // 1 (Universal), 2 (Team 扩展)
  // ...existing fields...
  incompatible_tags?: string[];  // 互斥标签（v2 新增）
}

interface TeamBullet extends Bullet {
  author_id: string;              // 假名标识（GDPR 友好）
  origin_id?: string;             // Supersede 提议的起源 Bullet ID
  enforcement: "mandatory" | "suggestion";
  upvotes: number;
  downvotes: number;
  status: "pending" | "approved" | "archived" | "tombstone";
  deleted_at?: Date;
  context_summary?: string;       // 贡献者附加的业务场景摘要
}
```

| 版本 | 兼容策略 |
|------|----------|
| v1 → v2 读取 | 自动填充默认值（`enforcement="suggestion"`, `upvotes=0`, `status="approved"`） |
| v2 → v1 写入 | `serde(flatten)` 保留未知字段，不静默丢弃 |

---

## 3. 解耦架构 (Decoupling Architecture)

### 3.1 核心原则

Team Memory 与 Local Memory 之间必须做到**充分解耦**。具体要求：

| 原则 | 要求 | 验证方法 |
|------|------|----------|
| **零侵入** | 不修改 Local Memory 的任何现有接口签名、数据结构、存储格式 | Local Pool 的全部现有测试在 Team 功能禁用时 100% 通过，零改动 |
| **可剥离** | Team 功能可完整移除（删除代码 / 不安装可选依赖）而不影响 Local Memory | `pip install memx`（不含 team extra）运行正常 |
| **无感降级** | Team Server 不可达 / Team Cache 为空时，系统行为与纯 Local 模式完全一致 | 断网测试：延迟、结果与纯 Local 无差异 |
| **独立生命周期** | Team 组件可独立升级、独立配置、独立测试 | Team 模块有独立的 test suite 和 changelog |

### 3.2 架构边界

```text
┌──────────────────────────────────────────────────┐
│                   ACE Engine                      │
│                                                   │
│   ┌───────────────────────────────────────────┐   │
│   │         Core (不可修改)                     │   │
│   │  Reflector │ Generator │ Curator │ Decay   │   │
│   │  Storage   │ Config    │ Types   │ Privacy │   │
│   └──────────────────┬────────────────────────┘   │
│                      │ 扩展接口 (Extension Points) │
│   ┌──────────────────▼────────────────────────┐   │
│   │         Team Layer (可选扩展)               │   │
│   │  TeamCacheStorage │ GitFallbackStorage     │   │
│   │  SyncClient       │ Redactor │ Nominator   │   │
│   │  TeamMerger       │ TeamConfig             │   │
│   └───────────────────────────────────────────┘   │
│                                                   │
└──────────────────────────────────────────────────┘
```

### 3.3 扩展接口设计

Core 不依赖 Team Layer，Team Layer 通过以下**扩展接口**注入 Core：

```python
# Core 定义抽象接口（已有，无需修改）
class StorageBackend(Protocol):
    def search(self, query: str, top_k: int) -> list[Bullet]: ...

# Team Layer 实现额外的 StorageBackend
class TeamCacheStorage(StorageBackend):
    """Read-only storage backed by local team cache."""
    ...

class GitFallbackStorage(StorageBackend):
    """Read-only storage backed by .ace/playbook.jsonl."""
    ...
```

Generator 的多路检索通过**组合模式**实现，而非修改 Generator 内部逻辑：

```python
# Team Layer 提供的组合器（Generator 外部）
class MultiPoolRetriever:
    def __init__(self, local: StorageBackend, team: StorageBackend | None = None):
        self.pools = [local]
        if team:
            self.pools.append(team)

    def search(self, query: str, top_k: int) -> list[Bullet]:
        results = []
        for pool in self.pools:
            results.extend(pool.search(query, top_k))
        return self._shadow_merge(results)
```

### 3.4 依赖方向

```
Core ←── Team Layer ←── Sync Server
  ↑ 不可反转          ↑ 不可反转
```

- Core **不 import** Team Layer 的任何模块
- Team Layer **不 import** Sync Server 的内部实现
- 依赖注入在**应用初始化层**（`__init__.py` / `app.py`）完成

### 3.5 包结构

```
memx/
├── core/                    # 现有代码，零修改
│   ├── engines/
│   ├── storage/
│   ├── types.py
│   └── config.py
├── team/                    # 新增可选包
│   ├── __init__.py
│   ├── cache_storage.py     # TeamCacheStorage
│   ├── git_storage.py       # GitFallbackStorage
│   ├── sync_client.py       # AceSyncClient
│   ├── merger.py            # Shadow Merge + Conflict Detection
│   ├── redactor.py          # Sanitization Pipeline
│   ├── nominator.py         # Promotion Pipeline
│   └── config.py            # TeamConfig (独立于 core config)
└── ext/                     # 初始化 / 胶水层
    └── team_bootstrap.py    # 检测 team 配置，注入 Team Layer
```

### 3.6 数据隔离

| 维度 | Local Memory | Team Memory |
|------|-------------|-------------|
| 存储路径 | `~/.ace/{product}/` | `~/.ace/team_cache/{team_id}/` |
| 数据格式 | 现有混合存储（元数据 + 向量索引） | 独立缓存格式（Team 自行管理） |
| 生命周期 | Decay 引擎管理 | Team Cache TTL + 墓碑机制 |
| Schema | `Bullet` (schema_version=1) | `TeamBullet extends Bullet` (schema_version=2) |
| 写入权限 | 读写 | **只读**（本地缓存） |

**关键约束**：Team Layer 对 Local Pool **只读不写**。Team 信息只通过 Shadow Merge 在检索结果层面与 Local 结果合并，**不会写入 Local Pool**。

---

## 4. Federation Mode（主方案）

### 4.1 拓扑结构

```text
[ 本地 ACE (User A) ]              [ 本地 ACE (User B) ]
       |         |                         |         |
       | 检索    | 贡献/纠正               | 检索    | 贡献/纠正
       v         |                         v         |
[Local Pool A]   |                  [Local Pool B]   |
                 v                                   v
         +--------------------------------------------------+
         |              ACE Sync Server                      |
         |  (Auth, API, 向量存储, RBAC, Tag Taxonomy)          |
         |                                                   |
         |  +-----------+  +--------+  +---------+          |
         |  | Team Pool |  | Staging|  | Taxonomy|          |
         |  +-----------+  +--------+  +---------+          |
         +--------------------------------------------------+
```

### 4.2 多路检索与影子合并 (Shadow Merging)

检索时 Generator 进行**多路召回**，**全部在本地完成**：

1. `Query` → 本地 Local Pool
2. `Query` → 本地 Team Cache（预缓存的团队知识）

**关键约束：Pre-Inference 阶段不做实时远程请求**。Team Cache 含完整向量数据，缓存刷新策略见 3.5。

#### 4.2.1 强制规则 (Mandatory)

标记为 `enforcement: "mandatory"` 的 Bullet（安全红线、架构强约束），**跳过加权计算直接优先**。

**Mandatory 逃生舱**（防止遗留项目死锁）：
```json
{
  "ace": { "team": { "mandatory_overrides": [
    { "bullet_id": "xxx", "reason": "遗留项目依赖 Node 14", "expires": "2026-06-30" }
  ]}}
}
```
- 必须 `reason` + `expires`，过期后自动恢复
- 引擎注入偏离提示，审计上报 Team Server

#### 4.2.2 建议规则的加权合并

```
effective_score = base_score × decay_weight × layer_boost

Layer Boost:  Local 1.5 | Team 1.0
```

#### 4.2.3 冲突 vs 互补判定 (Incompatible Tags)

高语义相似度 ≠ 冲突。通过 Reflector 蒸馏时生成的**互斥标签 (incompatible_tags)** 判定：

- **标签互斥**（A 的 `tags` ∩ B 的 `incompatible_tags` 非空）→ 矛盾冲突，保留高分
- **无标签互斥但语义相似 >= 0.8** → 互补，两条都保留
- **兜底**（无 incompatible_tags 的旧数据）→ 相似度 >= 0.95 视为冲突

**标签归一化**（防止自由文本漂移）：
- Team Server 维护中心化 **Tag Taxonomy**，客户端同步时下载最新词表
- Reflector 生成标签时强制对齐 Taxonomy
- 兜底：向量相似度 >= 0.9 视为同一标签

### 4.3 贡献流水线 (Promotion Pipeline)

```
Local Reflector 发现高质量 Bullet (recall_count > 10, score > 80)
    |
    v
确定性脱敏 (正则替换路径/凭证/IP + custom_patterns)
    |
    v
用户确认 (展示脱敏后内容 + 可选附加上下文摘要)
    |
    v
上传 Team Server → Staging 池
    |
    v
三层审核 (见 3.4)
```

**提名频率控制**：
- 每会话最多 1 次弹窗
- Session 结束时批量汇总
- 可标记永久忽略 / 静默模式（`ace nominate list` 主动查看）

**脱敏引擎 (Redactor)**：

| 层级 | 方法 | 覆盖范围 |
|------|------|----------|
| L1 确定性规则（默认） | 正则替换路径、凭证、IP、`custom_patterns` | 结构化敏感信息 |
| L2 用户审核（必须） | 展示脱敏后最终内容，不可跳过 | 自然语言中的隐含敏感信息 |
| L3 LLM 泛化（可选） | `redactor.llm_generalize = true` | 将具体经验抽象为通用规则 |

### 4.4 审查与治理 (Governance)

**三层审核机制**（防止审核瓶颈）：

| 层级 | 条件 | 处理 |
|------|------|------|
| **自动审批** | `score >= 90` + 贡献者高信誉 + 非敏感标签 | 直接入 Team Pool（初始低权重） |
| **P2P 验证** | 自动审批入池后 | 显式 `ace upvote/downvote` + Supersede 纠正信号调权。不采纳 AI 执行结果（噪音太高） |
| **人工 Curator** | 敏感标签（`security`, `architecture`, `mandatory`）或低信誉贡献者 | 必须人工审核 |

**防积压**：Staging 超 50 条 / 最早 Pending 超 7 天 → 通知 Curator。超 30 天未审核 → 自动拒绝。

### 4.5 Team Cache 同步策略

| 时机 | 动作 |
|------|------|
| **Session Start** | 后台异步拉取增量（`updated_at` 差分 + 墓碑记录） |
| **定时** | 每 1 小时检查增量 |
| **向量** | 增量同步时完整拉取向量数据 |
| **离线** | 使用上次缓存快照 |

**规模控制**：本地上限 2000 条（约 3MB 向量），按 `effective_score` 保留 Top-N。服务端按订阅 Tags 分片返回。

**墓碑机制**：服务端删除 → 软删除（`status: tombstone`），保留 90 天。客户端同步时清理。

**Full Sync Check**：`last_sync_timestamp` 早于墓碑清理时间 → 强制全量 ID 校验，删除本地多余 Bullet。

### 4.6 团队知识纠正 (Team Supersede)

```
User A 本地纠正 → Reflector 检测 Supersede 模式 → 识别来源于 Team Pool
    |
    +-- 拒绝提交 → 仅 Local Pool 保留（影子合并覆盖）
    |
    +-- 同意提交 → Supersede Proposal → Curator/自动审核
                      |
                      +-- Accept → 团队 Bullet 更新，全员下次同步获得新版本
                      +-- Reject → 提议关闭，Local 版本继续生效
```

**防止知识孤岛**：Team Bullet 被更新后，检测到本地存在旧版覆盖 → 通知用户重新评估。

### 4.7 订阅与分发

- **按标签订阅**：前端订阅 `#frontend, #react`，后端订阅 `#rust, #k8s`
- **全部在本地缓存**：`~/.ace/team_cache/{team_id}/`

---

## 5. Git Fallback（退行备选方案）

### 5.1 定位与优势

当团队**无法或不愿部署 Sync Server** 时，可使用 Git Fallback 作为降级方案。

| 优势 | 说明 |
|------|------|
| **零基础设施** | 不需要任何服务器、数据库、域名 |
| **零成本** | 无运维、无云服务费用 |
| **与现有工作流无缝** | 知识跟随代码仓库，Clone 即获得 |
| **离线完全可用** | 不依赖任何网络连接 |
| **版本可追溯** | Git 历史记录知识的每一次变更 |
| **权限复用** | 复用 Git 仓库的现有权限控制 |

### 5.2 能力边界

| 能力 | Federation Mode | Git Fallback |
|------|----------------|--------------|
| 团队知识共享 | ✅ 双向 | ⚠️ 只读（手动维护的静态文件） |
| 自动贡献流水线 | ✅ | ❌ |
| 知识纠正提议 | ✅ | ❌ |
| 审核/治理 | ✅ 三层 | ❌ 由 Git PR 审核替代 |
| 语义去重 | ✅ 自动 | ⚠️ 读时内存去重 |
| 标签归一化 | ✅ 中心化 Taxonomy | ⚠️ 项目级 taxonomy.json |
| 墓碑/同步 | ✅ | ❌ 不需要 |
| 冲突检测 | ✅ Incompatible Tags | ⚠️ 纯语义相似度兜底 |

### 5.3 工作方式

Git Fallback 将团队共享知识存储为仓库内的**只读静态文件**：

```
.ace/
├── playbook.jsonl          # Git 追踪（手动维护或 Curator 编写）
├── taxonomy.json           # 可选：标签归一化词表
├── .gitignore              # 包含: playbook.vec, playbook.cache
├── playbook.vec            # gitignored（本地生成的向量缓存）
└── playbook.cache          # gitignored（去重后的内存快照缓存）
```

**为什么 Git Fallback 使用 JSONL 而非数据库**：
Local Pool 使用混合存储（元数据文件 + 向量索引，高性能读写 + 语义检索），但 Git Fallback 刻意选择纯文本 JSONL，因为：Git 无法 diff/review 二进制文件（数据库、向量索引），而 JSONL 支持 PR Review、手动编辑、文本级合并，且零外部依赖——任何语言都能解析。向量数据由引擎在本地按需生成（gitignored 的 `.ace/playbook.vec`）。

**关键简化**：
- `.ace/playbook.jsonl` 由 Tech Lead / Curator **手动编写或从个人 Playbook 导出**，通过 Git PR 审核合入
- 引擎对 `.ace/playbook.jsonl` **只读不写**——不做 append、不做 auto-compact、不依赖 `merge=union`
- 所有 Git 冲突问题（union 局限、compact 冲突、squash merge 不兼容）**彻底消除**
- 开发者只需 `git pull` 即可获得最新团队知识，引擎自动加载

**模型指纹**：`playbook.jsonl` 首行为 Header，记录 Embedding 模型信息。不匹配时柔性降级为纯关键词检索（Graceful Degradation）。

**读时去重**：加载时一次性执行并缓存到 `.ace/playbook.cache`，日常检索零开销。

### 5.4 从 Git Fallback 升级到 Federation Mode

当团队决定部署 Sync Server 时，迁移路径：

```
1. 部署 ACE Sync Server
2. ace import --from .ace/playbook.jsonl --to team:{id}  # 一键导入
3. 开发者配置 server_url
4. .ace/playbook.jsonl 可保留（只读备份）或删除
```

---

## 6. 系统接口

```typescript
enum Scope { Local, Team }

// Federation Bridge（仅 Federation Mode）
interface AceSyncClient {
  pull_index(since: Date, tags: string[]): Promise<BulletIndex[]>;
  fetch_bullets(ids: string[]): Promise<TeamBullet[]>;
  nominate_bullet(sanitized_bullet: Bullet): Promise<string>;
  propose_supersede(team_bullet_id: string, new_bullet: Bullet): Promise<string>;
  cast_vote(team_bullet_id: string, vote: "up" | "down"): Promise<void>;
  pull_taxonomy(): Promise<TagTaxonomy>;
}
```

---

## 7. 与 Universal Solution 的整合点

**解耦后的整合策略**（参照第 3 节解耦架构）：

| 组件 | 修改方式 | 侵入性 | 说明 |
|------|----------|--------|------|
| `Bullet` | Core 新增 `schema_version`, `incompatible_tags` 字段 | 低 | 向后兼容，默认值不影响现有逻辑 |
| `Bullet.scope` | Core 新增 `team:{id}` 枚举值 | 低 | 现有 `global` / `project` 不受影响 |
| `Generator` | **不修改**。Team Layer 提供 `MultiPoolRetriever` 组合器，在初始化层注入 | **零** | Core Generator 代码零改动 |
| `Curator` | **不修改**。Team Layer 提供 `TeamSupersedeCurator` 装饰器 | **零** | Core Curator 代码零改动 |
| `Storage` | **不修改**。Team Layer 新增 `TeamCacheStorage` / `GitFallbackStorage`，实现 Core 的 `StorageBackend` 协议 | **零** | 新增实现，非修改现有 |
| `Daemon` | Team Layer 新增 `TeamSyncDaemon` | **零** | 独立后台任务 |
| `Config` | Team Layer 新增独立的 `TeamConfig`，不修改 Core Config | **零** | 通过 `ext/team_bootstrap.py` 加载 |

**Core 零修改清单**：Reflector、Decay、Privacy、Generator、Curator、Storage、Config 的现有代码**均不修改**。

**性能预算**：

| 场景 | 目标延迟 |
|------|----------|
| Local Pool 检索 | < 50ms |
| + Team Cache / Git Fallback | < 90ms |
| 影子合并 | < 5ms |
| **端到端** | **< 100ms** |

---

## 8. 技术实现细节

### 8.1 ACE Sync Server 技术栈

本方案不绑定特定技术实现，仅定义**功能需求**。实现者可根据团队技术栈自由选择。

| 功能需求 | 需要实现的能力 | 参考实现方案 | 可替代方案 |
|----------|---------------|-------------|-----------|
| **API 层** | 高效序列化、流式传输、类型安全的 RPC 接口 | gRPC (protobuf) | GraphQL, REST+OpenAPI, tRPC, Connect |
| **向量存储** | 高维向量索引 + 元数据过滤 + 多租户隔离 | Qdrant, Milvus | Weaviate, Pinecone, pgvector (PostgreSQL), ChromaDB, LanceDB |
| **身份认证** | 企业级 SSO 集成 + Token 签发 | OIDC (Keycloak) + JWT | LDAP, SAML, OAuth2 (Auth0/Okta), 自建 API Key |
| **元数据存储** | Bullet 元数据、审核状态、投票记录的事务性存储 | PostgreSQL | MySQL, SQLite (小规模), MongoDB, CockroachDB |
| **Tag Taxonomy** | 中心化词表管理 + 版本化同步 | 数据库表 + 版本号 | 独立配置文件 + Git 管理, etcd/Consul KV |

**选型原则**：
- **小团队 (< 20 人)**：单机部署，SQLite/PostgreSQL + 内嵌向量索引即可满足
- **中大型团队**：独立向量数据库 + 关系数据库 + 消息队列
- **已有基础设施**：优先复用团队已有的数据库和认证系统，降低运维成本

### 8.2 同步机制

- **差分同步**：基于 `updated_at` 增量拉取
- **完整向量**：增量同步时包含向量数据
- **墓碑清理**：90 天软删除 + full_sync_check 兜底

---

## 9. 安全与隐私

| 层级 | 机制 | 负责方 |
|------|------|--------|
| L1 自动脱敏 | 正则 + custom_patterns | 本地 Redactor |
| L2 用户审核 | 展示脱敏后内容（不可跳过） | 用户 |
| L3 服务端隔离 | RBAC + 多租户 | Sync Server |
| L4 LLM 泛化 | 可选增强 | 本地 LLM |

---

## 10. 配置

### 10.1 零配置原则

- Git Fallback：仓库中存在 `.ace/playbook.jsonl` 即自动生效，无需任何配置
- Federation Mode：仅需 `server_url` 一项

### 10.2 完整配置

```json
{
  "ace": {
    "team": {
      "enabled": false,
      "server_url": "",
      "subscribed_tags": [],
      "cache_max_bullets": 2000,
      "cache_ttl_minutes": 60,
      "auto_nominate": {
        "enabled": true,
        "min_recall_count": 10,
        "min_instructivity_score": 80,
        "max_prompts_per_session": 1,
        "silent": false
      },
      "redactor": {
        "llm_generalize": false,
        "custom_patterns": []
      },
      "layer_boost": { "local": 1.5, "team": 1.0 },
      "mandatory_overrides": []
    }
  }
}
```

### 10.3 最小配置示例

**Git Fallback（零配置）**：仓库有 `.ace/playbook.jsonl` 即可。

**Federation Mode**：
```json
{ "ace": { "team": { "server_url": "https://ace.company.com" } } }
```

---

## 11. 风险分析与缓解 (Risk Analysis & Mitigations)

### 11.1 Team Cache 容量上限

**风险**：`cache_max_bullets = 2000` 对中大型团队（50+ 人）可能不足。按 `effective_score` Top-N 截断意味着长尾但有价值的冷门知识被丢弃，用户完全无感知。

**缓解**：
- 明确 2000 条上限是**按 `subscribed_tags` 分片后**的容量，而非全局共享。不同角色（前端/后端/SRE）各自 2000 条
- 增加**按需远程补充查询**：检索结果置信度低于阈值时，异步触发远程查询，结果写入 Team Cache 并在下次 prompt 生效
- Session 结束时可选"你可能错过的团队知识"回顾（`ace team missed`）

### 11.2 Pre-Inference 无远程请求 vs 知识覆盖率

**风险**：严格的 Pre-Inference 本地化约束保障了延迟，但牺牲了覆盖率。如果 Team Cache 未缓存到相关知识（Top-N 截断或 tags 订阅不全），用户完全不知道团队已有解决方案。

**缓解**：
- **Post-Inference 异步补充**：本次检索结束后，异步向 Server 发送查询，若命中高分结果则缓存并在下次 session 提示
- **Cache Miss 指标**：统计本地检索未命中但远程有结果的比例，用于动态调整 `cache_max_bullets` 和订阅 tags
- **不破坏核心约束**：Pre-Inference 阶段始终纯本地，补充查询严格异步

### 11.3 Tag Taxonomy 冷启动

**风险**：中心化 Taxonomy 谁来初始化？第一批 tags 从哪来？空 Taxonomy 会导致初期贡献的标签碎片化。

**缓解**：
- 提供**预设 Taxonomy 模板**（按语言/框架/领域分类，如 `rust`, `python`, `react`, `security`, `architecture`, `testing`）
- **种子聚合**：允许从团队成员 Local Pool 的高频 tags 自动提取候选词，由 Curator 审核后入 Taxonomy
- Taxonomy 初始化为 `ace team init` 命令的一部分

### 11.4 Supersede 时间窗口

**风险**：从提交 Supersede Proposal → Curator 审核 → 全员同步，可能有数天延迟。在此期间其他用户基于旧知识做错误决策。

**缓解**：
- 增加 `priority: "urgent" | "normal"` 字段。`urgent` 级别（安全漏洞、严重错误）触发即时推送通知
- `urgent` Supersede 可跳过 Staging 直接入池（初始低权重），同时通知 Curator 补审
- 本地提交 Supersede 后，提交者的 Local Pool 立即生效新版本（不等远程审核）

### 11.5 Federation Server 落地复杂度

**风险**：功能需求实际包含 API + 向量数据库 + OIDC 认证 + 关系数据库 + RBAC，对于"记忆框架"来说运维跳跃过大。小团队可能宁愿用 Git Fallback 也不碰这套。

**缓解**：
- 提供**官方 Docker Compose 参考实现**（SQLite + 内嵌向量 + 简单 API Key 认证），一键启动
- 明确 **Lite / Full 两档部署模式**：
  - **Lite**（< 20 人）：单容器，SQLite，API Key 认证，5 分钟部署
  - **Full**（20+ 人）：PostgreSQL + Qdrant + OIDC，按需扩展
- Federation Server 作为**独立项目**发布，不与 memx 核心库耦合

---

## 12. 实施优先级 (Implementation Priority)

### 12.1 路线图

| 阶段 | 内容 | 前置条件 | 预期产出 |
|------|------|----------|----------|
| **P0: 解耦重构** | 按第 3 节要求重构包结构，确保 Core/Team 边界清晰 | 无 | `memx/core/` + `memx/team/` 分离，现有测试 100% 通过 |
| **P1: Git Fallback** | 实现 `GitFallbackStorage`（只读 JSONL 加载 + 向量缓存 + 读时去重） | P0 | 仓库有 `.ace/playbook.jsonl` 即可使用团队知识 |
| **P2: Federation MVP** | 实现 `TeamCacheStorage` + `SyncClient`（pull/push） + 自动审批 | P0 | 最小可用的 Federation Mode |
| **P3: 治理能力** | P2P 投票、Supersede、Curator 审核、Redactor | P2 | 完整治理流水线 |
| **P4: 运维成熟** | Docker Compose 参考实现、监控、Cache Miss 指标 | P2 | 生产可用 |

### 12.2 关键原则

- **P0 先行**：不做解耦重构，后续所有工作都会侵入 Core，代价越晚越高
- **P1 验证需求**：Git Fallback 零成本上线，用真实用户验证"团队记忆"是否有真需求
- **P2 再建 Server**：确认需求后再投入 Server 开发，避免过早投入重资产
- **每个阶段独立可交付**：P1 完成即可发版，不依赖后续阶段

---

## 13. 总结

通过 **Federation Mode（联邦网络）** 实现"一人避坑，全员免疫"的自适应协同记忆。**Git Fallback** 作为零成本退行方案，确保任何团队都能以最低门槛开始使用团队记忆。

**充分解耦**是本方案的硬约束：Team Memory 作为纯可选扩展层，Core 代码零侵入、可剥离、无感降级。这确保了单机用户不受任何影响，也确保了 Team 功能可以独立迭代而不引入回归风险。

当团队规模、需求或基础设施条件成熟时，可随时从 Git Fallback 一键升级到 Federation Mode，无数据丢失。

---

## Review Feedback Log

历经 v1.1~v1.4 共 5 轮审查（Gemini + Red Team + Deep Review + Architecture Hardening），累计处理 30+ 个问题。v2.0 通过架构简化（砍掉 Phase 1/2/3 分阶段路线，统一为 Federation + Git Fallback）消除了大量 Git 工作流相关的设计复杂度。v2.1 新增解耦架构、风险分析、实施优先级。

关键决策记录：
- Git 从"主方案的第一步"降级为"只读退行备选"，消除 merge=union、auto-compact、append-only 写入等问题
- Git Fallback 的 `.ace/playbook.jsonl` 由引擎只读加载，不写入，彻底消除 Git 冲突
- 原 Phase 2（只读注册中心）与 Phase 3（双向联邦）合并为统一的 Federation Mode
- P2P 验证砍掉 AI 执行结果投票（噪音源），仅保留显式投票 + Supersede 信号
- 模型指纹不匹配时柔性降级（非拒绝加载），遵循 Graceful Degradation
- Incompatible Tags + 中心化 Taxonomy 替代简单动作词匹配
- Mandatory 逃生舱增加 reason + expires 防破窗
- 墓碑 90 天 + full_sync_check 防止长期离线后脏缓存
- **v2.1**: Team/Local 充分解耦列为硬约束，Core 代码零修改（Generator/Curator/Storage 通过组合模式+依赖注入扩展）
- **v2.1**: 新增 5 项风险分析（Cache 容量、覆盖率、Taxonomy 冷启动、Supersede 时间窗口、Server 落地复杂度）及缓解方案
- **v2.1**: 新增分阶段实施路线（P0 解耦重构 → P1 Git Fallback → P2 Federation MVP → P3 治理 → P4 运维）
