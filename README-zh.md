# MemX — AI 智能体的自适应记忆引擎

> 基于 mem0 + ACE 智能层：自动学习、自动遗忘、自动召回

MemX 在 [mem0](https://github.com/mem0ai/mem0) 基础上构建了 **ACE（自适应上下文引擎）** —— 一套自动提炼、去重、衰退和检索知识的流水线，让你的 AI 智能体记住重要的、遗忘无关的。

---

## 为什么选择 MemX：本地优先的智能记忆

### mem0 的核心问题

mem0 是优秀的记忆框架，但它的核心流程**强依赖远程 LLM API**。每次调用 `add()` 都需要 LLM 提取事实、判断去重、决策更新：

| 痛点 | 影响 |
|------|------|
| **成本累积** | 每次写入消耗 2–5K tokens（约 $0.001–0.01/次），高频交互下账单失控 |
| **离线不可用** | 内网开发、本地调试、飞行模式下完全无法工作 |
| **无本地可控性** | 知识提取质量完全取决于远端 LLM，无法本地调优 |
| **隐私合规风险** | 用户数据必须发送到云端 LLM，无法通过金融/军工等行业审查 |

### MemX 的解法：Local-First

MemX 的核心设计原则是**本地优先**——所有核心能力默认在本地完成，零外部依赖。LLM 是可选增强，不是必需前提。

| 能力 | mem0 | MemX (rules 模式) | MemX (llm/hybrid 模式) |
|------|------|-------------------|----------------------|
| **知识提取** | 每次 add() 调 LLM (~2–5K tokens) | 规则引擎，**0 API 调用，0 成本** | LLM 语义评估 + 结构化蒸馏 |
| **去重** | LLM 判断 UPDATE/DELETE | 余弦相似度 >= 0.8 自动合并 | 同左 |
| **遗忘** | 无——记忆永久留存 | 指数衰退 + 召回强化 | 同左 |
| **检索** | 纯向量相似度 | 4 层混合：精确 + 模糊 + 元数据 + 向量 | 同左 |
| **隐私脱敏** | 无内置方案 | 12 种内置规则 + 自定义正则 | 同左 |
| **离线能力** | 不可用 | **完全离线运行**（ONNX 本地嵌入） | 需 LLM API（失败自动降级到 rules） |
| **每次写入成本** | ~$0.001–0.01 | **$0** | ~$0.0005（仅有价值的交互触发） |
| **向量嵌入** | 需外部 API | ONNX Runtime 本地推理 | 同左 |
| **作用域** | 扁平 (user_id / agent_id) | 层级：`global` / `project:name` / `workspace:id` | 同左 |
| **Token 预算** | 调用方自行管理 | 内置裁剪器（CJK 感知） | 同左 |
| **命令行工具** | 无 | 10 条 CLI 命令 | 同左 |

> **核心优势：即使在 llm/hybrid 模式下，LLM 调用失败也会自动降级到 rules 模式——永远不会因为 API 故障而丢失数据。**

---

## 三种 Reflector 模式

Reflector 是 MemX 的知识蒸馏引擎，负责从对话中提炼出结构化的知识规则（Bullet）。它提供三种运行模式：

### `rules` 模式（默认）

纯规则引擎，0 LLM 调用，0 成本，完全离线。

- 5 条内置检测规则（error_fix / retry_success / config_change / new_tool / repetitive_op）
- 基于关键词匹配的分类和评分
- 适合：高频写入、成本敏感、离线环境、CI/CD 管道

### `llm` 模式

每次交互调用 LLM 做语义级评估和知识蒸馏。

- LLMEvaluator：判断 should_record + 分类 + 评分（1 次 API 调用）
- LLMDistiller：生成 "When [条件], [动作], because [原因]" 格式的结构化规则（1 次 API 调用）
- 适合：低频高价值场景、需要捕获隐性知识和用户偏好

### `hybrid` 模式（推荐）

规则预筛 + LLM 精评，质量/成本最优平衡。

- 规则先快速过滤 70%+ 的琐碎内容（0 成本）
- 规则命中的候选交给 LLM 做精评和蒸馏
- 规则漏掉的交互由 LLM 兜底评估（捕获用户偏好等隐性知识）
- 适合：生产环境日常使用

```
rules 模式:    交互事件 → 规则检测 → 规则评分 → 脱敏 → 规则蒸馏           (0 API calls)
llm 模式:      交互事件 → LLM 评估 → 脱敏 → LLM 蒸馏                  (1-2 API calls)
hybrid 模式:   交互事件 → 规则预筛 → LLM 精评/兜底 → 脱敏 → LLM 蒸馏    (0-2 API calls)
```

### 实测效果对比

| 场景 | rules | llm | hybrid |
|------|-------|-----|--------|
| Docker+PG 连接错误修复 | 2 bullets, 关键词截断, <1ms | 1 bullet, 语义蒸馏出结构化规则, ~14s | 2 bullets, 规则+LLM 混合精炼, ~21s |
| "几点了"闲聊 | 正确跳过, 0ms | 正确跳过, ~3s | 正确跳过, ~3s |
| "Python 用单引号"偏好 | **漏掉** | 捕获并蒸馏为规则 | 捕获并蒸馏为规则 |

---

## 功能概览

- **Reflector** — 3 模式知识蒸馏引擎（rules / llm / hybrid），自动将对话噪声提炼为结构化知识规则
- **Curator** — 语义去重（余弦相似度）+ 冲突检测（Semantic / Negation）
- **Decay** — 艾宾浩斯指数衰退 + 召回强化，模拟人类"用进废退"认知机制
- **Generator** — 4 层混合检索（精确匹配 + 模糊匹配 + 元数据匹配 + 向量语义）
- **Privacy** — 12 种内置 PII 脱敏规则 + 可插拔自定义正则
- **ONNX** — 本地嵌入推理（all-MiniLM-L6-v2, 384 维），完全离线可用
- **CLI** — 10 条命令完整管理知识库
- **Daemon** — 可选后台进程，支持多 Agent 共享内存

---

## 环境要求

| 项目 | 说明 |
|------|------|
| **Python** | >= 3.9, <= 3.14 |
| **mem0 后端** | 至少配置一个向量存储（如 Qdrant, Chroma） |
| **API 密钥** | 取决于后端——ONNX 嵌入不需要任何密钥 |

## 安装

```bash
# 核心包 — rules 模式，0 LLM 调用，0 成本
pip install memx

# LLM 增强（支持 llm/hybrid 模式）
pip install memx[llm]

# 本地嵌入（无需 API 密钥）
pip install memx[onnx]

# Neo4j 图数据库支持
pip install memx[graph]

# 全部功能
pip install memx[all]

# 开发环境
pip install memx[dev]
```

> **推荐组合**：`pip install memx[onnx,llm]` — 嵌入完全本地 + Reflector 可选 LLM 增强

### 核心依赖

```
mem0ai >= 1.0.0
pydantic >= 2.0
click >= 8.0
```

### 可选依赖

| 扩展 | 包 | 用途 |
|------|---|------|
| `llm` | litellm >= 1.40 | LLM/hybrid Reflector 模式（支持 OpenAI、Anthropic、Deepseek、Ollama 等） |
| `onnx` | onnxruntime >= 1.16, tokenizers >= 0.15 | 本地嵌入推理（无需 API） |
| `graph` | neo4j >= 5.0 | 图关系记忆 |
| `dev` | pytest, mypy, ruff 等 | 测试与代码检查 |

---

## 快速开始

```python
from memx import Memory

# 初始化，启用 ACE
m = Memory(config={"ace_enabled": True})

# 从对话中学习知识
result = m.add(
    [{"role": "user", "content": "pytest 运行时总是加 -v 参数以获取详细输出"}],
    user_id="dev1",
)
# result 包含: bullets_added, bullets_merged, bullets_skipped

# 检索知识库
results = m.search("pytest 详细模式", user_id="dev1")
for r in results.get("results", []):
    print(f"[{r['score']:.2f}] {r['memory']}")
```

### 使用 LLM 增强模式

```python
m = Memory(config={
    "ace_enabled": True,
    "reflector": {
        "mode": "hybrid",                        # 推荐模式
        "llm_model": "deepseek/deepseek-chat",   # 任何 litellm 兼容的模型
        "llm_api_base": "https://api.deepseek.com",
    },
})
```

### 完全离线模式

```python
# 嵌入 + 知识提取全部本地完成，无需任何 API 密钥
m = Memory(config={
    "ace_enabled": True,
    "reflector": {"mode": "rules"},     # 规则引擎，0 API 调用
    "embedder": {
        "provider": "onnx",             # ONNX 本地嵌入
        "config": {"model": "all-MiniLM-L6-v2"},
    },
})
```

### ACE 关闭（零开销代理）

```python
# 不启用 ACE 时，MemX 是 mem0 的透明代理，行为完全一致
m = Memory()
m.add("某个事实", user_id="u1")
```

---

## 从 mem0 迁移

```python
# 迁移前 (mem0)
from mem0 import Memory
m = Memory.from_config({"vector_store": {"provider": "qdrant", ...}})
m.add("fact", user_id="u1")
results = m.search("query", user_id="u1")

# 迁移后 (MemX, ACE 关闭 — 行为完全一致，零开销)
from memx import Memory
m = Memory(config={"vector_store": {"provider": "qdrant", ...}})
m.add("fact", user_id="u1")
results = m.search("query", user_id="u1")

# 迁移后 (MemX, ACE 开启 — 完整智能管道)
from memx import Memory
m = Memory(config={
    "ace_enabled": True,
    "vector_store": {"provider": "qdrant", ...},
})
m.add(
    [{"role": "user", "content": "pytest 总是加 -v 参数"}],
    user_id="u1",
)
results = m.search("pytest 详细模式", user_id="u1")
```

**ACE 启用后的变化：**
- `add()` 返回 `ace_ingest` 信封，包含 `bullets_added`、`bullets_merged`、`bullets_skipped` 计数
- `search()` 返回 `ace_search` 信封，包含 `mode`（"full" / "degraded" / "fallback"）和 `total_candidates`
- ACE 配置键（`reflector`、`curator`、`decay`、`retrieval`、`privacy`）为保留字段

**完全向后兼容：**
- 所有 mem0 向量存储提供者（Qdrant、Chroma、Pinecone、PGVector 等）照常工作
- 所有 mem0 嵌入提供者（OpenAI、Ollama、HuggingFace 等）照常工作
- 所有 mem0 LLM 提供者照常工作
- ACE 关闭时，`add()` / `search()` / `get()` / `delete()` / `update()` / `get_all()` / `history()` / `reset()` 行为与 mem0 完全一致

---

## API 参考

### 构造函数

```python
Memory(config: dict | None = None)
```

`config` 是一个包含 mem0 键和 ACE 键的字典。MemX 自动分离——ACE 键（`ace_enabled`、`reflector`、`curator`、`decay`、`retrieval`、`privacy`、`integration`、`daemon`）由 ACE 引擎消费；其余全部转发给 mem0 后端。

### 方法（mem0 兼容）

| 方法 | 签名 | 说明 |
|------|------|------|
| `add()` | `add(messages, user_id, agent_id, run_id, metadata, filters, prompt, scope, **kwargs)` | 添加记忆。ACE 开启时：蒸馏 → 去重 → 持久化。关闭时：直接透传 mem0。 |
| `search()` | `search(query, user_id, agent_id, run_id, limit=100, filters, scope, **kwargs)` | 搜索。ACE 开启时：4 层混合检索 + 衰退评分。关闭时：纯向量搜索。 |
| `get()` | `get(memory_id)` | 按 ID 获取单条记忆。 |
| `get_all()` | `get_all(user_id, agent_id, **kwargs)` | 获取用户/Agent 的全部记忆。 |
| `update()` | `update(memory_id, data)` | 更新记忆内容。 |
| `delete()` | `delete(memory_id)` | 删除单条记忆。 |
| `delete_all()` | `delete_all(user_id, agent_id, **kwargs)` | 删除匹配条件的所有记忆。 |
| `history()` | `history(memory_id)` | 获取记忆的修改历史。 |
| `reset()` | `reset()` | 清空所有记忆。 |

### 方法（ACE 专有）

| 方法 | 签名 | 说明 |
|------|------|------|
| `status()` | `status(user_id=None)` | 知识库统计：总数、分区、类型分布、平均衰退权重。 |
| `detect_conflicts()` | `detect_conflicts(user_id=None)` | 检测矛盾记忆（需开启 `curator.conflict_detection`）。 |
| `export()` | `export(format="json", scope=None)` | 导出知识库（JSON 或 Markdown）。 |
| `import_data()` | `import_data(data, format="json")` | 导入并自动去重。返回 `{imported, skipped, merged}`。 |
| `run_decay_sweep()` | `run_decay_sweep()` | 手动触发全量衰退扫描。 |
| `from_config()` | `Memory.from_config(config_dict)` | 工厂方法（类方法）。 |

---

## 配置参考

MemX 使用与 mem0 相同的配置字典格式，附加 ACE 专用键。所有 ACE 键均可选——默认值针对通用场景优化。

```python
config = {
    # --- ACE 总开关 ---
    "ace_enabled": True,

    # --- Reflector：知识蒸馏引擎 ---
    "reflector": {
        "mode": "rules",             # "rules"（默认，0 LLM）| "llm" | "hybrid"（推荐）
        "min_score": 30.0,           # [0-100] 最低指导性评分阈值
        "max_content_length": 500,   # 每条 Bullet 最大字符数
        "max_code_lines": 3,         # Bullet 中最大代码行数
        # LLM 设置（仅 mode = "llm" 或 "hybrid" 时生效）
        "llm_model": "openai/gpt-4o-mini",  # 任何 litellm 兼容的模型标识符
        "llm_api_base": None,        # 自定义 API 地址（如 "https://api.deepseek.com"）
        "llm_api_key": None,         # API 密钥（为 None 时使用环境变量）
        "max_eval_tokens": 512,      # LLM 评估响应最大 token 数
        "max_distill_tokens": 256,   # LLM 蒸馏响应最大 token 数
        "llm_temperature": 0.1,      # 低温度保证提取结果的确定性
    },

    # --- Curator：去重与冲突检测 ---
    "curator": {
        "similarity_threshold": 0.8,       # [0-1] 合并阈值
        "merge_strategy": "keep_best",     # "keep_best" | "merge_content"
        "conflict_detection": False,       # 启用矛盾检测
        "conflict_min_similarity": 0.5,    # 冲突窗口下界
        "conflict_max_similarity": 0.8,    # 冲突窗口上界
    },

    # --- Decay：时间衰退 ---
    "decay": {
        "half_life_days": 30.0,        # 指数衰退半衰期（天）
        "boost_factor": 0.1,           # 召回加权：weight *= (1 + boost * recall_count)
        "protection_days": 7,          # 新记忆保护期（天）
        "permanent_threshold": 15,     # 召回次数 >= 此值 → 永久保留
        "archive_threshold": 0.02,     # 权重低于此值 → 归档候选
        "sweep_on_session_end": True,  # 会话结束时自动执行衰退扫描
    },

    # --- Retrieval：混合检索调优 ---
    "retrieval": {
        "keyword_weight": 0.6,         # 关键词层权重（精确+模糊+元数据）
        "semantic_weight": 0.4,        # 向量语义层权重
        "recency_boost_days": 7,       # 近 N 天的记忆获得加成
        "recency_boost_factor": 1.2,   # 时效加成倍数
        "scope_boost": 1.3,            # 作用域匹配加成倍数
        "max_results": 5,              # 裁剪后最大返回条数
        "token_budget": 2000,          # LLM 上下文的最大 token 预算
    },

    # --- Privacy：PII / 密钥脱敏 ---
    "privacy": {
        "always_sanitize": False,      # ACE 关闭时也执行脱敏
        "sanitize_paths": True,        # 脱敏操作系统用户路径
        "custom_patterns": [           # 自定义正则模式
            r"INTERNAL-\d{6}",
        ],
    },

    # --- Integration：Agent 集成行为 ---
    "integration": {
        "auto_recall": True,           # 推理前自动召回
        "auto_reflect": True,          # 工具执行后自动反思
        "sweep_on_exit": True,         # 会话结束时自动衰退
        "context_template": "xml",     # "xml" | "markdown" | "plain"
    },

    # --- Daemon：多进程共享内存 ---
    "daemon": {
        "enabled": False,              # 启用后台守护进程
        "idle_timeout_seconds": 300,   # 空闲超时（秒）
        "socket_path": None,           # IPC 套接字路径（None 时自动选择）
    },

    # --- mem0 后端配置（原样透传） ---
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333},
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini"},
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"},
    },
}
```

---

## 系统架构

### 写入管道 (`add()`)

```
原始输入（消息列表 / 字符串）
    |
    v
+----------------------+
|   Privacy Sanitizer   |  脱敏：PII、API 密钥、Token、路径
+-----------+----------+
            v
+----------------------+
|   Reflector (蒸馏器)  |  rules:  规则检测 → 规则评分 → 脱敏 → 规则蒸馏
|                      |  llm:    LLM 评估 → 脱敏 → LLM 蒸馏
|                      |  hybrid: 规则预筛 → LLM 精评/兜底 → 脱敏 → LLM 蒸馏
+-----------+----------+
            v
+----------------------+
|   Curator (整理器)    |  对比现有知识库：
|                      |    相似度 >= 0.8 → 合并
|                      |    相似度 0.5-0.8 → 冲突告警
|                      |    其他 → 新增
+-----------+----------+
            v
+----------------------+
|   mem0 后端           |  持久化到向量存储（Qdrant、Chroma 等）
+----------------------+
```

### 检索管道 (`search()`)

```
查询字符串
    |
    v
+----------------------+
|   Generator (4 层)    |  L1: ExactMatcher    — 全词精确匹配
|                      |  L2: FuzzyMatcher    — 模糊匹配（SequenceMatcher）
|                      |  L3: MetadataMatcher — 工具/实体/标签匹配（Jaccard）
|                      |  L4: VectorSearcher  — 嵌入余弦相似度
+-----------+----------+
            v
+----------------------+
|   Score Merger        |  final = (keyword * 0.6 + semantic * 0.4)
|                      |        * decay_weight
|                      |        * recency_boost
|                      |        * scope_boost
+-----------+----------+
            v
+----------------------+
|   Token Trimmer       |  裁剪至 max_results (5) 和 token_budget (2000)
|                      |  CJK 感知：1.5 字符/token vs 4.0（拉丁字母）
|                      |  保证：至少返回 1 条结果
+-----------+----------+
            v
+----------------------+
|   Recall Reinforcer   |  异步：对被召回的 Bullet 执行 recall_count +1
|   (后台执行)          |  → 反馈到 Decay 引擎，实现自适应留存
+----------------------+
```

### 衰退公式

```
base_weight = 2^(-age_days / half_life)
boosted     = base_weight * (1 + boost_factor * recall_count)
final       = clamp(boosted, 0.0, 1.0)
```

特殊规则：
- `recall_count >= 15` → **永久保留**（weight = 1.0，永不衰退）
- `age <= 7 天` → **保护期**（weight = 1.0，暂不衰退）
- `weight < 0.02` → **归档候选**（可被清理）

---

## ONNX 本地嵌入

完全离线运行向量嵌入——无需 API 密钥，无需网络连接。

```bash
pip install memx[onnx]
```

```python
m = Memory(config={
    "ace_enabled": True,
    "embedder": {
        "provider": "onnx",
        "config": {"model": "all-MiniLM-L6-v2"},
    },
})
```

| 设置项 | 默认值 |
|--------|--------|
| 模型 | all-MiniLM-L6-v2 |
| 维度 | 384 |
| 最大 token | 256 |
| 缓存目录 | `~/.memx/models/` |
| 自动下载 | 是（HuggingFace Hub，仅首次运行） |

首次下载后模型缓存在本地，之后完全离线可用。如果 ONNX 依赖缺失，检索管道优雅降级（跳过向量层，关键词层仍然工作）。

---

## 隐私脱敏

脱敏器在**所有处理之前**运行，不可禁用。内置 12 种检测模式：

| # | 模式 | 示例 |
|---|------|------|
| 1 | 私钥块 (PEM) | `-----BEGIN RSA PRIVATE KEY-----` |
| 2 | Bearer / JWT Token | `Bearer eyJhbG...` |
| 3 | Anthropic API 密钥 | `sk-ant-api03-...` |
| 4 | OpenAI API 密钥 | `sk-proj-...` |
| 5 | GitHub Token | `ghp_xxxx`、`github_pat_...` |
| 6 | AWS Access Key ID | `AKIA...` |
| 7 | AWS Secret Access Key | 40 字符 base64 字符串 |
| 8 | 带凭据的数据库 URL | `postgres://user:pass@host/db` |
| 9 | 通用 API key 参数 | URL 中的 `api_key=...` |
| 10 | 密码/密钥字段 | `password: "..."` |
| 11 | Windows 用户路径 | `C:\Users\john\...` |
| 12 | Unix 用户路径 | `/home/john/...` |

添加自定义模式：

```python
config = {
    "privacy": {
        "custom_patterns": [
            r"INTERNAL-\d{6}",          # 公司内部 ID
            r"customer_[a-f0-9]{32}",   # 客户 Token
        ],
    },
}
```

---

## 多用户与多 Agent

### 作用域层级

```python
# 全局知识（所有用户共享）
m.add("时间戳统一使用 UTC", scope="global", user_id="alice")

# 项目级知识
m.add("本项目使用 FastAPI", scope="project:myapp", user_id="alice")

# 用户级知识
m.add("我偏好深色主题", user_id="alice")

# Agent 级知识
m.add("工具 X 需要 --force 参数", agent_id="tool_resolver")
```

### 跨 Agent 召回

```python
# 检索时自动匹配作用域并加权（默认 +30%）
results = m.search("API 模式", user_id="alice", scope="project:myapp")
# 返回：project:myapp 记忆（加权）+ global 记忆
```

### 守护进程模式（多进程共享内存）

```python
m = Memory(config={
    "ace_enabled": True,
    "daemon": {
        "enabled": True,
        "idle_timeout_seconds": 300,
    },
})
# 共享此配置的所有 Agent 通过 IPC 使用同一知识库
# 守护进程不可用时自动降级为直连模式
```

---

## 命令行工具

安装后可使用 `memx` 命令：

```bash
# 查看知识库状态
memx status

# 搜索记忆
memx search "pytest" --limit 10 --scope "project:myapp"

# 学习新知识（经过 Reflector 处理）
memx learn "总是使用 -v 参数"

# 原始写入（跳过 Reflector）
memx learn "原始事实" --raw

# 列出记忆（支持过滤）
memx list --type method --scope "project:myapp" --limit 20

# 导出知识库
memx export --format json
memx export --format markdown -o knowledge.md

# 导入知识库（自动去重）
memx import --file backup.json

# 检测矛盾记忆
memx conflicts

# 删除指定记忆
memx forget <memory-id>
memx forget <memory-id> --yes   # 跳过确认

# 执行衰退扫描
memx sweep
```

所有命令支持 `--json` 输出和 `--user-id` 多用户过滤。

---

## 知识类型与分区

MemX 将每条知识 Bullet 标记为一个**类型**和一个**分区**，实现结构化组织。

### 知识类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `method` | 解决问题的步骤/流程 | "用 `git rebase -i` 压缩提交" |
| `trick` | 效率小技巧 | "Ctrl+Shift+P 打开命令面板" |
| `pitfall` | 常见错误/避坑指南 | "不要用 `==` 比较 None" |
| `preference` | 用户/团队偏好 | "Python 总是用单引号" |
| `knowledge` | 领域知识/事实 | "PostgreSQL 支持 JSONB 索引" |

### 分区

`commands` · `debugging` · `architecture` · `workflow` · `tools` · `patterns` · `preferences` · `general`

---

## 常见问题

### ACE 开启后搜索无结果

Reflector 会过滤低质量内容。检查 `reflector.min_score`（默认 30.0）——如果有效知识被误过滤，适当降低此值。

### 向量搜索层被跳过（"degraded" 模式）

嵌入提供者不可用（API 故障或 ONNX 未安装）。关键词层（L1-L3）仍正常工作。安装 ONNX 以获得离线弹性：

```bash
pip install memx[onnx]
```

### 记忆随时间消失

这是 Decay 引擎的正常行为。经常被召回的记忆会被保留，不用的会逐渐衰退。要永久保留某条记忆，让它被召回 15 次以上，或增大 `decay.half_life_days`。

### PII 脱敏误报

内置模式不可禁用（设计如此）。如果出现误报，脱敏后的内容会包含 `[REDACTED]` 标记。检查 `privacy.sanitize_paths` 是否需要关闭路径脱敏。

### `add()` 返回 `raw_fallback: true`

Reflector 未能提取结构化 Bullet，回退到原始 mem0 `add()`。数据已保存，只是未经蒸馏。检查输入格式（role/content 字典列表效果最佳）。

### LLM/hybrid 模式降级为 rules

如果设置了 `reflector.mode` 为 `"llm"` 或 `"hybrid"` 但 Bullet 缺少 `distilled_rule` 字段，说明 LLM 调用失败并自动降级。检查：
1. 已安装 `litellm`（`pip install memx[llm]`）
2. API 密钥已设置（环境变量或 `reflector.llm_api_key`）
3. 模型标识符正确（如 `"openai/gpt-4o-mini"`、`"deepseek/deepseek-chat"`）

---

## 许可证

Apache-2.0
