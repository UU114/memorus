# Memorus — AI 智能体的自适应协同记忆引擎

> 基于 mem0 + ACE v2.0 联邦架构：自动学习、自动协同、自动遗忘

Memorus 在 [mem0](https://github.com/mem0ai/mem0) 基础上构建了 **ACE v2.0（自适应上下文引擎）** —— 一套联邦化的知识流水线，可自动提炼、去重、同步本地环境与团队池中的知识。实现“一人避坑，全员免疫”，让你的 AI 智能体记住重要的、修正错误的、分享有用的。

---

## 为什么选择 Memorus：本地优先 + 联邦协同

传统的记忆框架强依赖云端 API 且缺乏团队协作能力，Memorus 遵循**本地优先 + 可选联邦**原则：

| 能力 | mem0 | Memorus (v2.0) |
|------|------|----------------------|
| **知识提取** | 每次 add() 调 LLM (~$0.01) | **混合规则引擎 ($0)** + 可选 LLM |
| **去重机制** | 基于 LLM | **语义合并 + Supersede (知识纠正)** |
| **遗忘机制** | 无——记忆永久留存 | **艾宾浩斯衰退 + 召回强化** |
| **团队协作** | 无 | **联邦同步 + Git Fallback (v2.0 新增)** |
| **离线能力** | 不可用 | **完全离线运行** (ONNX + 规则 + Git) |
| **纠正机制** | 手动覆盖/删除 | **自动 Supersede (知识进化)** |
| **隐私脱敏** | 无内置方案 | **12 层 PII 过滤器 + 默认本地** |

---

## ACE v2.0 核心架构：联邦协同

Memorus v2.0 引入了**联邦上下文引擎**，支持多层级的记忆共享：

### 1. 三级记忆池模型
- **Local (Global)**：个人私有习惯与通用知识 (`scope: global`)。
- **Local (Project)**：个人在特定项目中的私有经验 (`scope: project:name`)。
- **Team (Org)**：团队共享的架构规约、最佳实践、避坑指南 (`scope: team:id`)。

### 2. 联邦运行模式
- **Federation Mode**：通过 **ACE Sync Server** 实现实时双向同步、P2P 验证与治理。
- **Git Fallback**：零成本协作，知识存储于仓库内的 `.ace/playbook.jsonl`，随 Git 版本控制共享，基于 PR 进行审核。

---

## 三种 Reflector 模式

Reflector 是 Memorus 的知识蒸馏引擎，负责从对话中提炼出结构化的知识规则（Bullet）。它提供三种运行模式：

### `rules` 模式
纯规则引擎，0 LLM 调用，0 成本，完全离线。适合高频写入、成本敏感场景。

### `llm` 模式
每次交互调用 LLM 做语义级评估和知识蒸馏。适合低频高价值、需要捕获隐性知识的场景。

### `hybrid` 模式（默认，推荐）
规则预筛 + LLM 精评，质量/成本最优平衡。**v2.0 新增滑动窗口机制**，用于检测用户的纠正意图。

---

## 功能概览

- **Reflector (v2.0)** — 3 模式蒸馏引擎。新增**滑动窗口**支持，能识别“不对，应该是 X”等纠正语境。
- **Curator (v2.0)** — 语义去重 + 冲突检测。新增 **Supersede 机制**，允许新知识精准覆盖/取代旧的错误知识。
- **Team Layer (v2.0)** — 影子合并 (Shadow Merging)。检索时自动合并团队知识，对本地库**只读不写**，确保环境纯净。
- **Decay** — 艾宾浩斯指数衰退 + 召回强化，模拟人类“用进废退”机制。
- **Generator** — 4 层混合检索（精确 + 模糊 + 元数据 + 向量）。
- **Privacy** — 12 种内置 PII 脱敏规则 + 可插拔自定义正则。
- **ONNX** — 本地嵌入推理（all-MiniLM-L6-v2），完全离线可用。

---

## 安装

```bash
# 核心包 — 本地优先混合模式
pip install memorus

# 团队协同支持 (Federation)
pip install memorus[team]

# 本地嵌入 (ONNX)
pip install memorus[onnx]

# 全部功能 (推荐团队使用)
pip install memorus[all]
```

---

## 快速开始

### 本地智能记忆
```python
from memorus import Memory

m = Memory(config={"ace_enabled": True})

# ACE 自动从对话中提炼知识
m.add([{"role": "user", "content": "本项目中总是使用 pnpm"}], user_id="dev_1")

# 检索时自动应用衰退权重和混合评分
results = m.search("包管理器", user_id="dev_1")
```

### 团队协同 (Federation)
```python
config = {
    "ace_enabled": True,
    "team": {
        "enabled": True,
        "server_url": "https://ace.your-company.com",
        "subscribed_tags": ["python", "security"]
    }
}
m = Memory(config=config)
```
*注：对于 Git Fallback 模式，只需在仓库根目录放置 `.ace/playbook.jsonl` 即可自动生效。*

---

## 关键机制

### 1. 知识纠正与覆盖 (Correction & Supersede)
当用户说“不对，换成 X 方式”时，ACE 会检测到纠正意图：
- **Reflector** 将该轮交互标记为 `is_correction`。
- **Curator** 寻找与之语义相关的旧知识，并执行 **Supersede**：用新内容替换旧内容，但保留其 ID 和历史召回统计，实现知识的“原地进化”。

### 2. 贡献流水线 (Promotion Pipeline)
1. **自动检测**：ACE 识别出本地高质量知识（如召回次数 > 10）。
2. **脱敏处理**：本地 Redactor 自动擦除所有个人路径、密钥等。
3. **提名上传**：用户审核脱敏后的内容，提交至团队 Staging 池。
4. **治理审批**：经过 P2P 投票或管理员审核后，合入团队主池。

---

## 配置参考

```json
{
  "ace": {
    "reflector": { "mode": "hybrid", "llm_model": "gpt-4o-mini" },
    "curator": { "similarity_threshold": 0.8, "correction_score_threshold": 50 },
    "decay": { "half_life_days": 30, "permanent_threshold": 15 },
    "team": {
      "enabled": true,
      "layer_boost": { "local": 1.5, "team": 1.0 },
      "auto_nominate": { "enabled": true, "min_recall_count": 10 }
    }
  }
}
```

---

## 命令行工具

- `memorus status`：查看知识库统计与健康度。
- `memorus search "query"`：测试检索与排序。
- `memorus learn "Always use X"`：手动添加蒸馏规则。
- `memorus nominate`：审核并提交本地知识到团队。
- `memorus team sync`：拉取最新的团队知识更新。
- `memorus conflicts [--type anchor_mismatch]`：检测冲突/矛盾的记忆，可按 wire-form 类型过滤。
- `memorus verify [--rehydrate-anchors] [--stale-only] [--scope] [--dry-run] [--json]`：批量验证 / 回迁锚点（详见 [doc/verification.md](doc/verification.md)）。
- `memorus sweep`：手动触发全量衰退扫描。

---

## Memory Trust（记忆信任层）

EPIC-R018 引入了**记忆信任层**：bullet 在入库时自动抽取 **anchors**（指向源代码片段），检索时由 `VerificationEngine` 比对实时文件状态，给出 `verified / stale / unverifiable / not_applicable` 四种状态以及 `trust_score`。三种策略 (`flag` / `demote` / `drop`) 决定 stale 行如何呈现给上层 prompt。

完整模型、配置、迁移步骤与 LLM prompt 集成模式见 [`doc/verification.md`](doc/verification.md)。Rust 版本对应 [`memorus-r/doc/verification.md`](memorus-r/doc/verification.md)。

---

## 与现有方案对比

| 维度 | RAG | ChatGPT Memory | Mem0 | **ACE (v2.0)** |
|------|----------|----------------|------|----------------|
| **知识来源** | 静态文档 | 用户明确告知 | API 式存取 | **自动蒸馏** |
| **时效维护** | 手动清理 | 永久存储 | 手动管理 | **指数衰退 + 强化** |
| **纠正覆盖** | 无 | 无 | 无 | **Supersede 覆盖** |
| **团队共享** | 静态 | 无 | 部分支持 | **联邦模式 / Git 共享** |
| **隐私安全** | 可能云端 | 云端 | 云端 | **完全本地 + 自动脱敏** |
| **部署方式** | 较重 | SaaS | SaaS | **SQLite + ONNX (本地)** |

---

## 许可证

Apache-2.0

---

*英文版本请参考 [README.md](README.md)。*
