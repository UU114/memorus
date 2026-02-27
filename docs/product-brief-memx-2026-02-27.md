# Product Brief: MemX

**Date:** 2026-02-27
**Author:** TPY
**Version:** 1.0
**Project Type:** AI 记忆引擎（mem0 Fork + ACE 智能层）
**Project Level:** Level 3（大型项目）

---

## Executive Summary

MemX 是基于 mem0 开源项目的深度改造 Fork，通过叠加 ACE（Adaptive Context Engine）智能层，将 mem0 从"被动存取型"记忆系统进化为"主动蒸馏型"自适应记忆引擎。它面向所有"用户与 AI 多轮交互"场景的产品开发者和企业客户，提供自动知识蒸馏、艾宾浩斯衰退遗忘、语义去重、混合三层检索等能力，同时完整保留 mem0 丰富的基础设施生态（23 个向量库、13+ LLM/Embedding Provider）。目标是成为 AI 产品的标准记忆基础设施，实现"越用越懂你"的智能体验。

---

## Problem Statement

### The Problem

当前所有 AI 产品面临共同的记忆困境——**五大痛点悖论**：

1. **噪声干扰悖论** — 全量记录导致信噪比极低，上下文被毒化
2. **成本与性能悖论** — Token 堆叠导致费用/延迟指数增长
3. **冲突演进悖论** — 知识更新后旧建议仍存，AI 自相矛盾
4. **隐私与智能悖论** — 云端记忆无法通过企业合规审查
5. **经验沉淀悖论** — 隐性经验散落日志，无法结构化复用

**现有解决方案的不足：**

| 方案 | 核心缺陷 |
|------|----------|
| 无限上下文窗口 | "大海捞针"效应，推理精度断崖式下降 |
| 传统 RAG | 仅处理静态文档，无法捕获动态交互经验 |
| ChatGPT Memory | 永久存储无衰退、无冲突检测、云端隐私风险 |
| mem0 原版 | 被动存取、每次 add 需 2 次 LLM 调用（高成本）、无衰退机制、无隐私脱敏 |
| 手动 MEMORY.md | 人工维护成本高、无结构化、无自动化 |

### Why Now?

- **AI Agent 爆发期**：2025-2026 年是 AI Agent 从概念到落地的关键窗口，记忆能力是 Agent 产品化的核心基础设施
- **mem0 已验证市场需求**：mem0 开源项目的快速增长证明了 AI 记忆赛道的真实需求
- **ACE 理论成熟**：ACE Framework 的设计方案已完成理论验证，具备工程化实施条件
- **Local-First 趋势**：企业对数据主权的要求日益严格，本地优先的记忆方案有明确的市场窗口

### Impact if Unsolved

- AI 产品始终停留在"无状态工具"阶段，无法形成用户粘性
- 开发者被迫在每个产品中重复发明记忆轮子，浪费大量工程资源
- 企业因隐私顾虑无法采用 AI 记忆方案，错失智能化升级机会
- 团队缺乏核心技术护城河，在 AI 基础设施赛道中被动

---

## Target Audience

### Primary Users

**AI 产品开发者**
- 正在构建 AI 应用（CLI 助手、IDE 插件、桌面 AI、Web 平台、Agent 框架）的技术团队
- 技术水平：中高级 Python 开发者，熟悉 LLM 生态
- 痛点：需要一个开箱即用、成本可控、隐私安全的 AI 记忆引擎
- 当前行为：手动管理 MEMORY.md / 使用 mem0 原版 / 自建简陋记忆系统

**企业客户**
- 正在进行 AI 智能化升级的企业技术团队
- 需要私有化部署、数据本地存储、符合合规要求的记忆解决方案
- 痛点：云端 SaaS 记忆方案（如 mem0 Cloud）无法满足数据主权要求

### Secondary Users

- **内部产品线**：团队自身的 AI 产品（ACEST Desktop、OpenClaw 等）作为 MemX 的首要集成用户
- **开源贡献者**：对 AI 记忆技术感兴趣的开发者社区
- **研究人员**：研究 AI 长期记忆、知识管理的学术/工业界研究者

### User Needs

1. **零配置快速启动** — 安装即用，合理默认值覆盖 90% 场景，不需要深入了解内部实现
2. **成本可控** — 默认模式下零 LLM 调用成本，可按需开启 LLM 增强
3. **数据主权** — 所有记忆数据存储在用户本地，零云端依赖，支持隐私自动脱敏

---

## Solution Overview

### Proposed Solution

在 mem0 开源项目之上叠加 ACE 智能层，形成 **MemX** 产品。核心策略是"ACE 的智能层 + mem0 的基础设施层"：

- **保留** mem0 全部基础设施（23 个 VectorStore、13+ LLM/Embedding Provider、图存储、Reranker、异步支持、Cloud Client、OpenMemory UI、TypeScript SDK）
- **新增** ACE 四大引擎：Reflector（知识蒸馏）、Curator（语义去重）、Generator（混合检索）、Decay（衰退遗忘）
- **新增** Integration Layer：Pre-Inference / Post-Action / Session-End 三个自动集成点
- **改造** 检索管线：从纯向量相似度升级为三层混合检索 + 衰退加权
- **扩展** 数据模型：在 mem0 metadata 中嵌入 Bullet 结构化字段

### Key Features

- **规则式知识蒸馏（Reflector）** — 纯规则模式检测 + 隐私脱敏 + 蒸馏为 Bullet，默认零 LLM 成本
- **艾宾浩斯衰退引擎（Decay）** — 知识自然新陈代谢，指数衰退 + 召回强化 + 永久保留阈值
- **混合三层检索（Generator）** — 精确匹配 + 模糊匹配 + 元数据匹配 + 语义向量，综合评分
- **语义去重（Curator）** — cosine similarity ≥ 0.8 自动 Merge，无需 LLM 调用
- **自动集成点（Integration Layer）** — Pre-Inference / Post-Action / Session-End 三时机自动学习与召回
- **Bullet 结构化知识单元** — 每条知识带 section / knowledge_type / instructivity_score / decay_weight 等元数据
- **隐私自动脱敏** — API Key / Token / 密码 / 用户路径自动过滤
- **优雅降级（Graceful Degradation）** — Embedding 不可用时纯关键词检索；Daemon 不可用时直接读存储
- **完全兼容 mem0 API** — `Memory.add() / search() / get_all()` 等公开接口签名不变

### Value Proposition

| 维度 | mem0 原版 | **MemX** |
|------|-----------|----------|
| 知识来源 | API 式手动存取 | **自动蒸馏** |
| 时效维护 | 永久存储 | **指数衰退 + 召回强化** |
| 冲突检测 | 无 | **语义冲突检测** |
| 检索质量 | 纯语义向量 | **混合三层检索 + 衰退加权** |
| 隐私安全 | 云端 / 本地均可 | **完全本地 + 自动脱敏** |
| 去重成本 | 每次 add 需 2 次 LLM 调用 | **cosine similarity 自动 Merge（零 LLM 成本）** |
| 默认成本 | 高（每条记忆 2000-5000 tokens） | **零（Rules-only 默认）** |

---

## Business Objectives

### Goals

- **开源影响力**：发布后 6 个月内获得 1000+ GitHub Star，建立 AI 记忆引擎赛道的技术品牌
- **PyPI 发布**：作为可安装的 Python 包发布到 PyPI（`pip install memx`），降低使用门槛
- **内部产品赋能**：为团队自身的 AI 产品线（ACEST Desktop、OpenClaw 等）提供核心记忆引擎
- **企业客户获取**：发布后 12 个月内获得 3-5 个企业客户试用/付费
- **商业化路径**：建立 MemX Enterprise 付费模式（私有化部署 + 高级功能 + 技术支持）

### Success Metrics

- **技术指标**：
  - 端到端检索延迟 < 50ms（5000 条记忆规模）
  - 默认模式零 LLM API 调用
  - 蒸馏规则命中率 > 70%（3-5 个会话后用户感知到历史经验被召回）
- **产品指标**：
  - PyPI 月下载量
  - GitHub Star 增长率
  - 社区 Issue / PR 活跃度
- **商业指标**：
  - 企业客户数量
  - 付费转化率
  - 技术品牌影响力（技术博客引用、会议演讲邀请）

### Business Value

- **短期**：为内部 AI 产品线提供差异化记忆能力，形成技术壁垒
- **中期**：通过开源建立行业影响力，吸引开发者生态
- **长期**：以 MemX Enterprise 实现商业变现，成为 AI 记忆基础设施的标准方案

---

## Scope

### In Scope

**Phase 1 — MVP（核心引擎）**
- Bullet 结构化数据模型（嵌入 mem0 metadata）
- Reflector 规则式蒸馏引擎（Rules-only 模式 + 隐私脱敏）
- Decay 衰退引擎（艾宾浩斯公式 + 召回强化 + 永久保留）
- 混合检索引擎（关键词 + 语义 + 衰退加权）
- Curator 语义去重（cosine similarity 自动 Merge）
- MemX 配置系统扩展（RetrievalConfig / ReflectorConfig / DecayConfig / PrivacyConfig）

**Phase 2 — 集成层**
- Integration Points 抽象接口（Pre-Inference / Post-Action / Session-End）
- ONNX Embedding Provider（本地 Embedding 能力）
- Daemon 常驻进程模式（IPC 通信、生命周期管理）
- 用户交互界面（CLI 命令 / Skill）

**Phase 3 — 高级功能**
- LLM 增强蒸馏（Reflector LLM-assisted / LLM-distill 模式）
- 语义冲突检测
- 层级 Scope 管理（global / project:{name}）
- 导入 / 导出（Playbook 可迁移）
- PyPI 发布 + 文档站点

### Out of Scope

- 移动端 SDK（iOS / Android）
- 多用户/团队实时协作记忆（当前仅支持单用户 Local-First）
- 自建 Embedding 模型训练
- 与 mem0 Cloud SaaS 的双向同步
- 非 Python 语言的原生 SDK 重写（TypeScript SDK 保留 mem0 现有）
- UI 重新设计（保留 OpenMemory UI 现有）

### Future Considerations

- 团队知识共享模式（多用户记忆合并与权限控制）
- 多语言 Embedding 模型支持（尤其中文优化）
- MemX Cloud 托管服务（企业 SaaS 模式）
- 与主流 Agent 框架（LangChain / CrewAI / AutoGen）的深度集成包
- VS Code / Cursor 插件形态发布
- 记忆可视化分析仪表板

---

## Key Stakeholders

- **TPY（项目负责人 / 技术决策者）** — 影响力：高。项目创始人，负责技术方向和商业决策。
- **开发团队（2-5 人）** — 影响力：高。负责具体工程实现，对技术选型有直接话语权。
- **早期用户 / Beta 测试者** — 影响力：中。提供产品反馈，影响功能优先级。
- **mem0 开源社区** — 影响力：中。Fork 关系需要维护，社区反馈影响项目声誉。
- **潜在企业客户** — 影响力：中。其需求影响 Enterprise 功能规划和商业化路径。

---

## Constraints and Assumptions

### Constraints

- **必须 Local-First**：核心记忆数据必须存储在用户本地，不强制依赖云端服务（LLM 评估为可选例外）
- **必须兼容 mem0 API**：`Memory.add() / search() / get_all() / get() / update() / delete()` 等公开接口签名保持不变，ACE 功能通过配置开关启用
- **零 LLM 成本默认**：默认模式（Rules-only Reflector + cosine Curator）不产生任何 LLM API 调用费用
- **mem0 开源协议约束**：需遵守 mem0 项目的开源许可证条款
- **小团队资源**：2-5 人的开发资源，需合理排期

### Assumptions

- mem0 项目将保持积极维护，不会突然停止更新或更改许可证
- all-MiniLM-L6-v2 ONNX 模型能满足多语言（尤其中英文）的基础语义检索需求
- 5000 条记忆规模内，SQLite + 内存 brute-force 的性能可满足 <50ms 检索目标
- AI 产品市场对"自适应记忆"能力有持续增长的需求
- 企业客户愿意为 Local-First 的隐私安全记忆方案付费

---

## Success Criteria

- **MVP 可用性**：在团队自身的 AI 产品中实际跑通 Reflector + Decay + 混合检索闭环，3-5 个会话后用户能感知到历史经验被自动召回
- **工程质量**：核心模块测试覆盖率 > 80%，检索延迟 < 50ms，无 P0 级 Bug
- **PyPI 发布成功**：`pip install memx` 可正常安装并运行，文档完整
- **社区认可**：GitHub 发布后首月获得 100+ Star，有外部贡献者提交 Issue/PR
- **API 兼容验证**：mem0 现有用户可以无缝迁移到 MemX，零代码修改即可运行

---

## Timeline and Milestones

### Target Launch

- **MVP 内部可用**：2026 Q2（约 2-3 个月）
- **PyPI 公开发布**：2026 Q3
- **Enterprise 版本**：2026 Q4

### Key Milestones

- **M1 — 基础框架搭建**（2 周）：Fork mem0、建立 MemX 项目结构、Bullet 数据模型定义、配置系统扩展
- **M2 — Reflector + Privacy**（3 周）：规则式蒸馏引擎、隐私脱敏模块、4-Stage 流水线
- **M3 — Decay + Curator**（2 周）：衰退引擎、语义去重、生命周期管理
- **M4 — Generator 改造**（3 周）：混合三层检索、衰退加权评分、Token 预算控制、降级模式
- **M5 — Integration Layer**（3 周）：三个集成点抽象、CLI Hook 实现、Daemon 模式
- **M6 — ONNX + 测试**（2 周）：ONNX Embedding Provider、全模块集成测试、性能基准测试
- **M7 — 高级功能**（4 周）：LLM 增强蒸馏、冲突检测、层级 Scope、导入导出
- **M8 — 发布准备**（2 周）：PyPI 打包、文档站点、README、CHANGELOG、许可证处理

---

## Risks and Mitigation

- **Risk:** mem0 上游频繁更新导致 Fork 同步成本高
  - **Likelihood:** High
  - **Mitigation:** 采用"新增模块独立存在"策略，不修改 mem0 现有 Provider 代码；建立定期 rebase 流程；核心改造集中在新增目录（`mem0/reflector/`, `mem0/curator/`, `mem0/decay/`, `mem0/integration/`），最小化与上游的冲突面

- **Risk:** 竞争对手推出类似的增强型记忆方案
  - **Likelihood:** Medium
  - **Mitigation:** 加速 MVP 发布抢占市场窗口；通过 ACE 理论框架建立技术壁垒；深度绑定自身产品线形成生态护城河

- **Risk:** Rules-only 蒸馏质量不足，用户感知不到价值
  - **Likelihood:** Medium
  - **Mitigation:** MVP 阶段重点打磨规则引擎的模式检测能力；预留 LLM-assisted 模式作为质量兜底；收集用户反馈持续迭代规则库

- **Risk:** 小团队资源不足以支撑 Level 3 项目全部计划
  - **Likelihood:** Medium
  - **Mitigation:** 严格按 Phase 分期交付，Phase 1 MVP 优先保证质量；非核心功能（冲突检测、导入导出）可延后；利用 mem0 现有基础设施减少重复劳动

- **Risk:** 开源许可证冲突或法律风险
  - **Likelihood:** Low
  - **Mitigation:** 在项目启动前确认 mem0 许可证条款（当前为 Apache 2.0），确保 Fork 和商业化路径合规

---

## Next Steps

1. 创建产品需求文档（PRD） - `/prd`
2. 进行技术架构设计 - `/architecture`
3. 用户研究（可选） - `/research`

---

**This document was created using BMAD Method v6 - Phase 1 (Analysis)**

*To continue: Run `/workflow-status` to see your progress and next recommended workflow.*
