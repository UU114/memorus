# STORY-048: 重构 memx/ → memx/core/ 包结构

**Epic:** EPIC-009 (Core/Team 解耦重构)
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 5

---

## User Story

As a MemX developer
I want the codebase restructured into core/ and team/ packages
So that Core and Team have clear boundaries and independent lifecycles

---

## Description

### Background
当前所有 MemX 代码平铺在 `memx/` 顶层目录下。为了支持 Team Memory 扩展且保持 Core/Team 充分解耦（NFR-014），需要将现有代码迁移到 `memx/core/` 子包，并预留 `memx/team/` 目录。

这是一次纯结构性重构，不修改任何业务逻辑。所有外部 import 通过 `memx/__init__.py` 的重新导出保持向后兼容。

### Scope
**In scope:**
- 所有现有代码从 `memx/` 移动到 `memx/core/`
- 所有内部 import path 更新
- `memx/__init__.py` 顶层导出保持不变
- `memx/team/` 空目录创建
- 全部现有测试通过

**Out of scope:**
- 任何业务逻辑修改
- Team Layer 实现（后续 Story）
- 外部 API 变更

### User Flow
1. 开发者运行 `from memx import Memory` — 仍然正常工作
2. 开发者运行 `from memx.core.memory import Memory` — 新路径也可用
3. 现有脚本和测试无需任何修改

---

## Acceptance Criteria

- [ ] 所有现有代码从 `memx/` 移动到 `memx/core/`
- [ ] 所有 import path 更新完成（内部引用从 `memx.xxx` → `memx.core.xxx`）
- [ ] `memx/__init__.py` 顶层导出保持不变（`from memx import Memory` 仍可用）
- [ ] 全部现有测试通过，零改动
- [ ] `memx/team/` 目录创建（空 `__init__.py`）
- [ ] `memx/core/__init__.py` 正确导出所有公开 API
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Technical Notes

### Components
- `memx/__init__.py` — 重写为从 `memx.core` 重新导出
- `memx/core/` — 所有现有模块迁入
- `memx/core/__init__.py` — 核心包导出
- `memx/team/__init__.py` — 空占位符

### Implementation Strategy
```
1. 创建 memx/core/ 目录
2. 移动所有现有模块到 memx/core/
3. 更新所有内部 import（使用 IDE/sed 批量替换）
4. 更新 memx/__init__.py 为重导出层：
   from memx.core.memory import Memory
   from memx.core.config import MemXConfig
   # ... 等所有公开 API
5. 创建 memx/team/__init__.py（空文件）
6. 运行全部测试验证
```

### Risks
- **影响面大**：所有文件都需要移动，import path 全部更新
- **Mitigation**：通过 `memx/__init__.py` 重导出，确保外部用户零感知

### Edge Cases
- 相对 import（`from .xxx import`）需要逐一检查
- `pyproject.toml` 或 `setup.py` 中的 package 配置可能需要更新
- CI/CD 配置中的路径引用

---

## Dependencies

**Prerequisite Stories:**
- Sprint 1-2 Core 代码完成

**Blocked Stories:**
- STORY-049: TeamConfig 独立配置
- STORY-050: TeamBullet 数据模型
- STORY-052: ext/team_bootstrap.py
- STORY-053: 解耦验证测试
- STORY-054: GitFallbackStorage

---

## Definition of Done

- [ ] 所有文件从 `memx/` 迁移到 `memx/core/`
- [ ] 内部 import path 全部更新
- [ ] `memx/__init__.py` 重导出保持向后兼容
- [ ] `memx/team/__init__.py` 空文件创建
- [ ] 全部现有单元测试通过
- [ ] 全部集成测试通过
- [ ] mypy --strict 通过
- [ ] ruff check 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **文件迁移:** 1 point
- **Import path 批量更新:** 2 points
- **重导出层 + 向后兼容验证:** 1 point
- **测试修复 + CI 验证:** 1 point
- **Total:** 5 points

**Rationale:** 纯结构性重构，逻辑简单但影响面广，需要仔细处理所有 import path 和兼容性。

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
