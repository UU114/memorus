# STORY-046: PyPI 打包与发布

**Epic:** EPIC-008 (用户界面与发布)
**Priority:** Must Have
**Story Points:** 5
**Status:** Done
**Assigned To:** Developer
**Created:** 2026-02-27
**Sprint:** 4

---

## User Story

As an end user
I want to install Memorus with pip
So that getting started is trivial

---

## Description

### Background
STORY-006 已创建项目骨架和初始 `pyproject.toml`。本 Story 需要：
1. 完善 `pyproject.toml`（版本号、描述、分类器、入口点、完整依赖声明）
2. 编写 README 快速开始指南
3. 创建 CHANGELOG
4. 配置 GitHub Actions 自动发布工作流
5. 确保 `pip install memorus` → `memorus status` 可直接运行

### Scope
**In scope:**
- `pyproject.toml` 完善（包名、版本 1.0.0、分类器、entry_points）
- 可选依赖分组：`[onnx]`, `[graph]`, `[all]`, `[dev]`
- `README.md` 快速开始指南 + 功能概览
- `CHANGELOG.md` 初始版本
- `memorus` CLI entry_point 注册
- `.github/workflows/publish.yml` — PyPI 自动发布
- 本地 `pip install -e .` 验证
- `python -m build` + `twine check` 验证

**Out of scope:**
- conda 包发布
- Docker 镜像
- 文档站（ReadTheDocs / MkDocs）
- API 参考文档自动生成

---

## Acceptance Criteria

- [ ] `pyproject.toml` 包含完整 metadata：name="memorus", version="1.0.0", description, authors, license, python_requires=">=3.9", classifiers
- [ ] `pip install memorus` 安装核心依赖（mem0ai, pydantic, click）
- [ ] `pip install memorus[onnx]` 额外安装 onnxruntime + tokenizers
- [ ] `pip install memorus[all]` 安装所有可选依赖
- [ ] `pip install memorus[dev]` 安装开发依赖（pytest, pytest-benchmark, mypy, ruff）
- [ ] 安装后 `memorus --help` 显示所有命令（entry_point 注册正确）
- [ ] `python -c "from memorus import Memory; m = Memory()"` 成功（零配置启动）
- [ ] `README.md` 包含：项目简介、安装命令、快速开始代码、CLI 示例、与 mem0 对比
- [ ] `CHANGELOG.md` 包含 v1.0.0 初始发布说明
- [ ] `.github/workflows/publish.yml` 在 tag push 时自动发布到 PyPI
- [ ] `python -m build` 成功生成 sdist + wheel
- [ ] `twine check dist/*` 通过（无警告）

---

## Technical Notes

### pyproject.toml 完善

```toml
[project]
name = "memorus"
version = "1.0.0"
description = "Intelligent memory engine for AI agents — mem0 fork with ACE layer"
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.9"
authors = [{name = "TPY"}]
keywords = ["memory", "ai", "agent", "mem0", "knowledge-base"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "mem0ai>=0.1.0",
    "pydantic>=2.0",
    "click>=8.0",
]

[project.optional-dependencies]
onnx = ["onnxruntime>=1.16", "tokenizers>=0.15"]
graph = ["neo4j>=5.0"]
all = ["memorus[onnx]", "memorus[graph]"]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-benchmark>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
    "build",
    "twine",
]

[project.scripts]
memorus = "memorus.cli.main:cli"

[project.urls]
Homepage = "https://github.com/user/memorus"
Documentation = "https://github.com/user/memorus#readme"
Repository = "https://github.com/user/memorus"
Issues = "https://github.com/user/memorus/issues"
Changelog = "https://github.com/user/memorus/blob/main/CHANGELOG.md"

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["memorus*"]
exclude = ["tests*", "docs*", "bmad*"]
```

### GitHub Actions Workflow

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: read

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install build tools
        run: pip install build twine
      - name: Build
        run: python -m build
      - name: Check
        run: twine check dist/*
      - name: Publish
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*
```

### README.md 结构

```markdown
# Memorus — Intelligent Memory for AI Agents

> mem0 fork + ACE intelligent layer: auto-learn, auto-forget, auto-recall

## Features
- Reflector: auto-distill knowledge from AI interactions
- Curator: semantic dedup, no bloated playbook
- Decay: time-based forgetting with recall reinforcement
- Generator: hybrid search (keyword + semantic + metadata)
- Privacy: hardcoded PII sanitization
- ONNX: local embedding, zero cloud dependency

## Install
pip install memorus
pip install memorus[onnx]  # for local embedding

## Quick Start
from memorus import Memory
m = Memory(config={"ace_enabled": True})
m.add([{"role": "user", "content": "..."}], user_id="dev1")
results = m.search("pytest verbose", user_id="dev1")

## CLI
memorus status
memorus search "pytest"
memorus learn "Always use -v flag with pytest"
memorus list --type tool_pattern
```

### Verification Steps
1. `pip install -e ".[dev]"` — 本地可编辑安装
2. `memorus --help` — CLI 可用
3. `python -m build` — 构建包
4. `twine check dist/*` — 包质量检查
5. `pip install dist/memorus-1.0.0-py3-none-any.whl` — 从 wheel 安装验证

### Edge Cases
- 包名 `memorus` 已被占用 → 备选 `memorus-ai` 或 `memorus-memory`
- mem0ai 版本冲突 → 设置宽松版本约束 `>=0.1.0`
- Python 3.9 兼容性 → 避免使用 `X | Y` union 语法（运行时），使用 `from __future__ import annotations`
- Windows 路径 → `pyproject.toml` 中使用 posix 路径

---

## Dependencies

**Prerequisite Stories:**
- STORY-006: 项目骨架（已完成）
- STORY-007: mem0 兼容测试（已完成）
- STORY-045: 性能基准测试（Sprint 4）

**Blocked Stories:**
- None（这是最终交付物）

---

## Definition of Done

- [ ] `pyproject.toml` 完善，所有字段填写
- [ ] `pip install -e .` 成功
- [ ] `memorus --help` 正确显示
- [ ] `python -m build` 生成 sdist + wheel
- [ ] `twine check dist/*` 无警告
- [ ] `README.md` 编写完成
- [ ] `CHANGELOG.md` v1.0.0 编写完成
- [ ] `.github/workflows/publish.yml` 创建
- [ ] 全部测试通过
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **pyproject.toml 完善 + 依赖分组:** 1 point
- **README.md + CHANGELOG.md:** 1.5 points
- **GitHub Actions 发布工作流:** 1 point
- **验证 + 兼容性测试:** 1 point
- **entry_point 注册修正:** 0.5 points *(如需)*
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
