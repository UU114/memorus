# STORY-053: 解耦验证测试套件

**Epic:** EPIC-009 (Core/Team 解耦重构)
**Priority:** Must Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 5

---

## User Story

As a Memorus maintainer
I want automated tests that verify Core/Team decoupling
So that future changes don't accidentally introduce coupling

---

## Description

### Background
Core/Team 解耦是 Memorus Team Memory 架构的核心原则（NFR-014）。需要自动化测试确保依赖方向严格单向（Core ← Team，不可反向），且 Core 在 Team 不存在时功能完全正常。

### Scope
**In scope:**
- CI 静态检查：Core 中不 import Team
- 测试：Core 独立运行
- 测试：Team 功能禁用时 Core 100% 通过
- 集成到 CI Pipeline

**Out of scope:**
- Team Layer 的功能测试（后续 Story）
- 性能回归测试

### User Flow
1. 开发者提交代码到 PR
2. CI 自动运行解耦验证
3. 如果 Core 中引入了 Team 依赖 → CI 失败，阻止合并
4. 如果 Core 测试在 Team 移除后失败 → CI 失败

---

## Acceptance Criteria

- [ ] CI 静态检查：`memorus/core/` 中无 `from memorus.team` 或 `import memorus.team`
- [ ] 测试：`pip install memorus`（无 team extra）→ Core 功能正常
- [ ] 测试：Team 功能禁用时所有 Core 测试 100% 通过
- [ ] 测试：删除 `memorus/team/` 后 Core 行为不变
- [ ] 集成到 CI Pipeline（GitHub Actions / pytest marker）
- [ ] 检查报告清晰指出违规的文件和行号

---

## Technical Notes

### Components
- `tests/unit/test_decoupling.py` — 解耦验证测试
- `.github/workflows/ci.yml` — CI 集成（或 pytest configuration）

### Implementation

```python
# tests/unit/test_decoupling.py
import ast
import os
from pathlib import Path

CORE_DIR = Path("memorus/core")
FORBIDDEN_IMPORTS = {"memorus.team", "from memorus.team", "from memorus import team"}


class TestDecoupling:
    """Verify Core/Team decoupling invariants."""

    def test_core_does_not_import_team(self):
        """Static check: no Team imports in Core code."""
        violations = []
        for py_file in CORE_DIR.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("memorus.team"):
                            violations.append(f"{py_file}:{node.lineno} import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("memorus.team"):
                        violations.append(f"{py_file}:{node.lineno} from {node.module}")

        assert not violations, f"Core imports Team:\n" + "\n".join(violations)

    def test_core_functions_without_team(self, tmp_path, monkeypatch):
        """Core works when memorus.team is not importable."""
        import sys
        # Block team imports
        monkeypatch.setitem(sys.modules, "memorus.team", None)

        from memorus.core.memory import Memory
        # Memory should initialize without error
        # (actual test depends on Memory constructor signature)

    def test_ext_bootstrap_handles_missing_team(self):
        """team_bootstrap gracefully handles missing Team package."""
        from memorus.ext.team_bootstrap import try_bootstrap_team
        # Should return False, not raise
        result = try_bootstrap_team(None)
        assert result is False
```

### CI Integration
```yaml
# In CI workflow
- name: Decoupling Check
  run: |
    python -m pytest tests/unit/test_decoupling.py -v
    # Optional: grep-based fast check
    ! grep -rn "from memorus.team\|import memorus.team" memorus/core/
```

---

## Dependencies

**Prerequisite Stories:**
- STORY-048: 重构 memorus/ → memorus/core/
- STORY-052: ext/team_bootstrap.py

**Blocked Stories:**
- None

---

## Definition of Done

- [ ] `test_decoupling.py` 测试文件创建
- [ ] AST 级别的静态 import 检查实现
- [ ] Core 独立运行测试实现
- [ ] Bootstrap 缺失处理测试实现
- [ ] CI Pipeline 集成
- [ ] 所有测试通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **AST 静态检查:** 1 point
- **Core 独立运行测试:** 1 point
- **CI 集成:** 1 point
- **Total:** 3 points

**Rationale:** 测试逻辑本身不复杂，但需要仔细设计测试场景确保解耦约束可靠检测。

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
