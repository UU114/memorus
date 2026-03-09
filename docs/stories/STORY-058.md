# STORY-058: Git Fallback 端到端集成测试

**Epic:** EPIC-010 (Git Fallback 团队记忆)
**Priority:** Should Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 5

---

## User Story

As a QA engineer
I want comprehensive integration tests for Git Fallback
So that the feature works reliably end-to-end

---

## Description

### Background
Git Fallback 涉及多个组件协作：GitFallbackStorage（JSONL 加载 + 向量缓存 + 去重缓存）、MultiPoolRetriever（并行查询 + Shadow Merge）、team_bootstrap（条件注入）。需要端到端集成测试验证整个链路。

### Scope
**In scope:**
- 测试夹具：创建测试用 `.ace/playbook.jsonl`
- 端到端检索测试（Local + Team 合并）
- Shadow Merge 行为验证
- mandatory Bullet 优先测试
- 文件不存在降级测试
- 模型指纹不匹配降级测试
- 性能测试

**Out of scope:**
- Federation Mode 测试（Sprint 6）
- UI/CLI 测试

### User Flow
1. CI 运行 `pytest tests/integration/test_team_retrieval.py`
2. 测试创建临时 `.ace/playbook.jsonl` 夹具
3. 验证 search 返回合并结果
4. 验证各种降级场景

---

## Acceptance Criteria

- [ ] 创建测试用 `.ace/playbook.jsonl` 夹具（含 Header + 多条 TeamBullet）
- [ ] 测试：search 返回 Local + Git Fallback 合并结果
- [ ] 测试：Shadow Merge 正确应用 layer boost（Local ×1.5, Team ×1.0）
- [ ] 测试：mandatory Bullet 正确优先（跳过 boost 加权）
- [ ] 测试：playbook.jsonl 不存在时纯 Local 结果
- [ ] 测试：模型指纹不匹配时降级为关键词检索
- [ ] 测试：incompatible_tags 冲突正确保留高分结果
- [ ] 性能测试：Team 检索增量 < 40ms

---

## Technical Notes

### Components
- `tests/integration/test_team_retrieval.py` — 端到端集成测试
- `tests/performance/test_team_search_latency.py` — 性能测试
- `tests/fixtures/` — 测试夹具

### Test Fixtures

```python
# conftest.py
import pytest
import json
from pathlib import Path

@pytest.fixture
def playbook_dir(tmp_path):
    """Create a temporary .ace/ directory with test playbook."""
    ace_dir = tmp_path / ".ace"
    ace_dir.mkdir()

    bullets = [
        {"_header": True, "model": "all-MiniLM-L6-v2", "dim": 384, "version": "1.0"},
        {
            "content": "Always use --locked with cargo build in CI",
            "section": "rust",
            "knowledge_type": "Method",
            "instructivity_score": 85,
            "schema_version": 2,
            "author_id": "anon-001",
            "enforcement": "suggestion",
            "tags": ["rust", "ci"],
            "incompatible_tags": [],
        },
        {
            "content": "Never commit .env files to git",
            "section": "security",
            "knowledge_type": "Pitfall",
            "instructivity_score": 95,
            "schema_version": 2,
            "enforcement": "mandatory",
            "tags": ["security", "git"],
            "incompatible_tags": [],
        },
        {
            "content": "Use snake_case for Python function names",
            "section": "python",
            "knowledge_type": "Preference",
            "instructivity_score": 70,
            "schema_version": 2,
            "enforcement": "suggestion",
            "tags": ["python", "style"],
            "incompatible_tags": ["camelCase"],
        },
    ]

    playbook = ace_dir / "playbook.jsonl"
    with playbook.open("w", encoding="utf-8") as f:
        for bullet in bullets:
            f.write(json.dumps(bullet, ensure_ascii=False) + "\n")

    return tmp_path
```

### Test Cases

```python
class TestGitFallbackIntegration:
    """End-to-end tests for Git Fallback team retrieval."""

    def test_search_returns_merged_results(self, playbook_dir):
        """Local + Team results are merged."""

    def test_shadow_merge_boost(self, playbook_dir):
        """Local results get 1.5x boost, Team gets 1.0x."""

    def test_mandatory_bullet_priority(self, playbook_dir):
        """Mandatory TeamBullets skip boost and appear first."""

    def test_no_playbook_returns_local_only(self, tmp_path):
        """Without playbook.jsonl, only Local results returned."""

    def test_model_mismatch_fallback(self, playbook_dir):
        """Wrong model fingerprint degrades to keyword search."""

    def test_incompatible_tags_conflict(self, playbook_dir):
        """Conflicting tags resolved by keeping higher score."""

    def test_empty_playbook(self, playbook_dir):
        """Empty playbook returns only Local results."""

    def test_invalid_json_lines_skipped(self, playbook_dir):
        """Invalid JSON lines are skipped with warning."""


class TestTeamSearchPerformance:
    """Performance benchmarks for team retrieval."""

    def test_team_search_latency_under_40ms(self, playbook_dir):
        """Team retrieval overhead should be < 40ms."""
        import time
        # ... benchmark code ...
```

### Edge Cases to Test
- playbook.jsonl 仅有 Header → 返回空 Team 结果
- playbook.jsonl 有 1000+ 条 → 性能验证
- 所有 TeamBullet 都是 mandatory → 全部优先返回
- Local 和 Team 有相同内容 → 近似去重（0.95 阈值）

---

## Dependencies

**Prerequisite Stories:**
- STORY-054: GitFallbackStorage JSONL 加载
- STORY-055: Git Fallback 向量缓存
- STORY-056: MultiPoolRetriever + Shadow Merge
- STORY-057: 读时去重 + playbook.cache

**Blocked Stories:**
- None

---

## Definition of Done

- [ ] 测试夹具创建（playbook.jsonl fixtures）
- [ ] 端到端集成测试全部通过
- [ ] 降级场景测试全部通过
- [ ] 性能测试：Team 检索增量 < 40ms
- [ ] 测试报告清晰显示各场景覆盖
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **测试夹具 + conftest:** 1 point
- **集成测试用例（7+ 场景）:** 2 points
- **性能测试:** 1 point
- **Total:** 4 points

**Rationale:** 不涉及生产代码修改，纯测试编写。测试场景多但每个场景逻辑清晰。

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
