# STORY-073: Team 治理集成测试

**Epic:** EPIC-012 (Team 治理与高级功能)
**Priority:** Could Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 7

---

## User Story

As a QA engineer
I want governance features thoroughly tested
So that voting, supersede, and taxonomy work correctly

---

## Description

### Background
Sprint 7 实现了完整的 Team 治理能力（三层审核、Supersede、Taxonomy、Mandatory 逃生舱）。本 Story 创建综合性集成测试套件，验证所有治理功能端到端正确工作，确保功能上线质量。

### Scope
**In scope:**
- 投票（upvote/downvote）→ effective_score 调整测试
- Supersede 流程端到端测试
- Taxonomy 标签对齐测试
- Mandatory 逃生舱过期后自动恢复测试
- 敏感标签强制 Curator 审核标记测试
- 治理功能与 Shadow Merge 集成测试

**Out of scope:**
- Server 端治理逻辑测试（独立项目）
- 性能基准测试（已有 test_bench_team_search.py 覆盖）
- UI/UX 测试

### Test Strategy
采用 Mock ACE Sync Server fixture（复用 Sprint 6 的 Mock Server），测试覆盖治理功能的完整客户端行为。每个测试用例独立可运行，不依赖外部服务。

---

## Acceptance Criteria

- [ ] 测试：投票正确调整 effective_score（upvote +5, downvote -10）
- [ ] 测试：Supersede 流程端到端（检测 → 用户确认 → 脱敏 → 上传）
- [ ] 测试：Supersede 拒绝时仅 Local 保留
- [ ] 测试：Taxonomy 对齐正确应用（精确匹配、别名、向量兜底）
- [ ] 测试：Mandatory 逃生舱过期后自动恢复
- [ ] 测试：敏感标签强制 Curator 审核标记
- [ ] 测试：偏离提示正确注入上下文
- [ ] 测试：审计上报失败不阻塞本地行为

---

## Technical Notes

### Test Files
- **File:** `tests/unit/team/test_governance.py` — 治理分类逻辑单元测试
- **File:** `tests/unit/team/test_supersede.py` — Supersede 检测和提交单元测试
- **File:** `tests/unit/team/test_taxonomy.py` — Taxonomy 加载和对齐单元测试
- **File:** `tests/unit/team/test_mandatory_override.py` — 逃生舱单元测试
- **File:** `tests/integration/test_team_governance.py` — 治理功能集成测试

### Test Fixtures
```python
@pytest.fixture
def mock_ace_server():
    """Mock ACE Sync Server with governance endpoints."""
    # Reuse Sprint 6 mock server fixture
    # Add: POST /api/v1/bullets/{id}/vote
    # Add: POST /api/v1/bullets/supersede
    # Add: GET /api/v1/taxonomy
    # Add: POST /api/v1/audit/deviation
    ...

@pytest.fixture
def sample_taxonomy():
    """Preset taxonomy for testing."""
    return TagTaxonomy(
        version=1,
        categories={
            "languages": ["python", "rust", "typescript"],
            "frameworks": ["react", "django", "fastapi"],
            "domains": ["security", "architecture", "testing"],
        },
        aliases={
            "reactjs": "react", "py": "python",
            "ts": "typescript", "k8s": "kubernetes",
        },
    )

@pytest.fixture
def mandatory_team_bullet():
    """A mandatory TeamBullet for override testing."""
    return TeamBullet(
        id="team-mandatory-001",
        content="All new APIs must use gRPC",
        enforcement="mandatory",
        metadata=BulletMetadata(
            tags=["architecture"],
            instructivity_score=95,
        ),
    )
```

### Key Test Cases

#### 1. Voting Tests
```python
def test_upvote_increases_effective_score(team_cache, mock_server):
    """upvote should increase effective_score by +5."""

def test_downvote_decreases_effective_score(team_cache, mock_server):
    """downvote should decrease effective_score by -10."""

def test_duplicate_vote_is_idempotent(team_cache, mock_server):
    """Voting on same bullet twice should be idempotent."""
```

#### 2. Supersede Tests
```python
def test_supersede_detection(reflector, team_bullets):
    """Should detect when local bullet corrects a team bullet."""

def test_supersede_submit_flow(nominator, redactor, mock_server):
    """Full flow: detect → redact → user confirm → upload."""

def test_supersede_reject_keeps_local_only(nominator):
    """Rejecting supersede should keep local bullet, no upload."""

def test_stale_override_notification(cache_storage):
    """After team bullet update, notify user about stale local override."""
```

#### 3. Taxonomy Tests
```python
def test_taxonomy_exact_match(taxonomy):
    """Exact tag should be preserved."""

def test_taxonomy_alias_normalization(taxonomy):
    """'reactjs' should normalize to 'react'."""

def test_taxonomy_vector_fallback(taxonomy, embedder):
    """Similar tag (>0.9) should normalize to canonical."""

def test_taxonomy_no_match_keeps_original(taxonomy):
    """Unrecognized tag should be preserved as-is."""

def test_taxonomy_unavailable_degrades_gracefully(reflector):
    """When no taxonomy available, tag generation unchanged."""
```

#### 4. Mandatory Override Tests
```python
def test_override_skips_mandatory_enforcement(merger, override_config):
    """Active override should prevent mandatory enforcement."""

def test_override_expired_restores_mandatory(merger, expired_override):
    """Expired override should restore mandatory behavior."""

def test_override_requires_reason(config):
    """Override without reason should raise validation error."""

def test_override_max_90_days(config):
    """Override expiring beyond 90 days should raise validation error."""

def test_deviation_hint_injected(generator, override_config):
    """Generator should inject deviation hint when override active."""

def test_audit_report_failure_non_blocking(merger, mock_server_down):
    """Audit report failure should not block local behavior."""
```

#### 5. Governance Classification Tests
```python
def test_sensitive_tag_requires_curator(classifier):
    """Bullet with 'security' tag → curator_required."""

def test_high_score_non_sensitive_auto_approve(classifier):
    """score >= 90 + non-sensitive → auto_approve."""

def test_default_p2p_review(classifier):
    """Other cases → p2p_review."""

def test_auto_approve_low_initial_weight(cache_storage):
    """Auto-approved bullets start with 0.5x weight."""
```

---

## Dependencies

**Prerequisite Stories:**
- STORY-069: 三层审核治理（客户端）
- STORY-070: Supersede 知识纠正
- STORY-071: Tag Taxonomy 标签归一化
- STORY-072: Mandatory 逃生舱

**Blocked Stories:** None (Sprint 7 最后一个 Story)

**External Dependencies:** None

---

## Definition of Done

- [ ] 所有测试文件创建完成
- [ ] 单元测试覆盖率 ≥ 80%（治理相关代码）
- [ ] 集成测试覆盖全部 8 个验收标准
- [ ] 全部测试在 CI 中通过
- [ ] 测试执行时间 < 30 秒（不含网络 I/O）
- [ ] Mock Server fixture 复用 Sprint 6 版本并扩展
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **单元测试（governance/supersede/taxonomy/override）:** 2 points
- **集成测试:** 1 point
- **Mock Server 扩展 + Fixture:** 1 point
- **Total:** 4 points

**Rationale:** 测试代码量较大但逻辑直接，主要工作在用例设计和 fixture 准备。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
