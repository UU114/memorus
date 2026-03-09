# STORY-072: 实现 Mandatory 逃生舱

**Epic:** EPIC-012 (Team 治理与高级功能)
**Priority:** Could Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 7

---

## User Story

As a developer on a legacy project
I want to override mandatory team rules locally
So that my project isn't blocked by rules that don't apply

---

## Description

### Background
Team Pool 中 `enforcement: "mandatory"` 的 Bullet 在 Shadow Merge 时跳过加权直接优先。但在遗留项目中，某些强制规则可能不适用（如"所有新接口必须用 gRPC"对纯 REST 的旧项目无效）。本 Story 提供安全的逃生舱机制，允许本地覆盖 mandatory 规则，同时保持审计和自动恢复。

### Scope
**In scope:**
- `mandatory_overrides` 配置支持 `bullet_id`, `reason`, `expires`
- `reason` 和 `expires` 为必填字段（无理由不允许覆盖）
- 过期后自动恢复 mandatory 行为
- 偏离时 Generator 注入偏离提示到上下文
- Federation Mode 下偏离事件审计上报

**Out of scope:**
- 团队管理员远程强制解除 override
- 偏离审批工作流（Server 端）

### User Flow
1. 开发者发现某个 mandatory 团队规则不适用于当前项目
2. 在 TeamConfig 中添加 `mandatory_overrides` 条目
3. 必须填写 `reason`（为什么不适用）和 `expires`（有效期，最长 90 天）
4. 生效后，Shadow Merge 不再强制优先该 Bullet
5. Generator 检索到该 Bullet 时注入偏离提示："注意：你的项目已覆盖此团队规则 [reason]"
6. Federation Mode 下，偏离事件异步上报 Team Server
7. 到期后自动恢复 mandatory 行为，Generator 恢复正常优先

---

## Acceptance Criteria

- [ ] `mandatory_overrides` 配置支持 `bullet_id`, `reason`, `expires` 字段
- [ ] `reason` 和 `expires` 为必填字段（校验失败 → 配置加载报错）
- [ ] `expires` 最长 90 天，超过 → 配置加载报错
- [ ] 过期后自动恢复 mandatory 行为（每次检索时检查有效期）
- [ ] 偏离时 Generator 注入偏离提示到上下文
- [ ] Federation Mode 下偏离事件审计上报 Team Server（异步，失败不阻塞）
- [ ] 偏离提示格式清晰，包含 bullet_id、reason、expires

---

## Technical Notes

### Components
- **File:** `memx/team/config.py` — `MandatoryOverride` 模型（已有签名，补充验证逻辑）
- **File:** `memx/team/merger.py` — MultiPoolRetriever 修改，检查 override 有效性
- **File:** `memx/team/sync_client.py` — 审计上报接口

### MandatoryOverride Model
```python
class MandatoryOverride(BaseModel):
    bullet_id: str
    reason: str  # required, non-empty
    expires: datetime  # required, max 90 days from now

    @validator("reason")
    def reason_not_empty(cls, v):
        if not v.strip():
            raise ValueError("reason is required for mandatory override")
        return v

    @validator("expires")
    def expires_within_90_days(cls, v):
        max_date = datetime.now() + timedelta(days=90)
        if v > max_date:
            raise ValueError("mandatory override cannot exceed 90 days")
        return v

    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires
```

### Shadow Merge Override Logic
```python
# In MultiPoolRetriever._shadow_merge():
def _should_enforce_mandatory(self, team_bullet: TeamBullet) -> bool:
    """Check if mandatory enforcement should apply."""
    if team_bullet.enforcement != "mandatory":
        return False
    # Check for active override
    override = self._find_override(team_bullet.id)
    if override and not override.is_expired:
        self._inject_deviation_hint(team_bullet, override)
        self._report_deviation(team_bullet, override)  # async, non-blocking
        return False  # override active, don't enforce
    return True  # no override or expired, enforce normally
```

### Deviation Hint Injection
```python
DEVIATION_HINT_TEMPLATE = (
    "[OVERRIDE] 你的项目已覆盖团队规则 [{bullet_id}]: {reason} "
    "(有效期至 {expires})"
)
```

### Audit Report
```python
async def report_deviation(self, bullet_id: str, reason: str, expires: datetime):
    """Report deviation to Team Server (async, non-blocking)."""
    try:
        await self._client.post("/api/v1/audit/deviation", json={
            "bullet_id": bullet_id,
            "reason": reason,
            "expires": expires.isoformat(),
            "user_id": self._config.user_alias,
        })
    except Exception:
        pass  # non-blocking, log warning
```

### Edge Cases
- 配置中 bullet_id 不存在于 Team Pool → 静默忽略（Bullet 可能尚未同步）
- 多个 override 同一 bullet_id → 取最后一个
- override 过期后立即恢复 → 无需 restart
- Server 审计上报失败 → 仅日志 WARNING，不影响本地行为

---

## Dependencies

**Prerequisite Stories:**
- STORY-056: MultiPoolRetriever + Shadow Merge
- STORY-049: TeamConfig 独立配置（MandatoryOverride 子模型）

**Blocked Stories:**
- STORY-073: 治理集成测试

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] MandatoryOverride 校验测试（reason 必填、expires 最长 90 天）
  - [ ] Shadow Merge 正确跳过被 override 的 mandatory Bullet
  - [ ] 过期后自动恢复测试
  - [ ] 偏离提示注入测试
  - [ ] 审计上报测试（Mock Server）
  - [ ] 边界情况测试（不存在的 bullet_id、重复 override）
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **MandatoryOverride 校验:** 0.5 point
- **Shadow Merge override 逻辑:** 1 point
- **偏离提示注入 + 审计上报:** 0.5 point
- **测试:** 1 point
- **Total:** 3 points

**Rationale:** 逻辑清晰、影响面小，主要修改 merger.py 和 config.py。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
