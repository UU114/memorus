# STORY-011: 实现 PrivacySanitizer（Stage 3）

**Epic:** EPIC-002 — Reflector 知识蒸馏引擎
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a privacy-conscious user
I want sensitive data automatically stripped from memories
So that API keys and passwords are never stored

---

## Description

### Background
PrivacySanitizer 是 Reflector 的 Stage 3，也是 MemX 安全架构中最关键的组件。它负责从记忆内容中自动检测并脱敏敏感信息（API Key、密码、token、用户路径等）。根据架构决策（NFR-003），Sanitizer 是 **hardcoded safety net**——其核心 pattern 列表不可通过配置关闭或移除，只能追加自定义 pattern。即使 Reflector 引擎整体被禁用，Sanitizer 仍可独立运行。

### Scope

**In scope:**
- ≥ 10 种 API Key 格式检测和脱敏
- 用户路径替换
- 密码字段脱敏
- SanitizeResult 返回值（clean content + filtered items 列表）
- hardcoded 内置 pattern（不可配置移除）
- 自定义 pattern 追加接口
- 单元测试（每种格式 ≥ 2 用例）

**Out of scope:**
- IngestPipeline 集成（STORY-014, STORY-015）
- 自然语言隐私检测（如"我的密码是 xxx"）
- PII 检测（姓名、电话、身份证号）— 未来功能

### Sensitive Data Patterns

| # | Pattern | Example | Replacement |
|---|---------|---------|-------------|
| 1 | OpenAI API Key | `sk-proj-abc123...` | `<OPENAI_KEY>` |
| 2 | Anthropic API Key | `sk-ant-api03-...` | `<ANTHROPIC_KEY>` |
| 3 | GitHub Token | `ghp_xxxx`, `gho_xxxx`, `github_pat_` | `<GITHUB_TOKEN>` |
| 4 | AWS Access Key | `AKIA...` (20 chars) | `<AWS_KEY>` |
| 5 | AWS Secret Key | 40-char base64 after aws_secret | `<AWS_SECRET>` |
| 6 | Generic Bearer Token | `Bearer eyJ...` | `<BEARER_TOKEN>` |
| 7 | Generic API Key param | `api_key=xxx`, `apikey=xxx` | `api_key=<REDACTED>` |
| 8 | Password fields | `password=xxx`, `passwd=`, `secret=` | `password=<REDACTED>` |
| 9 | Private Key blocks | `-----BEGIN.*PRIVATE KEY-----` | `<PRIVATE_KEY>` |
| 10 | User path (Windows) | `C:\Users\JohnDoe\...` | `<USER_PATH>\...` |
| 11 | User path (Unix) | `/home/johndoe/...`, `/Users/johndoe/` | `<USER_PATH>/...` |
| 12 | Database URL with creds | `postgres://user:pass@host` | `postgres://<REDACTED>@host` |

---

## Acceptance Criteria

- [ ] 检测并脱敏 ≥ 10 种 API Key/secret 格式
- [ ] OpenAI key (`sk-proj-*`, `sk-*` 50+ chars) 正确脱敏
- [ ] Anthropic key (`sk-ant-*`) 正确脱敏
- [ ] GitHub token (`ghp_*`, `gho_*`, `github_pat_*`) 正确脱敏
- [ ] AWS key (`AKIA*` 20 chars) 正确脱敏
- [ ] Bearer token 正确脱敏
- [ ] 密码字段（password=, secret=, token=）值脱敏
- [ ] 私钥 PEM block 整体替换
- [ ] 含用户名路径（Windows/Unix）替换为 `<USER_PATH>`
- [ ] Database URL 中的认证信息脱敏
- [ ] 返回 `SanitizeResult(clean_content: str, filtered_items: list[FilteredItem])`
- [ ] FilteredItem 记录：pattern_name, original_snippet（截断前8后4字符）, position
- [ ] 内置 pattern 为 hardcoded，`PrivacyConfig.custom_patterns` 仅允许追加
- [ ] 尝试通过配置移除内置 pattern → 无效（仍会运行）
- [ ] 单元测试：每种敏感信息格式 ≥ 2 用例（检测到 + 正常文本不误判）

---

## Technical Notes

### File Locations
- `memx/privacy/sanitizer.py` — PrivacySanitizer 类
- `memx/privacy/patterns.py` — 内置 pattern 定义

### Implementation Sketch

```python
import re
from dataclasses import dataclass

@dataclass
class FilteredItem:
    pattern_name: str
    snippet: str  # first 8 + "..." + last 4 chars
    position: int

@dataclass
class SanitizeResult:
    clean_content: str
    filtered_items: list[FilteredItem]
    was_modified: bool

class PrivacySanitizer:
    """Hardcoded privacy safety net. Cannot be disabled via config."""

    def __init__(self, custom_patterns: list[tuple[str, str, str]] = None):
        self._patterns = self._builtin_patterns()
        if custom_patterns:
            self._patterns.extend(custom_patterns)

    @staticmethod
    def _builtin_patterns() -> list[tuple[str, str, str]]:
        """Return (name, regex_pattern, replacement). HARDCODED."""
        return [
            ("openai_key", r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}", "<OPENAI_KEY>"),
            ("anthropic_key", r"sk-ant-(?:api\d+-)?[A-Za-z0-9_-]{20,}", "<ANTHROPIC_KEY>"),
            ("github_token", r"(?:ghp_|gho_|github_pat_)[A-Za-z0-9_]{20,}", "<GITHUB_TOKEN>"),
            ("aws_access_key", r"AKIA[A-Z0-9]{16}", "<AWS_KEY>"),
            ("bearer_token", r"Bearer\s+eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "<BEARER_TOKEN>"),
            ("password_field", r"(?:password|passwd|secret|token)\s*[=:]\s*\S+", lambda m: m.group().split("=")[0] + "=<REDACTED>"),
            ("private_key", r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----", "<PRIVATE_KEY>"),
            ("win_user_path", r"[A-Z]:\\Users\\[^\\]+", "<USER_PATH>"),
            ("unix_user_path", r"(?:/home/|/Users/)[^/\s]+", "<USER_PATH>"),
            ("db_url_creds", r"((?:postgres|mysql|mongodb)(?:ql)?://)([^:]+):([^@]+)@", r"\1<REDACTED>:<REDACTED>@"),
        ]

    def sanitize(self, content: str) -> SanitizeResult:
        filtered = []
        clean = content
        for name, pattern, replacement in self._patterns:
            for match in re.finditer(pattern, clean):
                snippet = self._truncate_match(match.group())
                filtered.append(FilteredItem(name, snippet, match.start()))
            clean = re.sub(pattern, replacement, clean)
        return SanitizeResult(clean, filtered, was_modified=len(filtered) > 0)

    @staticmethod
    def _truncate_match(text: str) -> str:
        if len(text) <= 16:
            return text[:4] + "..." + text[-4:]
        return text[:8] + "..." + text[-4:]
```

### Security Considerations
- Pattern 顺序影响替换结果（如 Bearer token 中可能含 sk- 开头的部分）→ 按特异性从高到低排序
- 正则不能太贪婪——避免误匹配正常文本
- 路径替换不应影响代码中的常量路径（如 `/usr/bin/`）→ 仅替换含用户名的路径

### Edge Cases
- 一段文本中含多种敏感信息 → 全部脱敏
- 敏感信息在多行文本中 → 正则需支持多行
- 非 ASCII 用户名路径（如中文用户名）→ 需要宽字符匹配
- `password=` 后跟空格或换行 → 不应替换空字符串

---

## Dependencies

**Prerequisite Stories:**
- STORY-003: PrivacyConfig（custom_patterns 配置）
- STORY-006: 项目骨架

**Blocked Stories:**
- STORY-013: ReflectorEngine（Stage 3 组装）
- STORY-015: Sanitizer safety net（独立于 Reflector 的运行）

**External Dependencies:** None

---

## Definition of Done

- [ ] Code in `memx/privacy/sanitizer.py` and `memx/privacy/patterns.py`
- [ ] ≥ 20 个测试用例（10 格式 × 2 正反例）
- [ ] 所有内置 pattern 有测试覆盖
- [ ] 误判率 < 1%（正常文本不被错误脱敏）
- [ ] `ruff check` + `mypy` 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Pattern definitions:** 1.5 points
- **Sanitizer logic:** 1.5 points
- **Tests (20+ cases):** 2 points
- **Total:** 5 points

**Rationale:** Pattern 数量多，每个需要精确正则，且需要大量正反例测试确保不误判。

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
