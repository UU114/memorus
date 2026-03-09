# STORY-054: 实现 GitFallbackStorage — JSONL 只读加载

**Epic:** EPIC-010 (Git Fallback 团队记忆)
**Priority:** Should Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 5

---

## User Story

As a team member
I want to `git clone` a repo and automatically get team knowledge
So that I benefit from team experience without any setup

---

## Description

### Background
Git Fallback 是 Team Memory 的零配置方案。团队知识以 `.ace/playbook.jsonl` 文件形式随仓库分发，开发者 `git clone` 后自动获得团队经验。这是最简单的 Team Memory 入口，不需要 Server、不需要配置，只要文件存在就工作。

`GitFallbackStorage` 实现 `StorageBackend` Protocol，提供只读的 TeamBullet 加载和检索能力。

### Scope
**In scope:**
- `GitFallbackStorage` 类实现 `StorageBackend` Protocol
- `.ace/playbook.jsonl` 解析（每行一个 TeamBullet JSON）
- 首行 Header 解析（Embedding 模型指纹）
- 模型指纹不匹配时降级
- 严格只读
- 文件不存在时返回空结果

**Out of scope:**
- 向量缓存（STORY-055）
- 去重缓存（STORY-057）
- 写入 playbook.jsonl

### User Flow
1. 开发者 `git clone` 一个包含 `.ace/playbook.jsonl` 的仓库
2. MemX 初始化时自动检测到文件
3. search 结果中自动包含 Git Fallback 的团队知识
4. 开发者无需任何配置

---

## Acceptance Criteria

- [ ] `GitFallbackStorage` 实现 `StorageBackend` Protocol（至少 `search()` 方法）
- [ ] 正确解析 `.ace/playbook.jsonl`（每行一个 TeamBullet JSON）
- [ ] 首行 Header 解析 Embedding 模型指纹（`{"_header": true, "model": "...", "dim": 384}`）
- [ ] 模型指纹不匹配时降级为纯关键词检索（WARNING 日志）
- [ ] 严格只读：无任何写入 `.ace/playbook.jsonl` 的代码路径
- [ ] 文件不存在时返回空结果（不报错）
- [ ] 支持 UTF-8 编码
- [ ] 无效 JSON 行跳过并记录 WARNING

---

## Technical Notes

### Components
- `memx/team/git_storage.py` — GitFallbackStorage 实现

### JSONL Format

```jsonl
{"_header": true, "model": "all-MiniLM-L6-v2", "dim": 384, "version": "1.0"}
{"content": "Always use --locked with cargo build", "section": "rust", "knowledge_type": "Method", "instructivity_score": 85, "schema_version": 2, "author_id": "anon-abc123", "enforcement": "suggestion", "tags": ["rust", "cargo"]}
{"content": "Never commit .env files to git", "section": "security", "knowledge_type": "Pitfall", "instructivity_score": 95, "schema_version": 2, "enforcement": "mandatory", "tags": ["security", "git"], "incompatible_tags": []}
```

### Implementation Sketch

```python
# memx/team/git_storage.py
from pathlib import Path
import json
import logging
from memx.team.types import TeamBullet

logger = logging.getLogger(__name__)

class GitFallbackStorage:
    """Read-only storage backend for .ace/playbook.jsonl."""

    def __init__(self, playbook_path: Path | None = None):
        self._path = playbook_path or self._find_playbook()
        self._bullets: list[TeamBullet] = []
        self._header: dict | None = None
        self._loaded = False

    def _find_playbook(self) -> Path | None:
        """Walk up from cwd to git root looking for .ace/playbook.jsonl."""
        current = Path.cwd()
        for parent in [current, *current.parents]:
            candidate = parent / ".ace" / "playbook.jsonl"
            if candidate.exists():
                return candidate
            if (parent / ".git").exists():
                break
        return None

    def _ensure_loaded(self):
        """Lazy load on first access."""
        if self._loaded:
            return
        self._loaded = True
        if not self._path or not self._path.exists():
            return

        with self._path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON at %s:%d, skipping", self._path, lineno)
                    continue

                if data.get("_header"):
                    self._header = data
                    continue

                try:
                    bullet = TeamBullet(**data)
                    if bullet.is_active:
                        self._bullets.append(bullet)
                except Exception:
                    logger.warning("Invalid TeamBullet at %s:%d, skipping", self._path, lineno)

        logger.info("Loaded %d team bullets from %s", len(self._bullets), self._path)

    def search(self, query: str, limit: int = 10, **kwargs) -> list[TeamBullet]:
        """Keyword search against loaded bullets."""
        self._ensure_loaded()
        # Simple keyword matching (vector search in STORY-055)
        query_lower = query.lower()
        scored = []
        for bullet in self._bullets:
            content_lower = bullet.content.lower()
            if query_lower in content_lower:
                scored.append((bullet, 1.0))
            else:
                # Check tag match
                tags = getattr(bullet, "tags", [])
                if any(query_lower in t.lower() for t in tags):
                    scored.append((bullet, 0.5))

        scored.sort(key=lambda x: (-x[1], -x[0].instructivity_score))
        return [b for b, _ in scored[:limit]]
```

### Edge Cases
- 空 playbook.jsonl → 返回空结果
- playbook.jsonl 仅有 Header 行 → 返回空结果
- 非 UTF-8 编码文件 → UnicodeDecodeError 捕获，WARNING 日志
- 超大文件（>10000 行）→ 全量加载到内存（后续由 STORY-057 去重优化）

---

## Dependencies

**Prerequisite Stories:**
- STORY-048: 重构 memx/ → memx/core/（Team 目录结构）
- STORY-050: TeamBullet 数据模型（隐含依赖 STORY-051）

**Blocked Stories:**
- STORY-055: Git Fallback 向量缓存
- STORY-057: 读时去重 + playbook.cache
- STORY-058: Git Fallback 集成测试

---

## Definition of Done

- [ ] `GitFallbackStorage` 实现 StorageBackend Protocol
- [ ] JSONL 解析正确（Header + Bullet 行）
- [ ] 模型指纹检测和降级逻辑
- [ ] 只读约束验证
- [ ] 单元测试覆盖：正常加载、空文件、无效 JSON、文件不存在
- [ ] mypy --strict 通过
- [ ] ruff check 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **JSONL 解析 + 数据加载:** 2 points
- **StorageBackend Protocol 实现:** 1.5 points
- **Header 解析 + 降级逻辑:** 0.5 points
- **测试:** 1 point
- **Total:** 5 points

**Rationale:** 文件 I/O + 数据解析 + Protocol 适配，逻辑清晰但需要处理多种文件格式边界。

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
