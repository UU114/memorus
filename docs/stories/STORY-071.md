# STORY-071: 实现 Tag Taxonomy 标签归一化

**Epic:** EPIC-012 (Team 治理与高级功能)
**Priority:** Could Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 7

---

## User Story

As a team member
I want consistent tag naming across the team
So that knowledge is organized and discoverable

---

## Description

### Background
目前 Reflector 蒸馏生成的标签完全自由格式，同一概念可能有多种写法（如 `react`, `reactjs`, `React.js`）。本 Story 实现 Tag Taxonomy 机制：Team Server 维护中心化词表，客户端同步后对齐标签，确保团队知识的标签一致性和可发现性。

### Scope
**In scope:**
- `pull_taxonomy()` 从 Server 下载最新 Taxonomy 到本地
- Taxonomy 本地缓存 `~/.ace/team_cache/{team_id}/taxonomy.json`
- Reflector 蒸馏时标签对齐 Taxonomy（如可用）
- 预设 Taxonomy 模板（rust, python, react, security, architecture, testing 等）
- Git Fallback 支持项目级 `.ace/taxonomy.json`
- 兜底：无 Taxonomy 匹配时向量相似度 ≥ 0.9 视为同一标签

**Out of scope:**
- Server 端 Taxonomy 管理 UI
- 自动 Taxonomy 种子聚合（从高频 tags 提取候选词，未来增强）
- `ace team init` 命令（属于 CLI 扩展）

### User Flow
1. Team Cache 同步时自动拉取最新 Taxonomy
2. Reflector 蒸馏新 Bullet 时，检查标签是否在 Taxonomy 中
3. 标签完全匹配 → 直接使用
4. 标签不匹配 → 向量相似度比较：≥ 0.9 → 替换为 Taxonomy 中的标准标签
5. 标签不匹配且无近似 → 保留原标签（允许新标签产生）

---

## Acceptance Criteria

- [ ] `AceSyncClient.pull_taxonomy()` 正确拉取最新 Taxonomy
- [ ] Taxonomy 缓存到 `~/.ace/team_cache/{team_id}/taxonomy.json`
- [ ] Reflector 蒸馏时标签对齐 Taxonomy 词表（完全匹配优先，向量相似度兜底）
- [ ] 提供预设 Taxonomy 模板（按语言/框架/领域分类）
- [ ] Git Fallback 支持项目级 `.ace/taxonomy.json`
- [ ] 兜底：无 Taxonomy 匹配时向量相似度 ≥ 0.9 视为同一标签
- [ ] Taxonomy 为空或不可用时，标签生成行为不变（降级）

---

## Technical Notes

### Components
- **File:** `memx/team/sync_client.py` — `pull_taxonomy()` 已有接口签名，实现对接
- **File:** `memx/team/taxonomy.py` — 新建，Taxonomy 加载/对齐逻辑
- **File:** `memx/team/cache_storage.py` — 同步时触发 Taxonomy 拉取
- **File:** Reflector 扩展接口 — 标签对齐钩子

### Taxonomy Data Model
```python
@dataclass
class TagTaxonomy:
    version: int
    updated_at: datetime
    categories: dict[str, list[str]]  # category -> canonical tags
    aliases: dict[str, str]  # alias -> canonical tag

    def normalize(self, tag: str) -> str:
        """Normalize a tag using taxonomy."""
        # 1. Exact match in aliases
        if tag in self.aliases:
            return self.aliases[tag]
        # 2. Case-insensitive match
        lower = tag.lower()
        for canonical in self._all_tags():
            if canonical.lower() == lower:
                return canonical
        # 3. No match — return original
        return tag
```

### Preset Templates
```python
PRESET_TAXONOMY = {
    "languages": ["python", "rust", "typescript", "go", "java"],
    "frameworks": ["react", "vue", "django", "fastapi", "actix"],
    "domains": ["security", "architecture", "testing", "devops", "database"],
    "practices": ["error-handling", "performance", "debugging", "logging"],
}

PRESET_ALIASES = {
    "reactjs": "react", "React.js": "react",
    "ts": "typescript", "TS": "typescript",
    "py": "python", "Python3": "python",
    "js": "javascript", "JS": "javascript",
    "k8s": "kubernetes", "K8S": "kubernetes",
}
```

### Vector Similarity Fallback
```python
def align_tag_fuzzy(tag: str, taxonomy: TagTaxonomy, embedder) -> str:
    """Fallback: use vector similarity when no exact/alias match."""
    if embedder is None:
        return tag  # no embedder, keep original
    tag_vec = embedder.embed(tag)
    best_match, best_score = None, 0.0
    for canonical in taxonomy.all_tags():
        score = cosine_similarity(tag_vec, embedder.embed(canonical))
        if score > best_score:
            best_match, best_score = canonical, score
    if best_score >= 0.9:
        return best_match
    return tag  # no match, keep original
```

### Edge Cases
- Taxonomy 文件损坏 → 使用预设模板降级
- Server 不可达时 → 使用本地缓存的 Taxonomy
- 本地缓存和 Server 版本冲突 → Server 版本优先
- Git Fallback 项目级 Taxonomy → 与 Team Taxonomy 合并（项目级优先）

---

## Dependencies

**Prerequisite Stories:**
- STORY-060: AceSyncClient 拉取接口（pull_taxonomy 方法签名）
- STORY-054: GitFallbackStorage（Git Fallback .ace/ 目录结构）

**Blocked Stories:**
- STORY-073: 治理集成测试

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] Taxonomy 加载和缓存测试
  - [ ] 精确匹配标签对齐测试
  - [ ] 别名归一化测试
  - [ ] 向量相似度兜底测试
  - [ ] 预设模板覆盖测试
  - [ ] Git Fallback 项目级 Taxonomy 测试
  - [ ] 降级场景测试（无 Taxonomy、无 embedder）
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **Taxonomy 数据模型 + 加载:** 1 point
- **标签对齐逻辑（精确 + 向量兜底）:** 1.5 points
- **预设模板 + Git Fallback 支持:** 0.5 point
- **测试:** 1 point
- **Total:** 4 points

**Rationale:** 核心是标签对齐算法，数据模型和预设模板相对简单。向量相似度兜底复用已有 embedder。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
