# STORY-057: 实现读时去重 + playbook.cache

**Epic:** EPIC-010 (Git Fallback 团队记忆)
**Priority:** Should Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 5

---

## User Story

As a developer
I want Git Fallback knowledge to be automatically deduplicated on load
So that redundant entries don't pollute search results

---

## Description

### Background
`.ace/playbook.jsonl` 可能随时间累积重复或近似重复的条目（例如多次导出、手动编辑合并）。为避免检索结果中出现冗余，需要在首次加载时执行一次性语义去重，并将去重结果缓存到 `.ace/playbook.cache`。

后续加载直接使用缓存，跳过去重计算。缓存与 playbook.jsonl 的修改时间绑定，文件更新后自动重建。

### Scope
**In scope:**
- 首次加载时语义去重
- 去重结果缓存到 `.ace/playbook.cache`
- 缓存过期检测和自动重建
- 日常检索零开销

**Out of scope:**
- playbook.jsonl 文件本身的修改/压缩
- 跨文件去重

### User Flow
1. 首次加载 playbook.jsonl → 执行语义去重 → 写入 playbook.cache
2. 后续加载 → 检测 cache 有效 → 直接使用缓存
3. playbook.jsonl 被修改（git pull 后）→ cache 过期 → 重新去重

---

## Acceptance Criteria

- [ ] 首次加载 playbook.jsonl 时执行一次性语义去重
- [ ] 去重阈值：content 相似度 ≥ 0.90 视为重复，保留 instructivity_score 更高的
- [ ] 去重结果缓存到 `.ace/playbook.cache`（gitignored）
- [ ] 后续加载直接使用缓存，跳过去重计算
- [ ] 缓存过期检测（playbook.jsonl 修改时间变化时重建）
- [ ] 日常检索零开销（去重仅在加载时发生）
- [ ] `.ace/.gitignore` 包含 `playbook.cache`
- [ ] mypy --strict 通过

---

## Technical Notes

### Components
- `memx/team/git_storage.py` — GitFallbackStorage 去重扩展

### Cache Format
```python
# .ace/playbook.cache — JSON format
{
    "source_mtime": 1709856000.0,
    "original_count": 150,
    "deduped_count": 120,
    "deduped_indices": [0, 1, 3, 5, 6, ...]  # indices into original JSONL
}
```

### Deduplication Algorithm

```python
def _deduplicate(self, bullets: list[TeamBullet]) -> list[TeamBullet]:
    """Remove near-duplicate bullets, keeping highest score."""
    if len(bullets) <= 1:
        return bullets

    kept = []
    for bullet in sorted(bullets, key=lambda b: -b.instructivity_score):
        is_dup = False
        for existing in kept:
            sim = self._text_similarity(bullet.content, existing.content)
            if sim >= 0.90:
                is_dup = True
                break
        if not is_dup:
            kept.append(bullet)

    return kept

@staticmethod
def _text_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)
```

### Integration with GitFallbackStorage

```python
def _ensure_loaded(self):
    if self._loaded:
        return
    self._loaded = True
    # ... load all bullets from JSONL ...

    # Try load dedup cache
    if not self._load_dedup_cache():
        # Run deduplication
        original_count = len(self._bullets)
        self._bullets = self._deduplicate(self._bullets)
        self._save_dedup_cache(original_count)
        logger.info(
            "Deduplicated: %d → %d bullets",
            original_count, len(self._bullets),
        )
```

### Performance Considerations
- 去重算法 O(n²)，但仅在首次加载时运行
- 对于 1000 条 bullet，去重耗时约 50-100ms
- 后续加载使用缓存，耗时 < 5ms

### Edge Cases
- 空 playbook → 不创建缓存
- 所有条目都是重复 → 保留 score 最高的一条
- 缓存文件损坏 → 删除缓存，重新去重
- 磁盘写入失败 → WARNING 日志，继续使用内存中的去重结果

---

## Dependencies

**Prerequisite Stories:**
- STORY-054: GitFallbackStorage JSONL 加载

**Blocked Stories:**
- STORY-058: Git Fallback 集成测试

---

## Definition of Done

- [ ] 语义去重算法实现
- [ ] playbook.cache 缓存生成和加载
- [ ] 缓存过期检测和自动重建
- [ ] 集成到 GitFallbackStorage 加载流程
- [ ] 单元测试覆盖：去重效果、缓存命中、缓存过期
- [ ] mypy --strict 通过
- [ ] ruff check 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **去重算法实现:** 1 point
- **缓存管理（序列化/过期检测）:** 1 point
- **集成 + 测试:** 1 point
- **Total:** 3 points

**Rationale:** 算法简单（Jaccard + 贪心），缓存管理模式与 STORY-055 类似，复杂度可控。

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
