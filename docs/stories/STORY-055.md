# STORY-055: 实现 Git Fallback 向量缓存

**Epic:** EPIC-010 (Git Fallback 团队记忆)
**Priority:** Should Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 5

---

## User Story

As a developer
I want vector search on Git Fallback knowledge
So that semantic matching works for team knowledge too

---

## Description

### Background
STORY-054 的 `GitFallbackStorage` 初始实现仅支持关键词检索。为了提供与 Local Pool 相同质量的语义检索，需要为 Git Fallback 的 TeamBullet 生成向量缓存。

向量缓存生成一次，存储在 `.ace/playbook.vec`（gitignored），后续检索直接使用内存中的向量索引，零磁盘 I/O。

### Scope
**In scope:**
- 首次加载时自动生成向量缓存
- 缓存文件 `.ace/playbook.vec`
- 缓存过期检测（playbook.jsonl 修改时间比较）
- ONNXEmbedder 集成
- `.ace/.gitignore` 自动维护
- 向量缓存常驻内存

**Out of scope:**
- 去重缓存（STORY-057）
- 远程 Embedding API 调用（仅 ONNX 本地）

### User Flow
1. 首次 `search()` 调用时检测无缓存 → 自动生成向量
2. 后续 `search()` → 从内存向量索引直接检索
3. `git pull` 后 playbook.jsonl 更新 → 下次 search 自动重建缓存

---

## Acceptance Criteria

- [ ] 首次加载时自动生成 `.ace/playbook.vec`（gitignored）
- [ ] 向量缓存与 playbook.jsonl 的修改时间比较，过期时自动重建
- [ ] 使用 ONNXEmbedder（如可用）生成向量，不可用时跳过向量生成
- [ ] `.ace/.gitignore` 自动创建/更新，包含 `playbook.vec` 和 `playbook.cache`
- [ ] 向量缓存加载后常驻内存，后续检索零磁盘 I/O
- [ ] 向量检索使用 cosine similarity
- [ ] 向量不可用时降级为关键词检索（STORY-054 的逻辑）
- [ ] mypy --strict 通过

---

## Technical Notes

### Components
- `memx/team/git_storage.py` — GitFallbackStorage 向量缓存扩展

### Vector Cache Format
```python
# .ace/playbook.vec — numpy binary format
{
    "model": "all-MiniLM-L6-v2",
    "dim": 384,
    "source_mtime": 1709856000.0,  # playbook.jsonl modification time
    "vectors": np.ndarray  # shape: (N, dim), float32
    "bullet_ids": list[str]  # aligned with vectors
}
```

### Implementation Sketch

```python
class GitFallbackStorage:
    # ... existing code from STORY-054 ...

    def _build_vector_cache(self):
        """Generate vector embeddings for all loaded bullets."""
        try:
            from memx.core.engines.onnx_embedder import ONNXEmbedder
            embedder = ONNXEmbedder()
        except (ImportError, Exception):
            logger.info("ONNX embedder not available, vector search disabled")
            self._vectors = None
            return

        texts = [b.content for b in self._bullets]
        vectors = embedder.encode(texts)  # np.ndarray (N, dim)
        self._vectors = vectors

        # Save cache
        self._save_vector_cache(vectors)

    def _save_vector_cache(self, vectors):
        """Persist vector cache to .ace/playbook.vec."""
        cache_path = self._path.parent / "playbook.vec"
        np.savez_compressed(
            cache_path,
            vectors=vectors,
            source_mtime=self._path.stat().st_mtime,
        )
        self._ensure_gitignore()

    def _load_vector_cache(self) -> bool:
        """Try loading existing vector cache. Returns True if valid."""
        cache_path = self._path.parent / "playbook.vec"
        if not cache_path.exists():
            return False
        data = np.load(cache_path, allow_pickle=True)
        if data["source_mtime"] != self._path.stat().st_mtime:
            return False  # stale cache
        self._vectors = data["vectors"]
        return True

    def search(self, query: str, limit: int = 10, **kwargs):
        """Search with vector similarity if available, else keyword."""
        self._ensure_loaded()

        if self._vectors is not None:
            return self._vector_search(query, limit)
        return self._keyword_search(query, limit)

    def _vector_search(self, query: str, limit: int):
        """Cosine similarity search."""
        query_vec = self._embedder.encode([query])[0]
        sims = np.dot(self._vectors, query_vec) / (
            np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_vec)
        )
        top_indices = np.argsort(sims)[-limit:][::-1]
        return [self._bullets[i] for i in top_indices if sims[i] > 0.3]

    def _ensure_gitignore(self):
        """Ensure .ace/.gitignore contains cache files."""
        gitignore_path = self._path.parent / ".gitignore"
        entries = {"playbook.vec", "playbook.cache"}
        existing = set()
        if gitignore_path.exists():
            existing = set(gitignore_path.read_text().splitlines())
        missing = entries - existing
        if missing:
            with gitignore_path.open("a") as f:
                for entry in missing:
                    f.write(f"\n{entry}")
```

### Performance Considerations
- 向量生成是一次性开销（首次加载）
- 后续检索纯内存计算，预期 < 5ms
- 缓存文件使用 numpy compressed format，磁盘占用小

### Edge Cases
- ONNXEmbedder 未安装 → 永久降级为关键词检索
- playbook.jsonl 为空 → 不生成向量缓存
- 磁盘空间不足无法写入缓存 → WARNING 日志，继续内存使用
- Header 中模型指纹与 ONNXEmbedder 不匹配 → 降级关键词检索

---

## Dependencies

**Prerequisite Stories:**
- STORY-054: GitFallbackStorage JSONL 加载
- ONNXEmbedder (STORY-036，Sprint 3)

**Blocked Stories:**
- STORY-058: Git Fallback 集成测试

---

## Definition of Done

- [ ] 向量缓存自动生成和加载
- [ ] 缓存过期检测和自动重建
- [ ] ONNXEmbedder 不可用时优雅降级
- [ ] `.ace/.gitignore` 自动维护
- [ ] cosine similarity 向量检索实现
- [ ] 单元测试覆盖：缓存生成、加载、过期、降级
- [ ] mypy --strict 通过
- [ ] ruff check 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **向量缓存生成 + 序列化:** 2 points
- **向量检索实现:** 1.5 points
- **缓存过期 + 降级逻辑:** 1 point
- **测试:** 0.5 points
- **Total:** 5 points

**Rationale:** 涉及 numpy 向量运算、缓存管理、ONNXEmbedder 集成，技术复杂度中等。

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
