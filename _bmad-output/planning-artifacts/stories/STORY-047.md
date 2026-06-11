# STORY-047: 实现冲突检测（Conflict Detector）

**Epic:** EPIC-008 (用户界面与发布)
**Priority:** Could Have
**Story Points:** 4
**Status:** Done
**Assigned To:** Developer
**Created:** 2026-02-27
**Sprint:** 4

---

## User Story

As a Memorus user
I want to know if my memories contain contradictions
So that I can resolve conflicting knowledge

---

## Description

### Background
当前 `memorus/engines/curator/conflict.py` 为空文件（占位符），`CuratorConfig.conflict_detection: bool = False` 配置项已存在但未被使用。`CuratorEngine.curate()` 完全忽略此配置。

冲突检测的目标是识别"相似但矛盾"的记忆对。典型场景：
- 记忆 A: "Use `--no-cache` flag with pip install"
- 记忆 B: "Never use `--no-cache` with pip, it breaks builds"

这类记忆的 cosine similarity 在 0.5~0.8 之间（相关但不完全重复），且内容语义矛盾。冲突检测不阻塞入库，仅标记冲突对供用户审查。

### Scope
**In scope:**
- `ConflictDetector` 类实现（`memorus/engines/curator/conflict.py`）
- 基于 similarity 范围（0.5~0.8）的冲突候选筛选
- 基于否定词/对立语义的矛盾判断（规则模式）
- `Memory.detect_conflicts()` 公开方法
- CLI `memorus conflicts` 命令
- 冲突结果数据结构 `Conflict`
- `CuratorConfig.conflict_detection` 配置项启用

**Out of scope:**
- LLM 辅助冲突判断
- 自动冲突解决
- 冲突 UI / 交互式解决
- 实时入库阻断

---

## Acceptance Criteria

- [ ] `ConflictDetector.detect(memories)` 扫描所有记忆对，返回 `list[Conflict]`
- [ ] `Conflict` 数据结构包含：`memory_a_id`, `memory_b_id`, `memory_a_content`, `memory_b_content`, `similarity: float`, `reason: str`
- [ ] 仅检测 similarity 在 `[conflict_min, conflict_max]` 范围内的记忆对（默认 0.5~0.8，可配置）
- [ ] 矛盾判断规则：否定词检测（not, never, don't, 不要, 禁止, 避免）+ 对立关键词（always/never, enable/disable, use/avoid）
- [ ] 不阻塞入库流程（detect_conflicts 是独立调用，不在 IngestPipeline 中）
- [ ] `Memory.detect_conflicts(user_id=None)` 公开方法
- [ ] `CuratorConfig.conflict_detection = True` 时，`curate()` 在结果中附带冲突警告（不阻止入库）
- [ ] `memorus conflicts [--json] [--user-id UID]` CLI 命令显示冲突列表
- [ ] 无冲突时返回空列表 / CLI 显示 "No conflicts detected"
- [ ] 大数据集优化：O(n²) 比较使用预过滤（先按 section 分组，仅组内比较）

---

## Technical Notes

### Components
- `memorus/engines/curator/conflict.py` — `ConflictDetector` 类（替换空文件）
- `memorus/engines/curator/engine.py` — `CuratorEngine` 集成冲突检测
- `memorus/memory.py` — `Memory.detect_conflicts()` 方法
- `memorus/config.py` — `CuratorConfig` 新增冲突检测参数
- `memorus/cli/main.py` — `memorus conflicts` 命令

### Data Structures

```python
# memorus/engines/curator/conflict.py
from dataclasses import dataclass, field

@dataclass
class Conflict:
    """A pair of memories that may contradict each other."""
    memory_a_id: str
    memory_b_id: str
    memory_a_content: str
    memory_b_content: str
    similarity: float
    reason: str  # e.g., "Negation detected: 'always' vs 'never'"

@dataclass
class ConflictResult:
    """Result of conflict detection scan."""
    conflicts: list[Conflict] = field(default_factory=list)
    total_pairs_checked: int = 0
    scan_time_ms: float = 0.0
```

### ConflictDetector Implementation

```python
class ConflictDetector:
    # Negation indicators
    NEGATION_WORDS = {
        "en": {"not", "never", "don't", "dont", "shouldn't", "avoid", "disable", "without", "no"},
        "zh": {"不要", "禁止", "避免", "不可", "别", "勿", "不能", "不应"},
    }

    # Opposing pairs
    OPPOSING_PAIRS = [
        ("always", "never"), ("enable", "disable"), ("use", "avoid"),
        ("with", "without"), ("add", "remove"), ("true", "false"),
        ("yes", "no"), ("allow", "deny"), ("open", "close"),
    ]

    def __init__(self, config: CuratorConfig | None = None):
        self._config = config or CuratorConfig()
        self._conflict_min = 0.5  # from config if needed
        self._conflict_max = 0.8

    def detect(self, memories: list[ExistingBullet]) -> ConflictResult:
        """Scan all memory pairs for potential contradictions."""
        conflicts = []
        total_checked = 0

        # Group by section for O(n²/k) instead of O(n²)
        groups = self._group_by_section(memories)

        for section, group_memories in groups.items():
            for i, mem_a in enumerate(group_memories):
                for mem_b in group_memories[i + 1:]:
                    total_checked += 1
                    sim = CuratorEngine.text_similarity(mem_a.content, mem_b.content)

                    if self._conflict_min <= sim <= self._conflict_max:
                        reason = self._check_contradiction(mem_a.content, mem_b.content)
                        if reason:
                            conflicts.append(Conflict(
                                memory_a_id=mem_a.bullet_id,
                                memory_b_id=mem_b.bullet_id,
                                memory_a_content=mem_a.content,
                                memory_b_content=mem_b.content,
                                similarity=sim,
                                reason=reason,
                            ))

        return ConflictResult(conflicts=conflicts, total_pairs_checked=total_checked)

    def _check_contradiction(self, text_a: str, text_b: str) -> str | None:
        """Check if two texts express contradictory information."""
        # 1. Check opposing pairs
        a_lower = text_a.lower()
        b_lower = text_b.lower()

        for word_a, word_b in self.OPPOSING_PAIRS:
            if word_a in a_lower and word_b in b_lower:
                return f"Opposing terms: '{word_a}' vs '{word_b}'"
            if word_b in a_lower and word_a in b_lower:
                return f"Opposing terms: '{word_b}' vs '{word_a}'"

        # 2. Check negation asymmetry
        neg_a = self._count_negations(a_lower)
        neg_b = self._count_negations(b_lower)
        if abs(neg_a - neg_b) >= 1 and (neg_a == 0 or neg_b == 0):
            return f"Negation asymmetry: one statement negates, the other affirms"

        return None

    @staticmethod
    def _group_by_section(memories: list[ExistingBullet]) -> dict[str, list[ExistingBullet]]:
        groups: dict[str, list[ExistingBullet]] = {}
        for mem in memories:
            section = mem.metadata.get("memorus_section", "general")
            groups.setdefault(section, []).append(mem)
        return groups
```

### CuratorEngine Integration

```python
# In CuratorEngine.curate() — optional conflict warning
def curate(self, candidates, existing):
    result = ...  # existing logic

    if self._config.conflict_detection:
        detector = ConflictDetector(self._config)
        # Check new candidates against existing for conflicts
        all_bullets = existing  # Only check existing pool
        conflict_result = detector.detect(all_bullets)
        if conflict_result.conflicts:
            logger.warning(
                "Detected %d potential conflicts in knowledge base",
                len(conflict_result.conflicts),
            )
        result.conflicts = conflict_result.conflicts  # Attach to result

    return result
```

### CuratorConfig Additions

```python
class CuratorConfig(BaseModel):
    similarity_threshold: float = 0.8
    merge_strategy: str = "keep_best"
    conflict_detection: bool = False       # already exists
    conflict_min_similarity: float = 0.5   # NEW
    conflict_max_similarity: float = 0.8   # NEW
```

### CLI Command

```python
@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--user-id", default=None, help="Filter by user ID")
@click.pass_context
def conflicts(ctx, as_json, user_id):
    """Detect contradictory memories."""
    memory = ctx.obj["memory"]
    result = memory.detect_conflicts(user_id=user_id)

    if as_json:
        click.echo(json.dumps([
            {"a_id": c.memory_a_id, "b_id": c.memory_b_id,
             "a": c.memory_a_content, "b": c.memory_b_content,
             "similarity": round(c.similarity, 3), "reason": c.reason}
            for c in result
        ], indent=2, ensure_ascii=False))
    else:
        if not result:
            click.echo("No conflicts detected.")
            return
        click.echo(f"Found {len(result)} potential conflict(s):\n")
        for i, c in enumerate(result, 1):
            click.echo(f"  {i}. [{c.similarity:.2f}] {c.reason}")
            click.echo(f"     A ({c.memory_a_id}): {c.memory_a_content[:80]}")
            click.echo(f"     B ({c.memory_b_id}): {c.memory_b_content[:80]}")
            click.echo()
```

### Dependencies on Existing Code
- `memorus/engines/curator/engine.py:CuratorEngine` — `text_similarity()` 静态方法复用
- `memorus/engines/curator/engine.py:ExistingBullet` — 记忆数据结构
- `memorus/memory.py:Memory` — `get_all()` 获取全部记忆
- `memorus/utils/bullet_factory.py:BulletFactory` — payload 解析

### Edge Cases
- 空数据库 → 返回空 `ConflictResult`
- 仅 1 条记忆 → 无需比较，返回空
- 所有记忆 similarity < 0.5 → 无冲突候选
- 所有记忆 similarity > 0.8 → 不算冲突（属于重复，由 Curator 处理）
- 中英文混合内容 → 否定词同时检查中英文
- 大数据集（>1000 条）→ 按 section 分组预过滤降低复杂度

---

## Dependencies

**Prerequisite Stories:**
- STORY-017: Curator 核心去重（已完成）

**Blocked Stories:**
- None

---

## Definition of Done

- [ ] `ConflictDetector` 类实现并替换空 `conflict.py`
- [ ] `Conflict` 和 `ConflictResult` 数据结构定义
- [ ] 否定词 + 对立关键词检测规则覆盖中英文
- [ ] `Memory.detect_conflicts()` 方法实现
- [ ] `CuratorEngine` 集成冲突警告（`conflict_detection=True` 时）
- [ ] CLI `memorus conflicts` 命令实现
- [ ] 单元测试覆盖：正例矛盾对、反例无矛盾、边界 similarity
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **ConflictDetector 核心逻辑:** 1.5 points
- **矛盾判断规则（否定词 + 对立词）:** 1 point
- **Memory + CuratorEngine 集成:** 0.5 points
- **CLI 命令 + 测试:** 1 point
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
