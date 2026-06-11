# STORY-045: 性能基准测试套件

**Epic:** EPIC-008 (用户界面与发布)
**Priority:** Must Have
**Story Points:** 4
**Status:** Done
**Assigned To:** Developer
**Created:** 2026-02-27
**Sprint:** 4

---

## User Story

As a developer
I want automated performance benchmarks
So that we catch performance regressions in CI

---

## Description

### Background
Memorus 的 PRD 定义了明确的性能要求：
- **NFR-001**: 检索延迟 < 50ms（5000 条记忆）
- **蒸馏延迟**: 单次 Reflector < 20ms（rules 模式）
- **ONNX embed**: 单条 < 10ms

当前 `tests/performance/` 目录已存在但为空。需要构建完整的基准测试套件，使用 `pytest-benchmark` 集成到 CI 门禁中。

### Scope
**In scope:**
- 混合检索（GeneratorEngine）延迟基准
- Reflector 蒸馏延迟基准
- ONNX Embedding 延迟基准（条件测试，需安装 onnxruntime）
- Curator 去重延迟基准
- Decay sweep 延迟基准
- IngestPipeline / RetrievalPipeline 端到端基准
- pytest-benchmark 集成 + JSON 输出
- CI 门禁配置（超标则失败）

**Out of scope:**
- 负载测试 / 并发压测
- 内存占用 profiling
- 跨平台性能对比
- Daemon IPC 延迟基准（不稳定，依赖系统状态）

---

## Acceptance Criteria

- [ ] `tests/performance/test_bench_generator.py` — GeneratorEngine.search() 5000 条记忆 < 50ms（p95）
- [ ] `tests/performance/test_bench_reflector.py` — ReflectorEngine.reflect() 单次 < 20ms（p95）
- [ ] `tests/performance/test_bench_curator.py` — CuratorEngine.curate() 100 条现有 + 1 候选 < 10ms（p95）
- [ ] `tests/performance/test_bench_decay.py` — DecayEngine.sweep() 5000 条 < 30ms（p95）
- [ ] `tests/performance/test_bench_onnx.py` — ONNXEmbedder.embed() 单条 < 10ms（p95，skipif 无 onnxruntime）
- [ ] `tests/performance/test_bench_pipeline.py` — IngestPipeline + RetrievalPipeline 端到端基准
- [ ] 所有基准测试使用 `pytest-benchmark` fixture
- [ ] `pytest --benchmark-json=benchmark.json` 输出可解析的 JSON 报告
- [ ] CI 门禁：任一基准超标 → pytest 失败（使用 `--benchmark-max-time` 或自定义 assert）
- [ ] conftest.py 提供共享 fixture：生成 N 条模拟 Bullet 数据

---

## Technical Notes

### Test Files
- `tests/performance/__init__.py`
- `tests/performance/conftest.py` — 共享 fixture
- `tests/performance/test_bench_generator.py`
- `tests/performance/test_bench_reflector.py`
- `tests/performance/test_bench_curator.py`
- `tests/performance/test_bench_decay.py`
- `tests/performance/test_bench_onnx.py`
- `tests/performance/test_bench_pipeline.py`

### Shared Fixtures

```python
# tests/performance/conftest.py
import pytest
from datetime import datetime, timedelta
import random
from memorus.types import BulletMetadata, BulletSection, KnowledgeType
from memorus.engines.generator.engine import BulletForSearch, MetadataInfo

@pytest.fixture
def generate_bullets():
    """Factory fixture: generate N mock bullets for benchmarking."""
    def _generate(n: int = 5000) -> list[BulletForSearch]:
        sections = list(BulletSection)
        types = list(KnowledgeType)
        bullets = []
        for i in range(n):
            bullets.append(BulletForSearch(
                bullet_id=f"bench-{i:06d}",
                content=f"This is benchmark bullet {i} about {random.choice(['pytest', 'git', 'docker', 'vim', 'rust'])} with details on usage patterns and common pitfalls",
                metadata=MetadataInfo(
                    related_tools=[random.choice(["pytest", "git", "docker", "vim", "cargo"])],
                    key_entities=[f"entity_{i % 100}"],
                    tags=[random.choice(["testing", "devops", "editor", "lang"])],
                ),
                created_at=datetime.utcnow() - timedelta(days=random.randint(0, 90)),
                decay_weight=random.uniform(0.1, 1.0),
                extra={},
            ))
        return bullets
    return _generate
```

### Benchmark Examples

```python
# tests/performance/test_bench_generator.py
import pytest
from memorus.engines.generator.engine import GeneratorEngine
from memorus.config import RetrievalConfig

def test_generator_search_5000(benchmark, generate_bullets):
    """NFR-001: search latency < 50ms for 5000 bullets."""
    bullets = generate_bullets(5000)
    engine = GeneratorEngine(config=RetrievalConfig(), vector_searcher=None)

    result = benchmark.pedantic(
        engine.search,
        args=("pytest verbose output debugging",),
        kwargs={"bullets": bullets, "limit": 20},
        rounds=10,
        warmup_rounds=2,
    )

    assert len(result) <= 20
    # p95 assertion via benchmark stats
    assert benchmark.stats.stats.mean < 0.050  # 50ms

# tests/performance/test_bench_reflector.py
import pytest
from memorus.engines.reflector.engine import ReflectorEngine
from memorus.types import InteractionEvent
from memorus.config import ReflectorConfig

def test_reflector_rules_mode(benchmark):
    """Reflector rules mode < 20ms per event."""
    engine = ReflectorEngine(config=ReflectorConfig(mode="rules"))
    event = InteractionEvent(
        user_message="How do I run pytest with verbose output?",
        assistant_message="Use pytest -v flag for verbose output. You can also use -vv for extra verbosity.",
        metadata={"tool_name": "bash", "success": True},
    )

    result = benchmark.pedantic(
        engine.reflect,
        args=(event,),
        rounds=50,
        warmup_rounds=5,
    )

    assert benchmark.stats.stats.mean < 0.020  # 20ms
```

### CI Integration

```yaml
# In existing CI config, add benchmark step:
- name: Performance Benchmarks
  run: |
    pytest tests/performance/ \
      --benchmark-enable \
      --benchmark-json=benchmark.json \
      -v
  # pytest-benchmark built-in: --benchmark-max-time=1.0
```

### pytest-benchmark 依赖
- 新增 `pytest-benchmark` 到 `pyproject.toml` 的 `[project.optional-dependencies]` dev 组
- 或直接加入 `[tool.pytest.ini_options]` 配置

### Edge Cases
- CI 机器性能差异 → 使用较宽松阈值（2× 本地），或用 `--benchmark-disable` 跳过门禁
- ONNX 未安装 → `@pytest.mark.skipif` 跳过 ONNX 基准
- 首次运行无历史数据 → 仅记录，不对比

---

## Dependencies

**Prerequisite Stories:**
- STORY-030: RetrievalPipeline（已完成）
- STORY-013: ReflectorEngine（已完成）

**Blocked Stories:**
- STORY-046: PyPI 打包发布（性能基准需在发布前通过）

---

## Definition of Done

- [ ] 6 个基准测试文件全部实现
- [ ] 所有基准在本地通过阈值
- [ ] conftest.py 共享 fixture 可生成 1~10000 条数据
- [ ] `pytest tests/performance/ --benchmark-enable` 可运行
- [ ] benchmark.json 输出可解析
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **conftest + 数据生成 fixture:** 0.5 points
- **Generator + Decay 基准:** 1 point
- **Reflector + Curator 基准:** 1 point
- **ONNX + Pipeline 基准:** 1 point
- **CI 门禁配置 + 阈值调优:** 0.5 points
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
