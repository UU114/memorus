# STORY-031: Generator 单元测试全覆盖

**Epic:** EPIC-005 (Generator 混合检索引擎)
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 3

---

## User Story

As a QA engineer
I want Generator engine fully tested
So that search quality and degradation are guaranteed with regression safety

---

## Description

### Background
Sprint 2 已完成 Generator 全部组件（ExactMatcher, FuzzyMatcher, MetadataMatcher, VectorSearcher, ScoreMerger, GeneratorEngine, TokenBudgetTrimmer, RetrievalPipeline）。各组件有各自的单元测试，但缺少系统级覆盖率审查和端到端集成测试。本 Story 补齐测试缺口，确保 > 85% 覆盖率。

### Scope
**In scope:**
- 每个 Matcher 的中英文混合输入测试
- ScoreMerger 权重计算精确性测试
- GeneratorEngine 降级模式切换测试
- RetrievalPipeline 端到端集成测试（add→search→reinforce）
- 覆盖率报告生成

**Out of scope:**
- 性能基准测试（STORY-045）
- 大数据量压力测试

---

## Acceptance Criteria

- [ ] ExactMatcher 测试覆盖：纯英文、纯中文、中英混合、空查询、特殊字符
- [ ] FuzzyMatcher 测试覆盖：词干还原、bigram 匹配、边界输入
- [ ] MetadataMatcher 测试覆盖：tools/entities/tags 三种字段、前缀匹配、空 metadata
- [ ] VectorSearcher 测试覆盖：正常结果、空结果、异常降级、两种 mem0 输出格式
- [ ] ScoreMerger 测试覆盖：权重归一化、degraded 模式自动调权、recency boost 计算
- [ ] GeneratorEngine 端到端：full 模式 + degraded 模式 + 单 Matcher 故障隔离
- [ ] RetrievalPipeline 集成测试：Generator→Trimmer→Reinforcer 完整流程 + fallback 路径
- [ ] 覆盖率 > 85%（`memorus/engines/generator/` + `memorus/pipeline/retrieval.py` + `memorus/utils/token_counter.py`）
- [ ] 所有测试 < 5s 完成（无真实网络调用）

---

## Technical Notes

### Test Files
- `tests/unit/test_exact_matcher.py` — 补充中英混合、边界用例
- `tests/unit/test_fuzzy_matcher.py` — 补充词干还原验证
- `tests/unit/test_metadata_matcher.py` — 补充多字段交叉
- `tests/unit/test_vector_searcher.py` — 补充异常路径
- `tests/unit/test_score_merger.py` — 补充降级权重
- `tests/unit/test_generator_engine.py` — 补充端到端
- `tests/unit/test_retrieval_pipeline.py` — 补充 fallback 路径
- `tests/unit/test_token_counter.py` — 补充 CJK 估算精度
- `tests/unit/test_text_processing.py` — 补充词干还原表

### Testing Strategy
- 使用 `unittest.mock.MagicMock` 模拟 VectorSearcher 和 mem0 search_fn
- 使用 `pytest.approx` 验证浮点分数计算
- 使用 `pytest-cov` 生成覆盖率报告

### Dependencies on Existing Code
- `memorus/engines/generator/` — 全部 Matcher、ScoreMerger、GeneratorEngine
- `memorus/pipeline/retrieval.py` — RetrievalPipeline
- `memorus/utils/token_counter.py` — TokenBudgetTrimmer
- `memorus/utils/text_processing.py` — tokenize/stem 工具函数

---

## Dependencies

**Prerequisite Stories:**
- STORY-030: RetrievalPipeline ✓（已完成）
- STORY-023~029: Generator 全组件 ✓（已完成）

**Blocked Stories:**
- None

---

## Definition of Done

- [ ] 所有测试文件补充完毕
- [ ] `pytest tests/unit/test_*matcher*.py tests/unit/test_score_merger.py tests/unit/test_generator_engine.py tests/unit/test_retrieval_pipeline.py tests/unit/test_token_counter.py tests/unit/test_text_processing.py` 全部通过
- [ ] 覆盖率 > 85%
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **补充 Matcher 测试:** 1.5 points
- **ScoreMerger + GeneratorEngine 测试:** 1.5 points
- **RetrievalPipeline 集成测试:** 1 point
- **覆盖率补齐:** 1 point
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
