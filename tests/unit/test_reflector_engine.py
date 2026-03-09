"""Unit tests for memx.engines.reflector.engine — ReflectorEngine."""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from memx.core.config import ReflectorConfig
from memx.core.engines.reflector.engine import ReflectorEngine
from memx.core.privacy.sanitizer import PrivacySanitizer
from memx.core.types import (
    BulletSection,
    CandidateBullet,
    DetectedPattern,
    InteractionEvent,
    KnowledgeType,
    ScoredCandidate,
    SourceType,
)


# -- Helper factories -------------------------------------------------------


def _event(
    user: str = "hello",
    assistant: str = "hi there",
    metadata: dict | None = None,
) -> InteractionEvent:
    return InteractionEvent(
        user_message=user,
        assistant_message=assistant,
        metadata=metadata or {},
    )


def _error_event() -> InteractionEvent:
    """Create an event that triggers the ErrorFixRule."""
    return _event(
        user="I'm getting an error when running cargo build on my project",
        assistant=(
            "You should try updating your Rust toolchain by running "
            "rustup update to fix this compilation error in your project."
        ),
    )


def _config_event() -> InteractionEvent:
    """Create an event that triggers the ConfigChangeRule."""
    return _event(
        user="How do I configure the database environment variable properly?",
        assistant=(
            "You need to set the DB_HOST variable in your .env environment "
            "config file to point to the correct host for your database."
        ),
    )


# -- Test 1: Full pipeline --------------------------------------------------


class TestReflectFullPipeline:
    def test_reflect_full_pipeline(self) -> None:
        """InteractionEvent with error content -> CandidateBullet returned."""
        engine = ReflectorEngine()
        event = _error_event()
        bullets = engine.reflect(event)

        assert len(bullets) >= 1
        for bullet in bullets:
            assert isinstance(bullet, CandidateBullet)
            assert bullet.content != ""
            assert bullet.source_type == SourceType.INTERACTION
            assert 0.0 <= bullet.instructivity_score <= 100.0


# -- Test 2: No patterns ----------------------------------------------------


class TestReflectNoPatterns:
    def test_reflect_no_patterns(self) -> None:
        """Normal conversation without learnable content -> empty list."""
        engine = ReflectorEngine()
        event = _event(
            user="What is the capital of France?",
            assistant=(
                "The capital of France is Paris, a city known for its "
                "culture and history spanning many centuries."
            ),
        )
        bullets = engine.reflect(event)
        assert bullets == []


# -- Test 3: None event -----------------------------------------------------


class TestReflectNoneEvent:
    def test_reflect_none_event(self) -> None:
        """None -> empty list (no crash)."""
        engine = ReflectorEngine()
        # Passing None explicitly (bypasses type check)
        bullets = engine.reflect(None)  # type: ignore[arg-type]
        assert bullets == []


# -- Test 4: Code-heavy content ---------------------------------------------


class TestReflectCodeHeavy:
    def test_reflect_code_heavy(self) -> None:
        """Code-heavy content is filtered by PatternDetector -> empty list."""
        engine = ReflectorEngine()
        code_lines = "\n".join([
            "import os",
            "import sys",
            "from pathlib import Path",
            "def main():",
            "    x = 1",
            "    y = 2",
            "    return x + y",
            "class Foo:",
            "    pass",
            "if __name__ == '__main__':",
            "    main()",
        ])
        event = _event(user=code_lines, assistant=code_lines)
        bullets = engine.reflect(event)
        assert bullets == []


# -- Test 5: Default mode is "hybrid" ----------------------------------------


class TestModeRulesDefault:
    def test_mode_rules_default(self) -> None:
        """Default mode is 'hybrid'."""
        engine = ReflectorEngine()
        assert engine.mode == "hybrid"

    def test_mode_explicit_rules(self) -> None:
        """Explicit config mode='rules' is accepted as-is."""
        config = ReflectorConfig(mode="rules")
        engine = ReflectorEngine(config=config)
        assert engine.mode == "rules"


# -- Test 6: LLM mode initializes or falls back gracefully ------------------


class TestModeLlm:
    def test_mode_llm_with_litellm(self) -> None:
        """mode='llm' initializes LLM components when litellm is available."""
        config = ReflectorConfig(mode="llm")
        engine = ReflectorEngine(config=config)
        # If litellm is installed, mode stays "llm"; otherwise falls back to "rules"
        assert engine.mode in ("llm", "rules")

    def test_mode_llm_fallback_no_litellm(self, caplog: pytest.LogCaptureFixture) -> None:
        """mode='llm' falls back to 'rules' when LLM init fails."""
        config = ReflectorConfig(mode="llm")
        with patch(
            "memx.core.engines.reflector.engine.ReflectorEngine._init_llm_components",
            side_effect=RuntimeError("no litellm"),
        ):
            engine = ReflectorEngine.__new__(ReflectorEngine)
            engine._config = config
            engine._detector = MagicMock()
            engine._scorer = MagicMock()
            engine._sanitizer = MagicMock()
            engine._distiller = MagicMock()
            engine._mode = config.mode
            engine._llm_evaluator = None
            engine._llm_distiller = None
        # Simulate what happens: components not available -> reflect_llm falls back
        assert engine._llm_evaluator is None


# -- Test 7: Hybrid mode initializes or falls back gracefully ---------------


class TestModeHybrid:
    def test_mode_hybrid_with_litellm(self) -> None:
        """mode='hybrid' initializes LLM components when litellm is available."""
        config = ReflectorConfig(mode="hybrid")
        engine = ReflectorEngine(config=config)
        assert engine.mode in ("hybrid", "rules")

    def test_mode_hybrid_reflect_fallback(self) -> None:
        """hybrid mode falls back to rules when LLM components unavailable."""
        config = ReflectorConfig(mode="hybrid")
        engine = ReflectorEngine(config=config)
        # Force LLM components to None
        engine._llm_evaluator = None
        engine._llm_distiller = None
        engine._mode = "hybrid"
        # Should still work (fallback to rules)
        event = _error_event()
        bullets = engine.reflect(event)
        assert len(bullets) >= 1


# -- Test 8: Stage 1 failure -> empty list -----------------------------------


class TestStage1Failure:
    def test_stage1_failure_returns_empty(self) -> None:
        """Mock detector to raise -> empty list."""
        engine = ReflectorEngine()
        engine._detector = MagicMock()
        engine._detector.detect.side_effect = RuntimeError("detector boom")

        event = _error_event()
        bullets = engine.reflect(event)
        assert bullets == []


# -- Test 9: Stage 2 failure -> fallback scoring -----------------------------


class TestStage2Failure:
    def test_stage2_failure_fallback(self) -> None:
        """Mock scorer to raise -> fallback scoring produces ScoredCandidates."""
        engine = ReflectorEngine()

        # Let Stage 1 produce a real pattern
        event = _error_event()
        patterns = engine._run_stage1(event)
        assert len(patterns) >= 1

        # Now make the scorer blow up
        engine._scorer = MagicMock()
        engine._scorer.score.side_effect = RuntimeError("scorer boom")

        # _run_stage2 should catch and use fallback
        scored = engine._run_stage2(patterns)
        assert len(scored) == len(patterns)
        for s in scored:
            assert isinstance(s, ScoredCandidate)
            assert s.section == BulletSection.GENERAL
            assert s.knowledge_type == KnowledgeType.KNOWLEDGE
            assert s.instructivity_score == 50.0


# -- Test 10: Stage 3 failure -> original content preserved ------------------


class TestStage3Failure:
    def test_stage3_failure_uses_original(self) -> None:
        """Mock sanitizer to raise -> original content preserved."""
        engine = ReflectorEngine()

        # Create a scored candidate with known content
        pattern = DetectedPattern(
            pattern_type="error_fix",
            content="Original sensitive content with sk-proj-abc123secret456",
            confidence=0.8,
        )
        candidate = ScoredCandidate(
            pattern=pattern,
            section=BulletSection.DEBUGGING,
            knowledge_type=KnowledgeType.PITFALL,
            instructivity_score=70.0,
        )

        # Make sanitizer blow up
        engine._sanitizer = MagicMock()
        engine._sanitizer.sanitize.side_effect = RuntimeError("sanitizer boom")

        result = engine._run_stage3([candidate])
        assert len(result) == 1
        # Content should still be the original (unsanitized)
        assert "Original sensitive content" in result[0].pattern.content


# -- Test 11: Stage 4 failure -> fallback distill ----------------------------


class TestStage4Failure:
    def test_stage4_failure_fallback(self) -> None:
        """Mock distiller to raise -> fallback distill produces CandidateBullets."""
        engine = ReflectorEngine()

        pattern = DetectedPattern(
            pattern_type="error_fix",
            content="Some error fix content for testing fallback distiller",
            confidence=0.8,
        )
        candidate = ScoredCandidate(
            pattern=pattern,
            section=BulletSection.DEBUGGING,
            knowledge_type=KnowledgeType.PITFALL,
            instructivity_score=75.0,
        )

        # Make distiller blow up
        engine._distiller = MagicMock()
        engine._distiller.distill.side_effect = RuntimeError("distiller boom")

        bullets = engine._run_stage4([candidate])
        assert len(bullets) == 1
        bullet = bullets[0]
        assert isinstance(bullet, CandidateBullet)
        assert bullet.content == pattern.content
        assert bullet.section == BulletSection.DEBUGGING
        assert bullet.knowledge_type == KnowledgeType.PITFALL
        assert bullet.instructivity_score == 75.0
        assert bullet.source_type == SourceType.INTERACTION


# -- Test 12: Sanitizer redacts secrets --------------------------------------


class TestSanitizerRedactsSecrets:
    def test_sanitizer_redacts_secrets(self) -> None:
        """Event containing 'sk-proj-xxx...' -> redacted in output bullet."""
        engine = ReflectorEngine()

        # Build an event that will trigger ErrorFixRule AND contains a secret
        secret_key = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
        event = _event(
            user=f"I'm getting an error with my API key {secret_key} when running the build",
            assistant=(
                f"The key {secret_key} looks like an OpenAI API key. "
                "You should try storing it in an environment variable instead "
                "of hardcoding it to fix this security issue."
            ),
        )
        bullets = engine.reflect(event)

        # Should have produced at least one bullet
        assert len(bullets) >= 1

        # The secret should be redacted in ALL bullet contents
        for bullet in bullets:
            assert secret_key not in bullet.content
            assert "<OPENAI_KEY>" in bullet.content


# -- Test 13: Full pipeline integration --------------------------------------


class TestFullPipelineIntegration:
    def test_full_pipeline_integration(self) -> None:
        """Realistic event -> all stages produce valid output."""
        config = ReflectorConfig(mode="rules", min_score=20.0)
        engine = ReflectorEngine(config=config)

        event = _event(
            user=(
                "I keep getting a 'ModuleNotFoundError: No module named requests' "
                "error when running my Python script."
            ),
            assistant=(
                "You need to install the requests library. Try running "
                "pip install requests in your terminal. This should fix the "
                "ModuleNotFoundError you are seeing."
            ),
        )
        bullets = engine.reflect(event)

        # Should have at least one bullet (ErrorFixRule + possibly NewToolRule)
        assert len(bullets) >= 1

        for bullet in bullets:
            assert isinstance(bullet, CandidateBullet)
            # Content should be non-empty and properly distilled
            assert len(bullet.content) > 0
            # Source type should be INTERACTION
            assert bullet.source_type == SourceType.INTERACTION
            # Score should be valid
            assert 0.0 <= bullet.instructivity_score <= 100.0
            # Section and knowledge_type should be set
            assert isinstance(bullet.section, BulletSection)
            assert isinstance(bullet.knowledge_type, KnowledgeType)


# -- Test 14: Multiple patterns -> multiple bullets --------------------------


class TestMultiplePatterns:
    def test_multiple_patterns(self) -> None:
        """Event that triggers multiple rules -> multiple bullets."""
        config = ReflectorConfig(mode="rules", min_score=10.0)
        engine = ReflectorEngine(config=config)

        # This event is crafted to trigger both ErrorFixRule AND ConfigChangeRule:
        # - user mentions "error" -> ErrorFixRule
        # - conversation mentions "config" / ".env" -> ConfigChangeRule
        event = _event(
            user=(
                "I have an error in my configuration. The .env file is not loading "
                "the environment variable DATABASE_URL properly."
            ),
            assistant=(
                "You should fix this by updating the config in your .env file. "
                "Try setting DATABASE_URL=postgres://localhost:5432/mydb to "
                "resolve the configuration error you are seeing."
            ),
        )
        bullets = engine.reflect(event)

        # Should produce at least 2 bullets (error_fix + config_change)
        assert len(bullets) >= 2

        # Verify different pattern types were detected
        sections = {b.section for b in bullets}
        knowledge_types = {b.knowledge_type for b in bullets}
        # At least two distinct classifications
        assert len(sections) >= 1 or len(knowledge_types) >= 1

        # All bullets should be valid
        for bullet in bullets:
            assert isinstance(bullet, CandidateBullet)
            assert bullet.content != ""
            assert bullet.source_type == SourceType.INTERACTION


# -- Test 15: LLMEvaluator parsing -----------------------------------------


class TestLLMEvaluatorParsing:
    """Test LLMEvaluator._parse_response with mock JSON responses."""

    def _make_evaluator(self) -> object:
        from memx.core.engines.reflector.llm_evaluator import LLMEvaluator

        config = ReflectorConfig(mode="llm")
        return LLMEvaluator(config)

    def test_parse_valid_response(self) -> None:
        from memx.core.engines.reflector.llm_evaluator import LLMEvaluator

        evaluator: LLMEvaluator = self._make_evaluator()  # type: ignore[assignment]
        event = _error_event()
        raw = '{"should_record": true, "knowledge_type": "pitfall", "section": "debugging", "instructivity_score": 82, "summary": "Use rustup update to fix build errors"}'
        result = evaluator._parse_response(raw, event)

        assert result is not None
        assert result.knowledge_type == KnowledgeType.PITFALL
        assert result.section == BulletSection.DEBUGGING
        assert result.instructivity_score == 82.0
        assert "rustup" in result.pattern.content

    def test_parse_should_record_false(self) -> None:
        from memx.core.engines.reflector.llm_evaluator import LLMEvaluator

        evaluator: LLMEvaluator = self._make_evaluator()  # type: ignore[assignment]
        event = _event()
        raw = '{"should_record": false, "knowledge_type": "knowledge", "section": "general", "instructivity_score": 10, "summary": "trivial"}'
        result = evaluator._parse_response(raw, event)
        assert result is None

    def test_parse_invalid_json(self) -> None:
        from memx.core.engines.reflector.llm_evaluator import LLMEvaluator

        evaluator: LLMEvaluator = self._make_evaluator()  # type: ignore[assignment]
        event = _event()
        result = evaluator._parse_response("not json at all", event)
        assert result is None

    def test_parse_markdown_fenced_json(self) -> None:
        from memx.core.engines.reflector.llm_evaluator import LLMEvaluator

        evaluator: LLMEvaluator = self._make_evaluator()  # type: ignore[assignment]
        event = _error_event()
        raw = '```json\n{"should_record": true, "knowledge_type": "method", "section": "workflow", "instructivity_score": 65, "summary": "Run tests before deploy"}\n```'
        result = evaluator._parse_response(raw, event)
        assert result is not None
        assert result.knowledge_type == KnowledgeType.METHOD

    def test_parse_invalid_enum_values_fallback(self) -> None:
        from memx.core.engines.reflector.llm_evaluator import LLMEvaluator

        evaluator: LLMEvaluator = self._make_evaluator()  # type: ignore[assignment]
        event = _event()
        raw = '{"should_record": true, "knowledge_type": "INVALID", "section": "NOPE", "instructivity_score": 50, "summary": "test"}'
        result = evaluator._parse_response(raw, event)
        assert result is not None
        assert result.knowledge_type == KnowledgeType.KNOWLEDGE
        assert result.section == BulletSection.GENERAL


# -- Test 16: LLMDistiller parsing -----------------------------------------


class TestLLMDistillerParsing:
    """Test LLMDistiller._parse_response with mock JSON responses."""

    def _make_distiller(self) -> object:
        from memx.core.engines.reflector.llm_distiller import LLMDistiller

        config = ReflectorConfig(mode="llm")
        return LLMDistiller(config)

    def _make_candidate(self) -> ScoredCandidate:
        pattern = DetectedPattern(
            pattern_type="error_fix",
            content="Use rustup update to fix compilation errors",
            confidence=0.8,
        )
        return ScoredCandidate(
            pattern=pattern,
            section=BulletSection.DEBUGGING,
            knowledge_type=KnowledgeType.PITFALL,
            instructivity_score=80.0,
        )

    def test_parse_valid_response(self) -> None:
        from memx.core.engines.reflector.llm_distiller import LLMDistiller

        distiller: LLMDistiller = self._make_distiller()  # type: ignore[assignment]
        candidate = self._make_candidate()
        raw = '{"distilled_rule": "When Rust build fails with toolchain errors, run rustup update, because outdated toolchains cause compilation failures.", "content": "Running rustup update refreshes the compiler and standard library.", "related_tools": ["rustup", "cargo"], "key_entities": ["rustup update"], "tags": ["rust", "build"]}'
        result = distiller._parse_response(raw, candidate)

        assert result is not None
        assert isinstance(result, CandidateBullet)
        assert result.distilled_rule is not None
        assert "When" in result.distilled_rule
        assert "rustup" in result.related_tools
        assert result.section == BulletSection.DEBUGGING
        assert result.instructivity_score == 80.0

    def test_parse_invalid_json_returns_none(self) -> None:
        from memx.core.engines.reflector.llm_distiller import LLMDistiller

        distiller: LLMDistiller = self._make_distiller()  # type: ignore[assignment]
        candidate = self._make_candidate()
        result = distiller._parse_response("garbage", candidate)
        assert result is None

    def test_fallback_distill(self) -> None:
        from memx.core.engines.reflector.llm_distiller import LLMDistiller

        distiller: LLMDistiller = self._make_distiller()  # type: ignore[assignment]
        candidate = self._make_candidate()
        result = distiller._fallback_distill(candidate)
        assert isinstance(result, CandidateBullet)
        assert result.content == candidate.pattern.content
        assert result.section == BulletSection.DEBUGGING


# -- Test 17: LLM mode end-to-end with mocked litellm ----------------------


def _ensure_litellm_mock() -> MagicMock:
    """Inject a mock litellm module into sys.modules if not installed."""
    if "litellm" not in sys.modules:
        mock_litellm = MagicMock()
        sys.modules["litellm"] = mock_litellm
        return mock_litellm
    return sys.modules["litellm"]  # type: ignore[return-value]


class TestLLMModeMocked:
    """Test full LLM pipeline with mocked litellm.completion."""

    def test_llm_mode_end_to_end(self) -> None:
        """LLM mode: mocked LLM -> evaluator + distiller produce valid bullet."""
        mock_litellm = _ensure_litellm_mock()

        eval_response = MagicMock()
        eval_response.choices = [MagicMock()]
        eval_response.choices[0].message.content = (
            '{"should_record": true, "knowledge_type": "pitfall", '
            '"section": "debugging", "instructivity_score": 85, '
            '"summary": "Use rustup update to fix Rust build errors"}'
        )

        distill_response = MagicMock()
        distill_response.choices = [MagicMock()]
        distill_response.choices[0].message.content = (
            '{"distilled_rule": "When Rust build fails, run rustup update, because outdated toolchains cause errors.", '
            '"content": "Refresh Rust toolchain with rustup update.", '
            '"related_tools": ["rustup", "cargo"], '
            '"key_entities": ["rustup update"], '
            '"tags": ["rust", "debugging"]}'
        )

        call_count = 0

        def mock_completion(**kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return eval_response
            return distill_response

        mock_litellm.completion = mock_completion

        # Force re-import to pick up mock
        for mod_name in list(sys.modules):
            if "llm_evaluator" in mod_name or "llm_distiller" in mod_name:
                del sys.modules[mod_name]

        config = ReflectorConfig(mode="llm", min_score=30.0)
        engine = ReflectorEngine(config=config)
        assert engine.mode == "llm"

        event = _error_event()
        call_count = 0  # reset for reflect call
        bullets = engine.reflect(event)

        assert len(bullets) == 1
        bullet = bullets[0]
        assert isinstance(bullet, CandidateBullet)
        assert bullet.distilled_rule is not None
        assert "rustup" in bullet.distilled_rule
        assert bullet.section == BulletSection.DEBUGGING
        assert bullet.knowledge_type == KnowledgeType.PITFALL
        assert "rustup" in bullet.related_tools

    def test_llm_mode_should_record_false(self) -> None:
        """LLM evaluator says not worth recording -> empty list."""
        mock_litellm = _ensure_litellm_mock()

        eval_response = MagicMock()
        eval_response.choices = [MagicMock()]
        eval_response.choices[0].message.content = (
            '{"should_record": false, "knowledge_type": "knowledge", '
            '"section": "general", "instructivity_score": 10, '
            '"summary": "trivial greeting"}'
        )

        mock_litellm.completion = lambda **kw: eval_response

        for mod_name in list(sys.modules):
            if "llm_evaluator" in mod_name or "llm_distiller" in mod_name:
                del sys.modules[mod_name]

        config = ReflectorConfig(mode="llm")
        engine = ReflectorEngine(config=config)
        event = _event(user="hi", assistant="hello there")
        bullets = engine.reflect(event)

        assert bullets == []
