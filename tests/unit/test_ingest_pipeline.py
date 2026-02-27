"""Unit tests for memx.pipeline.ingest — IngestPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from memx.config import ReflectorConfig
from memx.engines.reflector.engine import ReflectorEngine
from memx.pipeline.ingest import IngestPipeline, IngestResult
from memx.privacy.sanitizer import PrivacySanitizer, SanitizeResult
from memx.types import (
    BulletSection,
    CandidateBullet,
    InteractionEvent,
    KnowledgeType,
    SourceType,
)
from memx.utils.bullet_factory import MEMX_PREFIX


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _error_messages() -> list[dict[str, str]]:
    """Messages that trigger the ErrorFixRule in Reflector."""
    return [
        {
            "role": "user",
            "content": (
                "I'm getting an error when running cargo build on my project"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "You should try updating your Rust toolchain by running "
                "rustup update to fix this compilation error in your project."
            ),
        },
    ]


def _trivial_messages() -> list[dict[str, str]]:
    """Messages that should NOT trigger any pattern detection rule."""
    return [
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "content": (
                "The capital of France is Paris, a city known for its "
                "culture and history spanning many centuries."
            ),
        },
    ]


def _make_candidate(
    content: str = "Use rustup update to fix compilation errors",
    section: BulletSection = BulletSection.DEBUGGING,
    knowledge_type: KnowledgeType = KnowledgeType.PITFALL,
    score: float = 65.0,
) -> CandidateBullet:
    """Create a CandidateBullet for mocking."""
    return CandidateBullet(
        content=content,
        section=section,
        knowledge_type=knowledge_type,
        source_type=SourceType.INTERACTION,
        instructivity_score=score,
        key_entities=["rust", "cargo"],
        related_tools=["rustup"],
        related_files=[],
        tags=["error-fix"],
        scope="global",
    )


# ---------------------------------------------------------------------------
# Test 1: Normal flow — error messages produce bullets
# ---------------------------------------------------------------------------


class TestProcessNormalFlow:
    """Messages with error patterns -> Reflector produces bullets -> bullets_added > 0."""

    def test_process_normal_flow(self) -> None:
        """Full pipeline: error messages -> reflector -> bullets written to mem0."""
        # Use a real ReflectorEngine with a low min_score to ensure detection
        config = ReflectorConfig(min_score=10.0)
        reflector = ReflectorEngine(config=config)
        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=reflector,
            mem0_add_fn=mem0_add,
        )

        result = pipeline.process(
            _error_messages(),
            metadata={"app": "test"},
            user_id="u1",
            agent_id="a1",
            run_id="r1",
        )

        assert isinstance(result, IngestResult)
        assert result.bullets_added >= 1
        assert result.raw_fallback is False
        assert result.errors == []
        # mem0_add should have been called once per bullet
        assert mem0_add.call_count == result.bullets_added


# ---------------------------------------------------------------------------
# Test 2: No patterns detected — raw fallback
# ---------------------------------------------------------------------------


class TestProcessNoPatterns:
    """Normal conversation -> no patterns -> raw_fallback=True."""

    def test_process_no_patterns(self) -> None:
        """Trivial messages produce no patterns, triggering raw fallback."""
        reflector = ReflectorEngine()
        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=reflector,
            mem0_add_fn=mem0_add,
        )

        result = pipeline.process(
            _trivial_messages(),
            user_id="u1",
        )

        assert result.raw_fallback is True
        assert result.bullets_added == 0
        # Raw add should have been called once (the fallback path)
        mem0_add.assert_called_once()


# ---------------------------------------------------------------------------
# Test 3: Empty messages — empty IngestResult
# ---------------------------------------------------------------------------


class TestProcessEmptyMessages:
    """None or empty input -> empty IngestResult, no processing."""

    def test_process_none_messages(self) -> None:
        """None messages -> immediate empty IngestResult."""
        reflector = MagicMock()
        mem0_add = MagicMock()

        pipeline = IngestPipeline(reflector=reflector, mem0_add_fn=mem0_add)
        result = pipeline.process(None)

        assert result.bullets_added == 0
        assert result.raw_fallback is False
        assert result.errors == []
        reflector.reflect.assert_not_called()
        mem0_add.assert_not_called()

    def test_process_empty_list(self) -> None:
        """Empty list -> immediate empty IngestResult."""
        reflector = MagicMock()
        mem0_add = MagicMock()

        pipeline = IngestPipeline(reflector=reflector, mem0_add_fn=mem0_add)
        result = pipeline.process([])

        assert result.bullets_added == 0
        assert result.raw_fallback is False
        reflector.reflect.assert_not_called()
        mem0_add.assert_not_called()

    def test_process_empty_string(self) -> None:
        """Empty string -> immediate empty IngestResult."""
        reflector = MagicMock()
        pipeline = IngestPipeline(reflector=reflector)
        result = pipeline.process("")

        assert result.bullets_added == 0
        assert result.raw_fallback is False
        reflector.reflect.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: String messages — parsed correctly
# ---------------------------------------------------------------------------


class TestProcessStringMessages:
    """String input parsed into InteractionEvent with user_message only."""

    def test_process_string_messages(self) -> None:
        """String input is parsed as user_message."""
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [_make_candidate()]
        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector, mem0_add_fn=mem0_add
        )

        result = pipeline.process("I got an error running cargo build")

        assert result.bullets_added == 1
        # Verify reflect was called with an InteractionEvent
        call_args = mock_reflector.reflect.call_args[0][0]
        assert isinstance(call_args, InteractionEvent)
        assert call_args.user_message == "I got an error running cargo build"
        assert call_args.assistant_message == ""


# ---------------------------------------------------------------------------
# Test 5: List messages — parsed correctly
# ---------------------------------------------------------------------------


class TestProcessListMessages:
    """List of dicts split by role into user/assistant messages."""

    def test_process_list_messages(self) -> None:
        """List of role dicts is correctly parsed into InteractionEvent."""
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [_make_candidate()]
        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector, mem0_add_fn=mem0_add
        )

        messages = [
            {"role": "user", "content": "How to fix error X?"},
            {"role": "assistant", "content": "Try running command Y."},
            {"role": "user", "content": "That worked, thanks!"},
        ]
        result = pipeline.process(messages, metadata={"k": "v"})

        assert result.bullets_added == 1
        call_args = mock_reflector.reflect.call_args[0][0]
        assert isinstance(call_args, InteractionEvent)
        assert "How to fix error X?" in call_args.user_message
        assert "That worked, thanks!" in call_args.user_message
        assert "Try running command Y." in call_args.assistant_message
        assert call_args.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# Test 6: Reflector failure fallback
# ---------------------------------------------------------------------------


class TestReflectorFailureFallback:
    """Reflector raises -> raw_fallback=True, mem0_add called with raw messages."""

    def test_reflector_failure_fallback(self) -> None:
        """When Reflector raises, pipeline falls back to raw add."""
        mock_reflector = MagicMock()
        mock_reflector.reflect.side_effect = RuntimeError("LLM timeout")
        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector, mem0_add_fn=mem0_add
        )

        messages = _error_messages()
        result = pipeline.process(
            messages, user_id="u1", metadata={"app": "test"}
        )

        assert result.raw_fallback is True
        assert result.bullets_added == 0
        # Raw add should have been called with original messages
        mem0_add.assert_called_once()
        call_args = mem0_add.call_args
        assert call_args[0][0] == messages  # original messages passed through


# ---------------------------------------------------------------------------
# Test 7: Sanitizer runs before Reflector
# ---------------------------------------------------------------------------


class TestSanitizerRunsBeforeReflector:
    """Messages with API key should be sanitized before reaching Reflector."""

    def test_sanitizer_runs_before_reflector(self) -> None:
        """API key in string message is redacted before Reflector sees it."""
        sanitizer = PrivacySanitizer()
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = []

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            sanitizer=sanitizer,
        )

        secret = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
        pipeline.process(f"My API key is {secret}")

        # Verify that Reflector received sanitised content
        call_args = mock_reflector.reflect.call_args[0][0]
        assert isinstance(call_args, InteractionEvent)
        assert secret not in call_args.user_message
        assert "<OPENAI_KEY>" in call_args.user_message

    def test_sanitizer_runs_on_list_messages(self) -> None:
        """API key in list messages is redacted before Reflector."""
        sanitizer = PrivacySanitizer()
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = []

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            sanitizer=sanitizer,
        )

        secret = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
        messages = [
            {"role": "user", "content": f"Key is {secret}"},
            {"role": "assistant", "content": "I see your key."},
        ]
        pipeline.process(messages)

        call_args = mock_reflector.reflect.call_args[0][0]
        assert secret not in call_args.user_message


# ---------------------------------------------------------------------------
# Test 8: Sanitizer failure is non-fatal
# ---------------------------------------------------------------------------


class TestSanitizerFailureNonfatal:
    """Sanitizer raises -> original messages passed through to Reflector."""

    def test_sanitizer_failure_nonfatal(self) -> None:
        """When sanitizer raises, pipeline continues with original messages."""
        mock_sanitizer = MagicMock()
        mock_sanitizer.sanitize.side_effect = RuntimeError("sanitizer crash")

        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [_make_candidate()]
        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            sanitizer=mock_sanitizer,
            mem0_add_fn=mem0_add,
        )

        original = "raw message with secret"
        result = pipeline.process(original)

        # Pipeline should still succeed
        assert result.bullets_added == 1
        assert result.errors == []
        # Reflector received the original (unsanitised) message
        call_args = mock_reflector.reflect.call_args[0][0]
        assert call_args.user_message == original


# ---------------------------------------------------------------------------
# Test 9: mem0_add_fn called with bullet content + metadata
# ---------------------------------------------------------------------------


class TestMem0AddCalled:
    """Verify mem0_add_fn is called with bullet content and merged metadata."""

    def test_mem0_add_called(self) -> None:
        """mem0_add_fn called with content, user_id, agent_id, run_id, merged metadata."""
        candidate = _make_candidate(content="Use rustup update to fix errors")
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [candidate]
        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            mem0_add_fn=mem0_add,
        )

        result = pipeline.process(
            "error message",
            metadata={"app": "myapp"},
            user_id="u1",
            agent_id="a1",
            run_id="r1",
        )

        assert result.bullets_added == 1
        mem0_add.assert_called_once()

        # Check the call arguments
        call_args = mem0_add.call_args
        assert call_args[0][0] == "Use rustup update to fix errors"
        assert call_args[1]["user_id"] == "u1"
        assert call_args[1]["agent_id"] == "a1"
        assert call_args[1]["run_id"] == "r1"

        # Merged metadata should contain both app key and memx_ prefixed keys
        meta = call_args[1]["metadata"]
        assert meta["app"] == "myapp"
        assert f"{MEMX_PREFIX}section" in meta
        assert f"{MEMX_PREFIX}knowledge_type" in meta


# ---------------------------------------------------------------------------
# Test 10: mem0_add_fn failure — error recorded
# ---------------------------------------------------------------------------


class TestMem0AddFailure:
    """mem0_add_fn raises -> error recorded in IngestResult.errors."""

    def test_mem0_add_failure(self) -> None:
        """When mem0_add_fn raises, error is captured and processing continues."""
        candidates = [
            _make_candidate(content="bullet 1"),
            _make_candidate(content="bullet 2"),
        ]
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = candidates

        mem0_add = MagicMock()
        # First call succeeds, second call fails
        mem0_add.side_effect = [None, RuntimeError("db write failed")]

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            mem0_add_fn=mem0_add,
        )

        result = pipeline.process("error message")

        # First bullet succeeds, second fails
        assert result.bullets_added == 1
        assert len(result.errors) == 1
        assert "db write failed" in result.errors[0]


# ---------------------------------------------------------------------------
# Test 11: No mem0_add_fn — bullets counted but not written
# ---------------------------------------------------------------------------


class TestNoMem0AddFn:
    """mem0_add_fn=None -> bullets counted in result but not written anywhere."""

    def test_no_mem0_add_fn(self) -> None:
        """With mem0_add_fn=None, bullets are counted but no write occurs."""
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [
            _make_candidate(content="bullet A"),
            _make_candidate(content="bullet B"),
        ]

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            mem0_add_fn=None,
        )

        result = pipeline.process("error message")

        assert result.bullets_added == 2
        assert result.errors == []
        assert result.raw_fallback is False


# ---------------------------------------------------------------------------
# Test 12: Bullet metadata with memx_ prefix in mem0 call
# ---------------------------------------------------------------------------


class TestBulletMetadataInMem0:
    """Verify memx_ prefixed metadata is correctly included in mem0 call."""

    def test_bullet_metadata_in_mem0(self) -> None:
        """Bullet fields appear as memx_-prefixed keys in merged metadata."""
        candidate = _make_candidate(
            content="fix content",
            section=BulletSection.DEBUGGING,
            knowledge_type=KnowledgeType.PITFALL,
            score=72.0,
        )
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [candidate]
        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            mem0_add_fn=mem0_add,
        )

        result = pipeline.process("error message", metadata={"env": "prod"})

        assert result.bullets_added == 1
        meta = mem0_add.call_args[1]["metadata"]

        # Check memx_ prefixed fields
        assert meta[f"{MEMX_PREFIX}section"] == "debugging"
        assert meta[f"{MEMX_PREFIX}knowledge_type"] == "pitfall"
        assert meta[f"{MEMX_PREFIX}instructivity_score"] == 72.0
        assert meta[f"{MEMX_PREFIX}source_type"] == "interaction"
        assert meta[f"{MEMX_PREFIX}scope"] == "global"

        # Check list fields are JSON-serialised strings
        import json

        assert json.loads(meta[f"{MEMX_PREFIX}related_tools"]) == ["rustup"]
        assert json.loads(meta[f"{MEMX_PREFIX}key_entities"]) == ["rust", "cargo"]
        assert json.loads(meta[f"{MEMX_PREFIX}tags"]) == ["error-fix"]

        # User metadata should also be present
        assert meta["env"] == "prod"


# ---------------------------------------------------------------------------
# Test 13: IngestResult defaults
# ---------------------------------------------------------------------------


class TestIngestResultDefaults:
    """IngestResult() has correct default values."""

    def test_ingest_result_defaults(self) -> None:
        """Newly created IngestResult has zeroed counters and empty lists."""
        result = IngestResult()

        assert result.bullets_added == 0
        assert result.bullets_merged == 0
        assert result.bullets_skipped == 0
        assert result.errors == []
        assert result.raw_fallback is False

    def test_ingest_result_errors_are_independent(self) -> None:
        """Each IngestResult instance has its own errors list."""
        r1 = IngestResult()
        r2 = IngestResult()

        r1.errors.append("error1")
        assert r2.errors == []

    def test_ingest_result_fields_mutable(self) -> None:
        """IngestResult fields can be updated after creation."""
        result = IngestResult()
        result.bullets_added = 5
        result.bullets_merged = 2
        result.bullets_skipped = 1
        result.raw_fallback = True
        result.errors.append("test error")

        assert result.bullets_added == 5
        assert result.bullets_merged == 2
        assert result.bullets_skipped == 1
        assert result.raw_fallback is True
        assert result.errors == ["test error"]


# ---------------------------------------------------------------------------
# Additional edge case: _parse_event with non-standard input
# ---------------------------------------------------------------------------


class TestParseEventEdgeCases:
    """Edge cases for _parse_event static method."""

    def test_parse_non_standard_type(self) -> None:
        """Non-string, non-list input is str() coerced."""
        event = IngestPipeline._parse_event(12345, {})
        assert event.user_message == "12345"
        assert event.assistant_message == ""

    def test_parse_list_with_mixed_items(self) -> None:
        """List with non-dict items are skipped during role parsing."""
        messages = [
            {"role": "user", "content": "hello"},
            "not a dict",
            {"role": "assistant", "content": "hi"},
            42,
        ]
        event = IngestPipeline._parse_event(messages, {"key": "val"})
        assert event.user_message == "hello"
        assert event.assistant_message == "hi"
        assert event.metadata == {"key": "val"}

    def test_parse_list_with_system_role(self) -> None:
        """System role messages are not included in user or assistant."""
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        event = IngestPipeline._parse_event(messages, {})
        assert event.user_message == "Question?"
        assert event.assistant_message == "Answer."


# ---------------------------------------------------------------------------
# Integration test: Memory.add() ACE path
# ---------------------------------------------------------------------------


class TestMemoryAddACEPath:
    """Verify Memory.add() delegates to IngestPipeline when ACE is enabled."""

    def test_memory_add_ace_path(self) -> None:
        """ACE enabled + pipeline exists -> returns ace_ingest dict."""
        from memx.config import MemXConfig
        from memx.memory import Memory

        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        # Create a mock pipeline that returns a known IngestResult
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = IngestResult(
            bullets_added=3, raw_fallback=False, errors=[]
        )
        m._ingest_pipeline = mock_pipeline

        result = m.add("test message", user_id="u1")

        assert "ace_ingest" in result
        assert result["ace_ingest"]["bullets_added"] == 3
        assert result["ace_ingest"]["raw_fallback"] is False
        assert result["ace_ingest"]["errors"] == []
        # mem0.add should NOT have been called (pipeline handled it)
        m._mem0.add.assert_not_called()

    def test_memory_add_ace_no_pipeline_falls_back(self) -> None:
        """ACE enabled but pipeline is None -> falls back to proxy."""
        from memx.config import MemXConfig
        from memx.memory import Memory

        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0.add.return_value = {"results": []}
        m._mem0_init_error = None
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        result = m.add("test message", user_id="u1")

        # Should proxy to mem0
        m._mem0.add.assert_called_once()
        assert result == {"results": []}
