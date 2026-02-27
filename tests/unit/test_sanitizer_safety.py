"""Tests for STORY-015: PrivacySanitizer safety net behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memx.config import MemXConfig, PrivacyConfig
from memx.memory import Memory
from memx.pipeline.ingest import IngestPipeline
from memx.privacy.sanitizer import PrivacySanitizer


class TestIngestPipelineSanitizerIndependence:
    """Sanitizer runs before and independently of Reflector in IngestPipeline."""

    def test_sanitizer_runs_when_reflector_disabled(self):
        """Even if reflector returns empty, sanitizer should have already processed."""
        sanitizer = PrivacySanitizer()
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = []  # No patterns
        mock_mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            sanitizer=sanitizer,
            mem0_add_fn=mock_mem0_add,
        )

        # Message with API key
        messages = "My key is sk-proj-abcdefghijklmnopqrstuvwxyz1234567890abcdef"
        result = pipeline.process(messages, user_id="u1")

        # Reflector got sanitized content (key replaced)
        call_args = mock_reflector.reflect.call_args[0][0]
        assert "<OPENAI_KEY>" in call_args.user_message
        assert "sk-proj-" not in call_args.user_message

        # Raw fallback happened (no patterns) but with sanitized content
        assert result.raw_fallback is True
        # mem0 add should have been called with sanitized content
        assert mock_mem0_add.called
        add_content = mock_mem0_add.call_args[0][0]
        assert "sk-proj-" not in str(add_content)

    def test_sanitizer_runs_when_reflector_raises(self):
        """Sanitizer runs even if Reflector completely fails."""
        sanitizer = PrivacySanitizer()
        mock_reflector = MagicMock()
        mock_reflector.reflect.side_effect = RuntimeError("Reflector crash")
        mock_mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            sanitizer=sanitizer,
            mem0_add_fn=mock_mem0_add,
        )

        messages = "password=super_secret_123"
        result = pipeline.process(messages, user_id="u1")

        assert result.raw_fallback is True
        # The reflector received sanitized input before it crashed
        call_event = mock_reflector.reflect.call_args[0][0]
        assert "super_secret_123" not in call_event.user_message

    def test_sanitizer_failure_nonfatal_in_pipeline(self):
        """If sanitizer itself crashes, pipeline continues with original content."""
        mock_sanitizer = MagicMock()
        mock_sanitizer.sanitize.side_effect = RuntimeError("Sanitizer crash")
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = []
        mock_mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            sanitizer=mock_sanitizer,
            mem0_add_fn=mock_mem0_add,
        )

        result = pipeline.process("normal content", user_id="u1")
        # Should not crash
        assert result.raw_fallback is True

    def test_no_sanitizer_in_pipeline(self):
        """Pipeline works fine without a sanitizer."""
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = []

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            sanitizer=None,
            mem0_add_fn=MagicMock(),
        )

        result = pipeline.process("test content", user_id="u1")
        assert result.raw_fallback is True  # No patterns detected


class TestMemoryAlwaysSanitize:
    """Memory class respects always_sanitize config."""

    @pytest.fixture()
    def _make_memory(self):
        def factory(ace_enabled=False, always_sanitize=False):
            m = Memory.__new__(Memory)
            m._config = MemXConfig.from_dict({
                "ace_enabled": ace_enabled,
                "privacy": {"always_sanitize": always_sanitize},
            })
            m._mem0 = MagicMock()
            m._mem0.add.return_value = {"results": []}
            m._ingest_pipeline = None
            m._retrieval_pipeline = None
            if always_sanitize:
                m._sanitizer = PrivacySanitizer()
            else:
                m._sanitizer = None
            return m
        return factory

    def test_ace_off_sanitize_off(self, _make_memory):
        """Default: ace_enabled=False, always_sanitize=False -> no sanitization."""
        m = _make_memory(ace_enabled=False, always_sanitize=False)
        secret = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890abcdef"
        m.add(secret, user_id="u1")
        # Should pass through raw
        assert m._mem0.add.call_args[0][0] == secret

    def test_ace_off_sanitize_on(self, _make_memory):
        """ace_enabled=False but always_sanitize=True -> sanitization runs."""
        m = _make_memory(ace_enabled=False, always_sanitize=True)
        secret = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890abcdef"
        m.add(secret, user_id="u1")
        # Should be sanitized
        called_msg = m._mem0.add.call_args[0][0]
        assert "sk-proj-" not in called_msg
        assert "<OPENAI_KEY>" in called_msg

    def test_ace_off_sanitize_on_list_messages(self, _make_memory):
        """always_sanitize works with list-of-dict messages too."""
        m = _make_memory(ace_enabled=False, always_sanitize=True)
        msgs = [
            {"role": "user", "content": "My key is sk-proj-abcdefghijklmnopqrstuvwxyz1234567890abcdef"},
            {"role": "assistant", "content": "Noted!"},
        ]
        m.add(msgs, user_id="u1")
        called_msgs = m._mem0.add.call_args[0][0]
        assert "sk-proj-" not in called_msgs[0]["content"]
        assert "<OPENAI_KEY>" in called_msgs[0]["content"]
        # Assistant message unchanged
        assert called_msgs[1]["content"] == "Noted!"

    def test_sanitizer_failure_in_memory_nonfatal(self, _make_memory):
        """If sanitizer crashes in Memory, add() still works."""
        m = _make_memory(ace_enabled=False, always_sanitize=True)
        m._sanitizer = MagicMock()
        m._sanitizer.sanitize.side_effect = RuntimeError("boom")

        # Should not crash
        m.add("test content", user_id="u1")
        # Original message passed through
        assert m._mem0.add.call_args[0][0] == "test content"
