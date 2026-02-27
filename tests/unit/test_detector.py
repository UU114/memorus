"""Unit tests for memx.engines.reflector.detector — PatternDetector."""

from __future__ import annotations

from typing import Optional, Sequence

from memx.engines.reflector.detector import PatternDetector
from memx.engines.reflector.patterns import (
    ConfigChangeRule,
    NewToolRule,
    PatternRule,
    RepetitiveOpRule,
)
from memx.types import DetectedPattern, InteractionEvent


# ── Helper factories ───────────────────────────────────────────────────


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


# ── Error-fix detection tests ─────────────────────────────────────────


class TestDetectErrorFix:
    def test_detect_error_fix_positive(self) -> None:
        """User message has error keyword, assistant has fix keyword -> pattern detected."""
        detector = PatternDetector()
        event = _event(
            user="I'm getting an error when running cargo build",
            assistant="You should try updating your Rust toolchain. Run rustup update to fix this.",
        )
        patterns = detector.detect(event)
        assert len(patterns) >= 1
        fix = next(p for p in patterns if p.pattern_type == "error_fix")
        assert fix.confidence == 0.8
        assert fix.source_event == event
        assert fix.metadata["detection_rule"] == "error_fix"
        assert "Error:" in fix.content
        assert "Fix:" in fix.content

    def test_detect_error_fix_no_error(self) -> None:
        """Normal conversation without error context -> no error_fix pattern."""
        detector = PatternDetector()
        event = _event(
            user="What is the capital of France?",
            assistant="The capital of France is Paris, a city known for its culture and history.",
        )
        patterns = detector.detect(event)
        error_patterns = [p for p in patterns if p.pattern_type == "error_fix"]
        assert len(error_patterns) == 0

    def test_detect_error_fix_no_fix_keywords(self) -> None:
        """Error mentioned in user message but assistant provides no fix keywords."""
        detector = PatternDetector()
        event = _event(
            user="I have an error in my code somewhere",
            assistant="That sounds concerning. I would need more information to help you out with that.",
        )
        patterns = detector.detect(event)
        error_patterns = [p for p in patterns if p.pattern_type == "error_fix"]
        assert len(error_patterns) == 0

    def test_detect_error_fix_short_response(self) -> None:
        """Error mentioned but assistant response is too short -> no pattern."""
        detector = PatternDetector()
        event = _event(
            user="I have an error in my code",
            assistant="Try again.",
        )
        patterns = detector.detect(event)
        error_patterns = [p for p in patterns if p.pattern_type == "error_fix"]
        assert len(error_patterns) == 0

    def test_detect_error_fix_metadata_error(self) -> None:
        """Error signal comes from metadata.error_msg instead of user message text."""
        detector = PatternDetector()
        event = _event(
            user="I ran the deploy script",
            assistant="You should try setting the environment variable DB_HOST before running deploy.",
            metadata={"error_msg": "ConnectionError: cannot connect to database"},
        )
        patterns = detector.detect(event)
        assert len(patterns) >= 1
        fix = next(p for p in patterns if p.pattern_type == "error_fix")
        assert fix.confidence == 0.8
        assert fix.metadata["detection_rule"] == "error_fix"


# ── Retry-success detection tests ─────────────────────────────────────


class TestDetectRetrySuccess:
    def test_detect_retry_success_positive(self) -> None:
        """Previous event had error on same topic, current event shows resolution."""
        detector = PatternDetector()

        # First interaction: user has an error about webpack build
        prev_event = _event(
            user="webpack build error in my project config files",
            assistant="Let me look into that for you.",
        )
        detector.detect(prev_event)  # adds to history

        # Second interaction: same topic, resolution language
        current_event = _event(
            user="webpack build with my project config files again",
            assistant="The webpack build works now after fixing the config. It is resolved.",
        )
        patterns = detector.detect(current_event)
        retry_patterns = [p for p in patterns if p.pattern_type == "retry_success"]
        assert len(retry_patterns) == 1
        retry = retry_patterns[0]
        assert retry.confidence == 0.7
        assert retry.metadata["detection_rule"] == "retry_success"
        assert "related_keywords" in retry.metadata
        assert "Previous issue:" in retry.content
        assert "Resolution:" in retry.content

    def test_detect_retry_success_no_history(self) -> None:
        """Empty history -> no retry_success pattern possible."""
        detector = PatternDetector()
        event = _event(
            user="webpack build looks good now",
            assistant="Yes, everything works perfectly and is resolved.",
        )
        patterns = detector.detect(event)
        retry_patterns = [p for p in patterns if p.pattern_type == "retry_success"]
        assert len(retry_patterns) == 0

    def test_detect_retry_success_no_overlap(self) -> None:
        """Previous and current topics are completely unrelated -> no pattern."""
        detector = PatternDetector()

        # First: error about database
        prev_event = _event(
            user="database connection error keeps happening",
            assistant="Check your connection string.",
        )
        detector.detect(prev_event)

        # Second: completely different topic with success language
        current_event = _event(
            user="frontend styling looks beautiful",
            assistant="The design works great and is resolved now.",
        )
        patterns = detector.detect(current_event)
        retry_patterns = [p for p in patterns if p.pattern_type == "retry_success"]
        assert len(retry_patterns) == 0


# ── Code-heavy filter tests ───────────────────────────────────────────


class TestIsCodeHeavy:
    def test_is_code_heavy_positive(self) -> None:
        """Content that is mostly code -> True."""
        code_content = "\n".join([
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
        assert PatternDetector._is_code_heavy(code_content) is True

    def test_is_code_heavy_negative(self) -> None:
        """Content that is mostly natural language prose -> False."""
        prose = "\n".join([
            "The weather today is quite nice and sunny.",
            "I went to the store to buy some groceries for dinner.",
            "My favorite color is blue, and I like dogs a lot.",
            "We should plan a meeting next week to discuss the project.",
            "The new restaurant downtown has really excellent food.",
        ])
        assert PatternDetector._is_code_heavy(prose) is False

    def test_is_code_heavy_empty(self) -> None:
        """Empty content -> False."""
        assert PatternDetector._is_code_heavy("") is False
        assert PatternDetector._is_code_heavy("   ") is False

    def test_is_code_heavy_short(self) -> None:
        """Very short content (fewer than 3 lines) -> False."""
        assert PatternDetector._is_code_heavy("import os") is False
        assert PatternDetector._is_code_heavy("import os\nimport sys") is False


# ── Safety / edge-case tests ──────────────────────────────────────────


class TestSafety:
    def test_detect_never_raises(self) -> None:
        """detect() must never raise, even with pathological input."""
        detector = PatternDetector()

        # None-like values won't pass pydantic, but empty strings are fine
        event = _event(user="", assistant="")
        result = detector.detect(event)
        assert isinstance(result, list)

        # Very long strings
        long_event = _event(user="error " * 10000, assistant="try " * 10000)
        result = detector.detect(long_event)
        assert isinstance(result, list)


class TestClearHistory:
    def test_clear_history(self) -> None:
        """clear_history() empties the internal deque."""
        detector = PatternDetector()

        # Add some events to history
        for i in range(5):
            detector.detect(_event(user=f"message {i}", assistant=f"reply {i}"))
        assert len(detector._history) == 5

        detector.clear_history()
        assert len(detector._history) == 0


# ── History bounded by max_history ─────────────────────────────────────


class TestHistoryCapacity:
    def test_history_respects_max(self) -> None:
        """History deque respects max_history parameter."""
        detector = PatternDetector(max_history=3)
        for i in range(10):
            detector.detect(_event(user=f"msg {i}", assistant=f"reply {i}"))
        assert len(detector._history) == 3


# ── Return type validation ─────────────────────────────────────────────


class TestReturnTypes:
    def test_detected_pattern_is_valid_model(self) -> None:
        """Returned patterns are valid DetectedPattern instances."""
        detector = PatternDetector()
        event = _event(
            user="I have an error with npm install failing",
            assistant="Try running npm cache clean --force, then use npm install again.",
        )
        patterns = detector.detect(event)
        assert len(patterns) >= 1
        for p in patterns:
            assert isinstance(p, DetectedPattern)
            assert 0.0 <= p.confidence <= 1.0
            assert p.pattern_type != ""
            assert p.content != ""

    def test_code_heavy_event_returns_empty_list(self) -> None:
        """Code-heavy content is filtered out and returns empty list."""
        detector = PatternDetector()
        code_event = _event(
            user="Here is my code:\nimport os\nimport sys\nfrom pathlib import Path\ndef main():\n    x = 1\n    y = 2\n    return x + y\nclass Foo:\n    pass\nif __name__:\n    main()",
            assistant="import logging\ndef handler():\n    return None\nclass Bar:\n    pass\nfor i in range(10):\n    print(i)\nif True:\n    pass\nreturn 0\nimport json\nfrom os import path",
        )
        patterns = detector.detect(code_event)
        assert patterns == []


# ── ConfigChangeRule tests ────────────────────────────────────────────


class TestConfigChangeRule:
    def test_config_change_positive(self) -> None:
        """Config keywords in conversation -> config_change pattern detected."""
        detector = PatternDetector()
        event = _event(
            user="How do I configure the database environment variable?",
            assistant="You need to set the DB_HOST variable in your environment config file to point to the correct host.",
        )
        patterns = detector.detect(event)
        config_patterns = [p for p in patterns if p.pattern_type == "config_change"]
        assert len(config_patterns) == 1
        cfg = config_patterns[0]
        assert cfg.confidence == 0.6
        assert cfg.metadata["detection_rule"] == "config_change"
        assert "Config:" in cfg.content

    def test_config_change_no_keywords(self) -> None:
        """Unrelated conversation -> no config_change pattern."""
        detector = PatternDetector()
        event = _event(
            user="Tell me about the history of ancient Rome and its emperors.",
            assistant="Ancient Rome was founded in 753 BC and grew into a vast empire spanning centuries of history.",
        )
        patterns = detector.detect(event)
        config_patterns = [p for p in patterns if p.pattern_type == "config_change"]
        assert len(config_patterns) == 0

    def test_config_change_file_extension(self) -> None:
        """Reference to .env or .yaml file -> config_change pattern detected."""
        detector = PatternDetector()
        event = _event(
            user="I need to update the values in my .env file for production",
            assistant="Open the .env file and change the DATABASE_URL to point to your production server address.",
        )
        patterns = detector.detect(event)
        config_patterns = [p for p in patterns if p.pattern_type == "config_change"]
        assert len(config_patterns) >= 1

    def test_config_change_short_response(self) -> None:
        """Config keyword present but assistant response too short -> no pattern."""
        detector = PatternDetector()
        event = _event(
            user="How to configure?",
            assistant="Check the docs.",
        )
        patterns = detector.detect(event)
        config_patterns = [p for p in patterns if p.pattern_type == "config_change"]
        assert len(config_patterns) == 0


# ── NewToolRule tests ─────────────────────────────────────────────────


class TestNewToolRule:
    def test_new_tool_install_command(self) -> None:
        """pip install command in conversation -> new_tool pattern detected."""
        detector = PatternDetector()
        event = _event(
            user="How do I install the requests library?",
            assistant="You can install it by running pip install requests in your terminal session.",
        )
        patterns = detector.detect(event)
        tool_patterns = [p for p in patterns if p.pattern_type == "new_tool"]
        assert len(tool_patterns) == 1
        tool = tool_patterns[0]
        assert tool.confidence == 0.65
        assert tool.metadata["detection_rule"] == "new_tool"
        assert "tools" in tool.metadata
        assert "Tool:" in tool.content

    def test_new_tool_metadata(self) -> None:
        """tool_name in metadata, first time -> new_tool pattern detected."""
        detector = PatternDetector()
        event = _event(
            user="I want to use the new linting tool",
            assistant="The ruff linter is very fast and has excellent coverage for Python code quality checks.",
            metadata={"tool_name": "ruff"},
        )
        patterns = detector.detect(event)
        tool_patterns = [p for p in patterns if p.pattern_type == "new_tool"]
        assert len(tool_patterns) == 1
        tool = tool_patterns[0]
        assert tool.metadata["tools"] == ["ruff"]

    def test_new_tool_seen_before(self) -> None:
        """tool_name already in history -> no new_tool pattern."""
        detector = PatternDetector()

        # First event introduces the tool
        first = _event(
            user="Using ruff for linting today",
            assistant="Ruff is a great choice for linting Python. It runs very fast and catches many issues.",
            metadata={"tool_name": "ruff"},
        )
        detector.detect(first)  # adds to history

        # Second event uses same tool name
        second = _event(
            user="Still using ruff for this project",
            assistant="Great, ruff should continue working well for you on this particular project going forward.",
            metadata={"tool_name": "ruff"},
        )
        patterns = detector.detect(second)
        tool_patterns = [p for p in patterns if p.pattern_type == "new_tool"]
        assert len(tool_patterns) == 0

    def test_new_tool_short_response(self) -> None:
        """Install command but short assistant response -> no pattern."""
        detector = PatternDetector()
        event = _event(
            user="pip install flask",
            assistant="Done.",
        )
        patterns = detector.detect(event)
        tool_patterns = [p for p in patterns if p.pattern_type == "new_tool"]
        assert len(tool_patterns) == 0


# ── RepetitiveOpRule tests ────────────────────────────────────────────


class TestRepetitiveOpRule:
    def test_repetitive_op_positive(self) -> None:
        """Same topic 3+ times -> repetitive_op pattern detected."""
        detector = PatternDetector()

        # Build up history with similar topic (shared words: deploy, production, server)
        for _ in range(3):
            prev = _event(
                user="deploy production server with latest changes",
                assistant="Deploying to production now with your latest changes applied to the server.",
            )
            detector.detect(prev)

        # Fourth event on same topic
        event = _event(
            user="deploy production server once more with changes",
            assistant="Running another deployment to the production server with the latest version.",
        )
        patterns = detector.detect(event)
        rep_patterns = [p for p in patterns if p.pattern_type == "repetitive_op"]
        assert len(rep_patterns) == 1
        rep = rep_patterns[0]
        assert rep.metadata["detection_rule"] == "repetitive_op"
        assert rep.metadata["occurrences"] >= 3
        assert rep.confidence >= 0.5
        assert "Repeated pattern" in rep.content

    def test_repetitive_op_insufficient(self) -> None:
        """Only 1 prior similar event (need 2+) -> no pattern."""
        detector = PatternDetector()

        # Only one prior similar event
        prev = _event(
            user="deploy production server with latest changes",
            assistant="Deploying to production now.",
        )
        detector.detect(prev)

        event = _event(
            user="deploy production server once more with changes",
            assistant="Running another deployment.",
        )
        patterns = detector.detect(event)
        rep_patterns = [p for p in patterns if p.pattern_type == "repetitive_op"]
        assert len(rep_patterns) == 0

    def test_repetitive_op_no_overlap(self) -> None:
        """Three events on different topics -> no repetitive_op pattern."""
        detector = PatternDetector()

        topics = [
            "deploy production server with changes",
            "compile frontend assets for release",
            "migrate database schema to latest",
        ]
        for topic in topics:
            detector.detect(_event(user=topic, assistant="Acknowledged and working on that."))

        # Fourth event on yet another unrelated topic
        event = _event(
            user="review pull request number forty two",
            assistant="Looking at the pull request now for review.",
        )
        patterns = detector.detect(event)
        rep_patterns = [p for p in patterns if p.pattern_type == "repetitive_op"]
        assert len(rep_patterns) == 0


# ── Custom rules injection tests ──────────────────────────────────────


class TestCustomRules:
    def test_custom_rules_list(self) -> None:
        """Passing a custom rules list replaces the default rules."""

        class AlwaysMatchRule(PatternRule):
            @property
            def name(self) -> str:
                return "always_match"

            def check(
                self,
                event: InteractionEvent,
                history: Sequence[InteractionEvent],
            ) -> Optional[DetectedPattern]:
                return DetectedPattern(
                    pattern_type=self.name,
                    content="matched",
                    confidence=1.0,
                    source_event=event,
                    metadata={"detection_rule": self.name},
                )

        detector = PatternDetector(rules=[AlwaysMatchRule()])
        event = _event(user="anything at all", assistant="some sufficiently long response text here")
        patterns = detector.detect(event)

        assert len(patterns) == 1
        assert patterns[0].pattern_type == "always_match"
        assert patterns[0].confidence == 1.0

    def test_empty_rules_list(self) -> None:
        """Empty rules list produces no patterns."""
        detector = PatternDetector(rules=[])
        event = _event(
            user="I have an error with my code",
            assistant="You should try reinstalling the package to fix the issue you are seeing.",
        )
        patterns = detector.detect(event)
        assert patterns == []

    def test_default_rules_count(self) -> None:
        """Default rules list has exactly 5 rules after STORY-009."""
        detector = PatternDetector()
        assert len(detector._rules) == 5
