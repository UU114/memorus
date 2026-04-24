"""BulletDistiller — Stage 4 of the Reflector pipeline. Distill scored candidates into Bullets."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import logging

from memorus.core.config import ReflectorConfig, VerificationConfig
from memorus.core.types import (
    Anchor,
    BulletSection,
    CandidateBullet,
    KnowledgeType,
    ScoredCandidate,
    SourceRef,
    SourceType,
)

logger = logging.getLogger(__name__)

# Source-file extensions that participate in anchor extraction.
# Non-source files (.log, .tmp, binaries) are excluded to avoid noise.
_ANCHOR_SOURCE_EXTS: frozenset[str] = frozenset({
    "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "rb",
    "md", "yaml", "yml", "toml", "json",
})

# Compose a run-time regex from the whitelist; the ``|`` join matches the
# source-extension set above. The pattern mirrors the sanitizer path regex
# (memorus/core/privacy/patterns.py:70-81) but scoped to source files.
_ANCHOR_PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:[A-Za-z]:[\\/])?"                           # optional Windows drive
    r"(?:[\w.\-]+[\\/])+"                           # one or more directory hops
    r"[\w.\-]+\.(?:" + "|".join(sorted(_ANCHOR_SOURCE_EXTS)) + r")"
    r")"
)

# Fenced code block: captures opening fence language and body.
_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

# Cap on anchor_text characters. Mirrors the Pydantic ``Anchor.anchor_text``
# ``max_length=200`` constraint — keep in sync.
_ANCHOR_TEXT_MAX = 200

# Read at most 1 MB from an anchor-target file; skip files larger than 5 MB
# entirely. Anchor extraction is best-effort — we never want to stall ingest
# on a giant file.
_ANCHOR_READ_BYTES = 1 * 1024 * 1024
_ANCHOR_FILE_MAX_BYTES = 5 * 1024 * 1024

# Minimum meaningful anchor_text length. Below this the window is too easy
# to collide on, so we skip the anchor rather than store a brittle match.
_ANCHOR_MIN_CHARS = 20

# Common tool/command names for extraction
_KNOWN_TOOLS = frozenset({
    "git", "docker", "npm", "pip", "cargo", "brew", "apt",
    "kubectl", "terraform", "ansible", "make", "cmake",
    "pytest", "ruff", "mypy", "black", "flake8",
    "curl", "wget", "ssh", "scp", "rsync",
    "python", "node", "java", "go", "rustc", "ruby",
    "yarn", "pnpm", "bun", "deno", "poetry", "uv",
    "postgres", "redis", "nginx", "systemctl",
})


class BulletDistiller:
    """Distill ScoredCandidates into compact CandidateBullets."""

    def __init__(
        self,
        config: Optional[ReflectorConfig] = None,
        verification: Optional[VerificationConfig] = None,
    ):
        cfg = config or ReflectorConfig()
        self._max_content = cfg.max_content_length
        self._max_code_lines = cfg.max_code_lines
        # Verification config carries the configured ``project_root`` used to
        # resolve anchor paths. None means: resolve on-demand from cwd via
        # ``_resolve_project_root``.
        self._verification = verification

    def distill(self, candidate: ScoredCandidate) -> CandidateBullet:
        """Convert a ScoredCandidate into a CandidateBullet."""
        raw_content = candidate.pattern.content
        content = self._truncate_content(raw_content)
        was_truncated = len(content) < len(raw_content)
        tools = self._extract_tools(raw_content, candidate.pattern.metadata)
        entities = self._extract_entities(raw_content)
        sources = self._build_sources(candidate)

        # STORY-R100 — Anchor extraction is best-effort; any failure inside
        # ``_extract_anchors`` is swallowed and falls back to an empty list so
        # a broken source tree never blocks bullet ingest.
        project_root = self._resolve_project_root()
        turn_text = self._turn_text_for(candidate)
        anchor_input = (
            f"{turn_text}\n{raw_content}" if turn_text else raw_content
        )
        anchors = self._extract_anchors(anchor_input, sources, project_root)

        logger.debug(
            "BulletDistiller.distill: content_len=%d->%d truncated=%s "
            "tools=%s entities=%s sources=%d anchors=%d",
            len(raw_content), len(content), was_truncated, tools, entities[:5],
            len(sources), len(anchors),
        )

        return CandidateBullet(
            content=content,
            section=candidate.section,
            knowledge_type=candidate.knowledge_type,
            source_type=SourceType.INTERACTION,
            instructivity_score=candidate.instructivity_score,
            related_tools=tools,
            key_entities=entities,
            sources=sources,
            anchors=anchors,
        )

    @staticmethod
    def _build_sources(candidate: ScoredCandidate) -> list[SourceRef]:
        """Build SourceRefs from the originating InteractionEvent (if any).

        We record the assistant turn because assistant messages carry the
        actionable knowledge; the user prompt is contextual. ``turn_offset``
        comes from ``event.metadata["turn_offset"]`` when the caller tracks
        conversation position, else falls back to 0.
        """
        event = candidate.pattern.source_event
        if event is None:
            return []

        conversation_id = event.session_id or event.user_id
        if not conversation_id:
            return []

        meta = event.metadata or {}
        try:
            turn_offset = int(meta.get("turn_offset", 0))
        except (TypeError, ValueError):
            turn_offset = 0
        turn_offset = max(turn_offset, 0)

        turn_text = event.assistant_message or event.user_message
        turn_hash = SourceRef.compute_turn_hash(turn_text)

        return [
            SourceRef(
                conversation_id=conversation_id,
                turn_hash=turn_hash,
                turn_offset=turn_offset,
                timestamp=event.timestamp,
                role="assistant" if event.assistant_message else "user",
            )
        ]

    def _truncate_content(self, content: str) -> str:
        """Truncate content at sentence boundary, respecting max length."""
        if len(content) <= self._max_content:
            return content

        truncated = content[:self._max_content]
        # Try to cut at sentence boundary
        for sep in [". ", "\u3002", "\n", "; ", ", "]:
            idx = truncated.rfind(sep)
            if idx > self._max_content * 0.5:
                return truncated[:idx + len(sep)].rstrip()
        return truncated.rstrip() + "..."

    def _truncate_code(self, code: str) -> str:
        """Truncate code to max_code_lines."""
        lines = code.splitlines()
        if len(lines) <= self._max_code_lines:
            return code
        return "\n".join(lines[:self._max_code_lines]) + "\n..."

    @staticmethod
    def _extract_tools(content: str, context: dict) -> list[str]:
        """Extract tool names from content and context."""
        tools: set[str] = set()

        # From context metadata
        if tool := context.get("tool"):
            tools.add(str(tool).lower())
        if tool_list := context.get("tools"):
            if isinstance(tool_list, list):
                tools.update(str(t).lower() for t in tool_list)

        # From content — match known tools
        words = set(re.findall(r'\b\w+\b', content.lower()))
        tools.update(words & _KNOWN_TOOLS)

        return sorted(tools)

    # ------------------------------------------------------------------
    # Anchor extraction (STORY-R100)
    # ------------------------------------------------------------------

    def _resolve_project_root(self) -> Optional[Path]:
        """Resolve the project root used for anchor path normalization.

        Priority:
        1. ``VerificationConfig.project_root`` if configured and exists.
        2. Walk up from cwd looking for ``.git`` / ``pyproject.toml`` /
           ``Cargo.toml`` — whichever comes first.
        Returns ``None`` when neither strategy succeeds; callers then skip
        anchor extraction rather than attach mis-rooted paths.
        """
        if self._verification and self._verification.project_root:
            candidate = Path(self._verification.project_root).expanduser()
            if candidate.is_dir():
                return candidate.resolve()

        cwd = Path.cwd().resolve()
        sentinels = (".git", "pyproject.toml", "Cargo.toml")
        for directory in (cwd, *cwd.parents):
            for sentinel in sentinels:
                if (directory / sentinel).exists():
                    return directory
        return None

    @staticmethod
    def _turn_text_for(candidate: ScoredCandidate) -> str:
        """Return the originating conversation turn text, if available.

        Anchor extraction scans both the distilled bullet content and the
        originating turn, because code blocks live in the turn (not the
        bullet). When no source event is attached, return empty string.
        """
        event = candidate.pattern.source_event
        if event is None:
            return ""
        parts = [event.user_message or "", event.assistant_message or ""]
        return "\n".join(p for p in parts if p)

    def _extract_anchors(
        self,
        content: str,
        sources: list[SourceRef],  # noqa: ARG002 — kept for API parity with spec
        project_root: Optional[Path],
    ) -> list[Anchor]:
        """Extract region anchors from ``content`` relative to ``project_root``.

        Best-effort: any failure (regex, I/O, encoding) is trapped and the
        method returns whatever anchors were already collected. Never raises.
        Returns an empty list when no anchors can be produced.
        """
        if project_root is None or not content:
            return []

        anchors: list[Anchor] = []
        seen: set[tuple[str, str]] = set()  # (file_path, anchor_hash)
        now = datetime.now(timezone.utc)

        try:
            # Pass 1: fenced code blocks paired with any path mentioned
            # anywhere in ``content``. We don't require co-location — the
            # path and the code block typically share a turn.
            code_blocks: list[str] = [
                m.group(1) for m in _CODE_BLOCK_RE.finditer(content)
            ]
            paths_in_content: list[str] = [
                m.group("path") for m in _ANCHOR_PATH_RE.finditer(content)
            ]

            for raw_path in paths_in_content:
                rel_path = self._normalize_anchor_path(raw_path, project_root)
                if rel_path is None:
                    continue

                abs_path = project_root / rel_path
                file_text = self._safe_read_file(abs_path)
                if file_text is None:
                    continue

                # Pass 1a: code-block anchors — take the first code block
                # that produces a ≥ _ANCHOR_MIN_CHARS window matching the
                # file. We build the window from the code block itself
                # since Verifier compares anchor_text to future file state.
                for block in code_blocks:
                    window = self._code_block_window(block)
                    if window and window in file_text:
                        self._try_add_anchor(
                            rel_path, window, now, anchors, seen,
                        )
                        break  # one code-block anchor per file

                # Pass 1b: symbol anchors — pick up `fn foo`, `def foo`,
                # etc., mentioned around the path and search the file for
                # the symbol, then take a ±N line window.
                for symbol in self._candidate_symbols(content):
                    window = self._symbol_window(file_text, symbol)
                    if window:
                        self._try_add_anchor(
                            rel_path, window, now, anchors, seen,
                        )
                        # Allow multiple symbols per file; they produce
                        # distinct anchor_hash values so dedupe still works.
        except Exception as exc:  # pragma: no cover — best-effort guard
            logger.debug("Anchor extraction swallowed error: %s", exc)
            return anchors
        return anchors

    @staticmethod
    def _try_add_anchor(
        file_path: str,
        anchor_text: str,
        created_at: datetime,
        anchors: list[Anchor],
        seen: set[tuple[str, str]],
    ) -> None:
        """Build, dedupe, and append an Anchor if the window is usable."""
        text = anchor_text[:_ANCHOR_TEXT_MAX]
        if len(text) < _ANCHOR_MIN_CHARS:
            return
        anchor_hash = Anchor.compute_hash(text)
        key = (file_path, anchor_hash)
        if key in seen:
            return
        seen.add(key)
        anchors.append(
            Anchor(
                file_path=file_path,
                anchor_text=text,
                anchor_hash=anchor_hash,
                created_at=created_at,
            )
        )

    @staticmethod
    def _normalize_anchor_path(
        raw_path: str, project_root: Path
    ) -> Optional[str]:
        """Normalize *raw_path* to a forward-slash path relative to project_root.

        Returns ``None`` when the path refers to a file outside project_root,
        when the file does not exist, or when the file is oversized/binary.
        Windows drive letters are lower-cased; everything else preserves
        case because Linux file systems are case-sensitive.
        """
        cleaned = raw_path.replace("\\", "/")
        # Lowercase a Windows drive letter if present ("C:/..." -> "c:/...")
        if len(cleaned) >= 2 and cleaned[1] == ":":
            cleaned = cleaned[0].lower() + cleaned[1:]

        path_obj = Path(cleaned)
        candidate_abs = (
            path_obj if path_obj.is_absolute() else project_root / path_obj
        )
        try:
            candidate_abs = candidate_abs.resolve(strict=False)
        except OSError:
            return None
        if not candidate_abs.is_file():
            return None

        try:
            size = candidate_abs.stat().st_size
        except OSError:
            return None
        if size > _ANCHOR_FILE_MAX_BYTES:
            return None

        # Must be inside the project root. ``os.path.commonpath`` raises
        # on cross-drive comparisons, so guard with a try/except.
        try:
            rel = candidate_abs.relative_to(project_root.resolve())
        except ValueError:
            return None
        return rel.as_posix()

    @staticmethod
    def _safe_read_file(abs_path: Path) -> Optional[str]:
        """Read at most _ANCHOR_READ_BYTES from *abs_path* as UTF-8 text.

        Returns ``None`` for binary files or read errors. This is best-effort;
        any failure makes the caller simply skip anchor generation for the
        given path.
        """
        try:
            with abs_path.open("rb") as fh:
                blob = fh.read(_ANCHOR_READ_BYTES)
        except OSError:
            return None
        # Cheap binary heuristic: presence of NUL byte
        if b"\x00" in blob:
            return None
        try:
            return blob.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @staticmethod
    def _code_block_window(block: str) -> str:
        """Return the first ≤ 200 chars of *block*, cut at a line boundary."""
        trimmed = block.strip()
        if len(trimmed) <= _ANCHOR_TEXT_MAX:
            return trimmed
        head = trimmed[:_ANCHOR_TEXT_MAX]
        # Prefer a line boundary near the end
        nl = head.rfind("\n")
        return head[:nl] if nl > _ANCHOR_MIN_CHARS else head

    @staticmethod
    def _symbol_window(file_text: str, symbol: str) -> Optional[str]:
        """Locate *symbol* in *file_text* and return ±N-line context window.

        Window: 1 line before the symbol occurrence + the line itself +
        4 lines after. Returns ``None`` when the symbol is absent. The
        returned text is truncated to _ANCHOR_TEXT_MAX chars at line
        boundaries so the Verifier has deterministic input.
        """
        if not symbol or symbol not in file_text:
            return None
        # Line-based slice: find the line containing the first occurrence.
        lines = file_text.splitlines()
        hit_idx: Optional[int] = None
        for idx, line in enumerate(lines):
            if symbol in line:
                hit_idx = idx
                break
        if hit_idx is None:
            return None
        start = max(hit_idx - 1, 0)
        end = min(hit_idx + 5, len(lines))  # +4 after +1 for exclusive end
        window = "\n".join(lines[start:end])
        if len(window) <= _ANCHOR_TEXT_MAX:
            return window
        # Cut at last line boundary ≤ cap
        head = window[:_ANCHOR_TEXT_MAX]
        nl = head.rfind("\n")
        return head[:nl] if nl > _ANCHOR_MIN_CHARS else head

    @staticmethod
    def _candidate_symbols(content: str) -> list[str]:
        """Pick out short symbol-ish tokens that make good anchor seeds.

        Supported patterns (order matters — earlier ones win on ties):
        * ``fn NAME`` (Rust)
        * ``def NAME`` (Python)
        * ``function NAME`` (JS/TS)
        * ``class NAME``
        * Backticked ``NAME`` where NAME looks like an identifier
        """
        seen: set[str] = set()
        out: list[str] = []

        def _push(sym: str) -> None:
            if sym and sym not in seen:
                seen.add(sym)
                out.append(sym)

        for pat in (
            r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)",
        ):
            for m in re.finditer(pat, content):
                # Anchor on the full signature head so the file match
                # lands on the definition rather than a stray mention.
                prefix = m.group(0).split()[0]
                _push(f"{prefix} {m.group(1)}")

        for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", content):
            _push(m.group(1))

        return out

    @staticmethod
    def _extract_entities(content: str) -> list[str]:
        """Extract key entities: quoted terms, CamelCase, dotted names."""
        entities: set[str] = set()

        # Quoted strings (backtick, double-quote)
        entities.update(re.findall(r'`([^`]+)`', content))
        entities.update(re.findall(r'"([^"]{2,30})"', content))

        # CamelCase / PascalCase identifiers
        entities.update(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', content))

        # File-like dotted names (e.g., "config.yaml", "main.py")
        entities.update(
            m for m in re.findall(r'\b[\w-]+\.[\w.]+\b', content)
            if not m[0].isdigit()  # Exclude version numbers like "2.0"
        )

        # Limit to 10 most relevant
        return sorted(entities)[:10]
