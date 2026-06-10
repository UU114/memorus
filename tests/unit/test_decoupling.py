"""Guard tests: verify Core/Team decoupling invariants (NFR-014).

These tests act as CI safeguards to prevent accidental coupling
between memorus.core and memorus.team.  If any test here fails, it means
a developer introduced a forbidden dependency direction.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List

import pytest

# Resolve CORE_DIR / EXT_DIR relative to the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = _REPO_ROOT / "memorus" / "core"
EXT_DIR = _REPO_ROOT / "memorus" / "ext"
MEMORUS_DIR = _REPO_ROOT / "memorus"


def _collect_team_imports(directory: Path) -> List[str]:
    """Walk *directory* and return a list of Team-import violations.

    Each entry has the form ``<file>:<lineno> <import-statement>``.
    """
    violations: list[str] = []
    for py_file in sorted(directory.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            # Skip files that cannot be parsed (shouldn't happen,
            # but don't let the guard test itself crash).
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("memorus.team"):
                        violations.append(
                            f"{py_file}:{node.lineno} import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("memorus.team"):
                    violations.append(
                        f"{py_file}:{node.lineno} from {node.module}"
                    )
    return violations


class TestDecoupling:
    """Verify Core/Team decoupling invariants."""

    # ── Static import checks ─────────────────────────────────────

    def test_core_does_not_import_team(self) -> None:
        """AST-level check: memorus/core/ must never import from memorus.team."""
        violations = _collect_team_imports(CORE_DIR)
        assert not violations, (
            "Core imports Team — this violates NFR-014:\n"
            + "\n".join(violations)
        )

    def test_ext_is_the_only_memorus_package_importing_team(self) -> None:
        """Only integration surfaces may reference memorus.team.

        Scan every .py file under memorus/ *except* the integration-surface
        packages (memorus/ext/, memorus/obsidian_adapter/) and memorus/team/
        itself, and verify zero Team imports. The rule protects the CORE
        engine from a Team dependency; adapters sit beside ext, not inside
        core, so they may consume Team storage directly.
        """
        violations: list[str] = []
        for py_file in sorted(MEMORUS_DIR.rglob("*.py")):
            # Skip integration surfaces (allowed) and memorus/team/ itself
            rel = py_file.relative_to(MEMORUS_DIR)
            parts = rel.parts
            if parts and parts[0] in ("ext", "team", "obsidian_adapter"):
                continue

            source = py_file.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("memorus.team"):
                            violations.append(
                                f"{py_file}:{node.lineno} import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("memorus.team"):
                        violations.append(
                            f"{py_file}:{node.lineno} from {node.module}"
                        )

        assert not violations, (
            "Non-ext package imports Team — only memorus/ext/ may do this:\n"
            + "\n".join(violations)
        )

    # ── Runtime isolation checks ─────────────────────────────────

    def test_core_functions_without_team(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Core Memory initialises when memorus.team is completely unavailable."""
        # Block every memorus.team sub-import by injecting None sentinels.
        blocked = [
            k for k in list(sys.modules) if k.startswith("memorus.team")
        ]
        for mod_name in blocked:
            monkeypatch.setitem(sys.modules, mod_name, None)
        monkeypatch.setitem(sys.modules, "memorus.team", None)

        # Re-import Memory; it must not raise.
        from memorus.core.memory import Memory  # noqa: F811

        mem = Memory()
        assert mem is not None
        assert mem._config is not None

    def test_core_config_without_team(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Core MemorusConfig loads without memorus.team present."""
        monkeypatch.setitem(sys.modules, "memorus.team", None)

        from memorus.core.config import MemorusConfig

        cfg = MemorusConfig.from_dict({})
        assert cfg is not None

    # ── ext/team_bootstrap graceful degradation ──────────────────

    def test_ext_bootstrap_handles_missing_team(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """team_bootstrap.try_bootstrap_team returns False when Team is absent."""
        # Make memorus.team un-importable
        monkeypatch.setitem(sys.modules, "memorus.team", None)
        monkeypatch.setitem(sys.modules, "memorus.team.config", None)

        # Force re-evaluation of the lazy import inside try_bootstrap_team
        if "memorus.ext.team_bootstrap" in sys.modules:
            monkeypatch.delitem(sys.modules, "memorus.ext.team_bootstrap")

        from memorus.ext.team_bootstrap import try_bootstrap_team

        result = try_bootstrap_team(None)
        assert result is False, (
            "try_bootstrap_team should return False when Team is not installed"
        )

    # ── Marker for CI pipeline ───────────────────────────────────

    def test_decoupling_marker_exists(self) -> None:
        """Sanity: this file is discoverable by pytest."""
        # A trivial assertion that proves the guard suite itself runs.
        assert CORE_DIR.is_dir(), f"CORE_DIR not found: {CORE_DIR}"
        assert EXT_DIR.is_dir(), f"EXT_DIR not found: {EXT_DIR}"
