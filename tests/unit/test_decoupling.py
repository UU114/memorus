"""Guard tests: verify Core/Team decoupling invariants (NFR-014).

These tests act as CI safeguards to prevent accidental coupling
between memx.core and memx.team.  If any test here fails, it means
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
CORE_DIR = _REPO_ROOT / "memx" / "core"
EXT_DIR = _REPO_ROOT / "memx" / "ext"
MEMX_DIR = _REPO_ROOT / "memx"


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
                    if alias.name.startswith("memx.team"):
                        violations.append(
                            f"{py_file}:{node.lineno} import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("memx.team"):
                    violations.append(
                        f"{py_file}:{node.lineno} from {node.module}"
                    )
    return violations


class TestDecoupling:
    """Verify Core/Team decoupling invariants."""

    # ── Static import checks ─────────────────────────────────────

    def test_core_does_not_import_team(self) -> None:
        """AST-level check: memx/core/ must never import from memx.team."""
        violations = _collect_team_imports(CORE_DIR)
        assert not violations, (
            "Core imports Team — this violates NFR-014:\n"
            + "\n".join(violations)
        )

    def test_ext_is_the_only_memx_package_importing_team(self) -> None:
        """Only memx/ext/ is allowed to reference memx.team.

        Scan every .py file under memx/ *except* memx/ext/ and
        memx/team/ itself, and verify zero Team imports.
        """
        violations: list[str] = []
        for py_file in sorted(MEMX_DIR.rglob("*.py")):
            # Skip memx/ext/ (allowed), memx/team/ (self-ref is fine)
            rel = py_file.relative_to(MEMX_DIR)
            parts = rel.parts
            if parts and parts[0] in ("ext", "team"):
                continue

            source = py_file.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("memx.team"):
                            violations.append(
                                f"{py_file}:{node.lineno} import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("memx.team"):
                        violations.append(
                            f"{py_file}:{node.lineno} from {node.module}"
                        )

        assert not violations, (
            "Non-ext package imports Team — only memx/ext/ may do this:\n"
            + "\n".join(violations)
        )

    # ── Runtime isolation checks ─────────────────────────────────

    def test_core_functions_without_team(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Core Memory initialises when memx.team is completely unavailable."""
        # Block every memx.team sub-import by injecting None sentinels.
        blocked = [
            k for k in list(sys.modules) if k.startswith("memx.team")
        ]
        for mod_name in blocked:
            monkeypatch.setitem(sys.modules, mod_name, None)
        monkeypatch.setitem(sys.modules, "memx.team", None)

        # Re-import Memory; it must not raise.
        from memx.core.memory import Memory  # noqa: F811

        mem = Memory()
        assert mem is not None
        assert mem._config is not None

    def test_core_config_without_team(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Core MemXConfig loads without memx.team present."""
        monkeypatch.setitem(sys.modules, "memx.team", None)

        from memx.core.config import MemXConfig

        cfg = MemXConfig.from_dict({})
        assert cfg is not None

    # ── ext/team_bootstrap graceful degradation ──────────────────

    def test_ext_bootstrap_handles_missing_team(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """team_bootstrap.try_bootstrap_team returns False when Team is absent."""
        # Make memx.team un-importable
        monkeypatch.setitem(sys.modules, "memx.team", None)
        monkeypatch.setitem(sys.modules, "memx.team.config", None)

        # Force re-evaluation of the lazy import inside try_bootstrap_team
        if "memx.ext.team_bootstrap" in sys.modules:
            monkeypatch.delitem(sys.modules, "memx.ext.team_bootstrap")

        from memx.ext.team_bootstrap import try_bootstrap_team

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
