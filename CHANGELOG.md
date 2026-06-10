# Changelog

All notable changes to Memorus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Python/Rust convergence

This batch converges the Python shim defaults and behavior toward the Rust core
(the single source of truth). Several defaults changed; read the migration notes
below before upgrading.

### Changed — defaults

- `ace_enabled` now defaults to `True` (was `False`). `add()` runs the ACE
  pipeline by default; set `ace_enabled=False` to keep the old raw-add behavior.
- `daemon.enabled` now defaults to `True` (was `False`). The daemon client
  degrades gracefully when no daemon is reachable, so this is safe to flip.
- Reflector `mode` now defaults to `rules` (was `hybrid`) on both Python and
  Rust. LLM-based distillation is now explicit opt-in (`mode="llm"` or
  `mode="hybrid"`); the default no longer makes LLM calls.
- `retrieval.token_budget` raised from `2000` to `4096`. Context blocks roughly
  double in size; downstream consumers with their own limits may truncate.
- `retrieval.scope_boost` raised from `1.3` to `1.5`.
- Curator dedup threshold unified at `0.9` (was `0.8`). A higher threshold
  yields fewer false-positive dedup merges.

### Changed — behavior

- `add()` with ACE enabled runs Sanitize → Reflect → Curate. The sanitizer is
  fail-closed: if sanitization raises, the entire ingest is dropped (raw,
  unsanitized content is never persisted).
- `search()` uses hybrid (keyword + semantic) ranking; result scores are
  normalized to `[0, 1]`.
- Project- and team-scoped `search()` now also returns `global` memories (the
  scope filter is a union of the requested scope and `global`).
- Recall reinforcement now persists on `search()`: recalled bullets have their
  recall count incremented (in a background thread) so frequently used knowledge
  resists decay.
- `add()` raw-fallback (Reflector failure or no candidates) now returns the
  per-item rows reported by the store, so callers can surface record ids.
- ONNX embeddings are L2-normalized after mean pooling, matching the Python
  reference. **Stores created by the older Rust path (which did not normalize)
  should be re-embedded or re-indexed**, otherwise cosine ranking will drift
  because old and new vectors have different magnitudes.

### Changed — configuration

- The canonical config schema is the Rust nested form (e.g. nested `ace.*`
  blocks). Canonical nested `ace` blocks are accepted by the Python shim.
- Python-era flat keys (`half_life_days`, `similarity_threshold`,
  `custom_patterns`, `conflict_min_similarity`, `conflict_max_similarity`, …)
  continue to load via serde aliases on the Rust side.

### Changed — server

- **Contract change:** the daemon `recall` and `curate` commands no longer
  return a silent empty result when the backing memory is not initialized or a
  required field is missing. They now return an explicit error response
  (`status="error"` with a message), so callers must handle the error case
  rather than mistaking an uninitialized daemon for "no results".
- The REST server is now fail-closed on CORS configuration: a malformed
  `--cors-origins` entry refuses to start (non-zero exit) instead of silently
  degrading to permissive CORS. Pass `*` explicitly for permissive CORS.
- The MCP server gained an `import_memories` tool for importing a memory payload
  (auto-deduplicated against existing knowledge).

## [1.0.0] - 2026-02-27

### Added

- **ACE Pipeline** — Adaptive Context Engine with full ingest and retrieval pipelines
- **Reflector Engine** — pattern detection, instructivity scoring, and knowledge distillation
- **Curator Engine** — semantic deduplication, merge suggestions, and conflict detection
- **Decay Engine** — time-based forgetting with configurable exponential/linear curves
- **Generator Engine** — hybrid search combining vector, exact match, fuzzy match, and metadata matching
- **Privacy Sanitizer** — PII detection and scrubbing with pluggable regex patterns
- **ONNX Embeddings** — optional local embedding via ONNX Runtime (no API calls needed)
- **CLI** — full Click-based command-line interface (`memorus status`, `search`, `learn`, `list`, `forget`, `sweep`, `conflicts`, `export`, `import`)
- **Async Support** — `AsyncMemory` class for async/await usage
- **Daemon Mode** — optional background daemon with IPC for shared memory across tools
- **Scoped Memories** — hierarchical scope support (global, project-level)
- **Token Budget Trimmer** — budget-aware result trimming for LLM context windows
- **Import/Export** — JSON and Markdown export, JSON import with Curator dedup
- **GitHub Actions** — automated PyPI publishing on tag push
- **PyPI Packaging** — `pip install memorus` with optional dependency groups (`onnx`, `graph`, `all`, `dev`)

### Notes

- Initial public release
- Requires Python 3.9+
- Built on top of [mem0](https://github.com/mem0ai/mem0) as the storage backend
