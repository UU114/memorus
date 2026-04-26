# Memory Trust / Verification Layer

This document explains the **Memory Trust** layer shipped in EPIC-R018
(STORY-R099 → R105). It covers the anchor model, the three retrieval policies,
the configuration knobs, the migration path for legacy bullets, and how to
read the verification signal from inside an LLM prompt.

For the Rust mirror, see `memorus-r/doc/verification.md`. The two docs are
intentionally kept in sync — anything that changes here MUST be reflected in
the Rust counterpart.

---

## 1. Why memory needs verification

Long-lived agent memory accumulates **stale rules**. A bullet that read
*"call `connect(addr: &str)` from `src/net.rs`"* is correct on day one and
silently wrong six months later when the function gets renamed. Without
verification the agent will happily quote the dead rule and either hallucinate
fixes or trigger build errors.

The Memory Trust layer attaches **anchors** to bullets at ingest time, then
re-checks those anchors against the live filesystem at retrieval time. Each
bullet exposes one of four `verified_status` values plus a numeric
`trust_score` so prompt builders can up-rank the verified rows and demote
the suspect ones.

Goals:

- Catch references to renamed/deleted symbols before they reach the LLM.
- Keep verification **cheap** — TTL-cached results mean a hot retrieval pays
  near-zero filesystem cost.
- Never block ingest or search: verification is best-effort and degrades
  silently when the anchor signal is missing.

Non-goals:

- Re-extracting bullet content (rehydrate adds anchors only).
- Cross-machine consensus — verification is local to the workspace.
- Replacing the conflict detector — anchor mismatches are surfaced as one
  more `ConflictType` rather than as a separate pipeline.

---

## 2. Anchor model

An **anchor** is a `(file_path, anchor_text, anchor_hash, created_at)` tuple
attached to a bullet. The `anchor_text` is a short window (≤ 200 chars) that
the bullet claims must remain present in `file_path` for the bullet to be
trustworthy. `anchor_hash = sha256(anchor_text)` defends against tampering of
the stored anchor row itself.

```
                    ┌──────────────────────────────┐
   ingest turn ───▶ │  BulletDistiller             │
                    │  ._extract_anchors()         │
                    └──────────────┬───────────────┘
                                   │  list[Anchor]
                                   ▼
                    ┌──────────────────────────────┐
                    │  BulletMetadata.anchors      │
                    │  (persisted in mem0 payload) │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
   retrieval ────▶ │  VerificationEngine.verify()  │
                    │   ├ NotApplicable (no anchors)│
                    │   ├ Verified                  │
                    │   ├ Stale                     │
                    │   └ Unverifiable              │
                    └──────────────────────────────┘
```

Extraction sources:

- **Code-block anchors**: a fenced code block paired with any source-file
  path mentioned in the same turn. The first block whose head window
  (≤ 200 chars, line-aligned) appears verbatim in the referenced file wins.
- **Symbol anchors**: a `fn foo` / `def foo` / `class Foo` mention paired
  with a path. The verifier reads the file, finds the symbol, and stores a
  ±N-line context window.

A bullet may carry multiple anchors. Verification is **AND** — any failing
anchor demotes the whole bullet to `stale`.

---

## 3. Three policies: flag / demote / drop

`VerificationConfig.policy` controls how stale bullets are surfaced at
retrieval time. The verifier itself **never decides policy** — it just emits
a `VerifiedStatus`. The retrieval pipeline branches once on `policy`:

| Policy   | Behaviour                                                                                  | When to use                                                                        |
|----------|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `flag`   | No-op. Stale rows are returned with their original score; the LLM sees the status field.   | Default. Trust the LLM to read `verified_status` and choose what to ignore.       |
| `demote` | Multiply `final_score` by `stale_trust_score` / `unverifiable_trust_score`; re-sort.       | You want stale rows to fall to the bottom of the list automatically.              |
| `drop`   | Filter out rows whose `verified_status == "stale"`. Unverifiable + not_applicable kept.    | Strict mode: never feed a known-stale rule to the LLM.                            |

Notes:

- A row with `verified_status is None` (verifier disabled or lookup failed)
  is always kept as-is. Missing signal is never treated as a failure.
- `not_applicable` (no anchors at all) is **always** kept across all
  policies — by definition it cannot be wrong.
- The drop count is surfaced through `ace_search.dropped_stale_count` so
  operators can monitor how often it fires.

---

## 4. Configuration reference

`memorus.core.config.VerificationConfig`:

| Field                       | Type                          | Default | Meaning                                                                                  |
|-----------------------------|-------------------------------|---------|------------------------------------------------------------------------------------------|
| `enabled`                   | `bool`                        | `True`  | Master switch. When `False` the verifier never runs and write-back never happens.        |
| `ttl_seconds`               | `int ≥ 0`                     | `60`    | Cache lifetime for a bullet's verification state. `0` disables the cache (always re-check). |
| `project_root`              | `str | None`                  | `None`  | Root directory anchor `file_path` is relative to. `None` walks up from cwd looking for `.git` / `pyproject.toml` / `Cargo.toml`. |
| `verified_trust_score`      | `float ∈ [0.0, 1.0]`          | `1.0`   | Multiplier applied to `final_score` for a verified row under `policy=demote`.            |
| `stale_trust_score`         | `float ∈ [0.0, 1.0]`          | `0.3`   | Multiplier for `stale` rows under `demote`. Lower = harsher penalty.                     |
| `unverifiable_trust_score`  | `float ∈ [0.0, 1.0]`          | `0.7`   | Multiplier for `unverifiable` (file missing / read failed). Less harsh than stale.       |
| `policy`                    | `"flag" | "demote" | "drop"`  | `"flag"` | See §3.                                                                                  |

TOML example (`pyproject.toml` or standalone `memorus.toml`):

```toml
[verification]
enabled = true
ttl_seconds = 300
project_root = "/abs/path/to/repo"   # optional
verified_trust_score = 1.0
stale_trust_score = 0.25
unverifiable_trust_score = 0.6
policy = "demote"
```

---

## 5. CLI commands

### `memorus verify`

Refresh `verified_status` across the store. Iterates every bullet (or the
filtered subset), runs `VerificationEngine.verify_many`, and writes the
outcome back to the mem0 payload unless `--dry-run` is passed.

Flags:

| Flag                   | Description                                                                                |
|------------------------|--------------------------------------------------------------------------------------------|
| `--rehydrate-anchors`  | Backfill anchors on bullets where `anchors == []`. Re-runs `_extract_anchors` on `bullet.content`. |
| `--stale-only`         | Iterate only bullets whose stored `verified_status` is `null` or `"stale"`.                 |
| `--scope SCOPE`        | Filter to a single scope (e.g. `project:myapp`).                                            |
| `--dry-run`            | Compute the report without any DB writes. Provably read-only.                              |
| `--json`               | Emit the report as JSON. Shape matches Rust `memorus-r verify --json` byte-for-byte.       |
| `--user-id USER`       | Forward to `mem0.get_all` for user-scoped iteration.                                        |

JSON report shape (identical on both sides):

```json
{
  "verified": 142,
  "stale": 18,
  "unverifiable": 5,
  "not_applicable": 63,
  "elapsed_ms": 820,
  "anchors_added": 41
}
```

Single-bullet failures are logged at WARNING and counted as `errors` in the
human report. They never abort the run.

### `memorus conflicts --type TYPE`

Filter the conflict report to a single wire-form type:

```bash
memorus conflicts --type anchor_mismatch
```

Supported wire forms (from `ConflictType`):
- `anchor_mismatch` (STORY-R104)
- `opposing_pair`
- `negation_asymmetry`
- `version_conflict`

Omitting `--type` returns the full set, identical to pre-R105 behaviour.

---

## 6. Migration from legacy bullets

Bullets ingested before EPIC-R018 carry no anchors and, by protocol, always
verify as `not_applicable`. To enrol them in the trust layer:

1. **Dry run the rehydrate**:
   ```bash
   memorus verify --rehydrate-anchors --dry-run --json
   ```
   Inspect `anchors_added` to estimate impact.

2. **Apply rehydrate**:
   ```bash
   memorus verify --rehydrate-anchors
   ```
   Each newly anchored bullet has its `memorus_anchors` payload field
   re-written. Bullets where extraction yields nothing are left untouched.

3. **Baseline verify**:
   ```bash
   memorus verify
   ```
   This populates `verified_status` + `verified_at` on every bullet so the
   first user search sees a TTL-cached state.

4. **Periodic refresh**:
   ```bash
   memorus verify --stale-only
   ```
   Schedule via cron / launchd / systemd timer at whatever cadence matches
   your codebase churn (daily for active repos, weekly for stable libs).

The rehydrate pass is **best-effort**: it can only attach anchors when the
bullet content still contains a recognisable file path or symbol. Bullets
about pure conventions (e.g. *"prefer kebab-case for branch names"*) have no
anchorable content and will stay `not_applicable` forever — that is the
correct behaviour.

---

## 7. LLM prompt integration patterns

When the verifier is enabled, every search row carries two extra keys:

```json
{
  "id": "abc123",
  "memory": "use FooClient.connect_v2() in src/net.rs",
  "score": 0.91,
  "verified_status": "verified",
  "trust_score": 1.0,
  "metadata": { ... }
}
```

Recommended prompt-side handling:

- **Default (`policy=flag`)**: render the status next to each row so the LLM
  can read it. Example template:
  ```
  [verified, score 1.00] use FooClient.connect_v2() in src/net.rs
  [stale,    score 0.30] connect() lives in src/legacy.rs   # ⚠ stale
  [n/a,      score —   ] always run formatter before commit
  ```
  Add a system-prompt clause: *"Treat `stale` rows as unreliable; cite
  `verified` rows preferentially. `n/a` rows have no automatic check."*

- **`policy=demote`**: the demoted rows are already at the bottom of the
  list, so a simple top-N feed is usually enough. Still surface
  `verified_status` so the LLM knows why a row was demoted.

- **`policy=drop`**: only verified / unverifiable / not_applicable rows
  reach the prompt; you can render without the badge column.

Tip: when injecting into a CLAUDE.md / AGENTS.md style preamble, keep the
trust badge **first** on each line so the model sees the signal before the
content. Models trained on long contexts attend to row prefixes more
reliably than to suffixes.

---

## 8. FAQ

**Q: My team syncs memories across machines. Anchors point at file paths
that may not exist on every machine. What happens?**

The verifier returns `unverifiable` for missing files (not `stale`). Under
the default `flag` / `demote` policies the bullet is still served, just with
a lower trust score. Under `drop` the bullet is kept (drop only filters
`stale`). To eliminate the noise on machines without the codebase, set
`enabled = false` in that environment's config — the verifier short-circuits
at the engine boundary.

**Q: My project has no git repo (or pyproject / Cargo.toml). Where does
`project_root` resolve?**

The walk-up returns the current working directory as a last resort. Anchor
paths are then evaluated relative to cwd. The cleanest fix is to set
`project_root` explicitly in `VerificationConfig`.

**Q: How should I tune `ttl_seconds`?**

- Active development repo: `60` to `300` — files churn fast, you want fresh
  signal but not on every keystroke.
- Stable library you cite from many bullets: `3600` or higher — saves
  redundant filesystem reads.
- CI / scheduled verify-only runs: `0` to force every check, then rely on
  the persisted `verified_at` to power TTL during normal retrieval.

**Q: Can the verifier run inside a daemon and update the store in the
background?**

Yes — call `memorus verify --stale-only` from a cron / scheduler. The MVP
is single-shot; running it on a 5-minute cadence is equivalent to a
background daemon for most workloads.

**Q: A bullet flips between `verified` and `stale` between runs. Bug?**

Most often this is a flaky build artefact (a test runner that rewrites a
file). Pin `project_root` to the source tree (not the build tree) and
re-run. If the flap persists with a stable source tree, file an issue with
the bullet id and the anchor that flips — the protocol guarantees
determinism for stable inputs.
