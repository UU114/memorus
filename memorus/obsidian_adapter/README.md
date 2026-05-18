# memorus Obsidian Adapter

External, opt-in bridge from memorus to an Obsidian vault. memorus stays the
source of truth: the vault is a **read-only mirror** plus a **draft inbox**
that funnels writes through the existing nominator / Redactor / governance
pipeline. memorus core is not modified.

## Vault layout

```
<vault>/memorus/
├── team/            # read-only mirror of team bullets (overwritten on export)
├── personal/        # read-only mirror of personal memories
├── inbox/           # write drafts here; watcher submits via AceSyncClient / Memory.add
└── .tombstones/     # drop a copy of a bullet here to request its removal
```

## Usage

```powershell
# Full mirror refresh
python -m memorus.obsidian_adapter.cli export `
    --vault E:/path/to/vault `
    --team-id myteam `
    --with-personal

# One-shot inbox sweep (process every .md once, write sidecars)
python -m memorus.obsidian_adapter.cli watch `
    --vault E:/path/to/vault `
    --team-id myteam `
    --server-url https://ace.example.com `
    --token $env:ACE_TOKEN `
    --author-id alice@example.com `
    --once

# Daemon mode (poll every 2 s)
python -m memorus.obsidian_adapter.cli watch --vault E:/path/to/vault --author-id alice

# Submit a single file
python -m memorus.obsidian_adapter.cli nominate --file E:/path/to/vault/memorus/inbox/new.md

# Read submission outcomes
python -m memorus.obsidian_adapter.cli status --vault E:/path/to/vault
```

## Draft frontmatter

Minimum:

```markdown
---
type: team        # team | personal
section: workflow # optional; see BulletSection enum
tags: [ci, deploy]
---

Pin docker base image digest in CI to avoid silent base drift.
```

Server-owned fields (`enforcement`, `status`, `governance_tier`, `upvotes`,
`downvotes`, `nominated_at`, `verified_*`, `trust_score`, `deleted_at`,
`origin_id`) are stripped before submission — the user cannot self-promote.

## Invariants

- memorus core code untouched
- Inbox edits go through `Redactor.redact_l1` locally; fully-redacted drafts
  are blocked before they hit the network
- Deleting a vault file does **not** delete the bullet; only files placed in
  `.tombstones/` are treated as removal requests
- `<file>.status.md` sidecars carry the submission outcome
