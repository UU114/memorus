---
name: memorus
description: Intelligent memory with auto-dedup, decay, and conflict detection
mcpServers:
  - memorus
---

# Memorus — Intelligent Memory

Replace flat Markdown memory files with a structured, deduplicated, decay-aware knowledge base.

## WHEN to SEARCH (`search_memory`)
- At the **START** of every new conversation
- Before answering about user preferences, past decisions, project context
- When user says "remember when...", "you told me...", or references prior context
- When unsure if a topic was discussed before

## WHEN to ADD (`add_memory`)
- User explicitly says "remember this"
- User reveals a preference or makes a decision
- A fact worth persisting long-term is established
- After completing a task, store the outcome and lessons learned
- Do **NOT** store: greetings, transient chat, unanswered questions, speculative content

## SCOPE Management
- `scope="project:<workspace>"` — for project-specific knowledge (configs, decisions, patterns)
- No scope (global) — for user-level preferences, general facts
- Search automatically queries both the specified scope AND global

## Tools Reference

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `search_memory` | Semantic search across knowledge base | Before answering questions; start of conversations |
| `add_memory` | Store a fact (auto-deduplicated) | When user reveals preferences or decisions |
| `list_memories` | Browse all stored memories | When user asks "what do you remember?" |
| `forget_memory` | Delete a memory by ID | When user asks to forget something |
| `memory_status` | Knowledge base statistics | Diagnostics, health checks |
| `detect_conflicts` | Find contradictory memories | Periodically in long sessions; after bulk imports |
| `run_decay_sweep` | Age and archive stale memories | Usually automatic; manual trigger if needed |
| `export_memories` | Export to JSON/Markdown file | Backup, migration, sharing |
| `import_memories` | Import from JSON file | Restore from backup, migrate from other systems |

## Best Practices
1. **Search first, add later** — always check if the knowledge already exists
2. **Keep content concise and factual** — one clear fact per memory
3. **Use scopes** — separate project knowledge from personal preferences
4. **Periodically detect conflicts** — especially after bulk imports or long sessions
5. **Trust deduplication** — if you add a similar fact, the system merges automatically
