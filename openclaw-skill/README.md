# Memorus × OpenClaw Skill

Intelligent memory system for OpenClaw with auto-deduplication, temporal decay, and conflict detection.

Replaces flat Markdown memory files with a structured, queryable knowledge base.

## Quick Start

### 1. Install

**Unix/macOS:**
```bash
bash setup.sh
```

**Windows:**
```cmd
setup.bat
```

**Manual:**
```bash
pip install memorus[mcp]
```

### 2. Configure OpenClaw

Merge `openclaw.json` into your OpenClaw MCP server configuration:

```json
{
  "mcpServers": {
    "memorus": {
      "command": "memorus-mcp",
      "args": ["--config", "~/.openclaw/memorus-config.json"]
    }
  }
}
```

### 3. Add the Skill

Copy `SKILL.md` to your OpenClaw skills directory.

### 4. Migrate Existing Memories (Optional)

If you have existing OpenClaw Markdown memories:

```bash
# Preview what will be migrated
python migrate.py --memory-dir ~/.openclaw --dry-run

# Run the migration
python migrate.py --memory-dir ~/.openclaw --config ~/.openclaw/memorus-config.json
```

## Configuration

Edit `~/.openclaw/memorus-config.json`:

```json
{
  "ace_enabled": true,
  "reflector": { "mode": "rules" },
  "privacy": { "always_sanitize": true },
  "retrieval": { "max_results": 10, "token_budget": 2000 }
}
```

| Option | Description |
|--------|-------------|
| `ace_enabled` | Enable ACE engine (dedup, decay, classification) |
| `reflector.mode` | `"rules"` (offline, zero cost) or `"hybrid"` (uses LLM) |
| `privacy.always_sanitize` | Sanitize PII before storage |
| `retrieval.max_results` | Maximum memories returned per search |
| `retrieval.token_budget` | Token limit for search results |

## Available Tools

| Tool | Description |
|------|-------------|
| `search_memory` | Semantic search across knowledge base |
| `add_memory` | Store a fact (auto-deduplicated) |
| `list_memories` | Browse all stored memories |
| `forget_memory` | Delete a memory by ID |
| `memory_status` | Knowledge base statistics |
| `detect_conflicts` | Find contradictory memories |
| `run_decay_sweep` | Age and archive stale memories |
| `export_memories` | Export to JSON/Markdown file |
| `import_memories` | Import from JSON file |

## Coexistence with OpenClaw Memory

Memorus works alongside OpenClaw's built-in memory. You can:

1. **Gradual migration** — keep using MEMORY.md while Memorus handles new memories
2. **Full migration** — run `migrate.py` and disable OpenClaw's built-in memory
3. **Hybrid** — use OpenClaw memory for session notes, Memorus for long-term knowledge

## FAQ

**Q: Does this require an API key?**
A: No. The default `rules` reflector mode works fully offline. Switch to `hybrid` mode if you want LLM-powered reflection (requires an API key).

**Q: What about my existing MEMORY.md?**
A: Use `migrate.py` to import it. The original file is not modified.

**Q: How is deduplication handled?**
A: The ACE engine automatically detects near-duplicate content and merges it, keeping the most complete version.

**Q: Can I use this with multiple projects?**
A: Yes. Use the `scope` parameter: `scope="project:myapp"` for project-specific memories.
