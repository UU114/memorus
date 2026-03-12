# Memorus — Intelligent & Collaborative Memory for AI Agents

> mem0 fork + ACE v2.0 Federated Architecture: auto-learn, auto-collaborate, auto-forget

Memorus extends [mem0](https://github.com/mem0ai/mem0) with the **Adaptive Context Engine (ACE) v2.0** — a federated pipeline that automatically distills, deduplicates, and synchronizes knowledge across local environments and team pools. It ensures your AI agent remembers what matters, corrects what's wrong, and shares what's useful.

---

## Why Memorus: Federated & Local-First

While traditional memory frameworks rely heavily on cloud APIs and offer no team collaboration, Memorus is built on the principle of **Local-First with Optional Federation**:

| Feature | mem0 | Memorus (v2.0) |
|---|---|---|
| **Knowledge Extraction** | Every `add()` calls LLM (~$0.01) | **Hybrid Rules Engine ($0)** + Optional LLM |
| **Deduplication** | LLM-based | **Semantic Merge + Supersede (Correction)** |
| **Forgetting** | None (Permanent) | **Ebbinghaus Decay + Recall Boost** |
| **Team Collaboration** | None | **Federated Sync + Git Fallback (New v2.0)** |
| **Offline Capability** | Unavailable | **Fully Offline** (ONNX + Rules + Git) |
| **Correction Mechanism**| Overwrite/Manual | **Automatic Supersede (Knowledge Evolution)** |
| **Privacy** | None | **12-layer PII Sanitizer + Local-only by default** |

---

## Core Architecture: ACE v2.0

Memorus v2.0 introduces the **Federated Context Engine**, allowing "One person encounters a pitfall, the whole team becomes immune."

### 1. Multi-Tier Memory Pools
- **Local (Global)**: Personal habits and general knowledge (`scope: global`).
- **Local (Project)**: Private experience for specific projects (`scope: project:name`).
- **Team (Org)**: Shared architectural rules and best practices (`scope: team:id`).

### 2. Federated Operation Modes
- **Federation Mode**: Real-time sync via **ACE Sync Server** with P2P validation and governance.
- **Git Fallback**: Zero-infra sharing via `.ace/playbook.jsonl` tracked in Git. PR-based review workflow.

---

## Features

- **Reflector (v2.0)**: Distills conversations into "Knowledge Bullets". Now supports **Sliding Window** to detect user corrections.
- **Curator (v2.0)**: Advanced deduplication. Features **Supersede** mechanism to replace old/wrong knowledge with new corrections.
- **Team Layer**: Seamlessly merges team knowledge into local recall via **Shadow Merging** (read-only, zero contamination).
- **Decay Engine**: Simulates human memory; unused knowledge fades while frequently recalled ones become permanent.
- **Generator**: 4-layer hybrid retrieval (Exact + Fuzzy + Metadata + Vector).
- **Privacy Sanitizer**: Automatically strips API keys, tokens, and PII before any processing.

---

## Installation

```bash
# Core — local-first hybrid mode
pip install memorus

# With Team Federation support
pip install memorus[team]

# With local ONNX embeddings (Full offline)
pip install memorus[onnx]

# Everything (Recommended for Teams)
pip install memorus[all]
```

---

## Quick Start: Local & Team

### Local Intelligent Memory
```python
from memorus import Memory

# Initialize with ACE enabled
m = Memory(config={"ace_enabled": True})

# Add a conversation - ACE automatically distills it
m.add([{"role": "user", "content": "Always use pnpm for this project"}], user_id="dev_1")

# Search - retrieves with hybrid ranking and decay weighting
results = m.search("package manager", user_id="dev_1")
```

### Team Memory (Git Fallback)
Just place a `.ace/playbook.jsonl` in your repository. Memorus will automatically load and merge team knowledge into your local search results.

### Team Memory (Federation)
```python
config = {
    "ace_enabled": True,
    "team": {
        "enabled": True,
        "server_url": "https://ace.your-company.com",
        "subscribed_tags": ["python", "security"]
    }
}
m = Memory(config=config)
```

---

## Key Mechanisms

### 1. Correction & Supersede (New in v1.1+)
When a user says "That's wrong, use X instead", ACE detects the correction意图:
- **Reflector** flags the interaction as `is_correction`.
- **Curator** finds the old related knowledge and **Supersedes** it with the new one.
- The original ID is preserved, but the content "evolves".

### 2. Promotion Pipeline (Team Contribution)
1. **Detection**: ACE identifies high-value local knowledge (recalled >10 times).
2. **Sanitization**: Local Redactor strips all PII/paths.
3. **Nomination**: User reviews and submits to the Team Staging Pool.
4. **Governance**: Team curators or P2P voting approves the knowledge into the Team Pool.

---

## Configuration Reference

```json
{
  "ace": {
    "reflector": { "mode": "hybrid", "llm_model": "gpt-4o-mini" },
    "curator": { "similarity_threshold": 0.8, "correction_score_threshold": 50 },
    "decay": { "half_life_days": 30, "permanent_threshold": 15 },
    "team": {
      "enabled": true,
      "layer_boost": { "local": 1.5, "team": 1.0 },
      "auto_nominate": { "enabled": true, "min_recall_count": 10 }
    }
  }
}
```

---

## CLI Commands

Manage your knowledge base directly from the terminal:

- `memorus status`: Show KB statistics and health.
- `memorus search "query"`: Test retrieval and ranking.
- `memorus learn "Always use X"`: Manually add a distilled rule.
- `memorus nominate`: Review and submit local knowledge to Team.
- `memorus team sync`: Pull latest team updates.
- `memorus conflicts`: Detect contradictory memories.
- `memorus sweep`: Manually trigger the decay engine.

---

## Comparison with Existing Solutions

| Dimension | RAG | ChatGPT Memory | Mem0 | **ACE (v2.0)** |
|------|----------|----------------|------|----------------|
| **Source** | Static Docs | User Input | API | **Auto-Distillation** |
| **Lifecycle** | Manual | Permanent | Manual | **Exponential Decay** |
| **Correction**| None | None | None | **Supersede Mechanism** |
| **Team Sync** | Static | None | Partial | **Federated / Git PR** |
| **Privacy** | Cloud | Cloud | Cloud | **Local-First + Redactor** |
| **Deployment**| Heavy | SaaS | SaaS | **SQLite + ONNX (Local)** |

---

## License

Apache-2.0

---

*For the Chinese version of this documentation, please refer to [README-zh.md](README-zh.md).*
