"""memorus Obsidian adapter.

External, opt-in bridge between memorus knowledge (team + personal) and an
Obsidian vault. memorus is the source of truth; the vault is a read-only
mirror plus a draft inbox that funnels writes through the existing
nominator / governance pipeline (AceSyncClient + Redactor + GovernanceManager).

memorus core is NOT modified by this package. All interactions go through
already-public APIs: Memory, AceSyncClient, TeamCacheStorage, Redactor.
"""

from memorus.obsidian_adapter.exporter import VaultExporter
from memorus.obsidian_adapter.watcher import InboxWatcher

__all__ = ["VaultExporter", "InboxWatcher"]
