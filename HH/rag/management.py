"""
HarnessHarvester - RAG Management Interface

CLI-friendly management operations for all RAG stores.
Provides unified access to snippets, errors, and prompt templates.
"""

import json
from typing import Any, Dict, List, Optional

from core.logging_setup import get_logger
from rag.snippets import SnippetStore
from rag.errors import ErrorStore
from rag.prompts import PromptStore
from rag.engine import RAGEntry

logger = get_logger("rag_management")


class RAGManager:
    """
    Unified management interface for all RAG stores.
    Provides a single API for CRUD, search, stats, and maintenance.
    """

    def __init__(self):
        self.snippets = SnippetStore()
        self.errors = ErrorStore()
        self.prompts = PromptStore()

    # ── Store Access ─────────────────────────────────────────

    def get_store(self, store_type: str):
        """Get a specific store by type name."""
        stores = {
            "snippets": self.snippets,
            "snippet": self.snippets,
            "errors": self.errors,
            "error": self.errors,
            "prompts": self.prompts,
            "prompt": self.prompts,
        }
        store = stores.get(store_type.lower())
        if not store:
            raise ValueError(f"Unknown store type: {store_type}. Use: snippets, errors, prompts")
        return store

    # ── Unified Search ───────────────────────────────────────

    def search_all(
        self,
        query: str,
        top_k: int = 10,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, List]:
        """Search across all stores."""
        return {
            "snippets": self.snippets.search(query, top_k=top_k, tags=tags),
            "errors": self.errors.search(query, top_k=top_k, tags=tags),
            "prompts": self.prompts.search(query, top_k=top_k, tags=tags),
        }

    # ── Stats ────────────────────────────────────────────────

    def all_stats(self) -> Dict[str, Any]:
        """Get statistics for all stores."""
        return {
            "snippets": self.snippets.stats(),
            "errors": self.errors.stats(),
            "prompts": self.prompts.stats(),
        }

    # ── Import/Export ────────────────────────────────────────

    def export_store(self, store_type: str, output_path: str):
        """Export a store's entries to a JSON file."""
        store = self.get_store(store_type)
        entries = store.list_snippets(limit=100000) if store_type in ("snippets", "snippet") else \
                  store.list_patterns(limit=100000) if store_type in ("errors", "error") else \
                  store.list_prompts(limit=100000)

        data = [e.to_dict() for e in entries]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(data)} entries to {output_path}")
        return len(data)

    def import_store(self, store_type: str, input_path: str) -> int:
        """Import entries from a JSON file into a store."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        count = 0
        store = self.get_store(store_type)
        engine = store._engine

        for item in data:
            try:
                entry = RAGEntry.from_dict(item)
                entry.id = ""  # Force new ID generation
                engine.add(entry)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to import entry: {e}")

        logger.info(f"Imported {count}/{len(data)} entries into {store_type}")
        return count

    # ── Maintenance ──────────────────────────────────────────

    def cleanup_low_quality(self, min_quality: float = 0.2) -> int:
        """Remove entries with very low quality scores."""
        removed = 0
        for store_name in ["snippets", "errors", "prompts"]:
            store = self.get_store(store_name)
            engine = store._engine
            to_remove = [
                eid for eid, entry in engine._entries.items()
                if entry.quality_score < min_quality
                and entry.usage_count < 3  # Don't remove frequently used even if low quality
            ]
            for eid in to_remove:
                engine.delete(eid)
                removed += 1

        logger.info(f"Cleaned up {removed} low-quality entries")
        return removed

    def save_all(self):
        """Persist all stores."""
        self.snippets.save()
        self.errors.save()
        self.prompts.save()
