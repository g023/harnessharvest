"""
HarnessHarvester - Code Snippet RAG Store

Manages validated, reusable code snippets that harness operations can
contribute to and query from. Includes validation and quality scoring.
"""

import os
from typing import Any, Dict, List, Optional

from core.constants import RAG_SNIPPETS_DIR, RAG_TYPE_SNIPPET
from core.storage import ensure_dir
from core.logging_setup import get_logger
from rag.engine import RAGEngine, RAGEntry, SearchResult

logger = get_logger("rag_snippets")


class SnippetStore:
    """
    Code snippet RAG store.
    Manages a collection of validated, reusable code snippets with
    metadata like language, tags, quality scores, and usage metrics.
    """

    def __init__(self, store_dir: str = None):
        self.store_dir = store_dir or RAG_SNIPPETS_DIR
        ensure_dir(self.store_dir)
        self._engine = RAGEngine(self.store_dir)

    # ── Add / Update ─────────────────────────────────────────

    def add_snippet(
        self,
        content: str,
        title: str = "",
        description: str = "",
        language: str = "python",
        tags: Optional[List[str]] = None,
        source: str = "",
        quality_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        validated: bool = False,
    ) -> str:
        """Add a code snippet to the store."""
        from core.config import config
        min_len = config.get("rag.min_snippet_length", 20)
        max_len = config.get("rag.max_snippet_length", 50000)

        if len(content.strip()) < min_len:
            raise ValueError(f"Snippet too short (min {min_len} chars)")
        if len(content) > max_len:
            raise ValueError(f"Snippet too long (max {max_len} chars)")

        entry = RAGEntry(
            id="",
            content=content,
            entry_type=RAG_TYPE_SNIPPET,
            title=title or self._auto_title(content),
            description=description,
            tags=tags or self._auto_tags(content, language),
            language=language,
            metadata=metadata or {},
            quality_score=quality_score,
            source=source,
            validated=validated,
        )

        return self._engine.add(entry)

    def update_snippet(self, snippet_id: str, **updates) -> bool:
        """Update a snippet's metadata."""
        return self._engine.update(snippet_id, updates)

    def delete_snippet(self, snippet_id: str) -> bool:
        """Delete a snippet."""
        return self._engine.delete(snippet_id)

    def validate_snippet(self, snippet_id: str, quality_score: float = None) -> bool:
        """Mark a snippet as validated."""
        updates = {"validated": True}
        if quality_score is not None:
            updates["quality_score"] = max(0.0, min(1.0, quality_score))
        return self._engine.update(snippet_id, updates)

    # ── Search ───────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        language: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_quality: float = 0.0,
        validated_only: bool = False,
    ) -> List[SearchResult]:
        """Search for code snippets."""
        results = self._engine.search(
            query=query,
            top_k=top_k * 2 if validated_only else top_k,
            entry_type=RAG_TYPE_SNIPPET,
            tags=tags,
            min_quality=min_quality,
        )

        if language:
            results = [r for r in results if r.entry.language == language]

        if validated_only:
            results = [r for r in results if r.entry.validated]

        return results[:top_k]

    def get_snippet(self, snippet_id: str) -> Optional[RAGEntry]:
        """Get a snippet by ID."""
        return self._engine.get(snippet_id)

    def record_usage(self, snippet_id: str, success: bool = True):
        """Record snippet usage for ranking."""
        self._engine.record_usage(snippet_id, success)

    # ── Listing ──────────────────────────────────────────────

    def list_snippets(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "updated_at",
        language: Optional[str] = None,
    ) -> List[RAGEntry]:
        """List snippets with pagination."""
        entries = self._engine.list_entries(
            entry_type=RAG_TYPE_SNIPPET,
            limit=limit * 2 if language else limit,
            offset=offset if not language else 0,
            sort_by=sort_by,
        )
        if language:
            entries = [e for e in entries if e.language == language]
            entries = entries[offset: offset + limit]
        return entries

    def stats(self) -> Dict[str, Any]:
        """Get snippet store statistics."""
        base = self._engine.stats()
        base["store_type"] = "snippets"
        return base

    def save(self):
        """Persist indexes to disk."""
        self._engine.save()

    # ── Auto-generation helpers ──────────────────────────────

    @staticmethod
    def _auto_title(content: str) -> str:
        """Generate a title from the first meaningful line of code."""
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                if stripped.startswith("def "):
                    return stripped.split("(")[0].replace("def ", "Function: ")
                if stripped.startswith("class "):
                    return stripped.split("(")[0].split(":")[0].replace("class ", "Class: ")
                return stripped[:80]
        return "Untitled Snippet"

    @staticmethod
    def _auto_tags(content: str, language: str) -> List[str]:
        """Auto-generate tags from code content."""
        import re
        tags = [language]

        # Extract function/class names
        for match in re.finditer(r"def\s+(\w+)", content):
            name = match.group(1)
            if not name.startswith("_"):
                tags.append(name)
        for match in re.finditer(r"class\s+(\w+)", content):
            tags.append(match.group(1))

        # Extract imports
        for match in re.finditer(r"(?:from|import)\s+([\w.]+)", content):
            tags.append(match.group(1).split(".")[0])

        return list(set(tags))[:20]  # Limit tags
