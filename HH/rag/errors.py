"""
HarnessHarvester - Error Pattern RAG Store

Manages error patterns and their fixes. When agents encounter errors,
they can search this store for known solutions and contribute new ones.
"""

import os
from typing import Any, Dict, List, Optional

from core.constants import RAG_ERRORS_DIR, RAG_TYPE_ERROR
from core.storage import ensure_dir
from core.logging_setup import get_logger
from rag.engine import RAGEngine, RAGEntry, SearchResult

logger = get_logger("rag_errors")


class ErrorStore:
    """
    Error pattern RAG store.
    Stores error messages, tracebacks, and their successful fixes.
    Agents can query for known solutions when debugging harness issues.
    """

    def __init__(self, store_dir: str = None):
        self.store_dir = store_dir or RAG_ERRORS_DIR
        ensure_dir(self.store_dir)
        self._engine = RAGEngine(self.store_dir)

    # ── Add / Update ─────────────────────────────────────────

    def add_error_pattern(
        self,
        error_message: str,
        fix_description: str,
        fix_code: str = "",
        error_type: str = "",
        traceback: str = "",
        tags: Optional[List[str]] = None,
        source: str = "",
        quality_score: float = 0.5,
    ) -> str:
        """
        Add an error pattern with its fix to the store.

        Args:
            error_message: The error message or pattern
            fix_description: Human-readable description of the fix
            fix_code: Code that fixes the error (if applicable)
            error_type: Type of error (e.g., TypeError, ImportError)
            traceback: Full traceback (optional)
            tags: Additional tags for categorization
            source: Who contributed this (harness_id, etc.)
            quality_score: Initial quality rating
        """
        # Build structured content
        content_parts = [
            f"ERROR: {error_message}",
        ]
        if error_type:
            content_parts.append(f"TYPE: {error_type}")
        if traceback:
            content_parts.append(f"TRACEBACK:\n{traceback}")
        content_parts.append(f"\nFIX: {fix_description}")
        if fix_code:
            content_parts.append(f"\nFIX CODE:\n{fix_code}")

        content = "\n".join(content_parts)

        auto_tags = tags or []
        if error_type and error_type not in auto_tags:
            auto_tags.append(error_type)
        auto_tags.extend(self._extract_error_tags(error_message))
        auto_tags = list(set(auto_tags))

        entry = RAGEntry(
            id="",
            content=content,
            entry_type=RAG_TYPE_ERROR,
            title=f"Fix: {error_message[:100]}",
            description=fix_description,
            tags=auto_tags,
            language="python",
            metadata={
                "error_message": error_message,
                "error_type": error_type,
                "fix_description": fix_description,
                "fix_code": fix_code,
                "traceback": traceback[:2000],  # Limit traceback size
            },
            quality_score=quality_score,
            source=source,
        )

        return self._engine.add(entry)

    def update_error_pattern(self, error_id: str, **updates) -> bool:
        """Update an error pattern."""
        return self._engine.update(error_id, updates)

    def delete_error_pattern(self, error_id: str) -> bool:
        """Delete an error pattern."""
        return self._engine.delete(error_id)

    # ── Search ───────────────────────────────────────────────

    def find_fix(
        self,
        error_message: str,
        error_type: str = "",
        context: str = "",
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Find known fixes for an error.
        Combines error message, type, and context for best matching.
        """
        query_parts = [error_message]
        if error_type:
            query_parts.append(error_type)
        if context:
            query_parts.append(context)
        query = " ".join(query_parts)

        return self._engine.search(
            query=query,
            top_k=top_k,
            entry_type=RAG_TYPE_ERROR,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        tags: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """General search in error patterns."""
        return self._engine.search(
            query=query,
            top_k=top_k,
            entry_type=RAG_TYPE_ERROR,
            tags=tags,
        )

    def get_error_pattern(self, error_id: str) -> Optional[RAGEntry]:
        """Get an error pattern by ID."""
        return self._engine.get(error_id)

    def record_usage(self, error_id: str, fix_worked: bool = True):
        """Record that a fix was tried."""
        self._engine.record_usage(error_id, success=fix_worked)

    # ── Listing ──────────────────────────────────────────────

    def list_patterns(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "usage_count",
    ) -> List[RAGEntry]:
        """List error patterns."""
        return self._engine.list_entries(
            entry_type=RAG_TYPE_ERROR,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
        )

    def stats(self) -> Dict[str, Any]:
        """Get error store statistics."""
        base = self._engine.stats()
        base["store_type"] = "errors"
        return base

    def save(self):
        """Persist indexes to disk."""
        self._engine.save()

    @staticmethod
    def _extract_error_tags(error_message: str) -> List[str]:
        """Extract useful tags from an error message."""
        import re
        tags = []

        # Common Python error types
        type_match = re.match(r"(\w+Error|\w+Exception|\w+Warning)", error_message)
        if type_match:
            tags.append(type_match.group(1))

        # Module names
        for match in re.finditer(r"module '(\w+)'", error_message):
            tags.append(match.group(1))

        # Variable/attribute names
        for match in re.finditer(r"'(\w+)'", error_message):
            name = match.group(1)
            if len(name) > 2 and name.lower() not in {"the", "and", "for"}:
                tags.append(name)

        return tags[:10]
