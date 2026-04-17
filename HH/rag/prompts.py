"""
HarnessHarvester - Prompt Template RAG Store

Manages successful prompts that agents can access and learn from.
Tracks performance metrics per prompt to surface the most effective templates.
"""

import os
from typing import Any, Dict, List, Optional

from core.constants import RAG_PROMPTS_DIR, RAG_TYPE_PROMPT
from core.storage import ensure_dir
from core.logging_setup import get_logger
from rag.engine import RAGEngine, RAGEntry, SearchResult

logger = get_logger("rag_prompts")


class PromptStore:
    """
    Prompt template RAG store.
    Stores successful prompts with performance metrics, allowing agents
    to learn from past interactions and improve prompt engineering.
    """

    def __init__(self, store_dir: str = None):
        self.store_dir = store_dir or RAG_PROMPTS_DIR
        ensure_dir(self.store_dir)
        self._engine = RAGEngine(self.store_dir)

    # ── Add / Update ─────────────────────────────────────────

    def add_prompt_template(
        self,
        template: str,
        task_type: str = "",
        model_used: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        source: str = "",
        quality_score: float = 0.5,
        performance_metrics: Optional[Dict[str, Any]] = None,
        variables: Optional[List[str]] = None,
    ) -> str:
        """
        Add a prompt template to the store.

        Args:
            template: The prompt template text (may contain {variable} placeholders)
            task_type: Type of task (e.g., code_generation, review, summarization)
            model_used: Which model this prompt works best with
            description: Human-readable description
            tags: Categorization tags
            source: Who contributed this
            quality_score: Initial quality rating
            performance_metrics: Dict of metrics (e.g., avg_score, success_rate)
            variables: List of template variables used
        """
        auto_tags = list(tags or [])
        if task_type and task_type not in auto_tags:
            auto_tags.append(task_type)
        if model_used and model_used not in auto_tags:
            auto_tags.append(model_used.split("/")[-1].split(":")[0])  # Short model name

        # Auto-detect variables
        if variables is None:
            import re
            variables = re.findall(r"\{(\w+)\}", template)
            variables = list(set(variables))

        entry = RAGEntry(
            id="",
            content=template,
            entry_type=RAG_TYPE_PROMPT,
            title=description or f"Prompt: {task_type or 'general'}",
            description=description,
            tags=auto_tags,
            language="prompt",
            metadata={
                "task_type": task_type,
                "model_used": model_used,
                "performance_metrics": performance_metrics or {},
                "variables": variables,
            },
            quality_score=quality_score,
            source=source,
        )

        return self._engine.add(entry)

    def update_prompt(self, prompt_id: str, **updates) -> bool:
        """Update a prompt template."""
        return self._engine.update(prompt_id, updates)

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt template."""
        return self._engine.delete(prompt_id)

    def rate_prompt(self, prompt_id: str, score: float, success: bool = True):
        """
        Rate a prompt's effectiveness after use.
        Updates quality score using exponential moving average.
        """
        entry = self._engine.get(prompt_id)
        if not entry:
            return

        # Exponential moving average for quality
        alpha = 0.3  # Learning rate
        new_quality = entry.quality_score * (1 - alpha) + score * alpha
        self._engine.update(prompt_id, {"quality_score": new_quality})
        self._engine.record_usage(prompt_id, success=success)

    # ── Search ───────────────────────────────────────────────

    def find_prompt(
        self,
        task_description: str,
        task_type: str = "",
        model: str = "",
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Find the best prompt templates for a given task.

        Args:
            task_description: What the prompt should accomplish
            task_type: Filter by task type
            model: Filter by model compatibility
            top_k: Number of results
        """
        query_parts = [task_description]
        if task_type:
            query_parts.append(task_type)

        tags = []
        if task_type:
            tags.append(task_type)
        if model:
            tags.append(model.split("/")[-1].split(":")[0])

        return self._engine.search(
            query=" ".join(query_parts),
            top_k=top_k,
            entry_type=RAG_TYPE_PROMPT,
            tags=tags if tags else None,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        tags: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """General search in prompt templates."""
        return self._engine.search(
            query=query,
            top_k=top_k,
            entry_type=RAG_TYPE_PROMPT,
            tags=tags,
        )

    def get_prompt(self, prompt_id: str) -> Optional[RAGEntry]:
        """Get a prompt template by ID."""
        return self._engine.get(prompt_id)

    # ── Listing ──────────────────────────────────────────────

    def list_prompts(
        self,
        limit: int = 50,
        offset: int = 0,
        task_type: str = "",
        sort_by: str = "quality_score",
    ) -> List[RAGEntry]:
        """List prompt templates."""
        entries = self._engine.list_entries(
            entry_type=RAG_TYPE_PROMPT,
            limit=limit * 2 if task_type else limit,
            offset=0 if task_type else offset,
            sort_by=sort_by,
        )
        if task_type:
            entries = [
                e for e in entries
                if e.metadata.get("task_type") == task_type
            ]
            entries = entries[offset: offset + limit]
        return entries

    def get_best_prompt_for_task(self, task_type: str) -> Optional[RAGEntry]:
        """Get the highest-rated prompt for a task type."""
        entries = self.list_prompts(limit=1, task_type=task_type, sort_by="quality_score")
        return entries[0] if entries else None

    def stats(self) -> Dict[str, Any]:
        """Get prompt store statistics."""
        base = self._engine.stats()
        base["store_type"] = "prompts"

        # Add task type breakdown
        entries = self._engine.list_entries(entry_type=RAG_TYPE_PROMPT, limit=10000)
        task_types = {}
        for e in entries:
            tt = e.metadata.get("task_type", "unknown")
            task_types[tt] = task_types.get(tt, 0) + 1
        base["task_types"] = task_types

        return base

    def save(self):
        """Persist indexes to disk."""
        self._engine.save()
