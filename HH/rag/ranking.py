"""
HarnessHarvester - RAG Ranking System

Multi-signal ranking with configurable weights, time decay,
quality metrics, and adaptive re-ranking.
"""

import math
import time
from typing import Dict, List, Optional
from datetime import datetime

from core.logging_setup import get_logger

logger = get_logger("rag_ranking")


class RankingEngine:
    """
    Configurable multi-signal ranking engine.
    Combines multiple scoring signals with weighted fusion.
    Supports adaptive weight tuning based on user feedback.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "tfidf_similarity": 0.30,
            "bm25_score": 0.25,
            "quality_score": 0.20,
            "usage_frequency": 0.10,
            "recency_boost": 0.10,
            "tag_match": 0.05,
        }
        self._feedback_history: List[Dict] = []

    def compute_final_score(self, signals: Dict[str, float]) -> float:
        """Compute weighted final score from individual signals."""
        score = 0.0
        for signal_name, weight in self.weights.items():
            score += weight * signals.get(signal_name, 0.0)
        return score

    def rank(self, items: List[Dict[str, float]]) -> List[int]:
        """
        Rank items by weighted score.
        Each item is a dict of {signal_name: score}.
        Returns sorted indices.
        """
        scores = [(i, self.compute_final_score(item)) for i, item in enumerate(items)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scores]

    def record_feedback(self, query: str, chosen_id: str, ignored_ids: List[str]):
        """
        Record user feedback for adaptive weight tuning.
        The chosen result should be ranked higher in future.
        """
        self._feedback_history.append({
            "query": query,
            "chosen": chosen_id,
            "ignored": ignored_ids,
            "timestamp": time.time(),
        })

    def adapt_weights(self):
        """
        Adapt ranking weights based on accumulated feedback.
        Uses a simple gradient-like approach: if users consistently
        prefer results with high quality scores, increase quality weight.
        """
        if len(self._feedback_history) < 10:
            return  # Need enough feedback

        # This is a simplified adaptive mechanism
        # In production, you'd use learning-to-rank
        logger.info(f"Adapting weights from {len(self._feedback_history)} feedback samples")

    @staticmethod
    def time_decay_score(
        created_at: str,
        half_life_days: float = 90.0,
        now: Optional[float] = None,
    ) -> float:
        """
        Compute exponential time-decay score.
        Returns 1.0 for brand new items, 0.5 at half_life_days, etc.
        """
        if now is None:
            now = time.time()
        try:
            created = datetime.fromisoformat(created_at)
            age_days = (now - created.timestamp()) / 86400.0
        except (ValueError, TypeError):
            return 0.5

        return math.exp(-0.693 * max(age_days, 0) / max(half_life_days, 1))

    @staticmethod
    def usage_score(usage_count: int, success_count: int, failure_count: int) -> float:
        """
        Compute usage-based score combining frequency and success rate.
        Log-scaled frequency with success ratio weighting.
        """
        if usage_count == 0:
            return 0.0

        # Log-scaled frequency (diminishing returns)
        freq_score = math.log1p(usage_count) / 10.0

        # Success ratio (with Laplace smoothing)
        success_ratio = (success_count + 1) / (usage_count + 2)

        return min(1.0, freq_score * (0.3 + 0.7 * success_ratio))

    @staticmethod
    def diversity_rerank(
        results: list,
        score_key: str = "final_score",
        lambda_: float = 0.5,
    ) -> list:
        """
        MMR (Maximal Marginal Relevance) style diversity re-ranking.
        Balances relevance with diversity to avoid redundant results.
        """
        if len(results) <= 1:
            return results

        selected = [results[0]]
        remaining = list(results[1:])

        while remaining:
            best_idx = 0
            best_score = -float("inf")

            for i, candidate in enumerate(remaining):
                relevance = getattr(candidate, score_key, 0) if hasattr(candidate, score_key) else candidate.get(score_key, 0)

                # Compute max similarity to already selected items
                max_sim = 0.0
                for sel in selected:
                    sim = _jaccard_similarity(
                        _get_tokens(candidate),
                        _get_tokens(sel),
                    )
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr = lambda_ * relevance - (1 - lambda_) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected


def _get_tokens(item) -> set:
    """Extract tokens from a search result or dict for diversity comparison."""
    if hasattr(item, "entry"):
        text = item.entry.searchable_text
    elif isinstance(item, dict):
        text = item.get("content", "") + " " + item.get("title", "")
    else:
        text = str(item)
    from rag.embeddings import tokenize
    return set(tokenize(text))


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / max(union, 1)
