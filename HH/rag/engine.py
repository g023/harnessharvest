"""
HarnessHarvester - RAG Engine

Core search engine combining FAISS vector search with BM25 ranking.
Provides hybrid retrieval with configurable weighting.
"""

import os
import time
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from core.storage import (
    atomic_write_json, read_json, locked_update_json,
    ensure_dir, generate_id, content_hash
)
from core.logging_setup import get_logger
from rag.embeddings import TFIDFVectorizer, BM25Scorer, tokenize

logger = get_logger("rag_engine")


# ─── Data Models ─────────────────────────────────────────────────

@dataclass
class RAGEntry:
    """A single entry in the RAG store."""
    id: str
    content: str
    entry_type: str  # snippet, error_pattern, prompt_template
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.5  # 0.0 to 1.0
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    content_hash: str = ""
    source: str = ""  # who contributed this (harness_id, user, etc.)
    validated: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "RAGEntry":
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @property
    def searchable_text(self) -> str:
        """Combined text for indexing."""
        parts = [self.title, self.description, self.content]
        parts.extend(self.tags)
        return " ".join(filter(None, parts))


@dataclass
class SearchResult:
    """A search result with scoring breakdown."""
    entry: RAGEntry
    final_score: float
    tfidf_score: float = 0.0
    bm25_score: float = 0.0
    quality_score: float = 0.0
    usage_score: float = 0.0
    recency_score: float = 0.0
    tag_score: float = 0.0


# ─── FAISS Index Wrapper ─────────────────────────────────────────

class VectorIndex:
    """
    FAISS-backed vector index with fallback to brute-force numpy search.
    Supports add, search, remove, and persistence.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._ids: List[str] = []  # Parallel array: index position -> entry ID
        self._id_to_idx: Dict[str, int] = {}  # entry ID -> index position
        self._vectors: Optional[np.ndarray] = None
        self._needs_rebuild: bool = False

        if FAISS_AVAILABLE:
            # Use IndexFlatIP for cosine similarity (vectors must be L2-normalized)
            self._index = faiss.IndexFlatIP(dimension)
            logger.info(f"FAISS index initialized (dim={dimension})")
        else:
            self._index = None
            logger.warning("FAISS not available, using numpy fallback")

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, entry_id: str, vector: np.ndarray):
        """Add a vector to the index."""
        vector = vector.astype(np.float32).reshape(1, -1)

        # Pad or truncate to match dimension
        if vector.shape[1] < self.dimension:
            padded = np.zeros((1, self.dimension), dtype=np.float32)
            padded[0, :vector.shape[1]] = vector[0]
            vector = padded
        elif vector.shape[1] > self.dimension:
            vector = vector[:, :self.dimension]

        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        if entry_id in self._id_to_idx:
            # Update existing
            idx = self._id_to_idx[entry_id]
            if self._vectors is not None:
                self._vectors[idx] = vector[0]
            # FAISS doesn't support in-place update, so we mark for rebuild
            self._needs_rebuild = True
        else:
            # Add new
            self._id_to_idx[entry_id] = len(self._ids)
            self._ids.append(entry_id)

        # Maintain numpy mirror
        if self._vectors is None:
            self._vectors = vector.copy()
        else:
            if entry_id not in self._id_to_idx or self._id_to_idx[entry_id] == len(self._ids) - 1:
                self._vectors = np.vstack([self._vectors, vector])

        # Add to FAISS
        if self._index is not None:
            if self._needs_rebuild:
                self._rebuild_faiss()
            else:
                self._index.add(vector)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for nearest neighbors. Returns [(entry_id, score), ...]."""
        if self.size == 0:
            return []

        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        # Pad or truncate
        if query_vector.shape[1] < self.dimension:
            padded = np.zeros((1, self.dimension), dtype=np.float32)
            padded[0, :query_vector.shape[1]] = query_vector[0]
            query_vector = padded
        elif query_vector.shape[1] > self.dimension:
            query_vector = query_vector[:, :self.dimension]

        # L2 normalize
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector /= norm

        k = min(top_k, self.size)

        if self._index is not None and self._index.ntotal > 0:
            scores, indices = self._index.search(query_vector, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self._ids):
                    results.append((self._ids[idx], float(score)))
            return results
        elif self._vectors is not None:
            # Numpy fallback: cosine similarity via dot product (vectors are normalized)
            similarities = self._vectors @ query_vector.T
            similarities = similarities.flatten()
            top_indices = np.argsort(similarities)[::-1][:k]
            return [
                (self._ids[idx], float(similarities[idx]))
                for idx in top_indices
                if idx < len(self._ids)
            ]

        return []

    def remove(self, entry_id: str):
        """Mark an entry for removal. Requires rebuild for FAISS."""
        if entry_id in self._id_to_idx:
            idx = self._id_to_idx.pop(entry_id)
            self._ids[idx] = None  # Mark as removed
            self._needs_rebuild = True

    def _rebuild_faiss(self):
        """Rebuild FAISS index from numpy vectors (after updates/removals)."""
        if self._vectors is None:
            return

        # Compact: remove None entries
        new_ids = []
        new_vectors = []
        for i, eid in enumerate(self._ids):
            if eid is not None and i < len(self._vectors):
                new_ids.append(eid)
                new_vectors.append(self._vectors[i])

        self._ids = new_ids
        self._id_to_idx = {eid: i for i, eid in enumerate(self._ids)}

        if new_vectors:
            self._vectors = np.vstack(new_vectors)
        else:
            self._vectors = None

        if self._index is not None:
            self._index.reset()
            if self._vectors is not None and len(self._vectors) > 0:
                self._index.add(self._vectors)

        self._needs_rebuild = False
        logger.debug(f"FAISS index rebuilt: {self.size} vectors")

    # ── Persistence ──────────────────────────────────────────

    def save(self, directory: str):
        """Save index to directory."""
        ensure_dir(directory)

        # Compact before saving to remove None entries from removals
        if any(eid is None for eid in self._ids):
            self._rebuild_faiss()

        # Save ID mapping
        atomic_write_json(
            os.path.join(directory, "vector_ids.json"),
            {"ids": self._ids, "dimension": self.dimension}
        )

        # Save vectors as binary
        if self._vectors is not None and len(self._vectors) > 0:
            vectors_path = os.path.join(directory, "vectors.npy")
            np.save(vectors_path, self._vectors)

        # Save FAISS index if available
        if self._index is not None and self._index.ntotal > 0:
            faiss_path = os.path.join(directory, "faiss.index")
            faiss.write_index(self._index, faiss_path)

    @classmethod
    def load(cls, directory: str) -> "VectorIndex":
        """Load index from directory."""
        ids_data = read_json(os.path.join(directory, "vector_ids.json"))
        if ids_data is None:
            return cls(512)

        dimension = ids_data.get("dimension", 512)
        index = cls(dimension)
        index._ids = ids_data.get("ids", [])
        index._id_to_idx = {eid: i for i, eid in enumerate(index._ids) if eid is not None}

        # Load vectors
        vectors_path = os.path.join(directory, "vectors.npy")
        if os.path.exists(vectors_path):
            index._vectors = np.load(vectors_path)

        # Load FAISS index
        if FAISS_AVAILABLE:
            faiss_path = os.path.join(directory, "faiss.index")
            if os.path.exists(faiss_path):
                index._index = faiss.read_index(faiss_path)
            elif index._vectors is not None and len(index._vectors) > 0:
                index._index = faiss.IndexFlatIP(dimension)
                index._index.add(index._vectors)

        return index


# ─── Hybrid RAG Engine ───────────────────────────────────────────

class RAGEngine:
    """
    Hybrid search engine combining:
    1. FAISS vector similarity (TF-IDF vectors)
    2. BM25 probabilistic ranking
    3. Quality scores, usage metrics, recency boost, tag matching

    This is the core search interface for all RAG stores.
    """

    def __init__(self, store_dir: str, embedding_dim: int = 512):
        self.store_dir = store_dir
        self.embedding_dim = embedding_dim

        # Sub-paths
        self._entries_dir = os.path.join(store_dir, "entries")
        self._index_dir = os.path.join(store_dir, "index")
        ensure_dir(self._entries_dir)
        ensure_dir(self._index_dir)

        # Components
        self._vectorizer: Optional[TFIDFVectorizer] = None
        self._bm25: Optional[BM25Scorer] = None
        self._vector_index: Optional[VectorIndex] = None

        # In-memory cache
        self._entries: Dict[str, RAGEntry] = {}
        self._content_hashes: Dict[str, str] = {}  # hash -> entry_id

        # Load existing state
        self._load()

    # ── CRUD Operations ──────────────────────────────────────

    def add(self, entry: RAGEntry) -> str:
        """Add an entry to the RAG store. Returns entry ID."""
        # Generate ID if not set
        if not entry.id:
            entry.id = generate_id("rag")

        # Set timestamps
        now = datetime.now().isoformat()
        if not entry.created_at:
            entry.created_at = now
        entry.updated_at = now

        # Compute content hash
        entry.content_hash = content_hash(entry.content)

        # Check for duplicates
        if entry.content_hash in self._content_hashes:
            existing_id = self._content_hashes[entry.content_hash]
            logger.info(f"Duplicate detected, updating existing entry {existing_id}")
            entry.id = existing_id
            existing = self._entries.get(existing_id)
            if existing:
                entry.usage_count = existing.usage_count
                entry.success_count = existing.success_count

        # Store entry
        self._entries[entry.id] = entry
        self._content_hashes[entry.content_hash] = entry.id

        # Persist entry file
        entry_path = os.path.join(self._entries_dir, f"{entry.id}.json")
        atomic_write_json(entry_path, entry.to_dict())

        # Rebuild indexes
        self._rebuild_indexes()

        logger.info(f"RAG entry added: {entry.id} ({entry.entry_type})")
        return entry.id

    def get(self, entry_id: str) -> Optional[RAGEntry]:
        """Get an entry by ID."""
        if entry_id in self._entries:
            return self._entries[entry_id]

        entry_path = os.path.join(self._entries_dir, f"{entry_id}.json")
        data = read_json(entry_path)
        if data:
            entry = RAGEntry.from_dict(data)
            self._entries[entry.id] = entry
            return entry
        return None

    def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an entry's fields."""
        entry = self.get(entry_id)
        if not entry:
            return False

        for key, value in updates.items():
            if hasattr(entry, key) and key not in ("id", "created_at", "content_hash"):
                setattr(entry, key, value)

        entry.updated_at = datetime.now().isoformat()
        if "content" in updates:
            entry.content_hash = content_hash(entry.content)
            self._content_hashes[entry.content_hash] = entry.id

        self._entries[entry.id] = entry
        entry_path = os.path.join(self._entries_dir, f"{entry.id}.json")
        atomic_write_json(entry_path, entry.to_dict())

        if "content" in updates or "title" in updates or "tags" in updates:
            self._rebuild_indexes()

        return True

    def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        entry = self._entries.pop(entry_id, None)
        if entry:
            self._content_hashes.pop(entry.content_hash, None)

        entry_path = os.path.join(self._entries_dir, f"{entry_id}.json")
        try:
            os.remove(entry_path)
        except FileNotFoundError:
            if not entry:
                return False

        self._rebuild_indexes()
        logger.info(f"RAG entry deleted: {entry_id}")
        return True

    def record_usage(self, entry_id: str, success: bool = True):
        """Record that an entry was used (for ranking)."""
        entry = self.get(entry_id)
        if entry:
            entry.usage_count += 1
            if success:
                entry.success_count += 1
            else:
                entry.failure_count += 1
            entry.updated_at = datetime.now().isoformat()
            entry_path = os.path.join(self._entries_dir, f"{entry.id}.json")
            atomic_write_json(entry_path, entry.to_dict())

    # ── Search ───────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        entry_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_quality: float = 0.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity, BM25, and metadata signals.

        Args:
            query: Search query text
            top_k: Number of results to return
            entry_type: Filter by entry type (snippet, error_pattern, prompt_template)
            tags: Filter by tags (any match)
            min_quality: Minimum quality score filter
            weights: Override default ranking weights
        """
        if not self._entries:
            return []

        # Default weights from config
        if weights is None:
            from core.config import config
            weights = config.get("rag.ranking_weights", {
                "tfidf_similarity": 0.30,
                "bm25_score": 0.25,
                "quality_score": 0.20,
                "usage_frequency": 0.10,
                "recency_boost": 0.10,
                "tag_match": 0.05,
            })

        # Get candidate entries (pre-filter)
        candidates = self._get_candidates(entry_type, tags, min_quality)
        if not candidates:
            return []

        # Build candidate ID list and index mapping
        candidate_ids = [e.id for e in candidates]
        id_to_candidate = {e.id: e for e in candidates}

        # 1. TF-IDF vector similarity via FAISS
        tfidf_scores = {}
        if self._vector_index and self._vectorizer and self._vectorizer._fitted:
            query_vec = self._vectorizer.transform(query)
            # Search more than top_k to account for filtered-out entries
            vec_results = self._vector_index.search(query_vec, top_k=min(top_k * 3, self._vector_index.size))
            for eid, score in vec_results:
                if eid in id_to_candidate:
                    tfidf_scores[eid] = max(0.0, float(score))

        # 2. BM25 scores
        bm25_scores = {}
        if self._bm25 and self._bm25._fitted:
            all_ids = list(self._entries.keys())
            raw_scores = self._bm25.score_all(query)
            max_bm25 = max(raw_scores) if raw_scores and max(raw_scores) > 0 else 1.0
            for idx, eid in enumerate(all_ids):
                if eid in id_to_candidate and idx < len(raw_scores):
                    bm25_scores[eid] = raw_scores[idx] / max_bm25  # Normalize to 0-1

        # 3. Compute final scores
        results = []
        now_epoch = time.time()

        for entry in candidates:
            eid = entry.id

            # Individual signal scores
            tfidf_s = tfidf_scores.get(eid, 0.0)
            bm25_s = bm25_scores.get(eid, 0.0)
            quality_s = entry.quality_score
            usage_s = self._compute_usage_score(entry)
            recency_s = self._compute_recency_score(entry, now_epoch)
            tag_s = self._compute_tag_score(entry, query, tags)

            # Weighted combination
            final = (
                weights.get("tfidf_similarity", 0.3) * tfidf_s
                + weights.get("bm25_score", 0.25) * bm25_s
                + weights.get("quality_score", 0.2) * quality_s
                + weights.get("usage_frequency", 0.1) * usage_s
                + weights.get("recency_boost", 0.1) * recency_s
                + weights.get("tag_match", 0.05) * tag_s
            )

            results.append(SearchResult(
                entry=entry,
                final_score=final,
                tfidf_score=tfidf_s,
                bm25_score=bm25_s,
                quality_score=quality_s,
                usage_score=usage_s,
                recency_score=recency_s,
                tag_score=tag_s,
            ))

        # Sort by final score descending
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    # ── Scoring Helpers ──────────────────────────────────────

    def _get_candidates(
        self,
        entry_type: Optional[str],
        tags: Optional[List[str]],
        min_quality: float,
    ) -> List[RAGEntry]:
        """Filter entries by type, tags, and quality."""
        candidates = []
        for entry in self._entries.values():
            if entry_type and entry.entry_type != entry_type:
                continue
            if min_quality > 0 and entry.quality_score < min_quality:
                continue
            if tags:
                entry_tags_lower = {t.lower() for t in entry.tags}
                if not any(t.lower() in entry_tags_lower for t in tags):
                    continue
            candidates.append(entry)
        return candidates

    @staticmethod
    def _compute_usage_score(entry: RAGEntry) -> float:
        """Compute usage-based score (log scale with success ratio)."""
        if entry.usage_count == 0:
            return 0.0
        base = math.log1p(entry.usage_count) / 10.0  # Log scale, max ~1.0 at ~22000 uses
        success_ratio = entry.success_count / max(entry.usage_count, 1)
        return min(1.0, base * (0.5 + 0.5 * success_ratio))

    @staticmethod
    def _compute_recency_score(entry: RAGEntry, now_epoch: float) -> float:
        """Compute time-decay recency score with configurable half-life."""
        try:
            created = datetime.fromisoformat(entry.updated_at or entry.created_at)
            age_days = (now_epoch - created.timestamp()) / 86400.0
        except (ValueError, TypeError):
            return 0.5

        half_life = 90.0  # days
        return math.exp(-0.693 * age_days / half_life)

    @staticmethod
    def _compute_tag_score(entry: RAGEntry, query: str, filter_tags: Optional[List[str]]) -> float:
        """Compute tag match score."""
        if not entry.tags:
            return 0.0

        query_tokens = set(tokenize(query))
        tag_tokens = set()
        for tag in entry.tags:
            tag_tokens.update(tokenize(tag))

        if not query_tokens or not tag_tokens:
            return 0.0

        overlap = query_tokens & tag_tokens
        score = len(overlap) / max(len(query_tokens), 1)

        # Boost if explicitly filtered tags match
        if filter_tags:
            filter_match = sum(1 for t in filter_tags if t.lower() in {tg.lower() for tg in entry.tags})
            score = max(score, filter_match / len(filter_tags))

        return min(1.0, score)

    # ── Index Management ─────────────────────────────────────

    def _rebuild_indexes(self):
        """Rebuild TF-IDF vectorizer, BM25 scorer, and FAISS index."""
        if not self._entries:
            return

        entries = list(self._entries.values())
        texts = [e.searchable_text for e in entries]
        ids = [e.id for e in entries]

        # TF-IDF
        self._vectorizer = TFIDFVectorizer(max_features=self.embedding_dim)
        vectors = self._vectorizer.fit_transform(texts)

        # FAISS index
        self._vector_index = VectorIndex(self.embedding_dim)
        for i, (eid, vec) in enumerate(zip(ids, vectors)):
            self._vector_index.add(eid, vec)

        # BM25
        from core.config import config
        self._bm25 = BM25Scorer(
            k1=config.get("rag.bm25_k1", 1.2),
            b=config.get("rag.bm25_b", 0.75),
        )
        self._bm25.fit(texts)

        logger.debug(f"Indexes rebuilt: {len(entries)} entries")

    def _save_indexes(self):
        """Persist all indexes to disk."""
        if self._vectorizer:
            self._vectorizer.save(os.path.join(self._index_dir, "tfidf.json"))
        if self._bm25:
            self._bm25.save(os.path.join(self._index_dir, "bm25.json"))
        if self._vector_index:
            self._vector_index.save(self._index_dir)

    def save(self):
        """Persist everything to disk."""
        self._save_indexes()

        # Save content hash index
        atomic_write_json(
            os.path.join(self.store_dir, "hash_index.json"),
            self._content_hashes
        )

    # ── Loading ──────────────────────────────────────────────

    def _load(self):
        """Load all entries and indexes from disk."""
        # Load entries
        if os.path.isdir(self._entries_dir):
            for filename in os.listdir(self._entries_dir):
                if filename.endswith(".json"):
                    path = os.path.join(self._entries_dir, filename)
                    data = read_json(path)
                    if data:
                        try:
                            entry = RAGEntry.from_dict(data)
                            self._entries[entry.id] = entry
                            if entry.content_hash:
                                self._content_hashes[entry.content_hash] = entry.id
                        except Exception as e:
                            logger.warning(f"Failed to load entry {filename}: {e}")

        # Load content hash index
        hash_idx = read_json(os.path.join(self.store_dir, "hash_index.json"))
        if hash_idx:
            self._content_hashes.update(hash_idx)

        # Rebuild indexes if we have entries
        if self._entries:
            self._rebuild_indexes()

        logger.info(f"RAG engine loaded: {len(self._entries)} entries from {self.store_dir}")

    # ── Statistics ────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Get RAG store statistics."""
        entries = list(self._entries.values())
        types = {}
        for e in entries:
            types[e.entry_type] = types.get(e.entry_type, 0) + 1

        return {
            "total_entries": len(entries),
            "entries_by_type": types,
            "validated_count": sum(1 for e in entries if e.validated),
            "avg_quality_score": (
                sum(e.quality_score for e in entries) / len(entries) if entries else 0
            ),
            "total_usage": sum(e.usage_count for e in entries),
            "faiss_available": FAISS_AVAILABLE,
            "index_dimension": self.embedding_dim,
        }

    def list_entries(
        self,
        entry_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "updated_at",
        ascending: bool = False,
    ) -> List[RAGEntry]:
        """List entries with pagination and sorting."""
        entries = list(self._entries.values())

        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]

        # Sort
        reverse = not ascending
        if sort_by == "quality_score":
            entries.sort(key=lambda e: e.quality_score, reverse=reverse)
        elif sort_by == "usage_count":
            entries.sort(key=lambda e: e.usage_count, reverse=reverse)
        elif sort_by == "created_at":
            entries.sort(key=lambda e: e.created_at or "", reverse=reverse)
        else:
            entries.sort(key=lambda e: e.updated_at or "", reverse=reverse)

        return entries[offset: offset + limit]
