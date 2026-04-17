"""
HarnessHarvester - TF-IDF Embeddings Engine

Pure Python + numpy TF-IDF vectorizer with BM25 scoring.
Provides text vectorization for the RAG system without external NLP libraries.
"""

import math
import re
import os
import json
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter

import numpy as np

from core.storage import atomic_write_json, read_json
from core.logging_setup import get_logger

logger = get_logger("embeddings")

# ─── Tokenizer ───────────────────────────────────────────────────

# Common programming stop words to filter out
_CODE_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "it", "its",
    "this", "that", "these", "those", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "and", "but", "or", "nor", "not", "so", "very", "just", "if",
})


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for TF-IDF. Handles both natural language and code.
    Splits camelCase, snake_case, preserves meaningful programming tokens.
    """
    # Lowercase
    text = text.lower()

    # Split camelCase: insertBefore -> insert before
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Split snake_case: insert_before -> insert before
    text = text.replace("_", " ")

    # Extract words and numbers (keep meaningful tokens)
    tokens = re.findall(r"[a-z][a-z0-9]*|[0-9]+", text)

    # Filter stop words and very short tokens
    tokens = [t for t in tokens if t not in _CODE_STOP_WORDS and len(t) > 1]

    return tokens


def tokenize_with_ngrams(text: str, ngram_range: Tuple[int, int] = (1, 2)) -> List[str]:
    """Tokenize with n-grams for better phrase matching."""
    unigrams = tokenize(text)
    tokens = list(unigrams)

    if ngram_range[1] >= 2:
        for i in range(len(unigrams) - 1):
            tokens.append(f"{unigrams[i]}_{unigrams[i+1]}")

    if ngram_range[1] >= 3:
        for i in range(len(unigrams) - 2):
            tokens.append(f"{unigrams[i]}_{unigrams[i+1]}_{unigrams[i+2]}")

    return tokens


# ─── TF-IDF Vectorizer ──────────────────────────────────────────

class TFIDFVectorizer:
    """
    Pure Python + numpy TF-IDF vectorizer.
    Builds vocabulary from corpus and produces sparse/dense vectors.
    """

    def __init__(self, max_features: int = 512, use_ngrams: bool = True):
        self.max_features = max_features
        self.use_ngrams = use_ngrams
        self.vocabulary: Dict[str, int] = {}  # token -> index
        self.idf: Dict[str, float] = {}  # token -> IDF score
        self.doc_count: int = 0
        self._fitted = False

    def fit(self, documents: List[str]):
        """Build vocabulary and compute IDF from a corpus."""
        self.doc_count = len(documents)
        if self.doc_count == 0:
            self._fitted = True
            return self

        # Count document frequency for each token
        df: Dict[str, int] = Counter()
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1

        # Select top features by document frequency
        sorted_tokens = sorted(df.items(), key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tokens[: self.max_features]

        # Build vocabulary and IDF
        self.vocabulary = {}
        self.idf = {}
        for idx, (token, freq) in enumerate(top_tokens):
            self.vocabulary[token] = idx
            # Smooth IDF: log((N + 1) / (df + 1)) + 1
            self.idf[token] = math.log((self.doc_count + 1) / (freq + 1)) + 1.0

        self._fitted = True
        logger.debug(f"TF-IDF fitted: {len(self.vocabulary)} features from {self.doc_count} docs")
        return self

    def transform(self, text: str) -> np.ndarray:
        """Transform a single document to a TF-IDF vector."""
        if not self._fitted or not self.vocabulary:
            return np.zeros(max(self.max_features, 1), dtype=np.float32)

        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(len(self.vocabulary), dtype=np.float32)

        tf = Counter(tokens)
        vector = np.zeros(len(self.vocabulary), dtype=np.float32)

        for token, count in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf_score = count / len(tokens)
                vector[idx] = tf_score * self.idf.get(token, 0.0)

        # L2 normalize for cosine similarity via dot product
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector

    def transform_batch(self, texts: List[str]) -> np.ndarray:
        """Transform multiple documents to a matrix of TF-IDF vectors."""
        if not texts:
            dim = len(self.vocabulary) if self.vocabulary else self.max_features
            return np.zeros((0, dim), dtype=np.float32)
        vectors = [self.transform(text) for text in texts]
        return np.vstack(vectors)

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform_batch(documents)

    def partial_fit(self, new_documents: List[str]):
        """
        Incrementally update vocabulary with new documents.
        Recomputes IDF but preserves existing vocabulary indices.
        """
        self.doc_count += len(new_documents)

        df_update: Dict[str, int] = Counter()
        for doc in new_documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df_update[token] += 1

        # Add new tokens if space available
        for token, freq in df_update.items():
            if token not in self.vocabulary and len(self.vocabulary) < self.max_features:
                self.vocabulary[token] = len(self.vocabulary)

        # Recompute IDF for all known tokens
        for token in self.vocabulary:
            old_df = 0
            if token in self.idf:
                # Reverse-engineer old df from old IDF
                old_idf = self.idf[token] - 1.0
                old_n = self.doc_count - len(new_documents)
                if old_idf > 0:
                    old_df = max(1, round((old_n + 1) / math.exp(old_idf) - 1))
            new_df = old_df + df_update.get(token, 0)
            self.idf[token] = math.log((self.doc_count + 1) / (new_df + 1)) + 1.0

        self._fitted = True

    def _tokenize(self, text: str) -> List[str]:
        if self.use_ngrams:
            return tokenize_with_ngrams(text)
        return tokenize(text)

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: str):
        """Save vectorizer state to JSON."""
        state = {
            "max_features": self.max_features,
            "use_ngrams": self.use_ngrams,
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "doc_count": self.doc_count,
        }
        atomic_write_json(path, state)

    @classmethod
    def load(cls, path: str) -> "TFIDFVectorizer":
        """Load vectorizer state from JSON."""
        state = read_json(path)
        if state is None:
            return cls()
        v = cls(
            max_features=state.get("max_features", 512),
            use_ngrams=state.get("use_ngrams", True),
        )
        v.vocabulary = state.get("vocabulary", {})
        v.idf = state.get("idf", {})
        v.doc_count = state.get("doc_count", 0)
        v._fitted = bool(v.vocabulary)
        return v


# ─── BM25 Scorer ─────────────────────────────────────────────────

class BM25Scorer:
    """
    BM25 (Okapi BM25) ranking implementation.
    Provides probabilistic relevance scoring for text retrieval.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lens: List[int] = []
        self.doc_freqs: Dict[str, int] = {}  # token -> number of docs containing it
        self.doc_tokens: List[List[str]] = []  # tokenized documents
        self._fitted = False

    def fit(self, documents: List[str]):
        """Index documents for BM25 scoring."""
        self.doc_count = len(documents)
        self.doc_tokens = []
        self.doc_lens = []
        self.doc_freqs = Counter()

        total_len = 0
        for doc in documents:
            tokens = tokenize(doc)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        self.avg_doc_len = total_len / max(self.doc_count, 1)
        self._fitted = True
        return self

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a query against a specific document."""
        if not self._fitted or doc_idx >= len(self.doc_tokens):
            return 0.0

        query_tokens = tokenize(query)
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        doc_tf = Counter(doc_tokens)

        score = 0.0
        for qt in query_tokens:
            if qt not in self.doc_freqs:
                continue
            df = self.doc_freqs[qt]
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            tf = doc_tf.get(qt, 0)
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
            )
            score += idf * tf_norm

        return score

    def score_all(self, query: str) -> List[float]:
        """Compute BM25 scores for a query against all documents."""
        return [self.score(query, i) for i in range(self.doc_count)]

    def add_document(self, document: str):
        """Incrementally add a document."""
        tokens = tokenize(document)
        self.doc_tokens.append(tokens)
        self.doc_lens.append(len(tokens))
        self.doc_count += 1

        total_len = sum(self.doc_lens)
        self.avg_doc_len = total_len / self.doc_count

        for token in set(tokens):
            self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: str):
        state = {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            "doc_lens": self.doc_lens,
            "doc_freqs": self.doc_freqs,
            "doc_tokens": self.doc_tokens,
        }
        atomic_write_json(path, state)

    @classmethod
    def load(cls, path: str) -> "BM25Scorer":
        state = read_json(path)
        if state is None:
            return cls()
        s = cls(k1=state.get("k1", 1.2), b=state.get("b", 0.75))
        s.doc_count = state.get("doc_count", 0)
        s.avg_doc_len = state.get("avg_doc_len", 0.0)
        s.doc_lens = state.get("doc_lens", [])
        s.doc_freqs = state.get("doc_freqs", {})
        s.doc_tokens = state.get("doc_tokens", [])
        s._fitted = s.doc_count > 0
        return s
