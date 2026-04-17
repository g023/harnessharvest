"""
HarnessHarvester - Configuration Management

Loads, validates, and provides access to config.json settings.
Thread-safe singleton pattern with live reload capability.
"""

import json
import os
import copy
import threading
from typing import Any, Dict, Optional

from core.constants import CONFIG_PATH, PROJECT_ROOT


class _ConfigManager:
    """Thread-safe configuration manager with lazy loading and live reload."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._data: Dict[str, Any] = {}
        self._rlock = threading.RLock()
        self._load()
        self._initialized = True

    # ── Loading ──────────────────────────────────────────────

    def _load(self):
        """Load configuration from disk."""
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = self._defaults()
            self.save()

    def reload(self):
        """Reload configuration from disk (thread-safe)."""
        with self._rlock:
            self._load()

    def save(self):
        """Persist current configuration to disk (atomic write)."""
        from core.storage import atomic_write_json
        with self._rlock:
            atomic_write_json(CONFIG_PATH, self._data)

    # ── Access ───────────────────────────────────────────────

    def get(self, dotpath: str, default: Any = None) -> Any:
        """
        Get a config value by dot-separated path.
        Example: config.get("ollama.models.judge")
        """
        with self._rlock:
            keys = dotpath.split(".")
            node = self._data
            for key in keys:
                if isinstance(node, dict) and key in node:
                    node = node[key]
                else:
                    return default
            return copy.deepcopy(node)

    def set(self, dotpath: str, value: Any, persist: bool = False):
        """Set a config value by dot-separated path."""
        with self._rlock:
            keys = dotpath.split(".")
            node = self._data
            for key in keys[:-1]:
                if key not in node or not isinstance(node[key], dict):
                    node[key] = {}
                node = node[key]
            node[keys[-1]] = value
            if persist:
                self.save()

    @property
    def data(self) -> Dict[str, Any]:
        """Return a deep copy of the full config."""
        with self._rlock:
            return copy.deepcopy(self._data)

    # ── Convenience ──────────────────────────────────────────

    def get_model(self, role: str) -> str:
        """Get model name for a given role (judge, reasoner, generator, etc.)."""
        return self.get(f"ollama.models.{role}", "qwen3.5:2b")

    def get_model_context(self, model: str) -> int:
        """Get context window size for a model."""
        return self.get(f"ollama.model_contexts.{model}", 32768)

    def get_options_profile(self, profile: str = "default") -> Dict[str, Any]:
        """Get an LLM options profile, falling back to default."""
        opts = self.get(f"ollama.options_profiles.{profile}")
        if opts is None:
            opts = self.get("ollama.options_profiles.default", {})
        return opts

    def get_options_for_model(self, model: str, profile: str = "default") -> Dict[str, Any]:
        """Get options profile with model-specific num_ctx injected."""
        opts = self.get_options_profile(profile)
        opts["num_ctx"] = self.get_model_context(model)
        return opts

    def get_path(self, name: str) -> str:
        """Get an absolute path from config, resolved relative to PROJECT_ROOT."""
        rel = self.get(f"paths.{name}", name)
        if os.path.isabs(rel):
            return rel
        return os.path.join(PROJECT_ROOT, rel)

    # ── Defaults ─────────────────────────────────────────────

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        return {
            "version": "1.0.0",
            "ollama": {
                "host": "http://localhost:11434",
                "timeout": 600,
                "models": {
                    "judge": "hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0",
                    "reasoner": "qwen3.5:2b",
                    "generator": "qwen3.5:4b-q4_K_M",
                },
                "model_contexts": {
                    "hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0": 40000,
                    "qwen3.5:2b": 240000,
                    "qwen3.5:4b-q4_K_M": 32768,
                },
                "options_profiles": {
                    "default": {
                        "num_predict": 16384,
                        "top_k": 40,
                        "top_p": 0.9,
                        "min_p": 0.05,
                        "temperature": 0.7,
                        "repeat_penalty": 1.1,
                        "num_batch": 512,
                        "num_thread": 4,
                    }
                },
            },
            "paths": {
                "db_root": "db",
                "sandbox_root": "sandbox",
                "harnesses_dir": "db/harnesses",
                "rag_dir": "db/rag",
                "metrics_dir": "db/metrics",
                "logs_dir": "logs",
            },
            "harness": {
                "max_repair_attempts": 5,
                "max_branches_per_version": 3,
                "auto_repair_threshold": 0.6,
                "checkpoint_interval_seconds": 60,
                "min_review_score": 0.6,
            },
            "sandbox": {
                "execution_timeout": 120,
                "blocked_patterns": ["os.system", "subprocess.call", "eval(", "exec("],
                "allowed_imports": ["os", "sys", "json", "math", "re", "random", "collections", "itertools", "functools", "datetime", "time", "pathlib", "io", "string", "textwrap", "typing", "dataclasses", "enum", "abc", "copy", "pprint", "argparse", "csv", "hashlib", "html", "http", "urllib"],
            },
            "autolearn": {
                "reflection_interval": 5,
                "min_score_to_keep": 0.4,
            },
            "autoimprove": {
                "max_improvement_iterations": 5,
                "min_improvement_threshold": 0.05,
                "preserve_functionality_threshold": 0.9,
            },
            "metrics": {
                "track_llm_calls": True,
            },
            "rag": {
                "embedding_dim": 512,
                "search_top_k": 10,
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
                "min_snippet_length": 10,
                "max_snippet_length": 10000,
                "ranking_weights": {
                    "tfidf_similarity": 0.30,
                    "bm25_score": 0.25,
                    "quality_score": 0.20,
                    "usage_frequency": 0.10,
                    "recency_boost": 0.10,
                    "tag_match": 0.05,
                },
            },
            "logging": {
                "level": "INFO",
                "max_file_size_mb": 10,
                "backup_count": 5,
            },
        }


# Module-level singleton
config = _ConfigManager()
