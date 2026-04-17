"""
HarnessHarvester - Atomic Storage Operations

Thread-safe file I/O with file locking and atomic writes.
All data persistence goes through this module.
"""

import json
import os
import sys
import time
import tempfile
import hashlib
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path

from core.constants import LOCK_TIMEOUT, LOCK_RETRY_INTERVAL

# ─── Platform-specific file locking ─────────────────────────────

if sys.platform == "win32":
    import msvcrt

    def _lock_file(f, timeout: float = LOCK_TIMEOUT):
        deadline = time.monotonic() + timeout
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Could not acquire lock on {f.name} within {timeout}s")
                time.sleep(LOCK_RETRY_INTERVAL)

    def _unlock_file(f):
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _lock_file(f, timeout: float = LOCK_TIMEOUT):
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Could not acquire lock on {f.name} within {timeout}s")
                time.sleep(LOCK_RETRY_INTERVAL)

    def _unlock_file(f):
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


class FileLock:
    """Context manager for advisory file locking."""

    def __init__(self, path: str, timeout: float = LOCK_TIMEOUT):
        self.path = path + ".lock"
        self.timeout = timeout
        self._file = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._file = open(self.path, "w")
        _lock_file(self._file, self.timeout)
        return self

    def __exit__(self, *args):
        if self._file:
            _unlock_file(self._file)
            self._file.close()
            try:
                os.remove(self.path)
            except OSError:
                pass


# ─── Atomic File Operations ─────────────────────────────────────

def atomic_write(path: str, content: str, encoding: str = "utf-8"):
    """Write content to a file atomically (temp file + rename)."""
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix=".tmp_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # Atomic on POSIX
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, data: Any, indent: int = 2):
    """Write JSON data atomically."""
    content = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    atomic_write(path, content)


def atomic_write_bytes(path: str, data: bytes):
    """Write binary data atomically."""
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix=".tmp_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── Safe Read Operations ───────────────────────────────────────

def read_json(path: str, default: Any = None) -> Any:
    """Read JSON file, returning default if not found or invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def read_text(path: str, default: str = "") -> str:
    """Read text file, returning default if not found."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, OSError):
        return default


# ─── Locked JSON Operations ─────────────────────────────────────

def locked_read_json(path: str, default: Any = None) -> Any:
    """Read JSON with advisory file lock."""
    with FileLock(path):
        return read_json(path, default)


def locked_write_json(path: str, data: Any, indent: int = 2):
    """Write JSON atomically with advisory file lock."""
    with FileLock(path):
        atomic_write_json(path, data, indent)


def locked_update_json(path: str, updater, default: Any = None):
    """
    Read-modify-write JSON atomically with lock.
    updater(data) -> modified_data
    """
    with FileLock(path):
        data = read_json(path, default)
        data = updater(data)
        atomic_write_json(path, data)
        return data


# ─── Index Operations ───────────────────────────────────────────

def append_to_jsonl(path: str, record: Dict):
    """Append a JSON record to a JSONL file (one JSON object per line)."""
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with FileLock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def read_jsonl(path: str) -> List[Dict]:
    """Read all records from a JSONL file."""
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return records


# ─── Directory Operations ───────────────────────────────────────

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def list_json_files(directory: str) -> List[str]:
    """List all .json files in a directory (non-recursive)."""
    try:
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".json") and os.path.isfile(os.path.join(directory, f))
        ]
    except FileNotFoundError:
        return []


# ─── Hashing ────────────────────────────────────────────────────

def content_hash(content: str) -> str:
    """Generate SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def file_hash(path: str) -> str:
    """Generate SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── ID Generation ──────────────────────────────────────────────

_id_counter = 0
_id_lock = threading.Lock()


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID using timestamp + counter + random."""
    global _id_counter
    import random
    with _id_lock:
        _id_counter += 1
        ts = int(time.time() * 1000)
        rand = random.randint(0, 0xFFFF)
        uid = f"{ts:x}{_id_counter:04x}{rand:04x}"
    if prefix:
        return f"{prefix}_{uid}"
    return uid
