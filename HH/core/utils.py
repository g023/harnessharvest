"""
HarnessHarvester - Shared Utilities

Code extraction, text processing, LLM call helpers, and common operations.
"""

import re
import os
import sys
import time
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

from core.constants import CHARS_PER_TOKEN, PROJECT_ROOT


# ─── LLM Interface ──────────────────────────────────────────────

# Import LLM functions from project root
sys.path.insert(0, PROJECT_ROOT)
import _new_ollama as ollama_api


def llm_call(
    messages: List[Dict[str, str]],
    model: str,
    options: Dict = None,
    stream: bool = False,
    thinking: bool = True,
) -> Dict[str, Any]:
    """
    Unified LLM call wrapper. Returns dict with reasoning, content, usage, time_taken.
    Handles option injection and metrics logging.
    """
    from core.config import config

    if options is None:
        options = config.get_options_for_model(model, "default")
    else:
        # Ensure num_ctx is set
        if "num_ctx" not in options:
            options["num_ctx"] = config.get_model_context(model)

    # Set model on the ollama API module
    old_model = ollama_api.G_MODEL
    ollama_api.G_MODEL = model

    try:
        if stream:
            result = ollama_api.llm_stream(
                conv=messages, thinking=thinking, options=options, the_model=model
            )
        else:
            result = ollama_api.llm_nonstream(
                conv=messages, thinking=thinking, options=options, the_model=model
            )
    finally:
        ollama_api.G_MODEL = old_model

    # Log metrics
    _log_llm_call(model, messages, result, stream)

    return result


def _log_llm_call(model: str, messages: List, result: Dict, stream: bool):
    """Log LLM call metrics to JSONL."""
    from core.config import config
    if not config.get("metrics.track_llm_calls", True):
        return
    try:
        from core.storage import append_to_jsonl
        from core.constants import METRICS_DIR
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "stream": stream,
            "input_messages": len(messages),
            "input_chars": sum(len(m.get("content", "")) for m in messages),
            "output_chars": len(result.get("content", "")),
            "reasoning_chars": len(result.get("reasoning", "") or ""),
            "time_taken": result.get("time_taken", 0),
            "usage": result.get("usage", {}),
        }
        append_to_jsonl(os.path.join(METRICS_DIR, "llm_calls.jsonl"), record)
    except Exception:
        pass  # Don't let metrics logging break the main flow


# ─── Code Extraction ────────────────────────────────────────────

def extract_code_blocks(text: str, language: str = "python") -> List[str]:
    """Extract code blocks from LLM output, trying multiple patterns."""
    blocks = []

    # Pattern 1: Fenced code blocks with language tag
    pattern1 = rf"```{language}\s*\n(.*?)```"
    blocks.extend(re.findall(pattern1, text, re.DOTALL))

    # Pattern 2: Fenced code blocks without language tag
    if not blocks:
        pattern2 = r"```\s*\n(.*?)```"
        blocks.extend(re.findall(pattern2, text, re.DOTALL))

    # Pattern 3: Indented code blocks (4+ spaces or tab)
    if not blocks:
        lines = text.split("\n")
        code_lines = []
        for line in lines:
            if line.startswith("    ") or line.startswith("\t") or line.strip() == "":
                code_lines.append(line)
            else:
                if code_lines and any(l.strip() for l in code_lines):
                    blocks.append("\n".join(code_lines))
                code_lines = []
        if code_lines and any(l.strip() for l in code_lines):
            blocks.append("\n".join(code_lines))

    # Pattern 4: If the entire response looks like code
    if not blocks and _looks_like_code(text):
        blocks.append(text)

    return [b.strip() for b in blocks if b.strip()]


def extract_primary_code(text: str, language: str = "python") -> str:
    """Extract the primary (longest) code block from LLM output."""
    blocks = extract_code_blocks(text, language)
    if not blocks:
        return ""
    return max(blocks, key=len)


def _looks_like_code(text: str) -> bool:
    """Heuristic to check if text is likely Python code."""
    indicators = ["def ", "class ", "import ", "from ", "if ", "for ", "while ", "return "]
    lines = text.strip().split("\n")
    if not lines:
        return False
    code_lines = sum(1 for line in lines if any(line.strip().startswith(i) for i in indicators))
    return code_lines / max(len(lines), 1) > 0.2


# ─── Text Processing ────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def summarize_for_context(text: str, max_tokens: int = 2000) -> str:
    """
    If text exceeds max_tokens, create a compressed version.
    For code: keep imports, class/function signatures, and docstrings.
    For prose: keep first and last paragraphs.
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    lines = text.split("\n")

    # Try code summarization
    if _looks_like_code(text):
        summary_lines = []
        in_body = False
        brace_depth = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                summary_lines.append(line)
            elif stripped.startswith(("def ", "class ")):
                summary_lines.append(line)
                in_body = True
                brace_depth = 0
            elif in_body and stripped.startswith(('"""', "'''")):
                summary_lines.append(line)
            elif in_body and stripped == "":
                in_body = False
                summary_lines.append("    ...")
                summary_lines.append("")
            elif not in_body and stripped.startswith("#"):
                summary_lines.append(line)
        result = "\n".join(summary_lines)
        if estimate_tokens(result) <= max_tokens:
            return result

    # Fallback: head + tail
    return truncate_to_tokens(text, max_tokens)


def normalize_content(content: str) -> str:
    """Normalize content for deduplication (strip whitespace, comments)."""
    lines = content.split("\n")
    normalized = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            normalized.append(stripped)
    return "\n".join(normalized)


# ─── JSON Helpers ────────────────────────────────────────────────

def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON with json-repair fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            repaired = repair_json(text, return_objects=True)
            return repaired
        except Exception:
            return default


def extract_json_from_text(text: str) -> Any:
    """Extract JSON object/array from text that may contain non-JSON content."""
    # Try the whole text first
    result = safe_json_loads(text)
    if result is not None:
        return result

    # Try to find JSON in code blocks
    patterns = [
        r"```json\s*\n(.*?)```",
        r"```\s*\n(\{.*?\})```",
        r"```\s*\n(\[.*?\])```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result = safe_json_loads(match.group(1))
            if result is not None:
                return result

    # Try to find bare JSON objects
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        result = safe_json_loads(match.group())
        if result is not None:
            return result

    return None


# ─── Timestamp Helpers ───────────────────────────────────────────

def now_iso() -> str:
    """Current time in ISO format."""
    return datetime.now().isoformat()


def now_epoch() -> float:
    """Current time as epoch seconds."""
    return time.time()


# ─── Import Validation ──────────────────────────────────────────

def extract_imports(code: str) -> List[str]:
    """Extract all imported module names from Python code."""
    imports = set()
    for match in re.finditer(r"^\s*import\s+([\w.]+)", code, re.MULTILINE):
        imports.add(match.group(1).split(".")[0])
    for match in re.finditer(r"^\s*from\s+([\w.]+)\s+import", code, re.MULTILINE):
        imports.add(match.group(1).split(".")[0])
    return sorted(imports)


def validate_imports(code: str, allowed: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Check if code only imports from allowed modules.
    Returns (is_valid, list_of_disallowed_imports).
    """
    if allowed is None:
        from core.config import config
        allowed = config.get("sandbox.allowed_imports", [])

    used = extract_imports(code)
    disallowed = [m for m in used if m not in allowed and m not in sys.stdlib_module_names]
    return len(disallowed) == 0, disallowed


# ─── Path Safety ─────────────────────────────────────────────────

def is_safe_path(path: str, sandbox_root: str) -> bool:
    """Check if a path is safely within the sandbox root."""
    real_sandbox = os.path.realpath(sandbox_root)
    real_path = os.path.realpath(os.path.join(real_sandbox, path))
    return real_path.startswith(real_sandbox + os.sep) or real_path == real_sandbox


# ─── Dataclass Serialization ────────────────────────────────────

def dataclass_to_dict(obj) -> Dict:
    """Convert a dataclass to a JSON-serializable dict."""
    return asdict(obj)


def check_user_md():
    """Check if USER.md has content (user requests)."""
    path = os.path.join(PROJECT_ROOT, "USER.md")
    try:
        with open(path, "r") as f:
            content = f.read().strip()
        if content:
            return content
    except FileNotFoundError:
        pass
    return None
