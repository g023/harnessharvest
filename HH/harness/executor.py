"""
HarnessHarvester - Sandboxed Executor

Executes harness code in a sandboxed subprocess with:
- Path restriction to sandbox directory
- Configurable timeouts
- State checkpointing for resume capability
- Output capture (stdout, stderr)
- Resource monitoring
- Agentic runtime injection for LLM-powered harnesses
"""

import os
import sys
import json
import time
import signal
import subprocess
import threading
from typing import Any, Dict, List, Optional

from core.constants import (
    SANDBOX_ROOT, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED,
    STATUS_ACTIVE, STATUS_DRAFT, PROJECT_ROOT, PROJECTS_ROOT,
    DEFAULT_EXECUTION_TIMEOUT, DEFAULT_CHECKPOINT_INTERVAL,
)
from core.config import config
from core.storage import (
    atomic_write, atomic_write_json, read_json,
    ensure_dir, generate_id, read_text,
)
from core.logging_setup import get_logger
from core.utils import now_iso, is_safe_path, extract_imports
from harness.models import ExecutionLog, CheckpointState

logger = get_logger("executor")


# ─── Agentic Runtime Injection ──────────────────────────────────

AGENTIC_RUNTIME_PREAMBLE = '''
"""Injected Agentic Runtime for HarnessHarvester"""
import os
import sys

# Setup paths - Allow access to project root for imports
_PROJECT_ROOT = {project_root!r}
_WORKSPACE_PATH = {workspace_path!r}
_PROJECT_OUTPUT_PATH = {project_output_path!r}
_HARNESS_ID = {harness_id!r}
_RUN_ID = {run_id!r}
_TASK_DESCRIPTION = {task_description!r}
_LLM_CONFIG = {llm_config!r}

# Add project to path BEFORE any sandbox restrictions
sys.path.insert(0, _PROJECT_ROOT)

# Import the agentic runtime
from harness.runtime import AgenticHarness, WorkflowStage

# Import RAG stores for knowledge access
try:
    from rag.snippets import SnippetStore
    from rag.prompts import PromptStore
    from rag.errors import ErrorStore
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

# Make globals available to harness code
WORKSPACE = _WORKSPACE_PATH
PROJECT_OUTPUT = _PROJECT_OUTPUT_PATH
TASK = _TASK_DESCRIPTION
HARNESS_ID = _HARNESS_ID
RUN_ID = _RUN_ID
RAG_AVAILABLE = _RAG_AVAILABLE

# Create project output directory
os.makedirs(_PROJECT_OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(_PROJECT_OUTPUT_PATH, "deliverables"), exist_ok=True)

# RAG helper functions available to harnesses
def rag_search_snippets(query, top_k=5):
    """Search code snippets RAG store."""
    if not _RAG_AVAILABLE:
        return []
    try:
        store = SnippetStore()
        results = store.search(query, top_k=top_k)
        return [{{"title": r.entry.title, "content": r.entry.content, "score": r.final_score}} for r in results]
    except Exception:
        return []

def rag_search_errors(query, top_k=5):
    """Search error patterns RAG store."""
    if not _RAG_AVAILABLE:
        return []
    try:
        store = ErrorStore()
        results = store.search(query, top_k=top_k)
        return [{{"pattern": r.entry.content, "solution": r.entry.description, "score": r.final_score}} for r in results]
    except Exception:
        return []

def rag_search_prompts(query, top_k=3):
    """Search successful prompts RAG store."""
    if not _RAG_AVAILABLE:
        return []
    try:
        store = PromptStore()
        results = store.search(query, top_k=top_k)
        return [{{"prompt": r.entry.content, "score": r.final_score}} for r in results]
    except Exception:
        return []
'''


# ─── Sandbox Wrapper Code ───────────────────────────────────────

SANDBOX_WRAPPER = '''
# ── Sandbox Restrictions ─────────────────────────────────────
import builtins

SANDBOX_ROOT = {sandbox_root!r}
CHECKPOINT_FILE = {checkpoint_file!r}
_STEP_INDEX = 0

# ── Path Restriction ─────────────────────────────────────────
_original_open = builtins.open

def _safe_open(file, mode="r", *args, **kwargs):
    if isinstance(file, str):
        resolved = os.path.realpath(os.path.join(SANDBOX_ROOT, file) if not os.path.isabs(file) else file)
        sandbox_real = os.path.realpath(SANDBOX_ROOT)
        project_real = os.path.realpath(globals().get('_PROJECT_ROOT', '')) if '_PROJECT_ROOT' in globals() else ""
        project_output_real = os.path.realpath(globals().get('_PROJECT_OUTPUT_PATH', '')) if '_PROJECT_OUTPUT_PATH' in globals() else ""
        
        # Allow access to:
        # 1. Sandbox directory
        # 2. Project root (for imports)
        # 3. Project output directory (for deliverables)
        # 4. System libraries (for reading)
        safe_read_prefixes = ["/usr/lib", "/usr/local/lib", sys.prefix]
        if project_real:
            safe_read_prefixes.append(project_real)
        
        in_sandbox = resolved.startswith(sandbox_real + os.sep) or resolved == sandbox_real
        in_project = project_real and (resolved.startswith(project_real + os.sep) or resolved == project_real)
        in_project_output = project_output_real and (resolved.startswith(project_output_real + os.sep) or resolved == project_output_real)
        in_safe = mode.startswith("r") and any(resolved.startswith(p) for p in safe_read_prefixes)
        
        if not (in_sandbox or in_project or in_project_output or in_safe):
            raise PermissionError(f"Access denied: {{file}} is outside allowed paths")
    return _original_open(file, mode, *args, **kwargs)

builtins.open = _safe_open

# Restrict os.listdir
_original_listdir = os.listdir
def _safe_listdir(path="."):
    resolved = os.path.realpath(os.path.join(SANDBOX_ROOT, path) if not os.path.isabs(path) else path)
    sandbox_real = os.path.realpath(SANDBOX_ROOT)
    project_real = os.path.realpath(globals().get('_PROJECT_ROOT', '')) if '_PROJECT_ROOT' in globals() else ""
    project_output_real = os.path.realpath(globals().get('_PROJECT_OUTPUT_PATH', '')) if '_PROJECT_OUTPUT_PATH' in globals() else ""
    
    in_sandbox = resolved.startswith(sandbox_real) or resolved == sandbox_real
    in_project = project_real and resolved.startswith(project_real)
    in_project_output = project_output_real and resolved.startswith(project_output_real)
    
    if not (in_sandbox or in_project or in_project_output):
        raise PermissionError(f"Access denied: {{path}} is outside allowed paths")
    return _original_listdir(path)
os.listdir = _safe_listdir

# ── Checkpoint Support ───────────────────────────────────────
def checkpoint(state_dict, step_name=""):
    """Save execution state for resume capability."""
    global _STEP_INDEX
    _STEP_INDEX += 1
    cp = {{
        "step_index": _STEP_INDEX,
        "step_name": step_name,
        "state": state_dict,
        "timestamp": __import__("time").time(),
    }}
    with _original_open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f)

def load_checkpoint():
    """Load the last checkpoint if it exists."""
    try:
        with _original_open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# ── Change to sandbox directory ──────────────────────────────
os.makedirs(SANDBOX_ROOT, exist_ok=True)
os.chdir(SANDBOX_ROOT)

# ── Execute the harness ──────────────────────────────────────
'''


class HarnessExecutor:
    """
    Executes harness code in a sandboxed subprocess.
    Provides timeout control, output capture, and checkpoint/resume.
    Supports both simple scripts and agentic harnesses.
    """

    def __init__(self):
        self.sandbox_root = config.get_path("sandbox_root")
        self.timeout = config.get("sandbox.execution_timeout", DEFAULT_EXECUTION_TIMEOUT)
        self.checkpoint_interval = config.get(
            "harness.checkpoint_interval_seconds", DEFAULT_CHECKPOINT_INTERVAL
        )
        ensure_dir(self.sandbox_root)

    def execute(
        self,
        harness_id: str,
        code: str,
        version: str = "v1",
        branch: str = "main",
        timeout: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None,
        interactive_confirm: bool = True,
        agentic: bool = False,
        task_description: str = "",
    ) -> ExecutionLog:
        """
        Execute harness code in sandbox.

        Args:
            harness_id: The harness identifier
            code: Python code to execute
            version: Version string
            branch: Branch string
            timeout: Override default timeout
            resume_from_checkpoint: Path to checkpoint file to resume from
            interactive_confirm: Whether to prompt for dangerous operations
            agentic: If True, inject agentic runtime with LLM access
            task_description: Task for agentic harnesses
        """
        effective_timeout = timeout or self.timeout
        run_id = generate_id("run")
        now = now_iso()
        start_time = time.time()

        # Create execution sandbox directory
        exec_dir = os.path.join(self.sandbox_root, harness_id, f"{run_id}")
        ensure_dir(exec_dir)

        # Create project output directory (final deliverables go here)
        project_output_path = os.path.join(PROJECTS_ROOT, harness_id, run_id)
        ensure_dir(project_output_path)
        ensure_dir(os.path.join(project_output_path, "deliverables"))

        # Checkpoint file path
        checkpoint_file = os.path.join(exec_dir, f"state_{run_id}.json")

        # Check for dangerous operations
        if interactive_confirm:
            dangers = self._detect_dangerous_ops(code)
            if dangers:
                logger.warning(f"Dangerous operations detected: {dangers}")

        # Build script based on mode
        if agentic:
            full_script = self._build_agentic_script(
                code=code,
                exec_dir=exec_dir,
                checkpoint_file=checkpoint_file,
                harness_id=harness_id,
                run_id=run_id,
                task_description=task_description,
                project_output_path=project_output_path,
            )
        else:
            full_script = self._build_simple_script(
                code=code,
                exec_dir=exec_dir,
                checkpoint_file=checkpoint_file,
                resume_from_checkpoint=resume_from_checkpoint,
            )

        # Write script to temp file
        script_path = os.path.join(exec_dir, "_run.py")
        atomic_write(script_path, full_script)

        # Execute
        log = ExecutionLog(
            harness_id=harness_id,
            version=version,
            branch=branch,
            run_id=run_id,
            status=STATUS_RUNNING,
            started_at=now,
            resumed_from=resume_from_checkpoint or "",
        )

        logger.info(f"Executing {'agentic' if agentic else 'simple'} harness {harness_id} (run {run_id})")

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=exec_dir,
                env=self._get_safe_env(),
            )

            log.exit_code = result.returncode
            log.stdout = result.stdout[-10000:] if result.stdout else ""  # Limit output size
            log.stderr = result.stderr[-10000:] if result.stderr else ""
            log.status = STATUS_COMPLETED if result.returncode == 0 else STATUS_FAILED

            if result.returncode != 0:
                log.error_message = self._extract_error(result.stderr)
                log.error_traceback = result.stderr[-5000:]
                logger.warning(f"Harness {harness_id} failed: {log.error_message[:200]}")

        except subprocess.TimeoutExpired:
            log.status = STATUS_FAILED
            log.error_message = f"Execution timed out after {effective_timeout}s"
            logger.warning(f"Harness {harness_id} timed out")

        except Exception as e:
            log.status = STATUS_FAILED
            log.error_message = str(e)
            logger.error(f"Execution error: {e}")

        # Record completion
        log.completed_at = now_iso()
        log.duration_seconds = time.time() - start_time

        # Load checkpoint/state if exists
        if os.path.exists(checkpoint_file):
            cp_data = read_json(checkpoint_file)
            if cp_data:
                log.last_checkpoint = checkpoint_file
                log.checkpoints.append(cp_data)
        
        # For agentic harnesses, also check state file
        state_path = os.path.join(exec_dir, "_state.json")
        if agentic and os.path.exists(state_path):
            state_data = read_json(state_path)
            if state_data:
                log.deliverables = state_data.get("deliverables", [])
                log.llm_calls = state_data.get("llm_calls", 0)

        # Save execution log
        self._save_execution_log(harness_id, version, branch, log)

        return log

    def _build_simple_script(
        self,
        code: str,
        exec_dir: str,
        checkpoint_file: str,
        resume_from_checkpoint: Optional[str] = None,
    ) -> str:
        """Build a simple sandboxed script (legacy mode)."""
        wrapper = SANDBOX_WRAPPER.format(
            sandbox_root=exec_dir,
            checkpoint_file=checkpoint_file,
        )

        # Handle resume
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            wrapper += f"\n_RESUME_STATE = load_checkpoint()\n"
        else:
            wrapper += f"\n_RESUME_STATE = None\n"

        return wrapper + "\n# ── Harness Code ──\n" + code

    def _build_agentic_script(
        self,
        code: str,
        exec_dir: str,
        checkpoint_file: str,
        harness_id: str,
        run_id: str,
        task_description: str,
        project_output_path: str,
    ) -> str:
        """Build an agentic script with runtime injection."""
        # LLM configuration
        llm_config = {
            "model": config.get_model("generator"),
            "options": config.get_options_for_model(config.get_model("generator"), "default"),
        }

        # Agentic runtime preamble - BEFORE sandbox restrictions
        preamble = AGENTIC_RUNTIME_PREAMBLE.format(
            project_root=PROJECT_ROOT,
            workspace_path=exec_dir,
            project_output_path=project_output_path,
            harness_id=harness_id,
            run_id=run_id,
            task_description=task_description,
            llm_config=llm_config,
        )

        # Sandbox wrapper - AFTER runtime imports are established
        wrapper = SANDBOX_WRAPPER.format(
            sandbox_root=exec_dir,
            checkpoint_file=checkpoint_file,
        )

        return preamble + "\n" + wrapper + "\n# ── Agentic Harness Code ──\n" + code

    def get_latest_checkpoint(self, harness_id: str) -> Optional[str]:
        """Find the most recent checkpoint file for a harness."""
        sandbox_dir = os.path.join(self.sandbox_root, harness_id)
        if not os.path.isdir(sandbox_dir):
            return None

        latest_path = None
        latest_time = 0

        for root, dirs, files in os.walk(sandbox_dir):
            for f in files:
                if f.startswith("state_") and f.endswith(".json"):
                    path = os.path.join(root, f)
                    mtime = os.path.getmtime(path)
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_path = path

        return latest_path

    # ── Private Helpers ──────────────────────────────────────

    def _get_safe_env(self) -> Dict[str, str]:
        """Create a restricted environment for subprocess execution."""
        env = os.environ.copy()
        # Remove potentially dangerous env vars
        for key in ["PYTHONSTARTUP", "PYTHONPATH"]:
            env.pop(key, None)
        return env

    def _detect_dangerous_ops(self, code: str) -> List[str]:
        """Detect potentially dangerous operations in code."""
        dangers = []
        blocked = config.get("sandbox.blocked_patterns", [])

        for pattern in blocked:
            if pattern in code:
                dangers.append(pattern)

        # Check for network access
        if "socket" in code or "urllib" in code or "requests" in code:
            dangers.append("network_access")

        # Check for system commands
        if "subprocess" in code or "os.system" in code:
            dangers.append("system_commands")

        return dangers

    @staticmethod
    def _extract_error(stderr: str) -> str:
        """Extract the primary error message from stderr."""
        if not stderr:
            return ""
        lines = stderr.strip().split("\n")
        # Find the last line that looks like an error
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("Traceback") and not stripped.startswith("File"):
                return stripped
        return lines[-1] if lines else ""

    def _save_execution_log(
        self,
        harness_id: str,
        version: str,
        branch: str,
        log: ExecutionLog,
    ):
        """Save execution log to the harness version directory."""
        from core.constants import HARNESSES_DIR
        harness_dir = os.path.join(
            config.get_path("harnesses_dir"), harness_id
        )

        # Save to version dir
        version_dir = os.path.join(harness_dir, "versions", f"{version}_{branch}")
        if not os.path.isdir(version_dir):
            version_dir = os.path.join(harness_dir, "versions", version)
        ensure_dir(version_dir)

        log_path = os.path.join(version_dir, f"execution_{log.run_id}.json")
        atomic_write_json(log_path, log.to_dict())

        # Update harness metadata
        meta_path = os.path.join(harness_dir, "metadata.json")
        meta = read_json(meta_path, {})
        meta["total_runs"] = meta.get("total_runs", 0) + 1
        meta["updated_at"] = now_iso()
        if log.status == STATUS_COMPLETED:
            meta["status"] = STATUS_ACTIVE
        elif log.status == STATUS_FAILED and meta.get("status") == STATUS_DRAFT:
            meta["status"] = STATUS_FAILED
        atomic_write_json(meta_path, meta)
