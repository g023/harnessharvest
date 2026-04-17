"""
HarnessHarvester - Agentic Runtime

This module provides the AgenticHarness base class that all generated harnesses
extend. It gives harnesses the ability to:
- Make LLM calls via _new_ollama.py
- Follow mandatory workflow stages
- Create deliverables (files, projects)
- Self-repair on errors
- Report progress
- Checkpoint and resume

This runtime is injected into the sandbox before harness execution.
"""

import os
import sys
import json
import time
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

# ─── Workflow Stages ────────────────────────────────────────────

class WorkflowStage(Enum):
    """Mandatory stages every agentic harness must complete."""
    INIT = "init"
    RESEARCH = "research"
    ANALYZE = "analyze"
    PLAN = "plan"
    DEVELOP = "develop"
    TEST = "test"
    REPAIR = "repair"
    FINALIZE = "finalize"
    COMPLETE = "complete"

STAGE_ORDER = [
    WorkflowStage.INIT,
    WorkflowStage.RESEARCH,
    WorkflowStage.ANALYZE,
    WorkflowStage.PLAN,
    WorkflowStage.DEVELOP,
    WorkflowStage.TEST,
    WorkflowStage.REPAIR,
    WorkflowStage.FINALIZE,
    WorkflowStage.COMPLETE,
]


# ─── Data Models ────────────────────────────────────────────────

@dataclass
class Deliverable:
    """A file or artifact produced by the harness."""
    path: str
    type: str  # file, directory, document, code
    description: str = ""
    created_at: str = ""
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WorkflowState:
    """Persistent state of the harness workflow."""
    current_stage: str = WorkflowStage.INIT.value
    stage_history: List[Dict] = field(default_factory=list)
    deliverables: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    repair_attempts: int = 0
    llm_calls: int = 0
    started_at: str = ""
    updated_at: str = ""
    progress_percent: float = 0.0
    context: Dict = field(default_factory=dict)  # Stage outputs carried forward

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "WorkflowState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ─── Agentic Harness Base Class ─────────────────────────────────

class AgenticHarness:
    """
    Base class for all agentic harnesses.
    
    Subclasses must implement the workflow stage methods:
    - research() - Gather context and knowledge
    - analyze() - Break down the task
    - plan() - Create execution strategy
    - develop() - Generate deliverables
    - test() - Validate outputs
    - repair() - Fix any issues (called automatically if needed)
    - finalize() - Package and report completion
    
    Usage:
        class MyHarness(AgenticHarness):
            def research(self):
                info = self.llm("What do I need to know about...")
                self.context["research"] = info
            
            def develop(self):
                code = self.llm("Generate code for...")
                self.write_file("output.py", code)
    """

    # Configuration
    MAX_REPAIR_ATTEMPTS = 5
    MAX_LLM_CALLS = 100
    MAX_STAGE_RETRIES = 3

    def __init__(
        self,
        task_description: str,
        workspace_path: str,
        harness_id: str = "",
        run_id: str = "",
        config: Dict = None,
    ):
        self.task_description = task_description
        self.workspace_path = os.path.abspath(workspace_path)
        self.harness_id = harness_id
        self.run_id = run_id
        self.config = config or {}

        # Ensure workspace exists
        os.makedirs(self.workspace_path, exist_ok=True)

        # State management
        self._state_path = os.path.join(self.workspace_path, "_state.json")
        self._log_path = os.path.join(self.workspace_path, "_log.jsonl")
        self.state = self._load_state()
        
        # Context carried between stages
        self.context: Dict[str, Any] = self.state.context.copy()
        
        # LLM configuration
        self._llm_model = self.config.get("model", "qwen3.5:4b-q4_K_M")
        self._llm_options = self.config.get("options", {})

    # ─── Core Run Loop ──────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Execute the full workflow.
        
        Returns:
            Dict with status, deliverables, errors
        """
        self.log("Starting agentic workflow", level="INFO")
        self.state.started_at = datetime.now().isoformat()
        
        try:
            # Execute each stage in order
            stages = [
                (WorkflowStage.RESEARCH, self.research),
                (WorkflowStage.ANALYZE, self.analyze),
                (WorkflowStage.PLAN, self.plan),
                (WorkflowStage.DEVELOP, self.develop),
                (WorkflowStage.TEST, self.test),
                (WorkflowStage.FINALIZE, self.finalize),
            ]

            for stage, method in stages:
                self._execute_stage(stage, method)

            self._advance_stage(WorkflowStage.COMPLETE)
            self.state.progress_percent = 100.0
            self._save_state()

            return {
                "status": "success",
                "deliverables": self.state.deliverables,
                "errors": self.state.errors,
                "llm_calls": self.state.llm_calls,
            }

        except Exception as e:
            self.log(f"Workflow failed: {e}", level="ERROR")
            self.state.errors.append({
                "stage": self.state.current_stage,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            })
            self._save_state()
            
            return {
                "status": "failed",
                "error": str(e),
                "deliverables": self.state.deliverables,
                "errors": self.state.errors,
            }

    def _execute_stage(self, stage: WorkflowStage, method: Callable):
        """Execute a workflow stage with error handling and repair."""
        self._advance_stage(stage)
        
        for attempt in range(self.MAX_STAGE_RETRIES):
            try:
                self.log(f"Executing stage: {stage.value} (attempt {attempt + 1})")
                method()
                
                # Stage completed successfully
                self._record_stage_completion(stage)
                return
                
            except Exception as e:
                self.log(f"Stage {stage.value} failed: {e}", level="ERROR")
                
                if attempt < self.MAX_STAGE_RETRIES - 1:
                    # Try to repair
                    if self._attempt_repair(stage, e):
                        continue  # Retry stage after repair
                
                # Max retries exceeded
                raise RuntimeError(f"Stage {stage.value} failed after {attempt + 1} attempts: {e}")

    def _attempt_repair(self, stage: WorkflowStage, error: Exception) -> bool:
        """
        Attempt to repair a failed stage using LLM.
        
        Returns:
            True if repair was successful and stage should be retried
        """
        if self.state.repair_attempts >= self.MAX_REPAIR_ATTEMPTS:
            self.log("Max repair attempts reached", level="WARNING")
            return False

        self.state.repair_attempts += 1
        self._advance_stage(WorkflowStage.REPAIR)

        try:
            repair_prompt = f"""
A stage in my workflow failed. Help me fix it.

Stage: {stage.value}
Error: {error}
Traceback: {traceback.format_exc()}

Task I'm working on: {self.task_description}

Current context:
{json.dumps(self.context, indent=2, default=str)[:2000]}

What went wrong and how should I fix it? Provide a specific solution.
"""
            repair_result = self.llm(
                repair_prompt,
                system="You are a debugging expert. Analyze errors and provide specific fixes."
            )
            
            # Store repair suggestion for context
            self.context["last_repair"] = repair_result.get("content", "")
            
            self.log(f"Repair suggestion: {repair_result.get('content', '')[:200]}")
            
            # Record repair attempt
            self.state.errors.append({
                "stage": stage.value,
                "error": str(error),
                "repair_attempt": self.state.repair_attempts,
                "repair_suggestion": repair_result.get("content", "")[:500],
                "timestamp": datetime.now().isoformat(),
            })
            
            self._save_state()
            return True
            
        except Exception as repair_error:
            self.log(f"Repair failed: {repair_error}", level="ERROR")
            return False

    # ─── Stage Methods (Override in subclasses) ─────────────────

    def research(self):
        """
        Stage 1: Research - Gather context and knowledge.
        
        Use self.llm() to ask questions about the domain.
        Store findings in self.context["research"].
        """
        research_prompt = f"""
I need to complete this task: {self.task_description}

What key concepts, patterns, and best practices should I understand before starting?
Provide a structured analysis.
"""
        result = self.llm(research_prompt, system="You are a research analyst.")
        self.context["research"] = result.get("content", "")
        self.report_progress(15, "Research complete")

    def analyze(self):
        """
        Stage 2: Analyze - Break down the task into components.
        
        Identify sub-tasks, requirements, and constraints.
        Store in self.context["analysis"].
        """
        analysis_prompt = f"""
Task: {self.task_description}

Research findings:
{self.context.get("research", "N/A")[:2000]}

Break this task down into:
1. Core requirements
2. Sub-tasks needed
3. Expected deliverables
4. Potential challenges

Provide a structured analysis.
"""
        result = self.llm(analysis_prompt, system="You are a technical analyst.")
        self.context["analysis"] = result.get("content", "")
        self.report_progress(30, "Analysis complete")

    def plan(self):
        """
        Stage 3: Plan - Create execution strategy.
        
        Define steps, file structure, and approach.
        Store in self.context["plan"].
        """
        plan_prompt = f"""
Task: {self.task_description}

Analysis:
{self.context.get("analysis", "N/A")[:2000]}

Create a detailed execution plan:
1. Step-by-step implementation order
2. Files/deliverables to create
3. Dependencies between steps
4. Validation approach

Provide the plan as a structured document.
"""
        result = self.llm(plan_prompt, system="You are a project planner.")
        self.context["plan"] = result.get("content", "")
        self.report_progress(45, "Planning complete")

    def develop(self):
        """
        Stage 4: Develop - Generate deliverables.
        
        Create files, code, documents using self.write_file().
        This is the main work stage.
        """
        develop_prompt = f"""
Task: {self.task_description}

Plan:
{self.context.get("plan", "N/A")[:3000]}

Generate the complete solution. Include:
1. All necessary files with their complete content
2. Proper structure and organization
3. Comments and documentation

For each file, clearly mark it with:
=== FILE: filename.ext ===
<file contents>
=== END FILE ===
"""
        result = self.llm(
            develop_prompt, 
            system="You are an expert developer. Generate complete, working code."
        )
        
        # Parse and create files from response
        content = result.get("content", "")
        self._parse_and_create_files(content)
        self.report_progress(70, "Development complete")

    def test(self):
        """
        Stage 5: Test - Validate outputs.
        
        Check deliverables for correctness.
        Use LLM to evaluate semantic quality.
        """
        # Get list of created deliverables
        deliverables_info = "\n".join([
            f"- {d['path']} ({d['type']})"
            for d in self.state.deliverables
        ])

        test_prompt = f"""
Task: {self.task_description}

Deliverables created:
{deliverables_info}

Review these deliverables:
1. Do they fulfill the task requirements?
2. Are there any obvious issues or missing pieces?
3. What validation checks should be performed?

Provide a test report with pass/fail assessment.
"""
        result = self.llm(test_prompt, system="You are a QA engineer.")
        self.context["test_report"] = result.get("content", "")
        
        # Validate each deliverable
        for deliv in self.state.deliverables:
            self._validate_deliverable(deliv)
        
        self.report_progress(85, "Testing complete")

    def finalize(self):
        """
        Stage 6: Finalize - Package and report completion.
        
        Create summary, clean up, report what was accomplished.
        """
        finalize_prompt = f"""
Task completed: {self.task_description}

Deliverables:
{json.dumps(self.state.deliverables, indent=2)}

Test report:
{self.context.get("test_report", "N/A")[:1500]}

Create a completion report summarizing:
1. What was accomplished
2. Files created and their purposes
3. Any notes for the user
4. How to use the deliverables
"""
        result = self.llm(finalize_prompt, system="You are a technical writer.")
        
        # Write completion report
        report = result.get("content", "Task completed.")
        self.write_file("COMPLETION_REPORT.md", f"# Completion Report\n\n{report}")
        
        self.report_progress(100, "Finalization complete")

    # ─── LLM Interface ──────────────────────────────────────────

    def llm(
        self,
        prompt: str,
        system: str = None,
        model: str = None,
        options: Dict = None,
    ) -> Dict[str, Any]:
        """
        Make an LLM call.
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            model: Override model (optional)
            options: Override options (optional)
            
        Returns:
            Dict with "reasoning", "content", "usage", "time_taken"
        """
        if self.state.llm_calls >= self.MAX_LLM_CALLS:
            raise RuntimeError(f"Max LLM calls ({self.MAX_LLM_CALLS}) exceeded")

        self.state.llm_calls += 1

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        effective_model = model or self._llm_model
        effective_options = {**self._llm_options, **(options or {})}

        # Import and call LLM
        try:
            from _new_ollama import llm_nonstream
            result = llm_nonstream(
                conv=messages,
                thinking=True,
                options=effective_options,
                the_model=effective_model,
            )
            
            self.log(f"LLM call #{self.state.llm_calls}: {len(prompt)} chars -> {len(result.get('content', ''))} chars")
            return result
            
        except Exception as e:
            self.log(f"LLM call failed: {e}", level="ERROR")
            raise

    # ─── File Operations ────────────────────────────────────────

    def write_file(self, path: str, content: str, file_type: str = "file") -> str:
        """
        Create a file in the workspace.
        
        Args:
            path: Relative path within workspace
            content: File content (string). If a dict is passed (common mistake from llm()),
                     will auto-extract content["content"].
            file_type: Type of file (file, code, document, config)
            
        Returns:
            Absolute path to created file
        """
        # Handle common mistake: passing llm() result dict instead of content string
        if isinstance(content, dict):
            if "content" in content:
                self.log(f"Warning: write_file received dict, auto-extracting content key", level="WARNING")
                content = content["content"]
            else:
                raise ValueError(
                    f"write_file expects a string, got dict without 'content' key. "
                    f"If using llm(), remember: result['content'] is the text string."
                )
        
        if not isinstance(content, str):
            raise ValueError(
                f"write_file expects string content, got {type(content).__name__}. "
                f"If using llm(), remember: result['content'] is the text string."
            )
        
        # Sanitize path
        safe_path = self._sanitize_path(path)
        full_path = os.path.join(self.workspace_path, safe_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write file
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Track deliverable
        deliverable = Deliverable(
            path=safe_path,
            type=file_type,
            description=f"Created by {self.state.current_stage}",
            created_at=datetime.now().isoformat(),
        )
        self.state.deliverables.append(deliverable.to_dict())
        
        self.log(f"Created file: {safe_path} ({len(content)} bytes)")
        return full_path

    def read_file(self, path: str) -> str:
        """Read a file from the workspace."""
        safe_path = self._sanitize_path(path)
        full_path = os.path.join(self.workspace_path, safe_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {safe_path}")
        
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    def list_files(self, path: str = ".") -> List[str]:
        """List files in a directory within workspace."""
        safe_path = self._sanitize_path(path)
        full_path = os.path.join(self.workspace_path, safe_path)
        
        if not os.path.isdir(full_path):
            return []
        
        return os.listdir(full_path)

    def _sanitize_path(self, path: str) -> str:
        """Ensure path is safe and within workspace."""
        # If it's an absolute path, check if it's within workspace
        workspace_real = os.path.realpath(self.workspace_path)
        if os.path.isabs(path):
            path_real = os.path.realpath(path)
            # Check if path is within workspace (handle various edge cases)
            if path_real == workspace_real:
                # Path is the workspace itself
                path = "."
            elif path_real.startswith(workspace_real + os.sep):
                # It's within workspace, make it relative
                path = os.path.relpath(path_real, workspace_real)
            elif path_real.startswith(workspace_real):
                # Might be workspace without trailing sep - double check
                rel_path = path_real[len(workspace_real):]
                if rel_path.startswith(os.sep):
                    path = rel_path.lstrip(os.sep)
                else:
                    # Not actually within workspace, strip leading slashes only
                    path = path.lstrip("/\\")
            else:
                # Not within workspace - strip leading slashes for relative access
                path = path.lstrip("/\\")
        
        path = os.path.normpath(path)
        
        # Prevent directory traversal
        if ".." in path:
            path = path.replace("..", "")
        
        return path

    def _parse_and_create_files(self, content: str):
        """Parse LLM output and create files marked with FILE markers."""
        import re
        
        # Pattern: === FILE: path === ... === END FILE ===
        pattern = r"===\s*FILE:\s*([^\n=]+?)\s*===\s*\n(.*?)===\s*END\s*FILE\s*==="
        matches = re.findall(pattern, content, re.DOTALL)
        
        if matches:
            for filepath, filecontent in matches:
                filepath = filepath.strip()
                filecontent = filecontent.strip()
                self.write_file(filepath, filecontent, file_type=self._detect_file_type(filepath))
        else:
            # Fallback: try to extract code blocks
            code_pattern = r"```(\w+)?\n(.*?)```"
            code_matches = re.findall(code_pattern, content, re.DOTALL)
            
            for i, (lang, code) in enumerate(code_matches):
                ext = self._lang_to_ext(lang)
                filename = f"output_{i+1}.{ext}"
                self.write_file(filename, code.strip(), file_type="code")

    def _detect_file_type(self, path: str) -> str:
        """Detect file type from extension."""
        ext = os.path.splitext(path)[1].lower()
        type_map = {
            ".py": "code",
            ".js": "code",
            ".ts": "code",
            ".html": "code",
            ".css": "code",
            ".json": "config",
            ".yaml": "config",
            ".yml": "config",
            ".md": "document",
            ".txt": "document",
            ".rst": "document",
        }
        return type_map.get(ext, "file")

    def _lang_to_ext(self, lang: str) -> str:
        """Convert language name to file extension."""
        lang = (lang or "").lower()
        ext_map = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "html": "html",
            "css": "css",
            "json": "json",
            "yaml": "yaml",
            "markdown": "md",
            "md": "md",
            "bash": "sh",
            "shell": "sh",
        }
        return ext_map.get(lang, "txt")

    # ─── Validation ─────────────────────────────────────────────

    def _validate_deliverable(self, deliverable: Dict):
        """Validate a deliverable based on its type."""
        path = deliverable["path"]
        full_path = os.path.join(self.workspace_path, path)
        
        if not os.path.exists(full_path):
            deliverable["validation_errors"] = ["File does not exist"]
            return
        
        errors = []
        ext = os.path.splitext(path)[1].lower()
        
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Syntax validation by type
            if ext == ".py":
                errors.extend(self._validate_python(content))
            elif ext == ".json":
                errors.extend(self._validate_json(content))
            elif ext == ".html":
                errors.extend(self._validate_html(content))
        
        except Exception as e:
            errors.append(f"Read error: {e}")
        
        deliverable["validated"] = len(errors) == 0
        deliverable["validation_errors"] = errors

    def _validate_python(self, content: str) -> List[str]:
        """Validate Python syntax."""
        errors = []
        try:
            compile(content, "<string>", "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error line {e.lineno}: {e.msg}")
        return errors

    def _validate_json(self, content: str) -> List[str]:
        """Validate JSON syntax."""
        errors = []
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            errors.append(f"JSON error: {e.msg}")
        return errors

    def _validate_html(self, content: str) -> List[str]:
        """Basic HTML validation."""
        errors = []
        # Check for basic structure
        if "<html" not in content.lower() and "<!doctype" not in content.lower():
            errors.append("Missing HTML structure")
        return errors

    # ─── State Management ───────────────────────────────────────

    def _load_state(self) -> WorkflowState:
        """Load state from disk or create new."""
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    data = json.load(f)
                return WorkflowState.from_dict(data)
            except Exception:
                pass
        return WorkflowState()

    def _save_state(self):
        """Save state to disk."""
        self.state.updated_at = datetime.now().isoformat()
        self.state.context = self.context
        
        with open(self._state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2, default=str)

    def _advance_stage(self, stage: WorkflowStage):
        """Advance to a new workflow stage."""
        self.state.current_stage = stage.value
        self.state.stage_history.append({
            "stage": stage.value,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_state()

    def _record_stage_completion(self, stage: WorkflowStage):
        """Record successful completion of a stage."""
        for entry in reversed(self.state.stage_history):
            if entry["stage"] == stage.value:
                entry["completed"] = True
                entry["completed_at"] = datetime.now().isoformat()
                break
        self._save_state()

    # ─── Progress & Logging ─────────────────────────────────────

    def report_progress(self, percent: float, message: str = ""):
        """Report progress to the executor."""
        self.state.progress_percent = min(100.0, max(0.0, percent))
        self._save_state()
        self.log(f"Progress: {percent:.0f}% - {message}")

    def log(self, message: str, level: str = "INFO"):
        """Log a message to the execution log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "stage": self.state.current_stage,
            "message": message,
        }
        
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        # Also print for live monitoring
        print(f"[{level}] {self.state.current_stage}: {message}")

    def checkpoint(self, name: str = ""):
        """Save a checkpoint for resume capability."""
        checkpoint_dir = os.path.join(self.workspace_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_name = name or f"checkpoint_{self.state.current_stage}"
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.json")
        
        checkpoint_data = {
            "state": self.state.to_dict(),
            "context": self.context,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        self.log(f"Checkpoint saved: {checkpoint_name}")

    def load_checkpoint(self, name: str) -> bool:
        """Load a checkpoint to resume from."""
        checkpoint_path = os.path.join(self.workspace_path, "checkpoints", f"{name}.json")
        
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
            
            self.state = WorkflowState.from_dict(data["state"])
            self.context = data.get("context", {})
            self.log(f"Checkpoint loaded: {name}")
            return True
            
        except Exception as e:
            self.log(f"Failed to load checkpoint: {e}", level="ERROR")
            return False


# ─── Runtime Injection Code ─────────────────────────────────────

RUNTIME_PREAMBLE = '''
"""Injected Agentic Runtime for HarnessHarvester"""
import os
import sys

# Setup paths
_PROJECT_ROOT = {project_root!r}
_WORKSPACE_PATH = {workspace_path!r}
_HARNESS_ID = {harness_id!r}
_RUN_ID = {run_id!r}
_TASK_DESCRIPTION = {task_description!r}
_CONFIG = {config!r}

# Add project to path for imports
sys.path.insert(0, _PROJECT_ROOT)

# Import the runtime
from harness.runtime import AgenticHarness, WorkflowStage

# Make globals available
WORKSPACE = _WORKSPACE_PATH
TASK = _TASK_DESCRIPTION
'''

def generate_runtime_injection(
    project_root: str,
    workspace_path: str,
    harness_id: str,
    run_id: str,
    task_description: str,
    config: Dict = None,
) -> str:
    """
    Generate the runtime injection code to prepend to harness scripts.
    
    This code:
    1. Sets up paths
    2. Makes AgenticHarness available
    3. Provides workspace and task globals
    """
    return RUNTIME_PREAMBLE.format(
        project_root=project_root,
        workspace_path=workspace_path,
        harness_id=harness_id,
        run_id=run_id,
        task_description=task_description,
        config=config or {},
    )
