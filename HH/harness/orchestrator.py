"""
HarnessHarvester - Orchestrator Base Classes

Enforces mandatory cognitive workflow stages for agentic harnesses.
All orchestrators must implement: reflect, brainstorm, plan, create_task_list, track_progress.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from core.constants import (
    STAGE_REFLECT, STAGE_BRAINSTORM, STAGE_PLAN,
    STAGE_TASK_LIST, STAGE_TRACK, STAGE_EXECUTE,
    ORCHESTRATOR_STAGES,
)
from core.logging_setup import get_logger
from core.utils import llm_call, now_iso

logger = get_logger("orchestrator")


class StageError(Exception):
    """Raised when a stage is accessed out of order or prerequisites not met."""
    pass


class ToolPermission(Enum):
    """Tool permissions per orchestrator stage."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    LLM_CALL = "llm_call"
    FILE_IO = "file_io"
    EXECUTE = "execute"
    RAG_QUERY = "rag_query"
    RAG_WRITE = "rag_write"


# Stage → allowed tool permissions
STAGE_PERMISSIONS: Dict[str, Set[ToolPermission]] = {
    STAGE_REFLECT: {ToolPermission.READ_ONLY, ToolPermission.LLM_CALL, ToolPermission.RAG_QUERY},
    STAGE_BRAINSTORM: {ToolPermission.READ_ONLY, ToolPermission.LLM_CALL, ToolPermission.RAG_QUERY},
    STAGE_PLAN: {ToolPermission.READ_ONLY, ToolPermission.LLM_CALL, ToolPermission.RAG_QUERY},
    STAGE_TASK_LIST: {ToolPermission.READ_ONLY, ToolPermission.LLM_CALL, ToolPermission.READ_WRITE},
    STAGE_TRACK: {ToolPermission.READ_ONLY, ToolPermission.READ_WRITE, ToolPermission.LLM_CALL},
    STAGE_EXECUTE: {
        ToolPermission.READ_ONLY, ToolPermission.READ_WRITE,
        ToolPermission.LLM_CALL, ToolPermission.FILE_IO,
        ToolPermission.EXECUTE, ToolPermission.RAG_QUERY,
        ToolPermission.RAG_WRITE,
    },
}


@dataclass
class StageOutput:
    """Captured output from a cognitive stage."""
    stage: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    model_used: str = ""
    timestamp: str = ""


@dataclass
class TaskItem:
    """A single task in the hierarchical task list."""
    id: str
    title: str
    description: str = ""
    status: str = "pending"  # pending, in_progress, completed, failed, skipped
    priority: int = 0  # 0 = highest
    parent_id: str = ""
    subtasks: List[str] = field(default_factory=list)
    success_criteria: str = ""
    result: str = ""


class BaseOrchestrator(ABC):
    """
    Abstract base class for agentic orchestrators.

    Enforces the mandatory cognitive workflow:
    1. reflect() - Analyze the task
    2. brainstorm() - Generate approaches
    3. plan() - Formulate concrete plan
    4. create_task_list() - Break down into tasks
    5. track_progress() - Monitor execution
    6. execute() - Perform the work

    Stages must be completed in order. Each stage has restricted
    tool permissions to enforce structured thinking.
    """

    def __init__(self, task_description: str, model: str = None):
        self.task_description = task_description
        self.model = model
        self._current_stage_idx = -1
        self._stage_outputs: Dict[str, StageOutput] = {}
        self._tasks: Dict[str, TaskItem] = {}
        self._completed_stages: Set[str] = set()
        self._start_time = time.time()

    # ── Stage Properties ─────────────────────────────────────

    @property
    def current_stage(self) -> Optional[str]:
        if 0 <= self._current_stage_idx < len(ORCHESTRATOR_STAGES):
            return ORCHESTRATOR_STAGES[self._current_stage_idx]
        return None

    @property
    def stage_outputs(self) -> Dict[str, StageOutput]:
        return dict(self._stage_outputs)

    @property
    def tasks(self) -> Dict[str, TaskItem]:
        return dict(self._tasks)

    @property
    def progress(self) -> float:
        if not self._tasks:
            return 0.0
        completed = sum(1 for t in self._tasks.values() if t.status == "completed")
        return completed / len(self._tasks)

    # ── Stage Enforcement ────────────────────────────────────

    def _advance_to_stage(self, stage: str):
        """Advance to a specific stage, validating prerequisites."""
        target_idx = ORCHESTRATOR_STAGES.index(stage)

        # Check all prior stages are completed
        for i in range(target_idx):
            prior_stage = ORCHESTRATOR_STAGES[i]
            if prior_stage not in self._completed_stages:
                raise StageError(
                    f"Cannot enter '{stage}': prerequisite '{prior_stage}' not completed"
                )

        self._current_stage_idx = target_idx

    def _complete_stage(self, stage: str, output: StageOutput):
        """Mark a stage as completed and record its output."""
        self._stage_outputs[stage] = output
        self._completed_stages.add(stage)
        logger.info(f"Stage '{stage}' completed in {output.duration_seconds:.1f}s")

    def check_permission(self, permission: ToolPermission) -> bool:
        """Check if a tool permission is allowed in the current stage."""
        if self.current_stage is None:
            return False
        allowed = STAGE_PERMISSIONS.get(self.current_stage, set())
        return permission in allowed

    def require_permission(self, permission: ToolPermission):
        """Raise StageError if permission not allowed in current stage."""
        if not self.check_permission(permission):
            raise StageError(
                f"Permission '{permission.value}' not allowed in stage '{self.current_stage}'"
            )

    # ── Mandatory Stage Implementations ──────────────────────

    @abstractmethod
    def reflect(self) -> StageOutput:
        """
        Stage 1: Reflection
        Analyze the task, identify concepts, constraints, expected outcomes.
        """
        pass

    @abstractmethod
    def brainstorm(self) -> StageOutput:
        """
        Stage 2: Brainstorming & Internal Questioning
        Generate multiple approaches, question feasibility.
        """
        pass

    @abstractmethod
    def plan(self) -> StageOutput:
        """
        Stage 3: Planning
        Formulate concrete plan with phases/milestones.
        """
        pass

    @abstractmethod
    def create_task_list(self) -> StageOutput:
        """
        Stage 4: Task List Generation
        Hierarchical breakdown with success criteria.
        """
        pass

    @abstractmethod
    def track_progress(self) -> StageOutput:
        """
        Stage 5: Progress Tracking
        Track status of tasks/subtasks, update plan dynamically.
        """
        pass

    @abstractmethod
    def execute(self) -> StageOutput:
        """
        Stage 6: Execution
        Perform the actual work.
        """
        pass

    # ── Run Pipeline ─────────────────────────────────────────

    def run(self) -> Dict[str, StageOutput]:
        """
        Run the full cognitive workflow pipeline.
        Returns dict of stage -> output.
        """
        stages = [
            (STAGE_REFLECT, self.reflect),
            (STAGE_BRAINSTORM, self.brainstorm),
            (STAGE_PLAN, self.plan),
            (STAGE_TASK_LIST, self.create_task_list),
            (STAGE_TRACK, self.track_progress),
            (STAGE_EXECUTE, self.execute),
        ]

        for stage_name, stage_fn in stages:
            self._advance_to_stage(stage_name)
            start = time.time()
            try:
                output = stage_fn()
                output.duration_seconds = time.time() - start
                output.timestamp = now_iso()
                self._complete_stage(stage_name, output)
            except Exception as e:
                logger.error(f"Stage '{stage_name}' failed: {e}")
                self._stage_outputs[stage_name] = StageOutput(
                    stage=stage_name,
                    content=f"FAILED: {e}",
                    duration_seconds=time.time() - start,
                    timestamp=now_iso(),
                )
                raise

        return self._stage_outputs

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing."""
        return {
            "task_description": self.task_description,
            "current_stage": self.current_stage,
            "completed_stages": list(self._completed_stages),
            "stage_outputs": {
                k: {
                    "stage": v.stage,
                    "content": v.content,
                    "metadata": v.metadata,
                    "duration_seconds": v.duration_seconds,
                }
                for k, v in self._stage_outputs.items()
            },
            "tasks": {
                k: {
                    "id": v.id, "title": v.title,
                    "status": v.status, "priority": v.priority,
                }
                for k, v in self._tasks.items()
            },
            "progress": self.progress,
        }


class LLMOrchestrator(BaseOrchestrator):
    """
    Concrete orchestrator that uses LLM calls for each cognitive stage.
    Suitable for standard harness execution.
    """

    def __init__(self, task_description: str, model: str = None):
        super().__init__(task_description, model)
        from core.config import config
        self.model = model or config.get_model("generator")

    def reflect(self) -> StageOutput:
        self._advance_to_stage(STAGE_REFLECT)
        result = llm_call(
            messages=[{
                "role": "system",
                "content": "You are a task analyst. Analyze the given task and identify: 1) Key concepts, 2) Constraints, 3) Expected outcomes, 4) Potential challenges. Be thorough and structured."
            }, {
                "role": "user",
                "content": f"Analyze this task:\n\n{self.task_description}"
            }],
            model=self.model,
        )
        return StageOutput(
            stage=STAGE_REFLECT,
            content=result.get("content", ""),
            model_used=self.model,
            metadata={"reasoning": result.get("reasoning", "")},
        )

    def brainstorm(self) -> StageOutput:
        self._advance_to_stage(STAGE_BRAINSTORM)
        reflection = self._stage_outputs.get(STAGE_REFLECT, StageOutput(stage="", content=""))
        result = llm_call(
            messages=[{
                "role": "system",
                "content": "You are a creative problem solver. Given a task analysis, generate multiple approaches. For each approach: describe it, assess feasibility, and note trade-offs."
            }, {
                "role": "user",
                "content": f"Task: {self.task_description}\n\nAnalysis:\n{reflection.content}\n\nGenerate 3+ approaches."
            }],
            model=self.model,
        )
        return StageOutput(
            stage=STAGE_BRAINSTORM,
            content=result.get("content", ""),
            model_used=self.model,
        )

    def plan(self) -> StageOutput:
        self._advance_to_stage(STAGE_PLAN)
        brainstorm = self._stage_outputs.get(STAGE_BRAINSTORM, StageOutput(stage="", content=""))
        result = llm_call(
            messages=[{
                "role": "system",
                "content": "You are a project planner. Given brainstormed approaches, select the best one and create a concrete plan with phases, milestones, and success criteria."
            }, {
                "role": "user",
                "content": f"Task: {self.task_description}\n\nApproaches:\n{brainstorm.content}\n\nCreate a detailed plan."
            }],
            model=self.model,
        )
        return StageOutput(
            stage=STAGE_PLAN,
            content=result.get("content", ""),
            model_used=self.model,
        )

    def create_task_list(self) -> StageOutput:
        self._advance_to_stage(STAGE_TASK_LIST)
        plan = self._stage_outputs.get(STAGE_PLAN, StageOutput(stage="", content=""))
        result = llm_call(
            messages=[{
                "role": "system",
                "content": "You are a task breakdown specialist. Convert a plan into a hierarchical task list. Each task must have: id, title, description, success_criteria. Output as JSON array."
            }, {
                "role": "user",
                "content": f"Plan:\n{plan.content}\n\nCreate a JSON task list."
            }],
            model=self.model,
        )
        content = result.get("content", "")
        from core.utils import extract_json_from_text
        tasks_data = extract_json_from_text(content)
        if isinstance(tasks_data, list):
            for i, td in enumerate(tasks_data):
                task = TaskItem(
                    id=td.get("id", f"task_{i}"),
                    title=td.get("title", f"Task {i}"),
                    description=td.get("description", ""),
                    success_criteria=td.get("success_criteria", ""),
                    priority=i,
                )
                self._tasks[task.id] = task

        return StageOutput(
            stage=STAGE_TASK_LIST,
            content=content,
            model_used=self.model,
            metadata={"task_count": len(self._tasks)},
        )

    def track_progress(self) -> StageOutput:
        self._advance_to_stage(STAGE_TRACK)
        summary = f"Tasks: {len(self._tasks)}, Progress: {self.progress*100:.0f}%\n"
        for tid, task in self._tasks.items():
            summary += f"  [{task.status}] {task.title}\n"
        return StageOutput(
            stage=STAGE_TRACK,
            content=summary,
            metadata={"progress": self.progress, "task_count": len(self._tasks)},
        )

    def execute(self) -> StageOutput:
        self._advance_to_stage(STAGE_EXECUTE)
        # Default execution: generate code based on the plan
        task_list = self._stage_outputs.get(STAGE_TASK_LIST, StageOutput(stage="", content=""))
        plan = self._stage_outputs.get(STAGE_PLAN, StageOutput(stage="", content=""))

        result = llm_call(
            messages=[{
                "role": "system",
                "content": "You are an expert Python developer. Given a task and plan, write complete, working Python code. Include proper error handling and documentation."
            }, {
                "role": "user",
                "content": f"Task: {self.task_description}\n\nPlan:\n{plan.content}\n\nTasks:\n{task_list.content}\n\nWrite the complete Python code."
            }],
            model=self.model,
        )

        # Mark all tasks as completed
        for task in self._tasks.values():
            task.status = "completed"

        return StageOutput(
            stage=STAGE_EXECUTE,
            content=result.get("content", ""),
            model_used=self.model,
        )


def detect_agentic_harness(code: str) -> bool:
    """
    Detect if harness code defines an agentic pattern
    (orchestrators, agents, tools, etc.).
    """
    agentic_indicators = [
        r"BaseOrchestrator",
        r"LLMOrchestrator",
        r"class\s+\w*Orchestrator",
        r"class\s+\w*Agent",
        r"def reflect\(",
        r"def brainstorm\(",
        r"def plan\(",
        r"def create_task_list\(",
        r"@tool",
        r"tool_registry",
    ]
    import re
    matches = sum(1 for pattern in agentic_indicators if re.search(pattern, code))
    return matches >= 2
