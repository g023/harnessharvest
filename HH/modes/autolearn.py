"""
HarnessHarvester - Autolearn Mode

Continuous autonomous discovery loop that:
1. Generates novel task ideas using LLM
2. Creates harnesses for each task
3. Executes and reviews them
4. Self-reflects to improve idea generation
5. Persists session state for resume
"""

import os
import signal
import time
import json
from typing import Any, Dict, List, Optional

from core.config import config
from core.constants import AUTOLEARN_DIR, MODEL_REASONER, MODEL_GENERATOR, STATUS_ACTIVE
from core.storage import (
    atomic_write_json, read_json, append_to_jsonl,
    ensure_dir, generate_id,
)
from core.logging_setup import get_logger
from core.utils import llm_call, extract_json_from_text, now_iso
from harness.generator import HarnessGenerator
from harness.executor import HarnessExecutor
from harness.reviewer import HarnessReviewer
from harness.repairer import HarnessRepairer, should_repair

logger = get_logger("autolearn")


IDEA_GENERATION_PROMPT = """You are a creative task generator for the HarnessHarvester system.
Generate a novel, useful Python programming task that can be expressed as a self-contained script.

Requirements:
- The task should be practical and demonstrate useful programming concepts
- It should be completable in a single Python script
- It should use only the Python standard library
- Avoid tasks that require external APIs, databases, or network access
- Be creative - think of tasks that developers actually need

{existing_tasks_context}

{reflection_context}

Respond with a JSON object:
{{
    "task_description": "Detailed description of the task",
    "complexity": "simple|medium|complex|agentic",
    "domain": "e.g., algorithms, data_processing, text_processing, file_operations, etc.",
    "tags": ["tag1", "tag2"],
    "expected_difficulty": 0.0 to 1.0
}}"""

REFLECTION_PROMPT = """Analyze the results of the last {n} autolearn iterations:

{iteration_summaries}

Based on these results:
1. What types of tasks succeeded? What failed?
2. What domains should be explored more?
3. What should be avoided?
4. How can idea generation be improved?

Respond with a brief strategy adjustment (2-3 sentences)."""


class AutolearnSession:
    """Persistent autolearn session with checkpoint/resume."""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or generate_id("als")
        self.session_dir = os.path.join(AUTOLEARN_DIR, "sessions", self.session_id)
        ensure_dir(self.session_dir)

        self.iterations: List[Dict] = []
        self.reflection_notes: List[str] = []
        self.total_iterations: int = 0
        self.successful_count: int = 0
        self.failed_count: int = 0
        self.started_at: str = now_iso()
        self.last_iteration_at: str = ""
        self._interrupted = False

        # Try to load existing session
        self._load()

    def save(self):
        """Save session state."""
        state = {
            "session_id": self.session_id,
            "total_iterations": self.total_iterations,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "started_at": self.started_at,
            "last_iteration_at": self.last_iteration_at,
            "iterations": self.iterations[-50:],  # Keep last 50 summaries
            "reflection_notes": self.reflection_notes[-10:],
        }
        atomic_write_json(
            os.path.join(self.session_dir, "session.json"), state
        )

    def _load(self):
        """Load session state if exists."""
        state = read_json(os.path.join(self.session_dir, "session.json"))
        if state:
            self.total_iterations = state.get("total_iterations", 0)
            self.successful_count = state.get("successful_count", 0)
            self.failed_count = state.get("failed_count", 0)
            self.started_at = state.get("started_at", self.started_at)
            self.iterations = state.get("iterations", [])
            self.reflection_notes = state.get("reflection_notes", [])
            logger.info(f"Resumed session {self.session_id} at iteration {self.total_iterations}")


class AutolearnMode:
    """
    Continuous autonomous discovery loop.
    Generates tasks, creates harnesses, executes and reviews them,
    and learns from the results.
    """

    def __init__(self):
        self.generator = HarnessGenerator()
        self.executor = HarnessExecutor()
        self.reviewer = HarnessReviewer()
        self.repairer = HarnessRepairer()

        self.idea_model = config.get_model("reasoner")  # 240k context, strong reasoning
        self.reflection_interval = config.get("autolearn.reflection_interval", 3)

    def run(
        self,
        max_iterations: int = 0,
        max_duration_seconds: int = 0,
        session_id: str = None,
    ):
        """
        Run the autolearn loop.

        Args:
            max_iterations: Stop after N iterations (0 = infinite)
            max_duration_seconds: Stop after N seconds (0 = infinite)
            session_id: Resume a previous session
        """
        session = AutolearnSession(session_id)
        start_time = time.time()

        # Setup signal handler for clean shutdown
        def signal_handler(signum, frame):
            logger.info("Interrupt received, finishing current iteration...")
            session._interrupted = True

        old_handler = signal.signal(signal.SIGINT, signal_handler)

        logger.info(f"Autolearn started (session: {session.session_id})")
        print(f"Autolearn session: {session.session_id}")
        print("Press Ctrl+C to stop gracefully...")

        try:
            while not session._interrupted:
                # Check limits
                if max_iterations > 0 and session.total_iterations >= max_iterations:
                    logger.info(f"Max iterations ({max_iterations}) reached")
                    break
                if max_duration_seconds > 0:
                    elapsed = time.time() - start_time
                    if elapsed >= max_duration_seconds:
                        logger.info(f"Max duration ({max_duration_seconds}s) reached")
                        break

                # Run one iteration
                iteration_result = self._run_iteration(session)
                session.iterations.append(iteration_result)
                session.total_iterations += 1
                session.last_iteration_at = now_iso()

                if iteration_result.get("success"):
                    session.successful_count += 1
                else:
                    session.failed_count += 1

                # Self-reflection every N iterations
                if session.total_iterations % self.reflection_interval == 0:
                    self._self_reflect(session)

                # Save session checkpoint
                session.save()

                # Log progress
                logger.info(
                    f"Iteration {session.total_iterations}: "
                    f"{'SUCCESS' if iteration_result.get('success') else 'FAILED'} "
                    f"(score: {iteration_result.get('score', 0):.2f})"
                )

        except Exception as e:
            logger.error(f"Autolearn error: {e}")
        finally:
            signal.signal(signal.SIGINT, old_handler)

            # Generate summary
            summary = self._generate_summary(session)
            session.save()

            print(f"\n{'='*60}")
            print(f"Autolearn Session Summary: {session.session_id}")
            print(f"Total iterations: {session.total_iterations}")
            print(f"Successful: {session.successful_count}")
            print(f"Failed: {session.failed_count}")
            print(f"Duration: {time.time() - start_time:.0f}s")
            print(f"{'='*60}")

            return summary

    def _run_iteration(self, session: AutolearnSession) -> Dict:
        """Run a single autolearn iteration."""
        result = {
            "iteration": session.total_iterations + 1,
            "timestamp": now_iso(),
            "success": False,
            "score": 0.0,
            "harness_id": "",
            "task": "",
            "error": "",
        }

        try:
            # 1. Generate idea
            idea = self._generate_idea(session)
            result["task"] = idea.get("task_description", "")
            result["domain"] = idea.get("domain", "")

            if not result["task"]:
                result["error"] = "Failed to generate task idea"
                return result

            logger.info(f"Generated idea: {result['task'][:100]}...")

            # 2. Create harness
            harness_id, metadata = self.generator.generate(
                task_description=result["task"],
                tags=idea.get("tags", []) + ["autolearn"],
                source="autolearn",
            )
            result["harness_id"] = harness_id

            # 3. Execute
            code = self.generator.get_harness_code(harness_id)
            exec_log = self.executor.execute(
                harness_id=harness_id,
                code=code,
                timeout=120,  # Shorter timeout for autolearn
                interactive_confirm=False,
            )

            # 4. Review
            review = self.reviewer.review(
                harness_id=harness_id,
                code=code,
                task_description=result["task"],
                execution_log=exec_log,
                is_agentic=metadata.is_agentic,
            )

            result["score"] = review.overall_score
            result["success"] = review.overall_score >= config.get(
                "autolearn.min_score_to_keep", 0.3
            )

            # 5. Auto-repair if needed
            if should_repair(review) and exec_log.error_message:
                self.repairer.repair(
                    harness_id=harness_id,
                    code=code,
                    task_description=result["task"],
                    execution_log=exec_log,
                    review=review,
                )

            # Store successful code as snippet in RAG
            if result["success"]:
                self._store_in_rag(code, result["task"], review.overall_score)

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Iteration error: {e}")

        return result

    def _generate_idea(self, session: AutolearnSession) -> Dict:
        """Generate a novel task idea using LLM."""
        # Build context from existing tasks
        existing_context = ""
        if session.iterations:
            recent = session.iterations[-10:]
            tasks = [it.get("task", "")[:100] for it in recent if it.get("task")]
            if tasks:
                existing_context = "Avoid repeating these recent tasks:\n" + "\n".join(f"- {t}" for t in tasks)

        # Build reflection context
        reflection_context = ""
        if session.reflection_notes:
            reflection_context = f"Strategy notes: {session.reflection_notes[-1]}"

        prompt = IDEA_GENERATION_PROMPT.format(
            existing_tasks_context=existing_context,
            reflection_context=reflection_context,
        )

        result = llm_call(
            messages=[
                {"role": "system", "content": "You are a creative task designer. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=self.idea_model,
            options=config.get_options_for_model(self.idea_model, "creative"),
        )

        idea = extract_json_from_text(result.get("content", ""))
        if isinstance(idea, dict) and "task_description" in idea:
            return idea

        # Fallback: extract task from text
        content = result.get("content", "")
        return {
            "task_description": content[:500],
            "complexity": "simple",
            "domain": "general",
            "tags": [],
        }

    def _self_reflect(self, session: AutolearnSession):
        """Analyze recent iterations and adjust strategy."""
        recent = session.iterations[-self.reflection_interval:]
        if not recent:
            return

        summaries = "\n".join(
            f"- Iteration {it.get('iteration')}: {'SUCCESS' if it.get('success') else 'FAILED'} "
            f"(score={it.get('score', 0):.2f}, domain={it.get('domain', '?')})"
            for it in recent
        )

        prompt = REFLECTION_PROMPT.format(
            n=len(recent),
            iteration_summaries=summaries,
        )

        result = llm_call(
            messages=[
                {"role": "system", "content": "You are a learning analyst. Be concise."},
                {"role": "user", "content": prompt},
            ],
            model=self.idea_model,
            options=config.get_options_for_model(self.idea_model, "precise"),
        )

        reflection = result.get("content", "")
        if reflection:
            session.reflection_notes.append(reflection)
            logger.info(f"Self-reflection: {reflection[:200]}")

    def _generate_summary(self, session: AutolearnSession) -> Dict:
        """Generate a session summary."""
        summary = {
            "session_id": session.session_id,
            "total_iterations": session.total_iterations,
            "successful": session.successful_count,
            "failed": session.failed_count,
            "success_rate": (
                session.successful_count / max(session.total_iterations, 1)
            ),
            "started_at": session.started_at,
            "ended_at": now_iso(),
            "reflections": session.reflection_notes,
        }

        # Save summary
        summary_path = os.path.join(session.session_dir, "summary.json")
        atomic_write_json(summary_path, summary)

        return summary

    @staticmethod
    def _store_in_rag(code: str, task: str, score: float):
        """Store successful autolearn code as RAG snippet."""
        try:
            from rag.snippets import SnippetStore
            store = SnippetStore()
            store.add_snippet(
                content=code,
                title=f"Autolearn: {task[:80]}",
                description=task,
                tags=["autolearn", "auto_generated"],
                source="autolearn",
                quality_score=score,
            )
        except Exception:
            pass
