"""
HarnessHarvester - Multi-Faceted Review System

Uses LLM judges to evaluate harness code and execution results.
Produces structured review reports with scores across multiple dimensions.
"""

import json
from typing import Any, Dict, List, Optional

from core.config import config
from core.constants import (
    MODEL_JUDGE, MODEL_REASONER,
    SCORE_FUNCTIONALITY, SCORE_COMPLETION, SCORE_PLANNING,
    SCORE_EFFICIENCY, SCORE_ROBUSTNESS,
)
from core.logging_setup import get_logger
from core.utils import (
    llm_call, extract_json_from_text, now_iso,
    summarize_for_context, estimate_tokens,
)
from harness.models import ReviewReport, ExecutionLog

logger = get_logger("reviewer")


REVIEW_SYSTEM_PROMPT = """You are an expert code reviewer and judge for the HarnessHarvester system.
Evaluate the given Python code and its execution results.

You MUST respond with a JSON object containing these fields:
{
    "scores": {
        "functionality": <0.0 to 1.0>,
        "completion": <0.0 to 1.0>,
        "planning_quality": <0.0 to 1.0>,
        "efficiency": <0.0 to 1.0>,
        "robustness": <0.0 to 1.0>
    },
    "overall_score": <0.0 to 1.0>,
    "critique": "<detailed critique>",
    "suggestions": ["<suggestion 1>", "<suggestion 2>"],
    "strengths": ["<strength 1>", "<strength 2>"]
}

Scoring guide:
- functionality (0-1): Does the code work as intended? Does it meet the task requirements?
- completion (0-1): How complete is the implementation? Are there missing features?
- planning_quality (0-1): Is the code well-structured? Good separation of concerns?
- efficiency (0-1): Is the code efficient? No unnecessary operations or allocations?
- robustness (0-1): Does it handle errors? Edge cases? Invalid inputs?

Be fair but critical. Only give 0.9+ if the code is truly excellent."""


AGENTIC_REVIEW_PROMPT = """You are reviewing an agentic harness that uses orchestrators and agents.
In addition to code quality, evaluate:
- Quality of the cognitive workflow (reflect, brainstorm, plan, task_list, track)
- Clarity and completeness of the planning stages
- Whether the orchestrator properly delegates to agents
- Quality of tool usage and permission management
- Checkpoint and resume capability

Adjust the planning_quality score to heavily weight orchestrator quality."""


class HarnessReviewer:
    """
    Multi-faceted review system using LLM judges.
    Evaluates code quality, execution results, and agentic planning.
    """

    def __init__(self):
        self.judge_model = config.get_model("judge")

    def review(
        self,
        harness_id: str,
        code: str,
        task_description: str,
        execution_log: Optional[ExecutionLog] = None,
        is_agentic: bool = False,
        version: str = "v1",
        branch: str = "main",
    ) -> ReviewReport:
        """
        Conduct a full review of a harness.

        Args:
            harness_id: The harness ID
            code: The harness Python code
            task_description: What the harness was supposed to do
            execution_log: Optional execution results
            is_agentic: Whether this is an agentic harness
            version: Version being reviewed
            branch: Branch being reviewed
        """
        logger.info(f"Reviewing harness {harness_id} {version}/{branch}")

        # Prepare context
        model_ctx = config.get_model_context(self.judge_model)
        max_code_tokens = int(model_ctx * 0.4)
        code_summary = summarize_for_context(code, max_code_tokens)

        # Build review prompt
        user_parts = [
            f"## Task Description\n{task_description}\n",
            f"## Code\n```python\n{code_summary}\n```\n",
        ]

        if execution_log:
            exec_summary = self._format_execution_summary(execution_log)
            user_parts.append(f"## Execution Results\n{exec_summary}\n")

        user_prompt = "\n".join(user_parts)

        # Select system prompt
        system_prompt = REVIEW_SYSTEM_PROMPT
        if is_agentic:
            system_prompt += "\n\n" + AGENTIC_REVIEW_PROMPT

        # LLM review call
        options = config.get_options_for_model(self.judge_model, "judge")

        result = llm_call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.judge_model,
            options=options,
        )

        # Parse review
        content = result.get("content", "")
        review_data = extract_json_from_text(content)

        if review_data and isinstance(review_data, dict):
            report = self._build_report(harness_id, version, branch, review_data)
        else:
            # Fallback: extract what we can from text
            logger.warning("Failed to parse structured review, using fallback scoring")
            report = self._fallback_review(harness_id, version, branch, content, execution_log)

        report.is_agentic_review = is_agentic
        report.reviewer_model = self.judge_model
        report.reviewed_at = now_iso()

        logger.info(
            f"Review complete: {harness_id} - overall={report.overall_score:.2f} "
            f"(func={report.scores.get(SCORE_FUNCTIONALITY, 0):.2f})"
        )

        # Store review in error RAG if there were issues
        if execution_log and execution_log.error_message:
            self._contribute_to_error_rag(execution_log, report)

        return report

    def quick_score(self, code: str, task_description: str) -> float:
        """Quick scoring without full review. Returns 0.0-1.0."""
        result = llm_call(
            messages=[
                {"role": "system", "content": "Rate the following code on a scale of 0.0 to 1.0 for how well it accomplishes the task. Respond with ONLY a number."},
                {"role": "user", "content": f"Task: {task_description[:500]}\n\nCode:\n{code[:2000]}"},
            ],
            model=self.judge_model,
            options=config.get_options_for_model(self.judge_model, "judge"),
        )
        try:
            content = result.get("content") or "0.5"
            score = float(content.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.5

    # ── Private Helpers ──────────────────────────────────────

    def _build_report(
        self,
        harness_id: str,
        version: str,
        branch: str,
        data: Dict,
    ) -> ReviewReport:
        """Build a ReviewReport from parsed JSON data."""
        scores = data.get("scores", {})

        return ReviewReport(
            harness_id=harness_id,
            version=version,
            branch=branch,
            scores={
                SCORE_FUNCTIONALITY: float(scores.get("functionality", 0.5)),
                SCORE_COMPLETION: float(scores.get("completion", 0.5)),
                SCORE_PLANNING: float(scores.get("planning_quality", 0.5)),
                SCORE_EFFICIENCY: float(scores.get("efficiency", 0.5)),
                SCORE_ROBUSTNESS: float(scores.get("robustness", 0.5)),
            },
            overall_score=float(data.get("overall_score", 0.5)),
            critique=data.get("critique", ""),
            suggestions=data.get("suggestions", []),
            strengths=data.get("strengths", []),
        )

    def _fallback_review(
        self,
        harness_id: str,
        version: str,
        branch: str,
        text: str,
        execution_log: Optional[ExecutionLog],
    ) -> ReviewReport:
        """Create a review from unstructured text with heuristic scores."""
        # Basic heuristics
        func_score = 0.5
        if execution_log:
            func_score = 0.7 if execution_log.exit_code == 0 else 0.2

        return ReviewReport(
            harness_id=harness_id,
            version=version,
            branch=branch,
            scores={
                SCORE_FUNCTIONALITY: func_score,
                SCORE_COMPLETION: 0.5,
                SCORE_PLANNING: 0.5,
                SCORE_EFFICIENCY: 0.5,
                SCORE_ROBUSTNESS: 0.3 if execution_log and execution_log.error_message else 0.5,
            },
            overall_score=func_score * 0.8 + 0.1,
            critique=text[:1000],
            suggestions=["Review manually - automated review parsing failed"],
        )

    @staticmethod
    def _format_execution_summary(log: ExecutionLog) -> str:
        """Format execution log for review context."""
        parts = [
            f"Status: {log.status}",
            f"Exit code: {log.exit_code}",
            f"Duration: {log.duration_seconds:.1f}s",
        ]
        if log.stdout:
            parts.append(f"Output (last 500 chars):\n{log.stdout[-500:]}")
        if log.error_message:
            parts.append(f"Error: {log.error_message}")
        if log.error_traceback:
            parts.append(f"Traceback:\n{log.error_traceback[-1000:]}")
        return "\n".join(parts)

    @staticmethod
    def _contribute_to_error_rag(log: ExecutionLog, report: ReviewReport):
        """Contribute error patterns to the RAG system."""
        try:
            if not log.error_message:
                return
            from rag.errors import ErrorStore
            store = ErrorStore()

            fix_desc = ""
            if report.suggestions:
                fix_desc = "; ".join(report.suggestions[:3])

            store.add_error_pattern(
                error_message=log.error_message,
                fix_description=fix_desc or report.critique[:500],
                traceback=log.error_traceback[:2000],
                source=f"harness_{log.harness_id}",
                quality_score=0.3,  # Low initial quality - needs validation
            )
        except Exception:
            pass
