"""
HarnessHarvester - Self-Repair System

Intelligent repair with branching strategy, checkpoint-aware fixes,
and model selection based on error type.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from core.config import config
from core.constants import (
    MODEL_JUDGE, MODEL_REASONER, MODEL_GENERATOR,
    MAX_REPAIR_ATTEMPTS, MAX_BRANCHES_PER_VERSION,
    STATUS_ACTIVE, STATUS_FAILED, STATUS_REPAIRING,
    AUTO_REPAIR_THRESHOLD,
)
from core.storage import atomic_write_json, read_json, ensure_dir
from core.logging_setup import get_logger
from core.utils import (
    llm_call, extract_primary_code, extract_json_from_text,
    summarize_for_context, now_iso,
)
from harness.models import (
    RepairAttempt, ReviewReport, ExecutionLog, HarnessMetadata,
)
from harness.generator import HarnessGenerator
from harness.reviewer import HarnessReviewer

logger = get_logger("repairer")


# ─── Repair Prompts ─────────────────────────────────────────────

REPAIR_SYSTEM_PROMPT = """You are an expert Python debugger and code repair specialist.
Given a failed harness (code + error + review critique), fix the code.

Important rules:
1. Fix ONLY the broken parts - preserve working code
2. Maintain the same function/class signatures
3. If checkpoint state is provided, ensure the fix can resume from that state
4. Include the complete fixed code in a ```python block
5. Explain what you fixed and why"""

STRATEGY_ANALYSIS_PROMPT = """Analyze this failed harness and suggest repair strategies.

Code (summarized):
{code_summary}

Error: {error_message}
Traceback: {traceback}
Review critique: {critique}

Respond with a JSON array of repair strategies:
[
    {{
        "strategy": "strategy_name",
        "description": "what this strategy does",
        "model_recommendation": "syntax_fix|reasoning_fix|structural_fix",
        "confidence": 0.0 to 1.0,
        "changes_needed": ["list of specific changes"]
    }}
]

Suggest 2-3 different strategies from different angles."""


class HarnessRepairer:
    """
    Self-repair system with branching strategy.
    Analyzes failures, generates multiple repair approaches,
    and creates branches for each approach.
    """

    def __init__(self):
        self.generator = HarnessGenerator()
        self.reviewer = HarnessReviewer()
        self.max_attempts = config.get("harness.max_repair_attempts", MAX_REPAIR_ATTEMPTS)
        self.max_branches = config.get("harness.max_branches_per_version", MAX_BRANCHES_PER_VERSION)

    def repair(
        self,
        harness_id: str,
        code: str,
        task_description: str,
        execution_log: Optional[ExecutionLog] = None,
        review: Optional[ReviewReport] = None,
        checkpoint_state: Optional[Dict] = None,
        version: str = "v1",
        branch: str = "main",
    ) -> List[RepairAttempt]:
        """
        Attempt to repair a failed harness using multiple strategies.

        Returns list of RepairAttempts, one per branch created.
        """
        logger.info(f"Starting repair for {harness_id} {version}/{branch}")

        # 1. Analyze failure and generate strategies
        strategies = self._analyze_failure(
            code, execution_log, review, checkpoint_state
        )

        if not strategies:
            strategies = [self._default_strategy(execution_log)]

        # Limit branches
        strategies = strategies[:self.max_branches]

        # 2. Execute each repair strategy as a branch
        attempts = []
        for i, strategy in enumerate(strategies):
            branch_name = f"repair_{i+1}_{strategy.get('strategy', 'fix')}"

            try:
                attempt = self._execute_repair(
                    harness_id=harness_id,
                    code=code,
                    task_description=task_description,
                    execution_log=execution_log,
                    review=review,
                    checkpoint_state=checkpoint_state,
                    strategy=strategy,
                    source_version=version,
                    source_branch=branch,
                    target_branch=branch_name,
                )
                attempts.append(attempt)

                if attempt.success:
                    logger.info(
                        f"Repair branch '{branch_name}' succeeded "
                        f"(score: {attempt.score_before:.2f} -> {attempt.score_after:.2f})"
                    )
            except Exception as e:
                logger.error(f"Repair strategy '{branch_name}' failed: {e}")
                attempts.append(RepairAttempt(
                    harness_id=harness_id,
                    source_version=version,
                    source_branch=branch,
                    target_version="",
                    target_branch=branch_name,
                    repair_strategy=strategy.get("strategy", "unknown"),
                    model_used=strategy.get("model", ""),
                    error_info=str(e),
                    success=False,
                    created_at=now_iso(),
                ))

        # 3. Find best branch and promote if improved
        self._promote_best_branch(harness_id, attempts)

        # 4. Store repair attempts
        self._save_attempts(harness_id, attempts)

        return attempts

    def _analyze_failure(
        self,
        code: str,
        execution_log: Optional[ExecutionLog],
        review: Optional[ReviewReport],
        checkpoint_state: Optional[Dict],
    ) -> List[Dict]:
        """Use LLM to analyze failure and generate repair strategies."""
        error_message = ""
        traceback = ""
        critique = ""

        if execution_log:
            error_message = execution_log.error_message or ""
            traceback = execution_log.error_traceback or ""
        if review:
            critique = review.critique or ""

        # Use reasoner model for analysis (strong reasoning)
        code_summary = summarize_for_context(code, 3000)

        prompt = STRATEGY_ANALYSIS_PROMPT.format(
            code_summary=code_summary,
            error_message=error_message[:500],
            traceback=traceback[:1000],
            critique=critique[:500],
        )

        result = llm_call(
            messages=[
                {"role": "system", "content": "You are a debugging strategy analyst. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=config.get_model("reasoner"),
            options=config.get_options_for_model(config.get_model("reasoner"), "precise"),
        )

        strategies = extract_json_from_text(result.get("content", ""))
        if isinstance(strategies, list):
            # Assign models based on recommendation
            for s in strategies:
                rec = s.get("model_recommendation", "")
                s["model"] = self._select_repair_model(rec)
            return strategies

        return []

    def _execute_repair(
        self,
        harness_id: str,
        code: str,
        task_description: str,
        execution_log: Optional[ExecutionLog],
        review: Optional[ReviewReport],
        checkpoint_state: Optional[Dict],
        strategy: Dict,
        source_version: str,
        source_branch: str,
        target_branch: str,
    ) -> RepairAttempt:
        """Execute a single repair strategy."""
        model = strategy.get("model", config.get_model("generator"))
        strategy_name = strategy.get("strategy", "general_fix")
        changes = strategy.get("changes_needed", [])

        # Build repair prompt
        user_parts = [
            f"## Task\n{task_description}\n",
            f"## Failed Code\n```python\n{summarize_for_context(code, 4000)}\n```\n",
            f"## Repair Strategy: {strategy_name}\n{strategy.get('description', '')}\n",
            f"## Required Changes\n" + "\n".join(f"- {c}" for c in changes) + "\n",
        ]

        if execution_log and execution_log.error_message:
            user_parts.append(f"## Error\n{execution_log.error_message}\n")
            if execution_log.error_traceback:
                user_parts.append(f"## Traceback\n{execution_log.error_traceback[-1500:]}\n")

        if review and review.critique:
            user_parts.append(f"## Reviewer Critique\n{review.critique[:500]}\n")

        if checkpoint_state:
            user_parts.append(f"## Last Checkpoint State\n{str(checkpoint_state)[:500]}\n")
            user_parts.append("Ensure the fix preserves resume capability from this checkpoint.\n")

        user_prompt = "\n".join(user_parts)

        # LLM repair call
        result = llm_call(
            messages=[
                {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            options=config.get_options_for_model(model, "default"),
        )

        fixed_code = extract_primary_code(result.get("content", ""))
        if not fixed_code:
            raise ValueError("No code generated by repair model")

        # Create new version with this branch
        version_info = self.generator.create_version(
            harness_id=harness_id,
            code=fixed_code,
            branch=target_branch,
            parent_version=source_version,
            parent_branch=source_branch,
            model_used=model,
            repair_strategy=strategy_name,
        )

        # Quick-score the repair
        score_before = review.overall_score if review else 0.0
        score_after = self.reviewer.quick_score(fixed_code, task_description)

        return RepairAttempt(
            harness_id=harness_id,
            source_version=source_version,
            source_branch=source_branch,
            target_version=version_info.version,
            target_branch=target_branch,
            repair_strategy=strategy_name,
            model_used=model,
            error_info=execution_log.error_message if execution_log else "",
            checkpoint_used=checkpoint_state.get("checkpoint_id", "") if checkpoint_state else "",
            success=score_after > score_before,
            score_before=score_before,
            score_after=score_after,
            created_at=now_iso(),
        )

    def _select_repair_model(self, recommendation: str) -> str:
        """Select the appropriate model based on repair type."""
        # Map repair types to existing model roles:
        # syntax_fix -> generator (fast, good at code)
        # reasoning_fix -> reasoner (strong reasoning)
        # structural_fix -> generator (code generation)
        model_map = {
            "syntax_fix": config.get_model("generator"),
            "reasoning_fix": config.get_model("reasoner"),
            "structural_fix": config.get_model("generator"),
        }
        return model_map.get(recommendation, config.get_model("generator"))

    @staticmethod
    def _default_strategy(execution_log: Optional[ExecutionLog]) -> Dict:
        """Generate a default repair strategy when analysis fails."""
        error_type = "general"
        if execution_log and execution_log.error_message:
            msg = execution_log.error_message
            if "SyntaxError" in msg or "IndentationError" in msg:
                error_type = "syntax_fix"
            elif "TypeError" in msg or "AttributeError" in msg:
                error_type = "reasoning_fix"
            elif "ImportError" in msg or "ModuleNotFoundError" in msg:
                error_type = "structural_fix"

        return {
            "strategy": "general_fix",
            "description": "Fix the identified error",
            "model_recommendation": error_type,
            "confidence": 0.5,
            "changes_needed": ["Fix the reported error"],
        }

    def _promote_best_branch(self, harness_id: str, attempts: List[RepairAttempt]):
        """Promote the best-scoring repair branch to active."""
        successful = [a for a in attempts if a.success and a.score_after > 0]
        if not successful:
            return

        best = max(successful, key=lambda a: a.score_after)
        logger.info(
            f"Promoting branch '{best.target_branch}' "
            f"(score: {best.score_after:.2f}) for {harness_id}"
        )

        # Update harness metadata
        harnesses_dir = config.get_path("harnesses_dir")
        meta_path = os.path.join(harnesses_dir, harness_id, "metadata.json")
        meta = read_json(meta_path, {})
        meta["active_version"] = best.target_version
        meta["active_branch"] = best.target_branch
        meta["best_score"] = best.score_after
        meta["status"] = STATUS_ACTIVE
        meta["updated_at"] = now_iso()
        atomic_write_json(meta_path, meta)

    def _save_attempts(self, harness_id: str, attempts: List[RepairAttempt]):
        """Save repair attempts to harness directory."""
        harnesses_dir = config.get_path("harnesses_dir")
        repair_dir = os.path.join(harnesses_dir, harness_id, "repairs")
        ensure_dir(repair_dir)

        for attempt in attempts:
            path = os.path.join(
                repair_dir,
                f"repair_{attempt.target_branch}_{attempt.created_at[:19].replace(':', '-')}.json"
            )
            atomic_write_json(path, attempt.to_dict())


def should_repair(review: ReviewReport) -> bool:
    """Determine if a harness should be auto-repaired based on review scores."""
    threshold = config.get("harness.auto_repair_threshold", AUTO_REPAIR_THRESHOLD)
    return review.overall_score < threshold


def classify_error(error_message: str) -> str:
    """Classify an error to determine repair approach."""
    msg = error_message.lower()

    if any(k in msg for k in ["syntaxerror", "indentationerror", "tabulation"]):
        return "syntax_fix"
    if any(k in msg for k in ["typeerror", "attributeerror", "nameerror", "keyerror"]):
        return "reasoning_fix"
    if any(k in msg for k in ["importerror", "modulenotfounderror", "filenotfounderror"]):
        return "structural_fix"
    if any(k in msg for k in ["timeout", "recursion", "memory"]):
        return "structural_fix"

    return "reasoning_fix"  # Default to reasoning model
