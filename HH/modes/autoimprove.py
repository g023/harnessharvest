"""
HarnessHarvester - Autoimprove Mode

Takes existing working harnesses and attempts to create improved versions.
Analyzes weaknesses, generates improvements, and compares against baseline.
"""

import os
from typing import Any, Dict, List, Optional

from core.config import config
from core.constants import STATUS_ACTIVE
from core.storage import atomic_write_json, read_json, ensure_dir
from core.logging_setup import get_logger
from core.utils import llm_call, extract_primary_code, summarize_for_context, now_iso
from harness.generator import HarnessGenerator
from harness.executor import HarnessExecutor
from harness.reviewer import HarnessReviewer
from harness.models import HarnessMetadata

logger = get_logger("autoimprove")


WEAKNESS_ANALYSIS_PROMPT = """Analyze this harness code and identify areas for improvement.

## Task Description
{task_description}

## Code
```python
{code}
```

## Review Scores
{review_summary}

Identify:
1. Performance bottlenecks
2. Error handling gaps
3. Code structure improvements
4. Algorithm optimizations
5. Robustness enhancements

Respond with a JSON object:
{{
    "weaknesses": [
        {{
            "area": "description of weakness",
            "severity": "low|medium|high",
            "suggestion": "how to fix it"
        }}
    ],
    "improvement_priority": "efficiency|robustness|readability|functionality"
}}"""

IMPROVE_PROMPT = """Improve this Python code based on the identified weaknesses.

## Original Task
{task_description}

## Current Code
```python
{code}
```

## Identified Weaknesses
{weaknesses}

## Improvement Focus: {focus}

Requirements:
1. Preserve ALL original functionality
2. Fix identified weaknesses
3. Improve {focus}
4. Keep the same interface/signatures
5. Include the COMPLETE improved code in a ```python block"""


class AutoimproveMode:
    """
    Improves existing working harnesses by analyzing weaknesses
    and generating enhanced versions.
    """

    def __init__(self):
        self.generator = HarnessGenerator()
        self.executor = HarnessExecutor()
        self.reviewer = HarnessReviewer()

    def improve(
        self,
        harness_id: str,
        max_iterations: int = None,
        improvement_threshold: float = None,
    ) -> Dict[str, Any]:
        """
        Attempt to improve an existing harness.

        Args:
            harness_id: The harness to improve
            max_iterations: Max improvement attempts
            improvement_threshold: Min score improvement to accept

        Returns:
            Summary of improvement attempts
        """
        if max_iterations is None:
            max_iterations = config.get("autoimprove.max_improvement_iterations", 5)
        if improvement_threshold is None:
            improvement_threshold = config.get("autoimprove.min_improvement_threshold", 0.05)

        # Load harness
        metadata = self.generator.get_metadata(harness_id)
        if not metadata:
            raise ValueError(f"Harness {harness_id} not found")

        code = self.generator.get_harness_code(harness_id)
        task_description = metadata.task_description

        logger.info(f"Starting autoimprove for {harness_id}")

        # Get baseline score
        baseline_review = self.reviewer.review(
            harness_id=harness_id,
            code=code,
            task_description=task_description,
        )
        baseline_score = baseline_review.overall_score

        logger.info(f"Baseline score: {baseline_score:.2f}")

        # Improvement loop
        results = {
            "harness_id": harness_id,
            "baseline_score": baseline_score,
            "iterations": [],
            "best_score": baseline_score,
            "best_branch": "main",
            "improved": False,
        }

        for i in range(max_iterations):
            logger.info(f"Improvement iteration {i+1}/{max_iterations}")

            try:
                iteration_result = self._improve_iteration(
                    harness_id=harness_id,
                    code=code,
                    task_description=task_description,
                    baseline_review=baseline_review,
                    iteration=i + 1,
                )

                results["iterations"].append(iteration_result)

                if iteration_result["score"] > results["best_score"] + improvement_threshold:
                    results["best_score"] = iteration_result["score"]
                    results["best_branch"] = iteration_result["branch"]
                    results["improved"] = True
                    logger.info(
                        f"Improvement found! Score: {baseline_score:.2f} -> {iteration_result['score']:.2f}"
                    )

            except Exception as e:
                logger.error(f"Improvement iteration {i+1} failed: {e}")
                results["iterations"].append({
                    "iteration": i + 1,
                    "error": str(e),
                    "score": 0.0,
                })

        # Promote best if improved
        if results["improved"]:
            self._promote_improvement(harness_id, results["best_branch"])

        # Save results
        harnesses_dir = config.get_path("harnesses_dir")
        results_path = os.path.join(
            harnesses_dir, harness_id, "autoimprove_results.json"
        )
        atomic_write_json(results_path, results)

        logger.info(
            f"Autoimprove complete: {'IMPROVED' if results['improved'] else 'NO IMPROVEMENT'} "
            f"({baseline_score:.2f} -> {results['best_score']:.2f})"
        )

        return results

    def _improve_iteration(
        self,
        harness_id: str,
        code: str,
        task_description: str,
        baseline_review,
        iteration: int,
    ) -> Dict:
        """Run one improvement iteration."""
        # 1. Analyze weaknesses
        weaknesses = self._analyze_weaknesses(
            code, task_description, baseline_review
        )

        focus = weaknesses.get("improvement_priority", "robustness")
        weakness_list = weaknesses.get("weaknesses", [])

        # 2. Generate improved version
        branch_name = f"autoimprove_v{iteration}"
        improved_code = self._generate_improvement(
            code, task_description, weakness_list, focus
        )

        if not improved_code:
            return {
                "iteration": iteration,
                "error": "No code generated",
                "score": 0.0,
                "branch": branch_name,
            }

        # 3. Create version branch
        version_info = self.generator.create_version(
            harness_id=harness_id,
            code=improved_code,
            branch=branch_name,
            parent_version="v1",
            parent_branch="main",
            model_used=config.get_model("generator"),
        )

        new_version = version_info.version

        # 4. Execute improved version
        exec_log = self.executor.execute(
            harness_id=harness_id,
            code=improved_code,
            version=new_version,
            branch=branch_name,
            timeout=120,
            interactive_confirm=False,
        )

        # 5. Review improved version
        review = self.reviewer.review(
            harness_id=harness_id,
            code=improved_code,
            task_description=task_description,
            execution_log=exec_log,
            version=new_version,
            branch=branch_name,
        )

        # 6. Check functionality preservation
        func_threshold = config.get(
            "autoimprove.preserve_functionality_threshold", 0.9
        )
        func_preserved = (
            review.scores.get("functionality", 0)
            >= baseline_review.scores.get("functionality", 0) * func_threshold
        )

        return {
            "iteration": iteration,
            "branch": branch_name,
            "score": review.overall_score,
            "focus": focus,
            "weaknesses_addressed": len(weakness_list),
            "functionality_preserved": func_preserved,
            "execution_status": exec_log.status,
        }

    def _analyze_weaknesses(
        self,
        code: str,
        task_description: str,
        review,
    ) -> Dict:
        """Analyze code weaknesses using LLM."""
        code_summary = summarize_for_context(code, 3000)
        review_summary = "\n".join(
            f"- {k}: {v:.2f}" for k, v in review.scores.items()
        )
        if review.critique:
            review_summary += f"\nCritique: {review.critique[:500]}"

        prompt = WEAKNESS_ANALYSIS_PROMPT.format(
            task_description=task_description[:500],
            code=code_summary,
            review_summary=review_summary,
        )

        result = llm_call(
            messages=[
                {"role": "system", "content": "You are a code improvement analyst. Respond with JSON."},
                {"role": "user", "content": prompt},
            ],
            model=config.get_model("judge"),
            options=config.get_options_for_model(config.get_model("judge"), "precise"),
        )

        from core.utils import extract_json_from_text
        data = extract_json_from_text(result.get("content", ""))
        return data if isinstance(data, dict) else {"weaknesses": [], "improvement_priority": "robustness"}

    def _generate_improvement(
        self,
        code: str,
        task_description: str,
        weaknesses: List[Dict],
        focus: str,
    ) -> str:
        """Generate improved code."""
        weakness_text = "\n".join(
            f"- [{w.get('severity', 'medium')}] {w.get('area', '')}: {w.get('suggestion', '')}"
            for w in weaknesses
        )

        prompt = IMPROVE_PROMPT.format(
            task_description=task_description[:500],
            code=summarize_for_context(code, 4000),
            weaknesses=weakness_text or "General improvements needed",
            focus=focus,
        )

        result = llm_call(
            messages=[
                {"role": "system", "content": "You are an expert Python developer. Generate improved code."},
                {"role": "user", "content": prompt},
            ],
            model=config.get_model("generator"),
            options=config.get_options_for_model(config.get_model("generator"), "default"),
        )

        return extract_primary_code(result.get("content", ""))

    @staticmethod
    def _promote_improvement(harness_id: str, branch: str):
        """Promote an improved branch to active."""
        harnesses_dir = config.get_path("harnesses_dir")
        meta_path = os.path.join(harnesses_dir, harness_id, "metadata.json")
        meta = read_json(meta_path, {})
        meta["active_branch"] = branch
        meta["updated_at"] = now_iso()
        atomic_write_json(meta_path, meta)
