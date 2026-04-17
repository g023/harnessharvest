#!/usr/bin/env python3
"""
HarnessHarvester - Main CLI Entry Point

Usage:
    python main.py harvest "task description"       - Generate a new harness
    python main.py run <harness_id>                 - Execute a harness
    python main.py review <harness_id>              - Review a harness
    python main.py repair <harness_id>              - Repair a failed harness
    python main.py full "task description"           - Full pipeline: generate → run → review → repair
    python main.py autolearn [--max-iterations N]   - Run autolearn mode
    python main.py autoimprove <harness_id>         - Improve an existing harness
    python main.py rag search <query>               - Search RAG
    python main.py rag stats                        - Show RAG statistics
    python main.py api                              - Start RAG API server
    python main.py list [--status S] [--source S]   - List harnesses
    python main.py report <harness_id>              - Show harness report
"""

import argparse
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import config
from core.logging_setup import setup_logging, get_logger
from core.utils import now_iso


def cmd_harvest(args):
    """Generate a new harness from a task description."""
    from harness.generator import HarnessGenerator

    gen = HarnessGenerator()
    harness_id, metadata = gen.generate(
        task_description=args.task,
        tags=args.tags.split(",") if args.tags else [],
        source="user",
        options_profile=args.profile,
    )

    print(f"Harness generated: {harness_id}")
    print(f"  Task: {metadata.task_description[:100]}")
    print(f"  Agentic: {metadata.is_agentic}")
    print(f"  Status: {metadata.status}")


def cmd_run(args):
    """Execute a harness."""
    from harness.generator import HarnessGenerator
    from harness.executor import HarnessExecutor

    gen = HarnessGenerator()
    executor = HarnessExecutor()

    code = gen.get_harness_code(args.harness_id)
    meta = gen.get_metadata(args.harness_id)

    log = executor.execute(
        harness_id=args.harness_id,
        code=code,
        version=meta.active_version if meta else "v1",
        branch=meta.active_branch if meta else "main",
        timeout=args.timeout,
    )

    print(f"Execution: {log.status}")
    print(f"  Run ID: {log.run_id}")
    print(f"  Duration: {log.duration_seconds:.1f}s")
    print(f"  Exit code: {log.exit_code}")
    if log.stdout:
        print(f"  Output:\n{log.stdout[:500]}")
    if log.error_message:
        print(f"  Error: {log.error_message}")


def cmd_review(args):
    """Review a harness."""
    from harness.generator import HarnessGenerator
    from harness.reviewer import HarnessReviewer

    gen = HarnessGenerator()
    reviewer = HarnessReviewer()

    code = gen.get_harness_code(args.harness_id)
    meta = gen.get_metadata(args.harness_id)

    report = reviewer.review(
        harness_id=args.harness_id,
        code=code,
        task_description=meta.task_description if meta else "",
        is_agentic=meta.is_agentic if meta else False,
    )

    print(f"Review for {args.harness_id}:")
    print(f"  Overall: {report.overall_score:.2f}")
    for k, v in report.scores.items():
        print(f"  {k}: {v:.2f}")
    if report.critique:
        print(f"  Critique: {report.critique[:300]}")
    if report.suggestions:
        print("  Suggestions:")
        for s in report.suggestions[:5]:
            print(f"    - {s}")


def cmd_repair(args):
    """Repair a failed harness."""
    from harness.generator import HarnessGenerator
    from harness.repairer import HarnessRepairer

    gen = HarnessGenerator()
    repairer = HarnessRepairer()

    code = gen.get_harness_code(args.harness_id)
    meta = gen.get_metadata(args.harness_id)

    attempts = repairer.repair(
        harness_id=args.harness_id,
        code=code,
        task_description=meta.task_description if meta else "",
        version=meta.active_version if meta else "v1",
        branch=meta.active_branch if meta else "main",
    )

    print(f"Repair attempts for {args.harness_id}:")
    for att in attempts:
        status = "SUCCESS" if att.success else "FAILED"
        print(f"  [{status}] {att.target_branch}: {att.score_before:.2f} -> {att.score_after:.2f}")


def cmd_full(args):
    """Full pipeline: generate → run → review → repair."""
    from harness.generator import HarnessGenerator
    from harness.executor import HarnessExecutor
    from harness.reviewer import HarnessReviewer
    from harness.repairer import HarnessRepairer, should_repair

    gen = HarnessGenerator()
    executor = HarnessExecutor()
    reviewer = HarnessReviewer()
    repairer = HarnessRepairer()

    print("="*60)
    print(f"Full Pipeline: {args.task[:80]}")
    print("="*60)

    # 1. Generate (agentic by default, or --no-agentic for simple mode)
    agentic_mode = not getattr(args, 'no_agentic', False)
    print(f"\n[1/4] Generating {'agentic' if agentic_mode else 'simple'} harness...")
    harness_id, metadata = gen.generate(
        task_description=args.task,
        tags=args.tags.split(",") if args.tags else [],
        agentic=agentic_mode,
    )
    print(f"  Generated: {harness_id} (agentic={metadata.is_agentic})")

    # 2. Execute (pass agentic flag and task for LLM context)
    print("\n[2/4] Executing harness...")
    code = gen.get_harness_code(harness_id)
    exec_log = executor.execute(
        harness_id=harness_id,
        code=code,
        timeout=args.timeout,
        agentic=metadata.is_agentic,
        task_description=args.task,
    )
    print(f"  Status: {exec_log.status}")
    if metadata.is_agentic:
        print(f"  LLM Calls: {exec_log.llm_calls}")
        print(f"  Deliverables: {len(exec_log.deliverables)}")
    if exec_log.stdout:
        print(f"  Output: {exec_log.stdout[:200]}")
    if exec_log.error_message:
        print(f"  Error: {exec_log.error_message}")

    # 3. Review
    print("\n[3/4] Reviewing harness...")
    review = reviewer.review(
        harness_id=harness_id,
        code=code,
        task_description=args.task,
        execution_log=exec_log,
        is_agentic=metadata.is_agentic,
    )
    print(f"  Score: {review.overall_score:.2f}")
    for k, v in review.scores.items():
        print(f"    {k}: {v:.2f}")

    # 4. Repair if needed
    if should_repair(review) or exec_log.error_message:
        print("\n[4/4] Repairing harness...")
        attempts = repairer.repair(
            harness_id=harness_id,
            code=code,
            task_description=args.task,
            execution_log=exec_log,
            review=review,
        )
        for att in attempts:
            status = "SUCCESS" if att.success else "FAILED"
            print(f"  [{status}] {att.target_branch}: {att.score_before:.2f} -> {att.score_after:.2f}")
    else:
        print("\n[4/4] No repair needed (score >= threshold)")

    print(f"\n{'='*60}")
    print(f"Pipeline complete. Harness: {harness_id}")
    print(f"Final score: {review.overall_score:.2f}")
    print(f"{'='*60}")


def cmd_autolearn(args):
    """Run autolearn mode."""
    from modes.autolearn import AutolearnMode

    mode = AutolearnMode()
    mode.run(
        max_iterations=args.max_iterations,
        max_duration_seconds=args.max_duration,
        session_id=args.session,
    )


def cmd_autoimprove(args):
    """Run autoimprove on a harness."""
    from modes.autoimprove import AutoimproveMode

    mode = AutoimproveMode()
    results = mode.improve(
        harness_id=args.harness_id,
        max_iterations=args.max_iterations,
    )

    print(f"Autoimprove results for {args.harness_id}:")
    print(f"  Baseline: {results['baseline_score']:.2f}")
    print(f"  Best: {results['best_score']:.2f}")
    print(f"  Improved: {results['improved']}")


def cmd_rag(args):
    """RAG operations."""
    from rag.management import RAGManager

    mgr = RAGManager()

    if args.rag_cmd == "search":
        results = mgr.search_all(args.query, top_k=args.top_k)
        for store_name, entries in results.items():
            if entries:
                print(f"\n[{store_name}]")
                for r in entries:
                    print(f"  {r.entry.title} (score={r.final_score:.3f})")
                    if r.entry.description:
                        print(f"    {r.entry.description[:100]}")

    elif args.rag_cmd == "stats":
        stats = mgr.all_stats()
        for store_name, st in stats.items():
            print(f"\n[{store_name}]")
            for k, v in st.items():
                print(f"  {k}: {v}")

    elif args.rag_cmd == "add":
        from rag.snippets import SnippetStore
        store = SnippetStore()
        # Read from stdin or file
        if args.file:
            with open(args.file) as f:
                content = f.read()
        else:
            print("Enter snippet content (Ctrl+D to finish):")
            content = sys.stdin.read()

        entry_id = store.add_snippet(
            content=content,
            title=args.title or "Untitled",
            description=args.description or "",
            tags=args.tags.split(",") if args.tags else [],
        )
        print(f"Added snippet: {entry_id}")


def cmd_api(args):
    """Start the RAG API server."""
    from rag.api import start_api_server
    start_api_server(port=args.port)


def cmd_list(args):
    """List harnesses."""
    from harness.generator import HarnessGenerator

    gen = HarnessGenerator()
    harnesses = gen.list_harnesses(
        status=args.status,
        source=args.source,
        limit=args.limit,
    )

    if not harnesses:
        print("No harnesses found.")
        return

    print(f"{'ID':<30} {'Status':<12} {'Score':<8} {'Source':<12} Task")
    print("-" * 100)
    for h in harnesses:
        print(f"{h.id:<30} {h.status:<12} {h.best_score:<8.2f} {h.source:<12} {h.task_description[:40]}")


def cmd_report(args):
    """Show detailed harness report."""
    from harness.generator import HarnessGenerator
    from core.storage import read_json

    gen = HarnessGenerator()
    meta = gen.get_metadata(args.harness_id)

    if not meta:
        print(f"Harness {args.harness_id} not found.")
        return

    print(f"Harness Report: {meta.id}")
    print(f"{'='*60}")
    print(f"Task: {meta.task_description}")
    print(f"Status: {meta.status}")
    print(f"Agentic: {meta.is_agentic}")
    print(f"Source: {meta.source}")
    print(f"Created: {meta.created_at}")
    print(f"Updated: {meta.updated_at}")
    print(f"Versions: {meta.total_versions}")
    print(f"Runs: {meta.total_runs}")
    print(f"Best Score: {meta.best_score:.2f}")
    print(f"Active: {meta.active_version}/{meta.active_branch}")
    print(f"Tags: {', '.join(meta.tags)}")

    # Show code
    if args.show_code:
        try:
            code = gen.get_harness_code(args.harness_id)
            print(f"\n{'='*60}")
            print(f"Code ({len(code.splitlines())} lines):")
            print(f"{'='*60}")
            print(code)
        except FileNotFoundError:
            print("\n(Code not available)")


def main():
    parser = argparse.ArgumentParser(
        description="HarnessHarvester - Self-learning LLM-powered harness management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # harvest
    p_harvest = subparsers.add_parser("harvest", help="Generate a new harness")
    p_harvest.add_argument("task", help="Task description")
    p_harvest.add_argument("--tags", default="", help="Comma-separated tags")
    p_harvest.add_argument("--profile", default="creative", help="Options profile")

    # run
    p_run = subparsers.add_parser("run", help="Execute a harness")
    p_run.add_argument("harness_id", help="Harness ID")
    p_run.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")

    # review
    p_review = subparsers.add_parser("review", help="Review a harness")
    p_review.add_argument("harness_id", help="Harness ID")

    # repair
    p_repair = subparsers.add_parser("repair", help="Repair a failed harness")
    p_repair.add_argument("harness_id", help="Harness ID")

    # full
    p_full = subparsers.add_parser("full", help="Full pipeline: generate → run → review → repair")
    p_full.add_argument("task", help="Task description")
    p_full.add_argument("--tags", default="", help="Comma-separated tags")
    p_full.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    p_full.add_argument("--no-agentic", action="store_true", help="Generate simple script instead of agentic harness")

    # autolearn
    p_autolearn = subparsers.add_parser("autolearn", help="Run autolearn mode")
    p_autolearn.add_argument("--max-iterations", type=int, default=0, help="Max iterations (0=infinite)")
    p_autolearn.add_argument("--max-duration", type=int, default=0, help="Max duration seconds (0=infinite)")
    p_autolearn.add_argument("--session", default=None, help="Resume session ID")

    # autoimprove
    p_autoimprove = subparsers.add_parser("autoimprove", help="Improve existing harness")
    p_autoimprove.add_argument("harness_id", help="Harness ID")
    p_autoimprove.add_argument("--max-iterations", type=int, default=5, help="Max improvement iterations")

    # rag
    p_rag = subparsers.add_parser("rag", help="RAG operations")
    rag_sub = p_rag.add_subparsers(dest="rag_cmd")

    p_rag_search = rag_sub.add_parser("search", help="Search RAG")
    p_rag_search.add_argument("query", help="Search query")
    p_rag_search.add_argument("--top-k", type=int, default=5, help="Number of results")

    p_rag_stats = rag_sub.add_parser("stats", help="RAG statistics")

    p_rag_add = rag_sub.add_parser("add", help="Add snippet to RAG")
    p_rag_add.add_argument("--title", help="Snippet title")
    p_rag_add.add_argument("--description", default="", help="Description")
    p_rag_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_rag_add.add_argument("--file", default=None, help="Read content from file")

    # api
    p_api = subparsers.add_parser("api", help="Start RAG API server")
    p_api.add_argument("--port", type=int, default=8420, help="Port to listen on")

    # list
    p_list = subparsers.add_parser("list", help="List harnesses")
    p_list.add_argument("--status", default=None, help="Filter by status")
    p_list.add_argument("--source", default=None, help="Filter by source")
    p_list.add_argument("--limit", type=int, default=50, help="Max results")

    # report
    p_report = subparsers.add_parser("report", help="Show harness report")
    p_report.add_argument("harness_id", help="Harness ID")
    p_report.add_argument("--show-code", action="store_true", help="Show harness code")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    setup_logging()

    # Dispatch
    commands = {
        "harvest": cmd_harvest,
        "run": cmd_run,
        "review": cmd_review,
        "repair": cmd_repair,
        "full": cmd_full,
        "autolearn": cmd_autolearn,
        "autoimprove": cmd_autoimprove,
        "rag": cmd_rag,
        "api": cmd_api,
        "list": cmd_list,
        "report": cmd_report,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            logger = get_logger("main")
            logger.error(f"Command '{args.command}' failed: {e}", exc_info=True)
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
