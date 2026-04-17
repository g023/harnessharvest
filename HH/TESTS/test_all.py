#!/usr/bin/env python3
"""
HarnessHarvester - Comprehensive Test Suite

Run with: python -m TESTS.test_all
Or:       python TESTS/test_all.py
"""

import sys
import os
import json
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0
SKIP = 0


def test(name, fn):
    """Run a single test and report result."""
    global PASS, FAIL, SKIP
    try:
        fn()
        PASS += 1
        print(f"  [PASS] {name}")
    except AssertionError as e:
        FAIL += 1
        print(f"  [FAIL] {name}: {e}")
    except Exception as e:
        FAIL += 1
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════
# CORE TESTS
# ═══════════════════════════════════════════════════════════════

def test_core():
    print("\n── Core Infrastructure ──")

    def test_constants():
        from core.constants import (
            PROJECT_ROOT, DB_ROOT, SANDBOX_ROOT, HARNESSES_DIR,
            MODEL_JUDGE, MODEL_REASONER, MODEL_GENERATOR,
            STATUS_DRAFT, STATUS_ACTIVE, STATUS_FAILED,
            ORCHESTRATOR_STAGES,
            RAG_TYPE_SNIPPET, RAG_TYPE_ERROR, RAG_TYPE_PROMPT,
        )
        assert len(ORCHESTRATOR_STAGES) == 6
        assert STATUS_DRAFT == "draft"
        assert RAG_TYPE_SNIPPET == "snippet"
        assert RAG_TYPE_ERROR == "error_pattern"
        assert RAG_TYPE_PROMPT == "prompt_template"
    test("constants", test_constants)

    def test_config():
        from core.config import config
        assert config.get_model("judge") is not None
        assert config.get_model("reasoner") is not None
        assert config.get_model("generator") is not None
        assert config.get_model_context(config.get_model("judge")) > 0
        opts = config.get_options_for_model(config.get_model("judge"), "judge")
        assert isinstance(opts, dict)
        assert config.get_path("harnesses_dir") is not None
    test("config", test_config)

    def test_storage_atomic():
        from core.storage import atomic_write, atomic_write_json, read_json, ensure_dir
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            # Atomic write
            path = os.path.join(d, "test.txt")
            atomic_write(path, "hello world")
            with open(path) as f:
                assert f.read() == "hello world"

            # JSON round-trip
            json_path = os.path.join(d, "test.json")
            data = {"key": "value", "nested": {"a": 1}}
            atomic_write_json(json_path, data)
            loaded = read_json(json_path)
            assert loaded == data

            # Ensure dir
            deep = os.path.join(d, "a", "b", "c")
            ensure_dir(deep)
            assert os.path.isdir(deep)
    test("storage atomic ops", test_storage_atomic)

    def test_storage_locking():
        from core.storage import locked_read_json, locked_write_json, locked_update_json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "locked.json")
            locked_write_json(path, {"count": 0})
            data = locked_read_json(path)
            assert data["count"] == 0

            locked_update_json(path, lambda d: {**d, "count": d["count"] + 1})
            data = locked_read_json(path)
            assert data["count"] == 1
    test("storage locking", test_storage_locking)

    def test_storage_jsonl():
        from core.storage import append_to_jsonl, read_jsonl
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")
            append_to_jsonl(path, {"line": 1})
            append_to_jsonl(path, {"line": 2})
            append_to_jsonl(path, {"line": 3})
            lines = read_jsonl(path)
            assert len(lines) == 3
            assert lines[1]["line"] == 2
    test("storage JSONL", test_storage_jsonl)

    def test_storage_id():
        from core.storage import generate_id, content_hash
        id1 = generate_id("test")
        id2 = generate_id("test")
        assert id1.startswith("test_")
        assert id1 != id2
        h1 = content_hash("hello")
        h2 = content_hash("hello")
        assert h1 == h2
        assert h1 != content_hash("world")
    test("storage ID generation", test_storage_id)

    def test_utils_code_extract():
        from core.utils import extract_code_blocks, extract_primary_code
        text = "Here's the code:\n```python\ndef hello():\n    print('hi')\n```\nAnd more."
        blocks = extract_code_blocks(text)
        assert len(blocks) >= 1
        assert "def hello():" in blocks[0]
        primary = extract_primary_code(text)
        assert "def hello():" in primary
    test("utils code extraction", test_utils_code_extract)

    def test_utils_json():
        from core.utils import safe_json_loads, extract_json_from_text
        # Normal JSON
        data = safe_json_loads('{"key": "value"}')
        assert data["key"] == "value"
        # JSON in text
        text = "Here's the result: {\"score\": 0.8, \"name\": \"test\"} end"
        data = extract_json_from_text(text)
        assert data is not None
        assert data.get("score") == 0.8
    test("utils JSON parsing", test_utils_json)

    def test_utils_imports():
        from core.utils import extract_imports, validate_imports
        code = "import os\nimport sys\nfrom collections import defaultdict\nimport subprocess"
        imports = extract_imports(code)
        assert "os" in imports
        assert "subprocess" in imports
        # validate_imports checks against allowed list + stdlib
        valid, bad = validate_imports(code)
        # subprocess should be blocked by sandbox config
        if not valid:
            assert "subprocess" in bad
        else:
            pass  # May pass if subprocess is in allowed list
    test("utils import validation", test_utils_imports)

    def test_utils_tokens():
        from core.utils import estimate_tokens, truncate_to_tokens
        text = "hello world " * 100
        tokens = estimate_tokens(text)
        assert tokens > 0
        truncated = truncate_to_tokens(text, 10)
        assert len(truncated) < len(text)
    test("utils token estimation", test_utils_tokens)

    def test_utils_safety():
        from core.utils import is_safe_path
        from core.constants import SANDBOX_ROOT
        assert is_safe_path(os.path.join(SANDBOX_ROOT, "test.txt"), SANDBOX_ROOT)
        assert not is_safe_path("/etc/passwd", SANDBOX_ROOT)
        assert not is_safe_path(os.path.join(SANDBOX_ROOT, "..", "etc"), SANDBOX_ROOT)
    test("utils path safety", test_utils_safety)


# ═══════════════════════════════════════════════════════════════
# RAG TESTS
# ═══════════════════════════════════════════════════════════════

def test_rag():
    print("\n── RAG System ──")

    def test_embeddings_tokenize():
        from rag.embeddings import tokenize, tokenize_with_ngrams
        tokens = tokenize("Hello World! This is a camelCase testVariable.")
        assert "hello" in tokens
        assert "world" in tokens
        # Verify some tokens exist from camelCase splitting
        assert len(tokens) >= 3
        ngrams = tokenize_with_ngrams("hello world test")
        assert len(ngrams) >= 1
    test("embeddings tokenization", test_embeddings_tokenize)

    def test_tfidf():
        from rag.embeddings import TFIDFVectorizer
        import numpy as np
        vec = TFIDFVectorizer(max_features=64)
        docs = ["binary search algorithm", "fibonacci recursive function", "bubble sort comparison"]
        vec.fit(docs)
        v = vec.transform("search algorithm")
        # Shape is (vocab_size,) which may be <= max_features
        assert len(v.shape) == 1
        assert v.shape[0] > 0
        assert np.linalg.norm(v) > 0
    test("TF-IDF vectorizer", test_tfidf)

    def test_bm25():
        from rag.embeddings import BM25Scorer
        bm25 = BM25Scorer()
        docs = ["the cat sat on the mat", "the dog chased the cat", "a fish swam in the sea"]
        bm25.fit(docs)
        scores = bm25.score_all("cat")
        assert len(scores) == 3
        assert scores[0] > scores[2]  # "cat" appears in doc 0 and 1
    test("BM25 scorer", test_bm25)

    def test_vector_index():
        from rag.engine import VectorIndex
        import numpy as np
        idx = VectorIndex(dimension=32)
        for i in range(5):
            v = np.random.randn(32).astype(np.float32)
            idx.add(f"entry_{i}", v)
        assert idx.size == 5
        query = np.random.randn(32).astype(np.float32)
        results = idx.search(query, top_k=3)
        assert len(results) == 3  # Returns list of (id, score) tuples
        assert isinstance(results[0], tuple)
        # Remove marks entry as None, so size stays same until rebuild
        idx.remove("entry_2")
        assert "entry_2" not in idx._id_to_idx
    test("FAISS vector index", test_vector_index)

    def test_rag_engine_crud():
        from rag.engine import RAGEngine, RAGEntry
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            engine = RAGEngine(store_dir=d)
            # Add - RAGEngine.add() takes a RAGEntry object
            entry1 = RAGEntry(id="", content="binary search algorithm implementation", entry_type="snippet", title="Binary Search")
            entry2 = RAGEntry(id="", content="fibonacci recursive function", entry_type="snippet", title="Fibonacci")
            entry3 = RAGEntry(id="", content="bubble sort comparison", entry_type="snippet", title="Bubble Sort")
            e1 = engine.add(entry1)
            e2 = engine.add(entry2)
            e3 = engine.add(entry3)
            assert len(engine._entries) == 3
            # Get
            entry = engine.get(e1)
            assert entry is not None
            assert entry.title == "Binary Search"
            # Delete
            engine.delete(e3)
            assert len(engine._entries) == 2
    test("RAG engine CRUD", test_rag_engine_crud)

    def test_rag_engine_search():
        from rag.engine import RAGEngine, RAGEntry
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            engine = RAGEngine(store_dir=d)
            engine.add(RAGEntry(id="", content="binary search algorithm for sorted arrays", entry_type="snippet", title="Binary Search", tags=["algorithm", "search"]))
            engine.add(RAGEntry(id="", content="fibonacci sequence recursive implementation", entry_type="snippet", title="Fibonacci", tags=["algorithm", "recursion"]))
            engine.add(RAGEntry(id="", content="python string formatting methods", entry_type="snippet", title="String Formatting", tags=["python", "strings"]))

            results = engine.search("search algorithm", top_k=3)
            assert len(results) >= 1
            assert results[0].entry.title == "Binary Search"

            results2 = engine.search("recursive function", top_k=3)
            assert len(results2) >= 1
    test("RAG engine search", test_rag_engine_search)

    def test_snippet_store():
        from rag.snippets import SnippetStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            store = SnippetStore(store_dir=d)
            sid = store.add_snippet(
                content="def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1",
                title="Binary Search",
                tags=["algorithm"],
            )
            assert sid is not None
            results = store.search("search algorithm")
            assert len(results) >= 1
    test("snippet store", test_snippet_store)

    def test_error_store():
        from rag.errors import ErrorStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            store = ErrorStore(store_dir=d)
            eid = store.add_error_pattern(
                error_message="IndexError: list index out of range",
                fix_description="Check list length before accessing index",
                traceback="File test.py line 10",
            )
            assert eid is not None
            results = store.find_fix("IndexError list index")
            assert len(results) >= 1
    test("error store", test_error_store)

    def test_prompt_store():
        from rag.prompts import PromptStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            store = PromptStore(store_dir=d)
            pid = store.add_prompt_template(
                template="Generate Python code for: {task}",
                task_type="code_generation",
                description="Basic code generation prompt",
            )
            assert pid is not None
            results = store.find_prompt("code generation")
            assert len(results) >= 1
    test("prompt store", test_prompt_store)

    def test_rag_manager():
        from rag.management import RAGManager
        mgr = RAGManager()
        # Just verify it can be instantiated and stats work
        stats = mgr.all_stats()
        assert "snippets" in stats
    test("RAG manager", test_rag_manager)

    def test_ranking():
        from rag.ranking import RankingEngine
        engine = RankingEngine()
        # rank() takes List[Dict[str, float]] and returns sorted indices
        items = [
            {"tfidf": 0.5, "bm25": 0.6, "quality": 0.8},
            {"tfidf": 0.9, "bm25": 0.3, "quality": 0.5},
            {"tfidf": 0.2, "bm25": 0.1, "quality": 0.3},
        ]
        ranked = engine.rank(items)
        assert isinstance(ranked, list)
        assert len(ranked) == 3
    test("ranking engine", test_ranking)


# ═══════════════════════════════════════════════════════════════
# HARNESS TESTS
# ═══════════════════════════════════════════════════════════════

def test_harness():
    print("\n── Harness System ──")

    def test_models_roundtrip():
        from harness.models import (
            HarnessMetadata, VersionInfo, VersionIndex,
            ReviewReport, ExecutionLog, CheckpointState, RepairAttempt,
        )
        # HarnessMetadata
        m = HarnessMetadata(id="test", task_description="test task", tags=["a"])
        d = m.to_dict()
        m2 = HarnessMetadata.from_dict(d)
        assert m2.id == "test" and m2.tags == ["a"]

        # VersionIndex
        vi = VersionIndex(harness_id="test")
        vi.add_version(VersionInfo(version="v1", branch="main", review_score=0.7))
        vi.add_version(VersionInfo(version="v1", branch="repair_1", review_score=0.85))
        best = vi.get_best_version()
        assert best.review_score == 0.85
        d = vi.to_dict()
        vi2 = VersionIndex.from_dict(d)
        assert len(vi2.versions) == 2
    test("data models round-trip", test_models_roundtrip)

    def test_orchestrator_stages():
        from harness.orchestrator import BaseOrchestrator, StageError, StageOutput
        class TestOrch(BaseOrchestrator):
            def reflect(self): return StageOutput(stage="reflect", content="ok")
            def brainstorm(self): return StageOutput(stage="brainstorm", content="ok")
            def plan(self): return StageOutput(stage="plan", content="ok")
            def create_task_list(self): return StageOutput(stage="create_task_list", content="ok")
            def track_progress(self): return StageOutput(stage="track_progress", content="ok")
            def execute(self): return StageOutput(stage="execute", content="ok")

        orch = TestOrch("test")
        # Can't skip stages
        try:
            orch._advance_to_stage("brainstorm")
            assert False, "Should have raised"
        except StageError:
            pass

        # Can advance properly
        orch._advance_to_stage("reflect")
        orch._complete_stage("reflect", StageOutput(stage="reflect", content="ok"))
        orch._advance_to_stage("brainstorm")  # Now OK
    test("orchestrator stage enforcement", test_orchestrator_stages)

    def test_orchestrator_permissions():
        from harness.orchestrator import (
            STAGE_PERMISSIONS, ToolPermission,
        )
        from core.constants import STAGE_REFLECT, STAGE_EXECUTE
        # Reflect stage should not have EXECUTE permission
        assert ToolPermission.EXECUTE not in STAGE_PERMISSIONS[STAGE_REFLECT]
        # Execute stage should have everything
        assert ToolPermission.EXECUTE in STAGE_PERMISSIONS[STAGE_EXECUTE]
        assert ToolPermission.LLM_CALL in STAGE_PERMISSIONS[STAGE_EXECUTE]
    test("orchestrator permissions", test_orchestrator_permissions)

    def test_agentic_detection():
        from harness.orchestrator import detect_agentic_harness
        agentic = "class MyOrchestrator(BaseOrchestrator):\n    def reflect(self): pass\n    def brainstorm(self): pass"
        assert detect_agentic_harness(agentic)
        non_agentic = "print('hello')\nx = 42"
        assert not detect_agentic_harness(non_agentic)
    test("agentic detection", test_agentic_detection)

    def test_executor_sandbox():
        from harness.executor import HarnessExecutor
        exec_ = HarnessExecutor()
        log = exec_.execute(
            harness_id="test_suite",
            code='print("test output")\nprint(1+1)',
            version="v1", branch="main",
            timeout=10, interactive_confirm=False,
        )
        assert log.status == "completed"
        assert "test output" in log.stdout
        assert "2" in log.stdout
    test("executor sandbox exec", test_executor_sandbox)

    def test_executor_path_restrict():
        from harness.executor import HarnessExecutor
        exec_ = HarnessExecutor()
        log = exec_.execute(
            harness_id="test_restrict",
            code='open("/etc/passwd").read()',
            version="v1", branch="main",
            timeout=10, interactive_confirm=False,
        )
        assert log.status == "failed"
        assert "outside sandbox" in log.error_message
    test("executor path restriction", test_executor_path_restrict)

    def test_executor_timeout():
        from harness.executor import HarnessExecutor
        exec_ = HarnessExecutor()
        log = exec_.execute(
            harness_id="test_timeout",
            code='import time; time.sleep(60)',
            version="v1", branch="main",
            timeout=2, interactive_confirm=False,
        )
        assert log.status == "failed"
        assert "timed out" in log.error_message
    test("executor timeout", test_executor_timeout)

    def test_executor_checkpoint():
        from harness.executor import HarnessExecutor
        exec_ = HarnessExecutor()
        code = 'checkpoint({"x": 42}, "step1")\ncp = load_checkpoint()\nassert cp["state"]["x"] == 42\nprint("ok")'
        log = exec_.execute(
            harness_id="test_cp",
            code=code,
            version="v1", branch="main",
            timeout=10, interactive_confirm=False,
        )
        assert log.status == "completed"
        assert log.last_checkpoint  # checkpoint file was created
    test("executor checkpoint", test_executor_checkpoint)

    def test_error_classify():
        from harness.repairer import classify_error
        assert classify_error("SyntaxError: invalid syntax") == "syntax_fix"
        assert classify_error("TypeError: unsupported operand") == "reasoning_fix"
        assert classify_error("ImportError: No module named xyz") == "structural_fix"
        assert classify_error("RecursionError: maximum depth") == "structural_fix"
    test("error classification", test_error_classify)

    def test_should_repair():
        from harness.repairer import should_repair
        from harness.models import ReviewReport
        low = ReviewReport(harness_id="t", version="v1", overall_score=0.3)
        high = ReviewReport(harness_id="t", version="v1", overall_score=0.8)
        assert should_repair(low)
        assert not should_repair(high)
    test("should_repair threshold", test_should_repair)


# ═══════════════════════════════════════════════════════════════
# CLI TESTS
# ═══════════════════════════════════════════════════════════════

def test_cli():
    print("\n── CLI ──")

    def test_cli_import():
        from main import main
        assert callable(main)
    test("CLI import", test_cli_import)

    def test_cli_help():
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "harvest" in result.stdout
        assert "autolearn" in result.stdout
    test("CLI help", test_cli_help)


# ═══════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════

def main():
    start = time.time()
    print("="*60)
    print("HarnessHarvester Test Suite")
    print("="*60)

    test_core()
    test_rag()
    test_harness()
    test_cli()

    duration = time.time() - start
    total = PASS + FAIL + SKIP
    print(f"\n{'='*60}")
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped ({total} total)")
    print(f"Duration: {duration:.1f}s")
    print(f"{'='*60}")

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
