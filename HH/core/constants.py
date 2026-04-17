"""
HarnessHarvester - Global Constants

All uppercase constants used across the system.
Values can be overridden by config.json at runtime.
"""

import os

# ─── Project Paths ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")
DB_ROOT = os.path.join(PROJECT_ROOT, "db")
SANDBOX_ROOT = os.path.join(PROJECT_ROOT, "sandbox")
PROJECTS_ROOT = os.path.join(PROJECT_ROOT, "projects")  # Final deliverables output
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, "KNOWLEDGE_BASE")

# ─── Database Subdirectories ────────────────────────────────────
HARNESSES_DIR = os.path.join(DB_ROOT, "harnesses")
RAG_DIR = os.path.join(DB_ROOT, "rag")
RAG_SNIPPETS_DIR = os.path.join(RAG_DIR, "snippets")
RAG_ERRORS_DIR = os.path.join(RAG_DIR, "errors")
RAG_PROMPTS_DIR = os.path.join(RAG_DIR, "prompts")
METRICS_DIR = os.path.join(DB_ROOT, "metrics")
AUTOLEARN_DIR = os.path.join(DB_ROOT, "autolearn")

# ─── Model Identifiers ─────────────────────────────────────────
MODEL_JUDGE = "hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0"
MODEL_REASONER = "qwen3.5:2b"
MODEL_GENERATOR = "qwen3.5:4b-q4_K_M"

# ─── Model Context Windows ─────────────────────────────────────
CTX_JUDGE = 40000
CTX_REASONER = 240000
CTX_GENERATOR = 32768

# ─── Execution Defaults ────────────────────────────────────────
DEFAULT_EXECUTION_TIMEOUT = 300
DEFAULT_CHECKPOINT_INTERVAL = 60
MAX_REPAIR_ATTEMPTS = 5
MAX_BRANCHES_PER_VERSION = 3
MIN_REVIEW_SCORE = 0.6
AUTO_REPAIR_THRESHOLD = 0.4

# ─── RAG Defaults ──────────────────────────────────────────────
RAG_EMBEDDING_DIM = 512
RAG_MAX_SNIPPETS = 100000
RAG_SEARCH_TOP_K = 10
RAG_MIN_SNIPPET_LENGTH = 20
RAG_MAX_SNIPPET_LENGTH = 50000
BM25_K1 = 1.2
BM25_B = 0.75

# ─── Ranking Weights ───────────────────────────────────────────
RANK_W_TFIDF = 0.30
RANK_W_BM25 = 0.25
RANK_W_QUALITY = 0.20
RANK_W_USAGE = 0.10
RANK_W_RECENCY = 0.10
RANK_W_TAGS = 0.05

# ─── Logging ────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_MAX_SIZE_MB = 10
LOG_BACKUP_COUNT = 5

# ─── File Lock Timeout ──────────────────────────────────────────
LOCK_TIMEOUT = 30
LOCK_RETRY_INTERVAL = 0.1

# ─── Harness Statuses ──────────────────────────────────────────
STATUS_DRAFT = "draft"
STATUS_ACTIVE = "active"
STATUS_FAILED = "failed"
STATUS_REPAIRING = "repairing"
STATUS_ARCHIVED = "archived"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"

# ─── Review Score Categories ───────────────────────────────────
SCORE_FUNCTIONALITY = "functionality"
SCORE_COMPLETION = "completion"
SCORE_PLANNING = "planning_quality"
SCORE_EFFICIENCY = "efficiency"
SCORE_ROBUSTNESS = "robustness"

# ─── RAG Entry Types ───────────────────────────────────────────
RAG_TYPE_SNIPPET = "snippet"
RAG_TYPE_ERROR = "error_pattern"
RAG_TYPE_PROMPT = "prompt_template"

# ─── Orchestrator Stages ───────────────────────────────────────
STAGE_REFLECT = "reflect"
STAGE_BRAINSTORM = "brainstorm"
STAGE_PLAN = "plan"
STAGE_TASK_LIST = "create_task_list"
STAGE_TRACK = "track_progress"
STAGE_EXECUTE = "execute"

ORCHESTRATOR_STAGES = [
    STAGE_REFLECT,
    STAGE_BRAINSTORM,
    STAGE_PLAN,
    STAGE_TASK_LIST,
    STAGE_TRACK,
    STAGE_EXECUTE,
]

# ─── Token Estimation ──────────────────────────────────────────
CHARS_PER_TOKEN = 3.245  # Rough estimate for token counting
