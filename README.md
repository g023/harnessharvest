# HarnessHarvester
## License
MIT
## Author
g023 (https://github.com/g023/harnessharvest/)

A self-learning, self-correcting, LLM-powered harness creation and management system with FAISS-powered RAG, sandboxed execution, and autonomous improvement modes. Powered by Ollama and offline models.

## Overview

**HarnessHarvester** generates executable Python harnesses from natural language task descriptions, executes them in a sandboxed environment, reviews them with multi-faceted LLM judges, and repairs failures using branching strategies. It includes two autonomous modes: **autolearn** (continuous discovery loop) and **autoimprove** (iterative enhancement of existing harnesses). This concept is designed to be an offline first harness/scaffolding builder where you get the harness instead of some remote api. 

**tl;dr**: You give it a task -> it makes a python "harness" to do the task -> it runs the harness with some safety attempted (but you never know so probably run in a container to be safer) -> it reviews how well it did -> if it failed, it tries to fix itself -> successful harnesses get stored in a knowledge base for future generations to learn from -> final deliverables can be found in the ./HH/sandbox/ directory.

---

Usually the normal process is:

1. You run "Write a function that implements binary search on a sorted array"`
2. Some frontend sends it off to some witchcraft in the backend (which happens to most likely be an agentic harness) and after it is completed 
3. you get the deliverable, but do not get the harness.

---

**That's what this project is about**. You get the harness (the agentic code), and the deliverable (the output of the agentic code). 

>> Who knows what kind of potential one might have with different models and different tasks. 

Try it out. Run `python main.py full "Create a text file analyzer that counts words, sentences, and paragraphs"` to run the full pipeline (generate → execute → review → repair) in one command.

## Architecture

```
HH/
├── main.py                  # CLI entry point
├── config.json              # System configuration
├── _new_ollama.py           # Ollama API wrapper (provided)
├── core/                    # Core infrastructure
│   ├── constants.py         # All system constants
│   ├── config.py            # Thread-safe configuration manager
│   ├── storage.py           # Atomic file I/O with locking
│   ├── logging_setup.py     # Rotating log setup
│   └── utils.py             # LLM wrapper, code extraction, JSON parsing
├── rag/                     # FAISS-powered RAG system
│   ├── embeddings.py        # TF-IDF vectorizer + BM25 scorer
│   ├── engine.py            # FAISS vector index + hybrid search engine
│   ├── ranking.py           # Weighted fusion ranking with MMR diversity
│   ├── snippets.py          # Code snippet store
│   ├── errors.py            # Error pattern store
│   ├── prompts.py           # Prompt template store
│   ├── management.py        # Unified RAG manager
│   └── api.py               # HTTP REST API for RAG (port 8420)
├── harness/                 # Harness management
│   ├── models.py            # Data models (metadata, versions, reviews, etc.)
│   ├── orchestrator.py      # Cognitive workflow with stage-gated permissions
│   ├── generator.py         # LLM-based code generation
│   ├── executor.py          # Sandboxed subprocess execution
│   ├── reviewer.py          # Multi-faceted LLM review system
│   ├── repairer.py          # Self-repair with branching strategy
│   └── runtime.py           # Agentic runtime base class
├── modes/                   # Autonomous modes
│   ├── autolearn.py         # Continuous discovery loop
│   └── autoimprove.py       # Iterative harness enhancement
├── TESTS/                   # Test suite (34 tests)
│   └── test_all.py
├── db/                      # Flat-file database
│   ├── harnesses/           # Generated harness storage
│   ├── rag/                 # RAG indexes and entries
│   │   ├── snippets/
│   │   ├── errors/
│   │   └── prompts/
│   ├── metrics/             # Performance metrics
│   └── autolearn/sessions/  # Autolearn session state
├── sandbox/                 # Sandboxed execution directory
├── api/                     # API package
├── logs/                    # Log files
└── projects/                # Project deliverables
```

## Requirements

- Python 3.10+
- Ollama (running locally on port 11434)
- `faiss-cpu` + `numpy`
- `json-repair`
- `requests`

### Setup

```bash
# install dependencies:
pip install faiss-cpu numpy json-repair requests
```

### Ollama Models

Configure models in `config.json`. Defaults:

| Role | Model | Context |
|------|-------|---------|
| Judge | `hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0` | 40,000 |
| Reasoner | `qwen3.5:2b` | 240,000 |
| Generator | `qwen3.5:4b-q4_K_M` | 32,000 |

## Usage

### Generate a Harness

```bash
python main.py harvest "Write a function that implements binary search on a sorted array"
```

### Full Pipeline (Generate → Execute → Review → Repair)

```bash
python main.py full "Create a text file analyzer that counts words, sentences, and paragraphs"
```

### Execute a Harness

```bash
python main.py run <harness_id> --timeout 120
```

### Review a Harness

```bash
python main.py review <harness_id>
```

### Repair a Failed Harness

```bash
python main.py repair <harness_id>
```

### Autolearn Mode

Continuous autonomous discovery loop that generates tasks, creates harnesses, executes, reviews, and self-reflects:

```bash
python main.py autolearn --max-iterations 10
python main.py autolearn --max-duration 3600    # Run for 1 hour
python main.py autolearn --session <session_id> # Resume session
```

### Autoimprove Mode

Analyze and improve an existing harness through iterative enhancement:

```bash
python main.py autoimprove <harness_id> --max-iterations 5
```

### RAG Operations

```bash
python main.py rag search "binary search algorithm"
python main.py rag stats
python main.py rag add --title "My Snippet" --file snippet.py
```

### RAG API Server

```bash
python main.py api --port 8420
```

REST endpoints:
- `GET /health` - Health check
- `GET /snippets/search?q=<query>` - Search snippets
- `POST /snippets` - Add snippet
- `GET /errors/search?q=<query>` - Search error patterns
- `POST /errors` - Add error pattern
- `GET /prompts/search?q=<query>` - Search prompts

### List Harnesses

```bash
python main.py list
python main.py list --status active --source autolearn
```

### Harness Report

```bash
python main.py report <harness_id> --show-code
```

## Key Features

### FAISS-Powered Hybrid Search
Combines TF-IDF cosine similarity, BM25 relevance, FAISS vector search, quality scores, usage frequency, and recency for optimal retrieval.

### Sandboxed Execution
- Path restriction (files confined to sandbox directory)
- Configurable timeouts
- Dangerous operation detection (subprocess, network access)
- Environment variable sanitization

### Checkpointing & Resume
Harnesses can call `checkpoint(state_dict, step_name)` during execution. Failed runs can resume from the last checkpoint.

### Branching Repair Strategy
When a harness fails, the repairer:
1. Analyzes the failure using LLM reasoning
2. Generates 2-3 repair strategies
3. Creates a branch for each strategy
4. Quick-scores each repair
5. Promotes the best-scoring branch

### Cognitive Workflow Orchestrator
Agentic harnesses follow **mandatory stages** with **permission-gated tools**:

| Stage | Permissions |
|-------|------------|
| Reflect | Read-only, LLM, RAG query |
| Brainstorm | Read-only, LLM, RAG query |
| Plan | Read-only, LLM, RAG query |
| Create Task List | Read-only, LLM, Read-write |
| Track Progress | Read-only, Read-write, LLM |
| Execute | All permissions |

### Self-Learning RAG
Successful harnesses are automatically stored in RAG. Error patterns and fixes are captured. Future generations benefit from accumulated knowledge.

## Configuration

All settings in `config.json`:

- **ollama**: Host, models, context windows, option profiles (default/creative/precise/judge)
- **paths**: Database, sandbox, RAG directories
- **sandbox**: Allowed imports, blocked patterns, execution timeout
- **harness**: Max repair attempts, branching limits, auto-repair threshold
- **rag**: Embedding dimensions, ranking weights, BM25 parameters
- **autolearn**: Reflection interval, minimum score thresholds
- **autoimprove**: Iteration limits, improvement thresholds

## Running Tests

```bash
cd HH
python -m pytest TESTS/test_all.py::test_core TESTS/test_all.py::test_rag TESTS/test_all.py::test_harness TESTS/test_all.py::test_cli -v
```

4 test suites covering: core infrastructure, RAG system (embeddings, FAISS, BM25, engine, stores, ranking), harness system (models, orchestrator, executor, sandbox, checkpoints), and CLI.

## Domain Testing Results

The full pipeline (generate → execute → review → repair) has been validated across 4 domains:

| Domain | Task | Score | Repair Needed |
|--------|------|-------|---------------|
| **Game** | Text-based number guessing game | 1.00 | No |
| **Book** | Short story generator (genre + chapters) | 0.50 | Yes (0.00 → 0.50) |
| **Application** | CLI todo list with JSON persistence | 1.00 | No |
| **Website** | Portfolio site generator (HTML/CSS/JS) | 0.90 | No |

## Changelog

### v1.1.0 - Robustness Fixes

**Critical fixes:**
- **autoimprove.py**: Fixed hardcoded version numbers (`"v1"`/`"v2"`) in `_improve_iteration()` — now captures `version_info` from `create_version()` and uses the actual version string
- **engine.py (VectorIndex)**: Fixed `save()` writing `None` entries after `remove()` — now compacts before saving
- **engine.py (VectorIndex)**: Initialized `_needs_rebuild` flag in `__init__` instead of relying on `hasattr()`
- **_new_ollama.py**: Fixed `llm_stream()` unconditionally overriding `num_ctx` with the global 40k constant — now respects per-model context windows passed via options
- **executor.py**: Fixed sandbox wrapper ordering (`os.makedirs` now runs before `os.chdir`)
- **executor.py**: Replaced fragile string-based duration calculation with direct `time.time()` delta
- **repairer.py**: Fixed `_select_repair_model()` referencing undefined config roles (`repair_syntax`, etc.) — now maps to existing `generator`/`reasoner` roles
- **reviewer.py**: Fixed `quick_score()` crash when LLM returns `None` content
- **utils.py**: Fixed file handle leak in `check_user_md()` (added `with` statement)
- **config.py**: Added missing config defaults for `sandbox`, `autolearn`, `autoimprove`, `metrics`, and `rag.ranking_weights` sections

## Storage

All data uses flat-file JSON storage with:
- Atomic writes (temp file + rename) for crash safety
- `fcntl` file locking for concurrent access
- JSONL append for logs and metrics
- Content hashing (SHA-256) for deduplication


