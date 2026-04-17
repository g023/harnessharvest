"""
HarnessHarvester - Harness Generator

Generates executable harness code from natural language task descriptions
using LLM models. Handles code extraction, validation, and versioning.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from core.constants import (
    HARNESSES_DIR, STATUS_DRAFT, STATUS_ACTIVE,
    MODEL_GENERATOR,
)
from core.config import config
from core.storage import (
    atomic_write, atomic_write_json, read_json,
    locked_update_json, ensure_dir, generate_id, content_hash,
)
from core.logging_setup import get_logger
from core.utils import (
    llm_call, extract_primary_code, extract_imports,
    validate_imports, now_iso, summarize_for_context,
    estimate_tokens,
)
from harness.models import (
    HarnessMetadata, VersionInfo, VersionIndex,
)
from harness.orchestrator import detect_agentic_harness

logger = get_logger("generator")


# ─── Prompt Templates ───────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = """You are an expert Python code generator for the HarnessHarvester system.
Generate complete, executable Python code based on the given task description.

Requirements:
- Write clean, well-structured Python 3.10+ code
- Include proper error handling
- Add docstrings and type hints
- Use only standard library modules unless specifically requested
- The code should be self-contained and runnable
- For complex tasks, structure code with classes and functions
- Include a main() function or if __name__ == '__main__' block

For agentic tasks:
- Define an Orchestrator class that follows the cognitive workflow pattern
- Include reflect(), brainstorm(), plan(), create_task_list(), track_progress(), execute() methods
- Use tool decorators for actions the agent can take

Output the complete code inside a ```python code block."""


# ─── Agentic Harness Prompt ────────────────────────────────────

AGENTIC_SYSTEM_PROMPT = """You are an expert Python developer creating an AGENTIC HARNESS for the HarnessHarvester system.

CRITICAL: The AgenticHarness base class is ALREADY IMPORTED. DO NOT add any import statements for it.
The following are ALREADY available in the global namespace:
- AgenticHarness (base class to extend)
- WorkflowStage (enum)
- WORKSPACE, PROJECT_OUTPUT, TASK, HARNESS_ID, RUN_ID (globals)
- _LLM_CONFIG (config dict)
- rag_search_snippets(query, top_k=5) - Search code snippets
- rag_search_errors(query, top_k=5) - Search error patterns and solutions
- rag_search_prompts(query, top_k=3) - Search successful prompts

Your code MUST NOT include:
- `from agentic_harness import ...`
- `from harness.runtime import ...`
- `import agentic_harness`

Your code MUST:
1. Define a class that extends AgenticHarness directly
2. Override stage methods: research(), analyze(), plan(), develop(), test(), finalize()
3. Use self.llm(prompt, system=None) to make LLM calls
4. Use self.write_file(path, content) to create deliverables in WORKSPACE
5. For final output files, also copy to PROJECT_OUTPUT + "/deliverables/"
6. Instantiate the class and call run()

The AgenticHarness base class provides:
- self.llm(prompt, system=None) → {"content": str, "reasoning": str}
- self.write_file(path, content) → Creates files in workspace
- self.read_file(path) → Reads files from workspace  
- self.context → Dict to store data between stages
- self.task_description → The task to accomplish
- self.report_progress(percent, message) → Report progress
- self.log(message) → Log messages

Output the complete code inside a ```python code block."""


AGENTIC_USER_TEMPLATE = """Create an agentic harness for this task:

{task_description}

{context}

IMPORTANT: AgenticHarness is ALREADY imported. DO NOT add import statements for it.

CRITICAL: self.llm() returns a dict with "content" key that is a STRING, not a nested dict.
- CORRECT: result["content"] gives you the text string directly
- WRONG: result["content"]["content"] will crash (string indices error)

AVAILABLE RAG FUNCTIONS (for knowledge retrieval):
- rag_search_snippets(query, top_k=5) - Find relevant code patterns
- rag_search_errors(query, top_k=5) - Find error solutions
- rag_search_prompts(query, top_k=3) - Find successful prompts

FILE OUTPUT:
- WORKSPACE = working directory for all file operations
- PROJECT_OUTPUT = final deliverables directory (copy important files here in finalize())
- Use shutil.copy2() to copy files to PROJECT_OUTPUT + "/deliverables/"

Here is the required structure:
```python
# DO NOT add any imports for AgenticHarness - it's already available!
import shutil  # For copying files to project output

class MyHarness(AgenticHarness):
    def research(self):
        # Research phase - gather information
        # Optionally use RAG: snippets = rag_search_snippets("relevant topic")
        result = self.llm("What do I need to know about this task?")
        self.context["research"] = result["content"]
        self.report_progress(15, "Research complete")
    
    def analyze(self):
        # Analyze requirements
        self.context["analysis"] = "Task analysis..."
        self.report_progress(25, "Analysis complete")
    
    def plan(self):
        # Create execution plan
        self.context["plan"] = ["step1", "step2"]
        self.report_progress(35, "Planning complete")
    
    def develop(self):
        # Create deliverables - write files to workspace
        result = self.llm("Generate the content for the file")
        self.write_file("output.txt", result["content"])
        self.report_progress(70, "Development complete")
    
    def test(self):
        # Verify deliverables
        self.report_progress(85, "Testing complete")
    
    def finalize(self):
        # Copy important deliverables to project output
        import os
        for f in os.listdir(WORKSPACE):
            if not f.startswith("_"):  # Skip internal files
                src = os.path.join(WORKSPACE, f)
                dst = os.path.join(PROJECT_OUTPUT, "deliverables", f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        self.report_progress(95, "Finalizing - deliverables copied to project output")

# Create and run the harness
harness = MyHarness(
    task_description=TASK,
    workspace_path=WORKSPACE,
    harness_id=HARNESS_ID,
    run_id=RUN_ID,
    config=_LLM_CONFIG,
)
result = harness.run()
print(f"Harness completed with status: {{result}}")
```

Generate the complete harness class for the given task. Remember: 
1. NO imports for AgenticHarness!
2. result["content"] IS the text string - don't use result["content"]["content"]!
3. Copy deliverables to PROJECT_OUTPUT in finalize()!
4. DO NOT embed large multi-line string literals in the code - use self.llm() to generate content!
5. Keep harness code clean and simple - let LLM generate file content at runtime!"""


GENERATION_USER_TEMPLATE = """Generate Python code for the following task:

{task_description}

{context}

Write complete, runnable Python code."""


class HarnessGenerator:
    """
    Generates harness code from natural language descriptions.
    Uses LLM to produce code, validates it, and saves as versioned harness.
    """

    def __init__(self):
        self.model = config.get_model("generator")
        self.harnesses_dir = config.get_path("harnesses_dir")
        ensure_dir(self.harnesses_dir)

    def generate(
        self,
        task_description: str,
        context: str = "",
        model: str = None,
        options_profile: str = "creative",
        tags: Optional[List[str]] = None,
        source: str = "user",
        agentic: bool = True,  # Default to agentic mode
    ) -> Tuple[str, HarnessMetadata]:
        """
        Generate a new harness from a task description.

        Args:
            task_description: Natural language task description
            context: Additional context for generation
            model: Override model (optional)
            options_profile: LLM options profile
            tags: Tags for the harness
            source: Source of the request (user, autolearn, etc.)
            agentic: If True, generate an agentic harness that extends AgenticHarness

        Returns:
            (harness_id, metadata)
        """
        effective_model = model or self.model

        # Check RAG for relevant prompts and snippets
        rag_context = self._get_rag_context(task_description)
        if rag_context:
            context = f"{context}\n\nRelevant snippets from knowledge base:\n{rag_context}"

        # Truncate context to fit model limits
        model_ctx = config.get_model_context(effective_model)
        max_prompt_tokens = int(model_ctx * 0.5)  # Leave half for output
        context = summarize_for_context(context, max_prompt_tokens)

        # Select prompt based on mode
        if agentic:
            system_prompt = AGENTIC_SYSTEM_PROMPT
            user_prompt = AGENTIC_USER_TEMPLATE.format(
                task_description=task_description,
                context=context if context else "(No additional context)",
            )
        else:
            system_prompt = GENERATION_SYSTEM_PROMPT
            user_prompt = GENERATION_USER_TEMPLATE.format(
                task_description=task_description,
                context=context if context else "(No additional context)",
            )

        options = config.get_options_for_model(effective_model, options_profile)

        logger.info(f"Generating {'agentic' if agentic else 'simple'} harness for: {task_description[:100]}...")

        # LLM call
        result = llm_call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=effective_model,
            options=options,
        )

        code = extract_primary_code(result.get("content", ""))
        if not code:
            logger.error("No code extracted from LLM response")
            code = result.get("content", "# No code generated")

        # Post-process agentic code to remove incorrect imports
        if agentic:
            code = self._clean_agentic_code(code)

        # Validate
        is_valid, bad_imports = validate_imports(code)
        if not is_valid:
            logger.warning(f"Code uses disallowed imports: {bad_imports}")

        # Detect if agentic (or force based on parameter)
        is_agentic = agentic or detect_agentic_harness(code)

        # Create harness
        harness_id = generate_id("hrn")
        harness_dir = os.path.join(self.harnesses_dir, harness_id)
        version_dir = os.path.join(harness_dir, "versions", "v1")
        ensure_dir(version_dir)

        # Save code
        code_path = os.path.join(version_dir, "harness.py")
        atomic_write(code_path, code)

        # Create metadata
        now = now_iso()
        metadata = HarnessMetadata(
            id=harness_id,
            task_description=task_description,
            status=STATUS_DRAFT,
            is_agentic=is_agentic,
            created_at=now,
            updated_at=now,
            tags=tags or [],
            source=source,
        )

        # Create version info
        version_info = VersionInfo(
            version="v1",
            branch="main",
            model_used=effective_model,
            created_at=now,
            status=STATUS_DRAFT,
        )

        # Create version index
        version_index = VersionIndex(harness_id=harness_id)
        version_index.add_version(version_info)

        # Save all metadata
        atomic_write_json(
            os.path.join(harness_dir, "metadata.json"),
            metadata.to_dict(),
        )
        atomic_write_json(
            os.path.join(harness_dir, "version_index.json"),
            version_index.to_dict(),
        )
        atomic_write_json(
            os.path.join(version_dir, "metadata.json"),
            version_info.to_dict(),
        )
        atomic_write_json(
            os.path.join(version_dir, "generation_info.json"),
            {
                "model": effective_model,
                "options_profile": options_profile,
                "task_description": task_description,
                "context_provided": bool(context),
                "code_length": len(code),
                "is_agentic": is_agentic,
                "imports": extract_imports(code),
                "import_validation": {"valid": is_valid, "disallowed": bad_imports},
                "reasoning": result.get("reasoning", "")[:2000],
                "time_taken": result.get("time_taken", 0),
                "usage": result.get("usage", {}),
                "generated_at": now,
            },
        )

        # Update master index
        self._update_master_index(harness_id, metadata)

        logger.info(
            f"Harness generated: {harness_id} "
            f"(agentic={is_agentic}, lines={len(code.splitlines())})"
        )

        # Store successful prompt in RAG
        self._store_prompt_in_rag(task_description, effective_model, code)

        return harness_id, metadata

    def create_version(
        self,
        harness_id: str,
        code: str,
        branch: str = "main",
        parent_version: str = "",
        parent_branch: str = "main",
        model_used: str = "",
        repair_strategy: str = "",
    ) -> VersionInfo:
        """Create a new version of an existing harness."""
        harness_dir = os.path.join(self.harnesses_dir, harness_id)
        if not os.path.isdir(harness_dir):
            raise FileNotFoundError(f"Harness {harness_id} not found")

        # Load version index
        vi_path = os.path.join(harness_dir, "version_index.json")
        vi_data = read_json(vi_path, {"harness_id": harness_id, "versions": {}})
        version_index = VersionIndex.from_dict(vi_data)

        # Determine next version number
        existing_versions = set()
        for key in version_index.versions:
            v = key.split("/")[0]
            existing_versions.add(v)
        next_num = len(existing_versions) + 1
        new_version = f"v{next_num}"

        # Create version directory
        version_dir = os.path.join(harness_dir, "versions", f"{new_version}_{branch}")
        ensure_dir(version_dir)

        # Save code
        atomic_write(os.path.join(version_dir, "harness.py"), code)

        # Create version info
        now = now_iso()
        version_info = VersionInfo(
            version=new_version,
            branch=branch,
            parent_version=parent_version,
            parent_branch=parent_branch,
            repair_strategy=repair_strategy,
            model_used=model_used,
            created_at=now,
            status=STATUS_DRAFT,
        )

        # Update version index
        version_index.add_version(version_info)
        atomic_write_json(vi_path, version_index.to_dict())
        atomic_write_json(
            os.path.join(version_dir, "metadata.json"),
            version_info.to_dict(),
        )

        # Update harness metadata
        meta_path = os.path.join(harness_dir, "metadata.json")
        meta_data = read_json(meta_path, {})
        meta_data["total_versions"] = len(version_index.versions)
        meta_data["updated_at"] = now
        atomic_write_json(meta_path, meta_data)

        logger.info(f"Version created: {harness_id} {new_version}/{branch}")
        return version_info

    def get_harness_code(
        self,
        harness_id: str,
        version: str = None,
        branch: str = "main",
    ) -> str:
        """Read harness code for a specific version."""
        harness_dir = os.path.join(self.harnesses_dir, harness_id)

        if version is None:
            meta = read_json(os.path.join(harness_dir, "metadata.json"), {})
            version = meta.get("active_version", "v1")
            branch = meta.get("active_branch", "main")

        # Try exact directory name
        version_dir = os.path.join(harness_dir, "versions", f"{version}_{branch}")
        if not os.path.isdir(version_dir):
            version_dir = os.path.join(harness_dir, "versions", version)

        code_path = os.path.join(version_dir, "harness.py")
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                return f.read()
        raise FileNotFoundError(f"Harness code not found: {harness_id} {version}/{branch}")

    def list_harnesses(
        self,
        status: str = None,
        source: str = None,
        limit: int = 50,
    ) -> List[HarnessMetadata]:
        """List all harnesses with optional filtering."""
        index_path = os.path.join(self.harnesses_dir, "index.json")
        index = read_json(index_path, {"harnesses": {}})

        harnesses = []
        for hid, meta_brief in index.get("harnesses", {}).items():
            if status and meta_brief.get("status") != status:
                continue
            if source and meta_brief.get("source") != source:
                continue
            # Load full metadata
            meta_path = os.path.join(self.harnesses_dir, hid, "metadata.json")
            meta_data = read_json(meta_path)
            if meta_data:
                harnesses.append(HarnessMetadata.from_dict(meta_data))

        harnesses.sort(key=lambda h: h.updated_at or "", reverse=True)
        return harnesses[:limit]

    def get_metadata(self, harness_id: str) -> Optional[HarnessMetadata]:
        """Get full metadata for a harness."""
        meta_path = os.path.join(self.harnesses_dir, harness_id, "metadata.json")
        data = read_json(meta_path)
        if data:
            return HarnessMetadata.from_dict(data)
        return None

    # ── Private Helpers ──────────────────────────────────────

    def _update_master_index(self, harness_id: str, metadata: HarnessMetadata):
        """Update the master harness index."""
        index_path = os.path.join(self.harnesses_dir, "index.json")

        def updater(data):
            if data is None:
                data = {"harnesses": {}}
            if "harnesses" not in data:
                data["harnesses"] = {}
            data["harnesses"][harness_id] = {
                "task": metadata.task_description[:200],
                "status": metadata.status,
                "is_agentic": metadata.is_agentic,
                "source": metadata.source,
                "created_at": metadata.created_at,
                "updated_at": metadata.updated_at,
            }
            return data

        locked_update_json(index_path, updater, {"harnesses": {}})

    def _clean_agentic_code(self, code: str) -> str:
        """
        Post-process agentic harness code to remove incorrect imports
        and validate syntax.
        The AgenticHarness class is injected by the runtime, so imports are not needed.
        """
        lines = code.split('\n')
        cleaned_lines = []
        
        # Patterns to remove - these are injected by the runtime
        bad_patterns = [
            'from agentic_harness import',
            'from harness.runtime import',
            'import agentic_harness',
            'import harness.runtime',
        ]
        
        for line in lines:
            stripped = line.strip()
            # Check if this line matches any bad pattern
            should_remove = False
            for pattern in bad_patterns:
                if stripped.startswith(pattern):
                    should_remove = True
                    break
            
            if not should_remove:
                cleaned_lines.append(line)
            else:
                # Add a comment explaining the removal
                indent = len(line) - len(line.lstrip())
                cleaned_lines.append(' ' * indent + '# (Runtime injects AgenticHarness automatically)')
        
        cleaned_code = '\n'.join(cleaned_lines)
        
        # Validate syntax - check for unterminated strings
        try:
            compile(cleaned_code, '<generated_harness>', 'exec')
        except SyntaxError as e:
            # Log the error but continue - execution will catch it with better context
            logger.warning(f"Generated harness has syntax error: {e}")
        
        return cleaned_code

    def _get_rag_context(self, task_description: str) -> str:
        """Query RAG for relevant snippets and prompts."""
        try:
            from rag.snippets import SnippetStore
            from rag.prompts import PromptStore

            snippets = SnippetStore()
            results = snippets.search(task_description, top_k=3, min_quality=0.3)

            context_parts = []
            for r in results:
                if r.final_score > 0.2:
                    context_parts.append(
                        f"# Snippet: {r.entry.title}\n{r.entry.content[:500]}"
                    )
                    snippets.record_usage(r.entry.id)

            return "\n\n".join(context_parts) if context_parts else ""
        except Exception:
            return ""

    def _store_prompt_in_rag(self, task_description: str, model: str, code: str):
        """Store the generation prompt in RAG for future reference."""
        try:
            from rag.prompts import PromptStore
            store = PromptStore()
            store.add_prompt_template(
                template=GENERATION_USER_TEMPLATE.format(
                    task_description="{task_description}",
                    context="{context}",
                ),
                task_type="code_generation",
                model_used=model,
                description=f"Generated code for: {task_description[:100]}",
                tags=["generation", "code"],
                source="generator",
                quality_score=0.5,
            )
        except Exception:
            pass
