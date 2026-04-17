"""
HarnessHarvester - Harness Data Models

Dataclass models for harnesses, versions, reviews, checkpoints, and branches.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime

from core.constants import (
    STATUS_DRAFT, STATUS_ACTIVE, STATUS_FAILED,
    SCORE_FUNCTIONALITY, SCORE_COMPLETION, SCORE_PLANNING,
    SCORE_EFFICIENCY, SCORE_ROBUSTNESS,
)


@dataclass
class HarnessMetadata:
    """Core metadata for a harness."""
    id: str
    task_description: str
    status: str = STATUS_DRAFT
    is_agentic: bool = False
    created_at: str = ""
    updated_at: str = ""
    active_version: str = "v1"
    active_branch: str = "main"
    total_versions: int = 1
    total_runs: int = 0
    best_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    source: str = "user"  # user, autolearn, autoimprove

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "HarnessMetadata":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class VersionInfo:
    """Information about a specific version/branch of a harness."""
    version: str  # e.g., "v1", "v2"
    branch: str = "main"  # e.g., "main", "repair_1", "autoimprove_1"
    parent_version: str = ""
    parent_branch: str = ""
    repair_strategy: str = ""  # repair strategy used to create this version
    model_used: str = ""
    created_at: str = ""
    status: str = STATUS_DRAFT
    review_score: float = 0.0
    run_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "VersionInfo":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class VersionIndex:
    """Tracks all versions and branches for a harness."""
    harness_id: str
    versions: Dict[str, VersionInfo] = field(default_factory=dict)
    # Key format: "v1/main", "v2/repair_1", etc.

    def add_version(self, version_info: VersionInfo) -> str:
        key = f"{version_info.version}/{version_info.branch}"
        self.versions[key] = version_info
        return key

    def get_version(self, version: str, branch: str = "main") -> Optional[VersionInfo]:
        key = f"{version}/{branch}"
        return self.versions.get(key)

    def get_best_version(self) -> Optional[VersionInfo]:
        if not self.versions:
            return None
        return max(self.versions.values(), key=lambda v: v.review_score)

    def get_branches(self, version: str) -> List[VersionInfo]:
        return [v for k, v in self.versions.items() if k.startswith(f"{version}/")]

    def to_dict(self) -> Dict:
        return {
            "harness_id": self.harness_id,
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VersionIndex":
        idx = cls(harness_id=data.get("harness_id", ""))
        for key, vdata in data.get("versions", {}).items():
            idx.versions[key] = VersionInfo.from_dict(vdata)
        return idx


@dataclass
class ReviewReport:
    """LLM review of a harness version."""
    harness_id: str
    version: str
    branch: str = "main"
    reviewer_model: str = ""
    scores: Dict[str, float] = field(default_factory=lambda: {
        SCORE_FUNCTIONALITY: 0.0,
        SCORE_COMPLETION: 0.0,
        SCORE_PLANNING: 0.0,
        SCORE_EFFICIENCY: 0.0,
        SCORE_ROBUSTNESS: 0.0,
    })
    overall_score: float = 0.0
    critique: str = ""
    suggestions: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    is_agentic_review: bool = False
    reviewed_at: str = ""

    @property
    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["average_score"] = self.average_score
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "ReviewReport":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ExecutionLog:
    """Log of a harness execution run."""
    harness_id: str
    version: str
    branch: str = "main"
    run_id: str = ""
    status: str = ""  # running, completed, failed, timeout
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    error_traceback: str = ""
    checkpoints: List[Dict] = field(default_factory=list)
    last_checkpoint: str = ""
    resumed_from: str = ""  # checkpoint ID if resumed
    # Agentic harness fields
    deliverables: List[Dict] = field(default_factory=list)  # Files created by agentic harness
    llm_calls: int = 0  # Number of LLM calls made by harness
    workflow_stage: str = ""  # Current/final workflow stage
    progress_percent: float = 0.0  # Progress percentage

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ExecutionLog":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class CheckpointState:
    """Captured state at a checkpoint during execution."""
    checkpoint_id: str
    harness_id: str
    run_id: str
    step_index: int = 0
    step_name: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0  # 0.0 to 1.0
    created_at: str = ""
    version_hash: str = ""  # Hash of harness code at checkpoint time

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "CheckpointState":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class RepairAttempt:
    """Record of a repair attempt on a harness."""
    harness_id: str
    source_version: str
    source_branch: str
    target_version: str
    target_branch: str
    repair_strategy: str  # syntax_fix, reasoning_fix, structural_fix
    model_used: str
    error_info: str = ""
    checkpoint_used: str = ""  # checkpoint ID used for context
    success: bool = False
    score_before: float = 0.0
    score_after: float = 0.0
    created_at: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "RepairAttempt":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
