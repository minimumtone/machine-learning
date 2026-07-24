"""状態定義（指示書 §6）。"""

from enum import Enum


class AgentState(str, Enum):
    IDLE = "idle"
    OBSERVING = "observing"
    PLANNING = "planning"
    AWAITING_HUMAN_INPUT = "awaiting_human_input"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    EVALUATING = "evaluating"
    REPLANNING = "replanning"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskState(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending"
    READY = "ready"
    AWAITING_APPROVAL = "awaiting_approval"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class HypothesisState(str, Enum):
    DRAFT = "draft"
    PROPOSED = "proposed"
    HUMAN_REVIEWED = "human_reviewed"
    APPROVED_FOR_TESTING = "approved_for_testing"
    UNDER_EVALUATION = "under_evaluation"
    SUPPORTED = "supported"
    FALSIFIED = "falsified"
    CONDITIONALLY_SUPPORTED = "conditionally_supported"
    INCONCLUSIVE = "inconclusive"
    REJECTED_BY_HUMAN = "rejected_by_human"
    ARCHIVED = "archived"


class ErrorType(str, Enum):
    VALIDATION_ERROR = "validation_error"
    SCHEMA_MISMATCH = "schema_mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    MISSING_INPUT = "missing_input"
    MODEL_LOAD_ERROR = "model_load_error"
    MODEL_RUNTIME_ERROR = "model_runtime_error"
    OUT_OF_MEMORY = "out_of_memory"
    TIMEOUT = "timeout"
    PERMISSION_ERROR = "permission_error"
    NETWORK_ERROR = "network_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    OUT_OF_DOMAIN = "out_of_domain"
    PHYSICAL_CONSTRAINT_VIOLATION = "physical_constraint_violation"
    UNKNOWN_ERROR = "unknown_error"


# 自動修正可能なエラー種別（§9.2）
AUTO_RECOVERABLE_ERRORS = {
    ErrorType.UNIT_MISMATCH,
    ErrorType.SCHEMA_MISMATCH,
    ErrorType.NETWORK_ERROR,
    ErrorType.TIMEOUT,
}

# 人間確認が必要なエラー種別（§9.3）
HUMAN_REVIEW_ERRORS = {
    ErrorType.OUT_OF_DOMAIN,
    ErrorType.PERMISSION_ERROR,
    ErrorType.PHYSICAL_CONSTRAINT_VIOLATION,
    ErrorType.MISSING_INPUT,
    ErrorType.UNKNOWN_ERROR,
}
