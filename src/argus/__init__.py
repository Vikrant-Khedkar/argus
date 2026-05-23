"""Argus — AI vendor risk audit SDK.

Public API.
"""

# --- v1 (existing) — kept working via the legacy modules ---------------------
from .adapter import (
    ChatProvider,
    ModalProvider,
    OpenRouterProvider,
    get_provider,
)
from .memory import Conversation
from .risk_score import (
    RiskScorer,
    RiskResult as LegacyRiskResult,  # v1's RiskResult dataclass
    AXES,
    DEFAULT_JUDGE_MODEL,
)
from .assistant import Assistant, load_persona, Guardrails
from .observability import log_row, read_log, new_request_id, LOG_PATH

# --- v2 (new) — pluggable type system + scorer abstractions ------------------
from .types import (
    Instance,
    ScoreResult,
    RiskResult,
    JudgeVerdict,
    MultiJudgeResult,
)
from .tier_mapping import (
    TierMapping,
    InsuranceTierMapping,
    RiskLevelMapping,
    RawScoreMapping,
    GradeLetterMapping,
)
from .axes import AxisSpec, DEFAULT_AXES
from .scorers import (
    Scorer,
    register_scorer,
    get_scorer_class,
    list_scorers,
    CompositeScorer,
    MultiJudgeScorer,
)

__version__ = "0.2.0-dev"

__all__ = [
    # v1
    "ChatProvider",
    "ModalProvider",
    "OpenRouterProvider",
    "get_provider",
    "Conversation",
    "RiskScorer",
    "LegacyRiskResult",
    "AXES",
    "DEFAULT_JUDGE_MODEL",
    "Assistant",
    "Guardrails",
    "load_persona",
    "log_row",
    "read_log",
    "new_request_id",
    "LOG_PATH",
    # v2 types
    "Instance",
    "ScoreResult",
    "RiskResult",
    "JudgeVerdict",
    "MultiJudgeResult",
    # v2 tier mapping
    "TierMapping",
    "InsuranceTierMapping",
    "RiskLevelMapping",
    "RawScoreMapping",
    "GradeLetterMapping",
    # v2 axes
    "AxisSpec",
    "DEFAULT_AXES",
    # v2 scorers
    "Scorer",
    "register_scorer",
    "get_scorer_class",
    "list_scorers",
    "CompositeScorer",
    "MultiJudgeScorer",
    "__version__",
]
