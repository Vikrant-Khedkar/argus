"""Argus — AI vendor risk audit SDK.

Public API.
"""

# --- v1 (existing) — kept working via the legacy modules ---------------------
from .providers import (
    ChatProvider,
    ProviderName,
    get_provider,
    ModalProvider,
    OpenRouterProvider,
    HTTPProvider,
    HuggingFaceProvider,
    GuardrailedProvider,
)
from .guardrails import (
    PreFlightPatternGuard,
    PreFlightClassifierGuard,
    PreFlightEmbeddingGuard,
    FailIndex,
    refresh_fail_index,
    PostFlightRegenGuard,
    PostFlightHardRefuseGuard,
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
from .probes import LiabilityProbe, TransformedProbe
from .conversation import (
    ConversationTurn,
    ConversationProbe,
    TurnResult,
    ConversationResult,
    score_conversation,
    DEFAULT_MULTI_TURN_PROBES,
)
from .storage import AuditWriter, AuditRow, AuditIndex, AuditReport
from .eval_config import EvalConfig
from .evaluator import Evaluator
from .datasets import (
    load_factual_probes,
    load_bias_probes,
    load_safety_probes,
    load_toxicity_probes,
    load_calibration_probes,
    load_pii_probes,
    default_probe_set,
    get_axis_loader,
)
from .transforms import (
    AttackTransform,
    register_transform,
    get_transform_class,
    list_transforms,
    IdentityTransform,
    TranslationLaunderingTransform,
    PersonaSwapTransform,
    MultiTurnEscalationTransform,
    TyposTransform,
    CaseChangeTransform,
    ParaphraseTransform,
    DEFAULT_TRANSFORMS,
)
from .scorers import (
    Scorer,
    register_scorer,
    get_scorer_class,
    list_scorers,
    ModelBasedScorer,
    CompositeScorer,
    MultiJudgeScorer,
    LLMJudgeScorer,
    LlamaGuardScorer,
    LlamaPromptGuardScorer,
    PerspectiveAPIScorer,
    PresidioScorer,
    ExactMatchPIIScorer,
    RefusalRegexScorer,
    KeyFactsScorer,
    HedgePhraseScorer,
)

__version__ = "0.2.0-dev"

__all__ = [
    # Providers
    "ChatProvider",
    "ProviderName",
    "get_provider",
    "ModalProvider",
    "OpenRouterProvider",
    "HTTPProvider",
    "HuggingFaceProvider",
    "GuardrailedProvider",
    # Guardrails
    "PreFlightPatternGuard",
    "PreFlightClassifierGuard",
    "PreFlightEmbeddingGuard",
    "FailIndex",
    "refresh_fail_index",
    "PostFlightRegenGuard",
    "PostFlightHardRefuseGuard",
    # v1 legacy
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
    "ModelBasedScorer",
    "CompositeScorer",
    "MultiJudgeScorer",
    "LLMJudgeScorer",
    "LlamaGuardScorer",
    "LlamaPromptGuardScorer",
    "PerspectiveAPIScorer",
    "PresidioScorer",
    "ExactMatchPIIScorer",
    "RefusalRegexScorer",
    "KeyFactsScorer",
    "HedgePhraseScorer",
    # v2 probes + transforms
    "LiabilityProbe",
    "TransformedProbe",
    "AttackTransform",
    "register_transform",
    "get_transform_class",
    "list_transforms",
    "IdentityTransform",
    "TranslationLaunderingTransform",
    "PersonaSwapTransform",
    "MultiTurnEscalationTransform",
    "TyposTransform",
    "CaseChangeTransform",
    "ParaphraseTransform",
    "DEFAULT_TRANSFORMS",
    # v2 multi-turn
    "ConversationTurn",
    "ConversationProbe",
    "TurnResult",
    "ConversationResult",
    "score_conversation",
    "DEFAULT_MULTI_TURN_PROBES",
    # v2 storage / reporting
    "AuditWriter",
    "AuditRow",
    "AuditIndex",
    "AuditReport",
    # v2 orchestration
    "EvalConfig",
    "Evaluator",
    # v2 datasets
    "load_factual_probes",
    "load_bias_probes",
    "load_safety_probes",
    "load_toxicity_probes",
    "load_calibration_probes",
    "load_pii_probes",
    "default_probe_set",
    "get_axis_loader",
    "__version__",
]
