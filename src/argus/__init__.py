"""Argus — AI vendor risk audit SDK.

Public API.
"""

# Provider abstraction
from .adapter import (
    ChatProvider,
    ModalProvider,
    OpenRouterProvider,
    get_provider,
)

# Conversation / memory
from .memory import Conversation

# Risk scoring (Phase 1: legacy 3-axis RiskScorer; Phase 2+ adds pluggable axes)
from .risk_score import (
    RiskScorer,
    RiskResult,
    AXES,
    DEFAULT_JUDGE_MODEL,
)

# Orchestrator
from .assistant import Assistant, load_persona, Guardrails

# Observability
from .observability import log_row, read_log, new_request_id, LOG_PATH

__version__ = "0.2.0-dev"

__all__ = [
    # Providers
    "ChatProvider",
    "ModalProvider",
    "OpenRouterProvider",
    "get_provider",
    # Memory
    "Conversation",
    # Scoring
    "RiskScorer",
    "RiskResult",
    "AXES",
    "DEFAULT_JUDGE_MODEL",
    # Orchestrator
    "Assistant",
    "Guardrails",
    "load_persona",
    # Observability
    "log_row",
    "read_log",
    "new_request_id",
    "LOG_PATH",
    "__version__",
]
