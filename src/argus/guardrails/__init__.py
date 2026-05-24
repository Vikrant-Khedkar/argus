"""Pre-flight and post-flight guards for `GuardrailedProvider`.

Pre-flight runs before the model call:
- `PreFlightPatternGuard` — regex jailbreak patterns; short-circuits with a
  refusal if matched.

Post-flight runs after the model call:
- `PostFlightRegenGuard` — scores the response via `RiskScorer`; if any axis
  scores below threshold, regenerates with a strengthened system prompt.
- `PostFlightHardRefuseGuard` — replaces unsafe responses with a fixed
  refusal string when a critical axis fails.
"""

from .preflight import PreFlightPatternGuard, PreFlightClassifierGuard
from .postflight import PostFlightRegenGuard, PostFlightHardRefuseGuard
from .embedding_guard import (
    PreFlightEmbeddingGuard,
    FailIndex,
    refresh_fail_index,
    DEFAULT_EMBED_MODEL,
)

__all__ = [
    "PreFlightPatternGuard",
    "PreFlightClassifierGuard",
    "PreFlightEmbeddingGuard",
    "FailIndex",
    "refresh_fail_index",
    "DEFAULT_EMBED_MODEL",
    "PostFlightRegenGuard",
    "PostFlightHardRefuseGuard",
]
