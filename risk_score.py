"""Backward-compat shim. Argus v2 moved this module into the `argus` package.

For new code, prefer:
    from argus import RiskScorer, RiskResult
"""

from argus.risk_score import *  # noqa: F401, F403
from argus.risk_score import RiskScorer, RiskResult, AXES, DEFAULT_JUDGE_MODEL  # noqa: F401
