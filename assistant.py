"""Backward-compat shim. Argus v2 moved this module into the `argus` package.

For new code, prefer:
    from argus import Assistant, Guardrails, load_persona
"""

from argus.assistant import *  # noqa: F401, F403
from argus.assistant import Assistant, Guardrails, load_persona  # noqa: F401
