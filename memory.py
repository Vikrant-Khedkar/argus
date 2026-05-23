"""Backward-compat shim. Argus v2 moved this module into the `argus` package.

For new code, prefer:
    from argus import Conversation
"""

from argus.memory import *  # noqa: F401, F403
from argus.memory import Conversation  # noqa: F401
