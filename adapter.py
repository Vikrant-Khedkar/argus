"""Backward-compat shim. Argus v2 moved this module into the `argus` package.

For new code, prefer:
    from argus import ChatProvider, ModalProvider, OpenRouterProvider, get_provider
"""

from argus.adapter import *  # noqa: F401, F403
from argus.adapter import (  # noqa: F401
    ChatProvider,
    ModalProvider,
    OpenRouterProvider,
    get_provider,
    MODAL_URL_DEFAULT,
    OPENROUTER_URL,
)
