"""Backward-compat shim. Provider classes moved to argus.providers.

For new code, prefer:
    from argus.providers import ChatProvider, ModalProvider, OpenRouterProvider, HTTPProvider, HuggingFaceProvider, GuardrailedProvider
or just:
    from argus import ChatProvider, ModalProvider, OpenRouterProvider, ...
"""

from .providers.base import ChatProvider, ProviderName, get_provider  # noqa: F401
from .providers.modal import ModalProvider, MODAL_URL_DEFAULT  # noqa: F401
from .providers.openrouter import OpenRouterProvider, OPENROUTER_URL  # noqa: F401
from .providers.http import HTTPProvider  # noqa: F401
from .providers.huggingface import HuggingFaceProvider  # noqa: F401
from .providers.guardrailed import GuardrailedProvider  # noqa: F401

__all__ = [
    "ChatProvider",
    "ProviderName",
    "get_provider",
    "ModalProvider",
    "MODAL_URL_DEFAULT",
    "OpenRouterProvider",
    "OPENROUTER_URL",
    "HTTPProvider",
    "HuggingFaceProvider",
    "GuardrailedProvider",
]
