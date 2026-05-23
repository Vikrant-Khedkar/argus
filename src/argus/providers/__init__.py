"""Providers — anything that wraps an LLM behind the ChatProvider interface.

Pluggable: subclass ChatProvider, implement chat(). Anyone can wrap a custom
endpoint in HTTPProvider or build a bespoke ChatProvider subclass.
"""

from .base import ChatProvider, ProviderName, get_provider
from .modal import ModalProvider, MODAL_URL_DEFAULT
from .openrouter import OpenRouterProvider, OPENROUTER_URL
from .http import HTTPProvider
from .huggingface import HuggingFaceProvider
from .guardrailed import GuardrailedProvider

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
