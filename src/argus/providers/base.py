"""ChatProvider ABC + factory.

All scoring and assistant code talks to LLMs through this single interface.
Adding a new provider = subclass `ChatProvider` and implement `chat()`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


class ChatProvider(ABC):
    """Abstract chat-completion provider."""

    name: str

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str: ...

    def model_under_test(self) -> str:
        """Stable identifier for the model this provider serves.

        Recorded on every AuditRow so reviewers can filter / group results
        by which model was actually audited. Subclasses that wrap a known
        model id should override to return ``<provider>:<model>``.
        """
        model = getattr(self, "model", None)
        return f"{self.name}:{model}" if model else self.name


ProviderName = Literal["modal", "openrouter", "http", "huggingface"]


def get_provider(name: ProviderName | None = None, **kwargs) -> ChatProvider:
    """Factory — instantiate a provider by name.

    Defaults to the value of the ``PROVIDER`` env var or "modal" if unset.
    Extra kwargs are forwarded to the provider constructor.
    """
    from .modal import ModalProvider
    from .openrouter import OpenRouterProvider
    from .http import HTTPProvider
    from .huggingface import HuggingFaceProvider

    resolved = (name or os.getenv("PROVIDER", "modal")).lower()
    if resolved == "modal":
        return ModalProvider(**kwargs)
    if resolved == "openrouter":
        return OpenRouterProvider(**kwargs)
    if resolved == "http":
        return HTTPProvider(**kwargs)
    if resolved == "huggingface":
        return HuggingFaceProvider(**kwargs)
    raise ValueError(f"unknown provider: {resolved!r}")


__all__ = ["ChatProvider", "ProviderName", "get_provider"]
