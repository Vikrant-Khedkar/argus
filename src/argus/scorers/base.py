"""Scorer ABC + plugin registry.

`Scorer` is the universal interface: `(prompt, response) → ScoreResult`. Every
concrete scorer in `argus` — model-based judges, classifiers, regex matchers,
perturbation comparators — implements this.

The registry pattern lets users instantiate scorers from a config (e.g., YAML)
without importing Python classes directly. New scorers register themselves via
the `@register_scorer("name")` decorator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TypeVar

from ..types import ScoreResult

_REGISTRY: dict[str, type["Scorer"]] = {}

ScorerT = TypeVar("ScorerT", bound="Scorer")


def register_scorer(name: str) -> Callable[[type[ScorerT]], type[ScorerT]]:
    """Register a Scorer subclass under a string name for config-driven instantiation.

    Usage:
        @register_scorer("my_safety_classifier")
        class MySafetyClassifier(Scorer):
            ...

        # later, in YAML or programmatically:
        cls = get_scorer_class("my_safety_classifier")
    """

    def decorator(cls: type[ScorerT]) -> type[ScorerT]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"Scorer name '{name}' already registered to {_REGISTRY[name].__name__}"
            )
        _REGISTRY[name] = cls
        cls._registry_name = name  # type: ignore[attr-defined]
        return cls

    return decorator


def get_scorer_class(name: str) -> type["Scorer"]:
    if name not in _REGISTRY:
        raise KeyError(
            f"No scorer registered under '{name}'. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_scorers() -> list[str]:
    return sorted(_REGISTRY.keys())


class Scorer(ABC):
    """Base class. Subclass and implement `score()`."""

    _registry_name: str | None = None

    @abstractmethod
    def score(self, prompt: str, response: str) -> ScoreResult:
        """Score one (prompt, response) tuple. Must return a `ScoreResult`."""

    @property
    def name(self) -> str:
        return self._registry_name or self.__class__.__name__


__all__ = [
    "Scorer",
    "register_scorer",
    "get_scorer_class",
    "list_scorers",
]
