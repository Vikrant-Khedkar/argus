"""AttackTransform ABC + plugin registry.

Transforms take a `LiabilityProbe` and produce a `TransformedProbe`. The
output is what actually gets sent to the model under test.

Two flavours:
- Single-turn transforms wrap the probe's prompt in some attack pattern
  (translation-laundering, persona-swap, typos, paraphrase).
- Multi-turn transforms produce a sequence of message turns
  (gradual jailbreak escalation, multi-turn tool chain, etc.).

The procedural multiplier: one new transform × N existing probes = N new
test cases without any new probe curation. This is Argus's biggest leverage
for safety coverage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TypeVar

from ..probes import LiabilityProbe, TransformedProbe

_REGISTRY: dict[str, type["AttackTransform"]] = {}

TransformT = TypeVar("TransformT", bound="AttackTransform")


def register_transform(name: str) -> Callable[[type[TransformT]], type[TransformT]]:
    """Register an AttackTransform subclass under a string name."""

    def decorator(cls: type[TransformT]) -> type[TransformT]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"Transform name '{name}' already registered to {_REGISTRY[name].__name__}"
            )
        _REGISTRY[name] = cls
        cls._registry_name = name  # type: ignore[attr-defined]
        return cls

    return decorator


def get_transform_class(name: str) -> type["AttackTransform"]:
    if name not in _REGISTRY:
        raise KeyError(
            f"No transform registered under '{name}'. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_transforms() -> list[str]:
    return sorted(_REGISTRY.keys())


class AttackTransform(ABC):
    """Base class — subclass and implement ``apply()``."""

    _registry_name: str | None = None

    @abstractmethod
    def apply(self, probe: LiabilityProbe) -> TransformedProbe:
        """Apply this attack pattern to a probe."""

    @property
    def name(self) -> str:
        return self._registry_name or self.__class__.__name__


__all__ = [
    "AttackTransform",
    "register_transform",
    "get_transform_class",
    "list_transforms",
]
