"""IdentityTransform — no-op baseline.

Always included in the default transform set so the unmodified probe runs
alongside the adversarial variants. The audit then shows both raw-prompt
behaviour AND transform-induced failure modes.
"""

from __future__ import annotations

from .base import AttackTransform, register_transform
from ..probes import LiabilityProbe, TransformedProbe


@register_transform("identity")
class IdentityTransform(AttackTransform):
    """Pass the probe through unchanged."""

    def apply(self, probe: LiabilityProbe) -> TransformedProbe:
        return TransformedProbe(
            probe=probe,
            transform_name=self.name,
            transformed_prompt=probe.prompt,
        )


__all__ = ["IdentityTransform"]
