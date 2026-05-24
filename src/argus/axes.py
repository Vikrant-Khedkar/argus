"""AxisSpec — declarative description of one liability axis.

Each axis bundles together:
  - a human name + description
  - the scorer that produces scores for this axis
  - the insurance coverage categories this axis maps to
  - the TierMapping used to render presentation tiers from raw scores
  - a weight used when aggregating across axes for an overall verdict

`DEFAULT_AXES` is populated downstream — once concrete scorer subclasses exist
(Phase 4), this module imports them and assembles the default 7-axis set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .tier_mapping import TierMapping, InsuranceTierMapping

if TYPE_CHECKING:
    from .scorers.base import Scorer


@dataclass
class AxisSpec:
    name: str
    description: str
    scorer: "Scorer"
    coverage_alignment: list[str] = field(default_factory=list)
    tier_mapping: TierMapping = field(default_factory=InsuranceTierMapping)
    weight: float = 1.0
    severity_floor: str | None = None  # optional: only run probes >= this severity


# Populated in Phase 4 once concrete scorers exist
DEFAULT_AXES: list[AxisSpec] = []


__all__ = ["AxisSpec", "DEFAULT_AXES"]
