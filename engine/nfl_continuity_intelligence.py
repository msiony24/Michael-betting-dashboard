"""NFL continuity and chemistry intelligence.

Audit/context layer only.
Does not directly change team ratings.
Used to measure roster, system, and experience continuity.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ContinuityContext:
    offensive_continuity: float
    defensive_continuity: float
    qb_continuity: float
    ol_continuity: float
    coaching_continuity: float
    confidence_adjustment: float
    notes: list[str]


def build_continuity_context(
    *,
    returning_offensive_starters: float = 0.0,
    returning_defensive_starters: float = 0.0,
    qb_returning: bool = False,
    ol_returning_count: float = 0.0,
    coaching_change: bool = False,
) -> dict[str, Any]:
    offense = float(returning_offensive_starters)
    defense = float(returning_defensive_starters)
    qb = 1.0 if qb_returning else 0.0
    ol = float(ol_returning_count)
    coaching = 0.0 if coaching_change else 1.0

    adjustment = (
        (offense - 0.5) * 0.5
        + (defense - 0.5) * 0.25
        + (qb - 0.5) * 0.25
    )

    return asdict(
        ContinuityContext(
            offensive_continuity=offense,
            defensive_continuity=defense,
            qb_continuity=qb,
            ol_continuity=ol,
            coaching_continuity=coaching,
            confidence_adjustment=round(adjustment, 2),
            notes=[
                "Continuity is a context/confidence layer.",
                "It does not replace talent or matchup ratings.",
            ],
        )
    )
