"""Stable NFL team-rating schema for the Macabets football brain.

The schema separates football reasoning from the source of the ratings. Today it
can be populated from Macabets' legacy team components. Later the same fields
can be filled by Madden 27, advanced metrics, depth charts, injuries, or a
blended ratings model without rewriting the exploit engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping


DEFAULT_RATING = 67.5


def _rating(source: Mapping[str, float], key: str, fallback: float = DEFAULT_RATING) -> float:
    try:
        value = float(source.get(key, fallback))
    except (TypeError, ValueError):
        value = float(fallback)
    return round(min(100.0, max(0.0, value)), 2)


def _blend(*pairs: tuple[float, float]) -> float:
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return DEFAULT_RATING
    return round(sum(value * weight for value, weight in pairs) / total_weight, 2)


@dataclass(frozen=True)
class OffenseRatings:
    quarterback: float
    pass_protection: float
    run_blocking: float
    receiving_weapons: float
    running_backs: float
    overall: float


@dataclass(frozen=True)
class DefenseRatings:
    pass_rush: float
    run_defense: float
    linebacker_coverage: float
    cornerbacks: float
    safeties: float
    overall: float


@dataclass(frozen=True)
class NFLTeamProfile:
    team: str
    offense: OffenseRatings
    defense: DefenseRatings
    coaching: float
    depth: float
    continuity: float
    special_teams: float
    source: str = "legacy_adapter"
    data_quality: str = "provisional"
    available_fields: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


def profile_from_legacy_components(
    team: str,
    components: Mapping[str, float],
) -> NFLTeamProfile:
    """Translate today's broad Macabets components into the stable brain schema.

    The adapter is intentionally explicit. Fields derived from broad legacy
    categories are provisional estimates, not claims that Macabets already has
    true player-level or scheme-level data.
    """

    quarterback = _rating(components, "quarterback")
    offense_overall = _rating(components, "offense")
    defense_overall = _rating(components, "defense")
    offensive_line = _rating(components, "offensive_line")
    defensive_line = _rating(components, "defensive_line")
    skill_positions = _rating(components, "skill_positions")
    secondary = _rating(components, "secondary")
    coaching = _rating(components, "coaching")
    continuity = _rating(components, "continuity")
    special_teams = _rating(components, "special_teams")

    offense = OffenseRatings(
        quarterback=quarterback,
        pass_protection=offensive_line,
        run_blocking=_blend((offensive_line, 0.72), (offense_overall, 0.28)),
        receiving_weapons=_blend((skill_positions, 0.78), (offense_overall, 0.22)),
        running_backs=_blend((skill_positions, 0.62), (offense_overall, 0.38)),
        overall=offense_overall,
    )

    defense = DefenseRatings(
        pass_rush=defensive_line,
        run_defense=_blend((defensive_line, 0.68), (defense_overall, 0.32)),
        linebacker_coverage=_blend((defense_overall, 0.55), (secondary, 0.45)),
        cornerbacks=secondary,
        safeties=_blend((secondary, 0.72), (defense_overall, 0.28)),
        overall=defense_overall,
    )

    depth = _blend(
        (offense_overall, 0.25),
        (defense_overall, 0.35),
        (skill_positions, 0.15),
        (continuity, 0.25),
    )

    present = tuple(
        sorted(
            key
            for key in (
                "quarterback",
                "offense",
                "defense",
                "coaching",
                "offensive_line",
                "defensive_line",
                "skill_positions",
                "secondary",
                "special_teams",
                "continuity",
            )
            if key in components
        )
    )

    quality = "complete_legacy" if len(present) == 10 else "partial_legacy"

    return NFLTeamProfile(
        team=team,
        offense=offense,
        defense=defense,
        coaching=coaching,
        depth=depth,
        continuity=continuity,
        special_teams=special_teams,
        source="legacy_components_adapter",
        data_quality=quality,
        available_fields=present,
    )


SCHEMA_FIELDS = {
    "offense": (
        "quarterback",
        "pass_protection",
        "run_blocking",
        "receiving_weapons",
        "running_backs",
        "overall",
    ),
    "defense": (
        "pass_rush",
        "run_defense",
        "linebacker_coverage",
        "cornerbacks",
        "safeties",
        "overall",
    ),
    "team": ("coaching", "depth", "continuity", "special_teams"),
}
