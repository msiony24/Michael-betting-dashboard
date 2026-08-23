"""NFL turnover regression intelligence (audit/context layer).

Does not directly change predictions. Tracks turnover sustainability so
Macabets can distinguish repeatable defensive/offensive performance from
high-variance turnover results.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class TurnoverContext:
    team: str
    turnover_margin: float
    forced_turnovers: float
    giveaways: float
    takeaways: float
    expected_turnover_margin: float
    regression_amount: float
    sustainability: str


def estimate_sustainability(
    turnover_margin: float,
    forced_turnovers: float,
    takeaways: float,
    giveaways: float,
) -> TurnoverContext:
    # Conservative: turnover data affects confidence/context first,
    # not the core spread calculation.
    expected = float(forced_turnovers) - float(giveaways)
    regression = float(turnover_margin) - expected

    if abs(regression) <= 2:
        label = "Sustainable"
    elif regression > 2:
        label = "Positive turnover regression risk"
    else:
        label = "Negative turnover regression risk"

    return TurnoverContext(
        team="",
        turnover_margin=float(turnover_margin),
        forced_turnovers=float(forced_turnovers),
        giveaways=float(giveaways),
        takeaways=float(takeaways),
        expected_turnover_margin=expected,
        regression_amount=regression,
        sustainability=label,
    )


def build_turnover_context(team_stats: dict[str, Any]) -> dict[str, Any]:
    context = estimate_sustainability(
        team_stats.get("turnover_margin", 0),
        team_stats.get("forced_turnovers", 0),
        team_stats.get("takeaways", 0),
        team_stats.get("giveaways", 0),
    )
    return asdict(context)
