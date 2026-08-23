"""NFL red zone and third down intelligence (context only).

Audit/model context layer. Designed to influence confidence and close-game
interpretation without replacing the core team power model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class SituationalEfficiency:
    team: str
    red_zone_td_rate: float
    red_zone_td_allowed: float
    third_down_rate: float
    third_down_allowed: float
    execution_score: float


def build_efficiency_score(
    red_zone_td_rate: float,
    red_zone_td_allowed: float,
    third_down_rate: float,
    third_down_allowed: float,
) -> float:
    offense = (red_zone_td_rate - 0.55) * 40
    defense = (0.55 - red_zone_td_allowed) * 40
    third = ((third_down_rate - third_down_allowed) * 25)
    return round(max(-10, min(10, offense + defense + third)), 2)


def compare_close_game_execution(home: dict, away: dict) -> dict:
    edge = float(home.get("execution_score", 0)) - float(away.get("execution_score", 0))
    return {
        "execution_edge": round(edge, 2),
        "leader": "home" if edge > 0 else "away" if edge < 0 else "even",
        "confidence_context": (
            "supports the projected winner in a close matchup"
            if abs(edge) >= 2
            else "does not create a meaningful separation"
        ),
    }
