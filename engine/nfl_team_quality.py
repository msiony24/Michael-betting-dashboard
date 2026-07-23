from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class TeamQualityInputs:
    quarterback: float
    offense: float
    defense: float
    coaching: float
    offensive_line: float
    defensive_line: float
    skill_positions: float
    secondary: float
    special_teams: float
    continuity: float
    injury_adjustment: float = 0.0
    rookie_adjustment: float = 0.0


@dataclass(frozen=True)
class TeamQualityResult:
    team: str
    base_rating: float
    injury_adjustment: float
    rookie_adjustment: float
    final_rating: float
    component_scores: Dict[str, float]


WEIGHTS: Mapping[str, float] = {
    "quarterback": 0.22,
    "offense": 0.15,
    "defense": 0.15,
    "coaching": 0.10,
    "offensive_line": 0.09,
    "defensive_line": 0.09,
    "skill_positions": 0.07,
    "secondary": 0.07,
    "special_teams": 0.03,
    "continuity": 0.03,
}


def _validate_score(name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")

    if value < 0 or value > 100:
        raise ValueError(f"{name} must be between 0 and 100.")


def calculate_team_quality(
    team: str,
    inputs: TeamQualityInputs,
) -> TeamQualityResult:
    """
    Calculate a team's overall quality rating on a 0-100 scale.
    """

    if not isinstance(team, str) or not team.strip():
        raise ValueError("team must be a non-empty string.")

    component_scores = {
        "quarterback": float(inputs.quarterback),
        "offense": float(inputs.offense),
        "defense": float(inputs.defense),
        "coaching": float(inputs.coaching),
        "offensive_line": float(inputs.offensive_line),
        "defensive_line": float(inputs.defensive_line),
        "skill_positions": float(inputs.skill_positions),
        "secondary": float(inputs.secondary),
        "special_teams": float(inputs.special_teams),
        "continuity": float(inputs.continuity),
    }

    for name, value in component_scores.items():
        _validate_score(name, value)

    base_rating = sum(
        component_scores[name] * weight
        for name, weight in WEIGHTS.items()
    )

    final_rating = (
        base_rating
        + float(inputs.injury_adjustment)
        + float(inputs.rookie_adjustment)
    )

    final_rating = max(0.0, min(100.0, final_rating))

    return TeamQualityResult(
        team=team.strip(),
        base_rating=round(base_rating, 2),
        injury_adjustment=round(float(inputs.injury_adjustment), 2),
        rookie_adjustment=round(float(inputs.rookie_adjustment), 2),
        final_rating=round(final_rating, 2),
        component_scores={
            name: round(value, 2)
            for name, value in component_scores.items()
        },
    )


def compare_team_quality(
    away_team: TeamQualityResult,
    home_team: TeamQualityResult,
    home_field_advantage: float = 1.5,
    points_per_rating_point: float = 0.55,
) -> Dict[str, float | str]:
    """
    Compare two teams and estimate the home team's fair spread.

    Positive fair_spread_home:
        Home team should be favored.

    Negative fair_spread_home:
        Away team should be favored.
    """

    adjusted_home_rating = (
        home_team.final_rating + float(home_field_advantage)
    )

    rating_difference = (
        adjusted_home_rating - away_team.final_rating
    )

    fair_spread_home = (
        rating_difference * float(points_per_rating_point)
    )

    if fair_spread_home > 0:
        favored_team = home_team.team
    elif fair_spread_home < 0:
        favored_team = away_team.team
    else:
        favored_team = "Pick'em"

    return {
        "away_team": away_team.team,
        "home_team": home_team.team,
        "away_rating": away_team.final_rating,
        "home_rating": home_team.final_rating,
        "home_field_advantage": round(
            float(home_field_advantage), 2
        ),
        "adjusted_home_rating": round(adjusted_home_rating, 2),
        "rating_difference": round(rating_difference, 2),
        "fair_spread_home": round(fair_spread_home, 1),
        "favored_team": favored_team,
    }
