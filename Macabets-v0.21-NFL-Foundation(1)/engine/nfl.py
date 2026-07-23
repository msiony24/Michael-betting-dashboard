"""Macabets NFL foundation engine.

This release creates an independent matchup framework without pretending that a
complete data pipeline already exists. Manual grades are explicit and auditable.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .confidence import clamp, confidence_score
from .nfl_data import HOME_FIELD_POINTS, TEAM_PRIORS


def _american_from_probability(probability: float) -> int:
    p = clamp(float(probability), 0.001, 0.999)
    if p >= 0.5:
        return -round(100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def _moneyline_probability_from_spread(spread_margin: float) -> float:
    # A conservative logistic conversion calibrated only for a first-pass framework.
    return clamp(1 / (1 + math.exp(-spread_margin / 6.5)), 0.02, 0.98)


def analyze_nfl_matchup(
    away_team: str,
    home_team: str,
    market_spread_home: float,
    market_total: float,
    venue_context: str,
    team_quality_home: float,
    team_quality_away: float,
    quarterback_home: float,
    quarterback_away: float,
    matchup_home: float,
    matchup_away: float,
    coaching_home: float,
    coaching_away: float,
    situational_home: float,
    situational_away: float,
    injury_adjustment_home: float = 0.0,
    injury_adjustment_away: float = 0.0,
    weather_total_adjustment: float = 0.0,
    data_quality: float = 4.0,
) -> Dict[str, object]:
    if away_team == home_team:
        raise ValueError("Select two different NFL teams.")

    weights = {
        "Team quality": 0.34,
        "Quarterback": 0.25,
        "Matchup": 0.18,
        "Coaching": 0.11,
        "Situational context": 0.12,
    }
    home_scores = {
        "Team quality": float(team_quality_home),
        "Quarterback": float(quarterback_home),
        "Matchup": float(matchup_home),
        "Coaching": float(coaching_home),
        "Situational context": float(situational_home),
    }
    away_scores = {
        "Team quality": float(team_quality_away),
        "Quarterback": float(quarterback_away),
        "Matchup": float(matchup_away),
        "Coaching": float(coaching_away),
        "Situational context": float(situational_away),
    }

    weighted_grade_diff = sum(
        weights[key] * (home_scores[key] - away_scores[key]) for key in weights
    )
    # Each net grade point is worth roughly 1.1 scoreboard points in this foundation.
    home_field = HOME_FIELD_POINTS.get(venue_context, 1.7)
    prior_diff = TEAM_PRIORS.get(home_team, 0.0) - TEAM_PRIORS.get(away_team, 0.0)
    fair_margin_home = (
        1.1 * weighted_grade_diff
        + home_field
        + prior_diff
        + float(injury_adjustment_home)
        - float(injury_adjustment_away)
    )

    fair_spread_home = -fair_margin_home
    fair_total = clamp(float(market_total) + float(weather_total_adjustment), 28.0, 70.0)
    home_score = fair_total / 2 + fair_margin_home / 2
    away_score = fair_total / 2 - fair_margin_home / 2
    home_probability = _moneyline_probability_from_spread(fair_margin_home)
    away_probability = 1 - home_probability

    market_margin_home = -float(market_spread_home)
    edge_points_home = fair_margin_home - market_margin_home
    agreement = clamp(10 - abs(edge_points_home) * 1.2, 1, 10)
    uncertainty = clamp(10 - float(data_quality), 0, 10)
    confidence = confidence_score(float(data_quality), agreement, uncertainty)

    if abs(edge_points_home) < 1.0:
        recommendation = "PASS"
        recommendation_reason = "Macabets does not show at least one full point of spread value."
    elif edge_points_home > 0:
        recommendation = f"LEAN {home_team}"
        recommendation_reason = f"Macabets makes {home_team} {abs(edge_points_home):.1f} points stronger than the market."
    else:
        recommendation = f"LEAN {away_team}"
        recommendation_reason = f"Macabets makes {away_team} {abs(edge_points_home):.1f} points stronger than the market."

    factor_rows: List[Dict[str, float]] = []
    for name, weight in weights.items():
        difference = home_scores[name] - away_scores[name]
        factor_rows.append({
            "factor": name,
            "home_grade": home_scores[name],
            "away_grade": away_scores[name],
            "difference": difference,
            "weighted_impact": difference * weight * 1.1,
        })
    factor_rows.sort(key=lambda row: abs(row["weighted_impact"]), reverse=True)

    favorite = home_team if fair_margin_home >= 0 else away_team
    underdog = away_team if favorite == home_team else home_team
    projected_script = (
        f"{favorite} is projected to play from ahead and force {underdog} into a more predictable game script."
        if abs(fair_margin_home) >= 3
        else "Macabets projects a one-score game in which turnovers, red-zone execution and late-down decisions carry extra weight."
    )

    return {
        "away_team": away_team,
        "home_team": home_team,
        "fair_margin_home": fair_margin_home,
        "fair_spread_home": fair_spread_home,
        "fair_total": fair_total,
        "projected_home_score": home_score,
        "projected_away_score": away_score,
        "home_win_probability": home_probability,
        "away_win_probability": away_probability,
        "fair_moneyline_home": _american_from_probability(home_probability),
        "fair_moneyline_away": _american_from_probability(away_probability),
        "market_spread_home": float(market_spread_home),
        "edge_points_home": edge_points_home,
        "confidence": confidence,
        "data_quality": int(round(data_quality)),
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "projected_script": projected_script,
        "favorite": favorite,
        "underdog": underdog,
        "factor_rows": factor_rows,
        "upset_risk": "High" if abs(fair_margin_home) < 3 else "Moderate" if abs(fair_margin_home) < 7 else "Lower",
        "foundation_warning": (
            "Foundation model: matchup grades are manually entered. Do not treat this as a fully automated NFL edge until the Team Quality and data pipelines are built."
        ),
    }
