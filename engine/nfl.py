"""Macabets NFL Team Power Rating Engine (v0.22)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from engine.confidence import confidence_band, recommendation_from_edge
from engine.nfl_data import NFL_TEAM_RATINGS, TEAM_RATING_WEIGHTS


def american_to_probability(odds: int | float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    return (-odds) / ((-odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def probability_to_american(probability: float) -> int:
    p = min(max(float(probability), 0.01), 0.99)
    return round(-100.0 * p / (1.0 - p)) if p >= 0.5 else round(100.0 * (1.0 - p) / p)


def team_power_score(team: str, overrides: dict | None = None) -> tuple[float, dict]:
    if team not in NFL_TEAM_RATINGS:
        raise ValueError(f"No NFL rating profile exists for {team}.")
    components = dict(NFL_TEAM_RATINGS[team])
    if overrides:
        for key in TEAM_RATING_WEIGHTS:
            if key in overrides:
                components[key] = min(max(float(overrides[key]), 0.0), 100.0)
    raw = sum(components[key] * weight for key, weight in TEAM_RATING_WEIGHTS.items())
    # Convert 0-100 composite to a deliberately compressed NFL point scale.
    power_points = (raw - 82.0) * 0.55
    return round(power_points, 2), components


def spread_to_home_probability(home_margin: float) -> float:
    # Practical logistic approximation: a 3-point favorite is roughly 59%.
    return 1.0 / (1.0 + math.exp(-float(home_margin) / 8.25))


@dataclass(frozen=True)
class NFLAnalysis:
    away_team: str
    home_team: str
    pick: str
    fair_spread_home: float
    fair_moneyline_home: int
    fair_total: float
    home_win_probability: float
    away_win_probability: float
    confidence: float
    confidence_band: str
    projected_away_score: float
    projected_home_score: float
    upset_risk: str
    recommendation: str
    market_edge_points: float
    game_script: str
    why_home_can_win: list[str]
    why_away_can_win: list[str]
    swing_factors: list[str]
    biggest_risk: str
    invalidation_conditions: list[str]
    vegas_difference: str
    foundation_notice: str
    away_power_rating: float
    home_power_rating: float
    home_field_points: float
    rating_breakdown: list[dict]


def analyze(
    *,
    away_team: str,
    home_team: str,
    market_spread_home: float,
    market_moneyline_away: int,
    market_moneyline_home: int,
    market_total: float,
    venue_type: str = "Outdoor",
    weather: str = "Normal",
    neutral_site: bool = False,
    away_rating_overrides: dict | None = None,
    home_rating_overrides: dict | None = None,
    home_field_points: float = 1.7,
) -> dict:
    if away_team == home_team:
        raise ValueError("Home and away teams must be different.")
    if market_total <= 0:
        raise ValueError("Market total must be greater than zero.")

    away_power, away_components = team_power_score(away_team, away_rating_overrides)
    home_power, home_components = team_power_score(home_team, home_rating_overrides)
    applied_hfa = 0.0 if neutral_site else float(home_field_points)

    # Positive projected margin means home team; spread notation is the inverse.
    projected_home_margin = home_power - away_power + applied_hfa
    fair_spread_home = round((-projected_home_margin) * 2.0) / 2.0
    edge = round(float(market_spread_home) - fair_spread_home, 2)

    home_probability = spread_to_home_probability(projected_home_margin)
    away_probability = 1.0 - home_probability
    fair_moneyline_home = probability_to_american(home_probability)

    # Team-quality v0.22 models the side independently; total remains market anchored.
    fair_total = float(market_total)
    projected_home = (fair_total + projected_home_margin) / 2.0
    projected_away = fair_total - projected_home
    projected_home = round(projected_home * 2.0) / 2.0
    projected_away = round(projected_away * 2.0) / 2.0

    if fair_spread_home < 0:
        pick = f"{home_team} {fair_spread_home:+.1f}"
        favorite, underdog = home_team, away_team
    elif fair_spread_home > 0:
        pick = f"{away_team} {-fair_spread_home:+.1f}"
        favorite, underdog = away_team, home_team
    else:
        pick = "No side — pick'em"
        favorite, underdog = home_team, away_team

    rating_gap = abs(projected_home_margin)
    confidence = min(82.0, 52.0 + min(abs(edge), 5.0) * 4.2 + min(rating_gap, 10.0) * 1.0)
    if abs(edge) < 0.75:
        confidence = min(confidence, 58.0)
    confidence = round(confidence, 1)
    recommendation = recommendation_from_edge(edge, confidence)
    upset_risk = "High" if rating_gap < 3 else "Medium" if rating_gap < 7 else "Low"

    market_favorite = home_team if market_spread_home < 0 else away_team if market_spread_home > 0 else "neither team"
    edge_team = home_team if edge < 0 else away_team if edge > 0 else "neither side"

    breakdown = []
    for key, weight in TEAM_RATING_WEIGHTS.items():
        breakdown.append({
            "Category": key.replace("_", " ").title(),
            away_team: round(away_components[key], 1),
            home_team: round(home_components[key], 1),
            "Weight": f"{weight:.0%}",
            "Home advantage": round(home_components[key] - away_components[key], 1),
        })

    result = NFLAnalysis(
        away_team=away_team,
        home_team=home_team,
        pick=pick,
        fair_spread_home=fair_spread_home,
        fair_moneyline_home=fair_moneyline_home,
        fair_total=fair_total,
        home_win_probability=home_probability,
        away_win_probability=away_probability,
        confidence=confidence,
        confidence_band=confidence_band(confidence),
        projected_away_score=projected_away,
        projected_home_score=projected_home,
        upset_risk=upset_risk,
        recommendation=recommendation,
        market_edge_points=edge,
        game_script=(
            f"Macabets rates {favorite} as the stronger team after combining offense, defense, quarterback, "
            f"coaching, schedule and special teams. The independent team-quality line is {home_team} "
            f"{fair_spread_home:+.1f}. {underdog} remains live if it can keep the game out of the favorite's "
            "preferred script and win high-variance possessions."
        ),
        why_home_can_win=[
            f"Team power rating: {home_power:+.2f} points versus {away_team} at {away_power:+.2f}.",
            f"Home-field contribution: {applied_hfa:+.1f} points.",
            f"Macabets projects a {home_probability:.1%} win probability.",
        ],
        why_away_can_win=[
            f"Team power rating: {away_power:+.2f} points versus {home_team} at {home_power:+.2f}.",
            "Turnovers, explosive plays and red-zone execution can overwhelm a modest rating gap.",
            f"Macabets still assigns {away_team} a {away_probability:.1%} win probability.",
        ],
        swing_factors=[
            "Starting-quarterback availability and form",
            "Offensive-line versus pass-rush matchup",
            "Injuries that materially alter a unit rating",
            "Weather, rest and late market movement",
        ],
        biggest_risk=(
            "The v0.22 ratings are editable starter priors rather than an automated weekly statistical feed. "
            "They must be refreshed as injuries, form and roster quality change."
        ),
        invalidation_conditions=[
            "A starting quarterback or major contributor is ruled out",
            "The market moves by 1.5 points or more before kickoff",
            "The entered component ratings no longer reflect current personnel",
        ],
        vegas_difference=(
            f"Vegas currently makes {market_favorite} the market favorite. Macabets makes the fair home line "
            f"{fair_spread_home:+.1f}, creating a {abs(edge):.1f}-point difference toward {edge_team}."
            if abs(edge) >= 0.25 else
            "Macabets and the current market are effectively aligned on the side."
        ),
        foundation_notice=(
            "NFL v0.22 now produces an independent spread from Macabets Team Power Ratings. The component "
            "ratings are editable starter priors; the total remains market-anchored until the scoring model is added."
        ),
        away_power_rating=away_power,
        home_power_rating=home_power,
        home_field_points=applied_hfa,
        rating_breakdown=breakdown,
    )
    return asdict(result)
