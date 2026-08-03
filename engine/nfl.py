"""Macabets NFL winner-prediction engine v2.

Primary objective: estimate the winner and fair moneyline. Fair spread is a
secondary expression of the same team-strength difference.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from engine.confidence import confidence_band, recommendation_from_edge
from engine.nfl_data import NFL_DATA_STATUS, NFL_TEAM_RATINGS, TEAM_RATING_WEIGHTS
from engine.nfl_brain import build_matchup_brain


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
    # Roughly 2.5 rating points equal one scoreboard point. The scale is
    # intentionally compressed until historical calibration is complete.
    power_points = (raw - 67.5) / 2.5
    return round(power_points, 2), components


def spread_to_home_probability(home_margin: float) -> float:
    # Calibratable logistic mapping. A 3-point favorite is approximately 59%.
    return 1.0 / (1.0 + math.exp(-float(home_margin) / 8.25))


def _market_no_vig_home_probability(away_odds: int, home_odds: int) -> float:
    away = american_to_probability(away_odds)
    home = american_to_probability(home_odds)
    total = away + home
    return home / total if total > 0 else 0.5


def _decisive_factors(away_team: str, home_team: str, away: dict, home: dict) -> list[dict]:
    factors = []
    for category, weight in TEAM_RATING_WEIGHTS.items():
        gap = home[category] - away[category]
        impact = gap * weight
        if abs(impact) < 0.35:
            continue
        leader = home_team if impact > 0 else away_team
        factors.append({
            "category": category.replace("_", " ").title(),
            "leader": leader,
            "rating_gap": round(abs(gap), 1),
            "weighted_impact": round(abs(impact), 2),
            "explanation": f"{leader} owns the stronger {category.replace('_', ' ')} profile by {abs(gap):.1f} rating points.",
        })
    factors.sort(key=lambda item: item["weighted_impact"], reverse=True)
    return factors[:4]


@dataclass(frozen=True)
class NFLAnalysis:
    away_team: str
    home_team: str
    projected_winner: str
    pick: str
    fair_spread_home: float
    fair_moneyline_home: int
    fair_moneyline_away: int
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
    moneyline_edge_home: float
    game_script: str
    decisive_factors: list[dict]
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
    matchup_brain: dict


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

    projected_home_margin = home_power - away_power + applied_hfa
    fair_spread_home = round((-projected_home_margin) * 2.0) / 2.0
    spread_edge = round(float(market_spread_home) - fair_spread_home, 2)

    home_probability = spread_to_home_probability(projected_home_margin)
    away_probability = 1.0 - home_probability
    fair_moneyline_home = probability_to_american(home_probability)
    fair_moneyline_away = probability_to_american(away_probability)
    market_home_probability = _market_no_vig_home_probability(market_moneyline_away, market_moneyline_home)
    moneyline_edge_home = home_probability - market_home_probability

    fair_total = float(market_total)
    projected_home = round(((fair_total + projected_home_margin) / 2.0) * 2.0) / 2.0
    projected_away = round((fair_total - projected_home) * 2.0) / 2.0

    projected_winner = home_team if home_probability >= 0.5 else away_team
    if fair_spread_home < 0:
        pick = f"{home_team} {fair_spread_home:+.1f}"
    elif fair_spread_home > 0:
        pick = f"{away_team} {-fair_spread_home:+.1f}"
    else:
        pick = "No side — pick'em"

    # Confidence reflects model separation and data quality, not market disagreement.
    rating_gap = abs(projected_home_margin)
    confidence = 50.0 + min(rating_gap, 12.0) * 2.2
    if not NFL_DATA_STATUS.get("available"):
        confidence -= 7.0
    confidence = round(min(78.0, max(50.0, confidence)), 1)

    recommendation_edge = moneyline_edge_home if projected_winner == home_team else -moneyline_edge_home
    recommendation = recommendation_from_edge(recommendation_edge * 100.0, confidence)
    upset_risk = "High" if max(home_probability, away_probability) < 0.58 else "Medium" if max(home_probability, away_probability) < 0.68 else "Low"

    decisive = _decisive_factors(away_team, home_team, away_components, home_components)
    matchup_brain = build_matchup_brain(
        away_team=away_team,
        home_team=home_team,
        away_components=away_components,
        home_components=home_components,
    )
    top_reason = matchup_brain["summary"]

    breakdown = []
    for key, weight in TEAM_RATING_WEIGHTS.items():
        breakdown.append({
            "Category": key.replace("_", " ").title(),
            away_team: round(away_components[key], 1),
            home_team: round(home_components[key], 1),
            "Weight": f"{weight:.1%}",
            "Home advantage": round(home_components[key] - away_components[key], 1),
        })

    result = NFLAnalysis(
        away_team=away_team,
        home_team=home_team,
        projected_winner=projected_winner,
        pick=pick,
        fair_spread_home=fair_spread_home,
        fair_moneyline_home=fair_moneyline_home,
        fair_moneyline_away=fair_moneyline_away,
        fair_total=fair_total,
        home_win_probability=home_probability,
        away_win_probability=away_probability,
        confidence=confidence,
        confidence_band=confidence_band(confidence),
        projected_away_score=projected_away,
        projected_home_score=projected_home,
        upset_risk=upset_risk,
        recommendation=recommendation,
        market_edge_points=spread_edge,
        moneyline_edge_home=moneyline_edge_home,
        game_script=(
            f"Macabets projects {projected_winner} to win. {top_reason} "
            f"The model's fair home line is {home_team} {fair_spread_home:+.1f}."
        ),
        decisive_factors=decisive,
        why_home_can_win=[
            f"Macabets assigns {home_team} a {home_probability:.1%} win probability.",
            f"Home-field contribution: {applied_hfa:+.1f} points.",
            *[item["explanation"] for item in decisive if item["leader"] == home_team][:2],
        ],
        why_away_can_win=[
            f"Macabets assigns {away_team} a {away_probability:.1%} win probability.",
            *[item["explanation"] for item in decisive if item["leader"] == away_team][:2],
            "Turnovers and explosive plays can overcome a modest underlying rating gap.",
        ],
        swing_factors=[
            "Starting-quarterback availability and current form",
            "Offensive-line injuries versus the opposing defensive front",
            "Late injury news, rest and weather",
            "Turnover and red-zone variance",
        ],
        biggest_risk=(
            "The weekly snapshot still rates team passing rather than a confirmed starting quarterback, and injury adjustments remain manual."
            if NFL_DATA_STATUS.get("available") else
            "No live weekly snapshot is loaded; the model is using manual priors and neutral recent form."
        ),
        invalidation_conditions=[
            "A starting quarterback or major contributor is ruled out",
            "The market moves materially before kickoff",
            "The entered ratings no longer reflect current personnel",
        ],
        vegas_difference=(
            f"Macabets' no-vig home probability is {home_probability:.1%} versus the market's {market_home_probability:.1%}, "
            f"a {moneyline_edge_home:+.1%} difference."
        ),
        foundation_notice=(
            f"NFL prediction engine v2: {NFL_DATA_STATUS.get('rating_mode', 'team-state priors')}. "
            "Quarterback is the largest component, followed by defense, offensive line, offense and recent form. "
            "The total remains market-anchored."
        ),
        away_power_rating=away_power,
        home_power_rating=home_power,
        home_field_points=applied_hfa,
        rating_breakdown=breakdown,
        matchup_brain=matchup_brain,
    )
    return asdict(result)
