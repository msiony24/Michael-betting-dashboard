"""Macabets NFL foundation engine.

Version 0.1 establishes the matchup contract and report structure. Team-quality,
quarterback, injury, coaching and matchup ratings will be layered into this module
in later versions without changing the UI-facing result keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from engine.confidence import confidence_band, recommendation_from_edge


def american_to_probability(odds: int | float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    return (-odds) / ((-odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def probability_to_american(probability: float) -> int:
    p = min(max(float(probability), 0.01), 0.99)
    return round(-100.0 * p / (1.0 - p)) if p >= 0.5 else round(100.0 * (1.0 - p) / p)


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
) -> dict:
    """Return the stable NFL report schema using market-derived v0.1 baselines.

    The foundation intentionally does not invent team ratings. Until the Team Quality
    Engine is added, the fair line equals the entered market line and confidence remains
    low. This makes the workspace functional without presenting placeholders as a model edge.
    """
    if away_team == home_team:
        raise ValueError("Home and away teams must be different.")
    if market_total <= 0:
        raise ValueError("Market total must be greater than zero.")

    away_raw = american_to_probability(market_moneyline_away)
    home_raw = american_to_probability(market_moneyline_home)
    total_raw = away_raw + home_raw
    home_probability = home_raw / total_raw
    away_probability = away_raw / total_raw

    fair_spread_home = float(market_spread_home)
    fair_total = float(market_total)
    fair_moneyline_home = probability_to_american(home_probability)

    projected_home = (fair_total - fair_spread_home) / 2.0
    projected_away = fair_total - projected_home
    projected_home = round(projected_home * 2.0) / 2.0
    projected_away = round(projected_away * 2.0) / 2.0

    if fair_spread_home < 0:
        pick = f"{home_team} {fair_spread_home:+.1f}"
        favorite = home_team
        underdog = away_team
    elif fair_spread_home > 0:
        away_line = -fair_spread_home
        pick = f"{away_team} {away_line:+.1f}"
        favorite = away_team
        underdog = home_team
    else:
        pick = "No side — pick'em"
        favorite = home_team
        underdog = away_team

    confidence = 45.0
    edge = 0.0
    recommendation = recommendation_from_edge(edge, confidence)
    upset_risk = "High" if abs(fair_spread_home) < 3 else "Medium" if abs(fair_spread_home) < 7 else "Low"

    context_bits = []
    if neutral_site:
        context_bits.append("neutral-site conditions")
    else:
        context_bits.append(f"{home_team}'s home field")
    if venue_type != "Outdoor":
        context_bits.append(venue_type.lower())
    if weather != "Normal":
        context_bits.append(weather.lower())
    context = ", ".join(context_bits)

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
            f"The v0.1 baseline follows the entered market: {favorite} is expected to play from the stronger "
            f"position, with {underdog} remaining live if the game stays within one possession. The current "
            f"projection accounts only for the market and basic setting ({context}); team-specific game-script "
            "logic will be added with the quality and matchup engines."
        ),
        why_home_can_win=[
            f"The market assigns {home_team} a {home_probability:.1%} no-vig win probability.",
            "Home-field and team-specific matchup effects will be quantified in the next engine layer.",
        ],
        why_away_can_win=[
            f"The market assigns {away_team} a {away_probability:.1%} no-vig win probability.",
            "An underdog path remains open through turnovers, explosive plays and red-zone variance.",
        ],
        swing_factors=[
            "Starting quarterback availability and form",
            "Offensive-line versus pass-rush matchup",
            "Turnover and explosive-play differential",
            "Weather and late injury news",
        ],
        biggest_risk="No team-quality data is active yet, so v0.1 cannot establish an independent edge over the market.",
        invalidation_conditions=[
            "A starting quarterback or major contributor is ruled out",
            "The market moves materially before kickoff",
            "Weather changes enough to alter passing or kicking conditions",
        ],
        vegas_difference="None in v0.1. The foundation deliberately uses the market as its baseline until independent ratings are installed.",
        foundation_notice=(
            "NFL v0.1 is the reporting and input foundation. Its numbers are market-derived, not yet an independent "
            "Macabets prediction. Do not treat this version as a betting signal."
        ),
    )
    return asdict(result)
