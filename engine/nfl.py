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
from engine.nfl_schedule_intelligence import build_schedule_context
from engine.nfl_game_quality import build_game_quality_context
from engine.nfl_personnel_matchup import build_personnel_matchup_context
from engine.nfl_matchup_intelligence import build_matchup_intelligence
from engine.nfl_scheme_tendencies import build_scheme_matchup_context
from engine.nfl_los_intelligence import build_los_matchup_context
from engine.nfl_situational_intelligence import build_situational_matchup_context
from engine.nfl_opponent_adjustment import build_opponent_adjusted_context
from engine.nfl_simulation import simulate_game


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




def _safe_num(value, default=0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _build_expected_game_script(
    *,
    away_team: str,
    home_team: str,
    projected_winner: str,
    projected_home_margin: float,
    fair_total: float,
    projected_away_score: float,
    projected_home_score: float,
    matchup_intelligence: dict,
    scheme_context: dict,
    los_context: dict,
    situational_context: dict,
    opponent_context: dict,
    simulation_context: dict,
) -> str:
    """Turn the model's existing matchup evidence into an actual game-flow forecast.

    This is presentation only: it does not create a new adjustment or feed back
    into the projection. Every statement is derived from signals that have
    already been scored elsewhere in the NFL engine.
    """
    favorite = projected_winner
    underdog = away_team if favorite == home_team else home_team
    margin = abs(float(projected_home_margin))
    one_score = _safe_num(simulation_context.get("one_score_probability"), 0.0)
    upset = _safe_num(simulation_context.get("upset_probability"), 0.0)
    volatility = str(simulation_context.get("volatility") or "Moderate").lower()

    if fair_total <= 41.5:
        scoring_shape = "a lower-scoring game where possessions carry extra weight"
    elif fair_total >= 50.0:
        scoring_shape = "a higher-scoring game with more room for explosive swings"
    else:
        scoring_shape = "a balanced scoring environment"

    if margin < 3.0 or one_score >= 0.55:
        closeness = "a one-score game deep into the fourth quarter"
    elif margin < 6.0 or one_score >= 0.42:
        closeness = "a competitive game that is likely to stay within reach into the second half"
    elif margin >= 7.0:
        closeness = f"{favorite} creating separation if its early advantages hold"
    else:
        closeness = "a competitive game with a modest chance of late separation"

    # Offensive identity / pace comes from the scheme layer when available.
    pace_bits = []
    if scheme_context.get("available"):
        for team_name, profile in ((away_team, scheme_context.get("away") or {}), (home_team, scheme_context.get("home") or {})):
            edp = profile.get("early_down_pass_rate")
            spp = profile.get("seconds_per_play")
            identity = None
            try:
                edp = float(edp)
                identity = "pass-leaning" if edp >= 0.62 else "run-leaning" if edp <= 0.50 else "balanced"
            except (TypeError, ValueError):
                pass
            pace = None
            try:
                spp = float(spp)
                pace = "faster tempo" if spp < 27.0 else "slower tempo" if spp > 31.0 else "average tempo"
            except (TypeError, ValueError):
                pass
            if identity and pace:
                pace_bits.append(f"{team_name} projects as {identity} at {pace}")
            elif identity:
                pace_bits.append(f"{team_name} projects as {identity}")
    opening = (
        f"Macabets expects {scoring_shape}, with {closeness}. "
        + (("; ".join(pace_bits[:2]) + ". ") if pace_bits else "")
    )

    # Pull the strongest already-scored football drivers for each side.
    top_drivers = list(matchup_intelligence.get("top_drivers") or [])
    fav_drivers = [str(d.get("factor")) for d in top_drivers if str(d.get("leader")) == favorite][:2]
    dog_drivers = [str(d.get("factor")) for d in top_drivers if str(d.get("leader")) == underdog][:2]

    extra_contexts = [
        ("player-style matchup", matchup_intelligence.get("overall_style_advantage")),
        ("scheme fit", scheme_context.get("overall_advantage") if scheme_context.get("available") else None),
        ("line-of-scrimmage matchup", los_context.get("overall_advantage") if los_context.get("available") else None),
        ("situational execution", situational_context.get("overall_advantage") if situational_context.get("available") else None),
        ("opponent-adjusted performance", opponent_context.get("overall_advantage") if opponent_context.get("available") else None),
    ]
    fav_context = [label for label, leader in extra_contexts if leader == favorite]
    dog_context = [label for label, leader in extra_contexts if leader == underdog]

    fav_reasons = fav_drivers + [x for x in fav_context if x not in fav_drivers]
    dog_reasons = dog_drivers + [x for x in dog_context if x not in dog_drivers]
    fav_reason_text = ", ".join(fav_reasons[:3]) if fav_reasons else "the stronger overall matchup profile"
    dog_reason_text = ", ".join(dog_reasons[:2]) if dog_reasons else "creating turnovers or explosive plays"

    favorite_path = (
        f"**Most likely winning path — {favorite}:** control the areas where Macabets sees the clearest edge — "
        f"{fav_reason_text} — stay on schedule, and avoid giving the underdog short fields."
    )
    upset_path = (
        f"**Upset path — {underdog}:** lean into {dog_reason_text}, create an early swing play, and force {favorite} "
        f"out of its preferred script. The simulation still gives the underdog roughly {upset:.0%} of outcomes."
    )

    if margin < 3.0 or one_score >= 0.50:
        finish = (
            f"**Most likely finish:** Macabets sees this as close late, with {favorite} holding the better chance to execute the final decisive possessions. "
            f"The center of the projection is {away_team} {projected_away_score:.1f} – {home_team} {projected_home_score:.1f}, with {volatility} simulation volatility."
        )
    else:
        finish = (
            f"**Most likely finish:** if {favorite} gets ahead and keeps its matchup advantages intact, it has the better path to create separation in the second half. "
            f"The center of the projection is {away_team} {projected_away_score:.1f} – {home_team} {projected_home_score:.1f}, with {volatility} simulation volatility."
        )

    return opening + "\n\n" + favorite_path + "\n\n" + upset_path + "\n\n" + finish

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
    weather_context: dict
    schedule_context: dict
    game_quality_context: dict
    personnel_context: dict
    scheme_context: dict
    los_context: dict
    situational_context: dict
    opponent_adjusted_context: dict
    simulation_context: dict
    matchup_intelligence: dict


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
    weather_context: dict | None = None,
    game_date=None,
    week: int | None = None,
    season: int | None = None,
) -> dict:
    if away_team == home_team:
        raise ValueError("Home and away teams must be different.")
    if market_total <= 0:
        raise ValueError("Market total must be greater than zero.")

    away_power, away_components = team_power_score(away_team, away_rating_overrides)
    home_power, home_components = team_power_score(home_team, home_rating_overrides)

    resolved_season = int(season or (getattr(game_date, "year", None) or NFL_DATA_STATUS.get("season") or 2026))
    power_map = {}
    for team_name in NFL_TEAM_RATINGS:
        try:
            power_map[team_name] = team_power_score(team_name)[0]
        except Exception:
            continue
    # Ensure any manually overridden matchup ratings are represented in the
    # opponent-quality scale used for this game.
    power_map[away_team] = away_power
    power_map[home_team] = home_power
    schedule_context = build_schedule_context(
        away_team=away_team, home_team=home_team, season=resolved_season,
        team_power=power_map, game_date=game_date, week=week,
    )
    scheduled_neutral = bool(schedule_context.get("scheduled_neutral", False))
    applied_hfa = 0.0 if (neutral_site or scheduled_neutral) else float(home_field_points)
    schedule_side_adjustment = float(schedule_context.get("home_margin_adjustment", 0.0) or 0.0)
    schedule_confidence_penalty = float(schedule_context.get("confidence_penalty", 0.0) or 0.0)
    game_quality_context = build_game_quality_context(
        away_team=away_team, home_team=home_team, season=resolved_season, game_date=game_date,
    )
    game_quality_side_adjustment = float(game_quality_context.get("home_margin_adjustment", 0.0) or 0.0)
    game_quality_confidence_penalty = float(game_quality_context.get("confidence_penalty", 0.0) or 0.0)
    weather_context = dict(weather_context or {})
    weather_side_adjustment = float(weather_context.get("home_margin_adjustment", 0.0) or 0.0)
    weather_total_adjustment = float(weather_context.get("total_adjustment", 0.0) or 0.0)
    weather_confidence_penalty = float(weather_context.get("confidence_penalty", 0.0) or 0.0)

    personnel_context = build_personnel_matchup_context(
        away_team=away_team,
        home_team=home_team,
        week=week,
    )
    scheme_context = build_scheme_matchup_context(
        away_team=away_team,
        home_team=home_team,
        season=resolved_season,
        week=week,
        personnel_context=personnel_context,
    )
    scheme_side_adjustment = float(scheme_context.get("home_margin_adjustment", 0.0) or 0.0)
    los_context = build_los_matchup_context(
        away_team=away_team, home_team=home_team, season=resolved_season, week=week,
    )
    los_side_adjustment = float(los_context.get("home_margin_adjustment", 0.0) or 0.0)
    situational_context = build_situational_matchup_context(
        away_team=away_team, home_team=home_team, season=resolved_season, week=week,
    )
    situational_side_adjustment = float(situational_context.get("home_margin_adjustment", 0.0) or 0.0)
    opponent_adjusted_context = build_opponent_adjusted_context(
        away_team=away_team, home_team=home_team, season=resolved_season, week=week,
    )
    opponent_adjusted_side_adjustment = float(opponent_adjusted_context.get("home_margin_adjustment", 0.0) or 0.0)
    matchup_intelligence = build_matchup_intelligence(
        away_team=away_team,
        home_team=home_team,
        away_components=away_components,
        home_components=home_components,
        away_power=away_power,
        home_power=home_power,
        home_field_points=applied_hfa,
        weather_home_adjustment=weather_side_adjustment,
        schedule_home_adjustment=schedule_side_adjustment,
        game_quality_home_adjustment=game_quality_side_adjustment,
        scheme_home_adjustment=scheme_side_adjustment,
        los_home_adjustment=los_side_adjustment,
        situational_home_adjustment=situational_side_adjustment,
        opponent_adjusted_home_adjustment=opponent_adjusted_side_adjustment,
        personnel_context=personnel_context,
    )
    personnel_side_adjustment = float(matchup_intelligence.get("matchup_adjustment_home", 0.0) or 0.0)
    projected_home_margin = float(matchup_intelligence.get("football_home_edge_points", 0.0) or 0.0)
    fair_spread_home = round((-projected_home_margin) * 2.0) / 2.0
    spread_edge = round(float(market_spread_home) - fair_spread_home, 2)

    home_probability = spread_to_home_probability(projected_home_margin)
    away_probability = 1.0 - home_probability
    fair_moneyline_home = probability_to_american(home_probability)
    fair_moneyline_away = probability_to_american(away_probability)
    market_home_probability = _market_no_vig_home_probability(market_moneyline_away, market_moneyline_home)
    moneyline_edge_home = home_probability - market_home_probability

    fair_total = max(1.0, float(market_total) + weather_total_adjustment)
    projected_home = round(((fair_total + projected_home_margin) / 2.0) * 2.0) / 2.0
    projected_away = round((fair_total - projected_home) * 2.0) / 2.0

    simulation_context = simulate_game(
        away_team=away_team,
        home_team=home_team,
        projected_home_margin=projected_home_margin,
        fair_total=fair_total,
        market_spread_home=market_spread_home,
        market_total=market_total,
        seed_context=f"{resolved_season}:{week or 0}:{getattr(game_date, 'isoformat', lambda: game_date)() if game_date is not None else ''}",
    )

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
    confidence -= weather_confidence_penalty
    confidence -= schedule_confidence_penalty
    confidence -= game_quality_confidence_penalty
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
        unified_context=matchup_intelligence,
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
        game_script=_build_expected_game_script(
            away_team=away_team,
            home_team=home_team,
            projected_winner=projected_winner,
            projected_home_margin=projected_home_margin,
            fair_total=fair_total,
            projected_away_score=projected_away,
            projected_home_score=projected_home,
            matchup_intelligence=matchup_intelligence,
            scheme_context=scheme_context,
            los_context=los_context,
            situational_context=situational_context,
            opponent_context=opponent_adjusted_context,
            simulation_context=simulation_context,
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
            "Whether recent final scores match the teams' underlying play quality",
        ],
        biggest_risk=(
            "Questionable/Doubtful availability can still change before kickoff; definitive Sleeper Out/IR/PUP statuses automatically activate the next Footballguys depth-chart option."
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
        weather_context=weather_context,
        schedule_context=schedule_context,
        game_quality_context=game_quality_context,
        personnel_context=personnel_context,
        scheme_context=scheme_context,
        los_context=los_context,
        situational_context=situational_context,
        opponent_adjusted_context=opponent_adjusted_context,
        simulation_context=simulation_context,
        matchup_intelligence=matchup_intelligence,
    )
    return asdict(result)
