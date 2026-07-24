"""Transparent situational adjustments for the Macabets NFL engine."""

from __future__ import annotations


QB_STATUS = {
    "Healthy starter": {"points": 0.0, "uncertainty": 0.0},
    "Limited starter": {"points": -1.5, "uncertainty": 2.0},
    "Questionable / uncertain": {"points": -2.5, "uncertainty": 6.0},
    "Confirmed backup": {"points": -4.0, "uncertainty": 1.0},
    "Emergency / third-string": {"points": -6.5, "uncertainty": 2.0},
}

MOTIVATION = {
    "Normal": 0.0,
    "Must win": 0.25,
    "Elimination game": 0.35,
    "Resting starters / meaningless": -3.0,
}

COACHING_CHANGE = {
    "None": {"points": 0.0, "uncertainty": 0.0},
    "Recent coordinator change": {"points": -0.35, "uncertainty": 1.5},
    "Recent head-coach change": {"points": -0.75, "uncertainty": 3.0},
}

TRAVEL_BURDEN = {
    "Standard": 0.0,
    "Cross-country": 0.40,
    "International / unusual": 0.75,
}

WEATHER_TOTAL_ADJUSTMENT = {
    "Normal": 0.0,
    "Rain": -1.0,
    "Snow": -2.0,
    "High wind": -3.0,
    "Extreme heat": 0.5,
    "Extreme cold": -1.5,
}

CONTEXT_CERTAINTY_PENALTY = {
    "High": 0.0,
    "Medium": 4.0,
    "Low": 8.0,
}


def _bounded(value, low, high):
    return max(low, min(high, float(value)))


def evaluate_nfl_context(
    *,
    away_team: str,
    home_team: str,
    weather: str,
    context: dict | None = None,
) -> dict:
    """Return a home-margin adjustment and an auditable factor breakdown.

    Positive point adjustments favor the home team. Negative adjustments favor
    the away team. Known personnel changes move the line; uncertainty lowers
    confidence separately.
    """
    context = dict(context or {})
    breakdown = []
    home_adjustment = 0.0
    confidence_penalty = 0.0

    game_stage = str(context.get("game_stage", "Regular season"))
    week = int(context.get("week", 1))
    division_game = bool(context.get("division_game", False))

    if game_stage == "Regular season" and week <= 3:
        confidence_penalty += 4.0
        breakdown.append({
            "Factor": "Early-season uncertainty",
            "Advantage": "Neither team",
            "Home margin adjustment": 0.0,
            "Confidence impact": -4.0,
            "Explanation": "Weeks 1-3 carry extra uncertainty because current-season samples are limited.",
        })
    elif game_stage == "Regular season" and week == 4:
        confidence_penalty += 2.0
        breakdown.append({
            "Factor": "Early-season uncertainty",
            "Advantage": "Neither team",
            "Home margin adjustment": 0.0,
            "Confidence impact": -2.0,
            "Explanation": "Week 4 still carries a smaller current-season sample penalty.",
        })

    if division_game:
        confidence_penalty += 1.5
        breakdown.append({
            "Factor": "Division familiarity",
            "Advantage": "Neither team",
            "Home margin adjustment": 0.0,
            "Confidence impact": -1.5,
            "Explanation": "Division familiarity increases matchup uncertainty but does not automatically favor either side.",
        })

    away_rest = _bounded(context.get("away_rest_days", 7), 3, 21)
    home_rest = _bounded(context.get("home_rest_days", 7), 3, 21)
    rest_adjustment = _bounded((home_rest - away_rest) * 0.15, -1.5, 1.5)
    if abs(rest_adjustment) >= 0.05:
        home_adjustment += rest_adjustment
        breakdown.append({
            "Factor": "Rest advantage",
            "Advantage": home_team if rest_adjustment > 0 else away_team,
            "Home margin adjustment": round(rest_adjustment, 2),
            "Confidence impact": 0.0,
            "Explanation": f"{home_team} has {home_rest:.0f} rest days; {away_team} has {away_rest:.0f}.",
        })

    travel_label = str(context.get("away_travel_burden", "Standard"))
    travel_adjustment = TRAVEL_BURDEN.get(travel_label, 0.0)
    if travel_adjustment:
        home_adjustment += travel_adjustment
        breakdown.append({
            "Factor": "Away travel burden",
            "Advantage": home_team,
            "Home margin adjustment": travel_adjustment,
            "Confidence impact": 0.0,
            "Explanation": f"{away_team} is assigned a {travel_label.lower()} travel burden.",
        })

    away_qb_label = str(context.get("away_qb_status", "Healthy starter"))
    home_qb_label = str(context.get("home_qb_status", "Healthy starter"))
    away_qb = QB_STATUS.get(away_qb_label, QB_STATUS["Healthy starter"])
    home_qb = QB_STATUS.get(home_qb_label, QB_STATUS["Healthy starter"])
    qb_adjustment = home_qb["points"] - away_qb["points"]
    confidence_penalty += away_qb["uncertainty"] + home_qb["uncertainty"]
    if away_qb_label != "Healthy starter" or home_qb_label != "Healthy starter":
        home_adjustment += qb_adjustment
        breakdown.append({
            "Factor": "Quarterback availability",
            "Advantage": home_team if qb_adjustment > 0 else away_team if qb_adjustment < 0 else "Neither team",
            "Home margin adjustment": round(qb_adjustment, 2),
            "Confidence impact": round(-(away_qb["uncertainty"] + home_qb["uncertainty"]), 1),
            "Explanation": f"{away_team}: {away_qb_label}. {home_team}: {home_qb_label}.",
        })

    away_injuries = _bounded(context.get("away_non_qb_injury_points", 0), 0, 6)
    home_injuries = _bounded(context.get("home_non_qb_injury_points", 0), 0, 6)
    injury_adjustment = away_injuries - home_injuries
    if away_injuries or home_injuries:
        home_adjustment += injury_adjustment
        breakdown.append({
            "Factor": "Non-QB injuries",
            "Advantage": home_team if injury_adjustment > 0 else away_team if injury_adjustment < 0 else "Neither team",
            "Home margin adjustment": round(injury_adjustment, 2),
            "Confidence impact": 0.0,
            "Explanation": (
                f"Entered injury deductions — {away_team}: {away_injuries:.1f}; "
                f"{home_team}: {home_injuries:.1f}."
            ),
        })

    away_motivation_label = str(context.get("away_motivation", "Normal"))
    home_motivation_label = str(context.get("home_motivation", "Normal"))
    away_motivation = MOTIVATION.get(away_motivation_label, 0.0)
    home_motivation = MOTIVATION.get(home_motivation_label, 0.0)
    motivation_adjustment = home_motivation - away_motivation
    if away_motivation_label != "Normal" or home_motivation_label != "Normal":
        home_adjustment += motivation_adjustment
        breakdown.append({
            "Factor": "Motivation / availability",
            "Advantage": home_team if motivation_adjustment > 0 else away_team if motivation_adjustment < 0 else "Neither team",
            "Home margin adjustment": round(motivation_adjustment, 2),
            "Confidence impact": 0.0,
            "Explanation": f"{away_team}: {away_motivation_label}. {home_team}: {home_motivation_label}.",
        })

    away_coaching_label = str(context.get("away_coaching_change", "None"))
    home_coaching_label = str(context.get("home_coaching_change", "None"))
    away_coaching = COACHING_CHANGE.get(away_coaching_label, COACHING_CHANGE["None"])
    home_coaching = COACHING_CHANGE.get(home_coaching_label, COACHING_CHANGE["None"])
    coaching_adjustment = home_coaching["points"] - away_coaching["points"]
    coaching_uncertainty = away_coaching["uncertainty"] + home_coaching["uncertainty"]
    confidence_penalty += coaching_uncertainty
    if away_coaching_label != "None" or home_coaching_label != "None":
        home_adjustment += coaching_adjustment
        breakdown.append({
            "Factor": "Coaching / personnel change",
            "Advantage": home_team if coaching_adjustment > 0 else away_team if coaching_adjustment < 0 else "Neither team",
            "Home margin adjustment": round(coaching_adjustment, 2),
            "Confidence impact": round(-coaching_uncertainty, 1),
            "Explanation": f"{away_team}: {away_coaching_label}. {home_team}: {home_coaching_label}.",
        })

    manual_advantage_team = str(context.get("manual_advantage_team", "Neutral"))
    manual_advantage_points = _bounded(context.get("manual_advantage_points", 0), 0, 4)
    if manual_advantage_team == home_team:
        home_adjustment += manual_advantage_points
        manual_signed_points = manual_advantage_points
    elif manual_advantage_team == away_team:
        home_adjustment -= manual_advantage_points
        manual_signed_points = -manual_advantage_points
    else:
        manual_signed_points = 0.0
    if manual_signed_points:
        breakdown.append({
            "Factor": "Manual matchup / weather edge",
            "Advantage": manual_advantage_team,
            "Home margin adjustment": round(manual_signed_points, 2),
            "Confidence impact": 0.0,
            "Explanation": str(
                context.get("manual_advantage_reason")
                or "User-entered matchup adjustment."
            ),
        })

    certainty = str(context.get("context_certainty", "High"))
    certainty_penalty = CONTEXT_CERTAINTY_PENALTY.get(certainty, 0.0)
    confidence_penalty += certainty_penalty
    if certainty_penalty:
        breakdown.append({
            "Factor": "Context information quality",
            "Advantage": "Neither team",
            "Home margin adjustment": 0.0,
            "Confidence impact": -certainty_penalty,
            "Explanation": f"Context certainty was marked {certainty.lower()}.",
        })

    weather_already_priced = bool(context.get("weather_already_priced", True))
    total_adjustment = (
        0.0
        if weather_already_priced
        else WEATHER_TOTAL_ADJUSTMENT.get(str(weather), 0.0)
    )
    if total_adjustment:
        breakdown.append({
            "Factor": "Weather total adjustment",
            "Advantage": "Total only",
            "Home margin adjustment": 0.0,
            "Confidence impact": 0.0,
            "Explanation": (
                f"{weather} weather moves the projected total {total_adjustment:+.1f} "
                "because the entered market total was marked as not weather-adjusted."
            ),
        })

    return {
        "game_stage": game_stage,
        "home_margin_adjustment": round(home_adjustment, 2),
        "total_adjustment": round(total_adjustment, 2),
        "confidence_penalty": round(min(confidence_penalty, 25.0), 1),
        "breakdown": breakdown,
    }
