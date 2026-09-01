"""Pure betting-math functions extracted from app.py.

This module has zero dependency on streamlit or any UI code, so unlike
app.py it can be imported and tested directly. It exists because this exact
class of logic -- odds conversion, expected value, verdict thresholds, Kelly
sizing -- has produced three real, found-and-fixed bugs across this project
(the "Worth Betting on negative EV" loophole, the price-compression bias in
bet-confidence scoring, and a duplicate copy of the verdict logic that still
carried the same loophole). All three were only caught by manually reading
the code, because none of this had automated tests. This module is that
fix: app.py now imports these functions instead of defining its own copies.

Every function here is pure: given the same inputs, it returns the same
output, with no side effects and no dependency on Streamlit session state.
"""
from __future__ import annotations

import math

VERDICT_ORDER = {
    "Complete Pass": 0,
    "Pass": 1,
    "Lean": 2,
    "Worth Betting": 3,
    "Strong Bet": 4,
}

LEGACY_PRICE_LABELS = {
    "Significantly Underpriced": "Very Underpriced",
    "Fairly Priced": "Fair",
    "Slightly Overpriced": "Premium",
    "Significantly Overpriced": "Very Overpriced",
}


# --- basic formatting ---------------------------------------------------------

def money(value):
    return f"${value:,.2f}"


def safe_int(value, default: int = 0) -> int:
    """Convert blank, missing, or numeric-looking values safely."""
    import pandas as pd
    try:
        if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


# --- odds conversion ------------------------------------------------------------

def american_to_decimal(odds):
    if odds == 0:
        return 1.0
    return 1 + (100 / abs(odds) if odds < 0 else odds / 100)


def probability_to_american(probability):
    """Convert a win probability (0-1) to fair American odds."""
    probability = min(max(float(probability), 0.0001), 0.9999)
    if probability >= 0.5:
        return -round(100 * probability / (1 - probability))
    return round(100 * (1 - probability) / probability)


def format_american(odds):
    odds = int(round(odds))
    return f"+{odds}" if odds > 0 else str(odds)


def implied_probability(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    if odds > 0:
        return 100 / (odds + 100)
    return 0.0


def no_vig_probabilities(odds_a, odds_b):
    """Remove sportsbook margin from a two-sided moneyline market."""
    raw_a = implied_probability(int(odds_a))
    raw_b = implied_probability(int(odds_b))
    total = raw_a + raw_b
    if total <= 0:
        return 0.5, 0.5, 0.0
    return raw_a / total, raw_b / total, total - 1


# --- value / ROI ------------------------------------------------------------

def minimum_acceptable_odds(model_probability, required_roi=0.02):
    """Worst American price that still preserves the required expected ROI."""
    probability = min(max(float(model_probability), 0.0001), 0.9999)
    required_decimal = (1 + required_roi) / probability
    if required_decimal <= 1:
        return -10000
    if required_decimal >= 2:
        return round((required_decimal - 1) * 100)
    return -round(100 / (required_decimal - 1))


def estimated_nfl_cover_probability(point_edge, margin_std=13.86):
    """Estimate cover probability from the gap between market and fair spread."""
    z_score = float(point_edge) / float(margin_std)
    return 0.5 * (1 + math.erf(z_score / math.sqrt(2)))


def required_nfl_spread_edge(american_odds, required_roi=0.05, margin_std=13.86):
    """Point edge required to reach a target ROI at the entered spread price."""
    decimal_price = american_to_decimal(int(american_odds))
    target_probability = min(max((1 + required_roi) / decimal_price, 0.0001), 0.9999)
    low, high = -30.0, 30.0
    for _ in range(60):
        midpoint = (low + high) / 2
        if estimated_nfl_cover_probability(midpoint, margin_std) < target_probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def stake_to_win(odds, target):
    if odds < 0:
        return target * abs(odds) / 100
    if odds > 0:
        return target * 100 / odds
    return 0.0


def potential_profit(odds, stake):
    if odds < 0:
        return stake * 100 / abs(odds)
    if odds > 0:
        return stake * odds / 100
    return 0.0


def kelly_fraction(model_prob, odds):
    dec = american_to_decimal(odds)
    b = dec - 1
    q = 1 - model_prob
    if b <= 0:
        return 0.0
    return max(0.0, (b * model_prob - q) / b)


# --- verdict / price-quality logic (the money-facing recommendation) --------

def verdict_probability_ceiling(model_probability):
    """Return the strongest verdict allowed by projected win probability.

    Price can make a bet attractive, but it cannot manufacture conviction that
    the win-probability model does not have. These are ceilings, not automatic
    recommendations.
    """
    probability = min(max(float(model_probability), 0.0), 1.0)
    if probability < 0.52:
        return "Pass"
    if probability < 0.575:
        return "Lean"
    if probability < 0.65:
        return "Worth Betting"
    return "Strong Bet"


def cap_verdict_by_probability(verdict, model_probability):
    """Prevent price/confidence or Challenge mode from exceeding conviction."""
    verdict = str(verdict or "Pass")
    ceiling = verdict_probability_ceiling(model_probability)
    if VERDICT_ORDER.get(verdict, VERDICT_ORDER["Pass"]) > VERDICT_ORDER[ceiling]:
        return ceiling
    return verdict


def moneyline_price_quality(model_probability, market_odds, confidence_score):
    """Separate mathematical price value from the probability-capped verdict."""
    probability = min(max(float(model_probability), 0.0001), 0.9999)
    market_odds = int(market_odds)
    confidence_score = float(confidence_score)
    if confidence_score <= 10:
        confidence_score *= 10

    expected_roi = probability * american_to_decimal(market_odds) - 1

    # Price-assessment label is deliberately edge-based (model probability
    # minus the market-implied probability), not raw ROI. ROI for a given
    # edge is mathematically compressed at short prices (the same edge pays
    # far less ROI% at -350 than at +150), so ROI-based breakpoints quietly
    # mislabeled well-supported heavy favorites as "Fair"/"Premium" even when
    # the model saw real value. These breakpoints are calibrated so a bet at
    # a typical near-even-money price (-110) gets the identical label it
    # always did; the fix is that the same edge now means the same label at
    # every price, not just around -110.
    market_implied_probability = implied_probability(market_odds)
    edge = probability - market_implied_probability

    if edge >= 0.08:
        quality = "Very Underpriced"
    elif edge >= 0.02:
        quality = "Underpriced"
    elif edge >= -0.01:
        quality = "Fair"
    elif edge >= -0.035:
        quality = "Premium"
    elif edge >= -0.06:
        quality = "Overpriced"
    else:
        quality = "Very Overpriced"

    # A Strong Bet must clear BOTH the value test and a minimum win-probability
    # conviction floor. A large pricing edge alone cannot turn a modest favorite
    # into Macabets' strongest recommendation.
    #
    # "Worth Betting" is an active recommendation to wager real money, so it
    # requires the model's own expected_roi to be non-negative. High confidence
    # alone can no longer promote a price the model itself considers -EV into
    # a betting recommendation.
    # Verdict tiers are edge-based (model probability vs. market-implied
    # probability), matching the price_assessment label above -- not raw
    # expected ROI. ROI compresses at short prices and expands at plus-money
    # prices for the *same* probability edge, so an ROI-based gate was
    # inconsistently easy to clear for underdogs and inconsistently hard to
    # clear for short favorites, even when the genuine edge was identical.
    # expected_roi is kept as a hard floor on every tier so a large edge can
    # never promote a price the model itself sees as -EV (or barely +EV)
    # into a recommendation -- this preserves the original "no Worth Betting
    # on negative EV" fix while removing the price-direction bias.
    if edge >= 0.08 and confidence_score >= 75 and probability >= 0.75 and expected_roi >= 0.0:
        verdict = "Strong Bet"
    elif edge >= 0.02 and confidence_score >= 62 and expected_roi >= -0.02:
        verdict = "Worth Betting"
    elif edge >= -0.01 and confidence_score >= 78 and expected_roi >= -0.05:
        verdict = "Lean"
    elif edge <= -0.08 or (edge <= -0.06 and confidence_score < 78):
        verdict = "Complete Pass"
    else:
        verdict = "Pass"

    # Final conviction gate. Market value can lower the price required to bet,
    # but it cannot promote a low-probability pick into a stronger verdict.
    verdict = cap_verdict_by_probability(verdict, probability)

    return {
        "expected_roi": expected_roi,
        "quality": quality,
        "price_assessment": quality,
        "verdict": verdict,
        # Backward compatibility for older display code and the existing DB column.
        "recommendation": verdict,
    }


def normalize_price_assessment(label):
    label = str(label or "").strip()
    return LEGACY_PRICE_LABELS.get(label, label or "—")


def decision_label(expected_roi, confidence):
    """Compatibility wrapper returning the new verdict language.

    Historically this duplicated moneyline_price_quality()'s threshold logic
    rather than delegating to it, and that duplicate copy drifted: it still
    carried the negative-EV "Worth Betting" loophole after the main function
    was fixed, because nothing kept the two in sync. Keep this in mind if
    the thresholds below ever need to change again -- moneyline_price_quality
    is the source of truth and this should be updated to match it.
    """
    expected_roi = float(expected_roi)
    confidence_score = float(confidence)
    if confidence_score <= 10:
        confidence_score *= 10

    # This compatibility helper does not know the underlying model probability,
    # so it must never manufacture a Strong Bet from ROI + confidence alone.
    # moneyline_price_quality() is the source of truth for that verdict
    # because it can enforce the win-probability floor.
    if expected_roi >= 0.08 and confidence_score >= 70:
        verdict = "Worth Betting"
    elif expected_roi >= 0.025 and confidence_score >= 62:
        verdict = "Worth Betting"
    elif expected_roi >= 0.0 and confidence_score >= 82:
        verdict = "Worth Betting"
    elif expected_roi >= -0.075 and confidence_score >= 78:
        verdict = "Lean"
    elif expected_roi <= -0.12 or (expected_roi <= -0.08 and confidence_score < 78):
        verdict = "Complete Pass"
    else:
        verdict = "Pass"

    reason = (
        f"Macabets' final verdict is {verdict.lower()} after weighing the offered price "
        "against the model edge and current model confidence."
    )
    return verdict, reason


def nfl_bottom_line(team, probability, quality, verdict, confidence_band):
    likely = f"Macabets expects {team} to win and assigns a {probability:.1%} win probability."
    if quality in {"Very Underpriced", "Underpriced"}:
        price = "The current moneyline is favorable relative to Macabets' fair price."
    elif quality == "Fair":
        price = "The market is close to Macabets' fair price, so the case rests more on conviction than value."
    elif quality == "Premium":
        price = "The projected winner is a little expensive, but the premium is understandable and can remain playable when conviction is high."
    elif quality == "Overpriced":
        price = "The projected winner is expensive, so Macabets requires stronger conviction before accepting the number."
    else:
        price = "The projected winner is priced well beyond Macabets' fair number."
    return f"{likely} {price} Verdict: {verdict}. Prediction confidence: {confidence_band}."


def nfl_grade_band(grade):
    if grade >= 88:
        return "Elite"
    if grade >= 82:
        return "Strong"
    if grade >= 76:
        return "Above Average"
    if grade >= 70:
        return "Average"
    if grade >= 64:
        return "Below Average"
    return "Weak"


# --- confidence scoring -------------------------------------------------------

def tennis_confidence_meter(result):
    """Combine model stability, data quality, sample size and context clarity."""
    model_score = min(max(float(result.get("confidence", 5)) * 10, 0), 100)
    data_score = min(max(float(result.get("data_quality", 5)) * 10, 0), 100)

    samples = []
    for profile_key in ("profile_a", "profile_b"):
        try:
            sample_value = float(result.get(profile_key, {}).get("sample", 0))
            samples.append(sample_value if math.isfinite(sample_value) else 0.0)
        except (TypeError, ValueError):
            samples.append(0.0)
    minimum_sample = min(samples) if samples else 0.0
    sample_score = min(max(minimum_sample / 40.0 * 100.0, 0), 100)

    health_penalties = {
        "Clear": 0,
        "Minor concern": 10,
        "Recent medical timeout": 18,
        "Returning from layoff": 22,
        "Recent retirement": 28,
        "Significant concern": 35,
    }
    health_a = str(result.get("injury_status_a", "Clear"))
    health_b = str(result.get("injury_status_b", "Clear"))
    context_penalty = min(
        health_penalties.get(health_a, 10) + health_penalties.get(health_b, 10),
        60,
    )
    context_score = 100 - context_penalty

    overall = round(
        model_score * 0.35
        + data_score * 0.35
        + sample_score * 0.20
        + context_score * 0.10
    )
    if overall >= 85:
        band = "High"
    elif overall >= 70:
        band = "Solid"
    elif overall >= 55:
        band = "Moderate"
    else:
        band = "Low"

    return {
        "overall": int(min(max(overall, 0), 100)),
        "band": band,
        "model": round(model_score),
        "data": round(data_score),
        "sample": round(sample_score),
        "context": round(context_score),
        "minimum_sample": int(minimum_sample),
    }


def tennis_probability_confidence_band(probability_a, reliability_score=100):
    """Return a user-facing confidence label anchored to win probability.

    Win probability sets the maximum label. Data/model reliability may downgrade
    the label, but can never promote a close matchup into a higher-confidence tier.
    """
    try:
        probability_a = float(probability_a)
    except (TypeError, ValueError):
        probability_a = 0.5
    favorite_probability = max(probability_a, 1.0 - probability_a)

    if favorite_probability >= 0.85:
        probability_level = 3
    elif favorite_probability >= 0.80:
        probability_level = 2
    elif favorite_probability >= 0.60:
        probability_level = 1
    else:
        probability_level = 0

    try:
        reliability_score = float(reliability_score)
    except (TypeError, ValueError):
        reliability_score = 0.0
    if reliability_score >= 85:
        reliability_level = 3
    elif reliability_score >= 70:
        reliability_level = 2
    elif reliability_score >= 55:
        reliability_level = 1
    else:
        reliability_level = 0

    labels = ["Low Confidence", "Moderate Confidence", "High Confidence", "Very High Confidence"]
    return labels[min(probability_level, reliability_level)]


def tennis_bet_confidence(analysis_confidence, edge, expected_roi):
    """Grade confidence in a specific price without changing the match analysis."""
    positive_edge = max(float(edge), 0.0)
    edge_score = min(positive_edge / 0.10 * 100.0, 100.0)
    # Deliberately price-agnostic: dollar ROI for a given probability edge is
    # mathematically compressed at short prices (the same 5-point edge pays
    # far less ROI% at -350 than at +150), so scoring on raw ROI meant a
    # heavy favorite with genuinely strong conviction still read as
    # "Cautious"/"Low" purely because of price. Edge (probability
    # disagreement with the market) is the actual price-agnostic measure of
    # conviction.
    score = (
        float(analysis_confidence) * 0.55
        + edge_score * 0.45
    )

    if positive_edge < 0.02:
        score = min(score, 49)
    elif positive_edge < 0.05:
        score = min(score, 69)
    score = int(round(min(max(score, 0), 100)))

    if score >= 80:
        band = "High"
    elif score >= 65:
        band = "Solid"
    elif score >= 50:
        band = "Cautious"
    else:
        band = "Low / Pass"
    return {"overall": score, "band": band}


def fair_line_probability(scores_favorite, scores_opponent, weights, confidence):
    """Create a first-pass fair probability from a weighted matchup scorecard.

    The confidence input shrinks uncertain estimates toward 50%, preventing
    low-information matchups from producing extreme prices.
    """
    weighted_difference = sum(
        weights[key] * (scores_favorite[key] - scores_opponent[key])
        for key in weights
    )
    raw_probability = 1 / (1 + math.exp(-0.45 * weighted_difference))
    confidence_factor = min(max(confidence / 10, 0.1), 1.0)
    adjusted_probability = 0.5 + (raw_probability - 0.5) * confidence_factor
    return min(max(adjusted_probability, 0.02), 0.98), weighted_difference
