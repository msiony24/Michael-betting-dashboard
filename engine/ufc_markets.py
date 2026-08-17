from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math


DERIVATIVE_MARKETS_VERSION = "Macabets UFC Derivative Markets v0.1"


@dataclass(frozen=True)
class UFCDerivativeMarketConfig:
    bet_roi_threshold: float = 0.08
    watch_roi_threshold: float = 0.03
    min_bet_confidence: int = 60
    min_probability_for_bet: float = 0.04


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def american_to_decimal(odds: int | float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    return 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def implied_probability(odds: int | float) -> float:
    return 1.0 / american_to_decimal(odds)


def probability_to_american(probability: float) -> int:
    p = _clip(float(probability), 0.001, 0.999)
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def _supported_total_lines(rounds: int) -> list[float]:
    return [1.5, 2.5] if int(rounds) == 3 else [1.5, 2.5, 3.5, 4.5]


def _round_finish_masses(simulation: dict[str, Any]) -> list[float]:
    rounds = int(simulation.get("rounds", 3) or 3)
    finish_probability = _clip(float(simulation.get("finish_probability", 0.0) or 0.0), 0.0, 1.0)
    conditional = simulation.get("finish_round_probabilities_given_finish") or {}
    masses: list[float] = []
    for round_no in range(1, rounds + 1):
        value = conditional.get(f"Round {round_no}", 0.0)
        try:
            probability = float(value)
        except (TypeError, ValueError):
            probability = 0.0
        masses.append(finish_probability * max(0.0, probability))
    total = sum(masses)
    if total > 0 and finish_probability > 0:
        scale = finish_probability / total
        masses = [mass * scale for mass in masses]
    return masses


def total_round_probabilities(simulation: dict[str, Any], line: float) -> tuple[float, float]:
    """Return over/under probability for a UFC half-round total.

    The simulator currently models finish round, not exact seconds. At an X.5 line,
    Macabets splits the modeled finish mass in that round 50/50 around the 2:30 mark.
    That is a deliberately neutral timing assumption until finish-time calibration is added.
    """
    rounds = int(simulation.get("rounds", 3) or 3)
    line = float(line)
    if line not in _supported_total_lines(rounds):
        raise ValueError(f"Unsupported {rounds}-round UFC total: {line}")
    target_round = int(math.ceil(line))
    masses = _round_finish_masses(simulation)
    under = sum(masses[: max(0, target_round - 1)])
    if 1 <= target_round <= len(masses):
        under += 0.5 * masses[target_round - 1]
    under = _clip(under, 0.0, 1.0)
    return 1.0 - under, under


def build_derivative_markets(
    simulation: dict[str, Any],
    fighter_a: str,
    fighter_b: str,
) -> dict[str, Any]:
    if not simulation or not simulation.get("available"):
        return {"available": False, "version": DERIVATIVE_MARKETS_VERSION}

    rounds = int(simulation.get("rounds", 3) or 3)
    method_probabilities = {
        "a_ko_tko": float(simulation.get("a_ko_tko_probability", 0.0) or 0.0),
        "a_submission": float(simulation.get("a_submission_probability", 0.0) or 0.0),
        "a_decision": float(simulation.get("a_decision_probability", 0.0) or 0.0),
        "b_ko_tko": float(simulation.get("b_ko_tko_probability", 0.0) or 0.0),
        "b_submission": float(simulation.get("b_submission_probability", 0.0) or 0.0),
        "b_decision": float(simulation.get("b_decision_probability", 0.0) or 0.0),
    }
    labels = {
        "a_ko_tko": f"{fighter_a} by KO/TKO",
        "a_submission": f"{fighter_a} by Submission",
        "a_decision": f"{fighter_a} by Decision",
        "b_ko_tko": f"{fighter_b} by KO/TKO",
        "b_submission": f"{fighter_b} by Submission",
        "b_decision": f"{fighter_b} by Decision",
    }

    methods = []
    for key, probability in method_probabilities.items():
        p = _clip(probability, 0.001, 0.999)
        methods.append({
            "key": key,
            "market": labels[key],
            "probability": probability,
            "fair_odds": probability_to_american(p),
        })

    goes = _clip(float(simulation.get("goes_distance_probability", 0.0) or 0.0), 0.0, 1.0)
    distance = {
        "yes_probability": goes,
        "no_probability": 1.0 - goes,
        "yes_fair_odds": probability_to_american(goes),
        "no_fair_odds": probability_to_american(1.0 - goes),
    }

    totals = []
    for line in _supported_total_lines(rounds):
        over, under = total_round_probabilities(simulation, line)
        totals.append({
            "line": line,
            "over_probability": over,
            "under_probability": under,
            "over_fair_odds": probability_to_american(over),
            "under_fair_odds": probability_to_american(under),
        })

    return {
        "available": True,
        "version": DERIVATIVE_MARKETS_VERSION,
        "rounds": rounds,
        "method_markets": methods,
        "distance_market": distance,
        "round_totals": totals,
        "supported_total_lines": _supported_total_lines(rounds),
        "guardrail": (
            "Derivative prices are downstream of the fight simulation. Method props use simulated KO/TKO, submission, and decision probabilities. "
            "Round totals use simulated finish-round mass; because exact finish seconds are not modeled yet, Macabets splits the target round 50/50 around the X.5 checkpoint."
        ),
    }


def _verdict(
    probability: float,
    roi: float,
    confidence: int,
    config: UFCDerivativeMarketConfig,
) -> str:
    if (
        roi >= config.bet_roi_threshold
        and confidence >= config.min_bet_confidence
        and probability >= config.min_probability_for_bet
    ):
        return "BET"
    if roi >= config.watch_roi_threshold:
        return "WATCH"
    return "PASS"


def _single_price_evaluation(
    probability: float,
    odds: int,
    confidence: int,
    config: UFCDerivativeMarketConfig,
) -> dict[str, Any]:
    p = _clip(float(probability), 0.0, 1.0)
    implied = implied_probability(int(odds))
    roi = p * (american_to_decimal(int(odds)) - 1.0) - (1.0 - p)
    return {
        "probability": p,
        "fair_odds": probability_to_american(p),
        "market_odds": int(odds),
        "implied_probability": implied,
        "edge": p - implied,
        "roi": roi,
        "verdict": _verdict(p, roi, confidence, config),
    }


def evaluate_derivative_market(
    markets: dict[str, Any],
    market_key: str | None,
    *,
    odds_primary: int | None = None,
    odds_secondary: int | None = None,
    total_line: float | None = None,
    confidence: int = 0,
    config: UFCDerivativeMarketConfig | None = None,
) -> dict[str, Any]:
    config = config or UFCDerivativeMarketConfig()
    if not markets.get("available") or not market_key or odds_primary in (None, 0):
        return {"available": False}

    key = str(market_key)
    if key == "total_rounds":
        if total_line is None:
            return {"available": False}
        row = next((r for r in markets.get("round_totals", []) if float(r.get("line")) == float(total_line)), None)
        if row is None:
            return {"available": False}
        primary_label = f"Over {float(total_line):.1f} rounds"
        secondary_label = f"Under {float(total_line):.1f} rounds"
        p_primary = float(row["over_probability"])
        p_secondary = float(row["under_probability"])
    elif key == "goes_distance":
        distance = markets.get("distance_market") or {}
        primary_label = "Goes the distance — Yes"
        secondary_label = "Goes the distance — No"
        p_primary = float(distance.get("yes_probability", 0.0) or 0.0)
        p_secondary = float(distance.get("no_probability", 0.0) or 0.0)
    else:
        method = next((r for r in markets.get("method_markets", []) if r.get("key") == key), None)
        if method is None:
            return {"available": False}
        result = _single_price_evaluation(float(method["probability"]), int(odds_primary), confidence, config)
        return {
            "available": True,
            "market_type": "method",
            "market_key": key,
            "primary_label": str(method["market"]),
            "primary": result,
            "paired_no_vig_available": False,
            "note": "Method-of-victory props are evaluated against the entered price directly. A paired no-vig probability is unavailable because the sportsbook's full mutually exclusive method market was not entered.",
        }

    primary = _single_price_evaluation(p_primary, int(odds_primary), confidence, config)
    secondary = None
    hold = None
    if odds_secondary not in (None, 0):
        secondary = _single_price_evaluation(p_secondary, int(odds_secondary), confidence, config)
        implied_a = implied_probability(int(odds_primary))
        implied_b = implied_probability(int(odds_secondary))
        denom = implied_a + implied_b
        if denom > 0:
            no_vig_a = implied_a / denom
            no_vig_b = implied_b / denom
            primary["no_vig_probability"] = no_vig_a
            primary["no_vig_edge"] = p_primary - no_vig_a
            secondary["no_vig_probability"] = no_vig_b
            secondary["no_vig_edge"] = p_secondary - no_vig_b
            hold = denom - 1.0

    return {
        "available": True,
        "market_type": "two_way",
        "market_key": key,
        "primary_label": primary_label,
        "secondary_label": secondary_label,
        "primary": primary,
        "secondary": secondary,
        "sportsbook_hold": hold,
        "paired_no_vig_available": secondary is not None,
    }
