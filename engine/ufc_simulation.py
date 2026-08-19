from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np


SIMULATION_VERSION = "Macabets UFC Simulation v0.5 — Round-State Matchup Aware"


@dataclass(frozen=True)
class UFCSimulationConfig:
    simulations: int = 20000
    seed: int = 2417
    min_finish_probability: float = 0.12
    max_finish_probability_3r: float = 0.82
    max_finish_probability_5r: float = 0.88
    finish_calibration: float = 0.05


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _num(mapping: dict[str, Any], key: str, default: float | None = None) -> float | None:
    try:
        value = float(mapping.get(key))
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) else value


def _pct01(value: float | None, default: float = 0.5) -> float:
    if value is None:
        return default
    # Composite / percentile fields are stored 0-100. Raw rate fields are 0-1.
    return _clip(value / 100.0 if value > 1.0 else value, 0.0, 1.0)


def _finish_profile(
    recent: dict[str, Any],
    performance: dict[str, Any],
    opponent_recent: dict[str, Any],
    opponent_performance: dict[str, Any],
    *,
    opponent_damage: dict[str, Any] | None = None,
    rounds: int,
    config: UFCSimulationConfig,
) -> dict[str, float]:
    own_finish = _pct01(_num(recent, "finish_rate"), 0.42)
    opp_finish_loss = _pct01(_num(opponent_recent, "loss_finish_rate"), 0.30)
    kd_pressure = _pct01(_num(performance, "kd_per15_pct"), 0.50)
    sub_pressure = _pct01(_num(performance, "sub_attempts_per15_pct"), 0.50)
    opp_durability = _pct01(_num(opponent_performance, "durability_score"), 0.50)
    pace = _pct01(_num(performance, "pace_score"), 0.50)
    damage_risk = _pct01(_num(opponent_damage or {}, "risk_score"), 0.0)
    damage_reliability = _clip(_num(opponent_damage or {}, "reliability", 0.0) or 0.0, 0.0, 1.0)

    # Historical finish tendency remains dominant. Percentile traits only refine the
    # conditional method distribution; they do not create a second winner model.
    finish_probability = (
        0.52 * own_finish
        + 0.18 * opp_finish_loss
        + 0.10 * kd_pressure
        + 0.08 * sub_pressure
        + 0.07 * (1.0 - opp_durability)
        + 0.05 * pace
    )
    # Damage Risk is trajectory/recovery context. It can modestly redistribute method
    # probabilities, but never changes the winner probability inside the simulator.
    finish_probability += 0.06 * damage_risk * damage_reliability
    # Historical Validation v0.1 found the original simulator was about five percentage
    # points too decision-heavy on a large leakage-safe retrospective sample. Apply a
    # modest global finish calibration before the existing 5-round exposure adjustment.
    finish_probability += config.finish_calibration
    if int(rounds) == 5:
        finish_probability += 0.07
    max_finish = config.max_finish_probability_5r if int(rounds) == 5 else config.max_finish_probability_3r
    finish_probability = _clip(finish_probability, config.min_finish_probability, max_finish)

    ko_hist = _pct01(_num(recent, "ko_win_rate"), 0.28)
    sub_hist = _pct01(_num(recent, "submission_win_rate"), 0.18)
    hist_total = ko_hist + sub_hist
    if hist_total > 0.05:
        ko_base = ko_hist / hist_total
    else:
        ko_base = 0.62
    trait_ko = kd_pressure / max(kd_pressure + sub_pressure, 1e-9)
    ko_share = _clip(0.72 * ko_base + 0.28 * trait_ko + 0.10 * damage_risk * damage_reliability, 0.15, 0.92)
    sub_share = 1.0 - ko_share

    return {
        "finish_probability": finish_probability,
        "ko_share_of_finishes": ko_share,
        "submission_share_of_finishes": sub_share,
        "decision_probability_if_win": 1.0 - finish_probability,
    }


def _round_weights(
    rounds: int,
    pace_a: float,
    pace_b: float,
    cardio_a: dict[str, Any] | None = None,
    cardio_b: dict[str, Any] | None = None,
) -> np.ndarray:
    if int(rounds) == 5:
        base = np.array([0.31, 0.24, 0.19, 0.15, 0.11], dtype=float)
    else:
        base = np.array([0.44, 0.33, 0.23], dtype=float)
    avg_pace = (_pct01(pace_a, 0.5) + _pct01(pace_b, 0.5)) / 2.0
    pace_tilt = (avg_pace - 0.5) * 0.18
    indices = np.linspace(1.0, -1.0, len(base))
    adjusted = base * (1.0 + pace_tilt * indices)

    # Round Cardio changes *when* finishes occur, not who wins. Poor retention
    # shifts some conditional finish mass toward later rounds, where fatigue-related
    # breakdowns are more plausible. Missing/low-reliability cardio remains neutral.
    profiles = [p for p in (cardio_a, cardio_b) if isinstance(p, dict) and p.get("available")]
    if profiles:
        weighted = []
        for profile in profiles:
            retention = _clip(float(profile.get("retention", 1.0) or 1.0), 0.5, 1.35)
            reliability = _clip(float(profile.get("reliability", 0.0) or 0.0), 0.0, 1.0)
            weighted.append((retention, reliability))
        denom = sum(w for _, w in weighted)
        if denom > 0:
            avg_retention = sum(r * w for r, w in weighted) / denom
            fatigue = _clip(1.0 - avg_retention, -0.20, 0.35) * min(1.0, denom / len(weighted))
            late_axis = np.linspace(-1.0, 1.0, len(base))
            adjusted = adjusted * (1.0 + fatigue * (0.34 if int(rounds) == 5 else 0.22) * late_axis)

    adjusted = np.maximum(0.01, adjusted)
    return adjusted / adjusted.sum()




def _matchup_signal(matchup: dict[str, Any] | None) -> tuple[float, float]:
    if not isinstance(matchup, dict) or not matchup.get("available"):
        return 0.0, 0.0
    gap = _clip(float(matchup.get("weighted_gap", 0.0) or 0.0) / 50.0, -1.0, 1.0)
    reliability = _clip(float(matchup.get("reliability", 0.0) or 0.0), 0.0, 1.0)
    return gap, reliability


def _conditional_method_refinement(
    path: dict[str, float],
    *,
    side: int,
    striking_matchup: dict[str, Any] | None,
    grappling_matchup: dict[str, Any] | None,
) -> dict[str, float]:
    """Refine KO-vs-submission share only; never alter side win probability."""
    strike_gap, strike_rel = _matchup_signal(striking_matchup)
    grapple_gap, grapple_rel = _matchup_signal(grappling_matchup)
    directional_strike = strike_gap * strike_rel * float(side)
    directional_grapple = grapple_gap * grapple_rel * float(side)
    ko = _clip(
        float(path["ko_share_of_finishes"])
        + 0.08 * directional_strike
        - 0.05 * directional_grapple,
        0.12,
        0.94,
    )
    result = dict(path)
    result["ko_share_of_finishes"] = ko
    result["submission_share_of_finishes"] = 1.0 - ko
    return result


def _side_method_round_weights(
    rounds: int,
    *,
    method: str,
    side: int,
    pace_a: float,
    pace_b: float,
    cardio: dict[str, Any] | None,
    opponent_damage: dict[str, Any] | None,
    striking_matchup: dict[str, Any] | None,
    grappling_matchup: dict[str, Any] | None,
) -> np.ndarray:
    base = _round_weights(rounds, pace_a, pace_b, cardio if side > 0 else None, cardio if side < 0 else None)
    axis = np.linspace(-1.0, 1.0, len(base))
    early_axis = -axis

    retention = 1.0
    cardio_rel = 0.0
    if isinstance(cardio, dict) and cardio.get("available"):
        retention = _clip(float(cardio.get("retention", 1.0) or 1.0), 0.55, 1.30)
        cardio_rel = _clip(float(cardio.get("reliability", 0.0) or 0.0), 0.0, 1.0)
    fatigue = _clip(1.0 - retention, -0.20, 0.35) * cardio_rel

    damage = 0.0
    damage_rel = 0.0
    if isinstance(opponent_damage, dict) and opponent_damage.get("available"):
        damage = _pct01(_num(opponent_damage, "risk_score"), 0.0)
        damage_rel = _clip(_num(opponent_damage, "reliability", 0.0) or 0.0, 0.0, 1.0)

    strike_gap, strike_rel = _matchup_signal(striking_matchup)
    grapple_gap, grapple_rel = _matchup_signal(grappling_matchup)
    strike_edge = strike_gap * strike_rel * float(side)
    grapple_edge = grapple_gap * grapple_rel * float(side)

    adjusted = base.copy()
    # A fading fighter's own finishing threat shifts earlier. Strong retention preserves
    # more threat into later rounds. This is conditional timing, not a winner re-price.
    adjusted *= 1.0 + fatigue * 0.30 * early_axis
    if method == "ko":
        adjusted *= 1.0 + (0.16 * strike_edge + 0.10 * damage * damage_rel) * early_axis
    else:
        # Grappling/submission advantages often compound through repeated exchanges;
        # tilt slightly later when the fighter can sustain pace/control.
        adjusted *= 1.0 + (0.12 * grapple_edge - 0.08 * fatigue) * axis
    adjusted = np.maximum(0.01, adjusted)
    return adjusted / adjusted.sum()

def simulate_fight(
    fighter_a: str,
    fighter_b: str,
    probability_a: float,
    recent_a: dict[str, Any],
    recent_b: dict[str, Any],
    performance_a: dict[str, Any],
    performance_b: dict[str, Any],
    *,
    cardio_a: dict[str, Any] | None = None,
    cardio_b: dict[str, Any] | None = None,
    damage_a: dict[str, Any] | None = None,
    damage_b: dict[str, Any] | None = None,
    striking_matchup: dict[str, Any] | None = None,
    grappling_matchup: dict[str, Any] | None = None,
    rounds: int = 3,
    config: UFCSimulationConfig | None = None,
) -> dict[str, Any]:
    """Simulate fight outcomes conditional on Macabets' final matchup win probability.

    The simulator does not change the side probability. It decomposes that already-built
    probability into KO/TKO, submission and decision paths, then Monte Carlo samples those
    paths and finish rounds. This prevents simulation from becoming a competing winner model.
    """
    config = config or UFCSimulationConfig()
    rounds = int(rounds)
    if rounds not in {3, 5}:
        raise ValueError("UFC simulation supports 3 or 5 rounds.")
    p_a = _clip(float(probability_a), 0.01, 0.99)
    p_b = 1.0 - p_a

    path_a = _finish_profile(recent_a, performance_a, recent_b, performance_b, opponent_damage=damage_b, rounds=rounds, config=config)
    path_b = _finish_profile(recent_b, performance_b, recent_a, performance_a, opponent_damage=damage_a, rounds=rounds, config=config)
    path_a = _conditional_method_refinement(path_a, side=1, striking_matchup=striking_matchup, grappling_matchup=grappling_matchup)
    path_b = _conditional_method_refinement(path_b, side=-1, striking_matchup=striking_matchup, grappling_matchup=grappling_matchup)

    theoretical = {
        "a_ko_tko": p_a * path_a["finish_probability"] * path_a["ko_share_of_finishes"],
        "a_submission": p_a * path_a["finish_probability"] * path_a["submission_share_of_finishes"],
        "a_decision": p_a * path_a["decision_probability_if_win"],
        "b_ko_tko": p_b * path_b["finish_probability"] * path_b["ko_share_of_finishes"],
        "b_submission": p_b * path_b["finish_probability"] * path_b["submission_share_of_finishes"],
        "b_decision": p_b * path_b["decision_probability_if_win"],
    }
    labels = list(theoretical)
    probs = np.array([theoretical[k] for k in labels], dtype=float)
    probs = probs / probs.sum()

    rng = np.random.default_rng(config.seed)
    draws = rng.choice(len(labels), size=int(config.simulations), p=probs)
    counts = np.bincount(draws, minlength=len(labels))
    empirical = {label: float(counts[i] / int(config.simulations)) for i, label in enumerate(labels)}

    pace_a = _num(performance_a, "pace_score", 50.0) or 50.0
    pace_b = _num(performance_b, "pace_score", 50.0) or 50.0
    round_vectors = {
        "a_ko_tko": _side_method_round_weights(rounds, method="ko", side=1, pace_a=pace_a, pace_b=pace_b, cardio=cardio_a, opponent_damage=damage_b, striking_matchup=striking_matchup, grappling_matchup=grappling_matchup),
        "a_submission": _side_method_round_weights(rounds, method="sub", side=1, pace_a=pace_a, pace_b=pace_b, cardio=cardio_a, opponent_damage=damage_b, striking_matchup=striking_matchup, grappling_matchup=grappling_matchup),
        "b_ko_tko": _side_method_round_weights(rounds, method="ko", side=-1, pace_a=pace_a, pace_b=pace_b, cardio=cardio_b, opponent_damage=damage_a, striking_matchup=striking_matchup, grappling_matchup=grappling_matchup),
        "b_submission": _side_method_round_weights(rounds, method="sub", side=-1, pace_a=pace_a, pace_b=pace_b, cardio=cardio_b, opponent_damage=damage_a, striking_matchup=striking_matchup, grappling_matchup=grappling_matchup),
    }
    finish_mass = sum(theoretical[k] for k in round_vectors)
    unconditional_round_finish = np.zeros(rounds, dtype=float)
    side_a_round_finish = np.zeros(rounds, dtype=float)
    side_b_round_finish = np.zeros(rounds, dtype=float)
    ko_round_finish = np.zeros(rounds, dtype=float)
    sub_round_finish = np.zeros(rounds, dtype=float)
    for key, vec in round_vectors.items():
        mass = float(theoretical[key])
        contribution = mass * vec
        unconditional_round_finish += contribution
        if key.startswith("a_"):
            side_a_round_finish += contribution
        else:
            side_b_round_finish += contribution
        if "ko" in key:
            ko_round_finish += contribution
        else:
            sub_round_finish += contribution
    if finish_mass > 0:
        round_probs = unconditional_round_finish / finish_mass
    else:
        round_probs = _round_weights(rounds, pace_a, pace_b, cardio_a, cardio_b)

    finish_count = int(round(int(config.simulations) * finish_mass))
    round_counts = np.rint(round_probs * max(finish_count, 0)).astype(int) if finish_count else np.zeros(rounds, dtype=int)

    a_win_emp = empirical["a_ko_tko"] + empirical["a_submission"] + empirical["a_decision"]
    b_win_emp = 1.0 - a_win_emp
    goes_distance = theoretical["a_decision"] + theoretical["b_decision"]
    finish_probability = 1.0 - goes_distance

    all_paths = {
        f"{fighter_a} by KO/TKO": theoretical["a_ko_tko"],
        f"{fighter_a} by Submission": theoretical["a_submission"],
        f"{fighter_a} by Decision": theoretical["a_decision"],
        f"{fighter_b} by KO/TKO": theoretical["b_ko_tko"],
        f"{fighter_b} by Submission": theoretical["b_submission"],
        f"{fighter_b} by Decision": theoretical["b_decision"],
    }
    most_likely_path, most_likely_path_probability = max(all_paths.items(), key=lambda item: item[1])

    likely_finish_round = None
    likely_finish_round_probability = 0.0
    if finish_count:
        likely_finish_round = int(np.argmax(round_counts) + 1)
        likely_finish_round_probability = float(round_counts[likely_finish_round - 1] / finish_count)

    # Volatility expresses how uncertain the fight path is, not confidence in the side.
    concentration = max(all_paths.values())
    if concentration >= 0.36:
        volatility = "Lower"
    elif concentration >= 0.25:
        volatility = "Moderate"
    else:
        volatility = "High"

    round_state_projection = []
    for idx in range(rounds):
        total = float(unconditional_round_finish[idx])
        a_share = float(side_a_round_finish[idx] / total) if total > 0 else 0.5
        ko_share = float(ko_round_finish[idx] / total) if total > 0 else 0.5
        if a_share >= 0.57:
            leader = fighter_a
        elif a_share <= 0.43:
            leader = fighter_b
        else:
            leader = "Balanced"
        method_label = "KO/TKO-heavy" if ko_share >= 0.62 else ("Submission-heavy" if ko_share <= 0.38 else "Mixed finish paths")
        round_state_projection.append({
            "round": idx + 1,
            "unconditional_finish_probability": total,
            "probability_given_finish": float(round_probs[idx]),
            "fighter_a_finish_share": a_share,
            "fighter_b_finish_share": 1.0 - a_share,
            "ko_share": ko_share,
            "submission_share": 1.0 - ko_share,
            "finish_edge": leader,
            "state": method_label,
        })

    return {
        "available": True,
        "version": SIMULATION_VERSION,
        "simulations": int(config.simulations),
        "seed": int(config.seed),
        "rounds": rounds,
        "model_win_probability_a": p_a,
        "model_win_probability_b": p_b,
        "simulated_win_probability_a": a_win_emp,
        "simulated_win_probability_b": b_win_emp,
        "a_ko_tko_probability": theoretical["a_ko_tko"],
        "a_submission_probability": theoretical["a_submission"],
        "a_decision_probability": theoretical["a_decision"],
        "b_ko_tko_probability": theoretical["b_ko_tko"],
        "b_submission_probability": theoretical["b_submission"],
        "b_decision_probability": theoretical["b_decision"],
        "goes_distance_probability": goes_distance,
        "finish_probability": finish_probability,
        "most_likely_path": most_likely_path,
        "most_likely_path_probability": float(most_likely_path_probability),
        "likely_finish_round": likely_finish_round,
        "likely_finish_round_probability_given_finish": likely_finish_round_probability,
        "finish_round_probabilities_given_finish": {
            f"Round {i + 1}": float(round_probs[i]) for i in range(rounds)
        },
        "round_state_projection": round_state_projection,
        "volatility": volatility,
        "fighter_a_finish_profile": path_a,
        "fighter_b_finish_profile": path_b,
        "cardio_timing_used": bool((cardio_a or {}).get("available") or (cardio_b or {}).get("available")),
        "damage_method_context_used": bool((damage_a or {}).get("available") or (damage_b or {}).get("available")),
        "advanced_striking_path_context_used": bool((striking_matchup or {}).get("available")),
        "advanced_grappling_path_context_used": bool((grappling_matchup or {}).get("available")),
        "guardrail": (
            "Simulation consumes Macabets' final win probability; it does not re-predict the winner. "
            "Method and round-state probabilities are conditional decompositions using recent finish tendencies, durability, knockdown/submission pressure, advanced striking/grappling compatibility, cardio retention, damage/recovery context and scheduled rounds. "
            "Advanced matchup, cardio and damage signals change how and when the modeled winner paths occur; the simulator preserves Macabets' already-finalized side win probability and never creates a competing winner model."
        ),
    }
