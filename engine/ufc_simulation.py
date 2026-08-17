from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np


SIMULATION_VERSION = "Macabets UFC Simulation v0.1"


@dataclass(frozen=True)
class UFCSimulationConfig:
    simulations: int = 20000
    seed: int = 2417
    min_finish_probability: float = 0.12
    max_finish_probability_3r: float = 0.82
    max_finish_probability_5r: float = 0.88


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
    rounds: int,
    config: UFCSimulationConfig,
) -> dict[str, float]:
    own_finish = _pct01(_num(recent, "finish_rate"), 0.42)
    opp_finish_loss = _pct01(_num(opponent_recent, "loss_finish_rate"), 0.30)
    kd_pressure = _pct01(_num(performance, "kd_per15_pct"), 0.50)
    sub_pressure = _pct01(_num(performance, "sub_attempts_per15_pct"), 0.50)
    opp_durability = _pct01(_num(opponent_performance, "durability_score"), 0.50)
    pace = _pct01(_num(performance, "pace_score"), 0.50)

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
    ko_share = _clip(0.72 * ko_base + 0.28 * trait_ko, 0.15, 0.92)
    sub_share = 1.0 - ko_share

    return {
        "finish_probability": finish_probability,
        "ko_share_of_finishes": ko_share,
        "submission_share_of_finishes": sub_share,
        "decision_probability_if_win": 1.0 - finish_probability,
    }


def _round_weights(rounds: int, pace_a: float, pace_b: float) -> np.ndarray:
    if int(rounds) == 5:
        base = np.array([0.31, 0.24, 0.19, 0.15, 0.11], dtype=float)
    else:
        base = np.array([0.44, 0.33, 0.23], dtype=float)
    avg_pace = (_pct01(pace_a, 0.5) + _pct01(pace_b, 0.5)) / 2.0
    # High pace modestly pulls finishes earlier; low pace shifts mass later.
    tilt = (avg_pace - 0.5) * 0.18
    indices = np.linspace(1.0, -1.0, len(base))
    adjusted = np.maximum(0.01, base * (1.0 + tilt * indices))
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

    path_a = _finish_profile(recent_a, performance_a, recent_b, performance_b, rounds=rounds, config=config)
    path_b = _finish_profile(recent_b, performance_b, recent_a, performance_a, rounds=rounds, config=config)

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

    finish_mask = np.array(["decision" not in labels[i] for i in draws], dtype=bool)
    finish_count = int(finish_mask.sum())
    round_counts = np.zeros(rounds, dtype=int)
    if finish_count:
        round_probs = _round_weights(
            rounds,
            _num(performance_a, "pace_score", 50.0) or 50.0,
            _num(performance_b, "pace_score", 50.0) or 50.0,
        )
        sampled_rounds = rng.choice(np.arange(1, rounds + 1), size=finish_count, p=round_probs)
        round_counts = np.bincount(sampled_rounds, minlength=rounds + 1)[1:]
    else:
        round_probs = _round_weights(rounds, 50.0, 50.0)

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
        "volatility": volatility,
        "fighter_a_finish_profile": path_a,
        "fighter_b_finish_profile": path_b,
        "guardrail": (
            "Simulation consumes Macabets' final win probability; it does not re-predict the winner. "
            "Method probabilities are conditional decompositions using recent finish tendencies, durability, knockdown/submission pressure, pace and scheduled rounds."
        ),
    }
