from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np


JUDGING_VERSION = "Macabets UFC Judging v0.1 — Round/Card Simulation"


@dataclass(frozen=True)
class UFCJudgingConfig:
    simulations: int = 12000
    seed: int = 9137
    judge_noise_sd: float = 0.055
    max_round_tilt: float = 0.085
    max_ten_eight_probability: float = 0.08


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _num(mapping: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(mapping, dict):
        return default
    try:
        value = float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) else value


def _signal(matchup: dict[str, Any] | None) -> tuple[float, float]:
    if not isinstance(matchup, dict) or not matchup.get("available"):
        return 0.0, 0.0
    gap = _clip(_num(matchup, "weighted_gap") / 50.0, -1.0, 1.0)
    reliability = _clip(_num(matchup, "reliability"), 0.0, 1.0)
    return gap, reliability


def _cardio_retention(profile: dict[str, Any] | None) -> tuple[float, float]:
    if not isinstance(profile, dict) or not profile.get("available"):
        return 1.0, 0.0
    return (
        _clip(_num(profile, "retention", 1.0), 0.55, 1.30),
        _clip(_num(profile, "reliability"), 0.0, 1.0),
    )


def _round_shape(
    rounds: int,
    *,
    striking_matchup: dict[str, Any] | None,
    grappling_matchup: dict[str, Any] | None,
    cardio_a: dict[str, Any] | None,
    cardio_b: dict[str, Any] | None,
    config: UFCJudgingConfig,
) -> np.ndarray:
    strike_gap, strike_rel = _signal(striking_matchup)
    grapple_gap, grapple_rel = _signal(grappling_matchup)
    retention_a, cardio_rel_a = _cardio_retention(cardio_a)
    retention_b, cardio_rel_b = _cardio_retention(cardio_b)

    # Striking/grappling edges influence round control, while cardio changes the *shape*
    # of that edge through the fight. The final decision-side probability is reconciled
    # later, so this layer cannot create a second moneyline model.
    matchup_edge = 0.55 * strike_gap * strike_rel + 0.45 * grapple_gap * grapple_rel
    cardio_edge = ((retention_a - retention_b) * min(cardio_rel_a, cardio_rel_b))
    axis = np.linspace(-1.0, 1.0, rounds)
    shape = (
        0.028 * matchup_edge
        + 0.055 * cardio_edge * axis
    )
    return np.clip(shape, -config.max_round_tilt, config.max_round_tilt)




def _single_judge_win_probability(round_probs_a: np.ndarray) -> float:
    # Exact Poisson-binomial probability of winning a majority of scheduled rounds.
    # Used only for fast calibration; the final three-judge simulation below still
    # includes judge-level noise and rare 10-8 rounds.
    dist = np.array([1.0], dtype=float)
    for p in round_probs_a:
        dist = np.convolve(dist, np.array([1.0 - float(p), float(p)], dtype=float))
    needed = len(round_probs_a) // 2 + 1
    return float(dist[needed:].sum())


def _panel_win_probability(round_probs_a: np.ndarray) -> float:
    q = _single_judge_win_probability(round_probs_a)
    return float(q ** 3 + 3.0 * q * q * (1.0 - q))

def _simulate_cards(round_probs_a: np.ndarray, config: UFCJudgingConfig, rng: np.random.Generator) -> dict[str, Any]:
    sims = int(config.simulations)
    a_card_wins = np.zeros(sims, dtype=int)
    split_flags = np.zeros(sims, dtype=bool)
    unanimous_flags = np.zeros(sims, dtype=bool)
    majority_flags = np.zeros(sims, dtype=bool)

    scoreline_counts: dict[str, int] = {}
    judge_margin_abs: list[float] = []

    for s in range(sims):
        judge_scores = [[0, 0] for _ in range(3)]
        # Judges observe the same underlying round, then a small disagreement layer
        # creates realistic card dispersion. Treating all judge round calls as fully
        # independent dramatically overstates split decisions.
        for p in round_probs_a:
            latent_a_round = bool(rng.random() < float(p))
            closeness = 1.0 - min(1.0, abs(float(p) - 0.5) / 0.5)
            disagree_p = _clip(0.035 + 0.115 * closeness, 0.03, 0.15)
            dominance = abs(float(p) - 0.5) * 2.0
            ten_eight_p = min(config.max_ten_eight_probability, max(0.0, dominance - 0.58) * 0.10)
            for j in range(3):
                judge_a_round = latent_a_round
                if rng.random() < disagree_p:
                    judge_a_round = not judge_a_round
                ten_eight = bool(rng.random() < ten_eight_p)
                if judge_a_round:
                    judge_scores[j][0] += 10
                    judge_scores[j][1] += 8 if ten_eight else 9
                else:
                    judge_scores[j][1] += 10
                    judge_scores[j][0] += 8 if ten_eight else 9

        judge_votes = []
        for a_points, b_points in judge_scores:
            if a_points > b_points:
                vote = 1
            elif a_points < b_points:
                vote = -1
            else:
                vote = 0
            judge_votes.append(vote)
            judge_margin_abs.append(abs(a_points - b_points))
            label = f"{max(a_points, b_points)}-{min(a_points, b_points)}"
            scoreline_counts[label] = scoreline_counts.get(label, 0) + 1

        a_votes = sum(v == 1 for v in judge_votes)
        b_votes = sum(v == -1 for v in judge_votes)
        if a_votes > b_votes:
            a_card_wins[s] = 1
        elif b_votes > a_votes:
            a_card_wins[s] = 0
        else:
            a_card_wins[s] = int(rng.random() < float(np.mean(round_probs_a)))

        unanimous_flags[s] = (a_votes == 3 or b_votes == 3)
        split_flags[s] = sorted([a_votes, b_votes]) == [1, 2]
        majority_flags[s] = not unanimous_flags[s] and not split_flags[s]

    return {
        "a_card_win_probability": float(a_card_wins.mean()),
        "unanimous_probability": float(unanimous_flags.mean()),
        "split_probability": float(split_flags.mean()),
        "majority_or_draw_probability": float(majority_flags.mean()),
        "most_common_judge_score": max(scoreline_counts.items(), key=lambda item: item[1])[0] if scoreline_counts else "—",
        "average_absolute_judge_margin": float(np.mean(judge_margin_abs)) if judge_margin_abs else 0.0,
    }


def build_judging_projection(
    fighter_a: str,
    fighter_b: str,
    *,
    decision_probability_a: float,
    decision_probability_b: float,
    striking_matchup: dict[str, Any] | None = None,
    grappling_matchup: dict[str, Any] | None = None,
    cardio_a: dict[str, Any] | None = None,
    cardio_b: dict[str, Any] | None = None,
    rounds: int = 3,
    config: UFCJudgingConfig | None = None,
) -> dict[str, Any]:
    """Project round-winning and judge-card uncertainty conditional on a decision.

    The target conditional decision winner share comes from the existing Macabets
    simulation. Round-state matchup signals only shape *which rounds are more likely*
    to swing to either fighter. A calibration offset is solved so the judge-card model
    stays aligned with the existing decision-side probability rather than re-pricing it.
    """
    config = config or UFCJudgingConfig()
    rounds = int(rounds)
    if rounds not in {3, 5}:
        raise ValueError("UFC judging supports 3 or 5 rounds.")

    decision_total = float(decision_probability_a) + float(decision_probability_b)
    if decision_total <= 1e-9:
        return {"available": False, "version": JUDGING_VERSION, "rounds": rounds}

    target_a = _clip(float(decision_probability_a) / decision_total, 0.03, 0.97)
    shape = _round_shape(
        rounds,
        striking_matchup=striking_matchup,
        grappling_matchup=grappling_matchup,
        cardio_a=cardio_a,
        cardio_b=cardio_b,
        config=config,
    )

    # Deterministically solve a common round-probability offset. Card-win probability
    # is estimated with fixed random streams so the bisection is stable and reproducible.
    base_seed = int(config.seed)
    low, high = -0.42, 0.42
    best_probs = np.full(rounds, target_a, dtype=float)
    probe_cfg = UFCJudgingConfig(
        simulations=min(2500, int(config.simulations)),
        seed=base_seed,
        judge_noise_sd=config.judge_noise_sd,
        max_round_tilt=config.max_round_tilt,
        max_ten_eight_probability=config.max_ten_eight_probability,
    )
    for _ in range(10):
        mid = (low + high) / 2.0
        probs = np.clip(0.5 + mid + shape, 0.08, 0.92)
        best_probs = probs
        probe = _simulate_cards(probs, probe_cfg, np.random.default_rng(base_seed))
        if probe["a_card_win_probability"] < target_a:
            low = mid
        else:
            high = mid

    final_projection = _simulate_cards(best_probs, config, np.random.default_rng(base_seed + 1))
    a_card = float(final_projection["a_card_win_probability"])
    b_card = 1.0 - a_card

    round_rows = []
    for idx, p in enumerate(best_probs):
        if p >= 0.57:
            edge = fighter_a
        elif p <= 0.43:
            edge = fighter_b
        else:
            edge = "Close"
        round_rows.append({
            "round": idx + 1,
            "fighter_a_round_win_probability": float(p),
            "fighter_b_round_win_probability": float(1.0 - p),
            "round_edge": edge,
        })

    closeness = 1.0 - min(1.0, abs(a_card - 0.5) / 0.5)
    controversy_risk = _clip(0.55 * final_projection["split_probability"] + 0.45 * closeness, 0.0, 1.0)
    if controversy_risk >= 0.62:
        controversy_label = "High"
    elif controversy_risk >= 0.38:
        controversy_label = "Moderate"
    else:
        controversy_label = "Lower"

    return {
        "available": True,
        "version": JUDGING_VERSION,
        "rounds": rounds,
        "conditional_decision_probability_a": target_a,
        "conditional_decision_probability_b": 1.0 - target_a,
        "judge_card_win_probability_a": a_card,
        "judge_card_win_probability_b": b_card,
        "round_win_probabilities": round_rows,
        "unanimous_decision_probability": float(final_projection["unanimous_probability"]),
        "split_decision_probability": float(final_projection["split_probability"]),
        "majority_or_draw_probability": float(final_projection["majority_or_draw_probability"]),
        "most_common_judge_score": final_projection["most_common_judge_score"],
        "average_absolute_judge_margin": float(final_projection["average_absolute_judge_margin"]),
        "judging_uncertainty": controversy_label,
        "judging_uncertainty_score": controversy_risk,
        "simulations": int(config.simulations),
        "guardrail": (
            "Judge-card simulation is conditional on the fight reaching a decision and is calibrated back to Macabets' existing decision-side probability. "
            "Striking, grappling and cardio only shape round-by-round scoring paths; this module does not add a second moneyline adjustment."
        ),
    }
