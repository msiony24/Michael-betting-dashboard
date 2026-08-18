from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import pandas as pd


STYLE_VERSION = "Macabets UFC Style Matchups v0.3 — Advanced Striking + Grappling"


@dataclass(frozen=True)
class UFCStyleConfig:
    min_sample_for_full_weight: int = 6
    max_probability_adjustment: float = 0.03


def _num(profile: dict[str, Any], key: str) -> float | None:
    value = profile.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _avg(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else None


def _attack_vs_defense(
    attack_profile: dict[str, Any],
    defense_profile: dict[str, Any],
    attack_keys: list[str],
    defense_keys: list[str],
) -> tuple[float | None, int]:
    attack = _avg([_num(attack_profile, key) for key in attack_keys])
    defense = _avg([_num(defense_profile, key) for key in defense_keys])
    used = int(attack is not None) + int(defense is not None)
    if attack is None or defense is None:
        return None, used
    # Percentile-based residual: positive means the attack has historically been
    # stronger than the opponent's corresponding defense.
    return float(attack - defense), used


def _edge_label(gap: float) -> str:
    magnitude = abs(float(gap))
    if magnitude < 5.0:
        return "Even"
    if magnitude < 12.0:
        return "Slight"
    if magnitude < 22.0:
        return "Moderate"
    return "Clear"


def _advantage(gap: float, fighter_a: str, fighter_b: str) -> str:
    if abs(gap) < 5.0:
        return "Even"
    return fighter_a if gap > 0 else fighter_b


def _archetype(profile: dict[str, Any]) -> str:
    distance = _num(profile, "distance_strike_share")
    clinch = _num(profile, "clinch_strike_share")
    ground = _num(profile, "ground_strike_share")
    td = _num(profile, "td_per15_pct")
    subs = _num(profile, "sub_attempts_per15_pct")
    pace = _num(profile, "pace_score")

    if ground is not None and ground >= 0.28 and (td or 0) >= 60:
        return "Ground-pressure wrestler"
    if subs is not None and subs >= 70 and (td or 0) >= 50:
        return "Submission-oriented grappler"
    if clinch is not None and clinch >= 0.22:
        return "Clinch-pressure fighter"
    if distance is not None and distance >= 0.72:
        return "Distance striker"
    if pace is not None and pace >= 70:
        return "High-pace pressure fighter"
    return "Balanced / mixed style"


def build_style_matchup(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    fighter_a: str,
    fighter_b: str,
    *,
    rounds: int = 3,
    advanced_grappling: dict[str, Any] | None = None,
    advanced_striking: dict[str, Any] | None = None,
    config: UFCStyleConfig | None = None,
) -> dict[str, Any]:
    """Build opponent-specific style interactions from underlying UFC performance traits.

    This layer intentionally uses attack-vs-defense residuals rather than standalone
    composite scores. That makes it a compatibility layer instead of a second copy of
    the general performance engine.
    """
    config = config or UFCStyleConfig()

    if advanced_grappling and advanced_grappling.get("available"):
        specs = []
        if not (advanced_striking and advanced_striking.get("available")):
            specs.append({
                "category": "Striking offense vs defense",
                "weight": 0.38 if int(rounds) == 3 else 0.34,
                "a_attack": ["sig_accuracy_pct", "kd_per15_pct"],
                "b_defense": ["sig_defense_pct", "kd_absorbed_per15_pct"],
                "b_attack": ["sig_accuracy_pct", "kd_per15_pct"],
                "a_defense": ["sig_defense_pct", "kd_absorbed_per15_pct"],
                "why_a": f"{fighter_a}'s striking efficiency and knockdown pressure line up favorably with {fighter_b}'s strike defense/durability profile.",
                "why_b": f"{fighter_b}'s striking efficiency and knockdown pressure line up favorably with {fighter_a}'s strike defense/durability profile.",
            })
        specs.append({
            "category": "Pace and attrition compatibility",
            "weight": 0.14 if int(rounds) == 3 else 0.18,
            "a_attack": ["pace_score"],
            "b_defense": ["durability_score", "pace_score"],
            "b_attack": ["pace_score"],
            "a_defense": ["durability_score", "pace_score"],
            "why_a": f"{fighter_a}'s pace is more likely to create an attritional advantage against {fighter_b}'s durability/pace profile.",
            "why_b": f"{fighter_b}'s pace is more likely to create an attritional advantage against {fighter_a}'s durability/pace profile.",
        })
    else:
        specs = [
            {
                "category": "Striking offense vs defense",
                "weight": 0.36 if int(rounds) == 3 else 0.32,
                "a_attack": ["sig_accuracy_pct", "kd_per15_pct"],
                "b_defense": ["sig_defense_pct", "kd_absorbed_per15_pct"],
                "b_attack": ["sig_accuracy_pct", "kd_per15_pct"],
                "a_defense": ["sig_defense_pct", "kd_absorbed_per15_pct"],
                "why_a": f"{fighter_a}'s striking efficiency and knockdown pressure line up favorably with {fighter_b}'s strike defense/durability profile.",
                "why_b": f"{fighter_b}'s striking efficiency and knockdown pressure line up favorably with {fighter_a}'s strike defense/durability profile.",
            },
            {
                "category": "Wrestling pressure vs takedown defense",
                "weight": 0.30 if int(rounds) == 3 else 0.28,
                "a_attack": ["td_per15_pct", "td_accuracy_pct", "control_share_pct"],
                "b_defense": ["td_defense_pct"],
                "b_attack": ["td_per15_pct", "td_accuracy_pct", "control_share_pct"],
                "a_defense": ["td_defense_pct"],
                "why_a": f"{fighter_a}'s takedown and control pressure is better matched to {fighter_b}'s takedown defense than the reverse interaction.",
                "why_b": f"{fighter_b}'s takedown and control pressure is better matched to {fighter_a}'s takedown defense than the reverse interaction.",
            },
            {
                "category": "Grappling threat vs defensive resistance",
                "weight": 0.18,
                "a_attack": ["sub_attempts_per15_pct", "control_share_pct"],
                "b_defense": ["td_defense_pct", "durability_score"],
                "b_attack": ["sub_attempts_per15_pct", "control_share_pct"],
                "a_defense": ["td_defense_pct", "durability_score"],
                "why_a": f"{fighter_a}'s submission/control pressure presents the more difficult grappling problem for {fighter_b}.",
                "why_b": f"{fighter_b}'s submission/control pressure presents the more difficult grappling problem for {fighter_a}.",
            },
            {
                "category": "Pace and attrition compatibility",
                "weight": 0.16 if int(rounds) == 3 else 0.22,
                "a_attack": ["pace_score"],
                "b_defense": ["durability_score", "pace_score"],
                "b_attack": ["pace_score"],
                "a_defense": ["durability_score", "pace_score"],
                "why_a": f"{fighter_a}'s pace is more likely to create an attritional advantage against {fighter_b}'s durability/pace profile.",
                "why_b": f"{fighter_b}'s pace is more likely to create an attritional advantage against {fighter_a}'s durability/pace profile.",
            },
        ]

    rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    used_weight = 0.0
    available_interactions = 0

    for spec in specs:
        a_vs_b, _ = _attack_vs_defense(
            profile_a, profile_b, spec["a_attack"], spec["b_defense"]
        )
        b_vs_a, _ = _attack_vs_defense(
            profile_b, profile_a, spec["b_attack"], spec["a_defense"]
        )
        if a_vs_b is None or b_vs_a is None:
            continue

        # Compare the two directional compatibility residuals. This makes the
        # matchup row symmetric and prevents rewarding raw offense by itself.
        gap = float(a_vs_b - b_vs_a)
        weight = float(spec["weight"])
        weighted_gap += gap * weight
        used_weight += weight
        available_interactions += 1
        leader = _advantage(gap, fighter_a, fighter_b)
        if leader == fighter_a:
            why = spec["why_a"]
        elif leader == fighter_b:
            why = spec["why_b"]
        else:
            why = "The attack-versus-defense interaction is close enough that this area does not create a meaningful matchup edge."
        rows.append(
            {
                "category": spec["category"],
                "advantage": leader,
                "strength": _edge_label(gap),
                "interaction_gap": gap,
                "why": why,
            }
        )

    if advanced_striking and advanced_striking.get("available"):
        advanced_weight_total = 0.38 if int(rounds) == 3 else 0.34
        for advanced in list(advanced_striking.get("rows", [])):
            gap = float(advanced.get("interaction_gap", 0.0) or 0.0)
            local_weight = float(advanced.get("weight", 0.0) or 0.0) * advanced_weight_total
            if local_weight <= 0:
                continue
            weighted_gap += gap * local_weight
            used_weight += local_weight
            available_interactions += 1
            rows.append({
                "category": advanced.get("category", "Advanced striking"),
                "advantage": advanced.get("advantage", "Even"),
                "strength": advanced.get("strength", _edge_label(gap)),
                "interaction_gap": gap,
                "why": advanced.get("why", "Advanced striking interaction."),
                "advanced_striking": True,
            })

    if advanced_grappling and advanced_grappling.get("available"):
        advanced_weight_total = 0.48 if int(rounds) == 3 else 0.48
        advanced_rows = list(advanced_grappling.get("rows", []))
        for advanced in advanced_rows:
            gap = float(advanced.get("interaction_gap", 0.0) or 0.0)
            local_weight = float(advanced.get("weight", 0.0) or 0.0) * advanced_weight_total
            if local_weight <= 0:
                continue
            weighted_gap += gap * local_weight
            used_weight += local_weight
            available_interactions += 1
            rows.append({
                "category": advanced.get("category", "Advanced grappling"),
                "advantage": advanced.get("advantage", "Even"),
                "strength": advanced.get("strength", _edge_label(gap)),
                "interaction_gap": gap,
                "why": advanced.get("why", "Advanced wrestling/grappling interaction."),
                "advanced_grappling": True,
            })

    if used_weight <= 0 or available_interactions == 0:
        return {
            "available": False,
            "adjustment_a": 0.0,
            "weighted_gap": 0.0,
            "reliability": 0.0,
            "rows": [],
            "fighter_a_archetype": _archetype(profile_a),
            "fighter_b_archetype": _archetype(profile_b),
        }

    weighted_gap /= used_weight
    sample_reliability = min(
        1.0,
        min(int(profile_a.get("sample", 0) or 0), int(profile_b.get("sample", 0) or 0))
        / float(config.min_sample_for_full_weight),
    )
    completeness = min(
        float(profile_a.get("data_completeness", 0.0) or 0.0),
        float(profile_b.get("data_completeness", 0.0) or 0.0),
    )
    expected_interactions = len(specs)
    if advanced_striking and advanced_striking.get("available"):
        expected_interactions += len(advanced_striking.get("rows", []))
    if advanced_grappling and advanced_grappling.get("available"):
        expected_interactions += len(advanced_grappling.get("rows", []))
    interaction_coverage = available_interactions / float(max(expected_interactions, 1))
    reliability = sample_reliability * (0.40 + 0.60 * completeness) * interaction_coverage

    raw_adjustment = (weighted_gap / 55.0) * config.max_probability_adjustment
    adjustment = float(
        np.clip(
            raw_adjustment * reliability,
            -config.max_probability_adjustment,
            config.max_probability_adjustment,
        )
    )

    return {
        "available": True,
        "adjustment_a": adjustment,
        "weighted_gap": float(weighted_gap),
        "reliability": float(reliability),
        "rows": rows,
        "fighter_a_archetype": _archetype(profile_a),
        "fighter_b_archetype": _archetype(profile_b),
        "five_round_weighting": int(rounds) == 5,
        "guardrail": (
            "Style Matchups uses opponent-specific attack-vs-defense residuals, is capped at ±3 percentage points, "
            "and is reliability-shrunk so it cannot simply re-award standalone performance strength. When available, advanced target/range/power striking replaces the generic striking row, and advanced chain-wrestling, control/escape and submission interactions replace the generic wrestling/grappling rows inside this same cap."
        ),
    }
