from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import pandas as pd

from engine.ufc_performance import build_performance_table, fighter_performance


OPPONENT_ADJUSTMENT_VERSION = "Macabets UFC Opponent-Adjusted Skills v0.1"


@dataclass(frozen=True)
class UFCOpponentAdjustmentConfig:
    recent_fights: int = 8
    min_opponents_for_full_weight: int = 6
    max_skill_adjustment: float = 8.0
    adjustment_strength: float = 0.30


def _num(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else None


def _score(profile: dict[str, Any], keys: list[str]) -> float | None:
    return _mean([_num(profile, key) for key in keys])


def _skill_scores(profile: dict[str, Any]) -> dict[str, float | None]:
    return {
        "striking_offense": _score(profile, ["sig_accuracy_pct", "sig_diff_per_min_pct", "kd_per15_pct"]),
        "striking_defense": _score(profile, ["sig_defense_pct", "kd_absorbed_per15_pct"]),
        "wrestling_offense": _score(profile, ["td_per15_pct", "td_accuracy_pct", "control_share_pct"]),
        "wrestling_defense": _score(profile, ["td_defense_pct", "control_share_pct"]),
        "grappling_offense": _score(profile, ["sub_attempts_per15_pct", "control_share_pct", "td_per15_pct"]),
        "grappling_defense": _score(profile, ["td_defense_pct", "durability_score"]),
        "power": _score(profile, ["kd_per15_pct", "sig_diff_per_min_pct"]),
        "durability": _score(profile, ["durability_score", "sig_defense_pct", "kd_absorbed_per15_pct"]),
        "pace": _score(profile, ["pace_score"]),
    }


OPPOSING_SKILL = {
    "striking_offense": "striking_defense",
    "striking_defense": "striking_offense",
    "wrestling_offense": "wrestling_defense",
    "wrestling_defense": "wrestling_offense",
    "grappling_offense": "grappling_defense",
    "grappling_defense": "grappling_offense",
    "power": "durability",
    "durability": "power",
    "pace": "pace",
}


def _recent_opponents(fights: pd.DataFrame, fighter: str, limit: int) -> list[str]:
    if fights is None or fights.empty or "fighter" not in fights.columns or "opponent" not in fights.columns:
        return []
    frame = fights.copy()
    if "event_date" in frame.columns:
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
        frame = frame.sort_values("event_date", ascending=False)
    rows = frame.loc[
        frame["fighter"].astype(str).str.casefold() == str(fighter).strip().casefold()
    ].head(limit)
    return [str(v).strip() for v in rows["opponent"].dropna().tolist() if str(v).strip()]


def _quality_label(value: float | None) -> str:
    if value is None:
        return "Unknown"
    if value >= 65:
        return "Elite"
    if value >= 57:
        return "Strong"
    if value <= 35:
        return "Weak"
    if value <= 43:
        return "Below average"
    return "Average"


def _adjust_value(base: float | None, opponent_quality: float | None, reliability: float, config: UFCOpponentAdjustmentConfig) -> tuple[float | None, float]:
    if base is None or opponent_quality is None:
        return base, 0.0
    raw = (opponent_quality - 50.0) * config.adjustment_strength * reliability
    movement = float(np.clip(raw, -config.max_skill_adjustment, config.max_skill_adjustment))
    return float(np.clip(base + movement, 0.0, 100.0)), movement


def _reference_lookup(fights: pd.DataFrame, ratings: pd.DataFrame) -> dict[str, dict[str, Any]]:
    # Historical/inactive opponents matter to strength of competition. Build a
    # reference table across the full rating universe, while the selected fighter's
    # displayed base performance table can remain restricted to the active pool.
    reference = build_performance_table(fights, ratings, active_only=False)
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in reference.iterrows():
        name = str(row.get("fighter", "")).strip()
        if name:
            lookup[name.casefold()] = row.to_dict()
    return lookup


def adjust_fighter_profile(
    fighter: str,
    base_profile: dict[str, Any],
    fights: pd.DataFrame,
    reference_lookup: dict[str, dict[str, Any]],
    *,
    config: UFCOpponentAdjustmentConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    opponents = _recent_opponents(fights, fighter, config.recent_fights)
    opponent_profiles = [reference_lookup.get(name.casefold()) for name in opponents]
    opponent_profiles = [p for p in opponent_profiles if p]
    reliability = min(1.0, len(opponent_profiles) / float(config.min_opponents_for_full_weight))

    base_skills = _skill_scores(base_profile)
    opponent_skill_rows = [_skill_scores(profile) for profile in opponent_profiles]
    adjusted_skills: dict[str, float | None] = {}
    skill_rows: list[dict[str, Any]] = []

    for skill, base in base_skills.items():
        opposing = OPPOSING_SKILL[skill]
        quality = _mean([row.get(opposing) for row in opponent_skill_rows])
        adjusted, movement = _adjust_value(base, quality, reliability, config)
        adjusted_skills[skill] = adjusted
        skill_rows.append({
            "skill": skill.replace("_", " ").title(),
            "base_score": base,
            "opponent_quality": quality,
            "opponent_quality_label": _quality_label(quality),
            "adjusted_score": adjusted,
            "adjustment": movement,
        })

    profile = dict(base_profile)

    # Feed opponent quality back into the exact percentile traits consumed by
    # Performance and Style Matchups. This upgrades the existing layers instead of
    # stacking a new independent probability modifier on top of correlated stats.
    trait_opposition = {
        "sig_accuracy_pct": "striking_defense",
        "sig_diff_per_min_pct": "striking_defense",
        "kd_per15_pct": "durability",
        "sig_defense_pct": "striking_offense",
        "kd_absorbed_per15_pct": "power",
        "td_per15_pct": "wrestling_defense",
        "td_accuracy_pct": "wrestling_defense",
        "control_share_pct": "wrestling_defense",
        "td_defense_pct": "wrestling_offense",
        "sub_attempts_per15_pct": "grappling_defense",
    }
    for trait, opposing_skill in trait_opposition.items():
        base = _num(profile, trait)
        quality = _mean([row.get(opposing_skill) for row in opponent_skill_rows])
        adjusted, _ = _adjust_value(base, quality, reliability, config)
        if adjusted is not None:
            profile[trait] = adjusted

    profile["striking_score"] = _mean([adjusted_skills["striking_offense"], adjusted_skills["striking_defense"], adjusted_skills["power"]])
    profile["wrestling_score"] = _mean([adjusted_skills["wrestling_offense"], adjusted_skills["wrestling_defense"]])
    profile["grappling_score"] = _mean([adjusted_skills["grappling_offense"], adjusted_skills["grappling_defense"]])
    profile["durability_score"] = adjusted_skills["durability"]
    profile["pace_score"] = adjusted_skills["pace"]
    profile["opponent_adjustment_reliability"] = reliability
    profile["opponent_sample"] = len(opponent_profiles)

    overall_quality = _mean([
        _mean(list(row.values())) for row in opponent_skill_rows if row
    ])
    report = {
        "fighter": fighter,
        "available": bool(opponent_profiles),
        "opponent_sample": len(opponent_profiles),
        "requested_sample": min(len(opponents), config.recent_fights),
        "reliability": reliability,
        "overall_opponent_skill_quality": overall_quality,
        "overall_opponent_quality_label": _quality_label(overall_quality),
        "skills": skill_rows,
        "opponents_used": opponents[: config.recent_fights],
    }
    return profile, report


def build_opponent_adjusted_matchup(
    fighter_a: str,
    fighter_b: str,
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    fights: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    config: UFCOpponentAdjustmentConfig | None = None,
) -> dict[str, Any]:
    config = config or UFCOpponentAdjustmentConfig()
    lookup = _reference_lookup(fights, ratings)
    adjusted_a, report_a = adjust_fighter_profile(
        fighter_a, profile_a, fights, lookup, config=config
    )
    adjusted_b, report_b = adjust_fighter_profile(
        fighter_b, profile_b, fights, lookup, config=config
    )
    reliability = min(float(report_a["reliability"]), float(report_b["reliability"]))
    return {
        "available": bool(report_a["available"] and report_b["available"]),
        "version": OPPONENT_ADJUSTMENT_VERSION,
        "fighter_a_profile": adjusted_a,
        "fighter_b_profile": adjusted_b,
        "fighter_a_report": report_a,
        "fighter_b_report": report_b,
        "reliability": reliability,
        "guardrail": (
            "Opponent-adjusted skills transform the existing Performance and Style inputs rather than adding another independent line adjustment. "
            "Recent opponents are evaluated by the specific skill needed to test the fighter, and every adjustment is sample-shrunk and capped at ±8 skill points."
        ),
    }
