"""QB replacement intelligence.

Uses Madden/player ratings as the talent baseline for backups and applies a
small replacement adjustment only when QB1 is unavailable. It does not replace
normal QB ratings; it estimates the drop from the expected starter to the
actual replacement.
"""
from __future__ import annotations

from typing import Any


def _num(value: Any, default: float = 67.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def apply_qb_replacement_adjustment(*, grade: float, healthy_starters, active_starters, depth):
    if healthy_starters is None or active_starters is None or healthy_starters.empty or active_starters.empty:
        return grade, {}

    starter = healthy_starters.iloc[0]
    replacement = active_starters.iloc[0]
    starter_rating = _num(starter.get("macabets_rating"))
    replacement_rating = _num(replacement.get("macabets_rating"))
    drop = starter_rating - replacement_rating

    # Keep the impact conservative because team ratings already include QB.
    adjustment = -min(max(drop * 0.25, 0), 5.0)
    context = {
        "starter": str(starter.get("player_name", "")),
        "replacement": str(replacement.get("player_name", "")),
        "starter_rating": round(starter_rating, 2),
        "replacement_rating": round(replacement_rating, 2),
        "rating_drop": round(drop, 2),
        "grade_adjustment": round(adjustment, 2),
        "severity": "Major" if drop >= 15 else "Moderate" if drop >= 8 else "Small",
    }
    return round(grade + adjustment, 2), context
