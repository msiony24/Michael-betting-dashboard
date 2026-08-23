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


def _experience_credit(player):
    """Give experienced backups credit without replacing talent evaluation."""
    starts = _num(player.get("career_starts", player.get("starts", 0)), 0)
    attempts = _num(player.get("career_attempts", player.get("attempts", 0)), 0)
    if starts >= 25 or attempts >= 750:
        return 3.0, "veteran experience"
    if starts >= 8 or attempts >= 250:
        return 1.5, "meaningful NFL experience"
    return 0.0, "limited NFL experience"


def apply_qb_replacement_adjustment(*, grade: float, healthy_starters, active_starters, depth):
    if healthy_starters is None or active_starters is None or healthy_starters.empty or active_starters.empty:
        return grade, {}

    starter = healthy_starters.iloc[0]
    replacement = active_starters.iloc[0]
    starter_rating = _num(starter.get("macabets_rating"))
    replacement_rating = _num(replacement.get("macabets_rating"))
    drop = max(0.0, starter_rating - replacement_rating)

    # Madden provides the talent baseline. Experience softens the replacement
    # penalty for proven backups without pretending they are the starter.
    experience_credit, experience_note = _experience_credit(replacement)
    effective_drop = max(0.0, drop - experience_credit)

    # Keep the impact conservative because team ratings already include QB.
    adjustment = -min(effective_drop * 0.25, 5.0)
    context = {
        "starter": str(starter.get("player_name", "")),
        "replacement": str(replacement.get("player_name", "")),
        "starter_rating": round(starter_rating, 2),
        "replacement_rating": round(replacement_rating, 2),
        "raw_rating_drop": round(drop, 2),
        "experience_credit": round(experience_credit, 2),
        "experience_context": experience_note,
        "effective_rating_drop": round(effective_drop, 2),
        "grade_adjustment": round(adjustment, 2),
        "severity": "Major" if effective_drop >= 15 else "Moderate" if effective_drop >= 8 else "Small",
    }
    return round(grade + adjustment, 2), context
