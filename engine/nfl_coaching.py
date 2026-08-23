"""2026 NFL head-coach priors for Macabets.

The coaching layer is intentionally conservative. It uses only sourced head-coach
identity/experience and the coach's 2025 record when supplied. New coaches are
not automatically treated as bad coaches; they stay close to neutral until
current-season evidence exists.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COACHING_PATH = PROJECT_ROOT / "data" / "coaching_2026.csv"
NEUTRAL_COACHING_RATING = 70.0


def _record_parts(value: Any) -> tuple[int, int] | None:
    text = str(value or "").strip()
    if not text or text in {"--", "—", "nan", "None", "0-0"}:
        return None
    try:
        wins_text, losses_text = text.split("-", 1)
        wins, losses = int(wins_text), int(losses_text)
    except (ValueError, TypeError):
        return None
    if wins + losses <= 0:
        return None
    return wins, losses


def coaching_rating(*, experience_years: float, prior_record: Any) -> tuple[float, dict[str, float | str]]:
    """Return a conservative 0-100 coaching prior and transparent components."""
    years = max(0.0, float(experience_years or 0.0))
    # Experience is useful evidence, but it must never create a giant edge by itself.
    experience_bonus = min(5.0, math.sqrt(years) * 1.05)

    record_adjustment = 0.0
    record = _record_parts(prior_record)
    if record is not None:
        wins, losses = record
        games = wins + losses
        win_pct = wins / games
        # One recent season is informative but noisy and partly reflects roster quality.
        record_adjustment = max(-3.5, min(3.5, (win_pct - 0.5) * 10.0))

    rating = NEUTRAL_COACHING_RATING + experience_bonus + record_adjustment
    rating = round(max(64.0, min(80.0, rating)), 1)
    return rating, {
        "neutral_base": NEUTRAL_COACHING_RATING,
        "experience_bonus": round(experience_bonus, 2),
        "recent_record_adjustment": round(record_adjustment, 2),
    }


def load_coaching_priors(path: Path | str = DEFAULT_COACHING_PATH) -> dict[str, dict[str, Any]]:
    coaching_path = Path(path)
    if not coaching_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with coaching_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            team = str(row.get("team", "")).strip()
            if not team:
                continue
            years = float(row.get("experience_years") or 0.0)
            record = str(row.get("record_2025") or "--").strip()
            rating, components = coaching_rating(experience_years=years, prior_record=record)
            has_record = _record_parts(record) is not None
            out[team] = {
                "rating": rating,
                "head_coach": str(row.get("head_coach") or "").strip(),
                "experience_years": years,
                "record_2025": record,
                "source_url": str(row.get("source_url") or "").strip(),
                "status": "returning / 2025 record available" if has_record else "2026 coach prior / no 2025 team record",
                "components": components,
            }
    return out
