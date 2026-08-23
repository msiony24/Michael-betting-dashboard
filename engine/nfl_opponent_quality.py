"""NFL opponent quality intelligence.

Audit/context layer. Measures whether team performance is inflated or suppressed
by opponent quality. Does not directly replace team ratings.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any


@dataclass
class OpponentQualitySnapshot:
    team: str
    games: int
    opponent_strength: float
    adjusted_offense: float
    adjusted_defense: float
    adjustment_note: str


def quality_bucket(value: float) -> str:
    if value >= 75:
        return "Elite schedule"
    if value >= 65:
        return "Above average schedule"
    if value >= 50:
        return "Average schedule"
    return "Below average schedule"


def save_snapshot(snapshot: OpponentQualitySnapshot, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.exists():
        try:
            rows = json.loads(path.read_text())
        except Exception:
            rows = []
    rows.append(asdict(snapshot))
    path.write_text(json.dumps(rows, indent=2))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "teams": len(records),
        "average_opponent_strength": (
            sum(float(r.get("opponent_strength", 0)) for r in records) / len(records)
            if records else 0
        ),
    }
