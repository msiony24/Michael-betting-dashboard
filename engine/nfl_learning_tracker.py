"""NFL learning foundation (audit-first).

Stores settled prediction outcomes so Macabets can measure calibration.
Does not change model weights automatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any


@dataclass
class NFLPredictionOutcome:
    game_id: str
    season: int
    week: int
    prediction: str
    predicted_probability: float
    fair_spread: float
    market_spread: float | None
    closing_spread: float | None
    result_margin: float
    correct_winner: bool
    closing_line_value: float | None


def record_outcome(row: NFLPredictionOutcome, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if path.exists():
        try:
            rows = json.loads(path.read_text())
        except Exception:
            rows = []

    rows.append(asdict(row))
    path.write_text(json.dumps(rows, indent=2))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"games": 0}

    return {
        "games": len(rows),
        "winner_accuracy": sum(bool(r.get("correct_winner")) for r in rows) / len(rows),
        "average_clv": (
            sum(float(r.get("closing_line_value") or 0) for r in rows) / len(rows)
        ),
    }
