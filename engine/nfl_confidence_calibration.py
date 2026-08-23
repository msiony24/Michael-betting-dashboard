"""NFL confidence calibration tracker (audit only).

Tracks whether confidence labels are predictive.
Does not alter model outputs.
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Any


BANDS = ("Low", "Moderate", "High", "Very High")


def record_confidence_result(
    confidence_band: str,
    winner_correct: bool,
    closing_line_value: float | None,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.exists():
        try:
            rows = json.loads(path.read_text())
        except Exception:
            rows = []
    rows.append({
        "confidence_band": confidence_band,
        "winner_correct": bool(winner_correct),
        "closing_line_value": closing_line_value,
    })
    path.write_text(json.dumps(rows, indent=2))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for band in BANDS:
        sample = [r for r in rows if r.get("confidence_band") == band]
        result[band] = {
            "games": len(sample),
            "accuracy": (
                sum(bool(r.get("winner_correct")) for r in sample) / len(sample)
                if sample else None
            ),
            "average_clv": (
                sum(float(r.get("closing_line_value") or 0) for r in sample) / len(sample)
                if sample else None
            ),
        }
    return result
