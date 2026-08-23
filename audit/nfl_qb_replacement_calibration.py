
"""NFL QB replacement calibration tracker (audit only).

This module records QB replacement situations after the season starts.
It does not alter predictions. It exists to measure whether the QB
replacement model is calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any


@dataclass
class QBReplacementAuditRow:
    season: int
    week: int
    team: str
    starter_qb: str
    replacement_qb: str
    starter_rating: float
    replacement_rating: float
    raw_drop: float
    effective_drop: float
    qb_adjustment: float
    experience_credit: float
    support_factor: float
    pressure_factor: float
    market_open_spread: float | None = None
    market_close_spread: float | None = None
    result_margin: float | None = None


def save_qb_audit_row(row: QBReplacementAuditRow, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if path.exists():
        try:
            rows = json.loads(path.read_text())
        except Exception:
            rows = []

    rows.append(asdict(row))
    path.write_text(json.dumps(rows, indent=2))


def calibration_bucket(drop: float) -> str:
    drop = abs(float(drop))
    if drop <= 5:
        return "0-5"
    if drop <= 10:
        return "6-10"
    if drop <= 15:
        return "11-15"
    return "16+"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {}
    for row in rows:
        bucket = calibration_bucket(row.get("effective_drop", 0))
        buckets.setdefault(bucket, {"games": 0, "avg_adjustment": 0.0})
        buckets[bucket]["games"] += 1
        buckets[bucket]["avg_adjustment"] += float(row.get("qb_adjustment", 0))

    for value in buckets.values():
        if value["games"]:
            value["avg_adjustment"] /= value["games"]

    return buckets
