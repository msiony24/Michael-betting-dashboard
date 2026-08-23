"""NFL closing line value tracker.

Audit layer only. Does not change predictions.
Stores opening vs closing market movement.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any


@dataclass
class NFLCLVRecord:
    game_id: str
    team: str
    market: str
    opening_line: float
    closing_line: float
    clv_points: float
    prediction: str | None = None
    result: str | None = None


def save_clv(record: NFLCLVRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if path.exists():
        try:
            rows = json.loads(path.read_text())
        except Exception:
            rows = []

    rows.append(asdict(record))
    path.write_text(json.dumps(rows, indent=2))


def summarize_clv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"games": 0}

    values = [float(r.get("clv_points", 0)) for r in rows]
    return {
        "games": len(values),
        "average_clv_points": sum(values) / len(values),
        "positive_clv_rate": sum(v > 0 for v in values) / len(values),
    }
