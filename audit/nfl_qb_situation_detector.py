"""NFL QB situation detector (audit/support layer).

Identifies pregame QB replacement situations. Does not alter predictions.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any


@dataclass
class QBReplacementSituation:
    game_id: str
    team: str
    qb1: str
    replacement_qb: str | None
    qb1_status: str
    detected_before_kickoff: bool
    replacement_expected: bool
    reason: str
    created_at: str


def is_unavailable(status: Any) -> bool:
    value = str(status or "").lower()
    return any(x in value for x in ["out", "ir", "pup", "suspended", "inactive"])


def detect_qb_situation(
    *,
    game_id: str,
    team: str,
    depth_chart: list[dict],
    availability: dict[str, str],
) -> QBReplacementSituation | None:
    qbs = [p for p in depth_chart if str(p.get("position","")).upper() == "QB"]
    if len(qbs) < 2:
        return None

    qbs = sorted(qbs, key=lambda x: x.get("depth", 99))
    qb1 = qbs[0]
    qb2 = qbs[1]

    qb1_name = str(qb1.get("name",""))
    status = availability.get(qb1_name, "")

    if not is_unavailable(status):
        return None

    return QBReplacementSituation(
        game_id=game_id,
        team=team,
        qb1=qb1_name,
        replacement_qb=str(qb2.get("name","")) or None,
        qb1_status=str(status),
        detected_before_kickoff=True,
        replacement_expected=True,
        reason="QB1 unavailable before kickoff; next active QB identified",
        created_at=datetime.utcnow().isoformat(),
    )
