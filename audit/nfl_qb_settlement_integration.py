"""NFL QB calibration settlement bridge (audit only).

Connects settled game results to previously logged QB replacement events.
It does not change predictions or ratings.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class QBCalibrationResult:
    event_id: str
    team: str
    eligible: bool
    predicted_qb_adjustment: float
    actual_margin: float | None
    notes: str


def attach_settlement_result(
    qb_event: dict[str, Any],
    *,
    event_id: str,
    actual_margin: float | None,
    game_completed: bool,
) -> QBCalibrationResult:
    """Attach a completed game result to a QB replacement audit event.

    Only events already marked eligible are graded. A backup appearing because
    of an in-game injury is never converted into a pregame calibration sample.
    """
    eligible = bool(qb_event.get("eligible_for_calibration")) and game_completed

    return QBCalibrationResult(
        event_id=str(event_id),
        team=str(qb_event.get("team", "")),
        eligible=eligible,
        predicted_qb_adjustment=float(qb_event.get("qb_adjustment", 0.0)),
        actual_margin=actual_margin if eligible else None,
        notes=(
            "Eligible QB replacement calibration sample."
            if eligible
            else "Ignored for QB calibration."
        ),
    )


def result_to_dict(result: QBCalibrationResult) -> dict[str, Any]:
    return asdict(result)
