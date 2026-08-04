from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Callable, Iterable


CORRECT_STATUSES = {"Correct", "Won"}
INCORRECT_STATUSES = {"Incorrect", "Lost"}
PENDING_STATUSES = {"", "Pending", None}
BET_VERDICTS = {"Strong Bet", "Worth Betting", "Bet"}
PASS_VERDICTS = {"Lean", "Pass", "Complete Pass", "Fair Price"}


def normalize_prediction_result(row: dict) -> str:
    status = row.get("status", "Pending")
    if status in CORRECT_STATUSES:
        return "Correct"
    if status in INCORRECT_STATUSES:
        return "Incorrect"
    if status in {"Push", "Void"}:
        return str(status)
    return "Pending"


def normalize_verdict(verdict: object) -> str:
    text = str(verdict or "").strip()
    if text in BET_VERDICTS:
        return "BET"
    if text in PASS_VERDICTS:
        return "PASS"
    lowered = text.casefold()
    if "bet" in lowered and "pass" not in lowered:
        return "BET"
    return "PASS"


def row_day(row: dict) -> str:
    raw = row.get("event_date") or row.get("created_at") or ""
    if isinstance(raw, (date, datetime)):
        return raw.isoformat()[:10]
    text = str(raw).strip()
    return text[:10] if len(text) >= 10 else text


def summarize_rows(
    rows: Iterable[dict],
    verdict_getter: Callable[[dict], str] | None = None,
) -> dict:
    rows = list(rows)
    verdict_getter = verdict_getter or (lambda row: str(row.get("recommendation", "")))

    completed = [
        row for row in rows
        if normalize_prediction_result(row) in {"Correct", "Incorrect"}
    ]
    correct = sum(normalize_prediction_result(row) == "Correct" for row in completed)
    incorrect = len(completed) - correct
    pending = sum(normalize_prediction_result(row) == "Pending" for row in rows)

    bet_rows = [row for row in rows if normalize_verdict(verdict_getter(row)) == "BET"]
    pass_rows = [row for row in rows if normalize_verdict(verdict_getter(row)) == "PASS"]

    def record(group: list[dict]) -> dict:
        graded = [
            row for row in group
            if normalize_prediction_result(row) in {"Correct", "Incorrect"}
        ]
        wins = sum(normalize_prediction_result(row) == "Correct" for row in graded)
        losses = len(graded) - wins
        return {
            "correct": wins,
            "incorrect": losses,
            "graded": len(graded),
            "accuracy": wins / len(graded) if graded else None,
            "pending": sum(normalize_prediction_result(row) == "Pending" for row in group),
            "total": len(group),
        }

    return {
        "total": len(rows),
        "graded": len(completed),
        "correct": correct,
        "incorrect": incorrect,
        "pending": pending,
        "accuracy": correct / len(completed) if completed else None,
        "bet": record(bet_rows),
        "pass": record(pass_rows),
    }


def group_rows_by_day(rows: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row_day(row)].append(row)
    return dict(sorted(grouped.items(), reverse=True))
