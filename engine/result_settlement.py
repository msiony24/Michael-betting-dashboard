from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
import statistics
import unicodedata
from typing import Any, Iterable


ACTIONABLE_VERDICTS = {"strong bet", "worth betting", "bet"}
PASS_VERDICTS = {"lean", "pass", "complete pass", "fair price"}
EXCEPTION_STATUS_TOKENS = (
    "ret",
    "walkover",
    "walk-over",
    "w/o",
    "cancel",
    "abandon",
    "suspend",
    "postpon",
)


@dataclass(frozen=True)
class NameSignature:
    surname: str
    first_initial: str


@dataclass(frozen=True)
class SettlementGrade:
    status: str
    prediction_correct: bool | None
    value_call_correct: bool | None
    value_call_result: str
    reason: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ascii_tokens(value: object) -> list[str]:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens = re.findall(r"[A-Za-z]+", text.casefold())
    return [t for t in tokens if t not in {"jr", "sr", "ii", "iii", "iv"}]


def canonical_name(value: object) -> str:
    """Normalize punctuation/accents without discarding identifying initials."""
    return " ".join(_ascii_tokens(value))


def name_signature(value: object) -> NameSignature:
    """Build a conservative surname + first-initial identity signature.

    Handles full API names ("Juan Manuel Cerundolo") and provider forms such as
    "Cerundolo J.M." without collapsing Francisco and Juan Manuel Cerundolo.
    """
    tokens = _ascii_tokens(value)
    if not tokens:
        return NameSignature("", "")

    # Provider form: Surname F. or Surname J.M.
    if len(tokens) >= 2 and len(tokens[0]) > 1 and all(len(t) == 1 for t in tokens[1:]):
        return NameSignature(tokens[0], tokens[1])

    if len(tokens) >= 2:
        return NameSignature(tokens[-1], tokens[0][0])
    return NameSignature(tokens[0], "")


def names_compatible(requested: object, candidate: object) -> bool:
    """Return True only for a strong player/team identity match."""
    a = canonical_name(requested)
    b = canonical_name(candidate)
    if not a or not b:
        return False
    if a == b:
        return True

    sa = name_signature(requested)
    sb = name_signature(candidate)
    return bool(
        sa.surname
        and sa.surname == sb.surname
        and sa.first_initial
        and sa.first_initial == sb.first_initial
    )


def event_participants_match(
    participant_a: object,
    participant_b: object,
    event_first: object,
    event_second: object,
) -> bool:
    direct = names_compatible(participant_a, event_first) and names_compatible(
        participant_b, event_second
    )
    reverse = names_compatible(participant_a, event_second) and names_compatible(
        participant_b, event_first
    )
    return direct or reverse


def american_implied_probability(odds: int | float) -> float:
    value = float(odds)
    if value < 0:
        return abs(value) / (abs(value) + 100.0)
    if value > 0:
        return 100.0 / (value + 100.0)
    raise ValueError("American odds cannot be zero")


def decimal_to_american(decimal_odds: float) -> int:
    value = float(decimal_odds)
    if not math.isfinite(value) or value <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if value >= 2.0:
        return int(round((value - 1.0) * 100.0))
    return int(round(-100.0 / (value - 1.0)))


def no_vig_two_way_probability(
    selected_odds: int | float,
    other_odds: int | float,
) -> float:
    selected = american_implied_probability(selected_odds)
    other = american_implied_probability(other_odds)
    total = selected + other
    return selected / total if total > 0 else 0.5


def prediction_entry_odds(row: dict[str, Any]) -> int | None:
    prediction = str(row.get("prediction") or "").strip()
    participant_a = str(row.get("participant_a") or "").strip()
    participant_b = str(row.get("participant_b") or "").strip()
    raw = None
    if prediction and names_compatible(prediction, participant_a):
        raw = row.get("market_odds_a")
    elif prediction and names_compatible(prediction, participant_b):
        raw = row.get("market_odds_b")
    elif row.get("market_line") not in (None, ""):
        raw = row.get("market_line")
    try:
        value = int(round(float(raw)))
        return value if value != 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def prediction_entry_no_vig_probability(row: dict[str, Any]) -> float | None:
    prediction = str(row.get("prediction") or "").strip()
    participant_a = str(row.get("participant_a") or "").strip()
    participant_b = str(row.get("participant_b") or "").strip()
    try:
        odds_a = int(round(float(row.get("market_odds_a"))))
        odds_b = int(round(float(row.get("market_odds_b"))))
    except (TypeError, ValueError, OverflowError):
        return None
    if odds_a == 0 or odds_b == 0:
        return None
    if names_compatible(prediction, participant_a):
        return no_vig_two_way_probability(odds_a, odds_b)
    if names_compatible(prediction, participant_b):
        return no_vig_two_way_probability(odds_b, odds_a)
    return None


def is_provider_exception(event_status: object) -> bool:
    text = str(event_status or "").strip().casefold()
    return any(token in text for token in EXCEPTION_STATUS_TOKENS)


def is_finished_status(event_status: object) -> bool:
    text = str(event_status or "").strip().casefold()
    return text in {"finished", "completed", "final"}


def grade_moneyline_prediction(
    *,
    prediction: object,
    actual_winner: object,
    recommendation: object,
    provider_status: object = "Finished",
) -> SettlementGrade:
    """Grade winner accuracy and the actionable value call separately.

    PASS/Lean decisions stay neutral for ``value_call_correct``. A single match
    result cannot prove that avoiding a wager was mathematically correct; it only
    tells us whether an actual recommended wager won or lost.
    """
    if is_provider_exception(provider_status):
        return SettlementGrade(
            status="Pending",
            prediction_correct=None,
            value_call_correct=None,
            value_call_result="Needs manual review",
            reason=f"Provider status requires manual review: {provider_status}",
        )

    if not is_finished_status(provider_status):
        return SettlementGrade(
            status="Pending",
            prediction_correct=None,
            value_call_correct=None,
            value_call_result="Pending",
            reason=f"Event is not final: {provider_status}",
        )

    if not str(actual_winner or "").strip():
        return SettlementGrade(
            status="Pending",
            prediction_correct=None,
            value_call_correct=None,
            value_call_result="Needs manual review",
            reason="Final event did not include a trustworthy winner",
        )

    correct = names_compatible(prediction, actual_winner)
    stored_status = "Won" if correct else "Lost"
    verdict = str(recommendation or "").strip().casefold()

    if verdict in ACTIONABLE_VERDICTS:
        value_correct = correct
        value_result = "Bet won" if correct else "Bet lost"
    else:
        value_correct = None
        if verdict in PASS_VERDICTS:
            value_result = "No wager — prediction won" if correct else "No wager — prediction lost"
        else:
            value_result = "Prediction graded; value verdict not actionable"

    return SettlementGrade(
        status=stored_status,
        prediction_correct=correct,
        value_call_correct=value_correct,
        value_call_result=value_result,
        reason="Moneyline prediction graded from final winner",
    )


def grade_spread(
    *,
    selected_score: float,
    opponent_score: float,
    spread: float,
) -> str:
    adjusted = float(selected_score) + float(spread)
    opponent = float(opponent_score)
    if math.isclose(adjusted, opponent, abs_tol=1e-9):
        return "Push"
    return "Won" if adjusted > opponent else "Lost"


def grade_total(*, combined_score: float, total_line: float, side: str) -> str:
    score = float(combined_score)
    line = float(total_line)
    if math.isclose(score, line, abs_tol=1e-9):
        return "Push"
    normalized = str(side or "").strip().casefold()
    if normalized == "over":
        return "Won" if score > line else "Lost"
    if normalized == "under":
        return "Won" if score < line else "Lost"
    raise ValueError("Total side must be Over or Under")


def consensus_moneyline_close(
    snapshots: Iterable[dict[str, Any]],
    *,
    prediction: object,
    participant_a: object,
    participant_b: object,
) -> dict[str, Any] | None:
    """Create a consensus close from the latest two-way book snapshot.

    The caller should pass rows from one capture timestamp only. Each bookmaker
    contributes one no-vig probability when both participants are present.
    """
    books: dict[str, dict[str, int]] = {}
    for row in snapshots:
        book = str(row.get("bookmaker") or "").strip()
        participant = str(row.get("participant") or "").strip()
        if not book or not participant:
            continue
        try:
            odds = int(round(float(row.get("american_odds"))))
        except (TypeError, ValueError, OverflowError):
            continue
        if odds == 0:
            continue
        if names_compatible(participant, participant_a):
            key = "a"
        elif names_compatible(participant, participant_b):
            key = "b"
        else:
            continue
        books.setdefault(book, {})[key] = odds

    prediction_is_a = names_compatible(prediction, participant_a)
    prediction_is_b = names_compatible(prediction, participant_b)
    if not prediction_is_a and not prediction_is_b:
        return None

    no_vig_probs: list[float] = []
    selected_prices: list[int] = []
    valid_books: list[str] = []
    for book, pair in books.items():
        if "a" not in pair or "b" not in pair:
            continue
        if prediction_is_a:
            selected, other = pair["a"], pair["b"]
        else:
            selected, other = pair["b"], pair["a"]
        no_vig_probs.append(no_vig_two_way_probability(selected, other))
        selected_prices.append(selected)
        valid_books.append(book)

    if not no_vig_probs:
        return None

    return {
        "closing_no_vig_probability": float(statistics.median(no_vig_probs)),
        "closing_odds_prediction": int(round(statistics.median(selected_prices))),
        "book_count": len(valid_books),
        "books": sorted(valid_books),
        "closing_book": "consensus_median",
    }


def clv_metrics(
    *,
    row: dict[str, Any],
    closing_no_vig_probability: float | None,
) -> dict[str, float | None]:
    entry_probability = prediction_entry_no_vig_probability(row)
    try:
        model_probability = float(row.get("predicted_probability"))
    except (TypeError, ValueError):
        model_probability = math.nan

    close = (
        float(closing_no_vig_probability)
        if closing_no_vig_probability is not None
        else math.nan
    )
    clv = (
        close - entry_probability
        if entry_probability is not None and math.isfinite(close)
        else None
    )
    model_edge_close = (
        model_probability - close
        if math.isfinite(model_probability) and math.isfinite(close)
        else None
    )
    return {
        "entry_no_vig_probability": entry_probability,
        "closing_no_vig_probability": close if math.isfinite(close) else None,
        "clv_probability": clv,
        "model_edge_at_close": model_edge_close,
    }
