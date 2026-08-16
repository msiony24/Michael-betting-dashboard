from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import math
import os
from typing import Any

from analysis_store import (
    insert_table_records,
    is_configured,
    list_analyses,
    list_table_records,
    update_analysis,
)
from engine.result_settlement import (
    clv_metrics,
    consensus_moneyline_close,
    grade_moneyline_prediction,
    is_finished_status,
    is_provider_exception,
    names_compatible,
    utc_now_iso,
)
from engine.settlement_providers import (
    APITennisSettlementClient,
    OddsApiClient,
    ProviderError,
)


ODDS_SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "WNBA": "basketball_wnba",
    "College Football": "americanfootball_ncaaf",
    "MLB": "baseball_mlb",
    "NHL": "icehockey_nhl",
}


def parse_event_date(row: dict[str, Any]) -> date | None:
    raw = row.get("event_date") or row.get("created_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)[:10]).date()
    except ValueError:
        return None


def parse_iso_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def nested_link(row: dict[str, Any]) -> dict[str, Any]:
    snapshot = row.get("input_snapshot") or {}
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except json.JSONDecodeError:
            snapshot = {}
    if not isinstance(snapshot, dict):
        return {}
    link = snapshot.get("settlement_link") or snapshot.get("external_event") or {}
    return link if isinstance(link, dict) else {}


def audit(
    analysis_id: object,
    *,
    action: str,
    source: str,
    reason: str = "",
    payload: dict[str, Any] | None = None,
    before_status: str | None = None,
    after_status: str | None = None,
) -> None:
    insert_table_records(
        "analysis_settlement_audit",
        {
            "analysis_id": str(analysis_id),
            "changed_at": utc_now_iso(),
            "action": action,
            "source": source,
            "reason": reason,
            "before_status": before_status,
            "after_status": after_status,
            "payload": payload or {},
        },
    )


def resolve_odds_link(
    row: dict[str, Any],
    odds_client: OddsApiClient | None,
) -> dict[str, Any] | None:
    if odds_client is None:
        return None

    participant_a = str(row.get("participant_a") or "").strip()
    participant_b = str(row.get("participant_b") or "").strip()
    if not participant_a or not participant_b:
        return None

    # Prefer an event ID already captured at analysis time.
    link = nested_link(row)
    provider = str(link.get("provider") or link.get("provider_name") or "").casefold()
    event_id = str(link.get("event_id") or "").strip()
    sport_key = str(link.get("sport_key") or "").strip()
    if provider in {"the_odds_api", "odds_api"} and event_id and sport_key:
        return {
            "provider": "the_odds_api",
            "event_id": event_id,
            "sport_key": sport_key,
            "commence_time": link.get("commence_time"),
            "link_method": "analysis_time_event_id",
        }

    if str(row.get("market_provider") or "").casefold() == "the_odds_api":
        existing_id = str(row.get("market_provider_event_id") or "").strip()
        existing_key = str(row.get("market_provider_sport_key") or "").strip()
        if existing_id and existing_key:
            return {
                "provider": "the_odds_api",
                "event_id": existing_id,
                "sport_key": existing_key,
                "commence_time": row.get("scheduled_start"),
                "link_method": "stored_event_id",
            }

    sport = str(row.get("sport") or "").strip()
    try:
        if sport == "Tennis":
            found = odds_client.find_tennis_event(
                participant_a=participant_a,
                participant_b=participant_b,
            )
            if not found:
                return None
            sport_key, event = found
        else:
            sport_key = ODDS_SPORT_KEYS.get(sport)
            if not sport_key:
                return None
            event = odds_client.find_event(
                sport_key=sport_key,
                participant_a=participant_a,
                participant_b=participant_b,
            )
            if not event:
                return None
    except ProviderError:
        return None

    return {
        "provider": "the_odds_api",
        "event_id": str(event.get("id") or ""),
        "sport_key": sport_key,
        "commence_time": event.get("commence_time"),
        "link_method": "strict_participant_match",
    }


def resolve_tennis_result_link(
    row: dict[str, Any],
    tennis_client: APITennisSettlementClient | None,
) -> dict[str, Any] | None:
    if tennis_client is None or str(row.get("sport")) != "Tennis":
        return None

    event_date = parse_event_date(row)
    if event_date is None:
        return None

    existing_id = str(row.get("result_provider_event_id") or "").strip()
    if str(row.get("result_provider") or "").casefold() == "api_tennis" and existing_id:
        fixture = tennis_client.fixture_by_event_id(existing_id, event_date=event_date)
        if fixture:
            return {"provider": "api_tennis", "event_id": existing_id, "fixture": fixture}

    try:
        fixture = tennis_client.find_fixture(
            participant_a=str(row.get("participant_a") or ""),
            participant_b=str(row.get("participant_b") or ""),
            event_date=event_date,
            tournament=str((row.get("input_snapshot") or {}).get("tournament") or row.get("event_name") or "")
            if isinstance(row.get("input_snapshot") or {}, dict)
            else str(row.get("event_name") or ""),
        )
    except ProviderError:
        return None

    if not fixture:
        return None
    event_id = str(fixture.get("event_key") or "").strip()
    return {
        "provider": "api_tennis",
        "event_id": event_id,
        "fixture": fixture,
    } if event_id else None


def persist_links(
    row: dict[str, Any],
    *,
    market_link: dict[str, Any] | None,
    result_link: dict[str, Any] | None,
) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    if market_link:
        changes.update({
            "market_provider": "the_odds_api",
            "market_provider_event_id": market_link.get("event_id"),
            "market_provider_sport_key": market_link.get("sport_key"),
            "scheduled_start": market_link.get("commence_time") or row.get("scheduled_start"),
            "provider_link_status": "linked",
        })
    if result_link:
        changes.update({
            "result_provider": result_link.get("provider"),
            "result_provider_event_id": result_link.get("event_id"),
            "provider_link_status": "linked",
        })
    if changes:
        updated = update_analysis(str(row["id"]), changes)
        if updated:
            row = updated
    return row


def snapshot_odds(
    row: dict[str, Any],
    market_link: dict[str, Any] | None,
    odds_client: OddsApiClient | None,
) -> int:
    if odds_client is None or not market_link:
        return 0
    if str(row.get("status") or "Pending") != "Pending":
        return 0

    commence = parse_iso_datetime(market_link.get("commence_time") or row.get("scheduled_start"))
    now = datetime.now(timezone.utc)
    if commence is not None and now >= commence:
        return 0
    # Do not burn quota on distant events. Hourly snapshots inside 12 hours are
    # enough to establish a defensible pre-start closing observation.
    if commence is not None and (commence - now).total_seconds() > 12 * 3600:
        return 0

    try:
        payload = odds_client.event_odds(
            str(market_link["sport_key"]),
            str(market_link["event_id"]),
            markets="h2h",
        )
    except ProviderError:
        return 0

    captured_at = utc_now_iso()
    records = []
    for bookmaker in payload.get("bookmakers", []) or []:
        book_key = str(bookmaker.get("key") or bookmaker.get("title") or "").strip()
        book_title = str(bookmaker.get("title") or bookmaker.get("key") or "").strip()
        for market in bookmaker.get("markets", []) or []:
            if str(market.get("key")) != "h2h":
                continue
            market_updated = market.get("last_update") or bookmaker.get("last_update")
            for outcome in market.get("outcomes", []) or []:
                participant = str(outcome.get("name") or "").strip()
                try:
                    price = int(round(float(outcome.get("price"))))
                except (TypeError, ValueError, OverflowError):
                    continue
                if not participant or price == 0:
                    continue
                records.append({
                    "analysis_id": str(row["id"]),
                    "provider": "the_odds_api",
                    "provider_event_id": str(market_link["event_id"]),
                    "provider_sport_key": str(market_link["sport_key"]),
                    "captured_at": captured_at,
                    "market_updated_at": market_updated,
                    "market_type": "h2h",
                    "bookmaker": book_title or book_key,
                    "bookmaker_key": book_key,
                    "participant": participant,
                    "american_odds": price,
                    "raw_payload": {"outcome": outcome},
                })
    if records:
        insert_table_records("analysis_odds_snapshots", records)
    return len(records)


def latest_close(row: dict[str, Any]) -> dict[str, Any] | None:
    records = list_table_records(
        "analysis_odds_snapshots",
        params={
            "select": "*",
            "analysis_id": f"eq.{row['id']}",
            "order": "captured_at.desc",
            "limit": 1000,
        },
    )
    if not records:
        return None

    start = parse_iso_datetime(row.get("scheduled_start"))
    eligible = []
    for item in records:
        captured = parse_iso_datetime(item.get("captured_at"))
        if captured is None:
            continue
        if start is None or captured <= start:
            eligible.append((captured, item))
    if not eligible:
        return None

    latest_time = max(item[0] for item in eligible)
    batch = [item[1] for item in eligible if item[0] == latest_time]
    result = consensus_moneyline_close(
        batch,
        prediction=row.get("prediction"),
        participant_a=row.get("participant_a"),
        participant_b=row.get("participant_b"),
    )
    if result:
        result["closing_snapshot_at"] = latest_time.isoformat()
    return result


def settle_tennis(
    row: dict[str, Any],
    result_link: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    fixture = result_link.get("fixture") or {}
    event_status = str(fixture.get("event_status") or "").strip()
    winner_label = str(fixture.get("event_winner") or "").strip().casefold()
    if winner_label == "first player":
        actual_winner = str(fixture.get("event_first_player") or "").strip()
    elif winner_label == "second player":
        actual_winner = str(fixture.get("event_second_player") or "").strip()
    else:
        actual_winner = None

    score_text = str(fixture.get("event_final_result") or "").strip() or None
    raw = {
        "event_status": event_status,
        "actual_winner": actual_winner,
        "actual_score": score_text,
        "provider_match_data": fixture,
    }
    exception_context = f"{event_status} {score_text or ''}".strip()
    if not is_finished_status(event_status) and not is_provider_exception(exception_context):
        return None, raw

    grading_status = "Retired / exceptional settlement" if is_provider_exception(exception_context) else event_status
    grade = grade_moneyline_prediction(
        prediction=row.get("prediction"),
        actual_winner=actual_winner,
        recommendation=row.get("recommendation"),
        provider_status=grading_status,
    )
    return {
        **raw,
        "status": grade.status,
        "automatic_status": grade.status,
        "prediction_correct": grade.prediction_correct,
        "value_call_correct": grade.value_call_correct,
        "value_call_result": grade.value_call_result,
        "settlement_source": "api_tennis",
        "settlement_version": "result-settlement-v1",
        "settled_at": utc_now_iso() if grade.status != "Pending" else None,
        "provider_link_status": "needs_review" if grade.status == "Pending" else "settled",
    }, raw


def settle_odds_api_score(
    row: dict[str, Any],
    market_link: dict[str, Any],
    odds_client: OddsApiClient,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        scores = odds_client.scores(str(market_link["sport_key"]), days_from=3)
    except ProviderError:
        return None, {}
    event = next(
        (item for item in scores if str(item.get("id") or "") == str(market_link["event_id"])),
        None,
    )
    if not event or not bool(event.get("completed")):
        return None, event or {}

    score_rows = event.get("scores") or []
    parsed = []
    for item in score_rows:
        try:
            parsed.append((str(item.get("name") or "").strip(), float(item.get("score"))))
        except (TypeError, ValueError):
            continue
    actual_winner = None
    if len(parsed) >= 2:
        top = max(score for _, score in parsed)
        winners = [name for name, score in parsed if math.isclose(score, top)]
        if len(winners) == 1:
            actual_winner = winners[0]

    score_text = " - ".join(f"{name} {score:g}" for name, score in parsed) or None
    provider_status = "Finished" if actual_winner else "Tie / needs review"
    grade = grade_moneyline_prediction(
        prediction=row.get("prediction"),
        actual_winner=actual_winner,
        recommendation=row.get("recommendation"),
        provider_status=provider_status,
    )
    raw = {
        "event_status": provider_status,
        "actual_winner": actual_winner,
        "actual_score": score_text,
        "provider_match_data": event,
    }
    return {
        **raw,
        "status": grade.status,
        "automatic_status": grade.status,
        "prediction_correct": grade.prediction_correct,
        "value_call_correct": grade.value_call_correct,
        "value_call_result": grade.value_call_result,
        "settlement_source": "the_odds_api",
        "settlement_version": "result-settlement-v1",
        "settled_at": utc_now_iso() if grade.status != "Pending" else None,
        "provider_link_status": "needs_review" if grade.status == "Pending" else "settled",
    }, raw


def apply_close_fields(row: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    close = latest_close(row)
    if not close:
        return changes
    metrics = clv_metrics(
        row=row,
        closing_no_vig_probability=close.get("closing_no_vig_probability"),
    )
    return {
        **changes,
        "closing_odds_prediction": close.get("closing_odds_prediction"),
        "closing_book": close.get("closing_book"),
        "closing_snapshot_at": close.get("closing_snapshot_at"),
        "closing_no_vig_probability": metrics.get("closing_no_vig_probability"),
        "entry_no_vig_probability": metrics.get("entry_no_vig_probability"),
        "clv_probability": metrics.get("clv_probability"),
        "model_edge_at_close": metrics.get("model_edge_at_close"),
    }


def process_row(
    row: dict[str, Any],
    *,
    odds_client: OddsApiClient | None,
    tennis_client: APITennisSettlementClient | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    result = {
        "analysis_id": str(row.get("id")),
        "event": row.get("event_name"),
        "action": "none",
    }

    market_link = resolve_odds_link(row, odds_client)
    result_link = resolve_tennis_result_link(row, tennis_client)

    if not dry_run:
        row = persist_links(row, market_link=market_link, result_link=result_link)

    snapshots = 0
    if not dry_run:
        snapshots = snapshot_odds(row, market_link, odds_client)
    result["odds_snapshot_rows"] = snapshots

    settlement_changes: dict[str, Any] | None = None
    raw: dict[str, Any] = {}
    if str(row.get("sport")) == "Tennis" and result_link:
        settlement_changes, raw = settle_tennis(row, result_link)
    elif market_link and odds_client is not None:
        settlement_changes, raw = settle_odds_api_score(row, market_link, odds_client)

    if raw and not settlement_changes and not dry_run:
        # Keep final/live provider facts attached even before grading is possible.
        update_analysis(str(row["id"]), raw)

    if not settlement_changes:
        result["action"] = "linked/snapshotted" if market_link or result_link else "unlinked"
        return result

    settlement_changes = apply_close_fields(row, settlement_changes)

    if bool(row.get("manual_override")):
        safe_raw = {
            key: value
            for key, value in settlement_changes.items()
            if key in {
                "event_status", "actual_winner", "actual_score", "provider_match_data",
                "closing_odds_prediction", "closing_book", "closing_snapshot_at",
                "closing_no_vig_probability", "entry_no_vig_probability",
                "clv_probability", "model_edge_at_close",
            }
        }
        if not dry_run and safe_raw:
            update_analysis(str(row["id"]), safe_raw)
            audit(
                row["id"],
                action="manual_override_preserved",
                source=str(settlement_changes.get("settlement_source") or "provider"),
                reason=str(row.get("manual_override_reason") or "Manual override locked"),
                payload=raw,
                before_status=str(row.get("status")),
                after_status=str(row.get("status")),
            )
        result["action"] = "manual_override_preserved"
        return result

    if dry_run:
        result["action"] = f"would_settle:{settlement_changes.get('status')}"
        return result

    before = str(row.get("status") or "Pending")
    updated = update_analysis(str(row["id"]), settlement_changes)
    after = str((updated or settlement_changes).get("status") or before)
    audit(
        row["id"],
        action="automatic_settlement" if after != "Pending" else "provider_exception",
        source=str(settlement_changes.get("settlement_source") or "provider"),
        reason=str(settlement_changes.get("value_call_result") or ""),
        payload=raw,
        before_status=before,
        after_status=after,
    )
    result["action"] = f"settled:{after}" if after != "Pending" else "needs_review"
    return result


def build_clients() -> tuple[OddsApiClient | None, APITennisSettlementClient | None]:
    try:
        odds_client = OddsApiClient()
    except ProviderError:
        odds_client = None
    try:
        tennis_client = APITennisSettlementClient()
    except ProviderError:
        tennis_client = None
    return odds_client, tennis_client


def run(*, limit: int = 500, dry_run: bool = False) -> list[dict[str, Any]]:
    if not is_configured():
        raise RuntimeError("Supabase settlement credentials are not configured")
    odds_client, tennis_client = build_clients()
    if odds_client is None and tennis_client is None:
        raise RuntimeError("No settlement data provider is configured")

    rows = list_analyses(limit, status="Pending")
    results = []
    for row in rows:
        try:
            results.append(
                process_row(
                    row,
                    odds_client=odds_client,
                    tennis_client=tennis_client,
                    dry_run=dry_run,
                )
            )
        except Exception as exc:
            results.append({
                "analysis_id": str(row.get("id")),
                "event": row.get("event_name"),
                "action": "error",
                "error": str(exc),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Settle pending Macabets predictions")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = run(limit=args.limit, dry_run=args.dry_run)
    print(json.dumps(results, indent=2, default=str))
    errors = [row for row in results if row.get("action") == "error"]
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
