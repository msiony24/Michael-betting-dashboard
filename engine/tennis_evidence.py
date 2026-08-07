"""Verified tennis evidence packets for the Macabets Challenge layer.

This module turns Macabets' local ATP match database into a compact, matchup-specific
record that the conversational layer can reason from directly. It does not change
core tennis predictions by itself.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from .tennis import canonical_player_key, resolve_player_name


def _date(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp(date.today())
    return pd.Timestamp(ts).normalize()


def _clean_number(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _match_record(row: pd.Series, player_key: str) -> dict[str, Any]:
    won = canonical_player_key(row.get("winner_name", "")) == player_key
    opponent = row.get("loser_name") if won else row.get("winner_name")
    opponent_rank = row.get("loser_rank") if won else row.get("winner_rank")
    player_rank = row.get("winner_rank") if won else row.get("loser_rank")
    date_value = pd.to_datetime(row.get("tourney_date"), errors="coerce")
    return {
        "date": date_value.strftime("%Y-%m-%d") if not pd.isna(date_value) else "Unknown",
        "result": "W" if won else "L",
        "opponent": str(opponent or "Unknown"),
        "opponent_rank": _clean_number(opponent_rank),
        "player_rank": _clean_number(player_rank),
        "tournament": str(row.get("tourney_name") or "Unknown"),
        "surface": str(row.get("surface") or "Unknown"),
        "round": str(row.get("round") or ""),
        "score": str(row.get("score") or "").strip(),
    }


def _record_summary(records: list[dict[str, Any]]) -> str:
    wins = sum(1 for row in records if row["result"] == "W")
    return f"{wins}-{len(records) - wins}"


def _notable_wins(records: list[dict[str, Any]], cutoff: int) -> list[dict[str, Any]]:
    return [
        row for row in records
        if row["result"] == "W"
        and row.get("opponent_rank") is not None
        and int(row["opponent_rank"]) <= cutoff
    ]


def build_player_evidence(
    matches: pd.DataFrame,
    player: str,
    event_date: Any,
    surface: str,
    tournament: str = "",
    lookback: int = 20,
) -> dict[str, Any]:
    """Build verified pre-match evidence for one player from local ATP history."""
    event_ts = _date(event_date)
    resolved, resolution = resolve_player_name(matches, player)
    target = resolved or player
    key = canonical_player_key(target)

    mask = (
        matches["winner_name"].map(canonical_player_key).eq(key)
        | matches["loser_name"].map(canonical_player_key).eq(key)
    ) & (matches["tourney_date"] < event_ts)

    player_matches = matches.loc[mask].sort_values("tourney_date", ascending=False)
    recent_rows = player_matches.head(max(int(lookback), 1))
    recent = [_match_record(row, key) for _, row in recent_rows.iterrows()]
    last_10 = recent[:10]
    last_5 = recent[:5]

    surface_key = str(surface or "").strip().casefold()
    surface_records = [
        row for row in recent
        if str(row.get("surface") or "").casefold() == surface_key
    ]

    tournament_key = str(tournament or "").strip().casefold()
    current_tournament = [
        row for row in recent
        if tournament_key and str(row.get("tournament") or "").strip().casefold() == tournament_key
    ]

    return {
        "player": player,
        "database_name": resolved,
        "name_resolution": resolution,
        "recent_record_5": _record_summary(last_5),
        "recent_record_10": _record_summary(last_10),
        "recent_record_20": _record_summary(recent),
        "recent_surface_record": _record_summary(surface_records),
        "surface": str(surface or "Unknown"),
        "current_tournament_results": current_tournament,
        "top_10_wins": _notable_wins(recent, 10),
        "top_20_wins": _notable_wins(recent, 20),
        "top_50_wins": _notable_wins(recent, 50),
        "recent_matches": recent,
        "matches_available_before_event": int(len(player_matches)),
    }


def build_tennis_evidence_packet(
    matches: pd.DataFrame,
    player_a: str,
    player_b: str,
    event_date: Any,
    surface: str,
    tournament: str = "",
    lookback: int = 20,
) -> dict[str, Any]:
    """Return the verified evidence packet supplied to Challenge Macabets."""
    event_ts = _date(event_date)
    available = matches[matches["tourney_date"] < event_ts]
    latest = available["tourney_date"].max() if not available.empty else pd.NaT
    latest_text = latest.strftime("%Y-%m-%d") if not pd.isna(latest) else "Unavailable"
    lag_days = None if pd.isna(latest) else max(int((event_ts - latest.normalize()).days), 0)

    return {
        "source": "Macabets local ATP match database",
        "status": "verified_local_history",
        "event_date": event_ts.strftime("%Y-%m-%d"),
        "latest_match_date_in_database": latest_text,
        "database_lag_days_at_event": lag_days,
        "freshness_note": (
            "Local evidence is authoritative for records present in this packet. "
            "A user claim dated after latest_match_date_in_database may simply be newer than the local feed."
        ),
        "player_a": build_player_evidence(
            matches, player_a, event_ts, surface, tournament=tournament, lookback=lookback
        ),
        "player_b": build_player_evidence(
            matches, player_b, event_ts, surface, tournament=tournament, lookback=lookback
        ),
    }
