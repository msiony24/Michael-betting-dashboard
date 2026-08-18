from __future__ import annotations

from typing import Any

import pandas as pd

from .tennis_identity import canonical_player_key, resolve_player_name


def _empty_summary(
    player_a: str,
    player_b: str,
    *,
    resolved_a: str | None = None,
    resolved_b: str | None = None,
    resolution_a: dict[str, Any] | None = None,
    resolution_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "meetings": 0,
        "wins_a": 0,
        "wins_b": 0,
        "surface_meetings": 0,
        "surface_wins_a": 0,
        "surface_wins_b": 0,
        "last_meeting": None,
        "requested_player_a": str(player_a),
        "requested_player_b": str(player_b),
        "resolved_player_a": resolved_a,
        "resolved_player_b": resolved_b,
        "resolution_a": resolution_a or {
            "requested": str(player_a),
            "resolved": resolved_a,
            "method": "not_attempted",
        },
        "resolution_b": resolution_b or {
            "requested": str(player_b),
            "resolved": resolved_b,
            "method": "not_attempted",
        },
    }


def build_head_to_head_summary(
    matches: pd.DataFrame,
    player_a: str,
    player_b: str,
    current_surface: str,
) -> dict[str, Any]:
    """Build opponent-specific H2H context using resolved historical identities.

    Live/API feeds commonly use full names (for example ``Hubert Hurkacz``),
    while the local tennis-data history uses provider display names such as
    ``Hurkacz H.``. Resolve both requested players against the historical match
    table before filtering, while preserving the requested full names for UI
    display.
    """
    if matches is None or matches.empty:
        return _empty_summary(player_a, player_b)

    def first_column(options: list[str]) -> str | None:
        return next((name for name in options if name in matches.columns), None)

    winner_col = first_column(["winner_name", "winner", "Winner", "w_name"])
    loser_col = first_column(["loser_name", "loser", "Loser", "l_name"])
    surface_col = first_column(["surface", "Surface"])
    date_col = first_column(["tourney_date", "match_date", "date", "Date"])
    event_col = first_column(["tourney_name", "tournament", "event", "Tournament"])
    score_col = first_column(["score", "Score"])
    round_col = first_column(["round", "Round"])

    if not winner_col or not loser_col:
        return _empty_summary(player_a, player_b)

    # The canonical resolver expects winner_name/loser_name. Create a tiny view
    # with those names so this helper remains compatible with alternate providers.
    resolution_frame = pd.DataFrame(
        {
            "winner_name": matches[winner_col],
            "loser_name": matches[loser_col],
        }
    )
    resolved_a, resolution_a = resolve_player_name(resolution_frame, str(player_a))
    resolved_b, resolution_b = resolve_player_name(resolution_frame, str(player_b))

    lookup_a = str(resolved_a or player_a).strip()
    lookup_b = str(resolved_b or player_b).strip()
    key_a = canonical_player_key(lookup_a)
    key_b = canonical_player_key(lookup_b)

    winner_keys = matches[winner_col].map(canonical_player_key)
    loser_keys = matches[loser_col].map(canonical_player_key)
    pair_mask = (
        ((winner_keys == key_a) & (loser_keys == key_b))
        | ((winner_keys == key_b) & (loser_keys == key_a))
    )
    meetings = matches.loc[pair_mask].copy()

    if meetings.empty:
        return _empty_summary(
            player_a,
            player_b,
            resolved_a=resolved_a,
            resolved_b=resolved_b,
            resolution_a=resolution_a,
            resolution_b=resolution_b,
        )

    meetings["_winner"] = meetings[winner_col].astype(str).str.strip()
    meetings["_winner_key"] = meetings[winner_col].map(canonical_player_key)
    wins_a = int((meetings["_winner_key"] == key_a).sum())
    wins_b = int((meetings["_winner_key"] == key_b).sum())

    if surface_col:
        surface_values = meetings[surface_col].astype(str).str.strip().str.casefold()
        surface_mask = surface_values == str(current_surface).strip().casefold()
        surface_meetings = meetings.loc[surface_mask].copy()
    else:
        surface_meetings = meetings.iloc[0:0].copy()

    surface_wins_a = int((surface_meetings["_winner_key"] == key_a).sum())
    surface_wins_b = int((surface_meetings["_winner_key"] == key_b).sum())

    if date_col:
        raw_dates = meetings[date_col]
        date_text = raw_dates.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        parsed_numeric = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
        parsed_general = pd.to_datetime(raw_dates, errors="coerce")
        meetings["_parsed_date"] = parsed_numeric.fillna(parsed_general)
        meetings = meetings.sort_values("_parsed_date", ascending=False, na_position="last")

    latest = meetings.iloc[0]
    latest_date = latest.get("_parsed_date")
    if pd.notna(latest_date):
        latest_date = pd.Timestamp(latest_date).date().isoformat()
    else:
        latest_date = "Date unavailable"

    details: list[str] = []
    if event_col and str(latest.get(event_col, "")).strip() not in {"", "nan", "None"}:
        details.append(str(latest.get(event_col)).strip())
    if round_col and str(latest.get(round_col, "")).strip() not in {"", "nan", "None"}:
        details.append(str(latest.get(round_col)).strip())

    score = ""
    if score_col and str(latest.get(score_col, "")).strip() not in {"", "nan", "None"}:
        score = str(latest.get(score_col)).strip()

    latest_winner_raw = str(latest["_winner"])
    latest_winner_key = canonical_player_key(latest_winner_raw)
    if latest_winner_key == key_a:
        latest_winner_display = str(player_a)
    elif latest_winner_key == key_b:
        latest_winner_display = str(player_b)
    else:
        latest_winner_display = latest_winner_raw

    return {
        "meetings": int(len(meetings)),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "surface_meetings": int(len(surface_meetings)),
        "surface_wins_a": surface_wins_a,
        "surface_wins_b": surface_wins_b,
        "last_meeting": {
            "date": latest_date,
            "winner": latest_winner_display,
            "event": " — ".join(details) if details else "Event unavailable",
            "score": score,
        },
        "requested_player_a": str(player_a),
        "requested_player_b": str(player_b),
        "resolved_player_a": resolved_a,
        "resolved_player_b": resolved_b,
        "resolution_a": resolution_a,
        "resolution_b": resolution_b,
    }
