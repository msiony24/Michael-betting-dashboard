"""Automatic NFL player availability from Sleeper.

Sleeper supplies current player status/injury metadata. Footballguys remains the
source of truth for depth order, and Madden/current NFL evidence determine player
quality. This module never invents an injury penalty: only definitive unavailable
statuses remove a player from the active depth chart. Questionable/Doubtful are
preserved as uncertainty flags until a definitive status is reported.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NFL_DIR = PROJECT_ROOT / "data" / "nfl"
DEFAULT_AVAILABILITY_PATH = DEFAULT_NFL_DIR / "sleeper_availability.csv"
DEFAULT_STATUS_PATH = DEFAULT_NFL_DIR / "sleeper_availability_status.json"
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"


def normalize_player_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text.lower())
    return re.sub(r"[^a-z0-9]", "", text)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def classify_availability(*, roster_status: Any = "", injury_status: Any = "", active: Any = None) -> tuple[str, bool]:
    """Return (state, definitively_unavailable).

    Only definitive states trigger depth-chart promotion. Doubtful and
    Questionable remain uncertainty states rather than being silently benched.
    """
    roster = _text(roster_status).lower()
    injury = _text(injury_status).lower()

    # Sleeper often places roster-list designations (IR/PUP/Sus) in
    # ``injury_status`` while leaving ``status`` as Active or Inactive. Treat
    # those exact designations as hard-unavailable too; otherwise an IR/PUP
    # starter can incorrectly remain in Macabets' active lineup.
    hard_injury = {
        "out", "ir", "pup", "sus", "suspended", "nfi", "cov",
        "injured reserve", "injured_reserve", "reserve/injured", "reserve injured",
        "physically unable", "commissioner exempt",
        "non-football injury", "non football injury", "non-football illness",
    }
    hard_roster_tokens = (
        "injured reserve", "injured_reserve", "reserve/injured", "reserve injured",
        "physically unable", "pup", "suspended", "commissioner exempt",
        "non-football injury", "non football injury", "non-football illness",
    )
    if injury in hard_injury or any(token in roster for token in hard_roster_tokens):
        return "Out", True
    if "doubt" in injury:
        return "Doubtful", False
    if "question" in injury:
        return "Questionable", False
    if injury in {"probable", "prob"}:
        return "Probable", False
    if active is False and roster and roster not in {"active", ""}:
        return "Inactive", True
    return "Active", False


def sleeper_payload_to_frame(payload: dict[str, Any], *, updated_at_utc: str | None = None) -> pd.DataFrame:
    updated = updated_at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    for player_id, player in (payload or {}).items():
        if not isinstance(player, dict):
            continue
        first = _text(player.get("first_name"))
        last = _text(player.get("last_name"))
        full = _text(player.get("full_name")) or " ".join(v for v in (first, last) if v).strip()
        team = _text(player.get("team")).upper()
        if not full or not team:
            continue
        roster_status = _text(player.get("status"))
        injury_status = _text(player.get("injury_status"))
        state, unavailable = classify_availability(
            roster_status=roster_status,
            injury_status=injury_status,
            active=player.get("active"),
        )
        rows.append({
            "sleeper_player_id": _text(player_id),
            "player_name": full,
            "name_key": normalize_player_name(full),
            "team_abbr": team,
            "position": _text(player.get("position")).upper(),
            "active": player.get("active"),
            "roster_status": roster_status,
            "injury_status": injury_status,
            "practice_participation": _text(player.get("practice_participation")),
            "injury_start_date": _text(player.get("injury_start_date")),
            "depth_chart_position": _text(player.get("depth_chart_position")),
            "news_updated": _text(player.get("news_updated")),
            "availability_state": state,
            "definitively_unavailable": bool(unavailable),
            "source": "Sleeper",
            "updated_at_utc": updated,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[
            "sleeper_player_id", "player_name", "name_key", "team_abbr", "position",
            "active", "roster_status", "injury_status", "practice_participation",
            "injury_start_date", "depth_chart_position", "news_updated",
            "availability_state", "definitively_unavailable", "source", "updated_at_utc",
        ])
    return frame.sort_values(["team_abbr", "player_name"]).drop_duplicates(["team_abbr", "name_key"], keep="last").reset_index(drop=True)


def refresh_sleeper_availability(
    *,
    output_path: Path | str = DEFAULT_AVAILABILITY_PATH,
    status_path: Path | str = DEFAULT_STATUS_PATH,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    client = session or requests.Session()
    response = client.get(SLEEPER_PLAYERS_URL, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Sleeper NFL players endpoint returned an unexpected payload.")

    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame = sleeper_payload_to_frame(payload, updated_at_utc=updated)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(output)

    definitive = int(frame["definitively_unavailable"].astype(bool).sum()) if not frame.empty else 0
    uncertain = int(frame["availability_state"].isin(["Questionable", "Doubtful"]).sum()) if not frame.empty else 0
    status = {
        "schema_version": "1.0",
        "source": "Sleeper",
        "source_url": SLEEPER_PLAYERS_URL,
        "updated_at_utc": updated,
        "players": int(len(frame)),
        "teams": int(frame["team_abbr"].nunique()) if not frame.empty else 0,
        "definitively_unavailable": definitive,
        "uncertain": uncertain,
        "file": str(output),
    }
    status_output = Path(status_path)
    status_output.parent.mkdir(parents=True, exist_ok=True)
    status_temp = status_output.with_suffix(status_output.suffix + ".tmp")
    status_temp.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    status_temp.replace(status_output)
    return status


def load_availability(path: Path | str = DEFAULT_AVAILABILITY_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()
    if "name_key" not in frame.columns and "player_name" in frame.columns:
        frame["name_key"] = frame["player_name"].map(normalize_player_name)
    # Reclassify on load instead of trusting an older CSV's cached state. This
    # makes classification fixes effective immediately and protects the rating
    # engine from stale/misclassified availability snapshots between refreshes.
    def _active_value(value: Any):
        text = _text(value).lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        return None

    classified = frame.apply(
        lambda row: classify_availability(
            roster_status=row.get("roster_status", ""),
            injury_status=row.get("injury_status", ""),
            active=_active_value(row.get("active", None)),
        ),
        axis=1,
    )
    frame["availability_state"] = [state for state, _ in classified]
    frame["definitively_unavailable"] = [bool(unavailable) for _, unavailable in classified]
    return frame


def load_availability_status(path: Path | str = DEFAULT_STATUS_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
