from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd

from engine.api_tennis import APITennisClient, APITennisError
from engine.player_intelligence_store import write_player_intelligence
from engine.player_profiles import build_player_profile, canonical_player_key

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DESTINATION = DATA_DIR / "player_intelligence_atp.json"


def load_matches() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(DATA_DIR.glob("atp_matches_*.csv")):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No ATP match files were found in data/.")

    matches = pd.concat(frames, ignore_index=True, sort=False)
    if "tourney_date" in matches.columns:
        raw = matches["tourney_date"].astype(str).str.replace(r"\.0$", "", regex=True)
        matches["tourney_date"] = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    return matches


def first_value(row: dict, *names: str):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def main() -> int:
    event_date = date.today()
    matches = load_matches()
    client = APITennisClient()

    try:
        standings_response = client.get_standings("ATP", force_refresh=True)
    except APITennisError as exc:
        print(f"Unable to refresh ATP standings: {exc}")
        return 1

    ranking_rows = standings_response.result
    profiles = []
    seen: set[str] = set()

    for row in ranking_rows:
        player_name = str(first_value(row, "player", "player_name", "standing_player") or "").strip()
        player_key = canonical_player_key(player_name)
        if not player_key or player_key in seen:
            continue
        seen.add(player_key)

        profile = build_player_profile(
            matches,
            player_name,
            event_date,
            api_client=client,
            include_api=False,
            use_store=False,
        )
        profile.ranking = _to_int(first_value(row, "place", "ranking", "position"))
        profile.ranking_points = _to_int(first_value(row, "points", "player_points"))
        raw_api_key = first_value(row, "player_key", "standing_player_key", "id")
        profile.api_player_key = str(raw_api_key) if raw_api_key not in (None, "") else None
        profile.api_source = standings_response.source
        profile.api_fetched_at = standings_response.fetched_at
        if "api_tennis_standings" not in profile.data_sources:
            profile.data_sources.append("api_tennis_standings")
        profiles.append(profile)

    destination = write_player_intelligence(
        profiles,
        destination=DESTINATION,
        as_of_date=event_date,
        metadata={
            "standings_source": standings_response.source,
            "standings_fetched_at": standings_response.fetched_at,
            "ranking_rows_received": len(ranking_rows),
            "historical_matches_loaded": len(matches),
            "api_requests_used": 1,
        },
    )
    print(f"Saved {len(profiles):,} ATP player profiles to {destination}")
    return 0


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
