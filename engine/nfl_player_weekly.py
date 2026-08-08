from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NFL_DATA_DIR = PROJECT_ROOT / "data" / "nfl"
DEFAULT_OUTPUT = NFL_DATA_DIR / "player_weekly_stats.csv"
DEFAULT_METADATA = NFL_DATA_DIR / "player_weekly_stats_metadata.json"

KEEP_COLUMNS = (
    "player_id", "player_display_name", "player_name", "position", "position_group",
    "recent_team", "team", "season", "week", "season_type",
    "attempts", "completions", "passing_yards", "passing_tds", "interceptions", "sacks",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "passing_epa", "rushing_epa", "receiving_epa", "dakota",
    "target_share", "air_yards_share", "wopr",
)


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def _load_one(season: int) -> pd.DataFrame:
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise RuntimeError("Install nflreadpy before refreshing NFL player stats.") from exc

    frame = _to_pandas(nfl.load_player_stats([int(season)], summary_level="week"))
    if "season" in frame.columns:
        frame = frame[pd.to_numeric(frame["season"], errors="coerce").eq(int(season))].copy()
    if "season_type" in frame.columns:
        frame = frame[frame["season_type"].astype(str).str.upper().eq("REG")].copy()
    return frame


def _prepare(frame: pd.DataFrame, *, active_season: int, requested_season: int, fallback: bool) -> pd.DataFrame:
    if frame.empty:
        return frame
    available = [column for column in KEEP_COLUMNS if column in frame.columns]
    clean = frame[available].copy()

    # nfl_rating_engine uses this field to keep previous-season production as a
    # modest prior, while allowing current-season production to grow toward 80%.
    clean["macabets_performance_cap"] = 0.20 if fallback else 0.80
    clean["macabets_requested_season"] = int(requested_season)
    clean["macabets_active_season"] = int(active_season)
    clean["macabets_is_fallback"] = bool(fallback)
    return clean.reset_index(drop=True)


def refresh_player_weekly_stats(
    requested_season: int | None = None,
    output_path: Path | str = DEFAULT_OUTPUT,
    metadata_path: Path | str = DEFAULT_METADATA,
) -> dict[str, Any]:
    requested = int(requested_season or date.today().year)
    active = requested
    fallback = False

    try:
        frame = _load_one(requested)
    except Exception as exc:
        active = requested - 1
        fallback = True
        print(
            f"Weekly player stats unavailable for {requested} "
            f"({type(exc).__name__}: {exc}); using {active} as a preseason prior."
        )
        frame = _load_one(active)

    if frame.empty and not fallback:
        active = requested - 1
        fallback = True
        print(f"No weekly player stats available for {requested}; using {active} as a preseason prior.")
        frame = _load_one(active)

    if frame.empty:
        raise RuntimeError(f"No usable nflverse weekly player stats found for {requested} or {active}.")

    clean = _prepare(frame, active_season=active, requested_season=requested, fallback=fallback)
    output = Path(output_path)
    metadata = Path(metadata_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    temp = output.with_suffix(output.suffix + ".tmp")
    clean.to_csv(temp, index=False)
    temp.replace(output)

    through_week = None
    if "week" in clean.columns and not clean.empty:
        week_values = pd.to_numeric(clean["week"], errors="coerce").dropna()
        if not week_values.empty:
            through_week = int(week_values.max())

    payload = {
        "source": "nflverse via nflreadpy.load_player_stats(summary_level='week')",
        "requested_season": requested,
        "active_season": active,
        "fallback_prior": fallback,
        "performance_cap": 0.20 if fallback else 0.80,
        "through_week": through_week,
        "rows": int(len(clean)),
        "players": int(clean.get("player_id", clean.get("player_display_name", pd.Series(dtype=object))).nunique()),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_path": str(output),
    }
    meta_temp = metadata.with_suffix(metadata.suffix + ".tmp")
    meta_temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    meta_temp.replace(metadata)
    return payload
