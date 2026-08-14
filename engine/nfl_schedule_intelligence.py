"""Schedule intelligence for the Macabets NFL model.

The module intentionally separates *schedule context* from team talent.  Future
schedule difficulty is informational only; it never boosts a team's win
probability just because a hard schedule is coming.  Only opponents already
faced can create a small opponent-quality correction, and that correction is
aggressively sample-shrunk.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd

from engine.nfl_fetch import TEAM_ABBR_TO_NAME

SCHEDULE_PATH = Path(__file__).resolve().parents[1] / "data" / "nfl" / "schedules.csv"
NAME_TO_ABBR = {name: abbr for abbr, name in TEAM_ABBR_TO_NAME.items()}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _load_schedule(path: Path | str = SCHEDULE_PATH) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(file_path)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["gameday"] = pd.to_datetime(frame.get("gameday"), errors="coerce")
    frame["week"] = pd.to_numeric(frame.get("week"), errors="coerce")
    return frame


def _season_frame(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame[pd.to_numeric(frame.get("season"), errors="coerce") == int(season)].copy()
    if "game_type" in result.columns:
        result = result[result["game_type"].astype(str).str.upper().eq("REG")]
    return result


def _opponent_rows(schedule: pd.DataFrame, team_abbr: str) -> pd.DataFrame:
    if schedule.empty:
        return schedule
    mask = schedule["away_team"].astype(str).eq(team_abbr) | schedule["home_team"].astype(str).eq(team_abbr)
    rows = schedule.loc[mask].copy()
    rows["opponent_abbr"] = rows.apply(
        lambda r: r["home_team"] if str(r["away_team"]) == team_abbr else r["away_team"], axis=1
    )
    rows["opponent"] = rows["opponent_abbr"].map(TEAM_ABBR_TO_NAME)
    return rows


def _average_opponent_power(rows: pd.DataFrame, team_power: Mapping[str, float]) -> tuple[float | None, int]:
    if rows.empty:
        return None, 0
    values = [float(team_power[name]) for name in rows["opponent"].dropna() if name in team_power]
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def _rest_adjustment(game_row: pd.Series | None) -> tuple[float, str]:
    if game_row is None:
        return 0.0, "Schedule rest data unavailable."
    away_rest = pd.to_numeric(game_row.get("away_rest"), errors="coerce")
    home_rest = pd.to_numeric(game_row.get("home_rest"), errors="coerce")
    if pd.isna(away_rest) or pd.isna(home_rest):
        return 0.0, "Schedule rest data unavailable."
    diff = float(home_rest - away_rest)
    # Rest matters, but ordinary 6/7/8-day differences should not dominate.
    if abs(diff) <= 1.0:
        adjustment = 0.0
    else:
        adjustment = _clip((abs(diff) - 1.0) * 0.15, 0.0, 0.60)
        if diff < 0:
            adjustment *= -1.0
    return adjustment, f"Rest: away {int(away_rest)} days, home {int(home_rest)} days."


def _find_game(schedule: pd.DataFrame, away_abbr: str, home_abbr: str, game_date: date | str | None, week: int | None):
    if schedule.empty:
        return None
    matches = schedule[
        schedule["away_team"].astype(str).eq(away_abbr)
        & schedule["home_team"].astype(str).eq(home_abbr)
    ].copy()
    if game_date is not None:
        target = pd.Timestamp(str(game_date)).normalize()
        dated = matches[matches["gameday"].dt.normalize().eq(target)]
        if not dated.empty:
            return dated.iloc[0]
    if week is not None:
        weekly = matches[pd.to_numeric(matches["week"], errors="coerce").eq(int(week))]
        if not weekly.empty:
            return weekly.iloc[0]
    return matches.iloc[0] if not matches.empty else None


def build_schedule_context(
    *,
    away_team: str,
    home_team: str,
    season: int,
    team_power: Mapping[str, float],
    game_date: date | str | None = None,
    week: int | None = None,
    schedule_path: Path | str = SCHEDULE_PATH,
) -> dict:
    """Return schedule/rest context for a single matchup.

    Opponent-quality correction now lives in the dedicated opponent-adjusted
    performance layer so schedule strength is not counted twice. Full/remaining
    schedule difficulty here is descriptive only.
    """
    frame = _season_frame(_load_schedule(schedule_path), int(season))
    away_abbr, home_abbr = NAME_TO_ABBR.get(away_team), NAME_TO_ABBR.get(home_team)
    if frame.empty or not away_abbr or not home_abbr:
        return {
            "available": False,
            "season": int(season),
            "home_margin_adjustment": 0.0,
            "confidence_penalty": 0.0,
            "summary": "Schedule intelligence unavailable.",
        }

    game_row = _find_game(frame, away_abbr, home_abbr, game_date, week)
    cutoff = pd.Timestamp(str(game_date)).normalize() if game_date is not None else None
    if cutoff is None and game_row is not None and pd.notna(game_row.get("gameday")):
        cutoff = pd.Timestamp(game_row["gameday"]).normalize()

    league_mean = sum(float(v) for v in team_power.values()) / max(len(team_power), 1)
    team_profiles = {}
    for team_name, abbr in ((away_team, away_abbr), (home_team, home_abbr)):
        rows = _opponent_rows(frame, abbr)
        full_avg, full_n = _average_opponent_power(rows, team_power)
        if cutoff is not None:
            played = rows[(rows["gameday"] < cutoff) & rows["away_score"].notna() & rows["home_score"].notna()]
            remaining = rows[rows["gameday"] >= cutoff]
        else:
            played = rows[rows["away_score"].notna() & rows["home_score"].notna()]
            remaining = rows[~(rows.index.isin(played.index))]
        # Recent opponent quality is more useful than an entire-season average.
        recent_played = played.sort_values("gameday").tail(5)
        played_avg, played_n = _average_opponent_power(recent_played, team_power)
        remaining_avg, remaining_n = _average_opponent_power(remaining, team_power)
        team_profiles[team_name] = {
            "full_schedule_avg_opponent_power": round(full_avg, 3) if full_avg is not None else None,
            "full_schedule_games": full_n,
            "recent_played_avg_opponent_power": round(played_avg, 3) if played_avg is not None else None,
            "recent_played_games": played_n,
            "remaining_avg_opponent_power": round(remaining_avg, 3) if remaining_avg is not None else None,
            "remaining_games": remaining_n,
            "full_schedule_vs_league": round((full_avg - league_mean), 3) if full_avg is not None else None,
        }

    # Opponent quality is still displayed here for schedule context, but no
    # scoreboard credit is applied in this module. The dedicated opponent-
    # adjusted performance layer owns that correction to prevent double count.
    sos_adjustment = 0.0

    rest_adjustment, rest_note = _rest_adjustment(game_row)
    div_game = bool(int(game_row.get("div_game", 0) or 0)) if game_row is not None and pd.notna(game_row.get("div_game")) else False
    scheduled_neutral = str(game_row.get("location") or "").strip().lower() == "neutral" if game_row is not None else False
    confidence_penalty = 1.5 if div_game else 0.0

    total_adjustment = _clip(rest_adjustment, -0.90, 0.90)
    notes = []
    notes.append(rest_note)
    if div_game:
        notes.append("Division familiarity modestly reduces confidence; it does not automatically force a closer spread.")
    if scheduled_neutral:
        notes.append("The schedule marks this as a neutral-site game.")

    return {
        "available": True,
        "season": int(season),
        "week": int(game_row["week"]) if game_row is not None and pd.notna(game_row.get("week")) else week,
        "game_id": str(game_row.get("game_id")) if game_row is not None else None,
        "div_game": div_game,
        "scheduled_neutral": scheduled_neutral,
        "rest_home_margin_adjustment": round(rest_adjustment, 2),
        "sos_home_margin_adjustment": round(sos_adjustment, 2),
        "home_margin_adjustment": round(total_adjustment, 2),
        "confidence_penalty": confidence_penalty,
        "league_mean_power": round(league_mean, 3),
        "away": team_profiles[away_team],
        "home": team_profiles[home_team],
        "summary": " ".join(notes),
    }
