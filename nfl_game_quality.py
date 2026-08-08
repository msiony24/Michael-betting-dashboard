"""Underlying game-quality context for the Macabets NFL model.

Final scores are noisy.  This module summarizes how well a team actually played
in recent games using efficiency, success, explosiveness and turnover context.
The signal is deliberately small and sample-shrunk; it refines recent form but
cannot override the core team-strength model by itself.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.nfl_fetch import TEAM_ABBR_TO_NAME

GAME_QUALITY_PATH = Path(__file__).resolve().parents[1] / "data" / "nfl" / "game_quality.csv"
NAME_TO_ABBR = {name: abbr for abbr, name in TEAM_ABBR_TO_NAME.items()}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _load(path: Path | str = GAME_QUALITY_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    frame = pd.read_csv(p)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["gameday"] = pd.to_datetime(frame.get("gameday"), errors="coerce")
    for col in (
        "season", "week", "quality_score", "underlying_edge", "score_margin",
        "turnover_margin", "turnover_luck_index", "net_epa", "success_margin",
        "yards_per_play_margin", "explosive_margin",
    ):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _recent_profile(frame: pd.DataFrame, team_abbr: str, season: int, cutoff=None) -> dict:
    if frame.empty:
        return {"available": False, "games": 0}
    rows = frame[(frame["team_abbr"].astype(str) == team_abbr) & frame["season"].eq(int(season))].copy()
    if cutoff is not None:
        rows = rows[rows["gameday"] < cutoff]
    rows = rows.sort_values(["gameday", "week"]).tail(5)
    if rows.empty:
        return {"available": False, "games": 0}

    # Latest game receives the most weight, but not enough to let one weird game dominate.
    base = [0.10, 0.13, 0.17, 0.24, 0.36]
    weights = pd.Series(base[-len(rows):], index=rows.index, dtype=float)
    weights = weights / weights.sum()

    def weighted(col: str, default: float = 0.0) -> float:
        if col not in rows:
            return default
        vals = pd.to_numeric(rows[col], errors="coerce").fillna(default)
        return float((vals * weights).sum())

    wins = pd.to_numeric(rows.get("score_margin"), errors="coerce").fillna(0).gt(0).sum()
    return {
        "available": True,
        "games": int(len(rows)),
        "record": f"{int(wins)}-{int(len(rows)-wins)}",
        "quality_score": round(weighted("quality_score", 50.0), 1),
        "underlying_edge": round(weighted("underlying_edge"), 3),
        "score_margin": round(weighted("score_margin"), 2),
        "net_epa": round(weighted("net_epa"), 3),
        "success_margin": round(weighted("success_margin"), 3),
        "yards_per_play_margin": round(weighted("yards_per_play_margin"), 2),
        "explosive_margin": round(weighted("explosive_margin"), 3),
        "turnover_margin": round(weighted("turnover_margin"), 2),
        "turnover_luck_index": round(weighted("turnover_luck_index"), 2),
    }


def build_game_quality_context(
    *,
    away_team: str,
    home_team: str,
    season: int,
    game_date=None,
    quality_path: Path | str = GAME_QUALITY_PATH,
) -> dict:
    """Compare recent underlying game quality for the two teams.

    The probability input is expressed as a small home-margin adjustment.  Five
    games are required for the full effect; one game receives only 20% weight.
    Positive turnover luck is *not* rewarded as quality and can slightly reduce
    confidence when scoreboard results are running ahead of underlying play.
    """
    frame = _load(quality_path)
    away_abbr, home_abbr = NAME_TO_ABBR.get(away_team), NAME_TO_ABBR.get(home_team)
    if frame.empty or not away_abbr or not home_abbr:
        return {
            "available": False,
            "home_margin_adjustment": 0.0,
            "confidence_penalty": 0.0,
            "summary": "Recent game-quality data unavailable.",
        }
    season_rows = frame[frame["season"].eq(int(season))]
    if season_rows.empty:
        return {
            "available": False,
            "season": int(season),
            "home_margin_adjustment": 0.0,
            "confidence_penalty": 0.0,
            "summary": "No current-season game-quality sample is available yet.",
        }

    cutoff = pd.Timestamp(str(game_date)).normalize() if game_date is not None else None
    away = _recent_profile(season_rows, away_abbr, season, cutoff)
    home = _recent_profile(season_rows, home_abbr, season, cutoff)
    if not away.get("available") or not home.get("available"):
        return {
            "available": False,
            "season": int(season),
            "away": away,
            "home": home,
            "home_margin_adjustment": 0.0,
            "confidence_penalty": 0.0,
            "summary": "Not enough current-season game-quality data for both teams.",
        }

    sample = min(int(away["games"]), int(home["games"]), 5) / 5.0
    # 10 quality-score points is meaningful but should still be worth well under a point.
    quality_gap = float(home["quality_score"]) - float(away["quality_score"])
    adjustment = _clip(quality_gap * 0.055 * sample, -0.75, 0.75)

    # If either team is substantially outperforming its underlying play through turnover
    # fortune, lower certainty a little instead of reversing the pick.
    luck_gap = abs(float(home["turnover_luck_index"]) - float(away["turnover_luck_index"]))
    confidence_penalty = _clip(max(0.0, luck_gap - 0.55) * 0.8 * sample, 0.0, 1.25)

    leader = home_team if quality_gap > 1.0 else away_team if quality_gap < -1.0 else None
    notes = []
    if leader:
        notes.append(f"{leader} has the stronger recent underlying game-quality profile.")
    else:
        notes.append("Recent underlying game quality is broadly similar.")
    if confidence_penalty >= 0.25:
        notes.append("Turnover-driven score results add extra uncertainty.")

    return {
        "available": True,
        "season": int(season),
        "away": away,
        "home": home,
        "quality_gap_home_minus_away": round(quality_gap, 2),
        "sample_weight": round(sample, 2),
        "home_margin_adjustment": round(adjustment, 2),
        "confidence_penalty": round(confidence_penalty, 2),
        "summary": " ".join(notes),
    }
