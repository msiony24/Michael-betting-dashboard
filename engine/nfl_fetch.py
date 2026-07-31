"""Download and aggregate public NFL data from nflverse.

The network dependency is isolated here so the Streamlit app can continue to run
from the last successful snapshot when nflverse is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

TEAM_ABBR_TO_NAME = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


@dataclass(frozen=True)
class FetchResult:
    season: int
    rows: int
    output_path: str
    fetched_at_utc: str


def _to_pandas(frame) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def _percentile_score(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        numeric = -numeric
    ranks = numeric.rank(method="average", pct=True)
    # Deliberately compressed to prevent small early-season samples from
    # creating false certainty.
    return (45.0 + ranks.fillna(0.5) * 45.0).clip(45.0, 90.0)


def _weighted_rating(frame: pd.DataFrame, columns: Iterable[tuple[str, float]]) -> pd.Series:
    result = pd.Series(0.0, index=frame.index)
    total_weight = 0.0
    for column, weight in columns:
        if column in frame:
            result += frame[column].fillna(67.5) * weight
            total_weight += weight
    return result / total_weight if total_weight else pd.Series(67.5, index=frame.index)


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default)


def _final_game_rows(pbp: pd.DataFrame) -> pd.DataFrame:
    """Return one final-score row per game when nflverse score fields exist."""
    required = {"game_id", "home_team", "away_team"}
    if not required.issubset(pbp.columns):
        return pd.DataFrame()

    home_score_col = next(
        (name for name in ("total_home_score", "home_score") if name in pbp.columns),
        None,
    )
    away_score_col = next(
        (name for name in ("total_away_score", "away_score") if name in pbp.columns),
        None,
    )
    if not home_score_col or not away_score_col:
        return pd.DataFrame()

    sortable = pbp.copy()
    if "play_id" in sortable.columns:
        sortable["_order"] = _numeric(sortable, "play_id")
    else:
        sortable["_order"] = range(len(sortable))

    final = sortable.sort_values("_order").groupby("game_id", as_index=False).tail(1)
    columns = ["game_id", "home_team", "away_team", home_score_col, away_score_col]
    if "week" in final.columns:
        columns.append("week")
    final = final[columns].copy()
    final = final.rename(columns={home_score_col: "home_score", away_score_col: "away_score"})
    final["home_score"] = pd.to_numeric(final["home_score"], errors="coerce")
    final["away_score"] = pd.to_numeric(final["away_score"], errors="coerce")
    return final.dropna(subset=["home_score", "away_score"])


def _recent_form_table(plays: pd.DataFrame, original_pbp: pd.DataFrame) -> pd.DataFrame:
    """Build a rolling recent-form signal from performance and game results.

    Performance receives 70% of the score and results 30%. The latest five weeks
    are used with exponential recency weights. If score fields are unavailable,
    the result component remains neutral rather than being invented.
    """
    if "week" not in plays.columns:
        return pd.DataFrame(columns=["team_abbr", "recent_form", "recent_performance", "recent_results"])

    weekly_offense = plays.groupby(["posteam", "week"], as_index=False).agg(
        offensive_epa=("epa", "mean"),
        offensive_success=("_success", "mean"),
    ).rename(columns={"posteam": "team_abbr"})
    weekly_defense = plays.groupby(["defteam", "week"], as_index=False).agg(
        defensive_epa_allowed=("epa", "mean"),
        defensive_success_allowed=("_success", "mean"),
    ).rename(columns={"defteam": "team_abbr"})
    weekly = weekly_offense.merge(weekly_defense, on=["team_abbr", "week"], how="outer")
    weekly["net_epa"] = weekly["offensive_epa"] - weekly["defensive_epa_allowed"]
    weekly["success_margin"] = weekly["offensive_success"] - weekly["defensive_success_allowed"]
    weekly["performance_raw"] = weekly["net_epa"] * 0.75 + weekly["success_margin"] * 0.25

    games = _final_game_rows(original_pbp)
    result_rows: list[dict[str, float | str | int]] = []
    if not games.empty:
        for _, game in games.iterrows():
            week = int(game.get("week", 0) or 0)
            margin = float(game["home_score"] - game["away_score"])
            for team, signed_margin in (
                (str(game["home_team"]), margin),
                (str(game["away_team"]), -margin),
            ):
                result_rows.append({
                    "team_abbr": team,
                    "week": week,
                    "win_value": 1.0 if signed_margin > 0 else 0.5 if signed_margin == 0 else 0.0,
                    "point_margin": signed_margin,
                })
    results = pd.DataFrame(result_rows)
    if not results.empty:
        weekly = weekly.merge(results, on=["team_abbr", "week"], how="left")
    else:
        weekly["win_value"] = 0.5
        weekly["point_margin"] = 0.0

    rows = []
    for team, group in weekly.groupby("team_abbr"):
        group = group.sort_values("week", ascending=False).head(5).copy()
        if group.empty:
            continue
        weights = pd.Series([0.38, 0.25, 0.17, 0.12, 0.08][: len(group)], index=group.index)
        weights = weights / weights.sum()
        performance = float((group["performance_raw"].fillna(0.0) * weights).sum())
        # Point margin is capped because a single blowout should not dominate form.
        margin_scaled = group["point_margin"].fillna(0.0).clip(-21, 21) / 21.0
        result_raw = group["win_value"].fillna(0.5) * 0.65 + ((margin_scaled + 1.0) / 2.0) * 0.35
        results_value = float((result_raw * weights).sum())
        rows.append({"team_abbr": team, "recent_performance_raw": performance, "recent_results_raw": results_value})

    form = pd.DataFrame(rows)
    if form.empty:
        return pd.DataFrame(columns=["team_abbr", "recent_form", "recent_performance", "recent_results"])
    form["recent_performance"] = _percentile_score(form["recent_performance_raw"])
    form["recent_results"] = _percentile_score(form["recent_results_raw"])
    form["recent_form"] = form["recent_performance"] * 0.70 + form["recent_results"] * 0.30
    return form[["team_abbr", "recent_form", "recent_performance", "recent_results"]]


def build_team_snapshot(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    required = {"posteam", "defteam", "epa"}
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"nflverse play-by-play data is missing required columns: {sorted(missing)}")

    original = pbp.copy()
    plays = pbp.copy()
    if "season" in plays:
        plays = plays[pd.to_numeric(plays["season"], errors="coerce") == int(season)]
    if "season_type" in plays:
        plays = plays[plays["season_type"].astype(str).eq("REG")]
    original = original.loc[plays.index.intersection(original.index)].copy()
    if "play_type" in plays:
        plays = plays[plays["play_type"].isin(["pass", "run"])]
    plays = plays[plays["posteam"].notna() & plays["defteam"].notna()]
    plays["epa"] = pd.to_numeric(plays["epa"], errors="coerce")
    plays = plays[plays["epa"].notna()]

    if plays.empty:
        raise ValueError(f"No usable regular-season plays were found for {season}.")

    success = _numeric(plays, "success", 0.0)
    if "success" not in plays.columns:
        success = plays["epa"].gt(0).astype(float)
    yards = _numeric(plays, "yards_gained", 0.0)
    pass_flag = _numeric(plays, "pass_attempt", 0.0).astype(bool)
    rush_flag = _numeric(plays, "rush_attempt", 0.0).astype(bool)
    if "pass_attempt" not in plays.columns and "play_type" in plays.columns:
        pass_flag = plays["play_type"].eq("pass")
    if "rush_attempt" not in plays.columns and "play_type" in plays.columns:
        rush_flag = plays["play_type"].eq("run")

    plays["_success"] = success
    plays["_explosive"] = ((pass_flag & (yards >= 20)) | (rush_flag & (yards >= 10))).astype(float)
    interception = _numeric(plays, "interception", 0.0)
    fumble_lost = _numeric(plays, "fumble_lost", 0.0)
    plays["_turnover"] = ((interception > 0) | (fumble_lost > 0)).astype(float)
    plays["_pass"] = pass_flag.astype(float)
    plays["_rush"] = rush_flag.astype(float)
    plays["_sack"] = _numeric(plays, "sack", 0.0)
    plays["_qb_hit"] = _numeric(plays, "qb_hit", 0.0)

    offense = plays.groupby("posteam", as_index=False).agg(
        plays=("epa", "size"),
        offense_epa_per_play=("epa", "mean"),
        offense_success_rate=("_success", "mean"),
        offense_explosive_rate=("_explosive", "mean"),
        offense_turnover_rate=("_turnover", "mean"),
    ).rename(columns={"posteam": "team_abbr"})

    defense = plays.groupby("defteam", as_index=False).agg(
        defense_epa_allowed=("epa", "mean"),
        defense_success_allowed=("_success", "mean"),
        defense_explosive_allowed=("_explosive", "mean"),
        defense_takeaway_rate=("_turnover", "mean"),
    ).rename(columns={"defteam": "team_abbr"})

    pass_plays = plays[pass_flag].copy()
    qb = pass_plays.groupby("posteam", as_index=False).agg(
        qb_epa_per_dropback=("epa", "mean"),
        qb_success_rate=("_success", "mean"),
        offense_sack_rate=("_sack", "mean"),
        offense_qb_hit_rate=("_qb_hit", "mean"),
    ).rename(columns={"posteam": "team_abbr"})
    qb["qb_cpoe"] = (
        pass_plays.groupby("posteam")["cpoe"].mean().reindex(qb["team_abbr"]).to_numpy()
        if "cpoe" in pass_plays.columns else 0.0
    )

    rush_plays = plays[rush_flag].copy()
    rush_offense = rush_plays.groupby("posteam", as_index=False).agg(
        rush_epa=("epa", "mean"), rush_success=("_success", "mean")
    ).rename(columns={"posteam": "team_abbr"})

    front = plays.groupby("defteam", as_index=False).agg(
        defense_sack_rate=("_sack", "mean"), defense_qb_hit_rate=("_qb_hit", "mean")
    ).rename(columns={"defteam": "team_abbr"})
    rush_defense = rush_plays.groupby("defteam", as_index=False).agg(
        rush_epa_allowed=("epa", "mean"), rush_success_allowed=("_success", "mean")
    ).rename(columns={"defteam": "team_abbr"})
    pass_defense = pass_plays.groupby("defteam", as_index=False).agg(
        pass_epa_allowed=("epa", "mean"),
        pass_success_allowed=("_success", "mean"),
        pass_explosive_allowed=("_explosive", "mean"),
    ).rename(columns={"defteam": "team_abbr"})

    if "special_teams_play" in original.columns:
        st_mask = pd.to_numeric(original["special_teams_play"], errors="coerce").fillna(0).astype(bool)
        st_plays = original[st_mask & original["posteam"].notna()].copy()
        special = st_plays.groupby("posteam", as_index=False)["epa"].mean().rename(
            columns={"posteam": "team_abbr", "epa": "special_teams_epa"}
        ) if not st_plays.empty else pd.DataFrame(columns=["team_abbr", "special_teams_epa"])
    else:
        special = pd.DataFrame(columns=["team_abbr", "special_teams_epa"])

    stats = offense.merge(defense, on="team_abbr", how="outer")
    for table in (qb, rush_offense, front, rush_defense, pass_defense, special):
        stats = stats.merge(table, on="team_abbr", how="left")

    net = stats.set_index("team_abbr")["offense_epa_per_play"].sub(
        stats.set_index("team_abbr")["defense_epa_allowed"]
    )
    opp_rows = []
    for team, group in plays.groupby("posteam"):
        opp_rows.append((team, group["defteam"].map(net).mean()))
    stats = stats.merge(
        pd.DataFrame(opp_rows, columns=["team_abbr", "sos_opponent_net_epa"]),
        on="team_abbr", how="left",
    )

    stats["offense"] = _weighted_rating(pd.DataFrame({
        "epa": _percentile_score(stats["offense_epa_per_play"]),
        "success": _percentile_score(stats["offense_success_rate"]),
        "explosive": _percentile_score(stats["offense_explosive_rate"]),
        "turnovers": _percentile_score(stats["offense_turnover_rate"], higher_is_better=False),
    }), [("epa", .45), ("success", .30), ("explosive", .15), ("turnovers", .10)])

    stats["defense"] = _weighted_rating(pd.DataFrame({
        "epa": _percentile_score(stats["defense_epa_allowed"], higher_is_better=False),
        "success": _percentile_score(stats["defense_success_allowed"], higher_is_better=False),
        "explosive": _percentile_score(stats["defense_explosive_allowed"], higher_is_better=False),
        "takeaways": _percentile_score(stats["defense_takeaway_rate"]),
    }), [("epa", .45), ("success", .30), ("explosive", .15), ("takeaways", .10)])

    stats["quarterback"] = _weighted_rating(pd.DataFrame({
        "epa": _percentile_score(stats["qb_epa_per_dropback"]),
        "success": _percentile_score(stats["qb_success_rate"]),
        "cpoe": _percentile_score(stats["qb_cpoe"]),
    }), [("epa", .55), ("success", .25), ("cpoe", .20)])

    stats["offensive_line"] = _weighted_rating(pd.DataFrame({
        "sack_avoidance": _percentile_score(stats["offense_sack_rate"], higher_is_better=False),
        "hit_avoidance": _percentile_score(stats["offense_qb_hit_rate"], higher_is_better=False),
        "rush_epa": _percentile_score(stats["rush_epa"]),
        "rush_success": _percentile_score(stats["rush_success"]),
    }), [("sack_avoidance", .45), ("hit_avoidance", .20), ("rush_epa", .20), ("rush_success", .15)])

    stats["defensive_line"] = _weighted_rating(pd.DataFrame({
        "sacks": _percentile_score(stats["defense_sack_rate"]),
        "hits": _percentile_score(stats["defense_qb_hit_rate"]),
        "rush_epa": _percentile_score(stats["rush_epa_allowed"], higher_is_better=False),
        "rush_success": _percentile_score(stats["rush_success_allowed"], higher_is_better=False),
    }), [("sacks", .35), ("hits", .20), ("rush_epa", .25), ("rush_success", .20)])

    stats["secondary"] = _weighted_rating(pd.DataFrame({
        "pass_epa": _percentile_score(stats["pass_epa_allowed"], higher_is_better=False),
        "pass_success": _percentile_score(stats["pass_success_allowed"], higher_is_better=False),
        "explosive": _percentile_score(stats["pass_explosive_allowed"], higher_is_better=False),
    }), [("pass_epa", .50), ("pass_success", .30), ("explosive", .20)])

    stats["strength_of_schedule"] = _percentile_score(stats["sos_opponent_net_epa"])
    stats["special_teams"] = _percentile_score(stats["special_teams_epa"])
    stats = stats.merge(_recent_form_table(plays, original), on="team_abbr", how="left")

    stats["team"] = stats["team_abbr"].map(TEAM_ABBR_TO_NAME)
    stats = stats[stats["team"].notna()].copy()
    stats["season"] = int(season)
    stats["through_week"] = int(pd.to_numeric(plays.get("week"), errors="coerce").max()) if "week" in plays else None
    stats["games_or_sample_plays"] = stats["plays"].fillna(0).astype(int)
    stats["data_source"] = "nflverse play-by-play"
    stats["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return stats.sort_values("team").reset_index(drop=True)


def fetch_and_build(season: int, output_path: str | Path) -> FetchResult:
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise RuntimeError("Install nflreadpy before refreshing NFL data.") from exc

    pbp = _to_pandas(nfl.load_pbp([int(season)]))
    snapshot = build_team_snapshot(pbp, int(season))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(output, index=False)
    return FetchResult(
        season=int(season), rows=len(snapshot), output_path=str(output),
        fetched_at_utc=snapshot["updated_at_utc"].iloc[0],
    )
