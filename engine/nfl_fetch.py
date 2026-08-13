"""Download and aggregate public NFL data from nflverse.

The network dependency is isolated here so the Streamlit app can continue to run
from the last successful snapshot when nflverse is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import time

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
    source_mode: str = "fresh"
    warning: str = ""


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




def build_scheme_snapshot(
    pbp: pd.DataFrame,
    season: int,
    *,
    ftn: pd.DataFrame | None = None,
    participation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build team-level scheme/tendency rates from public play data.

    Play-by-play supplies stable behavioral and situation splits. FTN charting
    adds motion/play-action/RPO/blitz detail when available. Participation adds
    man/zone coverage splits when that season is published. Missing optional
    charting fields remain NA rather than being guessed.
    """
    plays = pbp.copy()
    if "season" in plays.columns:
        plays = plays[pd.to_numeric(plays["season"], errors="coerce").eq(int(season))]
    if "season_type" in plays.columns:
        plays = plays[plays["season_type"].astype(str).eq("REG")]
    if "play_type" in plays.columns:
        plays = plays[plays["play_type"].isin(["pass", "run"])].copy()
    plays = plays[plays["posteam"].notna() & plays["defteam"].notna()].copy()
    if plays.empty:
        return pd.DataFrame()

    pass_flag = _numeric(plays, "pass_attempt", 0.0).astype(bool)
    rush_flag = _numeric(plays, "rush_attempt", 0.0).astype(bool)
    if "pass_attempt" not in plays.columns and "play_type" in plays.columns:
        pass_flag = plays["play_type"].eq("pass")
    if "rush_attempt" not in plays.columns and "play_type" in plays.columns:
        rush_flag = plays["play_type"].eq("run")
    plays["_pass"] = pass_flag.astype(float)
    plays["_rush"] = rush_flag.astype(float)
    plays["_success"] = _numeric(plays, "success", 0.0) if "success" in plays.columns else _numeric(plays, "epa", 0.0).gt(0).astype(float)
    yards = _numeric(plays, "yards_gained", 0.0)
    plays["_explosive"] = ((pass_flag & (yards >= 20)) | (rush_flag & (yards >= 10))).astype(float)
    plays["_pressure"] = pd.concat([_numeric(plays, "qb_hit", 0.0), _numeric(plays, "sack", 0.0)], axis=1).max(axis=1).clip(0, 1)

    down = _numeric(plays, "down", 0.0)
    early = plays[down.isin([1.0, 2.0])].copy()
    neutral = early.copy()
    if "wp" in neutral.columns:
        wp = pd.to_numeric(neutral["wp"], errors="coerce")
        neutral = neutral[wp.between(0.20, 0.80, inclusive="both")]
    rz = plays[_numeric(plays, "yardline_100", 999.0).le(20.0)].copy()

    offense = plays.groupby("posteam", as_index=False).agg(
        offensive_plays=("_pass", "size"),
        pass_rate=("_pass", "mean"),
        offense_explosive_rate=("_explosive", "mean"),
    ).rename(columns={"posteam": "team_abbr"})
    early_o = early.groupby("posteam", as_index=False)["_pass"].mean().rename(columns={"posteam": "team_abbr", "_pass": "early_down_pass_rate"})
    neutral_o = neutral.groupby("posteam", as_index=False)["_pass"].mean().rename(columns={"posteam": "team_abbr", "_pass": "neutral_early_down_pass_rate"})
    defense = plays.groupby("defteam", as_index=False).agg(
        defense_explosive_allowed=("_explosive", "mean"),
    ).rename(columns={"defteam": "team_abbr"})
    pass_plays = plays[pass_flag].copy()
    pressure_o = pass_plays.groupby("posteam", as_index=False)["_pressure"].mean().rename(columns={"posteam": "team_abbr", "_pressure": "pressure_rate_allowed"})
    pressure_d = pass_plays.groupby("defteam", as_index=False)["_pressure"].mean().rename(columns={"defteam": "team_abbr", "_pressure": "pressure_rate"})
    rz_o = rz.groupby("posteam", as_index=False)["_success"].mean().rename(columns={"posteam": "team_abbr", "_success": "red_zone_success_rate"})
    rz_d = rz.groupby("defteam", as_index=False)["_success"].mean().rename(columns={"defteam": "team_abbr", "_success": "red_zone_defense_success_allowed"})

    # Plays/game is a useful volume proxy even when clock-derived pace is absent.
    games = plays.groupby("posteam")["game_id"].nunique() if "game_id" in plays.columns else pd.Series(dtype=float)
    offense["plays_per_game"] = offense["team_abbr"].map(
        lambda t: float(offense.loc[offense["team_abbr"].eq(t), "offensive_plays"].iloc[0]) / max(1.0, float(games.get(t, 1)))
    )

    # Approximate between-play tempo from consecutive same-offense snaps. Clock
    # deltas are capped to remove quarter breaks, timeouts and possession gaps.
    pace_rows = []
    if {"game_id", "game_seconds_remaining", "posteam"}.issubset(plays.columns):
        ordered = plays.copy()
        if "play_id" in ordered.columns:
            ordered = ordered.sort_values(["game_id", "play_id"])
        else:
            ordered = ordered.sort_values(["game_id", "game_seconds_remaining"], ascending=[True, False])
        ordered["_prev_team"] = ordered.groupby("game_id")["posteam"].shift(1)
        ordered["_prev_clock"] = ordered.groupby("game_id")["game_seconds_remaining"].shift(1)
        delta = pd.to_numeric(ordered["_prev_clock"], errors="coerce") - pd.to_numeric(ordered["game_seconds_remaining"], errors="coerce")
        ordered["_snap_gap"] = delta.where(ordered["_prev_team"].eq(ordered["posteam"]) & delta.between(5, 45))
        pace_rows = ordered.groupby("posteam", as_index=False)["_snap_gap"].mean().rename(columns={"posteam": "team_abbr", "_snap_gap": "seconds_per_play"})

    stats = offense.merge(early_o, on="team_abbr", how="left").merge(neutral_o, on="team_abbr", how="left")
    for table in (defense, pressure_o, pressure_d, rz_o, rz_d):
        stats = stats.merge(table, on="team_abbr", how="left")
    if isinstance(pace_rows, pd.DataFrame) and not pace_rows.empty:
        stats = stats.merge(pace_rows, on="team_abbr", how="left")
    else:
        stats["seconds_per_play"] = pd.NA

    # FTN play charting: join team identity from PBP via game/play IDs.
    for col in ("no_huddle_rate", "motion_rate", "play_action_rate", "rpo_rate", "blitz_rate"):
        stats[col] = pd.NA
    if ftn is not None and not ftn.empty and {"nflverse_game_id", "nflverse_play_id"}.issubset(ftn.columns) and {"game_id", "play_id"}.issubset(plays.columns):
        identity = plays[["game_id", "play_id", "posteam", "defteam"]].drop_duplicates(["game_id", "play_id"])
        chart = ftn.copy().merge(identity, left_on=["nflverse_game_id", "nflverse_play_id"], right_on=["game_id", "play_id"], how="left")
        offense_specs = {
            "is_no_huddle": "no_huddle_rate",
            "is_motion": "motion_rate",
            "is_play_action": "play_action_rate",
            "is_rpo": "rpo_rate",
        }
        for source, target in offense_specs.items():
            if source in chart.columns:
                temp = chart[chart["posteam"].notna()].copy()
                temp["_value"] = temp[source].astype("boolean").astype(float)
                agg = temp.groupby("posteam", as_index=False)["_value"].mean().rename(columns={"posteam": "team_abbr", "_value": target})
                stats = stats.drop(columns=[target], errors="ignore").merge(agg, on="team_abbr", how="left")
        if "n_blitzers" in chart.columns:
            temp = chart[chart["defteam"].notna()].copy()
            blitzers = pd.to_numeric(temp["n_blitzers"], errors="coerce")
            temp["_blitz"] = blitzers.gt(0).where(blitzers.notna())
            agg = temp.groupby("defteam", as_index=False)["_blitz"].mean().rename(columns={"defteam": "team_abbr", "_blitz": "blitz_rate"})
            stats = stats.drop(columns=["blitz_rate"], errors="ignore").merge(agg, on="team_abbr", how="left")

    # Participation data can provide man/zone splits. This dataset may lag the
    # current season; if unavailable the fields stay NA rather than being guessed.
    stats["man_rate"] = pd.NA
    stats["zone_rate"] = pd.NA
    if participation is not None and not participation.empty and "defense_man_zone_type" in participation.columns:
        part = participation.copy()
        team_col = None
        if "defense_team" in part.columns:
            team_col = "defense_team"
        elif "possession_team" in part.columns and "nflverse_game_id" in part.columns and "play_id" in part.columns and {"game_id", "play_id", "defteam"}.issubset(plays.columns):
            identity = plays[["game_id", "play_id", "defteam"]].drop_duplicates(["game_id", "play_id"])
            part = part.merge(identity, left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="left")
            team_col = "defteam"
        if team_col:
            cov = part[part[team_col].notna()].copy()
            text = cov["defense_man_zone_type"].astype(str).str.upper()
            cov["_man"] = text.str.contains("MAN", na=False).astype(float)
            cov["_zone"] = text.str.contains("ZONE", na=False).astype(float)
            valid = ~(text.isin(["", "NAN", "NONE"]))
            cov = cov[valid]
            if not cov.empty:
                agg = cov.groupby(team_col, as_index=False).agg(man_rate=("_man", "mean"), zone_rate=("_zone", "mean")).rename(columns={team_col: "team_abbr"})
                stats = stats.drop(columns=["man_rate", "zone_rate"], errors="ignore").merge(agg, on="team_abbr", how="left")

    stats["team"] = stats["team_abbr"].map(TEAM_ABBR_TO_NAME)
    stats = stats[stats["team"].notna()].copy()
    stats["season"] = int(season)
    stats["through_week"] = int(pd.to_numeric(plays.get("week"), errors="coerce").max()) if "week" in plays.columns else None
    stats["data_source"] = "nflverse play-by-play + FTN charting/participation when available"
    stats["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return stats.sort_values("team").reset_index(drop=True)

def build_game_quality_snapshot(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """Build one underlying-performance row per team/game from nflverse PBP.

    The score is intentionally based on repeatable play quality more than final
    margin. Turnover margin is stored as context, not rewarded as sustainable
    quality.
    """
    required = {"game_id", "posteam", "defteam", "epa"}
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"nflverse play-by-play data is missing game-quality columns: {sorted(missing)}")

    original = pbp.copy()
    plays = pbp.copy()
    if "season" in plays.columns:
        plays = plays[pd.to_numeric(plays["season"], errors="coerce").eq(int(season))]
    if "season_type" in plays.columns:
        plays = plays[plays["season_type"].astype(str).eq("REG")]
    original = original.loc[plays.index.intersection(original.index)].copy()
    if "play_type" in plays.columns:
        plays = plays[plays["play_type"].isin(["pass", "run"])]
    plays = plays[plays["posteam"].notna() & plays["defteam"].notna()].copy()
    plays["epa"] = pd.to_numeric(plays["epa"], errors="coerce")
    plays = plays[plays["epa"].notna()]
    if plays.empty:
        return pd.DataFrame()

    yards = _numeric(plays, "yards_gained", 0.0)
    success = _numeric(plays, "success", 0.0)
    if "success" not in plays.columns:
        success = plays["epa"].gt(0).astype(float)
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
    plays["_yards"] = yards

    offense = plays.groupby(["game_id", "posteam"], as_index=False).agg(
        offensive_plays=("epa", "size"),
        offense_epa=("epa", "mean"),
        offense_success=("_success", "mean"),
        offense_ypp=("_yards", "mean"),
        offense_explosive=("_explosive", "mean"),
        turnovers=("_turnover", "sum"),
    ).rename(columns={"posteam": "team_abbr"})
    defense = plays.groupby(["game_id", "defteam"], as_index=False).agg(
        defense_epa_allowed=("epa", "mean"),
        defense_success_allowed=("_success", "mean"),
        defense_ypp_allowed=("_yards", "mean"),
        defense_explosive_allowed=("_explosive", "mean"),
        takeaways=("_turnover", "sum"),
    ).rename(columns={"defteam": "team_abbr"})
    quality = offense.merge(defense, on=["game_id", "team_abbr"], how="inner")
    quality["net_epa"] = quality["offense_epa"] - quality["defense_epa_allowed"]
    quality["success_margin"] = quality["offense_success"] - quality["defense_success_allowed"]
    quality["yards_per_play_margin"] = quality["offense_ypp"] - quality["defense_ypp_allowed"]
    quality["explosive_margin"] = quality["offense_explosive"] - quality["defense_explosive_allowed"]
    quality["turnover_margin"] = quality["takeaways"] - quality["turnovers"]

    # A compact underlying edge. Final score and turnovers are deliberately not
    # part of this performance score; they are stored for variance diagnostics.
    quality["underlying_edge"] = (
        quality["net_epa"] * 0.58
        + quality["success_margin"] * 0.22
        + quality["yards_per_play_margin"] * 0.035
        + quality["explosive_margin"] * 0.20
    )
    quality["quality_score"] = _percentile_score(quality["underlying_edge"])

    finals = _final_game_rows(original)
    if not finals.empty:
        score_rows = []
        for _, game in finals.iterrows():
            for team, opponent, margin in (
                (str(game["home_team"]), str(game["away_team"]), float(game["home_score"] - game["away_score"])),
                (str(game["away_team"]), str(game["home_team"]), float(game["away_score"] - game["home_score"])),
            ):
                score_rows.append({
                    "game_id": str(game["game_id"]), "team_abbr": team, "opponent_abbr": opponent,
                    "score_margin": margin, "week": game.get("week"),
                })
        quality = quality.merge(pd.DataFrame(score_rows), on=["game_id", "team_abbr"], how="left")
    else:
        quality["opponent_abbr"] = None
        quality["score_margin"] = 0.0
        quality["week"] = None

    # Score-over-performance plus turnover margin is descriptive variance, not a
    # talent bonus. Positive values mean the scoreboard has been friendlier than
    # the underlying play profile.
    score_scaled = pd.to_numeric(quality["score_margin"], errors="coerce").fillna(0).clip(-28, 28) / 14.0
    perf_scaled = (pd.to_numeric(quality["quality_score"], errors="coerce").fillna(67.5) - 67.5) / 15.0
    turnover_scaled = pd.to_numeric(quality["turnover_margin"], errors="coerce").fillna(0) / 2.0
    quality["turnover_luck_index"] = (score_scaled - perf_scaled) * 0.55 + turnover_scaled * 0.45

    meta_cols = [c for c in ("game_id", "gameday", "week") if c in original.columns]
    if "gameday" in meta_cols:
        meta = original[meta_cols].drop_duplicates("game_id", keep="last")
        if "week" in quality.columns and "week" in meta.columns:
            meta = meta.drop(columns=["week"])
        quality = quality.merge(meta, on="game_id", how="left")
    else:
        quality["gameday"] = None
    quality["season"] = int(season)
    quality["team"] = quality["team_abbr"].map(TEAM_ABBR_TO_NAME)
    quality["opponent"] = quality["opponent_abbr"].map(TEAM_ABBR_TO_NAME) if "opponent_abbr" in quality.columns else None
    quality["data_source"] = "nflverse play-by-play"
    quality["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    columns = [
        "season", "week", "gameday", "game_id", "team_abbr", "team",
        "opponent_abbr", "opponent", "score_margin", "quality_score",
        "underlying_edge", "net_epa", "success_margin", "yards_per_play_margin",
        "explosive_margin", "turnover_margin", "turnover_luck_index",
        "offensive_plays", "data_source", "updated_at_utc",
    ]
    return quality[[c for c in columns if c in quality.columns]].sort_values(["week", "game_id", "team_abbr"]).reset_index(drop=True)

def _load_pbp_with_retry(nfl, season: int, *, attempts: int = 4, base_delay_seconds: float = 2.0) -> pd.DataFrame:
    """Load nflverse play-by-play with bounded retries for transient network failures."""
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            return _to_pandas(nfl.load_pbp([int(season)]))
        except ValueError:
            # Availability/season errors are deterministic and should be handled by
            # the season-fallback logic rather than retried.
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = base_delay_seconds * (2 ** (attempt - 1))
            print(
                f"nflverse play-by-play download failed for {season} "
                f"(attempt {attempt}/{attempts}): {exc}. Retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
    raise ConnectionError(
        f"nflverse play-by-play download failed for {season} after {attempts} attempts: {last_error}"
    ) from last_error


def fetch_and_build(season: int, output_path: str | Path) -> FetchResult:
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise RuntimeError("Install nflreadpy before refreshing NFL data.") from exc

    pbp = _load_pbp_with_retry(nfl, int(season))
    snapshot = build_team_snapshot(pbp, int(season))
    game_quality = build_game_quality_snapshot(pbp, int(season))
    ftn = pd.DataFrame()
    participation = pd.DataFrame()
    try:
        ftn = _to_pandas(nfl.load_ftn_charting([int(season)]))
    except Exception:
        pass
    try:
        participation = _to_pandas(nfl.load_participation([int(season)]))
    except Exception:
        pass
    scheme = build_scheme_snapshot(pbp, int(season), ftn=ftn, participation=participation)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(output, index=False)
    game_quality.to_csv(output.parent / "game_quality.csv", index=False)
    scheme.to_csv(output.parent / "scheme_tendencies.csv", index=False)
    return FetchResult(
        season=int(season), rows=len(snapshot), output_path=str(output),
        fetched_at_utc=snapshot["updated_at_utc"].iloc[0],
    )
