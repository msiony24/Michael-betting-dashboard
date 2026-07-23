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
    # Keep early-season noise from producing fake 0/100 certainty.
    return (45.0 + ranks.fillna(0.5) * 45.0).clip(45.0, 90.0)


def _weighted_rating(frame: pd.DataFrame, columns: Iterable[tuple[str, float]]) -> pd.Series:
    result = pd.Series(0.0, index=frame.index)
    total_weight = 0.0
    for column, weight in columns:
        if column in frame:
            result += frame[column].fillna(67.5) * weight
            total_weight += weight
    return result / total_weight if total_weight else pd.Series(67.5, index=frame.index)


def build_team_snapshot(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    required = {"posteam", "defteam", "epa"}
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"nflverse play-by-play data is missing required columns: {sorted(missing)}")

    plays = pbp.copy()
    if "season" in plays:
        plays = plays[pd.to_numeric(plays["season"], errors="coerce") == int(season)]
    if "season_type" in plays:
        plays = plays[plays["season_type"].astype(str).eq("REG")]
    if "play_type" in plays:
        plays = plays[plays["play_type"].isin(["pass", "run"])]
    plays = plays[plays["posteam"].notna() & plays["defteam"].notna()]
    plays["epa"] = pd.to_numeric(plays["epa"], errors="coerce")
    plays = plays[plays["epa"].notna()]

    if plays.empty:
        raise ValueError(f"No usable regular-season plays were found for {season}.")

    success = pd.to_numeric(plays.get("success", plays["epa"].gt(0)), errors="coerce").fillna(0)
    yards = pd.to_numeric(plays.get("yards_gained", 0), errors="coerce").fillna(0)
    pass_flag = plays.get("pass_attempt", plays.get("play_type", "").eq("pass"))
    rush_flag = plays.get("rush_attempt", plays.get("play_type", "").eq("run"))
    pass_flag = pd.to_numeric(pass_flag, errors="coerce").fillna(0).astype(bool)
    rush_flag = pd.to_numeric(rush_flag, errors="coerce").fillna(0).astype(bool)
    plays["_success"] = success
    plays["_explosive"] = ((pass_flag & (yards >= 20)) | (rush_flag & (yards >= 10))).astype(float)
    if "turnover" in plays:
        turnover = pd.to_numeric(plays["turnover"], errors="coerce").fillna(0)
    else:
        interception = pd.to_numeric(plays.get("interception", pd.Series(0, index=plays.index)), errors="coerce").fillna(0)
        fumble_lost = pd.to_numeric(plays.get("fumble_lost", pd.Series(0, index=plays.index)), errors="coerce").fillna(0)
        turnover = ((interception > 0) | (fumble_lost > 0)).astype(float)
    plays["_turnover"] = turnover
    plays["_pass"] = pass_flag.astype(float)
    plays["_rush"] = rush_flag.astype(float)

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
    ).rename(columns={"posteam": "team_abbr"})
    if "cpoe" in pass_plays:
        cpoe = pass_plays.groupby("posteam", as_index=False)["cpoe"].mean().rename(
            columns={"posteam": "team_abbr", "cpoe": "qb_cpoe"}
        )
        qb = qb.merge(cpoe, on="team_abbr", how="left")
    else:
        qb["qb_cpoe"] = 0.0

    if "special_teams_play" in pbp and "epa" in pbp:
        st_plays = pbp[pd.to_numeric(pbp["special_teams_play"], errors="coerce").fillna(0).astype(bool)].copy()
        if "posteam" in st_plays:
            special = st_plays.groupby("posteam", as_index=False)["epa"].mean().rename(
                columns={"posteam": "team_abbr", "epa": "special_teams_epa"}
            )
        else:
            special = pd.DataFrame(columns=["team_abbr", "special_teams_epa"])
    else:
        special = pd.DataFrame(columns=["team_abbr", "special_teams_epa"])

    stats = offense.merge(defense, on="team_abbr", how="outer").merge(qb, on="team_abbr", how="left")
    stats = stats.merge(special, on="team_abbr", how="left")

    # Strength of schedule: average opponent net EPA quality faced.
    net = stats.set_index("team_abbr")["offense_epa_per_play"].sub(
        stats.set_index("team_abbr")["defense_epa_allowed"]
    )
    opp_rows = []
    for team, group in plays.groupby("posteam"):
        opponents = group["defteam"].map(net)
        opp_rows.append((team, opponents.mean()))
    sos = pd.DataFrame(opp_rows, columns=["team_abbr", "sos_opponent_net_epa"])
    stats = stats.merge(sos, on="team_abbr", how="left")

    stats["off_epa_score"] = _percentile_score(stats["offense_epa_per_play"])
    stats["off_success_score"] = _percentile_score(stats["offense_success_rate"])
    stats["off_explosive_score"] = _percentile_score(stats["offense_explosive_rate"])
    stats["off_turnover_score"] = _percentile_score(stats["offense_turnover_rate"], higher_is_better=False)
    stats["offense"] = _weighted_rating(stats, [
        ("off_epa_score", .45), ("off_success_score", .30),
        ("off_explosive_score", .15), ("off_turnover_score", .10),
    ])

    stats["def_epa_score"] = _percentile_score(stats["defense_epa_allowed"], higher_is_better=False)
    stats["def_success_score"] = _percentile_score(stats["defense_success_allowed"], higher_is_better=False)
    stats["def_explosive_score"] = _percentile_score(stats["defense_explosive_allowed"], higher_is_better=False)
    stats["def_takeaway_score"] = _percentile_score(stats["defense_takeaway_rate"])
    stats["defense"] = _weighted_rating(stats, [
        ("def_epa_score", .45), ("def_success_score", .30),
        ("def_explosive_score", .15), ("def_takeaway_score", .10),
    ])

    stats["qb_epa_score"] = _percentile_score(stats["qb_epa_per_dropback"])
    stats["qb_success_score"] = _percentile_score(stats["qb_success_rate"])
    stats["qb_cpoe_score"] = _percentile_score(stats["qb_cpoe"])
    stats["quarterback"] = _weighted_rating(stats, [
        ("qb_epa_score", .55), ("qb_success_score", .25), ("qb_cpoe_score", .20),
    ])
    stats["strength_of_schedule"] = _percentile_score(stats["sos_opponent_net_epa"])
    stats["special_teams"] = _percentile_score(stats["special_teams_epa"])

    stats["team"] = stats["team_abbr"].map(TEAM_ABBR_TO_NAME)
    stats = stats[stats["team"].notna()].copy()
    stats["season"] = int(season)
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
