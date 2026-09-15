"""Walk-forward backtest of the Macabets NFL rating engine's performance weighting.

WHAT THIS ANSWERS
    1. Does blending real NFL performance into Madden ratings improve game
       predictions at all (vs. Madden ratings alone)?
    2. Is the v1.5 weighting (efficiency-based, sample-weighted) better than the
       old v1.4 weighting (season totals, ramps to 80%)?
    3. Would different v1.5 settings (player stability, team cap, team ramp)
       have done better?
    4. How do all of these compare with the betting market's closing moneyline?

HOW IT WORKS
    The 2025 season is replayed week by week. Madden 26 "Week 1 Ratings" (set by
    EA before any 2025 game) are the talent baseline. To predict week N, the
    engine only sees nflverse player stats and play-by-play from weeks 1..N-1,
    plus the week-N roster (who is active that week, which team they play for).
    Nothing from week N or later touches a week-N prediction.

    Each team's rating is turned into a projected margin exactly the way the
    production core does it (team-state category weights, 1 rating point = 1
    point, 1.7-point home field, normal margin distribution with SD 13.5).

WHAT IT DOES NOT TEST
    Only the personnel/performance rating core is replayed. Production also adds
    coaching, continuity, recent form, schedule, weather, scheme, line-of-
    scrimmage, situational and opponent-adjusted layers; those are held neutral
    here because historical snapshots of them do not exist. Depth charts and
    Sleeper injury feeds are not available historically either, so units use
    rating-order starters from each week's active roster. Every variant gets the
    same inputs, so the comparison between variants is fair even though the
    absolute numbers will not match the full production model.

HOW TO RUN (GitHub Actions: "NFL Rating Backtest" workflow, or locally with
network access):
    python audit/nfl_rating_backtest.py
    python audit/nfl_rating_backtest.py --quick      # skip the settings grid

If EA no longer serves the Madden 26 Week 1 iteration, the script falls back to
the stored Madden 26 Week 8 file and only scores weeks 9-18 (those ratings were
public before week 9). That fallback is labeled LIMITED in the report because
EA's Week 8 update already reflects weeks 1-7 of real play.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import time
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import nfl_rating_engine as current_engine  # noqa: E402
from engine.nfl_fetch import build_team_snapshot  # noqa: E402
from engine.nfl_player_weekly import KEEP_COLUMNS as PLAYER_STAT_COLUMNS  # noqa: E402
from engine.nfl_rating_engine import _name_key  # noqa: E402
from engine.nfl_ratings_loader import _madden_category_ratings  # noqa: E402
from engine.nfl_team_state import TEAM_STATE_WEIGHTS  # noqa: E402

DEFAULT_SEASON = 2025
DEFAULT_OUTPUT_DIR = ROOT / "audit" / "results_nfl_rating_backtest"
STORED_MADDEN_PATH = ROOT / "data" / "madden_26_players.csv"
LEGACY_ENGINE_PATH = ROOT / "audit" / "legacy_nfl_rating_engine_v14.py"
LAUNCH_ITERATION = "2-week-1"
STORED_ITERATION_FIRST_SAFE_WEEK = 9  # stored file is EA "Week 8 Ratings"

# Mirrors engine/nfl.py (analyze default home_field_points and NFL_MARGIN_SD).
# Hard-coded so this script does not import the full app-facing NFL module.
HOME_FIELD_POINTS = 1.7
NFL_MARGIN_SD = 13.5
MARGIN_DIST = NormalDist(0.0, NFL_MARGIN_SD)
NEUTRAL_COMPONENTS = {"coaching": 70.0, "continuity": 67.5, "recent_form": 67.5}

TEAM_FIX = {"AZ": "ARI", "LAR": "LA", "STL": "LA", "OAK": "LV", "SD": "LAC", "WSH": "WAS", "JAC": "JAX"}
ACTIVE_STATUSES = {"ACT"}
COARSE_POSITION = {
    "QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "OL", "OT": "OL", "G": "OL", "OG": "OL", "C": "OL", "OL": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "OLB": "LB", "ILB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "SAF": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
    "K": "K", "P": "P", "LS": "LS",
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

@dataclass
class SeasonInputs:
    season: int
    schedules: pd.DataFrame
    player_stats: pd.DataFrame
    weekly_rosters: pd.DataFrame
    pbp: pd.DataFrame


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def _fix_team(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip().replace(TEAM_FIX)


def _regular_season(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    out = frame.copy()
    if "season" in out.columns:
        out = out[pd.to_numeric(out["season"], errors="coerce").eq(season)]
    for column in ("season_type", "game_type"):
        if column in out.columns:
            out = out[out[column].astype(str).str.upper().eq("REG")]
    return out.reset_index(drop=True)


def load_season_inputs(season: int) -> SeasonInputs:
    import nflreadpy as nfl

    print(f"Downloading nflverse {season} data...")
    schedules = _regular_season(_to_pandas(nfl.load_schedules([season])), season)
    player_stats = _regular_season(_to_pandas(nfl.load_player_stats([season], summary_level="week")), season)
    weekly_rosters = _to_pandas(nfl.load_rosters_weekly([season]))
    weekly_rosters = weekly_rosters[pd.to_numeric(weekly_rosters["season"], errors="coerce").eq(season)].reset_index(drop=True)
    pbp = _regular_season(_to_pandas(nfl.load_pbp([season])), season)
    print(f"  schedules={len(schedules)} player_stats={len(player_stats)} rosters={len(weekly_rosters)} pbp={len(pbp)}")
    return SeasonInputs(season, schedules, player_stats, weekly_rosters, pbp)


def fetch_madden_iteration(iteration: str, stored_path: Path = STORED_MADDEN_PATH) -> pd.DataFrame:
    """Download one EA ratings iteration and verify it is the same Madden edition as the stored file."""
    from engine.madden_ratings_loader import (
        EA_API_BASE, PAGE_SIZE, _find_player_list, _find_total, _request_json, normalize_players,
    )

    records: list[dict[str, Any]] = []
    total = None
    for page in range(60):
        query = urllib.parse.urlencode({"limit": PAGE_SIZE, "offset": page * PAGE_SIZE, "iteration": iteration})
        payload = _request_json(f"{EA_API_BASE}?{query}")
        items = _find_player_list(payload)
        if page == 0:
            total = _find_total(payload)
        if not items:
            break
        records.extend(items)
        if (total is not None and len(records) >= total) or len(items) < PAGE_SIZE:
            break
        time.sleep(0.25)
    if not records:
        raise RuntimeError(f"EA returned no players for iteration {iteration}.")

    served = {str((r.get("iteration") or {}).get("id", "")) for r in records}
    if served != {iteration}:
        raise RuntimeError(f"EA ignored the iteration request (served {sorted(served)[:3]}).")

    players = normalize_players(records)
    stored = pd.read_csv(stored_path)
    if "id" in stored.columns and "yearsPro" in stored.columns:
        overlap = players.merge(
            stored[["id", "yearsPro", "position"]].rename(columns={"id": "ea_player_id", "yearsPro": "_stored_years"}),
            on="ea_player_id", how="inner",
        )
        if len(overlap) < 500:
            raise RuntimeError(f"Iteration {iteration} shares only {len(overlap)} players with the stored Madden 26 file.")
        same_edition = (pd.to_numeric(overlap["years_pro"], errors="coerce") == pd.to_numeric(overlap["_stored_years"], errors="coerce")).mean()
        if same_edition < 0.90:
            raise RuntimeError(
                f"Iteration {iteration} does not look like Madden 26 (only {same_edition:.0%} of years-pro values match); "
                "EA may now be serving a different edition."
            )
        by_id = stored.drop_duplicates("id").set_index("id")
        players["position"] = players["ea_player_id"].map(by_id["position"])
        # Stored team is only a tie-breaker for same-name players; weekly rosters decide teams.
        players["team"] = players["ea_player_id"].map(by_id["team"]) if "team" in by_id.columns else ""
    else:
        players["position"] = np.nan
    players["iteration"] = iteration
    print(f"  Madden iteration {iteration}: {len(players)} players verified as Madden 26")
    return players


def load_stored_madden(stored_path: Path = STORED_MADDEN_PATH) -> pd.DataFrame:
    stored = pd.read_csv(stored_path)
    out = stored.rename(columns={"id": "ea_player_id", "yearsPro": "years_pro"}).copy()
    out["iteration"] = str(stored.get("iteration_id", pd.Series(["stored"])).iloc[0])
    return out


# --------------------------------------------------------------------------- #
# Weekly engine inputs
# --------------------------------------------------------------------------- #

def madden_for_week(madden: pd.DataFrame, roster_week: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assign each Madden player to the team he is ACTIVE for this week (or drop him)."""
    roster = roster_week.copy()
    name_col = "full_name" if "full_name" in roster.columns else "player_name"
    roster["name_key"] = roster[name_col].map(_name_key)
    roster["team_abbr"] = _fix_team(roster["team"])
    roster["status"] = roster.get("status", pd.Series("ACT", index=roster.index)).astype(str).str.upper()
    roster["coarse_position"] = roster.get("position", pd.Series("", index=roster.index)).astype(str).str.upper().map(COARSE_POSITION)
    roster = roster[roster["name_key"].ne("")]

    counts = roster.groupby("name_key")["team_abbr"].transform("size")
    unique_roster = roster[counts.eq(1)].set_index("name_key")
    shared_roster = roster[counts.gt(1)]

    work = madden.copy()
    work["name_key"] = work["player_name"].map(_name_key)
    work["_stored_team"] = _fix_team(work["team"].fillna("")) if "team" in work.columns else ""
    work = work[work["name_key"].ne("")]
    work = work[~work.duplicated("name_key", keep=False) | work["_stored_team"].ne("")]

    rows = []
    for _, player in work.iterrows():
        key = player["name_key"]
        if key in unique_roster.index:
            match = unique_roster.loc[key]
        else:
            candidates = shared_roster[shared_roster["name_key"].eq(key) & shared_roster["team_abbr"].eq(player["_stored_team"])]
            if len(candidates) != 1:
                continue
            match = candidates.iloc[0]
        if match["status"] not in ACTIVE_STATUSES:
            continue
        row = player.to_dict()
        row["team"] = match["team_abbr"]
        if not isinstance(row.get("position"), str) or not row["position"].strip() or row["position"] == "nan":
            row["position"] = match["coarse_position"]
        row["gsis_id"] = match.get("gsis_id", "")
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[out["position"].notna()].drop(columns=["name_key", "_stored_team"], errors="ignore")
    stats = {"madden_players": int(len(madden)), "active_matched": int(len(out)),
             "active_roster": int(roster["status"].isin(ACTIVE_STATUSES).sum())}
    return out, stats


def prepare_week(inputs: SeasonInputs, madden: pd.DataFrame, week: int, root: Path) -> dict[str, Any]:
    """Write the files the engine reads for a week-N prediction, using only pre-week-N data."""
    root.mkdir(parents=True, exist_ok=True)
    season = inputs.season

    prior_stats = inputs.player_stats[pd.to_numeric(inputs.player_stats["week"], errors="coerce") < week].copy()
    assert prior_stats.empty or prior_stats["week"].max() < week, "player-stat leakage"
    keep = [c for c in PLAYER_STAT_COLUMNS if c in prior_stats.columns]
    prior_stats = prior_stats[keep].copy()
    if "team" in prior_stats.columns:
        prior_stats["team"] = _fix_team(prior_stats["team"])
    prior_stats.to_csv(root / "player_weekly_stats_base.csv", index=False)

    roster_week = inputs.weekly_rosters[pd.to_numeric(inputs.weekly_rosters["week"], errors="coerce").eq(week)].copy()
    roster_week["team"] = _fix_team(roster_week["team"])
    roster_week.to_csv(root / "weekly_rosters.csv", index=False)

    week_madden, match_stats = madden_for_week(madden, roster_week)
    week_madden.to_csv(root / "madden.csv", index=False)

    prior_pbp = inputs.pbp[pd.to_numeric(inputs.pbp["week"], errors="coerce") < week]
    snapshot_rows = 0
    if not prior_pbp.empty:
        assert prior_pbp["week"].max() < week, "play-by-play leakage"
        snapshot = build_team_snapshot(prior_pbp, season)
        snapshot["team_abbr"] = _fix_team(snapshot["team_abbr"])
        snapshot.to_csv(root / "team_snapshot.csv", index=False)
        snapshot_rows = len(snapshot)
    return {"week": week, "prior_stat_rows": int(len(prior_stats)), "snapshot_teams": snapshot_rows, **match_stats}


def _stats_with_cap(root: Path, cap: float) -> Path:
    target = root / f"cap_{cap:.2f}"
    if (target / "player_weekly_stats.csv").exists():
        return target
    target.mkdir(exist_ok=True)
    stats = pd.read_csv(root / "player_weekly_stats_base.csv")
    stats["macabets_performance_cap"] = cap
    stats.to_csv(target / "player_weekly_stats.csv", index=False)
    shutil.copy(root / "weekly_rosters.csv", target / "weekly_rosters.csv")
    return target


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Variant:
    name: str
    engine: str = "current"          # "current" or "legacy"
    model: str = "v1.5"              # current engine's RATING_MODEL for this variant
    player_cap: float = 0.80
    player_stability_mult: float = 1.0
    team_cap: float = current_engine.TEAM_PERFORMANCE_CAP
    team_stability: float = current_engine.TEAM_PERFORMANCE_STABILITY
    group: str = "main"


def default_variants(include_grid: bool = True) -> list[Variant]:
    variants = [
        Variant("madden_only", player_cap=0.0, team_cap=0.0),
        Variant("v1_4_legacy", engine="legacy"),
        Variant("current_engine", model=current_engine.RATING_MODEL),
        Variant("v1_5_efficiency"),
    ]
    if include_grid:
        for mult in (0.5, 1.0, 2.0):
            for cap in (0.3, 0.6, 0.8):
                for k in (3.0, 6.0, 12.0):
                    variants.append(Variant(
                        f"grid_pm{mult:g}_tc{cap:g}_tk{k:g}", player_stability_mult=mult,
                        team_cap=cap, team_stability=k, group="grid",
                    ))
    return variants


def _load_legacy_engine():
    spec = importlib.util.spec_from_file_location("legacy_nfl_rating_engine_v14", LEGACY_ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@contextlib.contextmanager
def _current_settings(variant: Variant) -> Iterator[None]:
    saved = (dict(current_engine.PERFORMANCE_STABILITY), current_engine.TEAM_PERFORMANCE_CAP,
             current_engine.TEAM_PERFORMANCE_STABILITY, current_engine.RATING_MODEL)
    current_engine.RATING_MODEL = variant.model
    current_engine.PERFORMANCE_STABILITY = {k: v * variant.player_stability_mult for k, v in saved[0].items()}
    current_engine.TEAM_PERFORMANCE_CAP = variant.team_cap
    current_engine.TEAM_PERFORMANCE_STABILITY = variant.team_stability
    try:
        yield
    finally:
        (current_engine.PERFORMANCE_STABILITY, current_engine.TEAM_PERFORMANCE_CAP,
         current_engine.TEAM_PERFORMANCE_STABILITY, current_engine.RATING_MODEL) = saved


@contextlib.contextmanager
def _frozen_year(module: Any, season: int) -> Iterator[None]:
    original = module.datetime

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401
            return datetime(season, 12, 31, tzinfo=tz)

    module.datetime = _Frozen
    try:
        yield
    finally:
        module.datetime = original


def team_power(team: dict[str, Any]) -> float:
    components = _madden_category_ratings({}, team)
    components.update(NEUTRAL_COMPONENTS)
    raw = sum(float(components[name]) * weight for name, weight in TEAM_STATE_WEIGHTS.items())
    return raw - 67.5


class WeekRater:
    """Builds team power for every variant, caching player ratings across team settings."""

    def __init__(self, season: int, legacy: Any):
        self.season = season
        self.legacy = legacy
        self._player_cache: dict[tuple, pd.DataFrame] = {}
        self.no_depth_chart = Path(tempfile.gettempdir()) / "macabets_no_depth_chart.csv"

    def powers(self, variant: Variant, week_root: Path) -> dict[str, float]:
        stats_dir = _stats_with_cap(week_root, variant.player_cap)
        snapshot = week_root / "team_snapshot.csv"
        madden = week_root / "madden.csv"
        if variant.engine == "legacy":
            key = (str(week_root), "legacy", variant.player_cap)
            with _frozen_year(self.legacy, self.season):
                if key not in self._player_cache:
                    self._player_cache[key] = self.legacy.build_player_ratings(madden, stats_dir, depth_chart_path=self.no_depth_chart)
                teams = self.legacy.build_team_ratings(self._player_cache[key], snapshot, self.no_depth_chart)
        else:
            key = (str(week_root), "current", variant.model, variant.player_cap, variant.player_stability_mult)
            with _current_settings(variant):
                if key not in self._player_cache:
                    self._player_cache[key] = current_engine.build_player_ratings(madden, stats_dir, depth_chart_path=self.no_depth_chart)
                teams = current_engine.build_team_ratings(
                    self._player_cache[key], snapshot, self.no_depth_chart, current_season=self.season,
                )
        return {str(t["team_abbr"]): team_power(t) for t in teams.values()}

    def drop_week(self, week_root: Path) -> None:
        for key in [k for k in self._player_cache if k[0] == str(week_root)]:
            del self._player_cache[key]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _american_to_prob(line: Any) -> float:
    value = pd.to_numeric(line, errors="coerce")
    if pd.isna(value) or value == 0:
        return float("nan")
    return float(-value / (-value + 100.0)) if value < 0 else float(100.0 / (value + 100.0))


def game_table(schedules: pd.DataFrame, weeks: list[int]) -> pd.DataFrame:
    games = schedules.copy()
    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    games = games[games["week"].isin(weeks)].copy()
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    games = games.dropna(subset=["home_score", "away_score"])
    games = games[games["home_score"].ne(games["away_score"])].copy()  # ties are not scoreable win/loss
    games["home_team"] = _fix_team(games["home_team"])
    games["away_team"] = _fix_team(games["away_team"])
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games["neutral"] = games.get("location", pd.Series("", index=games.index)).astype(str).str.lower().eq("neutral")
    home_raw = games.get("home_moneyline", pd.Series(np.nan, index=games.index)).map(_american_to_prob)
    away_raw = games.get("away_moneyline", pd.Series(np.nan, index=games.index)).map(_american_to_prob)
    games["market_home_prob"] = home_raw / (home_raw + away_raw)
    keep = ["game_id", "week", "home_team", "away_team", "home_score", "away_score", "home_win", "neutral", "market_home_prob"]
    return games[keep].reset_index(drop=True)


def predict(games: pd.DataFrame, powers: dict[str, float]) -> pd.Series:
    margins = []
    for _, game in games.iterrows():
        home, away = powers.get(game["home_team"]), powers.get(game["away_team"])
        if home is None or away is None:
            margins.append(np.nan)
            continue
        hfa = 0.0 if bool(game["neutral"]) else HOME_FIELD_POINTS
        margins.append(home - away + hfa)
    return pd.Series(margins, index=games.index, dtype=float)


def _probs(margins: pd.Series, scale: float = 1.0) -> np.ndarray:
    return np.array([MARGIN_DIST.cdf(float(m) * scale) for m in margins], dtype=float)


def _log_losses(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "games": int(len(y)),
        "log_loss": round(float(_log_losses(p, y).mean()), 5),
        "brier": round(float(np.mean((p - y) ** 2)), 5),
        "accuracy": round(float(np.mean((p > 0.5) == (y == 1))), 4),
        "mean_pick_confidence": round(float(np.mean(np.maximum(p, 1 - p))), 4),
    }


def best_scale(margins: pd.Series, y: np.ndarray) -> tuple[float, float]:
    """In-sample: the margin multiplier that minimizes log loss (1.0 = production scale)."""
    best = (1.0, float("inf"))
    for scale in np.arange(0.25, 4.001, 0.05):
        loss = float(_log_losses(_probs(margins, scale), y).mean())
        if loss < best[1]:
            best = (round(float(scale), 2), round(loss, 5))
    return best


def bootstrap_difference(a: np.ndarray, b: np.ndarray, y: np.ndarray, draws: int = 2000, seed: int = 7) -> dict[str, float]:
    """Log-loss(a) - log-loss(b) with a game-resampled 95% interval. Negative = a is better."""
    diff = _log_losses(a, y) - _log_losses(b, y)
    rng = np.random.default_rng(seed)
    samples = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(draws)])
    return {
        "mean_difference": round(float(diff.mean()), 5),
        "ci_low": round(float(np.percentile(samples, 2.5)), 5),
        "ci_high": round(float(np.percentile(samples, 97.5)), 5),
        "share_of_resamples_better": round(float((samples < 0).mean()), 3),
    }


def calibration_table(p: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    fav_p = np.maximum(p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    rows = []
    for low, high in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
        mask = (fav_p >= low) & (fav_p < high)
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket": f"{int(low*100)}-{min(int(high*100), 100)}%", "games": int(mask.sum()),
            "predicted": round(float(fav_p[mask].mean()), 3), "actual": round(float(fav_won[mask].mean()), 3),
        })
    return rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_backtest(
    inputs: SeasonInputs,
    madden: pd.DataFrame,
    weeks: list[int],
    variants: list[Variant],
    *,
    work_dir: Path,
    legacy: Any = None,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    legacy = legacy if legacy is not None else _load_legacy_engine()
    rater = WeekRater(inputs.season, legacy)
    games = game_table(inputs.schedules, weeks)
    predictions = games.copy()
    week_log: list[dict[str, Any]] = []
    for name in (v.name for v in variants):
        predictions[f"margin__{name}"] = np.nan

    for week in weeks:
        week_games = games.index[games["week"].eq(week)]
        if len(week_games) == 0:
            continue
        root = work_dir / f"week_{week:02d}"
        info = prepare_week(inputs, madden, week, root)
        week_log.append(info)
        log(f"Week {week}: {len(week_games)} games | active Madden players {info['active_matched']} "
            f"| prior stat rows {info['prior_stat_rows']} | snapshot teams {info['snapshot_teams']}")
        for variant in variants:
            powers = rater.powers(variant, root)
            predictions.loc[week_games, f"margin__{variant.name}"] = predict(games.loc[week_games], powers)
        rater.drop_week(root)
        shutil.rmtree(root, ignore_errors=True)
    return predictions, week_log


def summarize(predictions: pd.DataFrame, variants: list[Variant], reference: str = "current_engine") -> dict[str, Any]:
    margin_cols = [f"margin__{v.name}" for v in variants]
    scored = predictions.dropna(subset=margin_cols).copy()
    y = scored["home_win"].to_numpy(dtype=float)
    probs = {v.name: _probs(scored[f"margin__{v.name}"]) for v in variants}

    rows = []
    for v in variants:
        scale, scaled_loss = best_scale(scored[f"margin__{v.name}"], y)
        early = scored["week"] <= scored["week"].median()
        rows.append({
            "variant": v.name, "group": v.group, **metrics(probs[v.name], y),
            "log_loss_early_weeks": round(float(_log_losses(probs[v.name][early.to_numpy()], y[early.to_numpy()]).mean()), 5),
            "log_loss_late_weeks": round(float(_log_losses(probs[v.name][~early.to_numpy()], y[~early.to_numpy()]).mean()), 5),
            "best_margin_scale": scale, "log_loss_at_best_scale": scaled_loss,
            **{f"setting_{k}": val for k, val in asdict(v).items() if k not in {"name", "group"}},
        })
    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)

    comparisons = {}
    if reference in probs:
        for other in ("madden_only", "v1_4_legacy", "v1_5_efficiency"):
            if other in probs:
                comparisons[f"{reference}_vs_{other}"] = bootstrap_difference(probs[reference], probs[other], y)
        grid = table[table["group"].eq("grid")]
        if not grid.empty:
            best_grid = grid.iloc[0]["variant"]
            comparisons[f"best_grid_vs_{reference}"] = {"best_grid": best_grid, **bootstrap_difference(probs[best_grid], probs[reference], y)}
            if "v1_4_legacy" in probs:
                comparisons["best_grid_vs_v1_4_legacy"] = {"best_grid": best_grid, **bootstrap_difference(probs[best_grid], probs["v1_4_legacy"], y)}

    market = {}
    has_market = scored["market_home_prob"].notna().to_numpy()
    if has_market.sum() > 0:
        m_y = y[has_market]
        market = {"market_closing_moneyline": metrics(scored["market_home_prob"].to_numpy()[has_market], m_y)}
        for name in ("madden_only", "v1_4_legacy", "v1_5_efficiency", reference):
            if name in probs:
                market[f"{name}_same_games"] = metrics(probs[name][has_market], m_y)

    return {
        "table": table,
        "comparisons": comparisons,
        "market": market,
        "calibration": {name: calibration_table(probs[name], y) for name in ("madden_only", "v1_4_legacy", "v1_5_efficiency", reference) if name in probs},
        "games_scored": int(len(scored)),
        "weeks_scored": sorted(int(w) for w in scored["week"].unique()),
    }


def _fmt_diff(entry: dict[str, float]) -> str:
    return (f"{entry['mean_difference']:+.4f} (95% range {entry['ci_low']:+.4f} to {entry['ci_high']:+.4f}; "
            f"better in {entry['share_of_resamples_better']:.0%} of resamples)")


def write_report(summary: dict[str, Any], meta: dict[str, Any], output_dir: Path) -> Path:
    table: pd.DataFrame = summary["table"]
    main = table[table["group"].eq("main")]
    lines = [
        "# Macabets NFL Rating Backtest",
        "",
        f"Season replayed: **{meta['season']}** | Madden baseline: **{meta['madden_iteration']}** "
        f"| Weeks scored: {summary['weeks_scored'][0]}-{summary['weeks_scored'][-1]} | Games: {summary['games_scored']}",
        "",
    ]
    if meta.get("limited"):
        lines += [
            "> **LIMITED RUN.** EA's Madden 26 Week 1 ratings could not be downloaded, so the stored",
            f"> Week 8 ratings were used and only weeks {STORED_ITERATION_FIRST_SAFE_WEEK}+ were scored. Those ratings already reflect",
            "> weeks 1-7 of real play, so this under-states the value of performance blending.",
            f"> Reason: {meta.get('limited_reason', '')}",
            "",
        ]
    lines += [
        f"`current_engine` = the engine's live setting ({current_engine.RATING_MODEL}); the settings grid varies v1.5.",
        "",
        "Lower log loss and Brier = better probabilities. Differences of ~0.005 in log loss are small;",
        "the 95% ranges below show whether a difference is distinguishable from luck.",
        "",
        "## Main comparison",
        "",
        "| Version | Log loss | Brier | Accuracy | Early weeks LL | Late weeks LL |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in main.iterrows():
        lines.append(f"| {r['variant']} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['accuracy']:.1%} "
                     f"| {r['log_loss_early_weeks']:.4f} | {r['log_loss_late_weeks']:.4f} |")
    lines += ["", "## Head-to-head (negative = first version is better)", ""]
    for key, entry in summary["comparisons"].items():
        label = key.replace("_vs_", " vs ").replace("best_grid", f"best grid ({entry.get('best_grid', '')})")
        lines.append(f"- **{label}:** {_fmt_diff(entry)}")
    if summary["market"]:
        lines += ["", "## Versus the betting market (same games)", "",
                  "| Source | Log loss | Brier | Accuracy |", "|---|---|---|---|"]
        for name, m in summary["market"].items():
            lines.append(f"| {name} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['accuracy']:.1%} |")
    lines += ["", "## Calibration (favorite's predicted vs actual win rate)", ""]
    for name, rows in summary["calibration"].items():
        cells = ", ".join(f"{r['bucket']}: {r['predicted']:.0%} pred / {r['actual']:.0%} actual (n={r['games']})" for r in rows)
        lines.append(f"- **{name}:** {cells}")
    grid = table[table["group"].eq("grid")].head(8)
    if not grid.empty:
        lines += ["", "## Best settings from the grid (in-sample; treat as a hint, not proof)", "",
                  "| Variant | Log loss | Early LL | Late LL | Player stability x | Team cap | Team ramp K |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in grid.iterrows():
            lines.append(f"| {r['variant']} | {r['log_loss']:.4f} | {r['log_loss_early_weeks']:.4f} | {r['log_loss_late_weeks']:.4f} "
                         f"| {r['setting_player_stability_mult']:g} | {r['setting_team_cap']:g} | {r['setting_team_stability']:g} |")
    lines += [
        "", "## Scale check",
        "",
        "`best_margin_scale` in summary.csv is the in-sample multiplier on projected margins that would",
        "minimize log loss (1.0 = production). Well above 1 means projections are too timid; below 1 too bold.",
        "",
        "## What this does and does not test",
        "",
        "Only the personnel/performance rating core is replayed. Coaching, continuity, recent form,",
        "schedule, weather, scheme and matchup layers are held neutral; depth charts and Sleeper injuries",
        "are replaced by each week's active roster. All versions get identical inputs.",
        "",
        f"Generated {meta['generated_at_utc']}.",
    ]
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--iteration", default=LAUNCH_ITERATION)
    parser.add_argument("--first-week", type=int, default=2)
    parser.add_argument("--last-week", type=int, default=18)
    parser.add_argument("--quick", action="store_true", help="skip the settings grid")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"season": args.season, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    first_week = args.first_week
    try:
        madden = fetch_madden_iteration(args.iteration)
        meta["madden_iteration"] = f"Madden 26 {args.iteration} (downloaded)"
    except Exception as exc:  # fall back to the stored file, honestly labeled
        print(f"Could not use EA iteration {args.iteration}: {exc}")
        madden = load_stored_madden()
        meta.update({"madden_iteration": f"Madden 26 stored file ({madden['iteration'].iloc[0]})",
                     "limited": True, "limited_reason": str(exc)[:300]})
        first_week = max(first_week, STORED_ITERATION_FIRST_SAFE_WEEK)

    inputs = load_season_inputs(args.season)
    weeks = list(range(first_week, args.last_week + 1))
    variants = default_variants(include_grid=not args.quick)
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        predictions, week_log = run_backtest(inputs, madden, weeks, variants, work_dir=Path(tmp))
    print(f"Replayed {len(weeks)} weeks x {len(variants)} versions in {time.time() - started:.0f}s")

    summary = summarize(predictions, variants)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary["table"].to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(week_log).to_csv(output_dir / "weekly_inputs.csv", index=False)
    payload = {k: v for k, v in summary.items() if k != "table"}
    payload["meta"] = meta
    payload["main_table"] = summary["table"][summary["table"]["group"].eq("main")].to_dict("records")
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report = write_report(summary, meta, output_dir)
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
