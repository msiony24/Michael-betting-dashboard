"""Automated NFL data foundation for Macabets.

The updater downloads a small set of nflverse datasets, writes normalized CSV
snapshots atomically, and records one metadata manifest. The Streamlit app only
reads the compact team snapshot; the larger files remain available to the brain
for future player, roster, injury, and matchup calculations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from engine.nfl_fetch import FetchResult, build_scheme_snapshot, fetch_and_build


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "nfl"
MANIFEST_NAME = "foundation_status.json"


PERFORMANCE_REQUIRED_COLUMNS = {
    # Core situational-execution fields added to the compact team snapshot.
    "third_down_conversion_rate",
    "third_down_conversion_allowed",
    "red_zone_td_rate",
    "red_zone_td_rate_allowed",
    "high_leverage_epa",
    "high_leverage_epa_allowed",
    # Real-performance LOS fields that should travel with the same snapshot.
    "qb_epa_when_disrupted",
    "defense_disruption_rate",
    "offense_run_stuff_rate",
    "defense_run_stuff_rate",
    # Opponent-adjusted performance fields.
    "sos_opponent_offense_epa",
    "sos_opponent_defense_epa_allowed",
    "opponent_quality_epa",
    "opponent_adjusted_net_epa",
}


def _performance_snapshot_has_current_schema(path: Path) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, sorted(PERFORMANCE_REQUIRED_COLUMNS)
    try:
        columns = set(pd.read_csv(path, nrows=1).columns)
    except Exception:
        return False, sorted(PERFORMANCE_REQUIRED_COLUMNS)
    missing = sorted(PERFORMANCE_REQUIRED_COLUMNS.difference(columns))
    return not missing, missing


def _ensure_current_performance_schema(performance: FetchResult, output_path: Path) -> FetchResult:
    """Force one rebuild when an old cached team snapshot predates new metrics.

    A network fallback is allowed to keep Macabets online, but a successful
    refresh must not silently preserve an older CSV schema after new model
    fields are introduced. This mirrors the scheme-snapshot schema guard.
    """
    current, missing = _performance_snapshot_has_current_schema(output_path)
    if current:
        return performance
    print(
        "NFL team snapshot: rebuilding stale schema; missing current columns: "
        + ", ".join(missing)
    )
    try:
        rebuilt = fetch_and_build(int(performance.season), output_path)
    except Exception as exc:
        print(
            "NFL team snapshot: forced schema rebuild could not complete; "
            f"keeping last-good snapshot for core analysis: {exc}"
        )
        return performance
    current, missing = _performance_snapshot_has_current_schema(output_path)
    if not current:
        print(
            "NFL team snapshot: rebuild completed but required columns are still missing: "
            + ", ".join(missing)
        )
    return rebuilt


@dataclass(frozen=True)
class DatasetStatus:
    name: str
    file: str
    rows: int
    available: bool
    error: str = ""


@dataclass(frozen=True)
class FoundationResult:
    requested_season: int
    performance_season: int
    updated_at_utc: str
    data_dir: str
    datasets: tuple[DatasetStatus, ...]

    @property
    def available_count(self) -> int:
        return sum(item.available for item in self.datasets)


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _season_filter(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    if frame.empty or "season" not in frame.columns:
        return frame
    numeric = pd.to_numeric(frame["season"], errors="coerce")
    return frame[numeric.eq(int(season))].copy()


def _latest_week_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest weekly roster/depth-chart row for each player and team."""
    if frame.empty or "week" not in frame.columns:
        return frame
    identity = next((c for c in ("gsis_id", "player_id", "pfr_id", "full_name") if c in frame.columns), None)
    team = next((c for c in ("team", "club_code", "recent_team") if c in frame.columns), None)
    if not identity:
        return frame
    keys = [identity] + ([team] if team else [])
    working = frame.copy()
    working["_week_sort"] = pd.to_numeric(working["week"], errors="coerce").fillna(-1)
    working = working.sort_values("_week_sort").drop_duplicates(keys, keep="last")
    return working.drop(columns=["_week_sort"])


def _safe_dataset(
    *,
    name: str,
    filename: str,
    loader: Callable[[], Any],
    data_dir: Path,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> DatasetStatus:
    path = data_dir / filename
    try:
        frame = _to_pandas(loader())
        if transform is not None:
            frame = transform(frame)
        _atomic_csv(frame, path)
        return DatasetStatus(name=name, file=str(path), rows=len(frame), available=True)
    except Exception as exc:  # one optional dataset must not destroy the full refresh
        return DatasetStatus(name=name, file=str(path), rows=0, available=False, error=str(exc))


def _cached_performance_result(output_path: Path, error: Exception) -> FetchResult | None:
    """Return the last-good team snapshot when nflverse is temporarily unavailable."""
    if not output_path.exists():
        return None
    try:
        frame = pd.read_csv(output_path)
    except Exception:
        return None
    if frame.empty:
        return None
    season = pd.to_numeric(frame.get("season"), errors="coerce").dropna()
    if season.empty:
        return None
    fetched_at = "cached"
    if "updated_at_utc" in frame.columns:
        values = frame["updated_at_utc"].dropna().astype(str)
        if not values.empty:
            fetched_at = values.iloc[-1]
    warning = f"Using last-good cached NFL performance snapshot because refresh failed: {error}"
    print(warning)
    return FetchResult(
        season=int(season.max()),
        rows=len(frame),
        output_path=str(output_path),
        fetched_at_utc=fetched_at,
        source_mode="cached",
        warning=warning,
    )


def _fetch_performance_with_fallback(
    requested_season: int,
    output_path: Path,
    *,
    minimum_season: int = 1999,
) -> FetchResult:
    last_error: Exception | None = None
    for season in range(int(requested_season), minimum_season - 1, -1):
        try:
            return fetch_and_build(season, output_path)
        except ValueError as exc:
            last_error = exc
            message = str(exc).lower()
            if "season must be between" in message or "no usable regular-season plays" in message:
                continue
            raise
        except (ConnectionError, OSError) as exc:
            last_error = exc
            cached = _cached_performance_result(output_path, exc)
            if cached is not None:
                return cached
            raise
    cached = _cached_performance_result(output_path, last_error or RuntimeError("No supported season"))
    if cached is not None:
        return cached
    raise RuntimeError("No supported NFL performance season could be loaded.") from last_error



def _ensure_scheme_snapshot(
    *,
    source_season: int,
    scheme_path: Path,
    nfl_module: Any,
) -> tuple[int, str]:
    """Ensure a usable regular-season scheme snapshot exists.

    During preseason, current-season regular-season PBP does not exist yet.
    Macabets should therefore use the same latest completed regular season that
    powers the team-performance fallback (normally the prior season) as a
    reduced scheme prior instead of showing scheme as unavailable.
    """
    try:
        if scheme_path.exists():
            existing = pd.read_csv(scheme_path)
            if not existing.empty and "team" in existing.columns:
                seasons = pd.to_numeric(existing.get("season"), errors="coerce").dropna()
                # A complete prior/current snapshot should cover essentially the
                # whole league. Also require the current scheme schema. This matters
                # when a metric definition changes (for example, the 2026-08 red-zone
                # update from play-level success to drive-level TD rate): an older
                # full-league CSV must be rebuilt rather than silently reused.
                required_scheme_columns = {
                    "red_zone_td_rate",
                    "red_zone_td_rate_allowed",
                }
                has_current_schema = required_scheme_columns.issubset(existing.columns)
                existing_season = int(seasons.max()) if not seasons.empty else None
                season_matches_source = existing_season == int(source_season)
                if (
                    not seasons.empty
                    and existing["team"].nunique() >= 30
                    and has_current_schema
                    and season_matches_source
                ):
                    return len(existing), ""
                if has_current_schema and not season_matches_source:
                    print(
                        "Scheme tendencies: rebuilding snapshot for current regular "
                        f"season {source_season}; existing snapshot is season "
                        f"{existing_season}. Prior-season scheme is discarded as soon "
                        "as current-season regular-season data is available."
                    )
                if not has_current_schema:
                    missing = sorted(required_scheme_columns.difference(existing.columns))
                    print(
                        "Scheme tendencies: rebuilding stale snapshot; missing current "
                        f"columns: {', '.join(missing)}"
                    )
    except Exception:
        pass

    try:
        pbp = _to_pandas(nfl_module.load_pbp([int(source_season)]))
        ftn = pd.DataFrame()
        participation = pd.DataFrame()
        try:
            ftn = _to_pandas(nfl_module.load_ftn_charting([int(source_season)]))
        except Exception:
            pass
        try:
            participation = _to_pandas(nfl_module.load_participation([int(source_season)]))
        except Exception:
            pass
        scheme = build_scheme_snapshot(
            pbp, int(source_season), ftn=ftn, participation=participation
        )
        if scheme.empty:
            return 0, f"no usable regular-season scheme data for {source_season}"
        _atomic_csv(scheme, scheme_path)
        print(
            f"Scheme tendencies: using {source_season} regular-season data. "
            "When this is the requested current season, no prior-season scheme "
            "data is blended into the snapshot."
        )
        return len(scheme), ""
    except Exception as exc:
        return 0, str(exc)

def refresh_nfl_foundation(
    requested_season: int,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    nfl_module: Any | None = None,
) -> FoundationResult:
    """Refresh team performance, schedules, rosters, stats, injuries, and depth charts.

    Required performance data falls back to the latest available season. Other
    datasets are independent and report their own status in the manifest.
    """
    if nfl_module is None:
        try:
            import nflreadpy as nfl_module
        except ImportError as exc:
            raise RuntimeError("Install nflreadpy before refreshing NFL data.") from exc

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    team_snapshot_path = root / "team_snapshot.csv"
    performance = _fetch_performance_with_fallback(
        int(requested_season), team_snapshot_path
    )
    performance = _ensure_current_performance_schema(performance, team_snapshot_path)
    scheme_path = root / "scheme_tendencies.csv"
    scheme_rows, scheme_error = _ensure_scheme_snapshot(
        source_season=int(performance.season),
        scheme_path=scheme_path,
        nfl_module=nfl_module,
    )
    statuses: list[DatasetStatus] = [
        DatasetStatus(
            name="team_performance",
            file=str(root / "team_snapshot.csv"),
            rows=performance.rows,
            available=True,
            error=performance.warning,
        ),
        DatasetStatus(
            name="scheme_tendencies",
            file=str(scheme_path),
            rows=scheme_rows,
            available=scheme_rows > 0,
            error="" if scheme_rows > 0 else (scheme_error or "scheme snapshot unavailable"),
        ),
    ]

    season = int(requested_season)
    specs = [
        ("schedules", "schedules.csv", lambda: nfl_module.load_schedules([season]), lambda f: _season_filter(f, season)),
        ("rosters", "rosters.csv", lambda: nfl_module.load_rosters([season]), lambda f: _season_filter(f, season)),
        ("prior_rosters", "prior_rosters.csv", lambda: nfl_module.load_rosters([season - 1]), lambda f: _season_filter(f, season - 1)),
        ("weekly_rosters", "weekly_rosters.csv", lambda: nfl_module.load_rosters_weekly([season]), lambda f: _latest_week_rows(_season_filter(f, season))),
        ("player_weekly_stats", "player_weekly_stats.csv", lambda: nfl_module.load_player_stats([season], summary_level="week"), lambda f: _season_filter(f, season)),
        ("team_weekly_stats", "team_weekly_stats.csv", lambda: nfl_module.load_team_stats([season], summary_level="week"), lambda f: _season_filter(f, season)),
        ("snap_counts", "snap_counts.csv", lambda: nfl_module.load_snap_counts([season]), lambda f: _season_filter(f, season)),
        ("injuries", "injuries.csv", lambda: nfl_module.load_injuries([season]), lambda f: _season_filter(f, season)),
        ("depth_charts", "depth_charts.csv", lambda: nfl_module.load_depth_charts([season]), lambda f: _latest_week_rows(_season_filter(f, season))),
    ]

    for name, filename, loader, transform in specs:
        statuses.append(
            _safe_dataset(
                name=name,
                filename=filename,
                loader=loader,
                data_dir=root,
                transform=transform,
            )
        )

    result = FoundationResult(
        requested_season=season,
        performance_season=performance.season,
        updated_at_utc=updated_at,
        data_dir=str(root),
        datasets=tuple(statuses),
    )
    manifest = {
        "schema_version": "1.0",
        "requested_season": result.requested_season,
        "performance_season": result.performance_season,
        "updated_at_utc": result.updated_at_utc,
        "available_datasets": result.available_count,
        "total_datasets": len(result.datasets),
        "datasets": [asdict(item) for item in result.datasets],
    }
    _atomic_json(manifest, root / MANIFEST_NAME)
    return result


def load_foundation_status(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    path = Path(data_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
