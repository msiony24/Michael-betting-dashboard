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

from engine.nfl_fetch import FetchResult, fetch_and_build


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "nfl"
MANIFEST_NAME = "foundation_status.json"


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
    raise RuntimeError("No supported NFL performance season could be loaded.") from last_error


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

    performance = _fetch_performance_with_fallback(
        int(requested_season), root / "team_snapshot.csv"
    )
    statuses: list[DatasetStatus] = [
        DatasetStatus(
            name="team_performance",
            file=str(root / "team_snapshot.csv"),
            rows=performance.rows,
            available=True,
        )
    ]

    season = int(requested_season)
    specs = [
        ("schedules", "schedules.csv", lambda: nfl_module.load_schedules([season]), lambda f: _season_filter(f, season)),
        ("rosters", "rosters.csv", lambda: nfl_module.load_rosters([season]), lambda f: _season_filter(f, season)),
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
