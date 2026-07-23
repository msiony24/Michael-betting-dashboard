"""Turn an nflverse team-stat snapshot into Macabets component ratings."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

RATING_COLUMNS = [
    "offense", "defense", "quarterback", "strength_of_schedule", "special_teams"
]


def load_snapshot(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(file_path)
    if "team" not in frame:
        return pd.DataFrame()
    return frame


def snapshot_profiles(path: str | Path, coaching_priors: dict[str, float]) -> tuple[dict, dict]:
    frame = load_snapshot(path)
    if frame.empty:
        return {}, {"available": False, "reason": "No NFL snapshot file has been generated yet."}

    profiles = {}
    for _, row in frame.iterrows():
        team = str(row["team"])
        profile = {}
        for column in RATING_COLUMNS:
            value = pd.to_numeric(row.get(column), errors="coerce")
            profile[column] = float(value) if pd.notna(value) else 67.5
        profile["coaching"] = float(coaching_priors.get(team, 67.5))
        profiles[team] = profile

    meta = {
        "available": bool(profiles),
        "season": int(pd.to_numeric(frame.get("season"), errors="coerce").dropna().max()) if "season" in frame and frame["season"].notna().any() else None,
        "updated_at_utc": str(frame.get("updated_at_utc", pd.Series(["Unknown"])).iloc[0]),
        "data_source": str(frame.get("data_source", pd.Series(["Unknown"])).iloc[0]),
        "teams": len(profiles),
        "sample_plays": int(pd.to_numeric(frame.get("games_or_sample_plays"), errors="coerce").fillna(0).sum()) if "games_or_sample_plays" in frame else 0,
    }
    return profiles, meta
