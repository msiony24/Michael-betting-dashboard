from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROSTER_PATHS = (
    PROJECT_ROOT / "data" / "nfl" / "weekly_rosters.csv",
    PROJECT_ROOT / "data" / "nfl" / "rosters.csv",
)
DEFAULT_PRIOR_MADDEN_PATHS = (
    PROJECT_ROOT / "data" / "madden_26_players.csv",
    PROJECT_ROOT / "data" / "madden_25_players.csv",
)

TEAM_COLUMNS = ("team", "recent_team", "club_code", "team_abbr", "team_abbreviation")
POSITION_COLUMNS = ("position", "pos", "position_group", "depth_chart_position")
NAME_COLUMNS = ("full_name", "player_name", "display_name", "football_name", "name")
FIRST_NAME_COLUMNS = ("first_name", "firstname", "firstName")
LAST_NAME_COLUMNS = ("last_name", "lastname", "lastName")


def normalize_person_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("’", "'")
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", text)
    return "".join(ch for ch in text if ch.isalnum())


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in columns}
    for candidate in candidates:
        found = lookup.get(candidate.casefold())
        if found:
            return found
    return None


def _build_name(frame: pd.DataFrame) -> pd.Series:
    full = _first_existing(frame.columns, NAME_COLUMNS)
    if full:
        return frame[full].fillna("").astype(str).str.strip()

    first = _first_existing(frame.columns, FIRST_NAME_COLUMNS)
    last = _first_existing(frame.columns, LAST_NAME_COLUMNS)
    if first or last:
        first_values = frame[first].fillna("").astype(str) if first else ""
        last_values = frame[last].fillna("").astype(str) if last else ""
        return (first_values + " " + last_values).str.strip()

    return pd.Series([""] * len(frame), index=frame.index, dtype="object")


def load_identity_map(paths: Iterable[Path] | None = None) -> pd.DataFrame:
    """Build a name -> team/position lookup from nflverse and prior Madden files."""
    candidate_paths = tuple(paths or (*DEFAULT_ROSTER_PATHS, *DEFAULT_PRIOR_MADDEN_PATHS))
    chunks: list[pd.DataFrame] = []

    for priority, path in enumerate(candidate_paths):
        path = Path(path)
        if not path.exists():
            continue
        try:
            source = pd.read_csv(path)
        except Exception:
            continue
        if source.empty:
            continue

        team_col = _first_existing(source.columns, TEAM_COLUMNS)
        position_col = _first_existing(source.columns, POSITION_COLUMNS)
        if not team_col and not position_col:
            continue

        chunk = pd.DataFrame(
            {
                "player_name": _build_name(source),
                "team": source[team_col] if team_col else "",
                "position": source[position_col] if position_col else "",
                "source_priority": priority,
                "identity_source": path.name,
            }
        )
        chunk["name_key"] = chunk["player_name"].map(normalize_person_name)
        chunk["team"] = chunk["team"].fillna("").astype(str).str.strip()
        chunk["position"] = chunk["position"].fillna("").astype(str).str.strip().str.upper()
        chunk = chunk[chunk["name_key"].ne("") & (chunk["team"].ne("") | chunk["position"].ne(""))]
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=["name_key", "team", "position", "identity_source"])

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.sort_values(["source_priority"])

    # Prefer the first source, while filling an empty team/position from later sources.
    rows = []
    for name_key, group in combined.groupby("name_key", sort=False):
        team = next((value for value in group["team"] if value), "")
        position = next((value for value in group["position"] if value), "")
        source = next((value for value in group["identity_source"] if value), "")
        rows.append({"name_key": name_key, "team": team, "position": position, "identity_source": source})

    return pd.DataFrame(rows)


def enrich_player_identities(
    players: pd.DataFrame,
    identity_map: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = players.copy()
    if "team" not in frame:
        frame["team"] = ""
    if "position" not in frame:
        frame["position"] = ""

    frame["team"] = frame["team"].fillna("").astype(str).str.strip()
    frame["position"] = frame["position"].fillna("").astype(str).str.strip().str.upper()
    frame["name_key"] = frame["player_name"].map(normalize_person_name)

    lookup = identity_map if identity_map is not None else load_identity_map()
    if lookup.empty:
        frame["identity_source"] = "ea"
        return frame.drop(columns=["name_key"]), {
            "identity_records": 0,
            "team_filled": 0,
            "position_filled": 0,
            "fully_resolved": int((frame["team"].ne("") & frame["position"].ne("")).sum()),
            "unresolved": int((frame["team"].eq("") | frame["position"].eq("")).sum()),
        }

    merged = frame.merge(lookup, on="name_key", how="left", suffixes=("", "_lookup"))
    missing_team = merged["team"].eq("")
    missing_position = merged["position"].eq("")
    merged.loc[missing_team, "team"] = merged.loc[missing_team, "team_lookup"].fillna("")
    merged.loc[missing_position, "position"] = merged.loc[missing_position, "position_lookup"].fillna("")
    merged["identity_source"] = merged["identity_source"].fillna("ea")

    stats = {
        "identity_records": int(len(lookup)),
        "team_filled": int((missing_team & merged["team"].ne("")).sum()),
        "position_filled": int((missing_position & merged["position"].ne("")).sum()),
        "fully_resolved": int((merged["team"].ne("") & merged["position"].ne("")).sum()),
        "unresolved": int((merged["team"].eq("") | merged["position"].eq("")).sum()),
    }
    return merged.drop(columns=["name_key", "team_lookup", "position_lookup"]), stats
