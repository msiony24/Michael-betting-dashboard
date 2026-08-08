from __future__ import annotations

import io
import re
import unicodedata
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NFL_DATA_DIR = DATA_DIR / "nfl"

DEFAULT_SEASON = date.today().year
DEFAULT_MADDEN_EA_PATH = DATA_DIR / "madden_27_players_ea.csv"
DEFAULT_OUTPUT_PATH = DATA_DIR / "madden_27_players.csv"
DEFAULT_ROSTER_CACHE = NFL_DATA_DIR / "roster_2026.csv"

ROSTER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "rosters/roster_{season}.csv"
)

TEAM_ALIASES = {
    "AZ": "ARI", "ARZ": "ARI", "ARI": "ARI",
    "LAR": "LA", "LA": "LA",
    "JAC": "JAX", "JAX": "JAX",
    "KAN": "KC", "KC": "KC",
    "GNB": "GB", "GB": "GB",
    "LVR": "LV", "LV": "LV",
    "NWE": "NE", "NE": "NE",
    "NOR": "NO", "NO": "NO",
    "SFO": "SF", "SF": "SF",
    "TAM": "TB", "TB": "TB",
    "WSH": "WAS", "WAS": "WAS",
}

EXPECTED_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LA", "MIA", "MIN", "NE", "NO",
    "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    parts = [p for p in text.split() if p and p not in _SUFFIXES]
    return " ".join(parts)


def _clean_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_name = {str(c).casefold(): str(c) for c in columns}
    for candidate in candidates:
        found = by_name.get(candidate.casefold())
        if found:
            return found
    return None


def _download_csv(url: str, timeout: int = 45) -> pd.DataFrame:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 Macabets/0.62",
            "Accept": "text/csv,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read()
    return pd.read_csv(io.BytesIO(content), low_memory=False)


def fetch_nflverse_roster(
    season: int = DEFAULT_SEASON,
    cache_path: Path | str | None = None,
) -> pd.DataFrame:
    cache = Path(cache_path) if cache_path else NFL_DATA_DIR / f"roster_{season}.csv"
    url = ROSTER_URL.format(season=int(season))
    print(f"Downloading nflverse {season} roster: {url}")

    try:
        roster = _download_csv(url)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix(cache.suffix + ".tmp")
        roster.to_csv(temp, index=False)
        temp.replace(cache)
        print(f"Saved nflverse roster cache: {cache}")
        return roster
    except Exception as exc:
        if cache.exists():
            print(
                "Live nflverse roster download failed; using last known-good roster cache. "
                f"Reason: {exc}"
            )
            return pd.read_csv(cache, low_memory=False)
        raise RuntimeError(
            f"Could not download nflverse roster for {season}, and no cache exists: {exc}"
        ) from exc


def standardize_roster(roster: pd.DataFrame) -> pd.DataFrame:
    frame = roster.copy()
    name_col = _find_column(
        frame.columns,
        ("full_name", "player_display_name", "display_name", "player_name", "name"),
    )
    if not name_col:
        first = _find_column(frame.columns, ("first_name", "firstname"))
        last = _find_column(frame.columns, ("last_name", "lastname"))
        if first and last:
            frame["_built_name"] = (
                frame[first].fillna("").astype(str).str.strip()
                + " "
                + frame[last].fillna("").astype(str).str.strip()
            ).str.strip()
            name_col = "_built_name"

    team_col = _find_column(frame.columns, ("team", "team_abbr", "club"))
    pos_col = _find_column(
        frame.columns,
        ("position", "depth_chart_position", "position_group", "pos"),
    )
    birth_col = _find_column(
        frame.columns,
        ("birth_date", "birthdate", "date_of_birth", "dob"),
    )
    gsis_col = _find_column(frame.columns, ("gsis_id", "player_id"))
    status_col = _find_column(frame.columns, ("status", "roster_status"))
    height_col = _find_column(frame.columns, ("height",))
    weight_col = _find_column(frame.columns, ("weight",))

    missing = [
        label for label, col in
        (("name", name_col), ("team", team_col), ("position", pos_col))
        if col is None
    ]
    if missing:
        raise RuntimeError(
            "nflverse roster schema is missing required fields: " + ", ".join(missing)
        )

    out = pd.DataFrame({
        "roster_name": frame[name_col].fillna("").astype(str).str.strip(),
        "team": frame[team_col].fillna("").astype(str).str.upper().str.strip(),
        "position": frame[pos_col].fillna("").astype(str).str.upper().str.strip(),
        "birthdate_roster": (
            frame[birth_col].map(_clean_date) if birth_col else ""
        ),
        "gsis_id": (
            frame[gsis_col].fillna("").astype(str).str.strip() if gsis_col else ""
        ),
        "roster_status": (
            frame[status_col].fillna("").astype(str).str.strip() if status_col else ""
        ),
        "height_roster": (
            pd.to_numeric(frame[height_col], errors="coerce") if height_col else float("nan")
        ),
        "weight_roster": (
            pd.to_numeric(frame[weight_col], errors="coerce") if weight_col else float("nan")
        ),
    })
    out["team"] = out["team"].map(lambda x: TEAM_ALIASES.get(x, x))
    out["name_key"] = out["roster_name"].map(_normalize_name)
    out = out[
        out["name_key"].ne("")
        & out["team"].isin(EXPECTED_TEAMS)
        & out["position"].ne("")
    ].copy()

    # Prefer active/inactive roster rows over UFA/RET rows when duplicate names occur.
    status_rank = {
        "ACT": 0, "RES": 1, "INA": 2, "PUP": 3, "NFI": 4, "DEV": 5,
        "PRA": 6, "UFA": 20, "RET": 30,
    }
    out["_status_rank"] = out["roster_status"].str.upper().map(status_rank).fillna(10)
    return out.sort_values(["name_key", "_status_rank"]).reset_index(drop=True)


def _choose_candidate(madden_row: pd.Series, candidates: pd.DataFrame) -> tuple[pd.Series | None, str]:
    if candidates.empty:
        return None, "unmatched"

    if len(candidates) == 1:
        return candidates.iloc[0], "exact_name"

    m_birth = _clean_date(madden_row.get("birthdate"))
    if m_birth:
        birth_matches = candidates[candidates["birthdate_roster"].eq(m_birth)]
        if len(birth_matches) == 1:
            return birth_matches.iloc[0], "exact_name_birthdate"

    # For same-name collisions, height/weight can resolve without fuzzy identity guessing.
    m_height = pd.to_numeric(madden_row.get("height"), errors="coerce")
    m_weight = pd.to_numeric(madden_row.get("weight"), errors="coerce")
    if pd.notna(m_height) or pd.notna(m_weight):
        scored = candidates.copy()
        score = pd.Series(0.0, index=scored.index)
        if pd.notna(m_height):
            score += (scored["height_roster"] - float(m_height)).abs().fillna(50.0)
        if pd.notna(m_weight):
            score += (scored["weight_roster"] - float(m_weight)).abs().fillna(100.0) / 10.0
        scored["_identity_distance"] = score
        scored = scored.sort_values("_identity_distance")
        if len(scored) == 1 or (
            float(scored.iloc[0]["_identity_distance"]) + 0.5
            < float(scored.iloc[1]["_identity_distance"])
        ):
            return scored.iloc[0], "exact_name_body_match"

    return None, "ambiguous_name"


def enrich_madden_players(
    madden: pd.DataFrame,
    roster: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    nfl = standardize_roster(roster)
    by_name = {key: group for key, group in nfl.groupby("name_key", sort=False)}

    rows = []
    methods: dict[str, int] = {}
    for _, mrow in madden.iterrows():
        row = mrow.to_dict()
        key = _normalize_name(row.get("player_name"))
        candidate, method = _choose_candidate(mrow, by_name.get(key, pd.DataFrame()))
        methods[method] = methods.get(method, 0) + 1

        if candidate is not None:
            row["team"] = candidate["team"]
            row["position"] = candidate["position"]
            row["gsis_id"] = candidate["gsis_id"]
            row["roster_status"] = candidate["roster_status"]
            row["roster_match_name"] = candidate["roster_name"]
        else:
            # Keep any future EA-populated values, but never manufacture identity.
            row["team"] = str(row.get("ea_team") or "").strip()
            row["position"] = str(row.get("ea_position") or "").strip().upper()
            row["gsis_id"] = ""
            row["roster_status"] = ""
            row["roster_match_name"] = ""

        row["roster_match_method"] = method
        rows.append(row)

    result = pd.DataFrame(rows)
    result["team"] = result["team"].fillna("").astype(str).str.strip()
    result["position"] = result["position"].fillna("").astype(str).str.upper().str.strip()

    matched = result["team"].isin(EXPECTED_TEAMS) & result["position"].ne("")
    teams = sorted(result.loc[matched, "team"].unique().tolist())
    report = {
        "madden_players": int(len(result)),
        "matched_players": int(matched.sum()),
        "match_rate": round(float(matched.mean()), 4) if len(result) else 0.0,
        "teams_recognized": len(teams),
        "teams": teams,
        "match_methods": methods,
        "unmatched_players": int((~matched).sum()),
    }
    return result, report


def enrich_and_save_madden_players(
    madden_path: Path | str = DEFAULT_MADDEN_EA_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    season: int = DEFAULT_SEASON,
) -> tuple[pd.DataFrame, dict]:
    madden = pd.read_csv(madden_path, low_memory=False)
    roster = fetch_nflverse_roster(season=season)
    enriched, report = enrich_madden_players(madden, roster)

    # Validation guard: do not replace a known-good final database with a broken join.
    if report["teams_recognized"] < 32:
        missing = sorted(EXPECTED_TEAMS - set(report["teams"]))
        raise RuntimeError(
            "Madden-to-NFL roster join did not cover all 32 teams. "
            f"Recognized {report['teams_recognized']}; missing: {', '.join(missing)}"
        )
    if report["matched_players"] < 1000:
        raise RuntimeError(
            "Madden-to-NFL roster join matched fewer than 1,000 players; "
            "refusing to replace the last known-good dataset."
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    enriched.to_csv(temp, index=False)
    temp.replace(output)
    return enriched, report


if __name__ == "__main__":
    frame, report = enrich_and_save_madden_players()
    print(report)
