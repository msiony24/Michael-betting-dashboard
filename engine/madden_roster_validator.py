from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MADDEN_PLAYERS = PROJECT_ROOT / "data" / "madden_27_players.csv"
DEFAULT_PLAYER_RATINGS = PROJECT_ROOT / "data" / "nfl" / "player_ratings.csv"
DEFAULT_TEAM_RATINGS = PROJECT_ROOT / "data" / "nfl" / "team_ratings_auto.json"
DEFAULT_STATUS = PROJECT_ROOT / "data" / "nfl" / "rating_status.json"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "data" / "nfl" / "madden_27_validation.json"
DEFAULT_REPORT_CSV = PROJECT_ROOT / "data" / "nfl" / "madden_27_validation.csv"

EXPECTED_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LA", "MIA", "MIN", "NE", "NO",
    "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}

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


FULL_TEAM_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

POSITION_GROUPS = {
    "QB": {"QB"},
    "RB": {"RB", "HB", "FB"},
    "WR_TE": {"WR", "TE"},
    "OL": {"LT", "LG", "C", "RG", "RT", "OL", "G", "T"},
    "DL_EDGE": {"DE", "DT", "DL", "LE", "RE", "EDGE"},
    "LB": {"LB", "MLB", "ILB", "OLB", "LOLB", "ROLB"},
    "SECONDARY": {"CB", "DB", "FS", "SS", "S"},
    "ST": {"K", "P", "LS"},
}

MIN_COUNTS = {
    "QB": 2, "RB": 3, "WR_TE": 6, "OL": 7,
    "DL_EDGE": 6, "LB": 4, "SECONDARY": 7, "ST": 2,
}


def _canon_team(value: object) -> str:
    raw = str(value or "").strip()
    if raw in FULL_TEAM_NAMES:
        return FULL_TEAM_NAMES[raw]
    text = raw.upper()
    return TEAM_ALIASES.get(text, text)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def _top_names(frame: pd.DataFrame, n: int = 3) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    sort_col = "macabets_rating" if "macabets_rating" in frame.columns else "overall"
    ordered = frame.sort_values(sort_col, ascending=False).head(n)
    rows = []
    for _, row in ordered.iterrows():
        rows.append({
            "name": str(row.get("player_name", "")),
            "position": str(row.get("position", "")),
            "overall": float(row.get("overall", 0) or 0),
            "macabets_rating": (
                float(row.get("macabets_rating"))
                if pd.notna(row.get("macabets_rating")) else None
            ),
        })
    return rows


def build_validation_report(
    madden_players_path: Path | str = DEFAULT_MADDEN_PLAYERS,
    player_ratings_path: Path | str = DEFAULT_PLAYER_RATINGS,
    team_ratings_path: Path | str = DEFAULT_TEAM_RATINGS,
    status_path: Path | str = DEFAULT_STATUS,
) -> dict[str, Any]:
    madden = pd.read_csv(madden_players_path)
    madden["team_canon"] = madden["team"].map(_canon_team)
    madden["position"] = madden["position"].fillna("").astype(str).str.upper().str.strip()
    madden["overall"] = pd.to_numeric(madden["overall"], errors="coerce")

    rated = pd.read_csv(player_ratings_path) if Path(player_ratings_path).exists() else pd.DataFrame()
    if not rated.empty:
        rated["team_canon"] = rated["team_abbr"].map(_canon_team)
        rated["position"] = rated["position"].fillna("").astype(str).str.upper().str.strip()
        rated["overall"] = pd.to_numeric(rated["overall"], errors="coerce")
        if "macabets_rating" in rated.columns:
            rated["macabets_rating"] = pd.to_numeric(rated["macabets_rating"], errors="coerce")

    madden_teams = set(madden.loc[madden["team_canon"].isin(EXPECTED_TEAMS), "team_canon"])
    rated_teams = set(rated["team_canon"]) & EXPECTED_TEAMS if not rated.empty else set()

    rows: list[dict[str, Any]] = []
    team_details: dict[str, Any] = {}
    critical_errors: list[str] = []
    warnings: list[str] = []

    for team in sorted(EXPECTED_TEAMS):
        raw_team = madden[madden["team_canon"].eq(team)].copy()
        rated_team = rated[rated["team_canon"].eq(team)].copy() if not rated.empty else pd.DataFrame()
        source = rated_team if not rated_team.empty else raw_team

        unit_counts = {}
        unit_top = {}
        thin_units = []
        for unit, positions in POSITION_GROUPS.items():
            unit_frame = source[source["position"].isin(positions)] if not source.empty else source
            count = int(len(unit_frame))
            unit_counts[unit] = count
            unit_top[unit] = _top_names(unit_frame, 5 if unit in {"QB", "WR_TE", "OL"} else 3)
            if count < MIN_COUNTS[unit]:
                thin_units.append(f"{unit}:{count}")

        qb_names = [item["name"] for item in unit_top["QB"][:2]]
        status = "PASS"
        if team not in madden_teams:
            status = "FAIL"
            critical_errors.append(f"{team}: missing from Madden 27 player file")
        if team not in rated_teams:
            status = "FAIL"
            critical_errors.append(f"{team}: missing from automated player ratings")
        if unit_counts["QB"] == 0:
            status = "FAIL"
            critical_errors.append(f"{team}: no quarterback resolved")
        elif thin_units:
            if status != "FAIL":
                status = "WARN"
            warnings.append(f"{team}: thin position coverage ({', '.join(thin_units)})")

        row = {
            "team": team,
            "status": status,
            "madden_players": int(len(raw_team)),
            "rated_players": int(len(rated_team)),
            "qb1": qb_names[0] if qb_names else "",
            "qb2": qb_names[1] if len(qb_names) > 1 else "",
            **{f"count_{k.lower()}": v for k, v in unit_counts.items()},
        }
        rows.append(row)
        team_details[team] = {
            **row,
            "top_by_unit": unit_top,
            "thin_units": thin_units,
        }

    known_positions = set().union(*POSITION_GROUPS.values())
    unknown_positions = sorted(
        position for position in set(madden["position"].dropna())
        if position and position not in known_positions
    )
    if unknown_positions:
        warnings.append("Unmapped Madden positions: " + ", ".join(unknown_positions))

    rating_sources = (
        rated["rating_source"].fillna("").value_counts().to_dict()
        if not rated.empty and "rating_source" in rated.columns else {}
    )
    if any("Madden 26" in str(source) for source in rating_sources):
        warnings.append("Automated player_ratings.csv still labels Madden 27 data as Madden 26.")

    status_payload = _load_json(Path(status_path))
    perf_count = int(status_payload.get("players_with_performance_data", 0) or 0)
    if perf_count == 0:
        warnings.append(
            "0 players have nflverse performance data. This is not a Madden roster failure; "
            "the current NFL refresh does not generate data/nfl/player_weekly_stats.csv."
        )

    team_ratings = _load_json(Path(team_ratings_path))
    normalized_team_rating_keys = {_canon_team(key) for key in team_ratings}
    missing_team_ratings = sorted(EXPECTED_TEAMS - normalized_team_rating_keys)
    if missing_team_ratings:
        critical_errors.append(
            "Automated team ratings missing: " + ", ".join(missing_team_ratings)
        )

    report = {
        "summary": {
            "madden_player_rows": int(len(madden)),
            "madden_teams_raw": int(madden["team"].nunique()),
            "madden_teams_canonical": int(len(madden_teams)),
            "rated_player_rows": int(len(rated)),
            "rated_teams": int(len(rated_teams)),
            "expected_teams": 32,
            "players_with_nflverse_performance": perf_count,
            "critical_error_count": len(critical_errors),
            "warning_count": len(warnings),
            "validation_passed": len(critical_errors) == 0,
        },
        "critical_errors": critical_errors,
        "warnings": warnings,
        "rating_source_counts": rating_sources,
        "teams": team_details,
        "team_rows": rows,
    }
    return report


def save_validation_report(
    report: dict[str, Any],
    json_path: Path | str = DEFAULT_REPORT_JSON,
    csv_path: Path | str = DEFAULT_REPORT_CSV,
) -> tuple[Path, Path]:
    json_out, csv_out = Path(json_path), Path(csv_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(report["team_rows"]).to_csv(csv_out, index=False)
    return json_out, csv_out
