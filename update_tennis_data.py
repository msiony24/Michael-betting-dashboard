from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd
import requests

from engine.api_tennis import APITennisClient, APITennisError


DATA_DIR = Path(__file__).resolve().parent / "data"
START_YEAR = 2021
SOURCE_TEMPLATE = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"
LIVE_LOOKBACK_DAYS = 60
REFRESH_STATUS_PATH = DATA_DIR / "tennis_refresh_status.json"
MATCH_COLUMNS = [
    "tourney_date", "tourney_name", "surface", "tourney_level", "round",
    "winner_name", "loser_name", "winner_rank", "loser_rank", "score",
    "winner_age", "loser_age", "w_ace", "l_ace", "w_df", "l_df",
    "w_svpt", "l_svpt", "w_1stIn", "l_1stIn", "w_1stWon", "l_1stWon",
    "w_2ndWon", "l_2ndWon", "w_SvGms", "l_SvGms", "w_bpSaved",
    "l_bpSaved", "w_bpFaced", "l_bpFaced",
]


STAT_COLUMNS = [
    "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt",
    "w_1stIn", "l_1stIn", "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon",
    "w_SvGms", "l_SvGms", "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced",
]


def normalize_round(value: object) -> str:
    mapping = {
        "1st Round": "R128",
        "2nd Round": "R64",
        "3rd Round": "R32",
        "4th Round": "R16",
        "Round Robin": "RR",
        "Quarterfinals": "QF",
        "Quarterfinal": "QF",
        "Semifinals": "SF",
        "Semifinal": "SF",
        "The Final": "F",
        "Final": "F",
        "Round of 128": "R128",
        "Round of 64": "R64",
        "Round of 32": "R32",
        "Round of 16": "R16",
        "1/8-finals": "R16",
        "Quarter-finals": "QF",
        "Semi-finals": "SF",
    }
    text = str(value or "").strip()
    if " - " in text:
        text = text.rsplit(" - ", 1)[-1].strip()
    return mapping.get(text, text)


def normalize_level(value: object) -> str:
    text = str(value).strip().lower()
    if "grand slam" in text:
        return "G"
    if "masters" in text or "1000" in text:
        return "M"
    if "atp250" in text or "atp 250" in text:
        return "A"
    if "atp500" in text or "atp 500" in text:
        return "A"
    if "masters cup" in text or "tour finals" in text:
        return "F"
    return "A"


def build_score(row: pd.Series) -> str:
    parts: list[str] = []
    for number in range(1, 6):
        w_col = f"W{number}"
        l_col = f"L{number}"
        if w_col not in row.index or l_col not in row.index:
            continue
        w = row.get(w_col)
        l = row.get(l_col)
        if pd.isna(w) or pd.isna(l):
            continue
        try:
            parts.append(f"{int(float(w))}-{int(float(l))}")
        except (TypeError, ValueError):
            continue
    return " ".join(parts)


def convert_year(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    frame = frame.copy()

    required = ["Date", "Tournament", "Surface", "Winner", "Loser"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{year} workbook is missing required columns: {', '.join(missing)}"
        )

    dates = pd.to_datetime(frame["Date"], errors="coerce", dayfirst=True)

    output = pd.DataFrame({
        "tourney_date": dates.dt.strftime("%Y%m%d"),
        "tourney_name": frame["Tournament"].astype(str).str.strip(),
        "surface": frame["Surface"].astype(str).str.strip().str.title(),
        "tourney_level": frame.get("Series", "ATP").map(normalize_level)
            if "Series" in frame.columns else "A",
        "round": frame.get("Round", "").map(normalize_round)
            if "Round" in frame.columns else "",
        "winner_name": frame["Winner"].astype(str).str.strip(),
        "loser_name": frame["Loser"].astype(str).str.strip(),
        "winner_rank": pd.to_numeric(frame.get("WRank"), errors="coerce"),
        "loser_rank": pd.to_numeric(frame.get("LRank"), errors="coerce"),
        "score": frame.apply(build_score, axis=1),
    })

    for column in MATCH_COLUMNS[10:]:
        output[column] = pd.NA

    output = output.dropna(subset=["tourney_date", "winner_name", "loser_name"])
    output = output[
        (output["winner_name"] != "")
        & (output["loser_name"] != "")
        & (output["winner_name"].str.lower() != "nan")
        & (output["loser_name"].str.lower() != "nan")
    ]
    return output[MATCH_COLUMNS]


def download_year(year: int) -> pd.DataFrame:
    url = SOURCE_TEMPLATE.format(year=year)
    print(f"Downloading ATP {year} baseline: {url}")

    response = requests.get(
        url,
        timeout=180,
        headers={"User-Agent": "Macabets personal tennis analytics"},
    )
    response.raise_for_status()

    if len(response.content) < 1000:
        raise RuntimeError(f"Downloaded workbook was unexpectedly small: {url}")

    workbook = pd.ExcelFile(BytesIO(response.content))
    sheet = str(year) if str(year) in workbook.sheet_names else workbook.sheet_names[0]
    raw = pd.read_excel(workbook, sheet_name=sheet)
    return convert_year(raw, year)


def _name_tokens(value: object) -> list[str]:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"[A-Za-z]+", text.casefold())


def player_signature(value: object) -> tuple[str, str]:
    """Return a provider-tolerant (surname, first-initial) signature."""
    tokens = [t for t in _name_tokens(value) if t not in {"jr", "sr", "ii", "iii", "iv"}]
    if not tokens:
        return "", ""

    # tennis-data.co.uk: "Tirante T.A." or "Fritz T."
    if len(tokens) >= 2 and len(tokens[0]) > 1 and all(len(t) == 1 for t in tokens[1:]):
        return tokens[0], tokens[1]

    # API-Tennis may use "T. A. Tirante" or a full name such as
    # "Thiago Agustin Tirante".
    if len(tokens) >= 2:
        return tokens[-1], tokens[0][0]
    return tokens[0], ""


def _existing_name_map(frames: list[pd.DataFrame]) -> dict[tuple[str, str], str]:
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for column in ("winner_name", "loser_name"):
            if column not in frame:
                continue
            for name in frame[column].dropna().astype(str):
                signature = player_signature(name)
                if not signature[0]:
                    continue
                bucket = counts.setdefault(signature, {})
                bucket[name] = bucket.get(name, 0) + 1

    resolved: dict[tuple[str, str], str] = {}
    for signature, options in counts.items():
        ranked = sorted(options.items(), key=lambda item: item[1], reverse=True)
        resolved[signature] = ranked[0][0]
    return resolved


def resolve_display_name(value: object, existing_names: dict[tuple[str, str], str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return existing_names.get(player_signature(text), text)


def _normalize_tournament(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _surface_lookup(frames: list[pd.DataFrame]) -> dict[str, str]:
    counts: dict[str, dict[str, int]] = {}
    for frame in frames:
        if frame is None or frame.empty or "tourney_name" not in frame or "surface" not in frame:
            continue
        for _, row in frame[["tourney_name", "surface"]].dropna().iterrows():
            key = _normalize_tournament(row["tourney_name"])
            surface = str(row["surface"]).title().strip()
            if not key or not surface:
                continue
            bucket = counts.setdefault(key, {})
            bucket[surface] = bucket.get(surface, 0) + 1
    return {
        key: max(options.items(), key=lambda item: item[1])[0]
        for key, options in counts.items()
    }


def infer_surface(tournament_name: object, historical_surfaces: dict[str, str]) -> str:
    key = _normalize_tournament(tournament_name)
    if key in historical_surfaces:
        return historical_surfaces[key]

    # API-Tennis tournament labels often include city/country/type suffixes.
    for historical_name, surface in historical_surfaces.items():
        if len(historical_name) >= 5 and (
            historical_name in key or key in historical_name
        ):
            return surface

    keywords = {
        "montreal": "Hard", "toronto": "Hard", "canada": "Hard",
        "cincinnati": "Hard", "us open": "Hard", "washington": "Hard",
        "wimbledon": "Grass", "halle": "Grass", "queens": "Grass",
        "roland garros": "Clay", "french open": "Clay", "rome": "Clay",
        "madrid": "Clay", "monte carlo": "Clay", "barcelona": "Clay",
    }
    for keyword, surface in keywords.items():
        if keyword in key:
            return surface
    return "Hard"


def _rank_map(standings: list[dict]) -> dict[str, float]:
    ranks: dict[str, float] = {}
    for row in standings:
        player_key = str(row.get("player_key") or "").strip()
        try:
            rank = float(row.get("place"))
        except (TypeError, ValueError):
            continue
        if player_key:
            ranks[player_key] = rank
    return ranks


def _score_from_api(event: dict, winner_side: str) -> str:
    scores = event.get("scores") or []
    parts: list[str] = []
    for set_row in scores:
        first = str(set_row.get("score_first") or "").strip()
        second = str(set_row.get("score_second") or "").strip()
        if not first or not second:
            continue
        if winner_side == "first":
            parts.append(f"{first}-{second}")
        else:
            parts.append(f"{second}-{first}")
    return " ".join(parts)


def _stat_row(event: dict, player_key: str, *stat_names: str) -> dict | None:
    wanted = {str(name).strip().casefold() for name in stat_names if str(name).strip()}
    for row in event.get("statistics") or []:
        if str(row.get("player_key") or "") != str(player_key):
            continue
        if str(row.get("stat_period") or "match").casefold() != "match":
            continue
        if str(row.get("stat_name") or "").strip().casefold() in wanted:
            return row
    return None


def _numeric(value: object) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _stat_value(event: dict, player_key: str, *stat_names: str) -> float | None:
    row = _stat_row(event, player_key, *stat_names)
    return _numeric(row.get("stat_value")) if row else None


def _stat_won_total(event: dict, player_key: str, *stat_names: str) -> tuple[float | None, float | None]:
    row = _stat_row(event, player_key, *stat_names)
    if not row:
        return None, None
    won = _numeric(row.get("stat_won"))
    total = _numeric(row.get("stat_total"))
    if won is not None or total is not None:
        return won, total

    # Some providers encode count stats as "4/7" in stat_value.
    raw = str(row.get("stat_value") or "").strip()
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$", raw)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def convert_api_fixtures(
    fixtures: list[dict],
    *,
    existing_names: dict[tuple[str, str], str],
    historical_surfaces: dict[str, str],
    ranks_by_key: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict] = []

    for event in fixtures:
        if str(event.get("event_type_type") or "").strip().casefold() != "atp singles":
            continue
        if str(event.get("event_status") or "").strip().casefold() != "finished":
            continue

        winner_marker = str(event.get("event_winner") or "").strip().casefold()
        if winner_marker not in {"first player", "second player"}:
            continue

        first_name = resolve_display_name(event.get("event_first_player"), existing_names)
        second_name = resolve_display_name(event.get("event_second_player"), existing_names)
        if not first_name or not second_name:
            continue

        first_key = str(event.get("first_player_key") or "").strip()
        second_key = str(event.get("second_player_key") or "").strip()
        first_won = winner_marker == "first player"

        winner_name = first_name if first_won else second_name
        loser_name = second_name if first_won else first_name
        winner_key = first_key if first_won else second_key
        loser_key = second_key if first_won else first_key
        tournament = str(event.get("tournament_name") or "").strip()
        event_date = pd.to_datetime(event.get("event_date"), errors="coerce")
        if pd.isna(event_date):
            continue

        w_first_won, w_first_total = _stat_won_total(
            event, winner_key, "1st serve points won", "First serve points won"
        )
        l_first_won, l_first_total = _stat_won_total(
            event, loser_key, "1st serve points won", "First serve points won"
        )
        w_second_won, w_second_total = _stat_won_total(
            event, winner_key, "2nd serve points won", "Second serve points won"
        )
        l_second_won, l_second_total = _stat_won_total(
            event, loser_key, "2nd serve points won", "Second serve points won"
        )
        w_bp_saved, w_bp_faced = _stat_won_total(
            event, winner_key, "Break Points Saved", "Break points saved"
        )
        l_bp_saved, l_bp_faced = _stat_won_total(
            event, loser_key, "Break Points Saved", "Break points saved"
        )

        def _sum_if_known(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return float(a + b)

        row = {column: pd.NA for column in MATCH_COLUMNS}
        row.update({
            "tourney_date": event_date.strftime("%Y%m%d"),
            "tourney_name": tournament,
            "surface": infer_surface(tournament, historical_surfaces),
            "tourney_level": normalize_level(tournament),
            "round": normalize_round(event.get("tournament_round")),
            "winner_name": winner_name,
            "loser_name": loser_name,
            "winner_rank": ranks_by_key.get(winner_key, pd.NA),
            "loser_rank": ranks_by_key.get(loser_key, pd.NA),
            "score": _score_from_api(event, "first" if first_won else "second"),
            "w_ace": _stat_value(event, winner_key, "Aces"),
            "l_ace": _stat_value(event, loser_key, "Aces"),
            "w_df": _stat_value(event, winner_key, "Double Faults"),
            "l_df": _stat_value(event, loser_key, "Double Faults"),
            "w_svpt": _sum_if_known(w_first_total, w_second_total),
            "l_svpt": _sum_if_known(l_first_total, l_second_total),
            "w_1stIn": w_first_total,
            "l_1stIn": l_first_total,
            "w_1stWon": w_first_won,
            "l_1stWon": l_first_won,
            "w_2ndWon": w_second_won,
            "l_2ndWon": l_second_won,
            "w_SvGms": _stat_value(event, winner_key, "Service Games Played", "Service Games"),
            "l_SvGms": _stat_value(event, loser_key, "Service Games Played", "Service Games"),
            "w_bpSaved": w_bp_saved,
            "l_bpSaved": l_bp_saved,
            "w_bpFaced": w_bp_faced,
            "l_bpFaced": l_bp_faced,
        })
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def _match_key(row: pd.Series) -> str:
    signatures = sorted([player_signature(row.get("winner_name")), player_signature(row.get("loser_name"))])
    players = "|".join(f"{surname}:{initial}" for surname, initial in signatures)
    return f"{str(row.get('tourney_date') or '')}|{players}"


def merge_live_matches(baseline: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    if baseline is None or baseline.empty:
        combined = live.copy()
    elif live is None or live.empty:
        combined = baseline.copy()
    else:
        baseline = baseline.copy()
        live = live.copy()
        baseline["_match_key"] = baseline.apply(_match_key, axis=1)
        live["_match_key"] = live.apply(_match_key, axis=1)

        # Fresh API rows win when the same match already exists in the slower
        # yearly source. This lets corrected scores/results replace stale rows.
        baseline = baseline[~baseline["_match_key"].isin(set(live["_match_key"]))]
        combined = pd.concat([baseline, live], ignore_index=True, sort=False)
        combined = combined.drop(columns=["_match_key"], errors="ignore")

    for column in MATCH_COLUMNS:
        if column not in combined:
            combined[column] = pd.NA
    combined = combined[MATCH_COLUMNS]
    combined["tourney_date"] = combined["tourney_date"].astype(str).str.replace(".0", "", regex=False)
    combined = combined.sort_values(["tourney_date", "tourney_name", "round", "winner_name"])
    return combined.reset_index(drop=True)


def preserve_existing_statistics(baseline: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Carry forward API-enriched point stats when the yearly baseline is refreshed.

    The slower yearly workbook does not include serve/return point totals. Without
    this merge, every daily refresh would erase API-Tennis statistics once a match
    aged outside the rolling live window.
    """
    if baseline is None or baseline.empty or existing is None or existing.empty:
        return baseline.copy() if baseline is not None else pd.DataFrame(columns=MATCH_COLUMNS)

    out = baseline.copy()
    old = existing.copy()
    out["_match_key"] = out.apply(_match_key, axis=1)
    old["_match_key"] = old.apply(_match_key, axis=1)
    old = old.drop_duplicates("_match_key", keep="last").set_index("_match_key")

    for column in STAT_COLUMNS:
        if column not in out:
            out[column] = pd.NA
        if column not in old:
            continue
        lookup = old[column]
        missing = out[column].isna()
        if missing.any():
            out.loc[missing, column] = out.loc[missing, "_match_key"].map(lookup)

    return out.drop(columns=["_match_key"], errors="ignore")


def fetch_live_atp_matches(
    current_year_frames: list[pd.DataFrame],
    *,
    today: date | None = None,
    client: APITennisClient | None = None,
) -> tuple[pd.DataFrame, dict]:
    active_today = today or date.today()
    start_date = active_today - timedelta(days=LIVE_LOOKBACK_DAYS)
    active_client = client or APITennisClient()

    if not active_client.configured:
        raise APITennisError(
            "API_TENNIS_KEY is required for the daily ATP refresh."
        )

    fixtures_response = active_client.get_fixtures(
        start_date,
        active_today,
        timezone_name="America/New_York",
        force_refresh=True,
    )
    standings_response = active_client.get_standings("ATP", force_refresh=True)

    names = _existing_name_map(current_year_frames)
    surfaces = _surface_lookup(current_year_frames)
    ranks = _rank_map(standings_response.result)
    live = convert_api_fixtures(
        fixtures_response.result,
        existing_names=names,
        historical_surfaces=surfaces,
        ranks_by_key=ranks,
    )
    metadata = {
        "window_start": start_date.isoformat(),
        "window_stop": active_today.isoformat(),
        "fixtures_source": fixtures_response.source,
        "fixtures_fetched_at": fixtures_response.fetched_at,
        "all_fixtures_received": len(fixtures_response.result),
        "completed_atp_singles_imported": len(live),
        "matches_with_serve_return_stats": int(
            live[["w_svpt", "l_svpt", "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon"]]
            .notna().all(axis=1).sum()
        ) if not live.empty else 0,
    }
    return live, metadata


def _read_existing_year(year: int) -> pd.DataFrame:
    path = DATA_DIR / f"atp_matches_{year}.csv"
    if not path.exists():
        return pd.DataFrame(columns=MATCH_COLUMNS)
    frame = pd.read_csv(path)
    for column in MATCH_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[MATCH_COLUMNS]


def _write_refresh_status(payload: dict) -> None:
    REFRESH_STATUS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current_year = date.today().year
    failures: list[str] = []
    frames_by_year: dict[int, pd.DataFrame] = {}

    # Historical seasons do not need to be downloaded every morning. Preserve
    # existing snapshots and only download a historical year when it is missing.
    for year in range(START_YEAR, current_year):
        existing = _read_existing_year(year)
        if not existing.empty:
            frames_by_year[year] = existing
            print(f"Keeping ATP {year}: {len(existing):,} historical matches")
            continue
        try:
            converted = download_year(year)
            destination = DATA_DIR / f"atp_matches_{year}.csv"
            converted.to_csv(destination, index=False)
            frames_by_year[year] = converted
            print(f"Saved {destination.name}: {len(converted):,} matches")
        except Exception as exc:
            failures.append(f"{year}: {exc}")

    # Refresh the slower yearly baseline when possible, but never throw away an
    # existing current-year snapshot just because that provider is temporarily down.
    existing_current = _read_existing_year(current_year)
    try:
        baseline_current = download_year(current_year)
        print(f"Current-year baseline contains {len(baseline_current):,} matches")
    except Exception as exc:
        if existing_current.empty:
            failures.append(f"{current_year} baseline: {exc}")
            baseline_current = existing_current
        else:
            print(f"WARNING: current-year baseline refresh failed; using existing CSV: {exc}")
            baseline_current = existing_current

    # Keep point-level statistics previously harvested from API-Tennis even when
    # the slower yearly baseline is refreshed without those fields.
    baseline_current = preserve_existing_statistics(baseline_current, existing_current)

    context_frames = list(frames_by_year.values()) + [baseline_current, existing_current]
    live_metadata: dict = {}
    live_matches = pd.DataFrame(columns=MATCH_COLUMNS)
    try:
        live_matches, live_metadata = fetch_live_atp_matches(context_frames)
        print(
            "API-Tennis rolling refresh: "
            f"{live_metadata['completed_atp_singles_imported']:,} completed ATP singles matches "
            f"from {live_metadata['window_start']} through {live_metadata['window_stop']}"
        )
    except Exception as exc:
        failures.append(f"API-Tennis live refresh: {exc}")

    merged_current = merge_live_matches(baseline_current, live_matches)
    destination = DATA_DIR / f"atp_matches_{current_year}.csv"
    merged_current.to_csv(destination, index=False)
    latest_date = ""
    if not merged_current.empty:
        latest_raw = pd.to_datetime(merged_current["tourney_date"], format="%Y%m%d", errors="coerce").dropna()
        if not latest_raw.empty:
            latest_date = latest_raw.max().date().isoformat()

    status = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_year": current_year,
        "current_year_match_count": len(merged_current),
        "current_year_matches_with_serve_return_stats": int(
            merged_current[["w_svpt", "l_svpt", "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon"]]
            .notna().all(axis=1).sum()
        ) if not merged_current.empty else 0,
        "latest_match_date": latest_date,
        "live_refresh": live_metadata,
        "ok": not failures,
        "failures": failures,
    }
    _write_refresh_status(status)
    print(f"Saved {destination.name}: {len(merged_current):,} matches; latest date {latest_date or 'unknown'}")
    print(f"Saved {REFRESH_STATUS_PATH.name}")

    if failures:
        print("\nATP refresh had failures:")
        for failure in failures:
            print(failure)
        return 1

    print("\nMacabets ATP database update completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
