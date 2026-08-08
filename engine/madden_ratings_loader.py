from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

EA_API_BASE = "https://drop-api.ea.com/rating/madden-nfl"
PAGE_SIZE = 100
DEFAULT_RETRIES = 4
DEFAULT_RETRY_DELAY = 2.0

DEFAULT_EA_CSV_PATH = DATA_DIR / "madden_27_players_ea.csv"
DEFAULT_RAW_PATH = DATA_DIR / "madden_27_raw.json"
DEFAULT_METADATA_PATH = DATA_DIR / "madden_27_metadata.json"


def _request_json(
    url: str,
    timeout: int = 45,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 Macabets/0.62",
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://www.ea.com/games/madden-nfl/ratings",
                "Connection": "close",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
            return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in (408, 425, 429) and not 500 <= exc.code <= 599:
                raise RuntimeError(
                    f"EA ratings request returned HTTP {exc.code}: {detail[:300]}"
                ) from exc
            last_error = exc
        except (
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            socket.timeout,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc

        if attempt < retries:
            wait = retry_delay * attempt
            print(
                f"EA request failed/incomplete; retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{retries})"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"EA ratings request failed after {retries} attempts: {last_error}"
    ) from last_error


def fetch_madden_page(limit: int = PAGE_SIZE, offset: int = 0) -> Any:
    params = urllib.parse.urlencode({"limit": int(limit), "offset": int(offset)})
    return _request_json(f"{EA_API_BASE}?{params}")


def _find_player_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("items", "results", "players", "data", "docs", "ratings", "entries"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _find_player_list(value)
            if nested:
                return nested

    for value in payload.values():
        nested = _find_player_list(value)
        if nested:
            return nested
    return []


def _find_total(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("total", "totalCount", "total_count", "count", "numFound", "totalResults"):
        value = payload.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    for value in payload.values():
        if isinstance(value, dict):
            found = _find_total(value)
            if found is not None:
                return found
    return None


def fetch_all_madden_players(
    page_size: int = PAGE_SIZE,
    pause_seconds: float = 0.25,
    max_pages: int = 100,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_players: List[Dict[str, Any]] = []
    raw_pages: List[Any] = []
    total_expected: int | None = None

    for page_number in range(max_pages):
        offset = page_number * page_size
        print(f"Fetching EA Madden page {page_number + 1} (offset {offset})...")
        payload = fetch_madden_page(limit=page_size, offset=offset)
        raw_pages.append(payload)
        page_players = _find_player_list(payload)

        if page_number == 0:
            total_expected = _find_total(payload)
        if not page_players:
            break

        all_players.extend(page_players)
        if total_expected is not None and len(all_players) >= total_expected:
            break
        if len(page_players) < page_size:
            break
        time.sleep(max(float(pause_seconds), 0.0))

    deduplicated: List[Dict[str, Any]] = []
    seen = set()
    for player in all_players:
        player_id = player.get("id")
        token = f"id:{player_id}" if player_id is not None else json.dumps(
            player, sort_keys=True, default=str
        )
        if token not in seen:
            seen.add(token)
            deduplicated.append(player)

    if not deduplicated:
        raise RuntimeError("EA download produced zero Madden player records.")

    metadata = {
        "source": EA_API_BASE,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "total_expected": total_expected,
        "records_downloaded": len(deduplicated),
        "pages_downloaded": len(raw_pages),
        "ea_team_position_note": (
            "EA Madden 27 ratings payload currently returns null team/position "
            "for player records; Macabets enriches those fields from nflverse rosters."
        ),
        "raw_pages": raw_pages,
    }
    return deduplicated, metadata


def _flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_key = f"{prefix}_{key}" if prefix else str(key)
                walk(child, child_key)
        elif isinstance(value, list):
            # Keep scalar lists. Rich ability objects remain in raw JSON.
            if all(not isinstance(item, (dict, list)) for item in value):
                flat[prefix] = ", ".join(str(item) for item in value)
        else:
            flat[prefix] = value

    walk(record)
    return flat


def _lookup(record: Dict[str, Any], *aliases: str) -> Any:
    by_key = {str(k).casefold(): v for k, v in record.items()}
    for alias in aliases:
        key = alias.casefold()
        if key in by_key:
            return by_key[key]
        # EA's nested stats flatten to stats_<attribute>_value.
        stat_key = f"stats_{alias}_value".casefold()
        if stat_key in by_key:
            return by_key[stat_key]
    return None


def normalize_players(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for raw in records:
        flat = _flatten_record(raw)
        first_name = _lookup(flat, "firstName", "first_name", "firstname")
        last_name = _lookup(flat, "lastName", "last_name", "lastname")
        full_name = _lookup(flat, "player_name", "full_name", "display_name", "name")
        if not full_name:
            full_name = f"{first_name or ''} {last_name or ''}".strip()

        row = {
            "ea_player_id": _lookup(flat, "id"),
            "player_name": full_name,
            "first_name": first_name,
            "last_name": last_name,
            "birthdate": _lookup(flat, "birthdate", "birth_date", "date_of_birth"),
            "height": _lookup(flat, "height"),
            "weight": _lookup(flat, "weight"),
            "college": _lookup(flat, "college"),
            "age": _lookup(flat, "age"),
            "years_pro": _lookup(flat, "yearsPro", "years_pro"),
            # EA currently returns these as null. They remain optional here.
            "ea_team": _lookup(flat, "team", "team_name", "teamName"),
            "ea_position": _lookup(flat, "position", "position_name", "positionName"),
            "overall": _lookup(flat, "overallRating", "overall", "overall_rating"),
            "speed": _lookup(flat, "speed"),
            "strength": _lookup(flat, "strength"),
            "agility": _lookup(flat, "agility"),
            "acceleration": _lookup(flat, "acceleration"),
            "awareness": _lookup(flat, "awareness"),
            "injury": _lookup(flat, "injury"),
            "change_of_direction": _lookup(flat, "changeOfDirection", "change_of_direction"),
            "throw_power": _lookup(flat, "throwPower"),
            "throw_accuracy_short": _lookup(flat, "throwAccuracyShort"),
            "throw_accuracy_mid": _lookup(flat, "throwAccuracyMid"),
            "throw_accuracy_deep": _lookup(flat, "throwAccuracyDeep"),
            "throw_under_pressure": _lookup(flat, "throwUnderPressure"),
            "play_action": _lookup(flat, "playAction"),
            "catching": _lookup(flat, "catching"),
            "catch_in_traffic": _lookup(flat, "catchInTraffic"),
            "spectacular_catch": _lookup(flat, "spectacularCatch"),
            "release": _lookup(flat, "release"),
            "short_route_running": _lookup(flat, "shortRouteRunning"),
            "medium_route_running": _lookup(flat, "mediumRouteRunning"),
            "deep_route_running": _lookup(flat, "deepRouteRunning"),
            "pass_block": _lookup(flat, "passBlock"),
            "pass_block_power": _lookup(flat, "passBlockPower"),
            "pass_block_finesse": _lookup(flat, "passBlockFinesse"),
            "run_block": _lookup(flat, "runBlock"),
            "run_block_power": _lookup(flat, "runBlockPower"),
            "run_block_finesse": _lookup(flat, "runBlockFinesse"),
            "power_moves": _lookup(flat, "powerMoves"),
            "finesse_moves": _lookup(flat, "finesseMoves"),
            "block_shedding": _lookup(flat, "blockShedding"),
            "play_recognition": _lookup(flat, "playRecognition"),
            "pursuit": _lookup(flat, "pursuit"),
            "tackle": _lookup(flat, "tackle"),
            "man_coverage": _lookup(flat, "manCoverage"),
            "zone_coverage": _lookup(flat, "zoneCoverage"),
            "press": _lookup(flat, "press"),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("EA responded successfully, but no player records could be identified.")

    required = ("player_name", "overall")
    missing = [c for c in required if c not in frame.columns or frame[c].isna().all()]
    if missing:
        raise RuntimeError(
            "EA response is missing required Madden identity/rating fields: "
            + ", ".join(missing)
        )

    numeric_columns = [c for c in frame.columns if c not in {
        "player_name", "first_name", "last_name", "birthdate", "college",
        "ea_team", "ea_position"
    }]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["player_name"] = frame["player_name"].fillna("").astype(str).str.strip()
    frame = frame[frame["player_name"].ne("") & frame["overall"].notna()].copy()
    frame["overall"] = frame["overall"].clip(0, 99)
    frame = frame.drop_duplicates(subset=["ea_player_id"], keep="first")
    return frame.reset_index(drop=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def download_and_save_madden_ratings(
    csv_path: Path | str = DEFAULT_EA_CSV_PATH,
    raw_path: Path | str = DEFAULT_RAW_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    records, metadata = fetch_all_madden_players()
    players = normalize_players(records)

    csv_path = Path(csv_path)
    raw_path = Path(raw_path)
    metadata_path = Path(metadata_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    raw_pages = metadata.pop("raw_pages")
    metadata.update({
        "normalized_player_count": int(len(players)),
        "csv_path": str(csv_path),
    })

    # Only replace files after a complete fetch + normalization.
    csv_temp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    players.to_csv(csv_temp, index=False)
    csv_temp.replace(csv_path)
    _atomic_write_json(raw_path, raw_pages)
    _atomic_write_json(metadata_path, metadata)
    return players


if __name__ == "__main__":
    downloaded = download_and_save_madden_ratings()
    print(f"Downloaded and saved {len(downloaded)} Madden players.")
