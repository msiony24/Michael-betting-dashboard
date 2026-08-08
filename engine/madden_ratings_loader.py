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

DEFAULT_CSV_PATH = DATA_DIR / "madden_27_players.csv"
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
                "User-Agent": "Mozilla/5.0 Macabets/0.61",
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
            # Retry server/rate-limit errors; fail immediately on permanent 4xx errors.
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
        token = json.dumps(player, sort_keys=True, default=str)
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
        "raw_pages": raw_pages,
    }
    return deduplicated, metadata


def _first_value(record: Dict[str, Any], candidates: Iterable[str]) -> Any:
    lowered = {str(key).casefold(): value for key, value in record.items()}
    for candidate in candidates:
        if candidate.casefold() in lowered:
            return lowered[candidate.casefold()]
    return None


def _flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_key = f"{prefix}_{key}" if prefix else str(key)
                walk(child, child_key)
        elif isinstance(value, list):
            if all(not isinstance(item, (dict, list)) for item in value):
                flat[prefix] = ", ".join(str(item) for item in value)
        else:
            flat[prefix] = value
    walk(record)
    return flat


def normalize_players(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for raw_record in records:
        record = _flatten_record(raw_record)
        player_name = _first_value(
            record, ("player_name", "player", "full_name", "display_name", "name")
        )
        first_name = _first_value(record, ("first_name", "firstname", "firstName"))
        last_name = _first_value(record, ("last_name", "lastname", "lastName"))
        if not player_name and (first_name or last_name):
            player_name = f"{first_name or ''} {last_name or ''}".strip()

        row = {
            "player_name": player_name,
            "team": _first_value(record, ("team", "team_name", "teamName", "club", "team_label")),
            "position": _first_value(record, ("position", "pos", "position_name", "positionName")),
            "overall": _first_value(record, ("overall", "ovr", "overall_rating", "overallRating", "rating")),
            "speed": _first_value(record, ("speed", "spd")),
            "strength": _first_value(record, ("strength", "str")),
            "agility": _first_value(record, ("agility", "agi")),
            "change_of_direction": _first_value(record, ("change_of_direction", "changeOfDirection", "cod")),
            "injury": _first_value(record, ("injury", "inj")),
            "awareness": _first_value(record, ("awareness", "awr")),
        }
        for key, value in record.items():
            clean_key = str(key).strip().replace(" ", "_").replace("-", "_")
            if clean_key and clean_key not in row:
                row[clean_key] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("EA responded successfully, but no player records could be identified.")

    for column in ("overall", "speed", "strength", "agility", "change_of_direction", "injury", "awareness"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    required = ("player_name", "team", "position", "overall")
    missing = [c for c in required if c not in frame.columns or frame[c].isna().all()]
    if missing:
        raise RuntimeError(
            "The EA response format was recognized incompletely. Missing usable fields: "
            + ", ".join(missing)
        )

    frame["player_name"] = frame["player_name"].astype(str).str.strip()
    frame["team"] = frame["team"].astype(str).str.strip()
    frame["position"] = frame["position"].astype(str).str.strip().str.upper()
    frame = frame[
        frame["player_name"].ne("")
        & frame["team"].ne("")
        & frame["position"].ne("")
        & frame["overall"].notna()
    ].copy()
    frame = frame.drop_duplicates(subset=["player_name", "team", "position"], keep="first")
    return frame.reset_index(drop=True)


def _atomic_write_text(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def download_and_save_madden_ratings(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    raw_path: Path | str = DEFAULT_RAW_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    # Fetch and normalize completely BEFORE touching the last known-good files.
    records, metadata = fetch_all_madden_players()
    players = normalize_players(records)

    csv_path = Path(csv_path)
    raw_path = Path(raw_path)
    metadata_path = Path(metadata_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    raw_pages = metadata.pop("raw_pages")
    metadata.update(
        {
            "normalized_player_count": int(len(players)),
            "recognized_team_count": int(players["team"].nunique()),
            "recognized_position_count": int(players["position"].nunique()),
            "csv_path": str(csv_path),
        }
    )

    # Atomic replacement: failed downloads never overwrite the previous good database.
    csv_temp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    players.to_csv(csv_temp, index=False)
    csv_temp.replace(csv_path)
    _atomic_write_text(raw_path, json.dumps(raw_pages, indent=2))
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))

    return players


if __name__ == "__main__":
    downloaded = download_and_save_madden_ratings()
    print(f"Downloaded and saved {len(downloaded)} Madden players.")
    print(f"CSV: {DEFAULT_CSV_PATH}")
    print(f"Metadata: {DEFAULT_METADATA_PATH}")
