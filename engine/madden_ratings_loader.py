from __future__ import annotations

import json
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

DEFAULT_CSV_PATH = DATA_DIR / "madden_27_players.csv"
DEFAULT_RAW_PATH = DATA_DIR / "madden_27_raw.json"
DEFAULT_METADATA_PATH = DATA_DIR / "madden_27_metadata.json"


def _request_json(url: str, timeout: int = 30) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 Macabets/0.61",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.ea.com/games/madden-nfl/ratings",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"EA ratings request returned HTTP {exc.code}: {detail[:300]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to the EA ratings service: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("EA returned a response that was not valid JSON.") from exc


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
    pause_seconds: float = 0.15,
    max_pages: int = 100,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_players: List[Dict[str, Any]] = []
    raw_pages: List[Any] = []
    total_expected: int | None = None

    for page_number in range(max_pages):
        offset = page_number * page_size
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

    metadata = {
        "source": EA_API_BASE,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "total_expected": total_expected,
        "records_downloaded": len(deduplicated),
        "pages_downloaded": len(raw_pages),
        "raw_pages": raw_pages,
    }
    return deduplicated, metadata


def _normalize_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _first_value(record: Dict[str, Any], candidates: Iterable[str]) -> Any:
    """
    Resolve fields defensively.

    EA sometimes nests fields such as team/position under objects, which become
    flattened names such as `team_label`, `player_team_shortLabel`, or
    `position_fullName`. Exact matches are preferred, followed by safe suffix
    matches against normalized keys.
    """
    normalized_record = {
        _normalize_key(key): value
        for key, value in record.items()
        if value not in (None, "")
    }
    normalized_candidates = [_normalize_key(candidate) for candidate in candidates]

    for candidate in normalized_candidates:
        if candidate in normalized_record:
            return normalized_record[candidate]

    for candidate in normalized_candidates:
        for key, value in normalized_record.items():
            if key.endswith(candidate):
                return value

    return None


def _entity_label(record: Dict[str, Any], entity: str) -> Any:
    """Find a human-readable team or position value in EA's nested schema."""
    entity_key = _normalize_key(entity)
    preferred_terms = (
        "shortlabel",
        "label",
        "displayname",
        "fullname",
        "name",
        "abbr",
        "abbreviation",
        "slug",
    )

    candidates = []
    for raw_key, value in record.items():
        key = _normalize_key(raw_key)
        if entity_key not in key or value in (None, ""):
            continue
        if isinstance(value, (dict, list)):
            continue

        priority = 99
        for index, term in enumerate(preferred_terms):
            if key.endswith(term) or term in key:
                priority = index
                break

        # Avoid IDs and image/logo URLs unless no descriptive field exists.
        if key.endswith("id") or "logo" in key or "image" in key:
            priority += 50

        candidates.append((priority, len(key), value))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


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
            record,
            ("player_name", "player", "full_name", "display_name", "name"),
        )
        first_name = _first_value(record, ("first_name", "firstname", "firstName"))
        last_name = _first_value(record, ("last_name", "lastname", "lastName"))
        if not player_name and (first_name or last_name):
            player_name = f"{first_name or ''} {last_name or ''}".strip()

        row = {
            "player_name": player_name,
            "team": (
                _first_value(
                    record,
                    (
                        "team",
                        "team_name",
                        "teamName",
                        "club",
                        "team_label",
                        "team_shortLabel",
                        "team_displayName",
                        "team_abbreviation",
                    ),
                )
                or _entity_label(record, "team")
            ),
            "position": (
                _first_value(
                    record,
                    (
                        "position",
                        "pos",
                        "position_name",
                        "positionName",
                        "position_label",
                        "position_shortLabel",
                        "position_abbreviation",
                    ),
                )
                or _entity_label(record, "position")
            ),
            "overall": _first_value(
                record,
                ("overall", "ovr", "overall_rating", "overallRating", "rating"),
            ),
            "speed": _first_value(record, ("speed", "spd")),
            "strength": _first_value(record, ("strength", "str")),
            "agility": _first_value(record, ("agility", "agi")),
            "change_of_direction": _first_value(
                record,
                ("change_of_direction", "changeOfDirection", "cod"),
            ),
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
        raise RuntimeError(
            "EA responded successfully, but no player records could be identified."
        )

    for column in (
        "overall",
        "speed",
        "strength",
        "agility",
        "change_of_direction",
        "injury",
        "awareness",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    required = ("player_name", "team", "position", "overall")
    missing = [
        column
        for column in required
        if column not in frame.columns or frame[column].isna().all()
    ]
    if missing:
        sample_keys = sorted(
            {
                str(key)
                for raw_record in records[:3]
                for key in _flatten_record(raw_record).keys()
            }
        )
        raise RuntimeError(
            "The EA response format was recognized incompletely. Missing usable fields: "
            + ", ".join(missing)
            + ". Sample EA fields: "
            + ", ".join(sample_keys[:80])
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

    frame = frame.drop_duplicates(
        subset=["player_name", "team", "position"],
        keep="first",
    )
    return frame.reset_index(drop=True)


def download_and_save_madden_ratings(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    raw_path: Path | str = DEFAULT_RAW_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    records, metadata = fetch_all_madden_players()
    players = normalize_players(records)

    csv_path = Path(csv_path)
    raw_path = Path(raw_path)
    metadata_path = Path(metadata_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    players.to_csv(csv_path, index=False)

    raw_pages = metadata.pop("raw_pages")
    with raw_path.open("w", encoding="utf-8") as file:
        json.dump(raw_pages, file, indent=2)

    metadata.update(
        {
            "normalized_player_count": int(len(players)),
            "recognized_team_count": int(players["team"].nunique()),
            "recognized_position_count": int(players["position"].nunique()),
            "csv_path": str(csv_path),
        }
    )

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return players


if __name__ == "__main__":
    downloaded = download_and_save_madden_ratings()
    print(f"Downloaded and saved {len(downloaded)} Madden players.")
    print(f"CSV: {DEFAULT_CSV_PATH}")
    print(f"Metadata: {DEFAULT_METADATA_PATH}")
