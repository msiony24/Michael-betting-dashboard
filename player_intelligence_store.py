from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from .player_profiles import PlayerProfile, canonical_player_key

DEFAULT_PLAYER_INTELLIGENCE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "player_intelligence_atp.json"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_player_intelligence(
    profiles: Iterable[PlayerProfile],
    *,
    destination: str | Path = DEFAULT_PLAYER_INTELLIGENCE_PATH,
    as_of_date: date | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically persist normalized player profiles for fast daily reuse."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        key = canonical_player_key(profile.requested_name or profile.historical_name)
        if not key:
            continue
        normalized[key] = asdict(profile)

    payload = {
        "schema_version": 1,
        "tour": "ATP",
        "as_of_date": (as_of_date or date.today()).isoformat(),
        "generated_at": utc_now_iso(),
        "profile_count": len(normalized),
        "metadata": metadata or {},
        "profiles": normalized,
    }

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def read_player_intelligence(
    source: str | Path = DEFAULT_PLAYER_INTELLIGENCE_PATH,
) -> dict[str, Any]:
    path = Path(source)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), dict):
        return {}
    return payload


def get_stored_profile(
    player_name: str,
    *,
    event_date: date,
    source: str | Path = DEFAULT_PLAYER_INTELLIGENCE_PATH,
) -> PlayerProfile | None:
    """Return a profile only when the snapshot cannot leak future information."""
    payload = read_player_intelligence(source)
    if not payload:
        return None

    try:
        snapshot_date = date.fromisoformat(str(payload.get("as_of_date", "")))
    except ValueError:
        return None
    if snapshot_date > event_date:
        return None

    raw = payload["profiles"].get(canonical_player_key(player_name))
    if not isinstance(raw, dict):
        return None

    allowed = set(PlayerProfile.__dataclass_fields__)
    clean = {key: value for key, value in raw.items() if key in allowed}
    try:
        profile = PlayerProfile(**clean)
    except TypeError:
        return None

    if "player_intelligence_store" not in profile.data_sources:
        profile.data_sources.append("player_intelligence_store")
    return profile
