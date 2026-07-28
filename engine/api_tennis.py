from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


API_BASE_URL = "https://api.api-tennis.com/tennis/"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "api_tennis_cache"


class APITennisError(RuntimeError):
    """Raised when API-Tennis cannot return a valid response."""


@dataclass(frozen=True)
class APITennisResponse:
    result: list[dict[str, Any]]
    source: str
    fetched_at: str
    cache_key: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_streamlit_secret(name: str) -> str:
    try:
        import streamlit as st  # Optional dependency.
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def get_api_key(explicit_key: str | None = None) -> str:
    """Resolve the API key without ever logging or returning it in errors."""
    if explicit_key:
        return str(explicit_key).strip()

    for variable in ("API_TENNIS_KEY", "API_TENNIS_API_KEY"):
        value = os.environ.get(variable, "").strip()
        if value:
            return value

    for secret_name in ("API_TENNIS_KEY", "API_TENNIS_API_KEY"):
        value = _read_streamlit_secret(secret_name)
        if value:
            return value

    return ""


def _cache_key(method: str, params: dict[str, Any]) -> str:
    normalized = json.dumps(
        {"method": method, "params": params},
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _read_cache(path: Path, max_age: timedelta | None) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)

        if max_age is not None and _utc_now() - fetched_at > max_age:
            return None

        if not isinstance(payload.get("result"), list):
            return None
        return payload
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, method: str, params: dict[str, Any], result: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "params": params,
        "fetched_at": _utc_now().isoformat(),
        "result": result,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class APITennisClient:
    """Small resilient client for API-Tennis with disk caching and safe fallbacks."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        timeout: int = 30,
        retries: int = 2,
        user_agent: str = "Macabets API-Tennis client/0.54",
    ) -> None:
        self.api_key = get_api_key(api_key)
        self.cache_dir = Path(cache_dir)
        self.timeout = max(5, int(timeout))
        self.retries = max(0, int(retries))
        self.user_agent = user_agent

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def request(
        self,
        method: str,
        *,
        cache_ttl: timedelta | None = timedelta(hours=12),
        allow_stale_cache: bool = True,
        force_refresh: bool = False,
        **params: Any,
    ) -> APITennisResponse:
        clean_method = str(method).strip()
        clean_params = {
            key: value
            for key, value in params.items()
            if value is not None and str(value).strip() != ""
        }
        key = _cache_key(clean_method, clean_params)
        path = _cache_path(self.cache_dir, key)

        if not force_refresh:
            cached = _read_cache(path, cache_ttl)
            if cached is not None:
                return APITennisResponse(
                    result=cached["result"],
                    source="cache",
                    fetched_at=cached["fetched_at"],
                    cache_key=key,
                )

        if not self.configured:
            stale = _read_cache(path, None) if allow_stale_cache else None
            if stale is not None:
                return APITennisResponse(
                    result=stale["result"],
                    source="stale_cache",
                    fetched_at=stale["fetched_at"],
                    cache_key=key,
                )
            raise APITennisError(
                "API-Tennis is not configured. Add API_TENNIS_KEY to Streamlit secrets "
                "or the environment."
            )

        query = urllib.parse.urlencode(
            {"method": clean_method, "APIkey": self.api_key, **clean_params}
        )
        url = f"{API_BASE_URL}?{query}"
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                url,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                payload = json.loads(raw)

                if int(payload.get("success", 0)) != 1:
                    raise APITennisError(
                        f"{clean_method} failed: {payload.get('error', 'Unknown API error')}"
                    )

                result = payload.get("result", [])
                if result is None:
                    result = []
                if not isinstance(result, list):
                    raise APITennisError(
                        f"{clean_method} returned an unexpected result format."
                    )

                fetched_at = _utc_now().isoformat()
                _write_cache(path, clean_method, clean_params, result)
                return APITennisResponse(
                    result=result,
                    source="live",
                    fetched_at=fetched_at,
                    cache_key=key,
                )

            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                APITennisError,
            ) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 4))

        stale = _read_cache(path, None) if allow_stale_cache else None
        if stale is not None:
            return APITennisResponse(
                result=stale["result"],
                source="stale_cache",
                fetched_at=stale["fetched_at"],
                cache_key=key,
            )

        raise APITennisError(
            f"API-Tennis request failed after {self.retries + 1} attempt(s): {last_error}"
        ) from last_error

    def get_standings(
        self,
        event_type: str = "ATP",
        *,
        force_refresh: bool = False,
    ) -> APITennisResponse:
        return self.request(
            "get_standings",
            event_type=event_type,
            cache_ttl=timedelta(hours=12),
            force_refresh=force_refresh,
        )

    def get_fixtures(
        self,
        date_start: date | str,
        date_stop: date | str,
        *,
        timezone_name: str = "America/New_York",
        force_refresh: bool = False,
    ) -> APITennisResponse:
        return self.request(
            "get_fixtures",
            date_start=str(date_start),
            date_stop=str(date_stop),
            timezone=timezone_name,
            cache_ttl=timedelta(minutes=30),
            force_refresh=force_refresh,
        )

    def get_player(self, player_key: int | str, *, force_refresh: bool = False) -> APITennisResponse:
        return self.request(
            "get_players",
            player_key=player_key,
            cache_ttl=timedelta(days=1),
            force_refresh=force_refresh,
        )


def safe_request(
    method: str,
    *,
    client: APITennisClient | None = None,
    **params: Any,
) -> dict[str, Any]:
    """Return a non-throwing result suitable for Streamlit UI code."""
    active_client = client or APITennisClient()
    try:
        response = active_client.request(method, **params)
        return {
            "ok": True,
            "result": response.result,
            "source": response.source,
            "fetched_at": response.fetched_at,
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "result": [],
            "source": "unavailable",
            "fetched_at": None,
            "error": str(exc),
        }
