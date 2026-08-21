from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
import threading
import time
from typing import Any

try:
    import streamlit as st
except Exception:  # pragma: no cover - batch jobs do not require Streamlit.
    st = None


TABLE_NAME = "analysis_log"

# Lightweight analysis lists are requested repeatedly by Streamlit on every rerun.
# Keep a short in-process cache so Performance Center + Analysis Log can share one
# small response instead of repeatedly downloading the same archive.
_LIST_CACHE_TTL_SECONDS = 30.0
_LIST_CACHE: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]] = {}
_LIST_CACHE_LOCK = threading.Lock()


def _clear_list_cache() -> None:
    with _LIST_CACHE_LOCK:
        _LIST_CACHE.clear()


def _json_safe(value: Any) -> Any:
    """Convert nested payloads into strict JSON-safe values for PostgREST.

    Python's json module emits NaN/Infinity by default, but PostgREST rejects
    those tokens as invalid JSON.  Normalize them to null at the storage
    boundary so every sport benefits from the same protection.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # numpy/pandas scalar objects usually expose item(); handle them without
    # importing either heavy dependency into this persistence module.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except Exception:
            converted = value
        if converted is not value:
            return _json_safe(converted)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _streamlit_secret(name: str) -> str:
    if st is None:
        return ""
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def _config() -> tuple[str, str]:
    """Resolve Supabase credentials for Streamlit and trusted batch jobs.

    The service-role key is read from the environment only. This lets GitHub
    Actions settle results through RLS without ever placing a service-role key in
    Streamlit secrets or browser-facing application code.
    """
    base_url = (
        os.environ.get("SUPABASE_URL", "").strip()
        or _streamlit_secret("SUPABASE_URL")
    ).rstrip("/")

    api_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.environ.get("SUPABASE_KEY", "").strip()
        or os.environ.get("SUPABASE_ANON_KEY", "").strip()
        or _streamlit_secret("SUPABASE_KEY")
        or _streamlit_secret("SUPABASE_ANON_KEY")
    )
    return base_url, api_key


def is_configured() -> bool:
    base_url, api_key = _config()
    return bool(base_url and api_key)


def _request(
    method: str,
    path: str,
    *,
    body: Any = None,
    params: dict[str, Any] | None = None,
    prefer: str | None = None,
) -> Any:
    base_url, api_key = _config()
    if not base_url or not api_key:
        raise RuntimeError(
            "Analysis Log permanent storage is not configured. Add SUPABASE_URL "
            "and a Supabase key to Streamlit secrets or the batch-job environment."
        )

    query = urllib.parse.urlencode(params or {}, doseq=True)
    url = f"{base_url}/rest/v1/{path}"
    if query:
        url = f"{url}?{query}"

    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    elif method in {"POST", "PATCH"}:
        headers["Prefer"] = "return=representation"

    payload = None
    if body is not None:
        payload = json.dumps(_json_safe(body), default=str, allow_nan=False).encode("utf-8")

    request = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Analysis Log request failed ({exc.code}): {detail[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the Analysis Log database: {exc.reason}"
        ) from exc

    return json.loads(raw.decode("utf-8")) if raw else None


def create_analysis(record: dict[str, Any]) -> dict[str, Any] | None:
    rows = _request("POST", TABLE_NAME, body=record)
    _clear_list_cache()
    return rows[0] if isinstance(rows, list) and rows else None


def list_analyses(
    limit: int = 500,
    *,
    sport: str | None = None,
    status: str | None = None,
    select: str = "*",
    cache_ttl: float = 0.0,
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 5000))
    cache_key = (safe_limit, sport or "", status or "", select)
    ttl = max(0.0, float(cache_ttl or 0.0))
    if ttl:
        now = time.monotonic()
        with _LIST_CACHE_LOCK:
            cached = _LIST_CACHE.get(cache_key)
            if cached and now - cached[0] < ttl:
                # Return a shallow copy so callers cannot mutate the cached list.
                return list(cached[1])

    params: dict[str, Any] = {
        "select": select,
        "order": "event_date.desc,created_at.desc",
        "limit": safe_limit,
    }
    if sport:
        params["sport"] = f"eq.{sport}"
    if status:
        params["status"] = f"eq.{status}"
    rows = _request("GET", TABLE_NAME, params=params)
    result = rows if isinstance(rows, list) else []
    if ttl:
        with _LIST_CACHE_LOCK:
            _LIST_CACHE[cache_key] = (time.monotonic(), list(result))
    return result


def get_analysis(analysis_id: str) -> dict[str, Any] | None:
    """Fetch one full analysis only when its frozen snapshots are actually needed."""
    rows = _request(
        "GET",
        TABLE_NAME,
        params={"id": f"eq.{analysis_id}", "select": "*", "limit": 1},
    )
    return rows[0] if isinstance(rows, list) and rows else None


def update_analysis(
    analysis_id: str,
    changes: dict[str, Any],
) -> dict[str, Any] | None:
    rows = _request(
        "PATCH",
        TABLE_NAME,
        body=changes,
        params={"id": f"eq.{analysis_id}", "select": "*"},
    )
    _clear_list_cache()
    return rows[0] if isinstance(rows, list) and rows else None


def delete_analysis(analysis_id: str) -> None:
    _request("DELETE", TABLE_NAME, params={"id": f"eq.{analysis_id}"})
    _clear_list_cache()


def list_table_records(
    table_name: str,
    *,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Read a settlement/support table through the same Supabase connection."""
    rows = _request("GET", str(table_name), params=params or {"select": "*"})
    return rows if isinstance(rows, list) else []


def insert_table_records(
    table_name: str,
    records: dict[str, Any] | list[dict[str, Any]],
    *,
    ignore_duplicates: bool = False,
) -> list[dict[str, Any]]:
    """Insert one or more support-table records and return inserted rows."""
    prefer = "return=representation"
    if ignore_duplicates:
        prefer += ",resolution=ignore-duplicates"
    rows = _request(
        "POST",
        str(table_name),
        body=records,
        prefer=prefer,
    )
    return rows if isinstance(rows, list) else []
