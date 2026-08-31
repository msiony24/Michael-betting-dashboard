from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    import streamlit as st
except Exception:  # pragma: no cover - batch jobs do not require Streamlit.
    st = None


TABLE_NAME = "analysis_log"


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


def _json_safe(value: Any) -> Any:
    """Recursively replace NaN/Infinity with None.

    Python's json.dumps happily emits the literal tokens NaN/Infinity/-Infinity for
    those float values, but that is not valid JSON per the spec -- PostgREST's strict
    parser rejects the *entire* request body the moment one appears anywhere in it,
    surfacing as an opaque "Empty or invalid json" (PGRST102) with no indication of
    which field caused it. Scrubbing them here, once, for every save, prevents any
    single stray NaN deep in a nested snapshot from silently losing the whole record.
    """
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


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
        payload = json.dumps(_json_safe(body), default=str).encode("utf-8")

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
    return rows[0] if isinstance(rows, list) and rows else None


def list_analyses(
    limit: int = 500,
    *,
    sport: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "select": "*",
        "order": "event_date.desc,created_at.desc",
        "limit": max(1, min(int(limit), 5000)),
    }
    if sport:
        params["sport"] = f"eq.{sport}"
    if status:
        params["status"] = f"eq.{status}"
    rows = _request("GET", TABLE_NAME, params=params)
    return rows if isinstance(rows, list) else []


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
    return rows[0] if isinstance(rows, list) and rows else None


def delete_analysis(analysis_id: str) -> None:
    _request("DELETE", TABLE_NAME, params={"id": f"eq.{analysis_id}"})


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
