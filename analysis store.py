"""Permanent universal Analysis Log storage for Macabets using Supabase REST."""
from __future__ import annotations

from datetime import date, datetime
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import streamlit as st

TABLE = "analysis_log"


def _config() -> tuple[str, str]:
    try:
        url = str(st.secrets.get("SUPABASE_URL", "")).strip().rstrip("/")
        key = str(st.secrets.get("SUPABASE_KEY", "")).strip()
    except Exception:
        return "", ""
    return url, key


def is_configured() -> bool:
    url, key = _config()
    return bool(url and key)


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy/date values into JSON-safe Python values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _request(method: str, query: str = "", payload: Any = None, prefer: str | None = None) -> Any:
    url, key = _config()
    if not url or not key:
        raise RuntimeError("Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets.")

    endpoint = f"{url}/rest/v1/{TABLE}"
    if query:
        endpoint += "?" + query
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    body = None if payload is None else json.dumps(_json_safe(payload)).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Analysis Log database error ({exc.code}): {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the Analysis Log database: {exc.reason}") from exc


def create_analysis(record: dict) -> dict | None:
    rows = _request("POST", payload=record, prefer="return=representation,resolution=ignore-duplicates")
    return rows[0] if isinstance(rows, list) and rows else None


def list_analyses(limit: int = 500, sport: str | None = None, status: str | None = None) -> list[dict]:
    params = ["select=*", "order=created_at.desc", f"limit={int(limit)}"]
    if sport and sport != "All":
        params.append("sport=eq." + urllib.parse.quote(sport, safe=""))
    if status and status != "All":
        params.append("status=eq." + urllib.parse.quote(status, safe=""))
    rows = _request("GET", "&".join(params))
    return rows if isinstance(rows, list) else []


def update_analysis(analysis_id: str, changes: dict) -> dict | None:
    query = "id=eq." + urllib.parse.quote(str(analysis_id), safe="")
    rows = _request("PATCH", query=query, payload=changes, prefer="return=representation")
    return rows[0] if isinstance(rows, list) and rows else None


def delete_analysis(analysis_id: str) -> None:
    query = "id=eq." + urllib.parse.quote(str(analysis_id), safe="")
    _request("DELETE", query=query, prefer="return=minimal")
