from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from engine.api_tennis import (
    APITennisClient,
    APITennisError,
    _cache_key,
    _read_cache,
    _write_cache,
    get_api_key,
)


# --- get_api_key resolution order ------------------------------------------

def test_get_api_key_prefers_explicit_key(monkeypatch):
    monkeypatch.setenv("API_TENNIS_KEY", "env-key")
    assert get_api_key("explicit-key") == "explicit-key"


def test_get_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.delenv("API_TENNIS_KEY", raising=False)
    monkeypatch.delenv("API_TENNIS_API_KEY", raising=False)
    monkeypatch.setenv("API_TENNIS_KEY", "env-key")
    assert get_api_key() == "env-key"


def test_get_api_key_returns_empty_string_when_nothing_configured(monkeypatch):
    monkeypatch.delenv("API_TENNIS_KEY", raising=False)
    monkeypatch.delenv("API_TENNIS_API_KEY", raising=False)
    assert get_api_key() == ""


# --- cache key determinism --------------------------------------------------

def test_cache_key_is_deterministic_for_same_inputs():
    key1 = _cache_key("get_standings", {"event_type": "ATP"})
    key2 = _cache_key("get_standings", {"event_type": "ATP"})
    assert key1 == key2


def test_cache_key_ignores_param_ordering():
    key1 = _cache_key("get_fixtures", {"a": 1, "b": 2})
    key2 = _cache_key("get_fixtures", {"b": 2, "a": 1})
    assert key1 == key2


def test_cache_key_differs_for_different_params():
    key1 = _cache_key("get_standings", {"event_type": "ATP"})
    key2 = _cache_key("get_standings", {"event_type": "WTA"})
    assert key1 != key2


# --- cache read/write round trip -------------------------------------------

def test_write_then_read_cache_round_trip(tmp_path: Path):
    path = tmp_path / "entry.json"
    _write_cache(path, "get_standings", {"event_type": "ATP"}, [{"place": 1}])
    cached = _read_cache(path, max_age=timedelta(hours=1))
    assert cached is not None
    assert cached["result"] == [{"place": 1}]


def test_read_cache_returns_none_when_expired(tmp_path: Path):
    path = tmp_path / "entry.json"
    _write_cache(path, "get_standings", {}, [{"place": 1}])
    # A negative max_age means "already expired" regardless of write time.
    cached = _read_cache(path, max_age=timedelta(seconds=-1))
    assert cached is None


def test_read_cache_returns_none_for_missing_file(tmp_path: Path):
    assert _read_cache(tmp_path / "missing.json", max_age=None) is None


def test_read_cache_returns_none_for_malformed_json(tmp_path: Path):
    path = tmp_path / "entry.json"
    path.write_text("{not valid", encoding="utf-8")
    assert _read_cache(path, max_age=None) is None


# --- client behavior without a live network call ---------------------------

def test_unconfigured_client_uses_stale_cache_when_available(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("API_TENNIS_KEY", raising=False)
    monkeypatch.delenv("API_TENNIS_API_KEY", raising=False)
    client = APITennisClient(api_key="", cache_dir=tmp_path)
    assert client.configured is False

    key = _cache_key("get_standings", {"event_type": "ATP"})
    path = tmp_path / f"{key}.json"
    _write_cache(path, "get_standings", {"event_type": "ATP"}, [{"place": 1}])

    response = client.request("get_standings", event_type="ATP", cache_ttl=timedelta(seconds=-1))
    assert response.source == "stale_cache"
    assert response.result == [{"place": 1}]


def test_unconfigured_client_with_no_cache_raises_clear_error(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("API_TENNIS_KEY", raising=False)
    monkeypatch.delenv("API_TENNIS_API_KEY", raising=False)
    client = APITennisClient(api_key="", cache_dir=tmp_path)

    with pytest.raises(APITennisError):
        client.request("get_standings", event_type="ATP")


def test_configured_client_uses_fresh_cache_without_network(tmp_path: Path):
    client = APITennisClient(api_key="test-key", cache_dir=tmp_path)
    key = _cache_key("get_standings", {"event_type": "ATP"})
    path = tmp_path / f"{key}.json"
    _write_cache(path, "get_standings", {"event_type": "ATP"}, [{"place": 1}])

    response = client.request("get_standings", event_type="ATP", cache_ttl=timedelta(hours=12))
    assert response.source == "cache"
    assert response.result == [{"place": 1}]


def test_write_then_read_cache_round_trip_supports_odds_mapping(tmp_path: Path):
    path = tmp_path / "odds.json"
    result = {"159923": {"Home/Away": {"Home": {"bet365": "1.50"}, "Away": {"bet365": "2.75"}}}}
    _write_cache(path, "get_odds", {"date_start": "2026-08-24"}, result)
    cached = _read_cache(path, max_age=timedelta(hours=1))
    assert cached is not None
    assert cached["result"] == result


def test_normalize_prematch_odds_extracts_only_match_winner_market():
    from engine.api_tennis import normalize_prematch_odds

    raw = {
        "159923": {
            "Home/Away": {
                "Home": {"bet365": "1.50", "bwin": "1.48"},
                "Away": {"bet365": "2.75", "bwin": "2.80"},
                "6:4": {"bet365": "9.50"},
            },
            "Home/Away (1st Set)": {
                "Home": {"bet365": "1.60"},
                "Away": {"bet365": "2.40"},
            },
        }
    }
    parsed = normalize_prematch_odds(raw)
    assert parsed["159923"]["home_odds"] == -200
    assert parsed["159923"]["away_odds"] == 180
    assert parsed["159923"]["home_book"] == "bet365"
    assert parsed["159923"]["away_book"] == "bwin"


def test_normalize_prematch_odds_ignores_non_moneyline_markets():
    from engine.api_tennis import normalize_prematch_odds

    raw = {"1": {"Set Betting": {"2:0": {"bet365": "2.10"}}}}
    assert normalize_prematch_odds(raw) == {}
