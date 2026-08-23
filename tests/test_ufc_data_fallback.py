from __future__ import annotations

import pandas as pd
import pytest

import update_ufc_data as ufc_update
from engine.ufc_data import FetchConfig, UFCStatsError


def _enriched_cache() -> pd.DataFrame:
    return pd.DataFrame([{
        "fighter": "Alpha", "opponent": "Beta", "fight_url": "f1", "event_date": "2025-01-01",
        "sig_str_landed": 10, "sig_str_attempted": 20, "td_landed": 1, "td_attempted": 2,
        "control_seconds": 60, "r1_sig_str_landed": 5, "r2_sig_str_landed": 5,
    }])


def _compact_cache() -> pd.DataFrame:
    return pd.DataFrame([{"fighter": "Alpha", "opponent": "Beta", "fight_url": "f1", "event_date": "2025-01-01"}])


def _mirror_rows() -> pd.DataFrame:
    return pd.DataFrame([{"fighter": "Gamma", "opponent": "Delta", "fight_url": "mirror:1", "event_date": "2025-06-01"}])


def test_live_fetch_success_returns_fresh(monkeypatch):
    live_events = pd.DataFrame([{"event_url": "e1", "event_date": "2025-01-01"}])
    live_fights = pd.DataFrame([{"fighter": "Alpha", "event_date": "2025-01-01"}])
    monkeypatch.setattr(ufc_update, "fetch_completed_events", lambda config: live_events)
    monkeypatch.setattr(ufc_update, "fetch_fight_history", lambda events, since, config: live_fights)

    fights, mode, error = ufc_update._get_fight_history(FetchConfig())
    assert mode == "fresh"
    assert error == ""
    assert len(fights) == 1


def test_live_fetch_raises_empty_events_falls_back_to_enriched_cache(monkeypatch):
    monkeypatch.setattr(ufc_update, "fetch_completed_events", lambda config: pd.DataFrame())
    monkeypatch.setattr(ufc_update, "_load_cached_fights", lambda: _enriched_cache())

    fights, mode, error = ufc_update._get_fight_history(FetchConfig())
    assert mode == "cached_fallback"
    assert "UFCStatsError" in error
    assert len(fights) == 1


def test_live_fetch_network_error_falls_back_to_enriched_cache(monkeypatch):
    def _raise(config):
        raise UFCStatsError("Unable to fetch UFCStats after 4 attempts")
    monkeypatch.setattr(ufc_update, "fetch_completed_events", _raise)
    monkeypatch.setattr(ufc_update, "_load_cached_fights", lambda: _enriched_cache())

    fights, mode, error = ufc_update._get_fight_history(FetchConfig())
    assert mode == "cached_fallback"
    assert "Unable to fetch UFCStats" in error


def test_live_fails_no_cache_falls_back_to_mirror(monkeypatch):
    monkeypatch.setattr(ufc_update, "fetch_completed_events", lambda config: pd.DataFrame())
    monkeypatch.setattr(ufc_update, "_load_cached_fights", lambda: pd.DataFrame())
    monkeypatch.setattr(ufc_update, "_load_github_mirror_history", lambda: _mirror_rows())

    fights, mode, error = ufc_update._get_fight_history(FetchConfig())
    assert mode == "mirror_bootstrap"
    assert len(fights) == 1
    assert fights.iloc[0]["fighter"] == "Gamma"


def test_live_fails_compact_cache_prefers_mirror_over_compact_cache(monkeypatch):
    # A compact (older-schema) cache should not be silently accepted if a
    # richer mirror bootstrap is available -- the mirror path enriches
    # detailed/round-level stats the compact cache lacks.
    monkeypatch.setattr(ufc_update, "fetch_completed_events", lambda config: pd.DataFrame())
    monkeypatch.setattr(ufc_update, "_load_cached_fights", lambda: _compact_cache())
    monkeypatch.setattr(ufc_update, "_load_github_mirror_history", lambda: _mirror_rows())

    fights, mode, error = ufc_update._get_fight_history(FetchConfig())
    assert mode == "mirror_bootstrap"


def test_live_fails_mirror_fails_falls_back_to_whatever_cache_exists(monkeypatch):
    def _raise_mirror():
        raise RuntimeError("mirror unreachable")
    monkeypatch.setattr(ufc_update, "fetch_completed_events", lambda config: pd.DataFrame())
    monkeypatch.setattr(ufc_update, "_load_cached_fights", lambda: _compact_cache())
    monkeypatch.setattr(ufc_update, "_load_github_mirror_history", _raise_mirror)

    fights, mode, error = ufc_update._get_fight_history(FetchConfig())
    assert mode == "cached_fallback"
    assert len(fights) == 1


def test_live_fails_mirror_returns_empty_falls_back_to_cache(monkeypatch):
    monkeypatch.setattr(ufc_update, "fetch_completed_events", lambda config: pd.DataFrame())
    monkeypatch.setattr(ufc_update, "_load_cached_fights", lambda: _compact_cache())
    monkeypatch.setattr(ufc_update, "_load_github_mirror_history", lambda: pd.DataFrame())

    fights, mode, error = ufc_update._get_fight_history(FetchConfig())
    assert mode == "cached_fallback"


def test_everything_fails_raises_with_both_errors_included(monkeypatch):
    def _raise_mirror():
        raise RuntimeError("mirror unreachable")
    monkeypatch.setattr(ufc_update, "fetch_completed_events", lambda config: pd.DataFrame())
    monkeypatch.setattr(ufc_update, "_load_cached_fights", lambda: pd.DataFrame())
    monkeypatch.setattr(ufc_update, "_load_github_mirror_history", _raise_mirror)

    with pytest.raises(RuntimeError) as excinfo:
        ufc_update._get_fight_history(FetchConfig())
    message = str(excinfo.value)
    assert "Live error" in message
    assert "Mirror error" in message


def test_everything_empty_no_exceptions_still_raises(monkeypatch):
    # Live returns nothing, cache is empty, and the mirror returns an empty
    # frame without raising -- must still fail loudly rather than silently
    # produce an empty ratings run.
    monkeypatch.setattr(ufc_update, "fetch_completed_events", lambda config: pd.DataFrame())
    monkeypatch.setattr(ufc_update, "_load_cached_fights", lambda: pd.DataFrame())
    monkeypatch.setattr(ufc_update, "_load_github_mirror_history", lambda: pd.DataFrame())

    with pytest.raises(RuntimeError):
        ufc_update._get_fight_history(FetchConfig())
