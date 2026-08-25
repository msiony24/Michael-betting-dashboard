from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from settle_results import (
    nested_link,
    parse_event_date,
    parse_iso_datetime,
    settle_odds_api_score,
    settle_tennis,
)
from engine.settlement_providers import ProviderError


def test_parse_event_date_from_event_date_field():
    assert parse_event_date({"event_date": "2026-09-10T12:00:00"}) == date(2026, 9, 10)


def test_parse_event_date_falls_back_to_created_at():
    assert parse_event_date({"created_at": "2026-09-10T12:00:00"}) == date(2026, 9, 10)


def test_parse_event_date_missing_returns_none():
    assert parse_event_date({}) is None


def test_parse_event_date_invalid_returns_none():
    assert parse_event_date({"event_date": "not a date"}) is None


def test_parse_iso_datetime_handles_z_suffix():
    result = parse_iso_datetime("2026-09-10T12:00:00Z")
    assert result == datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_iso_datetime_naive_datetime_assumed_utc():
    result = parse_iso_datetime("2026-09-10T12:00:00")
    assert result.tzinfo == timezone.utc


def test_parse_iso_datetime_missing_or_invalid_returns_none():
    assert parse_iso_datetime(None) is None
    assert parse_iso_datetime("") is None
    assert parse_iso_datetime("garbage") is None


def test_nested_link_extracts_settlement_link_dict():
    row = {"input_snapshot": {"settlement_link": {"event_id": "abc"}}}
    assert nested_link(row) == {"event_id": "abc"}


def test_nested_link_falls_back_to_external_event():
    row = {"input_snapshot": {"external_event": {"event_id": "xyz"}}}
    assert nested_link(row) == {"event_id": "xyz"}


def test_nested_link_parses_json_string_snapshot():
    row = {"input_snapshot": '{"settlement_link": {"event_id": "abc"}}'}
    assert nested_link(row) == {"event_id": "abc"}


def test_nested_link_malformed_json_string_returns_empty():
    row = {"input_snapshot": "{not valid json"}
    assert nested_link(row) == {}


def test_nested_link_missing_snapshot_returns_empty():
    assert nested_link({}) == {}


def _fixture_link(**overrides) -> dict:
    base = {
        "event_status": "Finished",
        "event_first_player": "Player A",
        "event_second_player": "Player B",
        "event_winner": "First Player",
        "event_final_result": "6-3 6-4",
    }
    base.update(overrides)
    return {"fixture": base}


def test_settle_tennis_unfinished_event_returns_none_changes():
    row = {"prediction": "Player A", "recommendation": "Bet"}
    link = _fixture_link(event_status="Scheduled", event_winner="", event_final_result="")
    changes, raw = settle_tennis(row, link)
    assert changes is None
    assert raw["event_status"] == "Scheduled"


def test_settle_tennis_second_player_wins_maps_correctly():
    row = {"prediction": "Player B", "recommendation": "Bet"}
    link = _fixture_link(event_winner="Second Player")
    changes, raw = settle_tennis(row, link)
    assert raw["actual_winner"] == "Player B"
    assert changes["status"] == "Won"


def test_settle_tennis_finished_normal_match_grades_correctly():
    row = {"prediction": "Player A", "recommendation": "Strong Bet"}
    link = _fixture_link()
    changes, raw = settle_tennis(row, link)
    assert changes["status"] == "Won"
    assert changes["provider_link_status"] == "settled"
    assert changes["settled_at"] is not None


def test_settle_tennis_incorrect_prediction_grades_lost():
    row = {"prediction": "Player B", "recommendation": "Bet"}
    link = _fixture_link()
    changes, raw = settle_tennis(row, link)
    assert changes["status"] == "Lost"


def test_settle_tennis_missing_fixture_data_returns_none():
    row = {"prediction": "Player A", "recommendation": "Bet"}
    changes, raw = settle_tennis(row, {})
    assert changes is None


class _FakeOddsClient:
    def __init__(self, scores):
        self._scores = scores

    def scores(self, sport_key, *, days_from=3):
        return self._scores


class _RaisingOddsClient:
    def scores(self, sport_key, *, days_from=3):
        raise ProviderError("provider down")


def test_settle_odds_api_score_clear_winner_grades_correctly():
    row = {"prediction": "Home Team", "recommendation": "Bet"}
    link = {"sport_key": "americanfootball_nfl", "event_id": "evt1"}
    client = _FakeOddsClient([{
        "id": "evt1", "completed": True,
        "scores": [{"name": "Home Team", "score": "24"}, {"name": "Away Team", "score": "17"}],
    }])
    changes, raw = settle_odds_api_score(row, link, client)
    assert changes["status"] == "Won"
    assert raw["actual_winner"] == "Home Team"


def test_settle_odds_api_score_tie_is_not_graded():
    row = {"prediction": "Home Team", "recommendation": "Bet"}
    link = {"sport_key": "americanfootball_nfl", "event_id": "evt1"}
    client = _FakeOddsClient([{
        "id": "evt1", "completed": True,
        "scores": [{"name": "Home Team", "score": "24"}, {"name": "Away Team", "score": "24"}],
    }])
    changes, raw = settle_odds_api_score(row, link, client)
    assert raw["actual_winner"] is None
    assert changes["status"] == "Pending"


def test_settle_odds_api_score_event_not_found_returns_none():
    row = {"prediction": "Home Team", "recommendation": "Bet"}
    link = {"sport_key": "americanfootball_nfl", "event_id": "missing_event"}
    client = _FakeOddsClient([{"id": "some_other_event", "completed": True, "scores": []}])
    changes, raw = settle_odds_api_score(row, link, client)
    assert changes is None


def test_settle_odds_api_score_incomplete_event_returns_none():
    row = {"prediction": "Home Team", "recommendation": "Bet"}
    link = {"sport_key": "americanfootball_nfl", "event_id": "evt1"}
    client = _FakeOddsClient([{"id": "evt1", "completed": False, "scores": []}])
    changes, raw = settle_odds_api_score(row, link, client)
    assert changes is None


def test_settle_odds_api_score_provider_error_returns_none():
    row = {"prediction": "Home Team", "recommendation": "Bet"}
    link = {"sport_key": "americanfootball_nfl", "event_id": "evt1"}
    changes, raw = settle_odds_api_score(row, link, _RaisingOddsClient())
    assert changes is None
    assert raw == {}


# --- resolve_odds_link: deciding which external event a row matches --------

from settle_results import persist_links, resolve_odds_link


class _FakeFindOddsClient:
    """Duck-typed stand-in for OddsApiClient -- no real network calls."""
    def __init__(self, *, tennis_event=None, generic_event=None, raise_error=False):
        self._tennis_event = tennis_event
        self._generic_event = generic_event
        self._raise_error = raise_error

    def find_tennis_event(self, *, participant_a, participant_b):
        if self._raise_error:
            raise ProviderError("boom")
        return self._tennis_event

    def find_event(self, *, sport_key, participant_a, participant_b):
        if self._raise_error:
            raise ProviderError("boom")
        return self._generic_event


def test_resolve_odds_link_none_client_returns_none():
    row = {"participant_a": "A", "participant_b": "B", "sport": "NFL"}
    assert resolve_odds_link(row, None) is None


def test_resolve_odds_link_missing_participants_returns_none():
    row = {"participant_a": "", "participant_b": "B", "sport": "NFL"}
    client = _FakeFindOddsClient(generic_event={"id": "evt1", "commence_time": "2026-09-10T12:00:00Z"})
    assert resolve_odds_link(row, client) is None


def test_resolve_odds_link_prefers_analysis_time_cached_link():
    row = {
        "participant_a": "A", "participant_b": "B", "sport": "NFL",
        "input_snapshot": {"settlement_link": {
            "provider": "the_odds_api", "event_id": "cached_evt", "sport_key": "americanfootball_nfl",
            "commence_time": "2026-09-10T12:00:00Z",
        }},
    }
    # Client would raise if actually called -- confirms the cached path short-circuits.
    client = _FakeFindOddsClient(raise_error=True)
    link = resolve_odds_link(row, client)
    assert link["event_id"] == "cached_evt"
    assert link["link_method"] == "analysis_time_event_id"


def test_resolve_odds_link_uses_stored_event_id_when_no_cached_link():
    row = {
        "participant_a": "A", "participant_b": "B", "sport": "NFL",
        "market_provider": "the_odds_api",
        "market_provider_event_id": "stored_evt", "market_provider_sport_key": "americanfootball_nfl",
    }
    client = _FakeFindOddsClient(raise_error=True)
    link = resolve_odds_link(row, client)
    assert link["event_id"] == "stored_evt"
    assert link["link_method"] == "stored_event_id"


def test_resolve_odds_link_tennis_uses_find_tennis_event():
    row = {"participant_a": "A", "participant_b": "B", "sport": "Tennis"}
    client = _FakeFindOddsClient(tennis_event=("tennis_atp", {"id": "evt1", "commence_time": "2026-09-10T12:00:00Z"}))
    link = resolve_odds_link(row, client)
    assert link["sport_key"] == "tennis_atp"
    assert link["link_method"] == "strict_participant_match"


def test_resolve_odds_link_unknown_sport_returns_none():
    row = {"participant_a": "A", "participant_b": "B", "sport": "Curling"}
    client = _FakeFindOddsClient(generic_event={"id": "evt1"})
    assert resolve_odds_link(row, client) is None


def test_resolve_odds_link_no_event_found_returns_none():
    row = {"participant_a": "A", "participant_b": "B", "sport": "NFL"}
    client = _FakeFindOddsClient(generic_event=None)
    assert resolve_odds_link(row, client) is None


def test_resolve_odds_link_provider_error_returns_none():
    row = {"participant_a": "A", "participant_b": "B", "sport": "NFL"}
    client = _FakeFindOddsClient(raise_error=True)
    assert resolve_odds_link(row, client) is None


# --- persist_links -------------------------------------------------------------

def test_persist_links_updates_market_and_result_fields(monkeypatch):
    calls = []

    def fake_update_analysis(analysis_id, changes):
        calls.append((analysis_id, changes))
        return {"id": analysis_id, **changes}

    import settle_results as sr
    monkeypatch.setattr(sr, "update_analysis", fake_update_analysis)

    row = {"id": "row1"}
    market_link = {"event_id": "evt1", "sport_key": "americanfootball_nfl", "commence_time": "2026-09-10T12:00:00Z"}
    result_link = {"provider": "api_tennis", "event_id": "fixture1"}
    updated = persist_links(row, market_link=market_link, result_link=result_link)

    assert len(calls) == 1
    _, changes = calls[0]
    assert changes["market_provider_event_id"] == "evt1"
    assert changes["result_provider_event_id"] == "fixture1"
    assert changes["provider_link_status"] == "linked"
    assert updated["market_provider_event_id"] == "evt1"


def test_persist_links_no_links_does_not_call_update(monkeypatch):
    calls = []

    def fake_update_analysis(analysis_id, changes):
        calls.append((analysis_id, changes))
        return {"id": analysis_id, **changes}

    import settle_results as sr
    monkeypatch.setattr(sr, "update_analysis", fake_update_analysis)

    row = {"id": "row1"}
    result = persist_links(row, market_link=None, result_link=None)
    assert calls == []
    assert result == row


# --- latest_close / apply_close_fields: no fake CLV, no leakage -----------

class _FakeSnapshotList:
    def __init__(self, snapshots):
        self._snapshots = snapshots

    def __call__(self, table, params=None):
        return self._snapshots


def test_latest_close_returns_none_with_no_snapshots(monkeypatch):
    import settle_results as sr
    monkeypatch.setattr(sr, "list_table_records", _FakeSnapshotList([]))
    row = {"id": "row1", "scheduled_start": "2026-09-10T18:00:00Z"}
    assert sr.latest_close(row) is None


def test_latest_close_excludes_snapshots_captured_after_kickoff(monkeypatch):
    # Leakage guard: a snapshot captured after the game already started must
    # never be treated as the "closing" line.
    import settle_results as sr
    snapshots = [
        {"bookmaker": "BookA", "participant": "Home", "american_odds": -150,
         "captured_at": "2026-09-10T19:30:00Z"},  # after kickoff -- must be excluded
    ]
    monkeypatch.setattr(sr, "list_table_records", _FakeSnapshotList(snapshots))
    row = {"id": "row1", "scheduled_start": "2026-09-10T18:00:00Z",
           "prediction": "Home", "participant_a": "Home", "participant_b": "Away"}
    assert sr.latest_close(row) is None


def test_latest_close_uses_the_latest_pregame_snapshot(monkeypatch):
    import settle_results as sr
    snapshots = [
        {"bookmaker": "BookA", "participant": "Home", "american_odds": -160,
         "captured_at": "2026-09-10T12:00:00Z"},
        {"bookmaker": "BookA", "participant": "Away", "american_odds": 140,
         "captured_at": "2026-09-10T12:00:00Z"},
        {"bookmaker": "BookA", "participant": "Home", "american_odds": -150,
         "captured_at": "2026-09-10T17:00:00Z"},  # more recent, still pregame
        {"bookmaker": "BookA", "participant": "Away", "american_odds": 130,
         "captured_at": "2026-09-10T17:00:00Z"},
    ]
    monkeypatch.setattr(sr, "list_table_records", _FakeSnapshotList(snapshots))
    row = {"id": "row1", "scheduled_start": "2026-09-10T18:00:00Z",
           "prediction": "Home", "participant_a": "Home", "participant_b": "Away"}
    result = sr.latest_close(row)
    assert result is not None
    assert result["closing_odds_prediction"] == -150  # the 17:00 snapshot, not the earlier 12:00 one


def test_apply_close_fields_adds_nothing_when_no_close_available(monkeypatch):
    # This is the actual "no fake CLV" guarantee: with no real closing
    # snapshot, apply_close_fields must return the changes dict completely
    # untouched -- no clv_probability, no closing_odds_prediction, nothing.
    import settle_results as sr
    monkeypatch.setattr(sr, "list_table_records", _FakeSnapshotList([]))
    row = {"id": "row1", "scheduled_start": "2026-09-10T18:00:00Z"}
    changes = {"status": "Won"}
    result = sr.apply_close_fields(row, changes)
    assert result == {"status": "Won"}
    assert "clv_probability" not in result
    assert "closing_odds_prediction" not in result


def test_apply_close_fields_populates_clv_when_close_is_real(monkeypatch):
    import settle_results as sr
    snapshots = [
        {"bookmaker": "BookA", "participant": "Home", "american_odds": -170,
         "captured_at": "2026-09-10T17:00:00Z"},
        {"bookmaker": "BookA", "participant": "Away", "american_odds": 150,
         "captured_at": "2026-09-10T17:00:00Z"},
    ]
    monkeypatch.setattr(sr, "list_table_records", _FakeSnapshotList(snapshots))
    row = {
        "id": "row1", "scheduled_start": "2026-09-10T18:00:00Z",
        "prediction": "Home", "participant_a": "Home", "participant_b": "Away",
        "market_odds_a": -150, "market_odds_b": 130, "predicted_probability": 0.62,
    }
    result = sr.apply_close_fields(row, {"status": "Won"})
    assert result["closing_odds_prediction"] == -170
    assert result["clv_probability"] is not None
