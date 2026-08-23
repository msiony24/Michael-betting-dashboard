from __future__ import annotations

from datetime import date
from pathlib import Path

from engine.player_intelligence_store import (
    get_stored_profile,
    read_player_intelligence,
    write_player_intelligence,
)
from engine.player_profiles import PlayerProfile


def _profile(name: str, **overrides) -> PlayerProfile:
    base = dict(requested_name=name, historical_name=name, career_matches=42, ranking=7)
    base.update(overrides)
    return PlayerProfile(**base)


def test_write_then_read_round_trip(tmp_path: Path):
    destination = tmp_path / "intel.json"
    write_player_intelligence(
        [_profile("Jannik Sinner"), _profile("Carlos Alcaraz")],
        destination=destination,
        as_of_date=date(2024, 8, 1),
    )
    payload = read_player_intelligence(destination)
    assert payload["profile_count"] == 2
    assert payload["as_of_date"] == "2024-08-01"
    assert payload["schema_version"] == 1


def test_read_missing_file_returns_empty_dict(tmp_path: Path):
    assert read_player_intelligence(tmp_path / "does_not_exist.json") == {}


def test_read_malformed_json_returns_empty_dict_not_an_exception(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert read_player_intelligence(path) == {}


def test_get_stored_profile_returns_none_when_snapshot_is_from_after_event_date(tmp_path: Path):
    # This is the critical leakage guard: a snapshot generated after the
    # event date must never be used to "predict" that event, or every
    # backtest/analysis quietly cheats by seeing the future.
    destination = tmp_path / "intel.json"
    write_player_intelligence(
        [_profile("Jannik Sinner")],
        destination=destination,
        as_of_date=date(2024, 9, 15),
    )
    result = get_stored_profile("Jannik Sinner", event_date=date(2024, 9, 1), source=destination)
    assert result is None


def test_get_stored_profile_returns_profile_when_snapshot_predates_event(tmp_path: Path):
    destination = tmp_path / "intel.json"
    write_player_intelligence(
        [_profile("Jannik Sinner", career_matches=99)],
        destination=destination,
        as_of_date=date(2024, 8, 1),
    )
    result = get_stored_profile("Jannik Sinner", event_date=date(2024, 9, 1), source=destination)
    assert result is not None
    assert result.career_matches == 99
    assert "player_intelligence_store" in result.data_sources


def test_get_stored_profile_snapshot_on_same_day_as_event_is_allowed(tmp_path: Path):
    destination = tmp_path / "intel.json"
    write_player_intelligence(
        [_profile("Jannik Sinner")], destination=destination, as_of_date=date(2024, 9, 1),
    )
    result = get_stored_profile("Jannik Sinner", event_date=date(2024, 9, 1), source=destination)
    assert result is not None


def test_get_stored_profile_unknown_player_returns_none(tmp_path: Path):
    destination = tmp_path / "intel.json"
    write_player_intelligence([_profile("Jannik Sinner")], destination=destination, as_of_date=date(2024, 8, 1))
    result = get_stored_profile("Nobody Real", event_date=date(2024, 9, 1), source=destination)
    assert result is None


def test_get_stored_profile_missing_store_returns_none(tmp_path: Path):
    result = get_stored_profile("Jannik Sinner", event_date=date(2024, 9, 1), source=tmp_path / "missing.json")
    assert result is None


def test_write_player_intelligence_is_atomic_no_partial_file_left_behind(tmp_path: Path):
    destination = tmp_path / "intel.json"
    write_player_intelligence([_profile("Jannik Sinner")], destination=destination)
    assert destination.exists()
    assert not destination.with_suffix(destination.suffix + ".tmp").exists()
