from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine.ufc_context import (
    UFCContextConfig,
    _age_on,
    _decline_index,
    _division_transition,
    _parse_dob,
    _parse_inches,
    _parse_weight,
    _recent_activity,
    build_fight_context,
)


# --- small parsers -----------------------------------------------------------

def test_parse_inches_feet_and_inches_format():
    assert _parse_inches("6' 2\"") == 74.0


def test_parse_inches_plain_number():
    assert _parse_inches("74") == 74.0


def test_parse_inches_missing_or_placeholder():
    assert _parse_inches("--") is None
    assert _parse_inches(None) is None
    assert _parse_inches("") is None


def test_parse_weight_extracts_number():
    assert _parse_weight("155 lbs.") == 155.0


def test_parse_dob_valid_and_invalid():
    assert _parse_dob("1994-05-10") is not None
    assert _parse_dob("--") is None
    assert _parse_dob(None) is None


# --- age curve: no youth bonus, decline only after 32 -----------------------

def test_age_on_computes_correct_age():
    dob = _parse_dob("2000-01-01")
    age = _age_on(dob, date(2025, 1, 1))
    assert age == pytest.approx(25.0, abs=0.01)


def test_decline_index_zero_before_32():
    assert _decline_index(25.0) == 0.0
    assert _decline_index(31.9) == 0.0


def test_decline_index_linear_between_32_and_36():
    assert _decline_index(34.0) == pytest.approx(2.0)


def test_decline_index_steeper_after_36():
    at_36 = _decline_index(36.0)
    at_38 = _decline_index(38.0)
    # From 36 to 38: +2 from the base term, +0.75*2=1.5 from the extra term -> +3.5 total.
    assert (at_38 - at_36) == pytest.approx(3.5)


def test_decline_index_none_input():
    assert _decline_index(None) is None


# --- _recent_activity: as-of date leakage guard ------------------------------

def _fights(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_recent_activity_excludes_fights_after_the_as_of_date():
    fights = _fights([
        {"fighter": "Alpha", "event_date": "2025-01-01", "division": "Men's Lightweight"},
        {"fighter": "Alpha", "event_date": "2026-06-01", "division": "Men's Lightweight"},  # future
    ])
    activity = _recent_activity(fights, "Alpha", date(2025, 6, 1))
    assert activity["days_since_last_fight"] == 151  # from the Jan fight only
    assert activity["fights_365d"] == 1


def test_recent_activity_counts_180_and_365_day_windows():
    fights = _fights([
        {"fighter": "Alpha", "event_date": "2025-05-01", "division": "Men's Lightweight"},  # ~150 days back
        {"fighter": "Alpha", "event_date": "2024-10-01", "division": "Men's Lightweight"},  # ~333 days back
        {"fighter": "Alpha", "event_date": "2023-01-01", "division": "Men's Lightweight"},  # >365 days back
    ])
    activity = _recent_activity(fights, "Alpha", date(2025, 9, 28))
    assert activity["fights_180d"] == 1
    assert activity["fights_365d"] == 2


def test_recent_activity_no_history_returns_none_days():
    activity = _recent_activity(_fights([]), "Alpha", date(2025, 1, 1))
    assert activity["days_since_last_fight"] is None
    assert activity["fights_180d"] == 0


def test_recent_activity_case_insensitive_name_match():
    fights = _fights([{"fighter": "alpha fighter", "event_date": "2025-01-01", "division": "Men's Lightweight"}])
    activity = _recent_activity(fights, "Alpha Fighter", date(2025, 2, 1))
    assert activity["days_since_last_fight"] == 31


# --- _division_transition -----------------------------------------------------

def test_division_transition_detects_mismatch():
    activity = {"recent_divisions": ["Men's Welterweight", "Men's Welterweight"]}
    assert _division_transition(activity, "Men's Lightweight") is True


def test_division_transition_false_when_consistent():
    activity = {"recent_divisions": ["Men's Lightweight", "Men's Lightweight"]}
    assert _division_transition(activity, "Men's Lightweight") is False


def test_division_transition_ignores_catchweight_bouts():
    activity = {"recent_divisions": ["Catch Weight"]}
    assert _division_transition(activity, "Men's Lightweight") is False


def test_division_transition_false_with_no_recent_history():
    assert _division_transition({"recent_divisions": []}, "Men's Lightweight") is False


# --- build_fight_context: end-to-end shape and caps --------------------------

def _profiles(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _ratings(fighters: list[str], division: str = "Men's Lightweight") -> pd.DataFrame:
    return pd.DataFrame([{"fighter": f, "division": division} for f in fighters])


def test_build_fight_context_no_profile_data_still_returns_a_result():
    result = build_fight_context(
        "Alpha", "Bravo", _ratings(["Alpha", "Bravo"]), pd.DataFrame(),
        profiles=pd.DataFrame(), fight_date=date(2025, 1, 1),
    )
    assert result["available"] is True  # activity/stance rows always present
    assert result["adjustment_a"] == pytest.approx(0.0)


def test_build_fight_context_reach_advantage_favors_longer_reach():
    profiles = _profiles([
        {"fighter": "Alpha", "reach": "78", "height": "72", "dob": "1994-01-01", "stance": "Orthodox"},
        {"fighter": "Bravo", "reach": "68", "height": "70", "dob": "1994-01-01", "stance": "Orthodox"},
    ])
    result = build_fight_context(
        "Alpha", "Bravo", _ratings(["Alpha", "Bravo"]), pd.DataFrame(),
        profiles=profiles, fight_date=date(2025, 1, 1),
    )
    assert result["physical_adjustment_a"] > 0


def test_build_fight_context_physical_adjustment_is_capped(monkeypatch):
    config = UFCContextConfig(max_physical_adjustment=0.009)
    profiles = _profiles([
        {"fighter": "Alpha", "reach": "90", "height": "84"},  # absurd reach gap
        {"fighter": "Bravo", "reach": "60", "height": "60"},
    ])
    result = build_fight_context(
        "Alpha", "Bravo", _ratings(["Alpha", "Bravo"]), pd.DataFrame(),
        profiles=profiles, fight_date=date(2025, 1, 1), config=config,
    )
    assert abs(result["physical_adjustment_a"]) <= config.max_physical_adjustment + 1e-9


def test_build_fight_context_older_fighter_is_penalized_not_bonused():
    profiles = _profiles([
        {"fighter": "Alpha", "dob": "1985-01-01"},  # much older, well past decline threshold
        {"fighter": "Bravo", "dob": "2000-01-01"},  # younger
    ])
    result = build_fight_context(
        "Alpha", "Bravo", _ratings(["Alpha", "Bravo"]), pd.DataFrame(),
        profiles=profiles, fight_date=date(2025, 1, 1),
    )
    # Bravo (younger) should be favored by the age adjustment.
    assert result["age_adjustment_a"] < 0


def test_build_fight_context_age_adjustment_is_capped():
    config = UFCContextConfig(max_age_adjustment=0.010)
    profiles = _profiles([
        {"fighter": "Alpha", "dob": "1970-01-01"},  # extremely old
        {"fighter": "Bravo", "dob": "2005-01-01"},
    ])
    result = build_fight_context(
        "Alpha", "Bravo", _ratings(["Alpha", "Bravo"]), pd.DataFrame(),
        profiles=profiles, fight_date=date(2025, 1, 1), config=config,
    )
    assert abs(result["age_adjustment_a"]) <= config.max_age_adjustment + 1e-9


def test_build_fight_context_stance_is_never_directional():
    profiles = _profiles([
        {"fighter": "Alpha", "stance": "Southpaw"},
        {"fighter": "Bravo", "stance": "Orthodox"},
    ])
    result = build_fight_context(
        "Alpha", "Bravo", _ratings(["Alpha", "Bravo"]), pd.DataFrame(),
        profiles=profiles, fight_date=date(2025, 1, 1),
    )
    stance_row = next(r for r in result["rows"] if r["category"] == "Stance")
    assert stance_row["advantage"] == "Even"
    assert stance_row["line_impact_a"] == 0.0


def test_build_fight_context_overall_adjustment_never_exceeds_probability_cap():
    config = UFCContextConfig(max_probability_adjustment=0.02)
    profiles = _profiles([
        {"fighter": "Alpha", "reach": "90", "height": "84", "dob": "1970-01-01", "stance": "Orthodox"},
        {"fighter": "Bravo", "reach": "60", "height": "60", "dob": "2005-01-01", "stance": "Orthodox"},
    ])
    fights = pd.DataFrame([
        {"fighter": "Alpha", "event_date": "2025-01-01", "division": "Men's Lightweight"},
        {"fighter": "Bravo", "event_date": "2024-01-01", "division": "Men's Lightweight"},
    ])
    result = build_fight_context(
        "Alpha", "Bravo", _ratings(["Alpha", "Bravo"]), fights,
        profiles=profiles, fight_date=date(2025, 1, 10), config=config,
    )
    assert abs(result["adjustment_a"]) <= config.max_probability_adjustment + 1e-9


def test_build_fight_context_division_transition_lowers_confidence():
    profiles = pd.DataFrame()
    fights = pd.DataFrame([
        {"fighter": "Alpha", "event_date": "2025-01-01", "division": "Men's Welterweight"},
        {"fighter": "Alpha", "event_date": "2024-06-01", "division": "Men's Welterweight"},
        {"fighter": "Bravo", "event_date": "2025-01-01", "division": "Men's Lightweight"},
    ])
    result = build_fight_context(
        "Alpha", "Bravo", _ratings(["Alpha", "Bravo"], division="Men's Lightweight"), fights,
        profiles=profiles, fight_date=date(2025, 6, 1),
    )
    assert result["fighter_a_weight_class_transition"] is True
    assert result["confidence_modifier"] < 0
