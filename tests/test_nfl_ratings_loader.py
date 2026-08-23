"""Real tests for engine/nfl_ratings_loader.py.

The root-level test_nfl_ratings_loader.py is a print-only smoke script (no
assertions) run directly in CI -- it only catches a crash, never a wrong
value. This file replaces that gap with real coverage of the merge/blend
logic that feeds every team's rating.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import engine.nfl_ratings_loader as loader
from engine.nfl_ratings_loader import (
    REQUIRED_TEAM_FIELDS,
    _blend,
    _madden_category_ratings,
    _number,
    _unit_grade,
    load_all_team_ratings,
    load_team_quality,
    merge_team_ratings,
)


def test_number_handles_missing_and_invalid():
    assert _number("70.5") == 70.5
    assert _number(None, default=50.0) == 50.0
    assert _number("bad", default=60.0) == 60.0


def test_unit_grade_from_dict_with_grade_key():
    madden_team = {"units": {"quarterback": {"grade": 88.0}}}
    assert _unit_grade(madden_team, "quarterback", 50.0) == 88.0


def test_unit_grade_from_plain_number_unit():
    madden_team = {"units": {"quarterback": 82.0}}
    assert _unit_grade(madden_team, "quarterback", 50.0) == 82.0


def test_unit_grade_missing_unit_returns_default():
    madden_team = {"units": {}}
    assert _unit_grade(madden_team, "quarterback", 55.0) == 55.0


def test_unit_grade_missing_units_key_returns_default():
    assert _unit_grade({}, "quarterback", 55.0) == 55.0


def test_blend_full_madden_weight_uses_madden_value():
    assert _blend(50.0, 90.0, madden_weight=1.0) == 90.0


def test_blend_zero_weight_uses_manual_value():
    assert _blend(50.0, 90.0, madden_weight=0.0) == 50.0


def test_blend_half_weight_averages():
    assert _blend(40.0, 80.0, madden_weight=0.5) == 60.0


def test_blend_weight_is_clamped_to_0_1_range():
    # A weight above 1.0 must not overshoot past the madden value.
    assert _blend(0.0, 100.0, madden_weight=5.0) == 100.0
    assert _blend(0.0, 100.0, madden_weight=-5.0) == 0.0


def test_madden_category_ratings_computes_offense_defense_composites():
    manual = {"quarterback": 60, "skill_positions": 60, "offensive_line": 60,
              "defensive_line": 60, "defense": 60, "secondary": 60, "special_teams": 60}
    madden = {"units": {
        "quarterback": {"grade": 90}, "running_backs": {"grade": 80}, "receiving_weapons": {"grade": 80},
        "offensive_line": {"grade": 70}, "defensive_front": {"grade": 75}, "linebackers": {"grade": 70},
        "secondary": {"grade": 65}, "special_teams": {"grade": 60},
    }}
    result = _madden_category_ratings(manual, madden)
    assert result["quarterback"] == 90.0
    expected_offense = 90 * 0.35 + 80 * 0.15 + 80 * 0.25 + 70 * 0.25
    assert result["offense"] == pytest.approx(expected_offense)


def test_madden_category_ratings_falls_back_to_manual_when_madden_unit_missing():
    manual = {"quarterback": 65}
    result = _madden_category_ratings(manual, {"units": {}})
    assert result["quarterback"] == 65.0


# --- merge_team_ratings -------------------------------------------------------

def _manual_team(**overrides) -> dict:
    base = {field: 60.0 for field in REQUIRED_TEAM_FIELDS}
    base.update(overrides)
    return base


def _madden_team(**overrides) -> dict:
    base = {"units": {
        "quarterback": {"grade": 80}, "running_backs": {"grade": 70}, "receiving_weapons": {"grade": 70},
        "offensive_line": {"grade": 70}, "defensive_front": {"grade": 70}, "linebackers": {"grade": 70},
        "secondary": {"grade": 70}, "special_teams": {"grade": 70},
    }, "roster_grade": 75.0}
    base.update(overrides)
    return base


def test_merge_team_ratings_missing_required_field_raises(monkeypatch):
    monkeypatch.setattr(loader, "load_coaching_priors", lambda: {})
    monkeypatch.setattr(loader, "load_continuity_priors", lambda: {})
    manual = {"Buffalo Bills": {"quarterback": 60}}  # missing most required fields
    with pytest.raises(ValueError, match="missing required fields"):
        merge_team_ratings(manual, {})


def test_merge_team_ratings_team_missing_from_madden_stays_unblended(monkeypatch):
    monkeypatch.setattr(loader, "load_coaching_priors", lambda: {})
    monkeypatch.setattr(loader, "load_continuity_priors", lambda: {})
    manual = {"Buffalo Bills": _manual_team(quarterback=60.0)}
    result = merge_team_ratings(manual, {})
    assert result["Buffalo Bills"]["madden_status"] == "Unavailable"
    assert result["Buffalo Bills"]["madden_blend_weight"] == 0.0
    assert result["Buffalo Bills"]["quarterback"] == 60.0


def test_merge_team_ratings_full_madden_weight_uses_madden_grades(monkeypatch):
    monkeypatch.setattr(loader, "load_coaching_priors", lambda: {})
    monkeypatch.setattr(loader, "load_continuity_priors", lambda: {})
    manual = {"Buffalo Bills": _manual_team(quarterback=50.0)}
    madden = {"Buffalo Bills": _madden_team()}
    result = merge_team_ratings(manual, madden, madden_weight=1.0)
    assert result["Buffalo Bills"]["quarterback"] == 80.0
    assert result["Buffalo Bills"]["madden_status"] == "Integrated"


def test_merge_team_ratings_sleeper_availability_zeroes_injury_adjustment(monkeypatch):
    monkeypatch.setattr(loader, "load_coaching_priors", lambda: {})
    monkeypatch.setattr(loader, "load_continuity_priors", lambda: {})
    manual = {"Buffalo Bills": _manual_team(injury_adjustment=-5.0)}
    madden = {"Buffalo Bills": _madden_team(availability_source="Sleeper")}
    result = merge_team_ratings(manual, madden)
    assert result["Buffalo Bills"]["injury_adjustment"] == 0.0


def test_merge_team_ratings_non_sleeper_availability_keeps_manual_injury_adjustment(monkeypatch):
    monkeypatch.setattr(loader, "load_coaching_priors", lambda: {})
    monkeypatch.setattr(loader, "load_continuity_priors", lambda: {})
    manual = {"Buffalo Bills": _manual_team(injury_adjustment=-5.0)}
    madden = {"Buffalo Bills": _madden_team(availability_source="")}
    result = merge_team_ratings(manual, madden)
    assert result["Buffalo Bills"]["injury_adjustment"] == -5.0


def test_merge_team_ratings_no_madden_data_skips_coaching_and_continuity_priors(monkeypatch):
    # Documents real behavior: a team absent from madden_ratings short-circuits
    # to the "Unavailable" branch before coaching/continuity priors are ever
    # applied, even if a real coaching prior exists for that team.
    monkeypatch.setattr(loader, "load_coaching_priors", lambda: {
        "Buffalo Bills": {"rating": 85.0, "head_coach": "Sean McDermott"},
    })
    monkeypatch.setattr(loader, "load_continuity_priors", lambda: {})
    manual = {"Buffalo Bills": _manual_team(coaching=60.0)}
    result = merge_team_ratings(manual, {})
    assert result["Buffalo Bills"]["madden_status"] == "Unavailable"
    assert result["Buffalo Bills"]["coaching"] == 60.0  # untouched manual value, prior never applied


def test_merge_team_ratings_missing_coaching_prior_uses_neutral_fallback(monkeypatch):
    # Coaching/continuity fallback logic only runs in the "team has madden
    # data" branch -- a team entirely absent from madden_ratings short-circuits
    # before reaching it, so a madden entry must be present here.
    monkeypatch.setattr(loader, "load_coaching_priors", lambda: {})
    monkeypatch.setattr(loader, "load_continuity_priors", lambda: {})
    manual = {"Buffalo Bills": _manual_team()}
    madden = {"Buffalo Bills": _madden_team()}
    result = merge_team_ratings(manual, madden)
    assert result["Buffalo Bills"]["coaching"] == 70.0
    assert "fallback" in result["Buffalo Bills"]["coaching_status"].lower()


def test_merge_team_ratings_present_coaching_prior_is_used(monkeypatch):
    monkeypatch.setattr(loader, "load_coaching_priors", lambda: {
        "Buffalo Bills": {"rating": 85.0, "head_coach": "Sean McDermott"},
    })
    monkeypatch.setattr(loader, "load_continuity_priors", lambda: {})
    manual = {"Buffalo Bills": _manual_team()}
    madden = {"Buffalo Bills": _madden_team()}
    result = merge_team_ratings(manual, madden)
    assert result["Buffalo Bills"]["coaching"] == 85.0
    assert result["Buffalo Bills"]["head_coach"] == "Sean McDermott"


def test_merge_team_ratings_missing_continuity_prior_uses_neutral_fallback(monkeypatch):
    monkeypatch.setattr(loader, "load_coaching_priors", lambda: {})
    monkeypatch.setattr(loader, "load_continuity_priors", lambda: {})
    manual = {"Buffalo Bills": _manual_team()}
    madden = {"Buffalo Bills": _madden_team()}
    result = merge_team_ratings(manual, madden)
    assert result["Buffalo Bills"]["continuity"] == 67.5
    assert "fallback" in result["Buffalo Bills"]["continuity_status"].lower()


# --- load_all_team_ratings / load_team_quality -------------------------------

def test_load_all_team_ratings_missing_madden_file_falls_back_gracefully(tmp_path: Path):
    manual_path = tmp_path / "manual.json"
    manual = {"Buffalo Bills": _manual_team()}
    manual_path.write_text(json.dumps(manual))
    result = load_all_team_ratings(
        ratings_path=manual_path,
        madden_ratings_path=tmp_path / "does_not_exist.json",
    )
    assert result["Buffalo Bills"]["madden_status"] == "File not generated"
    assert result["Buffalo Bills"]["madden_blend_weight"] == 0.0


def test_load_all_team_ratings_use_madden_false_returns_manual_only(tmp_path: Path):
    manual_path = tmp_path / "manual.json"
    manual = {"Buffalo Bills": _manual_team()}
    manual_path.write_text(json.dumps(manual))
    result = load_all_team_ratings(ratings_path=manual_path, use_madden=False)
    assert result == manual


def test_load_manual_team_ratings_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        loader.load_manual_team_ratings(tmp_path / "does_not_exist.json")


def test_load_manual_team_ratings_non_object_json_raises(tmp_path: Path):
    path = tmp_path / "ratings.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="JSON object"):
        loader.load_manual_team_ratings(path)


def test_load_team_quality_unknown_team_raises_with_helpful_message(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(loader, "load_coaching_priors", lambda: {})
    monkeypatch.setattr(loader, "load_continuity_priors", lambda: {})
    manual_path = tmp_path / "manual.json"
    manual_path.write_text(json.dumps({"Buffalo Bills": _manual_team()}))
    with pytest.raises(KeyError, match="Team not found"):
        load_team_quality(
            "Springfield Isotopes",
            ratings_path=manual_path,
            madden_ratings_path=tmp_path / "no_madden.json",
        )


def test_load_team_quality_returns_a_real_result(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(loader, "load_coaching_priors", lambda: {})
    monkeypatch.setattr(loader, "load_continuity_priors", lambda: {})
    manual_path = tmp_path / "manual.json"
    manual_path.write_text(json.dumps({"Buffalo Bills": _manual_team(quarterback=85.0)}))
    result = load_team_quality(
        "Buffalo Bills",
        ratings_path=manual_path,
        madden_ratings_path=tmp_path / "no_madden.json",
    )
    assert result.team == "Buffalo Bills"
