from __future__ import annotations

import pandas as pd
import pytest

from engine.ufc_grappling_matchups import (
    UFCGrapplingConfig,
    _fighter_raw_profile,
    _interaction_gap,
    _safe_ratio,
    _score,
    _strength,
    build_advanced_grappling_matchup,
    fighter_grappling_profile,
)


# --- small helpers -----------------------------------------------------------

def test_safe_ratio_handles_zero_denominator():
    assert _safe_ratio(4.0, 0.0) is None
    assert _safe_ratio(2.0, 4.0) == pytest.approx(0.5)


def test_score_handles_missing_and_invalid():
    assert _score({"x": "8"}, "x") == 8.0
    assert _score({}, "x") is None
    assert _score({"x": "bad"}, "x") is None


def test_interaction_gap_none_when_any_input_missing():
    assert _interaction_gap(10.0, 10.0, None, 10.0) is None


def test_interaction_gap_combines_both_sides_of_exchange():
    gap = _interaction_gap(a_attack=80.0, b_defense=40.0, b_attack=30.0, a_defense=60.0)
    assert gap == pytest.approx((80.0 - 40.0) - (30.0 - 60.0))


def test_strength_boundaries():
    assert _strength(4.9) == "Even"
    assert _strength(5.0) == "Slight"
    assert _strength(12.0) == "Moderate"
    assert _strength(22.0) == "Clear"


# --- _fighter_raw_profile: the actual grappling stat math -------------------

def _rows(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def test_raw_profile_empty_input():
    profile = _fighter_raw_profile(pd.DataFrame())
    assert profile["sample"] == 0
    assert profile["data_completeness"] == 0.0


def test_raw_profile_td_defense_is_one_minus_opponent_landed_rate():
    rows = _rows([{
        "event_date": "2025-01-01", "round": 3, "time": "5:00",
        "td_landed": 2, "td_attempted": 4,
        "opponent_td_landed": 1, "opponent_td_attempted": 5,
        "control_seconds": 60, "opponent_control_seconds": 30,
    }])
    profile = _fighter_raw_profile(rows)
    # td_defense = 1 - (opponent_td_landed / opponent_td_attempted) = 1 - 1/5 = 0.8
    assert profile["td_defense"] == pytest.approx(0.8)
    assert profile["td_accuracy"] == pytest.approx(0.5)


def test_raw_profile_submission_loss_rate_counts_only_submission_losses():
    rows = _rows([
        {"event_date": "2025-01-01", "round": 1, "time": "3:00", "result": "L", "method": "Submission (RNC)"},
        {"event_date": "2025-02-01", "round": 3, "time": "5:00", "result": "L", "method": "Decision"},
        {"event_date": "2025-03-01", "round": 2, "time": "1:00", "result": "W", "method": "KO"},
    ])
    profile = _fighter_raw_profile(rows)
    # 2 losses total, 1 of them a submission -> rate 0.5
    assert profile["submission_loss_rate"] == pytest.approx(0.5)


def test_raw_profile_submission_loss_rate_zero_with_no_losses():
    rows = _rows([{"event_date": "2025-01-01", "round": 1, "time": "3:00", "result": "W", "method": "KO"}])
    profile = _fighter_raw_profile(rows)
    assert profile["submission_loss_rate"] == 0.0


def test_raw_profile_repeat_attempt_rate_uses_round_level_columns():
    rows = _rows([{
        "event_date": "2025-01-01", "round": 3, "time": "5:00",
        "r1_td_attempted": 3, "r2_td_attempted": 1, "r3_td_attempted": 0,
    }])
    profile = _fighter_raw_profile(rows)
    # 2 rounds had >=1 attempt (r1, r2); 1 round had >=2 attempts (r1) -> rate 0.5
    assert profile["active_wrestling_rounds"] == 2
    assert profile["repeat_attempt_rounds"] == 1
    assert profile["repeat_attempt_rate"] == pytest.approx(0.5)


def test_raw_profile_repeat_attempt_rate_none_without_round_level_data():
    rows = _rows([{"event_date": "2025-01-01", "round": 3, "time": "5:00", "td_attempted": 5}])
    profile = _fighter_raw_profile(rows)
    assert profile["repeat_attempt_rate"] is None


# --- fighter_grappling_profile lookup ---------------------------------------

def test_fighter_grappling_profile_empty_table():
    assert fighter_grappling_profile(pd.DataFrame(), "Alpha")["sample"] == 0


def test_fighter_grappling_profile_case_insensitive_match():
    table = pd.DataFrame([{"fighter": "Alpha", "sample": 3, "chain_wrestling_score": 60.0}])
    result = fighter_grappling_profile(table, "ALPHA")
    assert result["sample"] == 3


# --- build_advanced_grappling_matchup ---------------------------------------

def _full_profile(name: str, sample: int = 10, completeness: float = 1.0, **scores) -> dict:
    base = {
        "fighter": name, "sample": sample, "data_completeness": completeness,
        "chain_wrestling_score": 50.0, "takedown_resistance_score": 50.0,
        "control_retention_score": 50.0, "bottom_escape_score": 50.0,
        "submission_pressure_score": 50.0, "submission_resistance_score": 50.0,
    }
    base.update(scores)
    return base


def test_grappling_matchup_identical_fighters_are_even():
    table = pd.DataFrame([_full_profile("Alpha"), _full_profile("Bravo")])
    result = build_advanced_grappling_matchup(table, "Alpha", "Bravo")
    assert result["available"] is True
    assert result["weighted_gap"] == pytest.approx(0.0)
    assert all(row["advantage"] == "Even" for row in result["rows"])


def test_grappling_matchup_favors_the_better_wrestler():
    table = pd.DataFrame([
        _full_profile("Alpha", chain_wrestling_score=95.0),
        _full_profile("Bravo"),
    ])
    result = build_advanced_grappling_matchup(table, "Alpha", "Bravo")
    chain_row = next(r for r in result["rows"] if r["category"].startswith("Chain"))
    assert chain_row["advantage"] == "Alpha"
    assert result["weighted_gap"] > 0


def test_grappling_matchup_unavailable_with_no_score_data():
    table = pd.DataFrame([{"fighter": "Alpha", "sample": 0}, {"fighter": "Bravo", "sample": 0}])
    result = build_advanced_grappling_matchup(table, "Alpha", "Bravo")
    assert result["available"] is False
    assert result["reliability"] == 0.0


def test_grappling_matchup_reliability_bounded_and_scales_with_sample():
    config = UFCGrapplingConfig(min_sample_for_full_weight=10)
    full = build_advanced_grappling_matchup(
        pd.DataFrame([_full_profile("Alpha", sample=10), _full_profile("Bravo", sample=10)]),
        "Alpha", "Bravo", config=config,
    )
    thin = build_advanced_grappling_matchup(
        pd.DataFrame([_full_profile("Alpha", sample=1), _full_profile("Bravo", sample=1)]),
        "Alpha", "Bravo", config=config,
    )
    assert 0.0 <= thin["reliability"] < full["reliability"] <= 1.0


def test_grappling_matchup_chain_wrestling_carries_the_most_weight():
    # Chain wrestling (0.50) should dominate the weighted gap versus control
    # (0.28) or submissions (0.22) when only one category has a gap.
    table_chain = pd.DataFrame([
        _full_profile("Alpha", chain_wrestling_score=90.0), _full_profile("Bravo"),
    ])
    table_sub = pd.DataFrame([
        _full_profile("Alpha", submission_pressure_score=90.0), _full_profile("Bravo"),
    ])
    chain_result = build_advanced_grappling_matchup(table_chain, "Alpha", "Bravo")
    sub_result = build_advanced_grappling_matchup(table_sub, "Alpha", "Bravo")
    assert chain_result["weighted_gap"] > sub_result["weighted_gap"]
