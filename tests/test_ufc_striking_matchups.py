from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.ufc_striking_matchups import (
    UFCStrikingConfig,
    _avg,
    _fight_minutes,
    _gap,
    _safe_ratio,
    _score,
    _strength,
    build_advanced_striking_matchup,
    build_striking_table,
    fighter_striking_profile,
)


# --- small helpers -----------------------------------------------------------

def test_fight_minutes_accumulates_full_prior_rounds():
    # 2 finished rounds (rnd=2), ended at 1:00 -> 1 full round (5 min) + 1:00.
    assert _fight_minutes(2, "1:00") == pytest.approx(6.0)


def test_fight_minutes_never_zero():
    assert _fight_minutes(1, "0:00") == pytest.approx(1.0 / 60.0)


def test_safe_ratio_handles_zero_denominator():
    assert _safe_ratio(5.0, 0.0) is None
    assert _safe_ratio(3.0, 6.0) == pytest.approx(0.5)


def test_avg_ignores_missing_values():
    row = {"a": 10.0, "b": None, "c": float("nan")}
    assert _avg(row, ["a", "b", "c"]) == pytest.approx(10.0)
    assert _avg(row, ["b", "c"]) is None


def test_score_handles_missing_and_invalid():
    assert _score({"x": "12.5"}, "x") == 12.5
    assert _score({}, "x") is None
    assert _score({"x": "bad"}, "x") is None
    assert _score({"x": float("nan")}, "x") is None


def test_gap_none_when_any_input_missing():
    assert _gap(10.0, 10.0, 10.0, None) is None


def test_gap_math_is_symmetric_interaction():
    # a_attack=70 vs b_defense=50 (a is favored by 20), b_attack=40 vs a_defense=60 (b is
    # disfavored by -20) -> total gap should combine both sides of the exchange.
    gap = _gap(a_attack=70.0, b_defense=50.0, b_attack=40.0, a_defense=60.0)
    assert gap == pytest.approx((70.0 - 50.0) - (40.0 - 60.0))
    assert gap == pytest.approx(40.0)


def test_strength_boundaries():
    assert _strength(4.9) == "Even"
    assert _strength(5.0) == "Slight"
    assert _strength(11.9) == "Slight"
    assert _strength(12.0) == "Moderate"
    assert _strength(21.9) == "Moderate"
    assert _strength(22.0) == "Clear"
    assert _strength(-30.0) == "Clear"  # magnitude, direction-agnostic


# --- fighter_striking_profile lookup -----------------------------------------

def test_fighter_striking_profile_empty_table():
    result = fighter_striking_profile(pd.DataFrame(), "Alpha")
    assert result["sample"] == 0


def test_fighter_striking_profile_unknown_fighter():
    table = pd.DataFrame([{"fighter": "Someone Else", "sample": 5}])
    result = fighter_striking_profile(table, "Alpha")
    assert result["sample"] == 0


def test_fighter_striking_profile_known_fighter_returns_row():
    table = pd.DataFrame([{"fighter": "Alpha", "sample": 5, "head_attack_score": 70.0}])
    result = fighter_striking_profile(table, "alpha")  # case-insensitive
    assert result["sample"] == 5
    assert result["head_attack_score"] == 70.0


# --- build_advanced_striking_matchup ----------------------------------------

def _synthetic_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _full_profile(name: str, sample: int = 10, completeness: float = 1.0, **scores) -> dict:
    base = {
        "fighter": name, "sample": sample, "data_completeness": completeness,
        "head_attack_score": 50.0, "head_defense_score": 50.0,
        "body_attack_score": 50.0, "body_defense_score": 50.0,
        "leg_attack_score": 50.0, "leg_defense_score": 50.0,
        "distance_attack_score": 50.0, "distance_defense_score": 50.0,
        "power_score": 50.0, "knockdown_resistance_score": 50.0,
        "close_attack_score": 50.0, "close_defense_score": 50.0,
    }
    base.update(scores)
    return base


def test_advanced_striking_matchup_identical_fighters_are_even():
    table = _synthetic_table([_full_profile("Alpha"), _full_profile("Bravo")])
    result = build_advanced_striking_matchup(table, "Alpha", "Bravo")
    assert result["available"] is True
    assert result["weighted_gap"] == pytest.approx(0.0)
    assert all(row["advantage"] == "Even" for row in result["rows"])


def test_advanced_striking_matchup_favors_the_better_striker():
    table = _synthetic_table([
        _full_profile("Alpha", head_attack_score=90.0, distance_attack_score=90.0, power_score=90.0),
        _full_profile("Bravo"),
    ])
    result = build_advanced_striking_matchup(table, "Alpha", "Bravo")
    assert result["weighted_gap"] > 0
    head_row = next(r for r in result["rows"] if r["category"].startswith("Head"))
    assert head_row["advantage"] == "Alpha"


def test_advanced_striking_matchup_unavailable_with_no_score_columns():
    table = _synthetic_table([{"fighter": "Alpha", "sample": 0}, {"fighter": "Bravo", "sample": 0}])
    result = build_advanced_striking_matchup(table, "Alpha", "Bravo")
    assert result["available"] is False
    assert result["reliability"] == 0.0


def test_advanced_striking_matchup_reliability_scales_with_sample_and_completeness():
    config = UFCStrikingConfig(min_sample_for_full_weight=10)
    table_full = _synthetic_table([_full_profile("Alpha", sample=10), _full_profile("Bravo", sample=10)])
    table_thin = _synthetic_table([_full_profile("Alpha", sample=2), _full_profile("Bravo", sample=2)])
    full = build_advanced_striking_matchup(table_full, "Alpha", "Bravo", config=config)
    thin = build_advanced_striking_matchup(table_thin, "Alpha", "Bravo", config=config)
    assert full["reliability"] > thin["reliability"]


def test_advanced_striking_matchup_reliability_never_exceeds_one():
    table = _synthetic_table([_full_profile("Alpha", sample=999), _full_profile("Bravo", sample=999)])
    result = build_advanced_striking_matchup(table, "Alpha", "Bravo")
    assert result["reliability"] <= 1.0


# --- build_striking_table: light integration check ---------------------------

def _fight_row(fighter, opponent, fight_url, event_date, **stats):
    row = {"fighter": fighter, "opponent": opponent, "fight_url": fight_url, "event_date": event_date,
           "round": 3, "time": "5:00"}
    row.update(stats)
    return row


def test_build_striking_table_runs_without_crashing_on_realistic_shape():
    # Three fighters in the same division so percentile ranking (needs >= 3)
    # actually activates, each with one two-sided fight against the others.
    rows = []
    pairs = [("Alpha", "Beta", "f1"), ("Beta", "Gamma", "f2"), ("Gamma", "Alpha", "f3")]
    for a, b, url in pairs:
        rows.append(_fight_row(a, b, url, "2025-01-01",
                                sig_str_landed=40, sig_str_attempted=80, kd=1,
                                head_landed=20, head_attempted=40, body_landed=10, body_attempted=15,
                                leg_landed=5, leg_attempted=8, distance_landed=25, distance_attempted=50,
                                clinch_landed=5, clinch_attempted=10, ground_landed=5, ground_attempted=8))
        rows.append(_fight_row(b, a, url, "2025-01-01",
                                sig_str_landed=30, sig_str_attempted=70, kd=0,
                                head_landed=15, head_attempted=35, body_landed=8, body_attempted=15,
                                leg_landed=4, leg_attempted=8, distance_landed=20, distance_attempted=45,
                                clinch_landed=4, clinch_attempted=10, ground_landed=3, ground_attempted=8))
    fights = pd.DataFrame(rows)
    ratings = pd.DataFrame([
        {"fighter": "Alpha", "division": "Lightweight", "active_pool": True},
        {"fighter": "Beta", "division": "Lightweight", "active_pool": True},
        {"fighter": "Gamma", "division": "Lightweight", "active_pool": True},
    ])
    table = build_striking_table(fights, ratings)
    assert len(table) == 3
    assert "head_attack_score" in table.columns
