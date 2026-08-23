from __future__ import annotations

import pandas as pd
import pytest

from engine.ufc_performance import (
    UFCPerformanceConfig,
    _profile_for_rows,
    fighter_performance,
    matchup_performance_adjustment,
)


def _rows(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


# --- _profile_for_rows --------------------------------------------------------

def test_profile_for_rows_empty_input():
    profile = _profile_for_rows(pd.DataFrame())
    assert profile["sample"] == 0
    assert profile["data_completeness"] == 0.0


def test_profile_for_rows_falls_back_to_compact_sig_str_when_detailed_missing():
    rows = _rows([{
        "event_date": "2025-01-01", "round": 3, "time": "5:00",
        "sig_str": 40, "result": "W", "method": "Decision",
    }])
    profile = _profile_for_rows(rows)
    # No sig_str_landed column at all -> should fall back to the compact "sig_str" field.
    assert profile["sig_landed_per_min"] is not None
    assert profile["sig_landed_per_min"] > 0


def test_profile_for_rows_finish_loss_rate_counts_ko_tko_sub_losses_only():
    rows = _rows([
        {"event_date": "2025-01-01", "round": 1, "time": "3:00", "result": "L", "method": "KO/TKO"},
        {"event_date": "2025-02-01", "round": 3, "time": "5:00", "result": "L", "method": "Decision"},
        {"event_date": "2025-03-01", "round": 2, "time": "1:00", "result": "W", "method": "Submission"},
    ])
    profile = _profile_for_rows(rows)
    assert profile["finish_loss_rate"] == pytest.approx(0.5)  # 1 of 2 losses was a finish


def test_profile_for_rows_zero_losses_gives_zero_finish_loss_rate():
    rows = _rows([{"event_date": "2025-01-01", "round": 1, "time": "3:00", "result": "W", "method": "KO/TKO"}])
    profile = _profile_for_rows(rows)
    assert profile["finish_loss_rate"] == 0.0


def test_profile_for_rows_position_shares_sum_to_one_when_present():
    rows = _rows([{
        "event_date": "2025-01-01", "round": 3, "time": "5:00",
        "ground_landed": 10, "clinch_landed": 5, "distance_landed": 5,
    }])
    profile = _profile_for_rows(rows)
    total = profile["ground_strike_share"] + profile["clinch_strike_share"] + profile["distance_strike_share"]
    assert total == pytest.approx(1.0)


def test_profile_for_rows_position_shares_none_when_no_position_data():
    rows = _rows([{"event_date": "2025-01-01", "round": 3, "time": "5:00"}])
    profile = _profile_for_rows(rows)
    assert profile["ground_strike_share"] is None


# --- fighter_performance lookup ----------------------------------------------

def test_fighter_performance_unknown_fighter_returns_zero_sample():
    table = pd.DataFrame([{"fighter": "Someone Else", "sample": 5}])
    result = fighter_performance(table, "Alpha")
    assert result["sample"] == 0


def test_fighter_performance_case_insensitive_match():
    table = pd.DataFrame([{"fighter": "Alpha", "sample": 5, "striking_score": 70.0}])
    result = fighter_performance(table, "ALPHA")
    assert result["sample"] == 5
    assert result["striking_score"] == 70.0


# --- matchup_performance_adjustment: weighting and caps ----------------------

def _profile(sample=10, completeness=1.0, **scores) -> dict:
    base = {
        "sample": sample, "data_completeness": completeness,
        "striking_score": 50.0, "wrestling_score": 50.0, "grappling_score": 50.0,
        "durability_score": 50.0, "pace_score": 50.0,
    }
    base.update(scores)
    return base


def test_matchup_adjustment_identical_profiles_are_neutral():
    result = matchup_performance_adjustment(_profile(), _profile())
    assert result["available"] is True
    assert result["adjustment_a"] == pytest.approx(0.0)


def test_matchup_adjustment_favors_the_stronger_profile():
    result = matchup_performance_adjustment(_profile(striking_score=90.0), _profile())
    assert result["adjustment_a"] > 0


def test_matchup_adjustment_capped_at_config_max():
    config = UFCPerformanceConfig(max_probability_adjustment=0.05)
    result = matchup_performance_adjustment(
        _profile(striking_score=100.0, wrestling_score=100.0, grappling_score=100.0, durability_score=100.0, pace_score=100.0),
        _profile(striking_score=0.0, wrestling_score=0.0, grappling_score=0.0, durability_score=0.0, pace_score=0.0),
        config=config,
    )
    assert abs(result["adjustment_a"]) <= config.max_probability_adjustment + 1e-9


def test_matchup_adjustment_unavailable_when_no_metrics_present():
    result = matchup_performance_adjustment({"sample": 0}, {"sample": 0})
    assert result["available"] is False
    assert result["adjustment_a"] == 0.0


def test_matchup_adjustment_reliability_scales_with_thin_sample():
    config = UFCPerformanceConfig(min_sample_for_full_weight=10)
    full = matchup_performance_adjustment(
        _profile(sample=10, striking_score=90.0), _profile(sample=10), config=config,
    )
    thin = matchup_performance_adjustment(
        _profile(sample=1, striking_score=90.0), _profile(sample=1), config=config,
    )
    assert thin["reliability"] < full["reliability"]
    assert abs(thin["adjustment_a"]) < abs(full["adjustment_a"])


def test_matchup_adjustment_uses_different_weights_for_5_round_fights():
    result_3r = matchup_performance_adjustment(_profile(durability_score=90.0), _profile(), rounds=3)
    result_5r = matchup_performance_adjustment(_profile(durability_score=90.0), _profile(), rounds=5)
    # Durability is weighted higher for 5-round fights (0.20 vs 0.15), so the
    # same gap should move the adjustment more in a 5-round context.
    assert abs(result_5r["adjustment_a"]) > abs(result_3r["adjustment_a"])
    assert result_5r["five_round_weighting"] is True
    assert result_3r["five_round_weighting"] is False


def test_matchup_adjustment_skips_missing_metrics_gracefully():
    a = {"sample": 10, "data_completeness": 0.6, "striking_score": 80.0}  # missing other metrics
    b = {"sample": 10, "data_completeness": 0.6, "striking_score": 40.0}
    result = matchup_performance_adjustment(a, b)
    assert result["available"] is True
    assert len(result["components"]) == 1
    assert result["adjustment_a"] > 0
