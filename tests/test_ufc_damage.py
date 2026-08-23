from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from engine.ufc_damage import (
    UFCDamageConfig,
    _as_date,
    _fight_minutes,
    _method_is_ko,
    _safe,
    build_damage_matchup,
    fighter_damage_profile,
)


# --- small helpers -----------------------------------------------------------

def test_safe_handles_missing_and_invalid():
    assert _safe("12") == 12.0
    assert _safe(None, default=0.0) == 0.0
    assert _safe("nope", default=-1.0) == -1.0
    assert _safe(float("nan"), default=0.0) == 0.0


def test_as_date_variants():
    assert _as_date(date(2025, 1, 1)) == date(2025, 1, 1)
    assert _as_date(datetime(2025, 1, 1, 5, 30)) == date(2025, 1, 1)
    assert _as_date("2025-01-01") == date(2025, 1, 1)


def test_as_date_none_defaults_to_today():
    assert _as_date(None) == date.today()


def test_fight_minutes_accumulates_full_prior_rounds():
    # 3 finished rounds (rnd=3) that ended at 2:30 -> 2 full rounds (10 min) + 2.5.
    assert _fight_minutes(3, "2:30") == pytest.approx(12.5)


def test_fight_minutes_never_zero():
    assert _fight_minutes(1, "0:00") == pytest.approx(1.0 / 60.0)


def test_method_is_ko_detection():
    assert _method_is_ko("KO/TKO") is True
    assert _method_is_ko("ko (punches)") is True
    assert _method_is_ko("Submission (rear-naked choke)") is False
    assert _method_is_ko(None) is False


# --- fighter_damage_profile: availability gating -----------------------------

def test_damage_profile_unavailable_without_event_date_or_fighter_columns():
    result = fighter_damage_profile(pd.DataFrame({"opponent": ["Beta"]}), "Alpha")
    assert result["available"] is False


def test_damage_profile_unavailable_with_no_history_on_or_before_fight_date():
    fights = pd.DataFrame([{
        "fighter": "Alpha", "opponent": "Beta", "fight_url": "f1", "event_date": "2026-01-01",
        "kd": 0, "head_landed": 10, "result": "W", "method": "Decision", "round": 3, "time": "5:00",
    }])
    # Asking about Alpha's risk profile as of a date *before* their only fight
    # must not use that fight at all.
    result = fighter_damage_profile(fights, "Alpha", fight_date=date(2025, 1, 1))
    assert result["available"] is False
    assert result["sample"] == 0


def _paired_fight(fight_url: str, event_date: str, a_name: str, b_name: str, *, a_result: str, a_method: str,
                   a_kd: float, b_kd: float, a_head_landed: float, b_head_landed: float) -> list[dict]:
    row_a = {
        "fighter": a_name, "opponent": b_name, "fight_url": fight_url, "event_date": event_date,
        "kd": a_kd, "head_landed": a_head_landed, "result": a_result, "method": a_method, "round": 3, "time": "5:00",
    }
    row_b = {
        "fighter": b_name, "opponent": a_name, "fight_url": fight_url, "event_date": event_date,
        "kd": b_kd, "head_landed": b_head_landed,
        "result": "L" if a_result == "W" else "W", "method": a_method, "round": 3, "time": "5:00",
    }
    return [row_a, row_b]


def test_damage_profile_unavailable_without_opponent_strike_detail():
    # No "kd"/"head_landed"/"sig_str_landed"/"sig_str" columns at all -> the
    # opponent-damage merge has nothing to attach.
    fights = pd.DataFrame([{
        "fighter": "Alpha", "opponent": "Beta", "fight_url": "f1", "event_date": "2025-01-01",
        "result": "W", "method": "Decision", "round": 3, "time": "5:00",
    }])
    result = fighter_damage_profile(fights, "Alpha", fight_date=date(2025, 6, 1))
    assert result["available"] is False


def test_damage_profile_future_fight_is_excluded_from_an_earlier_as_of_date():
    rows = []
    # An early, low-damage fight.
    rows += _paired_fight("f1", "2024-01-01", "Alpha", "Beta", a_result="W", a_method="Decision",
                           a_kd=0, b_kd=0, a_head_landed=5, b_head_landed=5)
    # A much later fight with heavy damage absorbed by Alpha -- must NOT
    # affect a risk profile requested as of a date before this fight happened.
    rows += _paired_fight("f2", "2026-06-01", "Alpha", "Gamma", a_result="L", a_method="TKO",
                           a_kd=0, b_kd=3, a_head_landed=5, b_head_landed=90)

    fights = pd.DataFrame(rows)
    profile = fighter_damage_profile(fights, "Alpha", fight_date=date(2025, 1, 1))
    assert profile["available"] is True
    assert profile["sample"] == 1  # only the 2024 fight should be visible
    assert profile["ko_tko_losses_last365"] == 0


def test_damage_profile_heavy_recent_damage_scores_higher_than_light_damage():
    config = UFCDamageConfig(min_sample_full_weight=1)
    heavy_rows = []
    for i in range(3):
        heavy_rows += _paired_fight(f"heavy{i}", f"2025-0{i+1}-01", "Alpha", f"Foe{i}", a_result="L", a_method="TKO",
                                     a_kd=0, b_kd=2, a_head_landed=10, b_head_landed=100)
    light_rows = []
    for i in range(3):
        light_rows += _paired_fight(f"light{i}", f"2025-0{i+1}-01", "Bravo", f"Foe{i}", a_result="W", a_method="Decision",
                                     a_kd=0, b_kd=0, a_head_landed=10, b_head_landed=10)

    heavy_profile = fighter_damage_profile(pd.DataFrame(heavy_rows), "Alpha", fight_date=date(2025, 6, 1), config=config)
    light_profile = fighter_damage_profile(pd.DataFrame(light_rows), "Bravo", fight_date=date(2025, 6, 1), config=config)

    assert heavy_profile["risk_score"] > light_profile["risk_score"]
    assert heavy_profile["risk_label"] in {"Moderate", "Elevated"}
    assert light_profile["risk_label"] == "Lower"


def test_damage_profile_risk_score_is_clipped_to_0_100():
    config = UFCDamageConfig(min_sample_full_weight=1)
    rows = []
    for i in range(3):
        rows += _paired_fight(f"f{i}", f"2025-0{i+1}-01", "Alpha", f"Foe{i}", a_result="L", a_method="KO",
                               a_kd=0, b_kd=10, a_head_landed=10, b_head_landed=500)
    profile = fighter_damage_profile(pd.DataFrame(rows), "Alpha", fight_date=date(2025, 6, 1), config=config)
    assert profile["risk_score"] <= 100.0


def test_damage_profile_recent_ko_loss_recovery_windows():
    config = UFCDamageConfig(min_sample_full_weight=1)
    rows = _paired_fight("f1", "2025-05-01", "Alpha", "Beta", a_result="L", a_method="KO",
                          a_kd=0, b_kd=1, a_head_landed=10, b_head_landed=60)
    profile = fighter_damage_profile(pd.DataFrame(rows), "Alpha", fight_date=date(2025, 6, 1), config=config)
    assert profile["days_since_last_ko_tko_loss"] == 31
    assert profile["ko_tko_losses_last365"] == 1


# --- build_damage_matchup: capped adjustment ---------------------------------

def test_damage_matchup_unavailable_when_either_fighter_lacks_data():
    fights = pd.DataFrame([{"fighter": "Alpha", "opponent": "Beta", "event_date": "2025-01-01"}])
    result = build_damage_matchup(fights, "Alpha", "Beta")
    assert result["available"] is False
    assert result["adjustment_a"] == 0.0


def test_damage_matchup_adjustment_capped_and_favors_lower_risk_fighter():
    config = UFCDamageConfig(min_sample_full_weight=1)
    rows = []
    for i in range(3):
        rows += _paired_fight(f"heavy{i}", f"2025-0{i+1}-01", "Alpha", f"Foe{i}", a_result="L", a_method="TKO",
                               a_kd=0, b_kd=2, a_head_landed=10, b_head_landed=100)
    for i in range(3):
        rows += _paired_fight(f"light{i}", f"2025-0{i+1}-01", "Bravo", f"Foe{i}", a_result="W", a_method="Decision",
                               a_kd=0, b_kd=0, a_head_landed=10, b_head_landed=10)
    fights = pd.DataFrame(rows)
    result = build_damage_matchup(fights, "Alpha", "Bravo", fight_date=date(2025, 6, 1), config=config)

    assert result["available"] is True
    assert abs(result["adjustment_a"]) <= config.max_probability_adjustment + 1e-9
    # Alpha carries more risk, so the lower-risk fighter (Bravo) is favored.
    assert result["advantage"] == "Bravo"
    assert result["adjustment_a"] < 0


def test_damage_matchup_even_when_risk_gap_is_small():
    config = UFCDamageConfig(min_sample_full_weight=1)
    rows = _paired_fight("f1", "2025-01-01", "Alpha", "Bravo", a_result="W", a_method="Decision",
                          a_kd=0, b_kd=0, a_head_landed=10, b_head_landed=10)
    fights = pd.DataFrame(rows)
    result = build_damage_matchup(fights, "Alpha", "Bravo", fight_date=date(2025, 6, 1), config=config)
    assert result["advantage"] == "Even"
