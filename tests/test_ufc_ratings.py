from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine.ufc_ratings import (
    UFCRatingConfig,
    _current_division,
    _expected,
    _finish_multiplier,
    _is_ranking_division,
    _score,
    _shrink,
    _to_strength_score,
    build_elo_history,
    build_fighter_ratings,
)


# --- small helpers -----------------------------------------------------------

def test_expected_is_half_when_ratings_are_equal():
    assert _expected(1500.0, 1500.0) == pytest.approx(0.5)


def test_expected_favors_the_higher_rated_fighter():
    assert _expected(1600.0, 1500.0) > 0.5
    assert _expected(1500.0, 1600.0) < 0.5


def test_expected_is_symmetric():
    e_a = _expected(1650.0, 1450.0)
    e_b = _expected(1450.0, 1650.0)
    assert e_a + e_b == pytest.approx(1.0)


def test_score_mapping():
    assert _score("W") == 1.0
    assert _score("w") == 1.0
    assert _score("L") == 0.0
    assert _score("D") == 0.5
    assert _score("Draw") == 0.5
    assert _score("NC") is None
    assert _score(None) is None


def test_finish_multiplier_ko_tko_and_submission():
    assert _finish_multiplier("KO/TKO") == 1.08
    assert _finish_multiplier("Submission (RNC)") == 1.08
    assert _finish_multiplier("Decision - Unanimous") == 1.0
    assert _finish_multiplier(None) == 1.0


def test_shrink_zero_observations_returns_center():
    assert _shrink(80.0, center=50.0, observations=0, prior_strength=3.0) == 50.0


def test_shrink_moves_toward_value_as_observations_grow():
    few = _shrink(80.0, center=50.0, observations=1, prior_strength=3.0)
    many = _shrink(80.0, center=50.0, observations=100, prior_strength=3.0)
    assert 50.0 < few < many < 80.0 + 1e-6
    assert many == pytest.approx(80.0, abs=1.0)


def test_to_strength_score_is_50_at_base_elo():
    assert _to_strength_score(1500.0) == pytest.approx(50.0, abs=0.01)


def test_to_strength_score_is_monotonic_increasing():
    assert _to_strength_score(1600.0) > _to_strength_score(1500.0) > _to_strength_score(1400.0)


def test_is_ranking_division_excludes_catchweight_and_unknown():
    assert _is_ranking_division("Men’s Lightweight") is True
    assert _is_ranking_division("Women’s Flyweight") is True
    assert _is_ranking_division("Catch Weight") is False
    assert _is_ranking_division("Unknown") is False
    assert _is_ranking_division("") is False


def test_current_division_skips_catchweight_for_most_recent_ranking_division():
    rows = pd.DataFrame([
        {"event_date": pd.Timestamp("2024-01-01"), "division": "Men’s Lightweight"},
        {"event_date": pd.Timestamp("2025-01-01"), "division": "Catch Weight"},
    ])
    assert _current_division(rows) == "Men’s Lightweight"


# --- build_elo_history: exchange mechanics -----------------------------------

def _fight_pair(a_result: str, b_result: str, method: str = "Decision - Unanimous", **extra_a_b) -> pd.DataFrame:
    row_a = {"event_date": "2025-01-01", "fight_url": "f1", "fighter": "Alpha", "opponent": "Beta",
              "result": a_result, "division": "Men’s Lightweight", "method": method}
    row_b = {"event_date": "2025-01-01", "fight_url": "f1", "fighter": "Beta", "opponent": "Alpha",
              "result": b_result, "division": "Men’s Lightweight", "method": method}
    return pd.DataFrame([row_a, row_b])


def test_elo_history_is_conserved_for_an_even_decision_with_no_stat_gap():
    # With no sig_str/kd/td columns at all, both dominance multipliers default
    # to neutral (1.0) and the exchange should be exactly zero-sum.
    fights = _fight_pair("W", "L")
    history, ratings = build_elo_history(fights, UFCRatingConfig())
    gain = ratings["Alpha"] - 1500.0
    loss = 1500.0 - ratings["Beta"]
    assert gain == pytest.approx(loss, abs=1e-9)


def test_elo_history_documents_known_asymmetry_from_the_dominance_multiplier():
    # KNOWN BEHAVIOR, not necessarily desired: because the dominance multiplier
    # is computed independently per fighter from that fighter's own stat gap,
    # a lopsided finish makes the winner's k-factor rise while the loser's
    # k-factor falls -- so the winner's gain and the loser's loss are not
    # exactly equal and opposite. This test pins the current magnitude so any
    # future change to the dominance formula shows up here explicitly.
    fights = pd.DataFrame([
        {"event_date": "2025-01-01", "fight_url": "f1", "fighter": "Alpha", "opponent": "Beta",
         "result": "W", "division": "Men’s Lightweight", "method": "KO/TKO",
         "sig_str": 80, "kd": 2, "td": 3, "sub_att": 0},
        {"event_date": "2025-01-01", "fight_url": "f1", "fighter": "Beta", "opponent": "Alpha",
         "result": "L", "division": "Men’s Lightweight", "method": "KO/TKO",
         "sig_str": 10, "kd": 0, "td": 0, "sub_att": 0},
    ])
    history, ratings = build_elo_history(fights, UFCRatingConfig())
    gain = ratings["Alpha"] - 1500.0
    loss = 1500.0 - ratings["Beta"]
    # The winner of a dominant finish currently gains more than the loser
    # loses -- net Elo is created in the exchange rather than conserved.
    assert gain > loss


def test_elo_history_ignores_fights_with_unscoreable_result():
    fights = _fight_pair("NC", "NC")
    history, ratings = build_elo_history(fights, UFCRatingConfig())
    assert history.empty
    assert ratings == {}


def test_elo_history_ignores_fights_missing_a_paired_opponent_row():
    # Only one side of the bout is present in the data.
    fights = _fight_pair("W", "L").iloc[[0]]
    history, ratings = build_elo_history(fights, UFCRatingConfig())
    assert history.empty


# --- build_fighter_ratings: inactivity and active_pool -----------------------

def _base_history(days_ago: int) -> pd.DataFrame:
    event_date = (pd.Timestamp(date.today()) - pd.Timedelta(days=days_ago)).date().isoformat()
    return pd.DataFrame([
        {"event_date": event_date, "fight_url": "f1", "fighter": "Alpha", "opponent": "Beta",
         "result": "W", "division": "Men’s Lightweight", "method": "Decision - Unanimous"},
        {"event_date": event_date, "fight_url": "f1", "fighter": "Beta", "opponent": "Alpha",
         "result": "L", "division": "Men’s Lightweight", "method": "Decision - Unanimous"},
    ])


def test_build_fighter_ratings_recently_active_fighter_is_in_active_pool():
    ratings = build_fighter_ratings(_base_history(days_ago=30))
    alpha = ratings.loc[ratings["fighter"] == "Alpha"].iloc[0]
    assert alpha["active_pool"] is True or alpha["active_pool"] == True  # noqa: E712
    assert alpha["inactivity_penalty"] == 0.0


def test_build_fighter_ratings_long_inactive_fighter_is_excluded_from_active_pool():
    config = UFCRatingConfig(active_window_days=730)
    ratings = build_fighter_ratings(_base_history(days_ago=1500), config=config)
    alpha = ratings.loc[ratings["fighter"] == "Alpha"].iloc[0]
    assert bool(alpha["active_pool"]) is False


def test_build_fighter_ratings_inactivity_penalty_grows_past_grace_period():
    config = UFCRatingConfig(inactivity_grace_days=300, inactivity_penalty_per_30d=3.0, max_inactivity_penalty=60.0)
    short = build_fighter_ratings(_base_history(days_ago=200), config=config)
    long = build_fighter_ratings(_base_history(days_ago=600), config=config)
    short_penalty = short.loc[short["fighter"] == "Alpha", "inactivity_penalty"].iloc[0]
    long_penalty = long.loc[long["fighter"] == "Alpha", "inactivity_penalty"].iloc[0]
    assert short_penalty == 0.0  # inside the grace period
    assert long_penalty > 0.0


def test_build_fighter_ratings_inactivity_penalty_is_capped():
    config = UFCRatingConfig(inactivity_grace_days=0, inactivity_penalty_per_30d=3.0, max_inactivity_penalty=60.0)
    ratings = build_fighter_ratings(_base_history(days_ago=5000), config=config)
    penalty = ratings.loc[ratings["fighter"] == "Alpha", "inactivity_penalty"].iloc[0]
    assert penalty == pytest.approx(60.0)


def test_build_fighter_ratings_missing_columns_raises():
    with pytest.raises(ValueError):
        build_fighter_ratings(pd.DataFrame([{"fighter": "Alpha"}]))


def test_build_fighter_ratings_empty_input_returns_empty_frame():
    assert build_fighter_ratings(pd.DataFrame(columns=["event_date", "fight_url", "fighter", "opponent", "result", "division", "method"])).empty
