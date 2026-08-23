from __future__ import annotations

import numpy as np
import pytest

from engine.ufc_simulation import (
    UFCSimulationConfig,
    _finish_profile,
    _num,
    _pct01,
    _round_weights,
    simulate_fight,
)


# --- small helpers -----------------------------------------------------------

def test_num_handles_missing_and_invalid():
    assert _num({"x": "3.5"}, "x") == 3.5
    assert _num({}, "x", default=1.0) == 1.0
    assert _num({"x": "bad"}, "x", default=1.0) == 1.0
    assert _num({"x": float("nan")}, "x", default=1.0) == 1.0


def test_pct01_treats_values_over_1_as_percentile_scale():
    assert _pct01(75.0) == pytest.approx(0.75)


def test_pct01_treats_values_at_or_under_1_as_already_a_rate():
    assert _pct01(0.4) == pytest.approx(0.4)


def test_pct01_none_returns_default():
    assert _pct01(None, default=0.33) == 0.33


def test_pct01_clips_to_0_1():
    assert _pct01(150.0) == pytest.approx(1.0)


# --- _finish_profile bounds ---------------------------------------------------

def _empty_dict() -> dict:
    return {}


def test_finish_profile_bounded_within_config_limits_3_round():
    config = UFCSimulationConfig()
    profile = _finish_profile(_empty_dict(), _empty_dict(), _empty_dict(), _empty_dict(), rounds=3, config=config)
    assert config.min_finish_probability <= profile["finish_probability"] <= config.max_finish_probability_3r


def test_finish_profile_bounded_within_config_limits_5_round():
    config = UFCSimulationConfig()
    profile = _finish_profile(_empty_dict(), _empty_dict(), _empty_dict(), _empty_dict(), rounds=5, config=config)
    assert config.min_finish_probability <= profile["finish_probability"] <= config.max_finish_probability_5r


def test_finish_profile_5_round_finish_probability_at_least_as_high_as_3_round():
    config = UFCSimulationConfig()
    p3 = _finish_profile(_empty_dict(), _empty_dict(), _empty_dict(), _empty_dict(), rounds=3, config=config)
    p5 = _finish_profile(_empty_dict(), _empty_dict(), _empty_dict(), _empty_dict(), rounds=5, config=config)
    assert p5["finish_probability"] >= p3["finish_probability"]


def test_finish_profile_ko_and_submission_shares_sum_to_one_and_are_bounded():
    config = UFCSimulationConfig()
    profile = _finish_profile({"ko_win_rate": 0.9, "submission_win_rate": 0.05}, _empty_dict(), _empty_dict(), _empty_dict(), rounds=3, config=config)
    assert profile["ko_share_of_finishes"] + profile["submission_share_of_finishes"] == pytest.approx(1.0)
    assert 0.15 <= profile["ko_share_of_finishes"] <= 0.92


def test_finish_profile_high_own_finish_rate_raises_finish_probability():
    config = UFCSimulationConfig()
    low = _finish_profile({"finish_rate": 0.1}, _empty_dict(), _empty_dict(), _empty_dict(), rounds=3, config=config)
    high = _finish_profile({"finish_rate": 0.9}, _empty_dict(), _empty_dict(), _empty_dict(), rounds=3, config=config)
    assert high["finish_probability"] > low["finish_probability"]


# --- _round_weights: always a valid probability distribution ----------------

def test_round_weights_sums_to_one_and_correct_length():
    weights = _round_weights(3, 50.0, 50.0)
    assert len(weights) == 3
    assert weights.sum() == pytest.approx(1.0)

    weights5 = _round_weights(5, 50.0, 50.0)
    assert len(weights5) == 5
    assert weights5.sum() == pytest.approx(1.0)


def test_round_weights_front_loaded_by_default():
    weights = _round_weights(3, 50.0, 50.0)
    assert weights[0] > weights[1] > weights[2]


def test_round_weights_poor_cardio_shifts_mass_later():
    neutral = _round_weights(3, 50.0, 50.0)
    fading = _round_weights(3, 50.0, 50.0, cardio_a={"available": True, "retention": 0.7, "reliability": 1.0})
    # Poor retention (fading) should raise the relative share of the final round.
    assert (fading[-1] / fading[0]) > (neutral[-1] / neutral[0])


def test_round_weights_ignores_unavailable_cardio():
    neutral = _round_weights(3, 50.0, 50.0)
    ignored = _round_weights(3, 50.0, 50.0, cardio_a={"available": False, "retention": 0.5})
    assert np.allclose(neutral, ignored)


# --- simulate_fight: the core invariant --------------------------------------

def _neutral_profile() -> dict:
    return {}


def test_simulate_fight_rejects_invalid_round_count():
    with pytest.raises(ValueError):
        simulate_fight("Alpha", "Bravo", 0.6, {}, {}, {}, {}, rounds=4)


def test_simulate_fight_preserves_the_input_win_probability():
    # This is the central correctness property: the simulator decomposes an
    # already-decided win probability into methods/rounds -- it must not
    # silently drift the win probability itself, even after Monte Carlo
    # sampling noise with a large sample size.
    result = simulate_fight(
        "Alpha", "Bravo", 0.65,
        {"finish_rate": 0.4}, {"finish_rate": 0.4},
        {"pace_score": 50.0}, {"pace_score": 50.0},
        rounds=3,
    )
    assert result["model_win_probability_a"] == pytest.approx(0.65)
    assert result["simulated_win_probability_a"] == pytest.approx(0.65, abs=0.02)


def test_simulate_fight_clips_extreme_input_probabilities():
    result_high = simulate_fight("Alpha", "Bravo", 1.0, {}, {}, {}, {}, rounds=3)
    assert result_high["model_win_probability_a"] <= 0.99
    result_low = simulate_fight("Alpha", "Bravo", 0.0, {}, {}, {}, {}, rounds=3)
    assert result_low["model_win_probability_a"] >= 0.01


def test_simulate_fight_goes_distance_equals_one_minus_finish_probability():
    result = simulate_fight("Alpha", "Bravo", 0.55, {}, {}, {}, {}, rounds=3)
    assert result["goes_distance_probability"] == pytest.approx(1.0 - result["finish_probability"])


def test_simulate_fight_method_probabilities_sum_to_one():
    result = simulate_fight("Alpha", "Bravo", 0.55, {}, {}, {}, {}, rounds=3)
    total = (
        result["a_ko_tko_probability"] + result["a_submission_probability"] + result["a_decision_probability"]
        + result["b_ko_tko_probability"] + result["b_submission_probability"] + result["b_decision_probability"]
    )
    assert total == pytest.approx(1.0)


def test_simulate_fight_most_likely_path_is_the_actual_max():
    result = simulate_fight("Alpha", "Bravo", 0.8, {"finish_rate": 0.6}, {}, {}, {}, rounds=3)
    paths = {
        f"Alpha by KO/TKO": result["a_ko_tko_probability"],
        f"Alpha by Submission": result["a_submission_probability"],
        f"Alpha by Decision": result["a_decision_probability"],
        f"Bravo by KO/TKO": result["b_ko_tko_probability"],
        f"Bravo by Submission": result["b_submission_probability"],
        f"Bravo by Decision": result["b_decision_probability"],
    }
    expected_label = max(paths, key=paths.get)
    assert result["most_likely_path"] == expected_label
    assert result["most_likely_path_probability"] == pytest.approx(paths[expected_label])


def test_simulate_fight_finish_round_probabilities_sum_to_one():
    result = simulate_fight("Alpha", "Bravo", 0.7, {"finish_rate": 0.6}, {}, {}, {}, rounds=3)
    total = sum(result["finish_round_probabilities_given_finish"].values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_simulate_fight_is_deterministic_for_a_fixed_seed():
    config = UFCSimulationConfig(seed=123, simulations=5000)
    result1 = simulate_fight("Alpha", "Bravo", 0.6, {}, {}, {}, {}, rounds=3, config=config)
    result2 = simulate_fight("Alpha", "Bravo", 0.6, {}, {}, {}, {}, rounds=3, config=config)
    assert result1["simulated_win_probability_a"] == result2["simulated_win_probability_a"]
    assert result1["most_likely_path"] == result2["most_likely_path"]


def test_simulate_fight_volatility_label_reflects_concentration():
    lopsided = simulate_fight("Alpha", "Bravo", 0.95, {"finish_rate": 0.9}, {}, {}, {}, rounds=3)
    close = simulate_fight("Alpha", "Bravo", 0.51, {}, {}, {}, {}, rounds=3)
    assert lopsided["volatility"] == "Lower"
    assert close["volatility"] in {"Moderate", "High"}


def test_simulate_fight_flags_reflect_cardio_and_damage_usage():
    no_extras = simulate_fight("Alpha", "Bravo", 0.6, {}, {}, {}, {}, rounds=3)
    assert no_extras["cardio_timing_used"] is False
    assert no_extras["damage_method_context_used"] is False

    with_cardio = simulate_fight(
        "Alpha", "Bravo", 0.6, {}, {}, {}, {},
        cardio_a={"available": True, "retention": 0.9, "reliability": 0.8}, rounds=3,
    )
    assert with_cardio["cardio_timing_used"] is True

    with_damage = simulate_fight(
        "Alpha", "Bravo", 0.6, {}, {}, {}, {},
        damage_b={"available": True, "risk_score": 60.0, "reliability": 0.7}, rounds=3,
    )
    assert with_damage["damage_method_context_used"] is True
