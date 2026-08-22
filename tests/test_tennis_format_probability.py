from __future__ import annotations

import pytest

from engine.tennis import simulate_matches


def test_format_probability_is_deterministic():
    first = simulate_matches(0.70, 20_000, True)
    second = simulate_matches(0.70, 20_000, True)
    assert first == second
    assert first["method"] == "exact_format_probability"


def test_best_of_five_exact_score_distribution_sums_to_one():
    result = simulate_matches(0.63, 20_000, True)
    assert sum(result["set_scores"].values()) == pytest.approx(1.0)
    assert result["win_probability"] == pytest.approx(
        result["set_scores"]["3-0"]
        + result["set_scores"]["3-1"]
        + result["set_scores"]["3-2"]
    )
    assert result["deciding_set"] == pytest.approx(
        result["set_scores"]["3-2"] + result["set_scores"]["2-3"]
    )


def test_best_of_five_rewards_the_stronger_player_more_than_best_of_three():
    bo3 = simulate_matches(0.70, 20_000, False)["win_probability"]
    bo5 = simulate_matches(0.70, 20_000, True)["win_probability"]
    assert bo5 > bo3 > 0.5


def test_even_match_stays_even_in_both_formats():
    assert simulate_matches(0.50, 20_000, False)["win_probability"] == pytest.approx(0.5)
    assert simulate_matches(0.50, 20_000, True)["win_probability"] == pytest.approx(0.5)


def test_historical_decider_definition_is_format_aware():
    import pandas as pd
    from engine.tennis import _historical_match_is_decider

    assert not _historical_match_is_decider(pd.Series({
        "level": "G", "round": "R128", "score": "6-4 6-4 6-4"
    }))
    assert _historical_match_is_decider(pd.Series({
        "level": "G", "round": "R128", "score": "6-4 4-6 6-3 3-6 6-2"
    }))
    assert _historical_match_is_decider(pd.Series({
        "level": "A", "round": "R32", "score": "6-4 4-6 6-2"
    }))
    assert not _historical_match_is_decider(pd.Series({
        "level": "A", "round": "R32", "score": "6-4 6-2"
    }))
