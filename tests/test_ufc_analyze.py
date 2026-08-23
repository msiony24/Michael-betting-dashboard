from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import engine.ufc as ufc


def _ratings_row(fighter: str, division: str = "Men's Lightweight", **overrides) -> dict:
    base = {
        "fighter": fighter, "division": division, "macabets_rating": 1500.0,
        "strength_score": 50.0, "ranking_confidence": 80.0, "active_pool": True,
        "division_rank": 1, "recent_form_adjusted": 50.0, "schedule_rating": 1500.0,
        "ufc_wins": 5, "ufc_losses": 2, "ufc_draws": 0, "ufc_finishes": 3,
        "division_fights": 6, "days_inactive": 60,
    }
    base.update(overrides)
    return base


def _fight_row(fighter: str, opponent: str, event_date: str, result: str, method: str = "Decision - Unanimous") -> dict:
    return {
        "fighter": fighter, "opponent": opponent, "event_date": event_date, "result": result,
        "method": method, "fight_url": f"{fighter}-{opponent}-{event_date}", "division": "Men's Lightweight",
        "round": 3, "time": "5:00",
    }


def _base_ratings() -> pd.DataFrame:
    return pd.DataFrame([_ratings_row("Alpha"), _ratings_row("Bravo")])


def _base_fights() -> pd.DataFrame:
    return pd.DataFrame([
        _fight_row("Alpha", "Nobody Known", "2025-01-01", "W"),
        _fight_row("Bravo", "Nobody Else", "2025-02-01", "W"),
    ])


def _neutral_layer(**overrides) -> dict:
    base = {"available": True, "adjustment_a": 0.0, "reliability": 1.0}
    base.update(overrides)
    return base


def _patch_all_layers(monkeypatch, *, performance=None, style=None, cardio=None, damage=None, context=None):
    """Replace every downstream builder analyze() calls with controlled fakes,
    isolating analyze()'s own aggregation/capping logic from the (already
    separately tested) internals of each layer.
    """
    monkeypatch.setattr(ufc, "build_performance_table", lambda fights, ratings: pd.DataFrame())
    monkeypatch.setattr(ufc, "fighter_performance", lambda table, fighter: {"fighter": fighter, "sample": 5})
    monkeypatch.setattr(
        ufc, "build_opponent_adjusted_matchup",
        lambda a, b, pa, pb, fights, ratings: {"available": True, "fighter_a_profile": pa, "fighter_b_profile": pb},
    )
    monkeypatch.setattr(ufc, "matchup_performance_adjustment", lambda pa, pb, rounds: performance or _neutral_layer())
    monkeypatch.setattr(ufc, "build_striking_table", lambda fights, ratings: pd.DataFrame())
    monkeypatch.setattr(ufc, "build_advanced_striking_matchup", lambda table, a, b: {"available": False, "rows": []})
    monkeypatch.setattr(ufc, "build_grappling_table", lambda fights, ratings: pd.DataFrame())
    monkeypatch.setattr(ufc, "build_advanced_grappling_matchup", lambda table, a, b: {"available": False, "rows": []})
    monkeypatch.setattr(
        ufc, "build_style_matchup",
        lambda pa, pb, a, b, *, rounds, advanced_grappling, advanced_striking: style or _neutral_layer(),
    )
    monkeypatch.setattr(ufc, "load_fighter_profiles", lambda path=None: pd.DataFrame())
    monkeypatch.setattr(
        ufc, "build_fight_context",
        lambda a, b, ratings, fights, *, profiles, rounds, fight_date: context or _neutral_layer(confidence_modifier=0),
    )
    monkeypatch.setattr(ufc, "build_cardio_matchup", lambda fights, a, b, *, rounds: cardio or _neutral_layer())
    monkeypatch.setattr(ufc, "build_damage_matchup", lambda fights, a, b, *, fight_date: damage or _neutral_layer())
    monkeypatch.setattr(
        ufc, "simulate_fight",
        lambda a, b, p, pa, pb, perf_a, perf_b, **kwargs: {"available": True},
    )
    monkeypatch.setattr(ufc, "build_derivative_markets", lambda simulation, a, b: {"available": False})
    monkeypatch.setattr(
        ufc, "evaluate_derivative_market",
        lambda markets, key, **kwargs: {"available": False},
    )


# --- basic error handling -----------------------------------------------------

def test_analyze_rejects_identical_fighters(monkeypatch):
    _patch_all_layers(monkeypatch)
    with pytest.raises(ufc.UFCAnalysisError):
        ufc.analyze("Alpha", "Alpha", ratings=_base_ratings(), fights=_base_fights())


def test_analyze_rejects_invalid_round_count(monkeypatch):
    _patch_all_layers(monkeypatch)
    with pytest.raises(ufc.UFCAnalysisError):
        ufc.analyze("Alpha", "Bravo", rounds=4, ratings=_base_ratings(), fights=_base_fights())


def test_analyze_rejects_unknown_fighter(monkeypatch):
    _patch_all_layers(monkeypatch)
    with pytest.raises(ufc.UFCAnalysisError):
        ufc.analyze("Alpha", "Nobody Real", ratings=_base_ratings(), fights=_base_fights())


# --- baseline Elo probability + reliability shrinkage ------------------------

def test_analyze_equal_ratings_gives_50_50_baseline(monkeypatch):
    _patch_all_layers(monkeypatch)
    result = ufc.analyze("Alpha", "Bravo", ratings=_base_ratings(), fights=_base_fights())
    assert result["raw_rating_probability_a"] == pytest.approx(0.5)
    assert result["ranking_baseline_probability_a"] == pytest.approx(0.5)


def test_analyze_low_confidence_shrinks_baseline_toward_50_50(monkeypatch):
    _patch_all_layers(monkeypatch)
    ratings = pd.DataFrame([
        _ratings_row("Alpha", macabets_rating=1700.0, ranking_confidence=20.0),
        _ratings_row("Bravo", macabets_rating=1500.0, ranking_confidence=20.0),
    ])
    result = ufc.analyze("Alpha", "Bravo", ratings=ratings, fights=_base_fights())
    raw = result["raw_rating_probability_a"]
    shrunk = result["ranking_baseline_probability_a"]
    # Low ranking confidence should pull the baseline closer to 0.5 than the
    # raw Elo gap alone would suggest.
    assert abs(shrunk - 0.5) < abs(raw - 0.5)


def test_analyze_high_confidence_barely_shrinks_baseline(monkeypatch):
    _patch_all_layers(monkeypatch)
    ratings = pd.DataFrame([
        _ratings_row("Alpha", macabets_rating=1700.0, ranking_confidence=100.0),
        _ratings_row("Bravo", macabets_rating=1500.0, ranking_confidence=100.0),
    ])
    result = ufc.analyze("Alpha", "Bravo", ratings=ratings, fights=_base_fights())
    raw = result["raw_rating_probability_a"]
    shrunk = result["ranking_baseline_probability_a"]
    assert shrunk == pytest.approx(raw, abs=0.001)


# --- the layered adjustment cap chain: the highest-stakes property here -----

def test_analyze_combined_matchup_adjustment_is_capped_at_8point5_points(monkeypatch):
    # Feed every correlated layer (performance, style, cardio, damage) an
    # extreme adjustment in the same direction; the combined result must
    # never exceed the documented ±8.5pp cap.
    _patch_all_layers(
        monkeypatch,
        performance=_neutral_layer(adjustment_a=0.5),
        style=_neutral_layer(adjustment_a=0.5),
        cardio=_neutral_layer(adjustment_a=0.5),
        damage=_neutral_layer(adjustment_a=0.5),
    )
    result = ufc.analyze("Alpha", "Bravo", ratings=_base_ratings(), fights=_base_fights())
    assert abs(result["combined_matchup_adjustment_a"]) <= 0.085 + 1e-9


def test_analyze_total_adjustment_is_capped_at_10_points_even_with_extreme_context(monkeypatch):
    _patch_all_layers(
        monkeypatch,
        performance=_neutral_layer(adjustment_a=0.5),
        style=_neutral_layer(adjustment_a=0.5),
        cardio=_neutral_layer(adjustment_a=0.5),
        damage=_neutral_layer(adjustment_a=0.5),
        context=_neutral_layer(adjustment_a=0.5, confidence_modifier=0),
    )
    result = ufc.analyze("Alpha", "Bravo", ratings=_base_ratings(), fights=_base_fights())
    assert abs(result["total_adjustment_a"]) <= 0.10 + 1e-9


def test_analyze_final_probability_stays_within_0_08_and_0_92_under_extreme_inputs(monkeypatch):
    _patch_all_layers(
        monkeypatch,
        performance=_neutral_layer(adjustment_a=0.9),
        style=_neutral_layer(adjustment_a=0.9),
        cardio=_neutral_layer(adjustment_a=0.9),
        damage=_neutral_layer(adjustment_a=0.9),
        context=_neutral_layer(adjustment_a=0.9),
    )
    ratings = pd.DataFrame([
        _ratings_row("Alpha", macabets_rating=2200.0),  # a huge, unrealistic Elo gap too
        _ratings_row("Bravo", macabets_rating=800.0),
    ])
    result = ufc.analyze("Alpha", "Bravo", ratings=ratings, fights=_base_fights())
    assert 0.08 - 1e-9 <= result["win_probability_a"] <= 0.92 + 1e-9
    assert result["win_probability_a"] + result["win_probability_b"] == pytest.approx(1.0)


def test_analyze_negative_extreme_layers_are_also_capped(monkeypatch):
    _patch_all_layers(
        monkeypatch,
        performance=_neutral_layer(adjustment_a=-0.5),
        style=_neutral_layer(adjustment_a=-0.5),
        cardio=_neutral_layer(adjustment_a=-0.5),
        damage=_neutral_layer(adjustment_a=-0.5),
    )
    result = ufc.analyze("Alpha", "Bravo", ratings=_base_ratings(), fights=_base_fights())
    assert abs(result["combined_matchup_adjustment_a"]) <= 0.085 + 1e-9
    assert result["combined_matchup_adjustment_a"] < 0


def test_analyze_zero_adjustments_leave_probability_at_baseline(monkeypatch):
    _patch_all_layers(monkeypatch)
    ratings = pd.DataFrame([
        _ratings_row("Alpha", macabets_rating=1600.0),
        _ratings_row("Bravo", macabets_rating=1500.0),
    ])
    result = ufc.analyze("Alpha", "Bravo", ratings=ratings, fights=_base_fights())
    assert result["win_probability_a"] == pytest.approx(result["ranking_baseline_probability_a"], abs=1e-9)


# --- winner selection ---------------------------------------------------------

def test_analyze_winner_is_whichever_side_has_higher_final_probability(monkeypatch):
    _patch_all_layers(monkeypatch, performance=_neutral_layer(adjustment_a=0.3))
    ratings = pd.DataFrame([_ratings_row("Alpha", macabets_rating=1450.0), _ratings_row("Bravo", macabets_rating=1550.0)])
    result = ufc.analyze("Alpha", "Bravo", ratings=ratings, fights=_base_fights())
    if result["win_probability_a"] >= result["win_probability_b"]:
        assert result["projected_winner"] == "Alpha"
    else:
        assert result["projected_winner"] == "Bravo"
    assert result["projected_winner_probability"] == max(result["win_probability_a"], result["win_probability_b"])
