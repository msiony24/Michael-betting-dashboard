from __future__ import annotations

import pytest

from engine.ufc_style_matchups import (
    UFCStyleConfig,
    _advantage,
    _archetype,
    _attack_vs_defense,
    _edge_label,
    build_style_matchup,
)


# --- small helpers -----------------------------------------------------------

def test_attack_vs_defense_computes_residual():
    attack_profile = {"sig_accuracy_pct": 70.0, "kd_per15_pct": 60.0}
    defense_profile = {"sig_defense_pct": 50.0, "kd_absorbed_per15_pct": 50.0}
    gap, used = _attack_vs_defense(attack_profile, defense_profile, ["sig_accuracy_pct", "kd_per15_pct"], ["sig_defense_pct", "kd_absorbed_per15_pct"])
    assert gap == pytest.approx(65.0 - 50.0)
    assert used == 2


def test_attack_vs_defense_none_when_either_side_missing():
    gap, used = _attack_vs_defense({}, {"sig_defense_pct": 50.0}, ["sig_accuracy_pct"], ["sig_defense_pct"])
    assert gap is None


def test_edge_label_boundaries():
    assert _edge_label(4.9) == "Even"
    assert _edge_label(5.0) == "Slight"
    assert _edge_label(12.0) == "Moderate"
    assert _edge_label(22.0) == "Clear"


def test_advantage_threshold_at_5():
    assert _advantage(4.9, "A", "B") == "Even"
    assert _advantage(5.0, "A", "B") == "A"
    assert _advantage(-5.0, "A", "B") == "B"


# --- _archetype classification ------------------------------------------------

def test_archetype_ground_pressure_wrestler():
    profile = {"ground_strike_share": 0.30, "td_per15_pct": 65.0}
    assert _archetype(profile) == "Ground-pressure wrestler"


def test_archetype_submission_oriented_grappler():
    profile = {"sub_attempts_per15_pct": 75.0, "td_per15_pct": 55.0}
    assert _archetype(profile) == "Submission-oriented grappler"


def test_archetype_clinch_pressure_fighter():
    profile = {"clinch_strike_share": 0.25}
    assert _archetype(profile) == "Clinch-pressure fighter"


def test_archetype_distance_striker():
    profile = {"distance_strike_share": 0.75}
    assert _archetype(profile) == "Distance striker"


def test_archetype_high_pace_pressure_fighter():
    profile = {"pace_score": 75.0}
    assert _archetype(profile) == "High-pace pressure fighter"


def test_archetype_balanced_when_nothing_stands_out():
    profile = {"distance_strike_share": 0.40, "clinch_strike_share": 0.10, "ground_strike_share": 0.10, "pace_score": 50.0}
    assert _archetype(profile) == "Balanced / mixed style"


def test_archetype_empty_profile_is_balanced():
    assert _archetype({}) == "Balanced / mixed style"


# --- build_style_matchup: spec selection and combined cap --------------------

def _percentile_profile(sample=10, completeness=1.0, **overrides) -> dict:
    base = {
        "sample": sample, "data_completeness": completeness,
        "sig_accuracy_pct": 50.0, "kd_per15_pct": 50.0, "sig_defense_pct": 50.0, "kd_absorbed_per15_pct": 50.0,
        "td_per15_pct": 50.0, "td_accuracy_pct": 50.0, "control_share_pct": 50.0, "td_defense_pct": 50.0,
        "sub_attempts_per15_pct": 50.0, "durability_score": 50.0, "pace_score": 50.0,
    }
    base.update(overrides)
    return base


def test_build_style_matchup_unavailable_with_empty_profiles():
    result = build_style_matchup({}, {}, "Alpha", "Bravo")
    assert result["available"] is False
    assert result["adjustment_a"] == 0.0
    # Archetypes should still be computed even when unavailable.
    assert "fighter_a_archetype" in result


def test_build_style_matchup_identical_profiles_are_neutral():
    result = build_style_matchup(_percentile_profile(), _percentile_profile(), "Alpha", "Bravo")
    assert result["available"] is True
    assert result["adjustment_a"] == pytest.approx(0.0)


def test_build_style_matchup_uses_full_4_spec_generic_set_without_advanced_layers():
    result = build_style_matchup(_percentile_profile(), _percentile_profile(), "Alpha", "Bravo")
    categories = {row["category"] for row in result["rows"]}
    assert "Wrestling pressure vs takedown defense" in categories
    assert "Grappling threat vs defensive resistance" in categories


def test_build_style_matchup_drops_generic_wrestling_and_grappling_rows_when_advanced_grappling_available():
    advanced_grappling = {"available": True, "rows": []}
    result = build_style_matchup(
        _percentile_profile(), _percentile_profile(), "Alpha", "Bravo",
        advanced_grappling=advanced_grappling,
    )
    categories = {row["category"] for row in result["rows"]}
    # The generic wrestling/grappling rows are replaced, not kept alongside advanced ones.
    assert "Wrestling pressure vs takedown defense" not in categories
    assert "Grappling threat vs defensive resistance" not in categories


def test_build_style_matchup_keeps_generic_striking_row_when_advanced_striking_unavailable():
    advanced_grappling = {"available": True, "rows": []}
    result = build_style_matchup(
        _percentile_profile(), _percentile_profile(), "Alpha", "Bravo",
        advanced_grappling=advanced_grappling,
    )
    categories = {row["category"] for row in result["rows"]}
    assert "Striking offense vs defense" in categories


def test_build_style_matchup_favors_the_stronger_striker():
    profile_a = _percentile_profile(sig_accuracy_pct=90.0, kd_per15_pct=90.0)
    result = build_style_matchup(profile_a, _percentile_profile(), "Alpha", "Bravo")
    assert result["adjustment_a"] > 0


def test_build_style_matchup_combined_adjustment_capped_with_extreme_advanced_rows():
    config = UFCStyleConfig(max_probability_adjustment=0.03)
    extreme_advanced_striking = {
        "available": True,
        "rows": [{"category": "Head attack vs head defense", "advantage": "Alpha", "strength": "Clear",
                   "interaction_gap": 100.0, "weight": 1.0, "why": "test"}],
    }
    extreme_advanced_grappling = {
        "available": True,
        "rows": [{"category": "Chain wrestling pressure vs resistance", "advantage": "Alpha", "strength": "Clear",
                   "interaction_gap": 100.0, "weight": 1.0, "why": "test"}],
    }
    result = build_style_matchup(
        _percentile_profile(sig_accuracy_pct=100.0, td_per15_pct=100.0),
        _percentile_profile(sig_accuracy_pct=0.0, td_per15_pct=0.0),
        "Alpha", "Bravo",
        advanced_striking=extreme_advanced_striking,
        advanced_grappling=extreme_advanced_grappling,
        config=config,
    )
    assert abs(result["adjustment_a"]) <= config.max_probability_adjustment + 1e-9


def test_build_style_matchup_reliability_scales_with_thin_sample():
    config = UFCStyleConfig(min_sample_for_full_weight=10)
    full = build_style_matchup(
        _percentile_profile(sample=10, sig_accuracy_pct=90.0), _percentile_profile(sample=10), "Alpha", "Bravo", config=config,
    )
    thin = build_style_matchup(
        _percentile_profile(sample=1, sig_accuracy_pct=90.0), _percentile_profile(sample=1), "Alpha", "Bravo", config=config,
    )
    assert thin["reliability"] < full["reliability"]


def test_build_style_matchup_five_round_flag_reflects_input():
    result = build_style_matchup(_percentile_profile(), _percentile_profile(), "Alpha", "Bravo", rounds=5)
    assert result["five_round_weighting"] is True
