from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.matchup_engine import (
    _abbreviated_name_candidates,
    _confidence,
    _name_tokens,
    _severity,
    _winner,
    analyze_matchup,
    evaluate_baseline,
    evaluate_movement,
    evaluate_return,
    evaluate_serve,
    evaluate_surface,
    evaluate_variety,
    resolve_player_profile,
)
from engine.player_traits import PlayerTraitsDatabase

# --- shared fixtures -------------------------------------------------------

METADATA = {
    "version": "test",
    "rating_scale": {"1": "Below Average", "2": "Average", "3": "Good", "4": "Very Good", "5": "Elite"},
    "valid_playing_styles": ["Baseliner"],
    "valid_court_positions": ["Baseline"],
    "valid_rally_preferences": ["Short", "Short-Medium", "Medium", "Medium-Long", "Long"],
    "valid_signature_traits": [
        "Big Server", "Serve Plus One", "Varied Serving Patterns", "Elite Returner",
        "Baseline Pressure", "Excellent Depth", "Early Ball Striker", "Heavy Topspin",
        "Flat Ball Striking", "Relentless Defender", "Defensive Retrieval",
        "Drop Shot Threat", "Slice Variety", "Transition Game", "Net Rushing",
        "Point Construction",
    ],
    "required_skills": ["serve", "return", "forehand", "backhand", "movement", "variety", "net_play"],
}


def _profile(**overrides) -> dict:
    base = {
        "playing_style": "Baseliner",
        "court_position": "Baseline",
        "preferred_rally": "Medium",
        "skills": {"serve": 3, "return": 3, "forehand": 3, "backhand": 3, "movement": 3, "variety": 3, "net_play": 3},
        # The database requires at least one signature trait per player; use
        # a neutral one that evaluate_* functions don't score on so it
        # doesn't quietly bias tests that expect a "no edge" result.
        "signature_traits": ["Point Construction"],
        "aliases": [],
        "confidence": "medium",
    }
    base.update(overrides)
    return base


def _database(tmp_path: Path, players: dict) -> PlayerTraitsDatabase:
    payload = {"_metadata": METADATA, "players": players}
    path = tmp_path / "traits.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return PlayerTraitsDatabase(path)


def _court(**overrides) -> dict:
    base = {"speed": 3, "bounce": "medium", "surface": "Hard", "confidence": "medium"}
    base.update(overrides)
    return base


# --- name tokenization / abbreviation matching ------------------------------

def test_name_tokens_strips_accents_and_punctuation():
    assert _name_tokens("Félix Auger-Aliassime") == ["felix", "auger", "aliassime"]


def test_name_tokens_is_case_insensitive():
    assert _name_tokens("DE MINAUR A.") == ["de", "minaur", "a"]


def test_abbreviated_candidates_unique_surname_and_initial_match():
    canonical = ["Jannik Sinner", "Carlos Alcaraz"]
    assert _abbreviated_name_candidates("Sinner J.", canonical) == ["Jannik Sinner"]
    assert _abbreviated_name_candidates("J. Sinner", canonical) == ["Jannik Sinner"]


def test_abbreviated_candidates_ambiguous_when_surname_and_initial_collide():
    # Two different first names sharing a surname AND first initial must not
    # silently resolve to either one.
    canonical = ["Alexander Zverev", "Anton Zverev"]
    candidates = _abbreviated_name_candidates("Zverev A.", canonical)
    assert set(candidates) == {"Alexander Zverev", "Anton Zverev"}


def test_abbreviated_candidates_no_match_returns_empty():
    canonical = ["Jannik Sinner", "Carlos Alcaraz"]
    assert _abbreviated_name_candidates("Nadal R.", canonical) == []


def test_abbreviated_candidates_ignores_single_token_input():
    # A bare surname with no initial isn't a safe abbreviation match.
    assert _abbreviated_name_candidates("Sinner", ["Jannik Sinner"]) == []


def test_resolve_player_profile_exact_match(tmp_path):
    db = _database(tmp_path, {"Jannik Sinner": _profile(aliases=[])})
    profile, resolution = resolve_player_profile(db, "Jannik Sinner")
    assert profile is not None
    assert resolution["method"] == "exact_or_alias"
    assert resolution["resolved"] == "Jannik Sinner"


def test_resolve_player_profile_unique_abbreviation(tmp_path):
    db = _database(tmp_path, {
        "Jannik Sinner": _profile(),
        "Carlos Alcaraz": _profile(),
    })
    profile, resolution = resolve_player_profile(db, "Sinner J.")
    assert profile is not None
    assert resolution["method"] == "surname_initial"
    assert resolution["resolved"] == "Jannik Sinner"


def test_resolve_player_profile_ambiguous_returns_no_profile(tmp_path):
    db = _database(tmp_path, {
        "Alexander Zverev": _profile(),
        "Anton Zverev": _profile(),
    })
    profile, resolution = resolve_player_profile(db, "Zverev A.")
    assert profile is None
    assert resolution["method"] == "ambiguous"
    assert set(resolution["candidates"]) == {"Alexander Zverev", "Anton Zverev"}


def test_resolve_player_profile_unresolved_does_not_invent_profile(tmp_path):
    db = _database(tmp_path, {"Jannik Sinner": _profile()})
    profile, resolution = resolve_player_profile(db, "Totally Unknown Player")
    assert profile is None
    assert resolution["method"] == "unresolved"


# --- scoring primitives -----------------------------------------------------

def test_severity_bands():
    assert _severity(0.1) == "Neutral"
    assert _severity(1.0) == "Minor"
    assert _severity(1.5) == "Moderate"
    assert _severity(2.2) == "Major"
    assert _severity(3.0) == "Match-Defining"


def test_winner_requires_minimum_gap():
    assert _winner("A", "B", 0.5) is None
    assert _winner("A", "B", 0.75) == "A"
    assert _winner("A", "B", -0.75) == "B"


def test_confidence_is_bounded_and_penalizes_fallback_profiles():
    high_diff = _confidence(10.0, "high")
    low_diff = _confidence(10.0, "fallback")
    assert 0.35 <= low_diff <= high_diff <= 0.98


# --- individual edge evaluators: symmetric inputs stay neutral -------------

def test_equal_profiles_produce_no_winner_on_every_edge():
    a = _profile()
    a["name"] = "Player A"
    b = _profile()
    b["name"] = "Player B"
    court = _court()

    for edge in (
        evaluate_serve(a, b, court),
        evaluate_return(a, b),
        evaluate_baseline(a, b, court),
        evaluate_movement(a, b, court),
        evaluate_variety(a, b),
        evaluate_surface(a, b, court),
    ):
        assert edge.winner is None
        assert edge.strength == "Neutral"


def test_big_server_trait_creates_serve_edge():
    a = _profile(signature_traits=["Big Server"], skills={
        "serve": 4, "return": 3, "forehand": 3, "backhand": 3, "movement": 3, "variety": 3, "net_play": 3,
    })
    a["name"] = "Server"
    b = _profile()
    b["name"] = "Returner"
    edge = evaluate_serve(a, b, _court())
    assert edge.winner == "Server"
    assert edge.score_a > edge.score_b


def test_fast_court_amplifies_serve_multiplier():
    a = _profile(skills={"serve": 5, "return": 3, "forehand": 3, "backhand": 3, "movement": 3, "variety": 3, "net_play": 3})
    a["name"] = "A"
    b = _profile(skills={"serve": 1, "return": 3, "forehand": 3, "backhand": 3, "movement": 3, "variety": 3, "net_play": 3})
    b["name"] = "B"
    fast_edge = evaluate_serve(a, b, _court(speed=5))
    slow_edge = evaluate_serve(a, b, _court(speed=1))
    # Same skill gap, but the fast court should widen the serve-score gap.
    assert (fast_edge.score_a - fast_edge.score_b) > (slow_edge.score_a - slow_edge.score_b)


# --- full analyze_matchup integration --------------------------------------

def test_analyze_matchup_reports_missing_players_without_inventing_data(tmp_path):
    db = _database(tmp_path, {"Jannik Sinner": _profile()})
    result = analyze_matchup("Jannik Sinner", "Nobody Real", "US Open", "Hard", database=db)
    assert result["status"] == "insufficient_profile_data"
    assert "Nobody Real" in result["missing_players"]
    assert result["explanation_only"] is True


def test_analyze_matchup_full_report_shape(tmp_path):
    db = _database(tmp_path, {
        "Player A": _profile(signature_traits=["Big Server"], skills={
            "serve": 5, "return": 3, "forehand": 3, "backhand": 3, "movement": 3, "variety": 3, "net_play": 3,
        }),
        "Player B": _profile(skills={
            "serve": 1, "return": 3, "forehand": 3, "backhand": 3, "movement": 3, "variety": 3, "net_play": 3,
        }),
    })
    result = analyze_matchup("Player A", "Player B", "US Open", "Hard", database=db)
    assert result["status"] == "ok"
    assert len(result["edges"]) == 6
    assert "Player A" in result["path_counts"]
    assert "Player B" in result["path_counts"]
    # A clear serve-skill gap should show up as at least one edge won by Player A.
    assert result["path_counts"]["Player A"] >= 1


def test_analyze_matchup_is_explanation_only_not_a_predictor(tmp_path):
    # This module documents itself as explanation-only ("Prediction happens
    # elsewhere"). Guard the contract: no predicted-winner-style field, and
    # the explicit flag stays present and true.
    db = _database(tmp_path, {
        "Player A": _profile(skills={
            "serve": 5, "return": 3, "forehand": 3, "backhand": 3, "movement": 3, "variety": 3, "net_play": 3,
        }),
        "Player B": _profile(skills={
            "serve": 1, "return": 3, "forehand": 3, "backhand": 3, "movement": 3, "variety": 3, "net_play": 3,
        }),
    })
    result = analyze_matchup("Player A", "Player B", "US Open", "Hard", database=db)
    assert result["explanation_only"] is True
    assert "predicted_winner" not in result
    assert "win_probability" not in result
