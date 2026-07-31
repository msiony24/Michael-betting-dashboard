from __future__ import annotations

import json
from pathlib import Path

from engine.matchup_engine import analyze_matchup
from engine.player_traits import PlayerTraitsDatabase


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "player_traits.json"


def test_known_players_return_structured_report():
    db = PlayerTraitsDatabase(DB_PATH)
    result = analyze_matchup(
        "Jannik Sinner",
        "Carlos Alcaraz",
        "Indian Wells",
        "Hard",
        database=db,
    )

    assert result["status"] == "ok"
    assert result["explanation_only"] is True
    assert len(result["edges"]) == 6
    assert set(result["path_counts"]) == {"Jannik Sinner", "Carlos Alcaraz"}
    assert isinstance(result["tactical_read"], str)
    assert result["court_profile"]["matched_name"] == "indian wells"


def test_aliases_resolve_to_canonical_names():
    db = PlayerTraitsDatabase(DB_PATH)
    result = analyze_matchup(
        "Sinner",
        "C. Alcaraz",
        "US Open",
        "Hard",
        database=db,
    )

    assert result["player_a"] == "Jannik Sinner"
    assert result["player_b"] == "Carlos Alcaraz"


def test_unknown_player_does_not_receive_invented_profile():
    db = PlayerTraitsDatabase(DB_PATH)
    result = analyze_matchup(
        "Unknown Example",
        "Jannik Sinner",
        "Wimbledon",
        "Grass",
        database=db,
    )

    assert result["status"] == "insufficient_profile_data"
    assert result["missing_players"] == ["Unknown Example"]


def test_output_is_deterministic():
    db = PlayerTraitsDatabase(DB_PATH)
    first = analyze_matchup(
        "Taylor Fritz",
        "Alex de Minaur",
        "Cincinnati",
        "Hard",
        database=db,
    )
    second = analyze_matchup(
        "Taylor Fritz",
        "Alex de Minaur",
        "Cincinnati",
        "Hard",
        database=db,
    )

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_paths_do_not_double_count_subskills():
    db = PlayerTraitsDatabase(DB_PATH)
    result = analyze_matchup(
        "Reilly Opelka",
        "Jannik Sinner",
        "Wimbledon",
        "Grass",
        database=db,
    )

    for paths in result["paths_to_victory"].values():
        assert len(paths) == len(set(paths))
        assert set(paths).issubset(
            {"Serve", "Return", "Baseline", "Movement", "Variety", "Surface"}
        )
