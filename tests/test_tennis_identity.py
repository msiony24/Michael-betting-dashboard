from __future__ import annotations

from datetime import date

import pandas as pd

from engine.tennis import perspective
from engine.tennis_h2h import build_head_to_head_summary
from engine.tennis_identity import canonical_player_key, player_name_signature, resolve_player_name
from engine.tennis_serve_return import serve_return_profile
from update_tennis_data import player_signature


def test_compound_and_multi_given_names_share_identity_without_registry():
    pairs = [
        ("Felix Auger-Aliassime", "Auger-Aliassime F.", ("aliassime", "f")),
        ("Juan Manuel Cerundolo", "Cerundolo J.M.", ("cerundolo", "j")),
        ("Alex de Minaur", "De Minaur A.", ("minaur", "a")),
        ("Taylor Fritz", "Fritz T.", ("fritz", "t")),
    ]
    empty_registry = pd.DataFrame(columns=["player_key", "alias", "canonical_name"])
    for full_name, provider_name, expected in pairs:
        assert player_name_signature(full_name) == expected
        assert player_name_signature(provider_name) == expected
        assert player_signature(full_name) == expected
        assert player_signature(provider_name) == expected
        assert canonical_player_key(full_name, registry=empty_registry) == canonical_player_key(
            provider_name, registry=empty_registry
        )


def test_provider_player_id_resolves_reversed_and_historical_aliases():
    registry = pd.DataFrame([
        {"player_key": "388", "alias": "Juan Manuel Cerundolo", "canonical_name": "Cerundolo J.M."},
        {"player_key": "388", "alias": "Manuel Cerundolo Juan", "canonical_name": "Cerundolo J.M."},
        {"player_key": "388", "alias": "Cerundolo J.M.", "canonical_name": "Cerundolo J.M."},
    ])
    assert canonical_player_key("Juan Manuel Cerundolo", registry=registry) == "api:388"
    assert canonical_player_key("Manuel Cerundolo Juan", registry=registry) == "api:388"
    assert canonical_player_key("Cerundolo J.M.", registry=registry) == "api:388"

    matches = pd.DataFrame([
        {"winner_name": "Cerundolo J.M.", "loser_name": "Auger-Aliassime F."},
    ])
    resolved, resolution = resolve_player_name(
        matches,
        "Manuel Cerundolo Juan",
        registry=registry,
    )
    assert resolved == "Cerundolo J.M."
    assert resolution["player_key"] == "388"
    assert resolution["method"] == "provider_player_id"



def test_default_registry_resolves_juan_manuel_cerundolo():
    assert canonical_player_key("Juan Manuel Cerundolo") == "api:388"
    matches = pd.DataFrame([
        {"winner_name": "Cerundolo J.M.", "loser_name": "Opponent X."},
    ])
    resolved, resolution = resolve_player_name(matches, "Juan Manuel Cerundolo")
    assert resolved == "Cerundolo J.M."
    assert resolution["player_key"] == "388"


def test_resolver_and_perspective_merge_felix_aliases():
    matches = pd.DataFrame([
        {
            "tourney_date": pd.Timestamp("2025-04-25"),
            "tourney_name": "Mutua Madrid Open",
            "surface": "Clay",
            "tourney_level": "M",
            "round": "R64",
            "winner_name": "Cerundolo J.M.",
            "loser_name": "Auger-Aliassime F.",
            "winner_rank": 126,
            "loser_rank": 19,
            "score": "7-6 6-4",
        },
        {
            "tourney_date": pd.Timestamp("2026-02-05"),
            "tourney_name": "Open Sud de France",
            "surface": "Hard",
            "tourney_level": "A",
            "round": "R64",
            "winner_name": "Felix Auger-Aliassime",
            "loser_name": "Wawrinka S.",
            "winner_rank": 8,
            "loser_rank": 113,
            "score": "6-4 7-6",
        },
    ])
    resolved, resolution = resolve_player_name(matches, "Felix Auger-Aliassime")
    assert resolved in {"Felix Auger-Aliassime", "Auger-Aliassime F."}
    assert resolution["resolved"] == resolved
    history = perspective(matches, resolved, date(2026, 8, 18))
    assert len(history) == 2


def test_h2h_finds_madrid_2025_across_name_formats():
    matches = pd.DataFrame([{
        "tourney_date": 20250425,
        "tourney_name": "Mutua Madrid Open",
        "surface": "Clay",
        "tourney_level": "M",
        "round": "R64",
        "winner_name": "Cerundolo J.M.",
        "loser_name": "Auger-Aliassime F.",
        "score": "7-6 6-4",
    }])
    h2h = build_head_to_head_summary(
        matches,
        "Felix Auger-Aliassime",
        "Juan Manuel Cerundolo",
        "Hard",
    )
    assert h2h["meetings"] == 1
    assert h2h["wins_a"] == 0
    assert h2h["wins_b"] == 1
    assert h2h["surface_meetings"] == 0
    assert h2h["last_meeting"]["date"] == "2025-04-25"
    assert h2h["last_meeting"]["winner"] == "Juan Manuel Cerundolo"
    assert h2h["last_meeting"]["score"] == "7-6 6-4"


def test_serve_return_counts_abbreviated_and_full_felix_rows_together():
    rows = []
    for idx, name in enumerate(["Auger-Aliassime F.", "Felix Auger-Aliassime", "Auger-Aliassime F."]):
        rows.append({
            "tourney_date": pd.Timestamp("2026-01-10") + pd.Timedelta(days=idx),
            "surface": "Hard",
            "winner_name": name,
            "loser_name": f"Opponent {idx}",
            "w_svpt": 60,
            "w_1stIn": 36,
            "w_1stWon": 28,
            "w_2ndWon": 14,
            "w_ace": 8,
            "w_df": 2,
            "w_SvGms": 10,
            "w_bpSaved": 3,
            "w_bpFaced": 4,
            "l_svpt": 58,
            "l_1stIn": 34,
            "l_1stWon": 22,
            "l_2ndWon": 11,
            "l_ace": 3,
            "l_df": 3,
            "l_SvGms": 10,
            "l_bpSaved": 2,
            "l_bpFaced": 5,
        })
    matches = pd.DataFrame(rows)
    profile = serve_return_profile(matches, "Felix Auger-Aliassime", date(2026, 8, 18), "Hard")
    assert profile["matches_with_stats"] == 3
    assert profile["available"] is True
