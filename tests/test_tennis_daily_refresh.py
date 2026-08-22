from __future__ import annotations

import pandas as pd

from update_tennis_data import (
    MATCH_COLUMNS,
    build_player_identity_registry,
    convert_api_fixtures,
    merge_live_matches,
    normalize_api_draw_context,
    normalize_level,
    normalize_round,
    player_signature,
    preserve_existing_statistics,
)


def _blank_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in MATCH_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[MATCH_COLUMNS]


def test_player_signature_handles_provider_name_formats():
    assert player_signature("Tirante T.A.") == ("tirante", "t")
    assert player_signature("T. A. Tirante") == ("tirante", "t")
    assert player_signature("Thiago Agustin Tirante") == ("tirante", "t")
    assert player_signature("Fritz T.") == ("fritz", "t")
    assert player_signature("Taylor Fritz") == ("fritz", "t")
    assert player_signature("Felix Auger-Aliassime") == ("aliassime", "f")
    assert player_signature("Auger-Aliassime F.") == ("aliassime", "f")
    assert player_signature("Juan Manuel Cerundolo") == ("cerundolo", "j")
    assert player_signature("Cerundolo J.M.") == ("cerundolo", "j")


def test_convert_api_fixture_maps_winner_score_names_and_player_ids():
    fixtures = [{
        "event_date": "2026-08-05",
        "event_first_player": "T. A. Tirante",
        "first_player_key": "101",
        "event_second_player": "T. Fritz",
        "second_player_key": "202",
        "event_winner": "First Player",
        "event_status": "Finished",
        "event_type_type": "Atp Singles",
        "tournament_name": "ATP Montreal, Canada Men Singles",
        "tournament_round": "ATP Montreal, Canada Men Singles - Round of 64",
        "scores": [
            {"score_first": "7", "score_second": "5", "score_set": "1"},
            {"score_first": "6", "score_second": "3", "score_set": "2"},
        ],
        "statistics": [],
    }]
    existing_names = {
        ("tirante", "t"): "Tirante T.A.",
        ("fritz", "t"): "Fritz T.",
    }
    frame = convert_api_fixtures(
        fixtures,
        existing_names=existing_names,
        historical_surfaces={"canadian open": "Hard"},
        ranks_by_key={"101": 50.0, "202": 4.0},
    )
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["winner_name"] == "Tirante T.A."
    assert row["loser_name"] == "Fritz T."
    assert row["winner_player_key"] == "101"
    assert row["loser_player_key"] == "202"
    assert row["score"] == "7-5 6-3"
    assert row["round"] == "R64"
    assert row["surface"] == "Hard"
    assert row["winner_rank"] == 50.0
    assert row["loser_rank"] == 4.0


def test_identity_registry_uses_api_key_to_join_provider_aliases():
    fixtures = [{
        "event_first_player": "Juan Manuel Cerundolo",
        "first_player_key": "388",
        "event_second_player": "Felix Auger-Aliassime",
        "second_player_key": "2073",
    }]
    standings = [
        {"player": "Manuel Cerundolo Juan", "player_key": 388},
        {"player": "Felix Auger-Aliassime", "player_key": 2073},
    ]
    existing_names = {
        ("cerundolo", "j"): "Cerundolo J.M.",
        ("aliassime", "f"): "Auger-Aliassime F.",
    }
    registry = build_player_identity_registry(
        fixtures,
        standings,
        existing_names=existing_names,
        existing_registry=pd.DataFrame(),
    )
    cerundolo = registry[registry["player_key"].astype(str).eq("388")]
    assert set(cerundolo["alias"]) >= {
        "Juan Manuel Cerundolo",
        "Manuel Cerundolo Juan",
        "Cerundolo J.M.",
    }
    assert set(cerundolo["canonical_name"]) == {"Cerundolo J.M."}


def test_merge_live_matches_replaces_stale_duplicate():
    baseline = _blank_frame([{
        "tourney_date": "20260805",
        "tourney_name": "Montreal",
        "surface": "Hard",
        "tourney_level": "M",
        "round": "R64",
        "winner_name": "Fritz T.",
        "loser_name": "Tirante T.A.",
        "score": "6-4 6-4",
    }])
    live = _blank_frame([{
        "tourney_date": "20260805",
        "tourney_name": "ATP Montreal, Canada Men Singles",
        "surface": "Hard",
        "tourney_level": "A",
        "round": "R64",
        "winner_name": "Tirante T.A.",
        "loser_name": "Fritz T.",
        "winner_player_key": "101",
        "loser_player_key": "202",
        "score": "7-5 6-3",
    }])
    merged = merge_live_matches(baseline, live)
    assert len(merged) == 1
    assert merged.iloc[0]["winner_name"] == "Tirante T.A."
    assert merged.iloc[0]["winner_player_key"] == "101"
    assert merged.iloc[0]["score"] == "7-5 6-3"


def test_refresh_preserves_api_player_ids_when_yearly_baseline_replaces_row():
    baseline = _blank_frame([{
        "tourney_date": "20260805",
        "tourney_name": "Montreal",
        "surface": "Hard",
        "tourney_level": "M",
        "round": "R64",
        "winner_name": "Fritz T.",
        "loser_name": "Tirante T.A.",
        "score": "6-4 6-4",
    }])
    existing = baseline.copy()
    existing.loc[0, "winner_player_key"] = "202"
    existing.loc[0, "loser_player_key"] = "101"
    preserved = preserve_existing_statistics(baseline, existing)
    assert str(preserved.iloc[0]["winner_player_key"]) == "202"
    assert str(preserved.iloc[0]["loser_player_key"]) == "101"


def test_api_round_and_tournament_level_normalization_for_major_events():
    assert normalize_round("Wimbledon - 1/64-finals") == "R128"
    assert normalize_round("Wimbledon - 1/32-finals") == "R64"
    assert normalize_round("Wimbledon - 1/16-finals") == "R32"
    assert normalize_level("Wimbledon") == "G"
    assert normalize_level("US Open") == "G"
    assert normalize_level("Montreal") == "M"
    assert normalize_level("Cincinnati") == "M"


def test_draw_context_separates_slam_qualifying_from_main_draw():
    frame = _blank_frame([
        {
            "tourney_date": "20260827",
            "tourney_name": "US Open",
            "tourney_level": "G",
            "round": "Semi-finals",
            "winner_name": "Qualifier A",
            "loser_name": "Qualifier B",
            "score": "6-4 6-4",
        },
        {
            "tourney_date": "20260830",
            "tourney_name": "US Open",
            "tourney_level": "G",
            "round": "1/64-finals",
            "winner_name": "Main A",
            "loser_name": "Main B",
            "score": "6-4 6-4 6-4",
        },
        {
            "tourney_date": "20260901",
            "tourney_name": "US Open",
            "tourney_level": "G",
            "round": "1/32-finals",
            "winner_name": "Main A",
            "loser_name": "Main C",
            "score": "6-4 6-4 6-4",
        },
    ])
    fixed = normalize_api_draw_context(frame)
    qualifier = fixed.iloc[0]
    assert qualifier["round"] == "Q"
    assert qualifier["tourney_level"] == "A"
    assert fixed.iloc[1]["round"] == "R128"
    assert fixed.iloc[1]["tourney_level"] == "G"
    assert fixed.iloc[2]["round"] == "R64"
    assert fixed.iloc[2]["tourney_level"] == "G"

    fixed_again = normalize_api_draw_context(fixed)
    assert fixed_again.iloc[0]["round"] == "Q"
    assert fixed_again.iloc[0]["tourney_level"] == "A"


def test_merge_live_matches_removes_one_day_provider_duplicate_and_keeps_richer_row():
    baseline = _blank_frame([{
        "tourney_date": "20260804",
        "tourney_name": "Canadian Open",
        "surface": "Hard",
        "tourney_level": "M",
        "round": "R128",
        "winner_name": "Altmaier D.",
        "loser_name": "Vukic A.",
        "score": "6-7 7-6 6-4",
    }])
    live = _blank_frame([{
        "tourney_date": "20260803",
        "tourney_name": "Montreal",
        "surface": "Hard",
        "tourney_level": "M",
        "round": "1/64-finals",
        "winner_name": "Altmaier D.",
        "loser_name": "Vukic A.",
        "winner_player_key": "100",
        "loser_player_key": "200",
        "score": "6.2-7.7 7.7-6.5 6-4",
        "w_svpt": 90,
        "l_svpt": 88,
        "w_1stWon": 40,
        "l_1stWon": 38,
        "w_2ndWon": 20,
        "l_2ndWon": 19,
    }])
    merged = merge_live_matches(baseline, live)
    assert len(merged) == 1
    assert merged.iloc[0]["tourney_name"] == "Montreal"
    assert merged.iloc[0]["tourney_level"] == "M"
    assert merged.iloc[0]["round"] == "R128"
    assert str(merged.iloc[0]["winner_player_key"]) == "100"


def test_merge_live_matches_does_not_collapse_distinct_matches_two_days_apart():
    baseline = _blank_frame([{
        "tourney_date": "20260628",
        "tourney_name": "Eastbourne",
        "surface": "Grass",
        "tourney_level": "A",
        "round": "F",
        "winner_name": "Bergs Z.",
        "loser_name": "Humbert U.",
        "score": "3-6 6-1 6-4",
    }])
    live = _blank_frame([{
        "tourney_date": "20260630",
        "tourney_name": "Wimbledon",
        "surface": "Grass",
        "tourney_level": "G",
        "round": "R128",
        "winner_name": "Bergs Z.",
        "loser_name": "Humbert U.",
        "score": "6-2 7-5 4-6 3-6 6-3",
    }])
    merged = merge_live_matches(baseline, live)
    assert len(merged) == 2


def test_merge_live_matches_dedupes_compound_surname_alias():
    baseline = _blank_frame([{
        "tourney_date": "20260804",
        "tourney_name": "Canadian Open",
        "surface": "Hard",
        "tourney_level": "M",
        "round": "R128",
        "winner_name": "Van De Zandschulp B.",
        "loser_name": "Mpetshi G.",
        "score": "6-2 3-6 6-3",
    }])
    live = _blank_frame([{
        "tourney_date": "20260804",
        "tourney_name": "Montreal",
        "surface": "Hard",
        "tourney_level": "M",
        "round": "R128",
        "winner_name": "Botic van de Zandschulp",
        "loser_name": "G. Mpetshi Perricard",
        "winner_player_key": "111",
        "loser_player_key": "9222",
        "score": "6-2 3-6 6-3",
    }])
    merged = merge_live_matches(baseline, live)
    assert len(merged) == 1
    assert str(merged.iloc[0]["loser_player_key"]) == "9222"
