from datetime import date

import pandas as pd

from engine.tennis_serve_return import serve_return_profile, serve_return_matchup_adjustment
from update_tennis_data import convert_api_fixtures


def _stat(player, name, value=None, won=None, total=None):
    return {
        "player_key": str(player),
        "stat_period": "match",
        "stat_type": "Service",
        "stat_name": name,
        "stat_value": value,
        "stat_won": won,
        "stat_total": total,
    }


def test_api_fixture_imports_point_level_serve_stats():
    event = {
        "event_type_type": "ATP Singles",
        "event_status": "Finished",
        "event_winner": "First Player",
        "event_first_player": "Alpha A.",
        "event_second_player": "Beta B.",
        "first_player_key": "1",
        "second_player_key": "2",
        "tournament_name": "Test Open",
        "tournament_round": "Quarterfinals",
        "event_date": "2026-08-01",
        "scores": [{"score_first": "6", "score_second": "4"}, {"score_first": "6", "score_second": "3"}],
        "statistics": [
            _stat(1, "Aces", "8"), _stat(2, "Aces", "3"),
            _stat(1, "Double Faults", "2"), _stat(2, "Double Faults", "4"),
            _stat(1, "1st serve points won", "72%", 36, 50),
            _stat(1, "2nd serve points won", "55%", 11, 20),
            _stat(2, "1st serve points won", "65%", 26, 40),
            _stat(2, "2nd serve points won", "45%", 9, 20),
            _stat(1, "Break Points Saved", "4/5", 4, 5),
            _stat(2, "Break Points Saved", "3/6", 3, 6),
            _stat(1, "Service Games Played", "10"),
            _stat(2, "Service Games Played", "10"),
        ],
    }
    frame = convert_api_fixtures(
        [event], existing_names={}, historical_surfaces={}, ranks_by_key={"1": 10, "2": 20}
    )
    row = frame.iloc[0]
    assert row["w_svpt"] == 70
    assert row["l_svpt"] == 60
    assert row["w_1stIn"] == 50
    assert row["w_1stWon"] == 36
    assert row["w_2ndWon"] == 11
    assert row["w_bpSaved"] == 4
    assert row["w_bpFaced"] == 5
    assert row["w_SvGms"] == 10


def _match(day, winner, loser, w_svpt, w1, w2, l_svpt, l1, l2):
    return {
        "tourney_date": pd.Timestamp(day), "surface": "Hard", "winner_name": winner, "loser_name": loser,
        "w_svpt": w_svpt, "w_1stIn": int(w_svpt * .62), "w_1stWon": w1, "w_2ndWon": w2,
        "l_svpt": l_svpt, "l_1stIn": int(l_svpt * .62), "l_1stWon": l1, "l_2ndWon": l2,
        "w_ace": 8, "l_ace": 3, "w_df": 2, "l_df": 4,
        "w_SvGms": 10, "l_SvGms": 10, "w_bpSaved": 4, "w_bpFaced": 5, "l_bpSaved": 2, "l_bpFaced": 5,
    }


def test_serve_return_profile_and_matchup_are_sample_shrunk_and_capped():
    rows = []
    for i in range(8):
        rows.append(_match(f"2026-07-{10+i:02d}", "Alpha", "OppA", 70, 36, 12, 68, 28, 9))
        rows.append(_match(f"2026-07-{10+i:02d}", "OppB", "Beta", 68, 34, 11, 70, 29, 8))
    matches = pd.DataFrame(rows)
    a = serve_return_profile(matches, "Alpha", date(2026, 8, 1), "Hard")
    b = serve_return_profile(matches, "Beta", date(2026, 8, 1), "Hard")
    assert a["available"] is True
    assert b["available"] is True
    assert a["serve_points_won"] > b["serve_points_won"]
    out = serve_return_matchup_adjustment(a, b)
    assert out["available"] is True
    assert out["probability_adjustment_a"] > 0
    assert abs(out["probability_adjustment_a"]) <= .032


def test_missing_stats_fail_closed():
    matches = pd.DataFrame([{
        "tourney_date": pd.Timestamp("2026-07-20"), "surface": "Hard", "winner_name": "Alpha", "loser_name": "Beta"
    }])
    a = serve_return_profile(matches, "Alpha", date(2026, 8, 1), "Hard")
    b = serve_return_profile(matches, "Beta", date(2026, 8, 1), "Hard")
    out = serve_return_matchup_adjustment(a, b)
    assert a["available"] is False
    assert out["probability_adjustment_a"] == 0.0
