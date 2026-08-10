"""Regression test for Madden 27 + nflverse QB performance blending.

Replace the existing tests/test_nfl_rating_engine.py only if this matches
your current test structure. This version updates the stale Madden 26
rating_source assertion to Madden 27.
"""

def test_builds_players_and_blends_qb_performance(tmp_path):
    madden = tmp_path / "madden.csv"
    nfl = tmp_path / "nfl"
    nfl.mkdir()

    _madden_fixture(madden)
    pd.DataFrame([{
        "player_display_name": "Test Quarterback",
        "recent_team": "BUF",
        "position": "QB",
        "attempts": 600,
        "passing_yards": 5000,
        "passing_tds": 45,
        "interceptions": 5,
        "rushing_yards": 500,
        "rushing_tds": 5,
        "sacks": 20,
    }]).to_csv(nfl / "player_weekly_stats.csv", index=False)

    players = build_player_ratings(madden, nfl)
    qb = players.loc[players.player_name.eq("Test Quarterback")].iloc[0]

    assert qb.performance_weight > 0
    assert qb.rating_source == "Madden 27 + nflverse performance"
