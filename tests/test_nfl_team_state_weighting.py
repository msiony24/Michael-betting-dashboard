from engine.nfl_team_state import TEAM_STATE_WEIGHTS, performance_evidence_weight


def test_team_state_weights_sum_to_one():
    assert abs(sum(TEAM_STATE_WEIGHTS.values()) - 1.0) < 1e-9


def test_prior_season_snapshot_is_not_double_counted():
    assert performance_evidence_weight(2025, 18, target_season=2026) == 0.0


def test_current_season_evidence_ramps_with_games():
    week1 = performance_evidence_weight(2026, 1, target_season=2026)
    week5 = performance_evidence_weight(2026, 5, target_season=2026)
    week18 = performance_evidence_weight(2026, 18, target_season=2026)
    assert 0.0 < week1 < week5 < week18 <= 0.85


def test_future_snapshot_is_ignored():
    assert performance_evidence_weight(2027, 1, target_season=2026) == 0.0
