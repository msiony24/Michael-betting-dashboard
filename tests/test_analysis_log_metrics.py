from engine.analysis_log_metrics import (
    group_rows_by_day,
    normalize_prediction_result,
    normalize_verdict,
    summarize_rows,
)


def test_prediction_result_normalization():
    assert normalize_prediction_result({"status": "Won"}) == "Correct"
    assert normalize_prediction_result({"status": "Lost"}) == "Incorrect"
    assert normalize_prediction_result({"status": "Pending"}) == "Pending"


def test_verdict_normalization():
    assert normalize_verdict("Strong Bet") == "BET"
    assert normalize_verdict("Worth Betting") == "BET"
    assert normalize_verdict("Pass") == "PASS"
    assert normalize_verdict("Complete Pass") == "PASS"


def test_summary_counts_bet_and_pass_predictions():
    rows = [
        {"status": "Won", "recommendation": "Strong Bet"},
        {"status": "Lost", "recommendation": "Worth Betting"},
        {"status": "Won", "recommendation": "Pass"},
        {"status": "Pending", "recommendation": "Complete Pass"},
    ]
    summary = summarize_rows(rows)
    assert summary["correct"] == 2
    assert summary["incorrect"] == 1
    assert summary["pending"] == 1
    assert summary["bet"]["correct"] == 1
    assert summary["bet"]["incorrect"] == 1
    assert summary["pass"]["correct"] == 1
    assert summary["pass"]["pending"] == 1


def test_group_rows_by_day_uses_event_date_then_created_at():
    rows = [
        {"event_date": "2026-08-04", "id": "a"},
        {"created_at": "2026-08-03T12:00:00Z", "id": "b"},
        {"event_date": "2026-08-04", "id": "c"},
    ]
    grouped = group_rows_by_day(rows)
    assert list(grouped) == ["2026-08-04", "2026-08-03"]
    assert [row["id"] for row in grouped["2026-08-04"]] == ["a", "c"]
