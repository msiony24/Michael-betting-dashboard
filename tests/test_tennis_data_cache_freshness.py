from pathlib import Path

from engine import data


def test_match_file_signature_changes_when_current_year_file_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    path = Path(tmp_path) / "atp_matches_2026.csv"
    path.write_text("winner_name,loser_name,tourney_date\nA,B,20260806\n", encoding="utf-8")

    first = data._match_file_signature()
    path.write_text(
        "winner_name,loser_name,tourney_date\nA,B,20260806\nC,D,20260807\n",
        encoding="utf-8",
    )
    second = data._match_file_signature()

    assert first != second
    assert first[0][0] == "atp_matches_2026.csv"
    assert second[0][0] == "atp_matches_2026.csv"
