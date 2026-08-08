import json
from pathlib import Path

import pandas as pd

from engine.madden_roster_validator import build_validation_report


def test_az_alias_counts_as_arizona(tmp_path: Path):
    teams = [
        "AZ", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
        "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LA", "MIA", "MIN", "NE", "NO",
        "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
    ]
    rows = []
    rated = []
    for t in teams:
        canon = "ARI" if t == "AZ" else t
        for pos, n in [("QB", 2), ("RB", 3), ("WR", 6), ("OL", 7), ("DL", 6), ("LB", 4), ("DB", 7), ("K", 2)]:
            for i in range(n):
                name = f"{canon}-{pos}-{i}"
                rows.append({"player_name": name, "team": t, "position": pos, "overall": 70+i})
                rated.append({"player_name": name, "team_abbr": canon, "position": pos, "overall": 70+i, "macabets_rating": 70+i, "rating_source": "Madden 27 baseline"})
    m = tmp_path / "m.csv"; p = tmp_path / "p.csv"; tr = tmp_path / "teams.json"; st = tmp_path / "status.json"
    pd.DataFrame(rows).to_csv(m, index=False)
    pd.DataFrame(rated).to_csv(p, index=False)
    tr.write_text(json.dumps({("ARI" if t == "AZ" else t): {} for t in teams}))
    st.write_text(json.dumps({"players_with_performance_data": 0}))
    report = build_validation_report(m, p, tr, st)
    assert report["summary"]["madden_teams_canonical"] == 32
    assert report["summary"]["rated_teams"] == 32
    assert report["summary"]["validation_passed"] is True
