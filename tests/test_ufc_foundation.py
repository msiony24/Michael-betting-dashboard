from datetime import date

import pandas as pd

from engine.ufc_data import normalize_division, parse_completed_events, parse_event_fights
from engine.ufc_ratings import build_fighter_ratings


EVENTS_HTML = """
<table>
<tr class="b-statistics__table-row">
<td class="b-statistics__table-col"><i><a class="b-link" href="http://ufcstats.com/event-details/abc">UFC Test</a><span class="b-statistics__date">August 15, 2026</span></i></td>
<td class="b-statistics__table-col">Las Vegas, Nevada, USA</td>
</tr>
</table>
"""

EVENT_HTML = """
<table>
<tr class="b-fight-details__table-row b-fight-details__table-row__hover js-fight-table-row" data-link="http://ufcstats.com/fight-details/f1">
<td class="b-fight-details__table-col"><p>W</p><p>L</p></td>
<td class="b-fight-details__table-col"><p><a href="http://ufcstats.com/fighter-details/a">Alpha Fighter</a></p><p><a href="http://ufcstats.com/fighter-details/b">Beta Fighter</a></p></td>
<td class="b-fight-details__table-col"><p>2</p><p>0</p></td>
<td class="b-fight-details__table-col"><p>80</p><p>42</p></td>
<td class="b-fight-details__table-col"><p>1</p><p>0</p></td>
<td class="b-fight-details__table-col"><p>0</p><p>0</p></td>
<td class="b-fight-details__table-col">Lightweight Bout</td>
<td class="b-fight-details__table-col">KO/TKO</td>
<td class="b-fight-details__table-col">2</td>
<td class="b-fight-details__table-col">3:10</td>
</tr>
</table>
"""


def test_event_parser_and_division_normalization():
    events = parse_completed_events(EVENTS_HTML)
    assert len(events) == 1
    assert events.iloc[0]["event_date"] == "2026-08-15"
    assert normalize_division("Women's Flyweight Bout") == "Women’s Flyweight"
    assert normalize_division("Lightweight Bout") == "Men’s Lightweight"


def test_event_fight_parser_emits_one_row_per_fighter():
    event = {
        "event_name": "UFC Test",
        "event_date": "2026-08-15",
        "location": "Las Vegas, Nevada, USA",
    }
    fights = parse_event_fights(EVENT_HTML, event)
    assert len(fights) == 2
    assert set(fights["fighter"]) == {"Alpha Fighter", "Beta Fighter"}
    assert fights.loc[fights["fighter"] == "Alpha Fighter", "result"].iloc[0] == "W"
    assert fights.loc[fights["fighter"] == "Alpha Fighter", "sig_str"].iloc[0] == 80


def _fight(url, day, winner, loser, winner_sig=60, loser_sig=40, method="Decision - Unanimous"):
    return [
        {"event_date": day, "fight_url": url, "fighter": winner, "opponent": loser, "result": "W", "division": "Men’s Lightweight", "method": method, "sig_str": winner_sig, "kd": 1, "td": 1, "sub_att": 0},
        {"event_date": day, "fight_url": url, "fighter": loser, "opponent": winner, "result": "L", "division": "Men’s Lightweight", "method": method, "sig_str": loser_sig, "kd": 0, "td": 0, "sub_att": 0},
    ]


def test_opponent_quality_drives_rating_order():
    rows = []
    rows += _fight("f1", "2024-01-01", "Strong", "Mid")
    rows += _fight("f2", "2024-02-01", "Mid", "Weak")
    rows += _fight("f3", "2024-03-01", "Strong", "Weak", method="KO/TKO")
    rows += _fight("f4", "2024-04-01", "Mid", "Weak")
    rows += _fight("f5", "2024-05-01", "Strong", "Mid")
    ratings = build_fighter_ratings(pd.DataFrame(rows), as_of=date(2024, 6, 1))
    lookup = ratings.set_index("fighter")
    assert lookup.loc["Strong", "macabets_rating"] > lookup.loc["Mid", "macabets_rating"] > lookup.loc["Weak", "macabets_rating"]
    assert lookup.loc["Strong", "division_rank"] == 1
    assert lookup.loc["Strong", "strength_score"] > 50

CURRENT_EVENTS_HTML = """
<table>
<tr class="b-statistics__table-row_type_first">
<td class="b-statistics__table-col">
  <i class="b-statistics__table-content">
    <a href="http://ufcstats.com/event-details/current123">UFC Current Layout</a>
    <span class="b-statistics__date">August 16, 2026</span>
  </i>
</td>
<td class="b-statistics__table-col">Chicago, Illinois, USA</td>
</tr>
</table>
"""


def test_completed_events_parser_handles_current_ufcstats_cell_layout():
    events = parse_completed_events(CURRENT_EVENTS_HTML)
    assert len(events) == 1
    assert events.iloc[0]["event_name"] == "UFC Current Layout"
    assert events.iloc[0]["event_date"] == "2026-08-16"
    assert events.iloc[0]["location"] == "Chicago, Illinois, USA"
    assert events.iloc[0]["event_url"].startswith("https://ufcstats.com/event-details/")


def test_catchweight_does_not_become_fighters_ranking_division():
    rows = []
    rows += _fight("f1", "2025-01-01", "Alpha", "Beta")
    catch = _fight("f2", "2025-06-01", "Alpha", "Gamma")
    for row in catch:
        row["division"] = "Catch Weight"
    rows += catch
    ratings = build_fighter_ratings(pd.DataFrame(rows), as_of=date(2025, 7, 1)).set_index("fighter")
    assert ratings.loc["Alpha", "division"] == "Men’s Lightweight"
    assert "Catch Weight" not in set(ratings.loc[ratings["active_pool"], "division"])


def test_new_weight_class_is_not_treated_as_fully_proven_immediately():
    rows = []
    # Veteran builds most of his record at lightweight.
    for i in range(6):
        rows += _fight(f"lw{i}", f"2024-0{min(i + 1, 9)}-01", "Veteran", f"LW Opp {i}")
    # Only one win at welterweight.
    move = _fight("ww1", "2025-01-01", "Veteran", "WW Opp")
    for row in move:
        row["division"] = "Men’s Welterweight"
    rows += move

    ratings = build_fighter_ratings(pd.DataFrame(rows), as_of=date(2025, 2, 1)).set_index("fighter")
    assert ratings.loc["Veteran", "division"] == "Men’s Welterweight"
    assert ratings.loc["Veteran", "division_fights"] == 1
    assert ratings.loc["Veteran", "division_elo"] < ratings.loc["Veteran", "global_elo"]


def test_one_fight_newcomer_is_shrunk_toward_neutral():
    rows = []
    rows += _fight("f1", "2025-01-01", "Established", "Mid")
    rows += _fight("f2", "2025-02-01", "Established", "Mid")
    rows += _fight("f3", "2025-03-01", "Established", "Mid")
    rows += _fight("f4", "2025-04-01", "Prospect", "Mid", method="KO/TKO")
    ratings = build_fighter_ratings(pd.DataFrame(rows), as_of=date(2025, 5, 1)).set_index("fighter")
    assert ratings.loc["Prospect", "ranking_confidence"] < ratings.loc["Established", "ranking_confidence"]
    assert abs(ratings.loc["Prospect", "macabets_rating"] - 1500) < abs(ratings.loc["Prospect", "raw_elo"] - 1500)
