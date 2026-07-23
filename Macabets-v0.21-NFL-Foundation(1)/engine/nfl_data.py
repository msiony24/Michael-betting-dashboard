"""NFL constants and starter team-quality priors for Macabets.

The ratings are neutral placeholders for the foundation release. They are designed
for manual review and will be replaced by the independent Team Quality Engine.
"""

from __future__ import annotations

NFL_TEAMS = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
    "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
    "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
    "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders",
]

# Foundation priors intentionally begin at league average. Users can supply current
# matchup grades in the UI until the automated Team Quality Engine is implemented.
TEAM_PRIORS = {team: 0.0 for team in NFL_TEAMS}

HOME_FIELD_POINTS = {
    "Standard home field": 1.7,
    "Strong home field": 2.3,
    "Weak home field": 1.0,
    "Neutral site": 0.0,
}
