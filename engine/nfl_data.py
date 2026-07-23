"""Static NFL reference data and editable Macabets v0.22 starter priors.

The ratings below are model priors, not live statistics. They exist so the Team
Power Rating Engine can create an independent line before automated weekly data
feeds are added. Every component is exposed in the UI and can be overridden.
"""

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

VENUE_TYPES = ["Outdoor", "Dome", "Retractable roof"]
WEATHER_OPTIONS = ["Normal", "Rain", "Snow", "High wind", "Extreme heat", "Extreme cold"]

# Component weights sum to 1.00. Ratings use a 0-100 scale.
TEAM_RATING_WEIGHTS = {
    "offense": 0.25,
    "defense": 0.25,
    "quarterback": 0.20,
    "coaching": 0.12,
    "strength_of_schedule": 0.10,
    "special_teams": 0.08,
}

# Conservative starter priors. These are deliberately compressed and editable.
# They are not presented as current factual rankings.
_TEAM_TIERS = {
    "Kansas City Chiefs": (90, 85, 96, 94, 84, 82),
    "Buffalo Bills": (89, 84, 94, 88, 83, 80),
    "Baltimore Ravens": (88, 89, 92, 90, 84, 82),
    "Philadelphia Eagles": (90, 88, 89, 88, 84, 81),
    "Detroit Lions": (91, 82, 88, 87, 82, 79),
    "Green Bay Packers": (87, 84, 89, 85, 81, 80),
    "Houston Texans": (86, 84, 90, 84, 82, 79),
    "San Francisco 49ers": (87, 86, 86, 91, 83, 80),
    "Cincinnati Bengals": (88, 79, 93, 82, 82, 78),
    "Los Angeles Rams": (86, 81, 88, 90, 81, 79),
    "Washington Commanders": (87, 80, 89, 85, 80, 78),
    "Tampa Bay Buccaneers": (84, 82, 86, 84, 79, 80),
    "Minnesota Vikings": (84, 85, 84, 88, 81, 79),
    "Los Angeles Chargers": (84, 84, 87, 91, 80, 78),
    "Denver Broncos": (81, 86, 82, 89, 80, 82),
    "Pittsburgh Steelers": (78, 88, 80, 90, 82, 84),
    "Seattle Seahawks": (82, 82, 82, 83, 79, 80),
    "Miami Dolphins": (85, 78, 84, 82, 79, 81),
    "Dallas Cowboys": (83, 81, 84, 80, 80, 82),
    "Atlanta Falcons": (82, 80, 81, 82, 78, 79),
    "Arizona Cardinals": (81, 79, 82, 84, 78, 77),
    "Chicago Bears": (80, 82, 81, 83, 79, 80),
    "Indianapolis Colts": (81, 79, 81, 81, 78, 79),
    "Jacksonville Jaguars": (80, 79, 82, 80, 78, 78),
    "New York Jets": (77, 84, 78, 79, 81, 80),
    "New England Patriots": (78, 80, 79, 82, 79, 79),
    "Cleveland Browns": (75, 85, 74, 78, 82, 80),
    "New Orleans Saints": (77, 80, 77, 79, 78, 79),
    "Las Vegas Raiders": (76, 78, 75, 79, 79, 81),
    "New York Giants": (75, 78, 76, 77, 80, 78),
    "Carolina Panthers": (76, 76, 78, 78, 78, 77),
    "Tennessee Titans": (75, 77, 76, 78, 79, 78),
}

NFL_TEAM_RATINGS = {
    team: {
        "offense": values[0],
        "defense": values[1],
        "quarterback": values[2],
        "coaching": values[3],
        "strength_of_schedule": values[4],
        "special_teams": values[5],
    }
    for team, values in _TEAM_TIERS.items()
}
