from engine.nfl_team_quality import (
    TeamQualityInputs,
    calculate_team_quality,
    compare_team_quality,
)


chiefs = calculate_team_quality(
    "Kansas City Chiefs",
    TeamQualityInputs(
        quarterback=96,
        offense=90,
        defense=86,
        coaching=95,
        offensive_line=84,
        defensive_line=88,
        skill_positions=82,
        secondary=85,
        special_teams=83,
        continuity=92,
        injury_adjustment=0,
        rookie_adjustment=0,
    ),
)

bills = calculate_team_quality(
    "Buffalo Bills",
    TeamQualityInputs(
        quarterback=93,
        offense=89,
        defense=84,
        coaching=87,
        offensive_line=85,
        defensive_line=84,
        skill_positions=84,
        secondary=82,
        special_teams=80,
        continuity=88,
        injury_adjustment=0,
        rookie_adjustment=0,
    ),
)

matchup = compare_team_quality(
    away_team=bills,
    home_team=chiefs,
    home_field_advantage=1.5,
)

print("Chiefs rating:", chiefs.final_rating)
print("Bills rating:", bills.final_rating)
print("Fair spread for home team:", matchup["fair_spread_home"])
print("Favored team:", matchup["favored_team"])
