from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT = ROOT / 'data' / 'nfl' / 'team_snapshot.csv'
LOS_ADJUSTMENT_CAP = 0.45


def _num(v: Any, default=float('nan')) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _rank(frame: pd.DataFrame, col: str, higher=True) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(50.0, index=frame.index, dtype=float)
    s = pd.to_numeric(frame[col], errors='coerce')
    if not higher:
        s = -s
    return s.rank(pct=True).fillna(.5) * 100.0


def _profile(frame: pd.DataFrame, team: str) -> dict[str, Any] | None:
    row = frame[frame['team'].astype(str).eq(team)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {k: r.get(k) for k in frame.columns}


def _evidence_weight(row: dict[str, Any], season: int, week: int | None) -> float:
    source_season = int(_num(row.get('season'), 0) or 0)
    if source_season != int(season):
        return 0.20
    w = int(week or _num(row.get('through_week'), 1) or 1)
    return min(1.0, 0.35 + 0.13 * max(0, w - 1))


def build_los_matchup_context(*, away_team: str, home_team: str, season: int, week: int | None, snapshot_path: Path | str = DEFAULT_SNAPSHOT) -> dict[str, Any]:
    path = Path(snapshot_path)
    if not path.exists():
        return {'available': False, 'home_margin_adjustment': 0.0, 'summary': 'Real NFL line-of-scrimmage data is not available yet.'}
    frame = pd.read_csv(path)
    needed = {'team','offense_sack_rate','offense_qb_hit_rate','defense_sack_rate','defense_qb_hit_rate','rush_success','rush_success_allowed'}
    if frame.empty or not needed.issubset(frame.columns):
        return {'available': False, 'home_margin_adjustment': 0.0, 'summary': 'Real NFL line-of-scrimmage data is incomplete.'}
    away = _profile(frame, away_team); home = _profile(frame, home_team)
    if away is None or home is None:
        return {'available': False, 'home_margin_adjustment': 0.0, 'summary': 'Real NFL line-of-scrimmage data is incomplete for this matchup.'}

    grades = pd.DataFrame({'team': frame['team']})
    grades['protect'] = (_rank(frame,'offense_sack_rate',False)*.45 + _rank(frame,'offense_qb_hit_rate',False)*.25 + _rank(frame,'qb_epa_when_disrupted',True)*.30)
    grades['rush'] = (_rank(frame,'defense_sack_rate',True)*.50 + _rank(frame,'defense_qb_hit_rate',True)*.30 + _rank(frame,'defense_disruption_rate',True)*.20)
    grades['run_block'] = (_rank(frame,'rush_success',True)*.45 + _rank(frame,'rush_epa',True)*.30 + _rank(frame,'offense_run_stuff_rate',False)*.25)
    grades['run_front'] = (_rank(frame,'rush_success_allowed',False)*.45 + _rank(frame,'rush_epa_allowed',False)*.30 + _rank(frame,'defense_run_stuff_rate',True)*.25)
    g = grades.set_index('team')

    def side(offense: str, defense: str) -> dict[str, Any]:
        pass_edge = float(g.loc[offense,'protect'] - g.loc[defense,'rush'])
        run_edge = float(g.loc[offense,'run_block'] - g.loc[defense,'run_front'])
        return {'team': offense, 'opponent': defense, 'pass_edge': pass_edge, 'run_edge': run_edge, 'combined': pass_edge*.58 + run_edge*.42}

    away_side = side(away_team, home_team); home_side = side(home_team, away_team)
    raw_home = (home_side['combined'] - away_side['combined']) * 0.012
    ew = min(_evidence_weight(away, season, week), _evidence_weight(home, season, week))
    adj = max(-LOS_ADJUSTMENT_CAP, min(LOS_ADJUSTMENT_CAP, raw_home * ew))
    leader = 'Even' if abs(adj) < .06 else home_team if adj > 0 else away_team
    strength = 'Even' if leader == 'Even' else 'Slight' if abs(adj) < .22 else 'Moderate'

    def out(team, row, side_row):
        return {
            'team': team,
            'pass_protection_grade': round(float(g.loc[team,'protect']),1),
            'pass_rush_grade': round(float(g.loc[team,'rush']),1),
            'run_block_grade': round(float(g.loc[team,'run_block']),1),
            'run_front_grade': round(float(g.loc[team,'run_front']),1),
            'sack_rate_allowed': _num(row.get('offense_sack_rate')),
            'qb_hit_rate_allowed': _num(row.get('offense_qb_hit_rate')),
            'disruption_rate_allowed': _num(row.get('offense_disruption_rate')),
            'qb_epa_when_disrupted': _num(row.get('qb_epa_when_disrupted')),
            'run_stuff_rate_allowed': _num(row.get('offense_run_stuff_rate')),
            'sack_rate_generated': _num(row.get('defense_sack_rate')),
            'qb_hit_rate_generated': _num(row.get('defense_qb_hit_rate')),
            'disruption_rate_generated': _num(row.get('defense_disruption_rate')),
            'run_stuff_rate_forced': _num(row.get('defense_run_stuff_rate')),
            'rush_success': _num(row.get('rush_success')),
            'rush_success_allowed': _num(row.get('rush_success_allowed')),
            'pass_matchup_edge': round(side_row['pass_edge'],1),
            'run_matchup_edge': round(side_row['run_edge'],1),
            'season': int(_num(row.get('season'),0) or 0),
            'through_week': int(_num(row.get('through_week'),0) or 0),
        }

    summary = ('The real-performance trench matchup is effectively even.' if leader == 'Even' else f'{leader} has a {strength.lower()} real-performance line-of-scrimmage edge after pass protection/pass rush and run-block/run-front interactions are combined.')
    return {
        'available': True, 'away': out(away_team, away, away_side), 'home': out(home_team, home, home_side),
        'overall_advantage': leader, 'overall_strength': strength, 'home_margin_adjustment': round(adj,3),
        'evidence_weight': round(ew,3), 'summary': summary,
        'guardrail': 'This layer is capped at ±0.45 points. It uses opponent-specific interaction and disruption/stuff resilience as a small refinement, not a second full award for OL/DL talent.',
        'source': 'nflverse regular-season play-by-play',
    }
