from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path(__file__).with_name('bets.db')
SPORTS = ['NFL', 'College Football', 'NBA', 'Tennis', 'UFC', 'Boxing']
MARKETS = {
    'NFL': ['Moneyline', 'Spread', 'Game Total', 'Team Total', 'Player Prop'],
    'College Football': ['Moneyline', 'Spread', 'Game Total', 'Team Total', 'Player Prop'],
    'NBA': ['Moneyline', 'Spread', 'Game Total', 'Team Total', 'Player Prop'],
    'Tennis': ['Moneyline', 'Set Handicap', 'Game Handicap', 'Match Total', 'Exact Set Score'],
    'UFC': ['Moneyline', 'Method of Victory', 'Round Total', 'Fight Goes Distance'],
    'Boxing': ['Moneyline', 'Method of Victory', 'Round Total', 'Fight Goes Distance'],
}


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                sport TEXT NOT NULL,
                event TEXT NOT NULL,
                selection TEXT NOT NULL,
                market TEXT NOT NULL,
                american_odds INTEGER NOT NULL,
                estimated_probability REAL NOT NULL,
                stake REAL NOT NULL,
                target_profit REAL NOT NULL,
                starting_bankroll REAL NOT NULL,
                edge REAL NOT NULL,
                expected_value REAL NOT NULL,
                analysis_json TEXT,
                notes TEXT,
                result TEXT DEFAULT 'Pending',
                profit_loss REAL DEFAULT 0
            )
        ''')


def american_to_decimal(odds: int) -> float:
    if odds == 0:
        raise ValueError('Odds cannot be zero.')
    return 1 + (100 / abs(odds) if odds < 0 else odds / 100)


def implied_probability(odds: int) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    if odds > 0:
        return 100 / (odds + 100)
    raise ValueError('Odds cannot be zero.')


def stake_to_win(odds: int, target_profit: float) -> float:
    if odds < 0:
        return target_profit * abs(odds) / 100
    if odds > 0:
        return target_profit * 100 / odds
    raise ValueError('Odds cannot be zero.')


def kelly_fraction(decimal_odds: float, probability: float) -> float:
    b = decimal_odds - 1
    return max(0.0, (b * probability - (1 - probability)) / b)


def simulate(bankroll: float, odds: int, probability: float, stake: float,
             n_bets: int, n_runs: int, uncertainty: float) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    decimal = american_to_decimal(odds)
    rows = []
    for _ in range(n_runs):
        current = bankroll
        peak = bankroll
        max_dd = 0.0
        losing = 0
        longest = 0
        for _ in range(n_bets):
            if current <= 0:
                break
            true_p = float(np.clip(rng.normal(probability, uncertainty), .01, .99))
            risk = min(stake, current)
            if rng.random() < true_p:
                current += risk * (decimal - 1)
                losing = 0
            else:
                current -= risk
                losing += 1
                longest = max(longest, losing)
            peak = max(peak, current)
            max_dd = max(max_dd, 0 if peak == 0 else (peak-current)/peak)
        rows.append((current, max_dd, longest))
    return pd.DataFrame(rows, columns=['Final Bankroll', 'Max Drawdown', 'Longest Losing Streak'])


def text_area(label: str, placeholder: str = '') -> str:
    return st.text_area(label, placeholder=placeholder, height=75)


def matchup_inputs(sport: str) -> tuple[str, dict]:
    data = {}
    if sport in {'NFL', 'College Football'}:
        c1, c2 = st.columns(2)
        with c1:
            team_a = st.text_input('Team A')
            data['Team A recent form'] = text_area('Team A recent form', 'Last 3-5 games, efficiency trend, injuries, home/road form')
            data['Team A strengths'] = text_area('Team A strengths', 'Run game, pressure rate, coverage, explosive plays, red zone, etc.')
            data['Team A weaknesses'] = text_area('Team A weaknesses', 'Run defense, pass protection, turnovers, secondary, etc.')
        with c2:
            team_b = st.text_input('Team B')
            data['Team B recent form'] = text_area('Team B recent form')
            data['Team B strengths'] = text_area('Team B strengths')
            data['Team B weaknesses'] = text_area('Team B weaknesses')
        data['Style matchup'] = text_area('How their styles match up', 'Pace, run/pass tendencies, pressure vs protection, man/zone, explosive-play profile')
        data['Team A matchup advantages'] = text_area('Team A matchup advantages')
        data['Team A matchup disadvantages'] = text_area('Team A matchup disadvantages')
        data['Team B matchup advantages'] = text_area('Team B matchup advantages')
        data['Team B matchup disadvantages'] = text_area('Team B matchup disadvantages')
        data['Key injuries/weather'] = text_area('Key injuries, weather, travel, rest')
        return f'{team_a} vs {team_b}'.strip(' vs'), data

    if sport == 'NBA':
        c1, c2 = st.columns(2)
        with c1:
            team_a = st.text_input('Team A')
            data['Team A recent form'] = text_area('Team A recent form', 'Last 5-10 games, offensive/defensive trend, shooting, rest')
            data['Team A strengths'] = text_area('Team A strengths')
            data['Team A weaknesses'] = text_area('Team A weaknesses')
        with c2:
            team_b = st.text_input('Team B')
            data['Team B recent form'] = text_area('Team B recent form')
            data['Team B strengths'] = text_area('Team B strengths')
            data['Team B weaknesses'] = text_area('Team B weaknesses')
        data['Matchup advantages'] = text_area('Matchup advantages and disadvantages', 'Pace, rebounding, rim pressure, 3-point profile, switching, bench, size')
        data['Injuries/rest'] = text_area('Injuries, rest, back-to-back, travel')
        return f'{team_a} vs {team_b}'.strip(' vs'), data

    if sport == 'Tennis':
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.text_input('Player 1')
            data['Player 1 recent form'] = text_area('Player 1 recent form')
            data['Player 1 strengths'] = text_area('Player 1 strengths')
            data['Player 1 weaknesses'] = text_area('Player 1 weaknesses')
        with c2:
            p2 = st.text_input('Player 2')
            data['Player 2 recent form'] = text_area('Player 2 recent form')
            data['Player 2 strengths'] = text_area('Player 2 strengths')
            data['Player 2 weaknesses'] = text_area('Player 2 weaknesses')
        data['Tour'] = st.selectbox('Tour', ['ATP', 'WTA'])
        data['Surface'] = st.selectbox('Surface', ['Hard', 'Clay', 'Grass', 'Indoor hard'])
        data['Format'] = st.selectbox('Format', ['Best of 3', 'Best of 5'])
        data['Matchup'] = text_area('Matchup advantages and disadvantages', 'Serve/return profile, movement, backhand/forehand pattern, surface fit, fatigue')
        data['Injuries'] = text_area('Injuries, fatigue, travel, retirement risk')
        return f'{p1} vs {p2}'.strip(' vs'), data

    c1, c2 = st.columns(2)
    with c1:
        f1 = st.text_input('Fighter 1')
        data['Fighter 1 recent form'] = text_area('Fighter 1 recent form')
        data['Fighter 1 strengths'] = text_area('Fighter 1 strengths')
        data['Fighter 1 weaknesses'] = text_area('Fighter 1 weaknesses')
        data['Fighter 1 power at weight'] = text_area('Fighter 1 power at this weight')
        data['Fighter 1 chin durability'] = text_area('Fighter 1 chin durability')
    with c2:
        f2 = st.text_input('Fighter 2')
        data['Fighter 2 recent form'] = text_area('Fighter 2 recent form')
        data['Fighter 2 strengths'] = text_area('Fighter 2 strengths')
        data['Fighter 2 weaknesses'] = text_area('Fighter 2 weaknesses')
        data['Fighter 2 power at weight'] = text_area('Fighter 2 power at this weight')
        data['Fighter 2 chin durability'] = text_area('Fighter 2 chin durability')
    data['Weight class'] = st.text_input('Weight class')
    data['Matchup advantages'] = text_area('Matchup advantages and disadvantages', 'Range, stance, wrestling/grappling, pace, defense, body work, cardio')
    data['Injuries'] = text_area('Known injuries, layoffs, weight-cut concerns')
    return f'{f1} vs {f2}'.strip(' vs'), data


def save_bet(values: tuple) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''INSERT INTO bets (
            created_at,sport,event,selection,market,american_odds,estimated_probability,
            stake,target_profit,starting_bankroll,edge,expected_value,analysis_json,notes
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', values)


st.set_page_config(page_title='Michael Betting Dashboard', page_icon='📊', layout='wide')
init_db()
st.title('Michael Betting Dashboard')
st.caption('Decision-support and simulation only. No bets are placed through this app.')

with st.sidebar:
    st.header('Your Defaults')
    bankroll = st.number_input('Current bankroll', min_value=100.0, value=100000.0, step=1000.0)
    target_profit = st.number_input('Default profit target', min_value=100.0, value=10000.0, step=500.0,
                                    help='The app calculates how much must be risked to win this amount.')
    favorite_focus = st.checkbox('Favorite-focused model', value=True,
                                 help='Flags underdogs and emphasizes break-even risk on favorites.')
    max_risk_pct = st.slider('Maximum risk per bet (% of bankroll)', .5, 50.0, 20.0, .5) / 100

home, new_bet, simulations, history = st.tabs(['Dashboard', 'Analyze Bet', 'Simulations', 'History'])

with home:
    bets = pd.read_sql_query('SELECT * FROM bets ORDER BY id DESC', sqlite3.connect(DB_PATH))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Bankroll', f'${bankroll:,.0f}')
    c2.metric('Default Win Target', f'${target_profit:,.0f}')
    c3.metric('Logged Bets', len(bets))
    resolved = bets[bets['result'].isin(['Win','Loss'])] if not bets.empty else bets
    roi = 0 if resolved.empty else resolved['profit_loss'].sum() / max(resolved['stake'].sum(), 1)
    c4.metric('Recorded ROI', f'{roi:.1%}')
    st.subheader('Your model profile')
    st.write('Primary tendency: favorites. Default objective: calculate the risk required to win $10,000 unless you override it. The dashboard still evaluates whether the price and estimated probability justify that exposure.')
    if bets.empty:
        st.info('No bets logged yet. Open Analyze Bet to create the first matchup report.')
    else:
        st.dataframe(bets[['created_at','sport','event','selection','american_odds','stake','target_profit','edge','result','profit_loss']].head(10), use_container_width=True)

with new_bet:
    sport = st.selectbox('Sport', SPORTS)
    event, analysis = matchup_inputs(sport)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        market = st.selectbox('Market', MARKETS[sport])
        selection = st.text_input('Your selection', placeholder='Example: Bills moneyline or Sinner -1.5 sets')
        odds = st.number_input('American odds', min_value=-5000, max_value=5000, value=-250, step=5)
        probability_pct = st.slider('Your estimated win probability', 1.0, 99.0, 75.0, .5)
    with c2:
        use_default_target = st.checkbox('Use $10,000 default profit target', value=True)
        this_target = target_profit if use_default_target else st.number_input('Profit target for this bet', min_value=100.0, value=5000.0, step=500.0)
        uncertainty_pct = st.slider('Probability uncertainty', 0.0, 15.0, 3.0, .5)
        notes = st.text_area('Final reasoning / notes', height=120)

    if odds != 0:
        dec = american_to_decimal(int(odds))
        implied = implied_probability(int(odds))
        p = probability_pct / 100
        stake = stake_to_win(int(odds), this_target)
        edge = p - implied
        ev = p * this_target - (1-p) * stake
        full_kelly = kelly_fraction(dec, p)
        quarter_kelly = full_kelly * .25 * bankroll
        risk_pct = stake / bankroll
        favorite_flag = odds < 0

        st.subheader('Bet Decision Summary')
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric('Risk to Win', f'${stake:,.0f}')
        m2.metric('Break-even', f'{implied:.1%}')
        m3.metric('Your Edge', f'{edge:+.1%}')
        m4.metric('Expected Value', f'${ev:,.0f}')
        m5.metric('Quarter Kelly', f'${quarter_kelly:,.0f}')

        warnings = []
        if favorite_focus and not favorite_flag:
            warnings.append('This is an underdog, outside your usual favorite-focused profile.')
        if edge <= 0:
            warnings.append('Your estimated probability does not beat the sportsbook break-even probability.')
        if risk_pct > max_risk_pct:
            warnings.append(f'This requires risking {risk_pct:.1%} of bankroll, above your {max_risk_pct:.1%} limit.')
        if stake > quarter_kelly and quarter_kelly > 0:
            warnings.append('The risk required to win your target exceeds the quarter-Kelly amount.')
        if warnings:
            for warning in warnings: st.warning(warning)
        else:
            st.success('The bet clears the current edge and bankroll-risk checks. This does not guarantee the wager is good; the probability estimate remains the key input.')

        if st.button('Save Matchup Report', type='primary'):
            save_bet((datetime.now().isoformat(timespec='minutes'), sport, event or 'Unspecified event', selection or 'Unspecified selection', market, int(odds), p, stake, this_target, bankroll, edge, ev, json.dumps(analysis), notes))
            st.success('Saved to your dashboard history.')
    else:
        st.error('American odds cannot be zero.')

with simulations:
    st.subheader('Repeat-Bet Bankroll Simulation')
    c1,c2,c3 = st.columns(3)
    with c1:
        sim_odds = st.number_input('Odds', min_value=-5000, max_value=5000, value=-250, step=5, key='sim_odds')
        sim_prob = st.slider('Estimated win probability', 1.0, 99.0, 75.0, .5, key='sim_prob') / 100
    with c2:
        sim_target = st.number_input('Profit target per bet', min_value=100.0, value=10000.0, step=500.0)
        sim_bets = st.number_input('Number of similar bets', 1, 1000, 100, 10)
    with c3:
        sim_runs = st.number_input('Simulation runs', 100, 50000, 10000, 1000)
        sim_uncertainty = st.slider('Probability uncertainty (%)', 0.0, 15.0, 3.0, .5, key='sim_uncertainty') / 100
    if st.button('Run 10,000-Path Simulation') and sim_odds != 0:
        sim_stake = stake_to_win(int(sim_odds), sim_target)
        results = simulate(bankroll, int(sim_odds), sim_prob, sim_stake, int(sim_bets), int(sim_runs), sim_uncertainty)
        a,b,c,d = st.columns(4)
        a.metric('Median Final Bankroll', f"${results['Final Bankroll'].median():,.0f}")
        b.metric('Chance of Profit', f"{(results['Final Bankroll'] > bankroll).mean():.1%}")
        c.metric('Median Max Drawdown', f"{results['Max Drawdown'].median():.1%}")
        d.metric('Median Longest Losing Streak', f"{results['Longest Losing Streak'].median():.0f}")
        fig = px.histogram(results, x='Final Bankroll', nbins=60, title='Distribution of Final Bankrolls')
        st.plotly_chart(fig, use_container_width=True)

with history:
    bets = pd.read_sql_query('SELECT * FROM bets ORDER BY id DESC', sqlite3.connect(DB_PATH))
    if bets.empty:
        st.info('No saved bets yet.')
    else:
        st.dataframe(bets, use_container_width=True)
        st.download_button('Download history as CSV', bets.to_csv(index=False), 'bet_history.csv', 'text/csv')
