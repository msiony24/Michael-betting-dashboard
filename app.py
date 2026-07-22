
import io
import math
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from tennis_engine import (
    analyze_matchup, implied_probability as tennis_implied_probability,
    load_tennis_data, player_options, tournament_catalog
)

st.set_page_config(
    page_title="Macabets",
    page_icon="📊",
    layout="wide",
)

SPORTS = ["NFL", "College Football", "NBA", "Tennis", "UFC", "Boxing"]
STATUSES = ["Pending", "Won", "Lost", "Void", "Cashed Out"]
BET_TYPES = ["Moneyline", "Spread", "Total", "Prop", "Parlay", "Live"]
DEFAULT_COLUMNS = [
    "id", "date", "sport", "event", "selection", "bet_type", "odds",
    "stake", "target_profit", "status", "result_profit", "book",
    "confidence", "notes"
]


def money(value):
    return f"${value:,.2f}"


def american_to_decimal(odds):
    if odds == 0:
        return 1.0
    return 1 + (100 / abs(odds) if odds < 0 else odds / 100)


def implied_probability(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    if odds > 0:
        return 100 / (odds + 100)
    return 0.0


def stake_to_win(odds, target):
    if odds < 0:
        return target * abs(odds) / 100
    if odds > 0:
        return target * 100 / odds
    return 0.0


def potential_profit(odds, stake):
    if odds < 0:
        return stake * 100 / abs(odds)
    if odds > 0:
        return stake * odds / 100
    return 0.0


def kelly_fraction(model_prob, odds):
    dec = american_to_decimal(odds)
    b = dec - 1
    q = 1 - model_prob
    if b <= 0:
        return 0.0
    return max(0.0, (b * model_prob - q) / b)


def probability_to_american(probability):
    probability = min(max(float(probability), 0.0001), 0.9999)
    if probability >= 0.5:
        return int(round(-100 * probability / (1 - probability)))
    return int(round(100 * (1 - probability) / probability))


def no_vig_probability(selection_odds, opponent_odds):
    selection_implied = implied_probability(selection_odds)
    opponent_implied = implied_probability(opponent_odds)
    total = selection_implied + opponent_implied
    if total <= 0:
        return 0.5
    return selection_implied / total


def clamp_probability(probability):
    return min(max(float(probability), 0.01), 0.99)


def recommendation_from_edge(edge, confidence, data_quality):
    if data_quality <= 4:
        return "PASS", "Data quality is too weak for a strong recommendation."
    if confidence >= 8 and edge >= 0.05:
        return "GREEN LIGHT", "Strong model edge with high confidence."
    if confidence >= 7 and edge >= 0.03:
        return "LEAN", "Positive edge, but not strong enough for a full green light."
    if edge > 0:
        return "PASS", "Macabets sees some value, but the edge or confidence is insufficient."
    return "PASS", "The market price is equal to or better than the Macabets projection."


def line_value_label(edge):
    if edge >= 0.07:
        return "Major value"
    if edge >= 0.05:
        return "Strong value"
    if edge >= 0.03:
        return "Moderate value"
    if edge > 0:
        return "Small value"
    if edge == 0:
        return "Fairly priced"
    return "No value"


def context_adjustment(score, confidence):
    return (float(score) / 10.0) * (float(confidence) / 100.0) * 0.05


def empty_bets():
    return pd.DataFrame(columns=DEFAULT_COLUMNS)


def normalize_bets(df):
    clean = df.copy()
    for col in DEFAULT_COLUMNS:
        if col not in clean.columns:
            clean[col] = ""
    clean = clean[DEFAULT_COLUMNS]
    numeric = ["odds", "stake", "target_profit", "result_profit", "confidence"]
    for col in numeric:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0)
    clean["status"] = clean["status"].replace("", "Pending")
    return clean


if "bets" not in st.session_state:
    st.session_state.bets = empty_bets()

if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100000.0

if "target_profit" not in st.session_state:
    st.session_state.target_profit = 10000.0

if "analyses" not in st.session_state:
    st.session_state.analyses = []


st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.small-note {color: #777; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

st.title("Macabets")
st.caption("Betting intelligence, fair-line pricing, matchup analysis and bankroll risk control.")

with st.sidebar:
    st.header("Core Settings")
    st.session_state.bankroll = st.number_input(
        "Starting bankroll",
        min_value=0.0,
        value=float(st.session_state.bankroll),
        step=1000.0,
    )
    st.session_state.target_profit = st.number_input(
        "Default target profit",
        min_value=1.0,
        value=float(st.session_state.target_profit),
        step=500.0,
    )
    st.divider()
    st.subheader("Restore / Import")
    uploaded = st.file_uploader("Upload a prior bets CSV", type=["csv"])
    if uploaded is not None:
        try:
            imported = normalize_bets(pd.read_csv(uploaded))
            st.session_state.bets = imported
            st.success(f"Loaded {len(imported)} bets.")
        except Exception as exc:
            st.error(f"Could not load CSV: {exc}")

bets = normalize_bets(st.session_state.bets)
settled = bets[bets["status"].isin(["Won", "Lost", "Void", "Cashed Out"])]
pending = bets[bets["status"] == "Pending"]
net_profit = float(settled["result_profit"].sum()) if not settled.empty else 0.0
current_bankroll = st.session_state.bankroll + net_profit
total_staked = float(settled["stake"].sum()) if not settled.empty else 0.0
roi = net_profit / total_staked if total_staked else 0.0
decisions = settled[settled["status"].isin(["Won", "Lost"])]
wins = int((decisions["status"] == "Won").sum()) if not decisions.empty else 0
win_rate = wins / len(decisions) if len(decisions) else 0.0
pending_exposure = float(pending["stake"].sum()) if not pending.empty else 0.0

tabs = st.tabs([
    "Dashboard", "New Bet", "Bet Ledger", "Matchup Lab",
    "Risk Simulator", "Performance", "Backup"
])

with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Bankroll", money(current_bankroll), money(net_profit))
    c2.metric("Pending Exposure", money(pending_exposure))
    c3.metric("Settled ROI", f"{roi:.1%}")
    c4.metric("Win Rate", f"{win_rate:.1%}")
    c5.metric("Bets Logged", f"{len(bets)}")

    if current_bankroll > 0:
        exposure_pct = pending_exposure / current_bankroll
        if exposure_pct >= 0.25:
            st.error(f"Pending exposure is {exposure_pct:.1%} of bankroll. This is a major concentration risk.")
        elif exposure_pct >= 0.15:
            st.warning(f"Pending exposure is {exposure_pct:.1%} of bankroll. Proceed carefully.")
        elif exposure_pct > 0:
            st.info(f"Pending exposure is {exposure_pct:.1%} of bankroll.")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Recent Bets")
        if bets.empty:
            st.write("No bets logged yet.")
        else:
            view = bets.sort_values("id", ascending=False).head(10)
            st.dataframe(
                view[["date", "sport", "event", "selection", "odds", "stake", "status", "result_profit"]],
                use_container_width=True,
                hide_index=True,
            )
    with right:
        st.subheader("Profit by Sport")
        if settled.empty:
            st.write("Settle bets to populate this chart.")
        else:
            profit_sport = settled.groupby("sport", as_index=False)["result_profit"].sum()
            fig, ax = plt.subplots()
            ax.bar(profit_sport["sport"], profit_sport["result_profit"])
            ax.axhline(0, linewidth=1)
            ax.set_ylabel("Profit / Loss ($)")
            ax.tick_params(axis="x", rotation=35)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

with tabs[1]:
    st.subheader("Enter a Bet")
    a, b, c = st.columns(3)
    sport = a.selectbox("Sport", SPORTS)
    event_date = b.date_input("Event date", value=date.today())
    bet_type = c.selectbox("Bet type", BET_TYPES)

    event = st.text_input("Event / matchup", placeholder="Bills vs Jets")
    selection = st.text_input("Selection", placeholder="Bills moneyline")

    d, e, f = st.columns(3)
    odds = d.number_input("American odds", value=-200, step=5)
    target_profit = e.number_input(
        "Target profit",
        min_value=1.0,
        value=float(st.session_state.target_profit),
        step=500.0,
    )
    suggested_stake = stake_to_win(int(odds), target_profit)
    stake_mode = f.radio("Stake method", ["Risk enough to win target", "Enter my own stake"], horizontal=True)

    if stake_mode == "Risk enough to win target":
        stake = suggested_stake
        f.metric("Required stake", money(stake))
    else:
        stake = f.number_input("Stake", min_value=0.0, value=1000.0, step=100.0)

    p_profit = potential_profit(int(odds), stake)
    implied = implied_probability(int(odds))
    g, h, i = st.columns(3)
    g.metric("Potential profit", money(p_profit))
    h.metric("Implied probability", f"{implied:.1%}")
    i.metric("Total return", money(stake + p_profit))

    j, k, l = st.columns(3)
    book = j.text_input("Sportsbook", placeholder="Optional")
    confidence = k.slider("Confidence", 1, 10, 7)
    status = l.selectbox("Initial status", STATUSES, index=0)
    notes = st.text_area("Notes / thesis")

    stake_pct = stake / current_bankroll if current_bankroll > 0 else 0
    if odds <= -500:
        st.warning("Very heavy favorite: one loss can erase several wins. Confirm the price is justified.")
    if stake_pct >= 0.20:
        st.error(f"This stake is {stake_pct:.1%} of the current bankroll.")
    elif stake_pct >= 0.10:
        st.warning(f"This stake is {stake_pct:.1%} of the current bankroll.")
    elif stake_pct > 0:
        st.caption(f"Stake size: {stake_pct:.1%} of current bankroll.")

    if st.button("Add Bet", type="primary", use_container_width=True):
        if not event.strip() or not selection.strip():
            st.error("Enter both the event and selection.")
        elif stake <= 0:
            st.error("Stake must be greater than zero.")
        else:
            next_id = int(bets["id"].max()) + 1 if not bets.empty else 1
            initial_result = 0.0
            if status == "Won":
                initial_result = p_profit
            elif status == "Lost":
                initial_result = -stake
            row = {
                "id": next_id,
                "date": event_date.isoformat(),
                "sport": sport,
                "event": event.strip(),
                "selection": selection.strip(),
                "bet_type": bet_type,
                "odds": int(odds),
                "stake": float(stake),
                "target_profit": float(target_profit),
                "status": status,
                "result_profit": float(initial_result),
                "book": book.strip(),
                "confidence": confidence,
                "notes": notes.strip(),
            }
            st.session_state.bets = pd.concat([bets, pd.DataFrame([row])], ignore_index=True)
            st.success("Bet added. Open Bet Ledger to settle or edit it.")

with tabs[2]:
    st.subheader("Bet Ledger")
    if bets.empty:
        st.write("No bets have been added.")
    else:
        filter_col1, filter_col2 = st.columns(2)
        sport_filter = filter_col1.multiselect("Filter sport", SPORTS, default=SPORTS)
        status_filter = filter_col2.multiselect("Filter status", STATUSES, default=STATUSES)
        filtered = bets[bets["sport"].isin(sport_filter) & bets["status"].isin(status_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Settle or Update a Bet")
        selected_id = st.selectbox(
            "Bet ID",
            bets["id"].astype(int).tolist(),
            format_func=lambda x: f"#{x} — {bets.loc[bets['id'] == x, 'selection'].iloc[0]}",
        )
        selected = bets[bets["id"] == selected_id].iloc[0]
        u1, u2, u3 = st.columns(3)
        new_status = u1.selectbox(
            "Status",
            STATUSES,
            index=STATUSES.index(selected["status"]) if selected["status"] in STATUSES else 0,
        )
        default_result = float(selected["result_profit"])
        if new_status == "Won":
            default_result = potential_profit(int(selected["odds"]), float(selected["stake"]))
        elif new_status == "Lost":
            default_result = -float(selected["stake"])
        elif new_status == "Void":
            default_result = 0.0
        result_profit = u2.number_input("Net result", value=float(default_result), step=100.0)
        updated_notes = u3.text_input("Updated note", value=str(selected["notes"]))

        col_save, col_delete = st.columns(2)
        if col_save.button("Save Update", type="primary", use_container_width=True):
            idx = st.session_state.bets.index[st.session_state.bets["id"] == selected_id][0]
            st.session_state.bets.at[idx, "status"] = new_status
            st.session_state.bets.at[idx, "result_profit"] = float(result_profit)
            st.session_state.bets.at[idx, "notes"] = updated_notes
            st.success("Bet updated.")
            st.rerun()

        if col_delete.button("Delete Bet", use_container_width=True):
            st.session_state.bets = st.session_state.bets[st.session_state.bets["id"] != selected_id].reset_index(drop=True)
            st.success("Bet deleted.")
            st.rerun()

with tabs[3]:
    st.subheader("Macabets Matchup Lab")
    st.caption("Select the matchup and event. Macabets builds the statistical and contextual projection automatically.")

    sport_lab = st.selectbox("Choose sport", SPORTS, key="lab_sport")

    if sport_lab == "Tennis":
        try:
            with st.spinner("Loading ATP player and match database..."):
                tennis_matches, tennis_load_errors = load_tennis_data()

            available_players = player_options(tennis_matches)
            tournaments = tournament_catalog(tennis_matches)

            if tennis_load_errors:
                with st.expander("Data status"):
                    st.caption(
                        "Some yearly files were unavailable, but Macabets loaded the remaining database: "
                        + "; ".join(tennis_load_errors)
                    )

            st.markdown("### Matchup")
            p1, p2 = st.columns(2)
            player_a = p1.selectbox(
                "Player A",
                available_players,
                index=None,
                placeholder="Start typing a player's name",
                key="auto_player_a",
            )
            player_b = p2.selectbox(
                "Player B",
                available_players,
                index=None,
                placeholder="Start typing a player's name",
                key="auto_player_b",
            )

            event1, event2, event3, event4 = st.columns(4)
            tournament = event1.selectbox(
                "Tournament",
                list(tournaments.keys()),
                index=None,
                placeholder="Start typing a tournament",
            )
            round_label = event2.selectbox(
                "Round",
                ["Qualifying", "R128", "R64", "R32", "R16", "Quarterfinal", "Semifinal", "Final"],
                index=4,
            )

            detected_surface = tournaments.get(tournament, {}).get("surface", "Hard") if tournament else "Hard"
            surface_options = ["Hard", "Clay", "Grass", "Carpet"]
            surface_index = surface_options.index(detected_surface) if detected_surface in surface_options else 0
            surface = event3.selectbox("Surface", surface_options, index=surface_index)
            event_date = event4.date_input("Event date", value=date.today())

            st.markdown("### Your Form Read")
            st.caption(
                "This is the only judgment call Macabets asks from you. "
                "The database also calculates recent form independently."
            )
            f1, f2 = st.columns(2)
            form_a = f1.slider(
                f"{player_a or 'Player A'} form",
                1, 10, 5,
                help="1 = extremely poor current form; 5 = neutral; 10 = elite current form."
            )
            form_b = f2.slider(
                f"{player_b or 'Player B'} form",
                1, 10, 5,
                help="1 = extremely poor current form; 5 = neutral; 10 = elite current form."
            )

            st.markdown("### Market Price")
            st.caption(
                "Optional. Macabets can calculate its fair line without sportsbook odds. "
                "Enter current odds only when you want an edge and recommendation."
            )
            o1, o2 = st.columns(2)
            market_a_text = o1.text_input("Current odds for Player A", placeholder="-180")
            market_b_text = o2.text_input("Current odds for Player B", placeholder="+155")

            run_disabled = not player_a or not player_b or player_a == player_b or not tournament
            if st.button(
                "Run Automatic Macabets Analysis",
                type="primary",
                use_container_width=True,
                disabled=run_disabled,
            ):
                result = analyze_matchup(
                    tennis_matches,
                    player_a,
                    player_b,
                    tournament,
                    round_label,
                    surface,
                    form_a,
                    form_b,
                    event_date,
                )
                st.session_state["latest_tennis_analysis"] = result

            result = st.session_state.get("latest_tennis_analysis")
            if result and result["player_a"] == player_a and result["player_b"] == player_b:
                st.divider()
                st.markdown("## Macabets Projection")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Statistical base", f"{result['base_probability']:.1%}")
                m2.metric("Automatic context", f"{result['total_adjustment']:+.1%}")
                m3.metric("Macabets probability", f"{result['final_probability']:.1%}")
                m4.metric("Macabets fair line", f"{result['fair_line']:+d}")

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Overall Elo", f"{result['overall_elo_a']:.0f} vs {result['overall_elo_b']:.0f}")
                m6.metric("Surface Elo", f"{result['surface_elo_a']:.0f} vs {result['surface_elo_b']:.0f}")
                m7.metric("Confidence", f"{result['confidence']}/10")
                m8.metric("Data quality", f"{result['data_quality']}/10")

                market_a = None
                market_b = None
                try:
                    if market_a_text.strip():
                        market_a = int(market_a_text.replace("+", "").strip())
                    if market_b_text.strip():
                        market_b = int(market_b_text.replace("+", "").strip())
                except ValueError:
                    st.error("Enter American odds as a whole number, such as -180 or +155.")

                if market_a and market_b:
                    raw_a = tennis_implied_probability(market_a)
                    raw_b = tennis_implied_probability(market_b)
                    no_vig_a = raw_a / (raw_a + raw_b)
                    edge = result["final_probability"] - raw_a
                    no_vig_edge = result["final_probability"] - no_vig_a

                    e1, e2, e3 = st.columns(3)
                    e1.metric("Vegas implied", f"{raw_a:.1%}")
                    e2.metric("No-vig market", f"{no_vig_a:.1%}")
                    e3.metric("Macabets edge", f"{edge:+.1%}", f"{no_vig_edge:+.1%} vs no-vig")

                    if result["data_quality"] < 5:
                        st.error("PASS — insufficient data quality.")
                    elif edge >= 0.05 and result["confidence"] >= 7:
                        st.success("GREEN LIGHT — meaningful pricing edge with adequate confidence.")
                    elif edge >= 0.025 and result["confidence"] >= 6:
                        st.warning("LEAN — positive edge, but below Green Light standards.")
                    else:
                        st.error("PASS — the available price does not offer enough value.")
                else:
                    st.info(
                        "Macabets has produced its fair line. Enter both sportsbook prices above "
                        "to compare the projection with the market."
                    )

                st.markdown("### Automatic Context Report")
                context_rows = []
                for item in result["adjustments"]:
                    context_rows.append({
                        "Category": item["factor"],
                        "Probability impact": f"{item['adjustment']:+.2%}",
                        "Macabets reasoning": item["explanation"],
                    })
                st.dataframe(pd.DataFrame(context_rows), use_container_width=True, hide_index=True)

                st.markdown("### Player Comparison")
                pa = result["profile_a"]
                pb = result["profile_b"]
                comparison = pd.DataFrame([
                    {
                        "Metric": "Current ranking",
                        player_a: f"{pa['rank']:.0f}" if not pd.isna(pa["rank"]) else "Unknown",
                        player_b: f"{pb['rank']:.0f}" if not pd.isna(pb["rank"]) else "Unknown",
                    },
                    {"Metric": "Last-10 win rate", player_a: f"{pa['recent_win']:.0%}", player_b: f"{pb['recent_win']:.0%}"},
                    {"Metric": f"{surface} win rate", player_a: f"{pa['surface_win']:.0%}", player_b: f"{pb['surface_win']:.0%}"},
                    {"Metric": "Serve points won", player_a: f"{pa['serve_points_won']:.1%}", player_b: f"{pb['serve_points_won']:.1%}"},
                    {"Metric": "Return points won", player_a: f"{pa['return_points_won']:.1%}", player_b: f"{pb['return_points_won']:.1%}"},
                    {"Metric": "Matches in last 7 days", player_a: pa["matches_7"], player_b: pb["matches_7"]},
                    {"Metric": "Rest days", player_a: pa["rest_days"], player_b: pb["rest_days"]},
                    {"Metric": "Advanced-round win rate", player_a: f"{pa['advanced_round_win']:.0%}", player_b: f"{pb['advanced_round_win']:.0%}"},
                    {"Metric": "Deciding-match win rate", player_a: f"{pa['deciding_win']:.0%}", player_b: f"{pb['deciding_win']:.0%}"},
                ])
                st.dataframe(comparison, use_container_width=True, hide_index=True)

                st.caption(
                    "Fitness is estimated from workload, rest, inactivity and recorded retirements. "
                    "Psychological context is a historical performance proxy, not a claim about a player's private mental state. "
                    "Confirmed injuries and live odds require a separate live-data provider."
                )

                if "analyses" not in st.session_state:
                    st.session_state.analyses = []

                if st.button("Save Automatic Analysis", use_container_width=True):
                    save_record = {
                        "date": event_date.isoformat(),
                        "sport": "Tennis",
                        "event": tournament,
                        "selection": player_a,
                        "opponent": player_b,
                        "surface": surface,
                        "round": round_label,
                        "base_probability": result["base_probability"],
                        "context_adjustment": result["total_adjustment"],
                        "macabets_probability": result["final_probability"],
                        "fair_line": result["fair_line"],
                        "confidence": result["confidence"],
                        "data_quality": result["data_quality"],
                    }
                    st.session_state.analyses.append(save_record)
                    st.success("Automatic matchup analysis saved for this session.")

                if st.session_state.analyses:
                    analyses_df = pd.DataFrame(st.session_state.analyses)
                    st.download_button(
                        "Download Matchup Analyses CSV",
                        data=analyses_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"macabets_analyses_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as exc:
            st.error(f"The tennis database could not load: {exc}")
            st.caption(
                "Check that the Streamlit app has internet access, then reboot the app. "
                "No manual projection will be substituted for missing data."
            )

    else:
        st.info(
            "Automatic modeling is currently available for Tennis. "
            "NFL, college football, NBA and combat-sports engines will be added separately."
        )

with tabs[4]:
    st.subheader("Risk Simulator")
    r1, r2, r3, r4 = st.columns(4)
    sim_bankroll = r1.number_input("Simulation bankroll", min_value=100.0, value=float(current_bankroll), step=1000.0)
    sim_odds = r2.number_input("Odds per bet", value=-250, step=5)
    model_prob_pct = r3.slider("Your estimated true win probability", 1.0, 99.0, 75.0, 0.5)
    number_bets = r4.number_input("Number of bets", min_value=1, max_value=500, value=50, step=1)

    s1, s2, s3 = st.columns(3)
    target_each = s1.number_input("Target profit per bet", min_value=1.0, value=float(st.session_state.target_profit), step=500.0)
    simulations = s2.number_input("Monte Carlo runs", min_value=100, max_value=20000, value=3000, step=100)
    staking = s3.selectbox("Staking method", ["Target-profit stake", "Flat % bankroll", "Quarter Kelly"])
    flat_pct = st.slider("Flat stake %", 0.5, 25.0, 5.0, 0.5) / 100

    true_prob = model_prob_pct / 100
    implied = implied_probability(int(sim_odds))
    edge = true_prob - implied
    full_kelly = kelly_fraction(true_prob, int(sim_odds))
    quarter_kelly = full_kelly / 4

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Implied probability", f"{implied:.1%}")
    q2.metric("Estimated edge", f"{edge:.1%}")
    q3.metric("Full Kelly", f"{full_kelly:.1%}")
    q4.metric("Quarter Kelly", f"{quarter_kelly:.1%}")

    if edge <= 0:
        st.error("Your estimated probability does not beat the sportsbook's implied probability.")
    elif sim_odds <= -500:
        st.warning("The favorite may win often, but the payoff structure creates severe loss-recovery risk.")

    if st.button("Run Simulation", type="primary", use_container_width=True):
        rng = np.random.default_rng()
        paths = np.zeros((int(simulations), int(number_bets) + 1))
        paths[:, 0] = sim_bankroll
        ruined = np.zeros(int(simulations), dtype=bool)

        for run in range(int(simulations)):
            bank = sim_bankroll
            for n in range(1, int(number_bets) + 1):
                if staking == "Target-profit stake":
                    stake_n = stake_to_win(int(sim_odds), target_each)
                elif staking == "Flat % bankroll":
                    stake_n = bank * flat_pct
                else:
                    stake_n = bank * quarter_kelly
                stake_n = min(stake_n, bank)
                if stake_n <= 0:
                    paths[run, n:] = bank
                    ruined[run] = True
                    break
                if rng.random() < true_prob:
                    bank += potential_profit(int(sim_odds), stake_n)
                else:
                    bank -= stake_n
                paths[run, n] = bank
                if bank <= 0:
                    paths[run, n:] = 0
                    ruined[run] = True
                    break

        ending = paths[:, -1]
        median_path = np.median(paths, axis=0)
        p10_path = np.percentile(paths, 10, axis=0)
        p90_path = np.percentile(paths, 90, axis=0)
        probability_profit = float(np.mean(ending > sim_bankroll))
        probability_loss_half = float(np.mean(ending <= sim_bankroll * 0.5))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Median ending bankroll", money(float(np.median(ending))))
        m2.metric("Chance of profit", f"{probability_profit:.1%}")
        m3.metric("Chance of losing 50%+", f"{probability_loss_half:.1%}")
        m4.metric("5th percentile finish", money(float(np.percentile(ending, 5))))

        fig, ax = plt.subplots()
        x = np.arange(int(number_bets) + 1)
        ax.plot(x, median_path, label="Median")
        ax.fill_between(x, p10_path, p90_path, alpha=0.2, label="10th–90th percentile")
        ax.axhline(sim_bankroll, linewidth=1)
        ax.set_xlabel("Bet number")
        ax.set_ylabel("Bankroll ($)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tabs[5]:
    st.subheader("Performance Analysis")
    if settled.empty:
        st.write("Settle bets to populate performance analytics.")
    else:
        perf = settled.copy()
        perf["odds_band"] = pd.cut(
            perf["odds"],
            bins=[-10000, -500, -350, -250, -180, -110, 0, 10000],
            labels=["≤ -500", "-499 to -350", "-349 to -250", "-249 to -180", "-179 to -110", "Even/+ odds", "Other"],
            include_lowest=True,
        )
        p1, p2 = st.columns(2)
        with p1:
            by_sport = perf.groupby("sport").agg(
                bets=("id", "count"),
                staked=("stake", "sum"),
                profit=("result_profit", "sum"),
            )
            by_sport["roi"] = by_sport["profit"] / by_sport["staked"].replace(0, np.nan)
            st.markdown("#### By Sport")
            st.dataframe(by_sport.reset_index(), use_container_width=True, hide_index=True)
        with p2:
            by_band = perf.groupby("odds_band", observed=False).agg(
                bets=("id", "count"),
                staked=("stake", "sum"),
                profit=("result_profit", "sum"),
            )
            by_band["roi"] = by_band["profit"] / by_band["staked"].replace(0, np.nan)
            st.markdown("#### By Odds Range")
            st.dataframe(by_band.reset_index(), use_container_width=True, hide_index=True)

        ordered = perf.sort_values(["date", "id"]).copy()
        ordered["cumulative_profit"] = ordered["result_profit"].cumsum()
        ordered["bankroll_curve"] = st.session_state.bankroll + ordered["cumulative_profit"]
        fig, ax = plt.subplots()
        ax.plot(range(1, len(ordered) + 1), ordered["bankroll_curve"], marker="o")
        ax.axhline(st.session_state.bankroll, linewidth=1)
        ax.set_xlabel("Settled bet")
        ax.set_ylabel("Bankroll ($)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tabs[6]:
    st.subheader("Backup and Export")
    st.warning(
        "Streamlit Community Cloud may restart the app and clear temporary session data. "
        "Download your CSV after updates and upload it again when needed."
    )
    csv_data = bets.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download All Bets CSV",
        data=csv_data,
        file_name=f"michael_bets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("#### Current Data")
    st.dataframe(bets, use_container_width=True, hide_index=True)
