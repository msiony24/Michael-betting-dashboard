
import io
import math
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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
    st.caption("Price the matchup first. Then decide whether the current market offers enough value to bet.")

    sport_lab = st.selectbox("Choose sport", SPORTS, key="lab_sport")
    lab_date = st.date_input("Event date", value=date.today(), key="lab_date")

    if sport_lab == "Tennis":
        st.markdown("### Tennis Beta")

        top1, top2, top3 = st.columns(3)
        player_a = top1.text_input("Selection / favorite", placeholder="Etcheverry")
        player_b = top2.text_input("Opponent", placeholder="Rodionov")
        tournament = top3.text_input("Tournament", placeholder="ATP event")

        meta1, meta2, meta3, meta4 = st.columns(4)
        surface = meta1.selectbox("Surface", ["Hard", "Clay", "Grass", "Indoor Hard"])
        event_level = meta2.selectbox(
            "Event level",
            ["ATP 250", "ATP 500", "Masters 1000", "Grand Slam", "Davis Cup", "Other"],
        )
        round_name = meta3.selectbox(
            "Round",
            ["Qualifying", "R128", "R64", "R32", "R16", "Quarterfinal", "Semifinal", "Final"],
        )
        match_format = meta4.selectbox("Format", ["Best of 3", "Best of 5"])

        st.divider()
        st.markdown("### Market Price")

        market1, market2, market3 = st.columns(3)
        opening_odds = market1.number_input("Opening line", value=-180, step=5)
        current_odds = market2.number_input("Current Vegas line", value=-180, step=5)
        opponent_odds = market3.number_input("Opponent line", value=155, step=5)

        market_prob = implied_probability(int(current_odds))
        no_vig_prob = no_vig_probability(int(current_odds), int(opponent_odds))

        st.divider()
        st.markdown("### Statistical Base Projection")
        st.caption("Enter the probability Macabets assigns before matchup and situational context is applied.")

        base1, base2, base3 = st.columns(3)
        base_probability_pct = base1.slider(
            "Base win probability",
            min_value=1.0,
            max_value=99.0,
            value=float(round(no_vig_prob * 100, 1)),
            step=0.5,
        )
        model_confidence = base2.slider("Overall confidence", 1, 10, 7)
        data_quality = base3.slider("Data quality", 1, 10, 7)

        st.divider()
        st.markdown("### Context Engine")
        st.caption(
            "Score each category from -10 to +10 for the selected player. "
            "Positive values help the selection; negative values help the opponent."
        )

        context_definitions = [
            ("Matchup", "Serve, return, rally tolerance, weapons and exploitable weaknesses"),
            ("Recent Form", "Current level, opponent quality and whether results match the underlying play"),
            ("Surface & Conditions", "Surface fit, court speed, altitude, weather and indoor/outdoor conditions"),
            ("Fitness & Fatigue", "Injuries, workload, travel, recovery and scheduling"),
            ("Event & Pressure", "Tournament importance, round pressure and big-match experience"),
            ("Psychological", "Confidence, rivalry, crowd, composure and ability to close"),
        ]

        adjustments = []
        notes = []
        for idx, (factor_name, factor_help) in enumerate(context_definitions):
            with st.expander(factor_name, expanded=(idx < 2)):
                c1, c2 = st.columns([1, 1])
                score = c1.slider(
                    f"{factor_name} score",
                    min_value=-10,
                    max_value=10,
                    value=0,
                    key=f"ctx_score_{idx}",
                    help=factor_help,
                )
                factor_confidence = c2.slider(
                    f"{factor_name} confidence",
                    min_value=0,
                    max_value=100,
                    value=70,
                    step=5,
                    key=f"ctx_conf_{idx}",
                )
                explanation = st.text_area(
                    f"{factor_name} reasoning",
                    placeholder=factor_help,
                    key=f"ctx_note_{idx}",
                )
                adjustment = context_adjustment(score, factor_confidence)
                adjustments.append({
                    "Factor": factor_name,
                    "Score": score,
                    "Confidence": factor_confidence,
                    "Adjustment": adjustment,
                    "Reasoning": explanation,
                })
                notes.append(explanation.strip())

        total_adjustment = sum(item["Adjustment"] for item in adjustments)
        base_probability = base_probability_pct / 100
        final_probability = clamp_probability(base_probability + total_adjustment)
        fair_line = probability_to_american(final_probability)
        edge = final_probability - market_prob
        no_vig_edge = final_probability - no_vig_prob
        recommendation, recommendation_reason = recommendation_from_edge(
            edge, model_confidence, data_quality
        )

        fair_low = clamp_probability(final_probability - ((11 - data_quality) * 0.005))
        fair_high = clamp_probability(final_probability + ((11 - data_quality) * 0.005))
        fair_range_low = probability_to_american(fair_low)
        fair_range_high = probability_to_american(fair_high)

        preferred_entry_prob = clamp_probability(final_probability - 0.035)
        max_playable_prob = clamp_probability(final_probability - 0.015)
        preferred_entry = probability_to_american(preferred_entry_prob)
        max_playable = probability_to_american(max_playable_prob)

        st.divider()
        st.markdown("### Macabets Price")

        price1, price2, price3, price4 = st.columns(4)
        price1.metric("Vegas implied probability", f"{market_prob:.1%}")
        price2.metric("No-vig market probability", f"{no_vig_prob:.1%}")
        price3.metric("Macabets probability", f"{final_probability:.1%}", f"{total_adjustment:+.1%} context")
        price4.metric("Macabets fair line", f"{fair_line:+d}")

        price5, price6, price7, price8 = st.columns(4)
        price5.metric("Edge vs listed line", f"{edge:+.1%}")
        price6.metric("Edge vs no-vig market", f"{no_vig_edge:+.1%}")
        price7.metric("Preferred entry", f"{preferred_entry:+d}")
        price8.metric("Maximum playable", f"{max_playable:+d}")

        st.caption(
            f"Estimated fair-line range: {fair_range_low:+d} to {fair_range_high:+d}. "
            "The range widens when data quality is lower."
        )

        if recommendation == "GREEN LIGHT":
            st.success(f"GREEN LIGHT — {recommendation_reason}")
        elif recommendation == "LEAN":
            st.warning(f"LEAN — {recommendation_reason}")
        else:
            st.error(f"PASS — {recommendation_reason}")

        st.markdown(f"**Market assessment:** {line_value_label(edge)}")

        adjustment_table = pd.DataFrame(adjustments)
        adjustment_table["Probability impact"] = adjustment_table["Adjustment"].map(lambda x: f"{x:+.2%}")
        st.markdown("#### Context Breakdown")
        st.dataframe(
            adjustment_table[["Factor", "Score", "Confidence", "Probability impact", "Reasoning"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Macabets Explanation")
        active_factors = [item for item in adjustments if item["Score"] != 0]
        if not active_factors:
            st.info("No context adjustments have been entered. The final price currently equals the base projection.")
        else:
            strongest = sorted(active_factors, key=lambda x: abs(x["Adjustment"]), reverse=True)
            for item in strongest:
                direction = "helps" if item["Adjustment"] > 0 else "hurts"
                reason = item["Reasoning"] or "No written explanation entered."
                st.write(
                    f"**{item['Factor']} {direction} {player_a or 'the selection'} "
                    f"({item['Adjustment']:+.2%}):** {reason}"
                )

        st.divider()
        st.markdown("### Save Analysis to Session")
        analysis_notes = st.text_area(
            "Final thesis / risks",
            placeholder="What is the clearest path to victory, and how can the selection lose?",
            key="lab_final_thesis",
        )

        if "analyses" not in st.session_state:
            st.session_state.analyses = []

        if st.button("Save Matchup Analysis", type="primary", use_container_width=True):
            if not player_a.strip() or not player_b.strip():
                st.error("Enter both players before saving.")
            else:
                analysis_record = {
                    "date": lab_date.isoformat(),
                    "sport": sport_lab,
                    "event": tournament.strip(),
                    "selection": player_a.strip(),
                    "opponent": player_b.strip(),
                    "surface": surface,
                    "event_level": event_level,
                    "round": round_name,
                    "format": match_format,
                    "opening_line": int(opening_odds),
                    "current_line": int(current_odds),
                    "opponent_line": int(opponent_odds),
                    "market_probability": market_prob,
                    "no_vig_probability": no_vig_prob,
                    "base_probability": base_probability,
                    "context_adjustment": total_adjustment,
                    "macabets_probability": final_probability,
                    "fair_line": fair_line,
                    "edge": edge,
                    "confidence": model_confidence,
                    "data_quality": data_quality,
                    "recommendation": recommendation,
                    "preferred_entry": preferred_entry,
                    "maximum_playable": max_playable,
                    "thesis": analysis_notes.strip(),
                }
                st.session_state.analyses.append(analysis_record)
                st.success("Matchup analysis saved for this session.")

        if st.session_state.analyses:
            st.markdown("#### Saved Matchup Analyses")
            analyses_df = pd.DataFrame(st.session_state.analyses)
            display_cols = [
                "date", "sport", "selection", "opponent", "current_line",
                "macabets_probability", "fair_line", "edge",
                "recommendation", "confidence", "data_quality"
            ]
            analyses_view = analyses_df[display_cols].copy()
            analyses_view["macabets_probability"] = analyses_view["macabets_probability"].map(lambda x: f"{x:.1%}")
            analyses_view["edge"] = analyses_view["edge"].map(lambda x: f"{x:+.1%}")
            st.dataframe(analyses_view, use_container_width=True, hide_index=True)

            analyses_csv = analyses_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Matchup Analyses CSV",
                data=analyses_csv,
                file_name=f"macabets_matchup_analyses_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.info(
            "The full pricing engine is being introduced sport by sport. "
            "Tennis is the first Macabets beta because it is the current research priority."
        )

        if sport_lab in ["NFL", "College Football"]:
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Favorite team")
                st.text_area("Recent form")
                st.text_area("Offensive strengths / weaknesses")
                st.text_area("Defensive strengths / weaknesses")
                st.text_area("Injuries / availability")
            with c2:
                st.text_input("Opponent")
                st.text_area("Opponent recent form")
                st.text_area("Opponent offensive profile")
                st.text_area("Opponent defensive profile")
                st.text_area("Venue, travel, rest, weather, rivalry")
            st.text_area("Where does the favorite have the clearest matchup advantage?")
            st.text_area("How can the opponent realistically upset the favorite?")

        elif sport_lab == "NBA":
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Favorite team")
                st.text_area("Last 5–10 games")
                st.text_area("Offensive matchup")
                st.text_area("Defensive matchup")
                st.text_area("Injuries / minutes restrictions")
            with c2:
                st.text_input("Opponent")
                st.text_area("Opponent last 5–10 games")
                st.text_area("Pace and shot profile")
                st.text_area("Rest / back-to-back / travel")
                st.text_area("Rebounding and turnover matchup")
            st.text_area("Upset path and late-game risk")

        else:
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Favorite fighter")
                st.text_area("Recent form and quality of opposition")
                st.text_area("Power at this weight")
                st.text_area("Chin durability")
                st.text_area("Wrestling / grappling / clinch profile")
                st.text_area("Injuries, layoff, weight cut")
            with c2:
                st.text_input("Opponent fighter")
                st.text_area("Opponent recent form")
                st.text_area("Opponent power and finishing threat")
                st.text_area("Opponent chin and recovery")
                st.text_area("Opponent technical advantages")
                st.text_area("Age, mileage and camp changes")
            st.text_area("Favorite's clearest path to victory")
            st.text_area("Opponent's most realistic upset path")

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
