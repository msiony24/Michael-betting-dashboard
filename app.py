
import io
import math
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from engine.data import load_matches
    from engine.tennis import (
        analyze as analyze_tennis_match,
        player_names as tennis_player_names,
        tournament_names as tennis_tournament_names,
        tournament_surface as tennis_tournament_surface,
    )
    TENNIS_ENGINE_AVAILABLE = True
    TENNIS_ENGINE_IMPORT_ERROR = ""
except Exception as exc:
    TENNIS_ENGINE_AVAILABLE = False
    TENNIS_ENGINE_IMPORT_ERROR = str(exc)

APP_VERSION = "Macabets Tennis v0.8"
BUILD_DATE = "July 22, 2026"

st.set_page_config(
    page_title="Michael Betting Dashboard",
    page_icon="📊",
    layout="wide",
)

SPORTS = ["NFL", "College Football", "NBA", "Tennis", "UFC", "Boxing"]
STATUSES = ["Pending", "Won", "Lost", "Void", "Cashed Out"]
BET_TYPES = ["Moneyline", "Spread", "Total", "Prop", "Parlay", "Live"]
ANALYSIS_COLUMNS = [
    "analysis_id", "created_at", "match_date", "tournament", "surface", "round",
    "player_a", "player_b", "market_odds_a", "model_probability_a", "fair_odds_a",
    "market_odds_b", "no_vig_probability_a", "no_vig_edge", "decision",
    "minimum_acceptable_odds_a", "estimated_roi", "confidence", "prediction",
    "upset_path", "biggest_risk", "assumptions", "notes", "result",
    "closing_odds_a", "prediction_correct",
    "closing_line_value", "review", "lesson"
]

DEFAULT_COLUMNS = [
    "id", "date", "sport", "event", "selection", "bet_type", "odds",
    "stake", "target_profit", "status", "result_profit", "book",
    "confidence", "notes"
]


SLATE_COLUMNS = [
    "slate_id", "match_date", "tournament", "surface", "round",
    "player_a", "player_b", "market_odds_a", "market_odds_b",
    "model_probability_a", "confidence", "notes"
]


def money(value):
    return f"${value:,.2f}"


def american_to_decimal(odds):
    if odds == 0:
        return 1.0
    return 1 + (100 / abs(odds) if odds < 0 else odds / 100)


def probability_to_american(probability):
    """Convert a win probability (0-1) to fair American odds."""
    probability = min(max(float(probability), 0.0001), 0.9999)
    if probability >= 0.5:
        return -round(100 * probability / (1 - probability))
    return round(100 * (1 - probability) / probability)


def format_american(odds):
    odds = int(round(odds))
    return f"+{odds}" if odds > 0 else str(odds)


def fair_line_probability(scores_favorite, scores_opponent, weights, confidence):
    """Create a first-pass fair probability from a weighted matchup scorecard.

    The confidence input shrinks uncertain estimates toward 50%, preventing
    low-information matchups from producing extreme prices.
    """
    weighted_difference = sum(
        weights[key] * (scores_favorite[key] - scores_opponent[key])
        for key in weights
    )
    raw_probability = 1 / (1 + math.exp(-0.45 * weighted_difference))
    confidence_factor = min(max(confidence / 10, 0.1), 1.0)
    adjusted_probability = 0.5 + (raw_probability - 0.5) * confidence_factor
    return min(max(adjusted_probability, 0.02), 0.98), weighted_difference


def implied_probability(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    if odds > 0:
        return 100 / (odds + 100)
    return 0.0


def no_vig_probabilities(odds_a, odds_b):
    """Remove sportsbook margin from a two-sided moneyline market."""
    raw_a = implied_probability(int(odds_a))
    raw_b = implied_probability(int(odds_b))
    total = raw_a + raw_b
    if total <= 0:
        return 0.5, 0.5, 0.0
    return raw_a / total, raw_b / total, total - 1


def minimum_acceptable_odds(model_probability, required_roi=0.02):
    """Worst American price that still preserves the required expected ROI."""
    probability = min(max(float(model_probability), 0.0001), 0.9999)
    required_decimal = (1 + required_roi) / probability
    if required_decimal <= 1:
        return -10000
    if required_decimal >= 2:
        return round((required_decimal - 1) * 100)
    return -round(100 / (required_decimal - 1))


def decision_label(expected_roi, confidence):
    if expected_roi >= 0.05 and confidence >= 7:
        return "BET", "The estimated edge is meaningful and supported by sufficient confidence."
    if expected_roi >= 0.02 and confidence >= 6:
        return "WATCH", "There may be value, but the edge or confidence is not strong enough yet."
    return "PASS", "The current price does not provide enough model-supported value."


def build_matchup_brief(player_a, player_b, scores_a, scores_b, weights):
    contributions = []
    for factor, weight in weights.items():
        difference = scores_a[factor] - scores_b[factor]
        contributions.append((factor, difference * weight, difference))
    contributions.sort(key=lambda item: item[1], reverse=True)

    strengths = [item for item in contributions if item[1] > 0][:3]
    risks = sorted([item for item in contributions if item[1] < 0], key=lambda item: item[1])[:2]

    strength_text = (
        "; ".join(f"{factor} ({raw_diff:+.1f})" for factor, _, raw_diff in strengths)
        if strengths else "no clear category-level advantage"
    )
    risk_text = (
        "; ".join(f"{factor} ({raw_diff:+.1f})" for factor, _, raw_diff in risks)
        if risks else "no major scorecard disadvantage"
    )

    return (
        f"{player_a} grades best in {strength_text}. "
        f"The clearest concerns relative to {player_b} are {risk_text}. "
        "This summary reflects the current pre-match scorecard and should be revised if injury, "
        "weather, scheduling, or market information changes."
    )


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


def empty_analyses():
    return pd.DataFrame(columns=ANALYSIS_COLUMNS)


def normalize_analyses(df):
    """Keep old archive exports compatible as the archive gains new fields."""
    clean = df.copy()
    for col in ANALYSIS_COLUMNS:
        if col not in clean.columns:
            clean[col] = ""
    clean = clean[ANALYSIS_COLUMNS]

    numeric = [
        "analysis_id", "market_odds_a", "market_odds_b", "model_probability_a",
        "fair_odds_a", "no_vig_probability_a", "no_vig_edge",
        "minimum_acceptable_odds_a", "estimated_roi", "confidence",
        "closing_odds_a", "closing_line_value"
    ]
    for col in numeric:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean["analysis_id"] = clean["analysis_id"].fillna(0).astype(int)
    clean["result"] = clean["result"].replace("", "Pending").fillna("Pending")
    clean["prediction_correct"] = clean["prediction_correct"].fillna("")
    return clean


def closing_line_value(model_probability, closing_odds):
    """Expected ROI at the closing price; positive means the model still beat close."""
    if pd.isna(closing_odds) or float(closing_odds) == 0:
        return float("nan")
    decimal = american_to_decimal(int(closing_odds))
    probability = float(model_probability)
    return probability * (decimal - 1) - (1 - probability)


def empty_slate():
    return pd.DataFrame(columns=SLATE_COLUMNS)


def normalize_slate(df):
    clean = df.copy()
    for col in SLATE_COLUMNS:
        if col not in clean.columns:
            clean[col] = ""
    clean = clean[SLATE_COLUMNS]

    numeric = [
        "slate_id", "market_odds_a", "market_odds_b",
        "model_probability_a", "confidence"
    ]
    for col in numeric:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean["slate_id"] = clean["slate_id"].fillna(0).astype(int)
    clean["model_probability_a"] = clean["model_probability_a"].fillna(0.5).clip(0.01, 0.99)
    clean["confidence"] = clean["confidence"].fillna(5).clip(1, 10)
    return clean


def score_daily_slate(df):
    scored = normalize_slate(df)
    if scored.empty:
        return scored

    market_a = scored["market_odds_a"].apply(
        lambda value: implied_probability(int(value)) if pd.notna(value) and value != 0 else np.nan
    )
    market_b = scored["market_odds_b"].apply(
        lambda value: implied_probability(int(value)) if pd.notna(value) and value != 0 else np.nan
    )
    totals = market_a + market_b

    scored["no_vig_probability_a"] = np.where(
        totals > 0, market_a / totals, market_a
    )
    scored["sportsbook_hold"] = totals - 1
    scored["fair_odds_a"] = scored["model_probability_a"].apply(probability_to_american)
    scored["no_vig_edge"] = scored["model_probability_a"] - scored["no_vig_probability_a"]
    scored["estimated_roi"] = scored.apply(
        lambda row: (
            row["model_probability_a"] * (american_to_decimal(int(row["market_odds_a"])) - 1)
            - (1 - row["model_probability_a"])
        )
        if pd.notna(row["market_odds_a"]) and row["market_odds_a"] != 0
        else np.nan,
        axis=1,
    )
    scored["decision"] = scored.apply(
        lambda row: decision_label(
            float(row["estimated_roi"]) if pd.notna(row["estimated_roi"]) else -1,
            int(row["confidence"]),
        )[0],
        axis=1,
    )
    scored["minimum_acceptable_odds_a"] = scored["model_probability_a"].apply(
        lambda probability: minimum_acceptable_odds(probability, required_roi=0.02)
    )

    # A transparent ranking score: edge is primary, confidence and market disagreement are secondary.
    scored["opportunity_score"] = (
        scored["estimated_roi"].fillna(-1) * 100 * 0.60
        + scored["confidence"] * 0.25
        + scored["no_vig_edge"].fillna(0) * 100 * 0.15
    )
    return scored.sort_values(
        ["opportunity_score", "confidence"],
        ascending=[False, False],
    ).reset_index(drop=True)


if "pending_fair_line_prefill" in st.session_state:
    pending = st.session_state.pop("pending_fair_line_prefill")
    for key, value in pending.items():
        st.session_state[key] = value


if "bets" not in st.session_state:
    st.session_state.bets = empty_bets()

if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100000.0

if "target_profit" not in st.session_state:
    st.session_state.target_profit = 10000.0

if "analyses" not in st.session_state:
    st.session_state.analyses = empty_analyses()


if "daily_slate" not in st.session_state:
    st.session_state.daily_slate = empty_slate()


st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.small-note {color: #777; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

title_col, version_col = st.columns([4, 1])
with title_col:
    st.title("Michael Betting Dashboard")
    st.caption("Favorite-focused bet tracking, matchup analysis and bankroll risk control.")
with version_col:
    st.markdown(
        f"""
        <div style="
            margin-top: 0.65rem;
            padding: 0.55rem 0.75rem;
            border: 1px solid #d8d8d8;
            border-radius: 0.55rem;
            text-align: center;
            font-weight: 600;
        ">
            {APP_VERSION}<br>
            <span style="font-size: 0.78rem; font-weight: 400;">{BUILD_DATE}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    uploaded = st.file_uploader("Upload a prior bets CSV", type=["csv"], key="bets_restore")
    if uploaded is not None:
        try:
            imported = normalize_bets(pd.read_csv(uploaded))
            st.session_state.bets = imported
            st.success(f"Loaded {len(imported)} bets.")
        except Exception as exc:
            st.error(f"Could not load bets CSV: {exc}")

    analysis_upload = st.file_uploader(
        "Upload a prior analysis archive CSV",
        type=["csv"],
        key="analysis_restore",
    )
    if analysis_upload is not None:
        try:
            imported_analyses = normalize_analyses(pd.read_csv(analysis_upload))
            st.session_state.analyses = imported_analyses
            st.success(f"Loaded {len(imported_analyses)} archived analyses.")
        except Exception as exc:
            st.error(f"Could not load analysis CSV: {exc}")

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
    "Dashboard", "New Bet", "Bet Ledger", "Fair Line Engine",
    "Analysis Archive", "Matchup Lab", "Outcome Simulator", "Performance",
    "Backup", "Daily Slate"
])

with tabs[0]:
    with st.expander("What's New in Macabets Tennis v0.8", expanded=True):
        st.markdown(
            """
            - Direct explanation of why a considered bet grades BET, WATCH, or PASS
            - Separates player strength from betting-price value
            - Shows the strongest factors supporting and opposing your selected side
            - Identifies “good player, bad price” situations
            - Adds a concise Macabets verdict before the detailed model table
            """
        )

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
    st.subheader("Automatic Match Analyzer — Tennis v4")
    st.caption(
        "Select the matchup and event context. Macabets builds the probability from "
        "historical ATP results, Elo, surface performance, form, serve/return data, "
        "fatigue, rest, and event-pressure history."
    )

    if not TENNIS_ENGINE_AVAILABLE:
        st.error(
            "The tennis engine could not be imported. Confirm that engine/data.py and "
            f"engine/tennis.py are in the repository. Import error: {TENNIS_ENGINE_IMPORT_ERROR}"
        )
    else:
        try:
            matches, data_errors = load_matches()
        except Exception as exc:
            matches = pd.DataFrame()
            data_errors = []
            st.error(str(exc))
            st.info(
                "Run update_tennis_data.py or the GitHub Action named "
                "'Update Macabets Tennis Data', then reboot the Streamlit app."
            )

        if not matches.empty:
            if data_errors:
                with st.expander("Data-file warnings"):
                    for item in data_errors:
                        st.warning(item)

            players = tennis_player_names(matches)
            tournaments = tennis_tournament_names(matches)

            existing_a = str(st.session_state.get("fle_favorite", "")).strip()
            existing_b = str(st.session_state.get("fle_opponent", "")).strip()
            existing_tournament = str(st.session_state.get("fle_tournament", "")).strip()

            player_a_options = players.copy()
            player_b_options = players.copy()
            if existing_a and existing_a not in player_a_options:
                player_a_options.insert(0, existing_a)
            if existing_b and existing_b not in player_b_options:
                player_b_options.insert(0, existing_b)
            if not player_a_options:
                player_a_options = ["Player A"]
            if not player_b_options:
                player_b_options = ["Player B"]

            tournament_options = tournaments.copy()
            if existing_tournament and existing_tournament not in tournament_options:
                tournament_options.insert(0, existing_tournament)
            if not tournament_options:
                tournament_options = ["Montreal"]

            meta1, meta2, meta3, meta4 = st.columns(4)
            match_date = meta1.date_input("Match date", value=date.today(), key="auto_match_date")
            tournament = meta2.selectbox(
                "Tournament",
                tournament_options,
                index=(
                    tournament_options.index(existing_tournament)
                    if existing_tournament in tournament_options else 0
                ),
                key="auto_tournament",
            )

            detected_surface = tennis_tournament_surface(matches, tournament)
            surface_options = ["Hard", "Clay", "Grass", "Carpet"]
            surface = meta3.selectbox(
                "Surface",
                surface_options,
                index=surface_options.index(detected_surface) if detected_surface in surface_options else 0,
                key="auto_surface",
            )
            round_name = meta4.selectbox(
                "Round",
                ["Qualifying", "R128", "R64", "R32", "R16", "Quarterfinal", "Semifinal", "Final"],
                key="auto_round",
            )

            p1, p2 = st.columns(2)
            default_a_index = player_a_options.index(existing_a) if existing_a in player_a_options else 0
            default_b_index = player_b_options.index(existing_b) if existing_b in player_b_options else min(1, len(player_b_options) - 1)
            player_a = p1.selectbox(
                "Player A",
                player_a_options,
                index=default_a_index,
                key="auto_player_a",
            )
            player_b = p2.selectbox(
                "Player B",
                player_b_options,
                index=default_b_index,
                key="auto_player_b",
            )

            bet_side_options = [
                "Just analyze",
                f"{player_a} — Player A",
                f"{player_b} — Player B",
            ]
            considering_bet = st.radio(
                "Who are you considering betting on?",
                bet_side_options,
                horizontal=True,
                key="auto_considering_bet",
                help=(
                    "This does not influence the model. It only tells Macabets which "
                    "market position you want evaluated."
                ),
            )

            o1, o2, o3 = st.columns(3)
            market_odds_a = o1.number_input(
                f"Sportsbook odds — {player_a}",
                value=int(st.session_state.get("fle_market_a", -180)),
                step=5,
                key="auto_market_a",
            )
            market_odds_b = o2.number_input(
                f"Sportsbook odds — {player_b}",
                value=int(st.session_state.get("fle_market_b", 155)),
                step=5,
                key="auto_market_b",
            )
            simulations = o3.selectbox(
                "Simulations",
                [5000, 10000, 20000, 50000],
                index=2,
                key="auto_simulations",
            )

            analyze_disabled = player_a == player_b
            if analyze_disabled:
                st.warning("Select two different players.")

            if st.button(
                "Analyze Match",
                type="primary",
                use_container_width=True,
                disabled=analyze_disabled,
            ):
                with st.spinner("Macabets is analyzing the matchup..."):
                    try:
                        st.session_state.automatic_match_result = analyze_tennis_match(
                            matches=matches,
                            player_a=player_a,
                            player_b=player_b,
                            tournament=tournament,
                            round_label=round_name,
                            surface=surface,
                            event_date=match_date,
                            simulations=int(simulations),
                        )
                        st.session_state.automatic_match_market = {
                            "market_odds_a": int(market_odds_a),
                            "market_odds_b": int(market_odds_b),
                            "match_date": match_date.isoformat(),
                            "considering_bet": considering_bet,
                        }
                    except Exception as exc:
                        st.session_state.pop("automatic_match_result", None)
                        st.error(f"Analysis failed: {exc}")

            result = st.session_state.get("automatic_match_result")
            market_snapshot = st.session_state.get("automatic_match_market", {})

            if result:
                analyzed_a = result["player_a"]
                analyzed_b = result["player_b"]
                listed_a = int(market_snapshot.get("market_odds_a", market_odds_a))
                listed_b = int(market_snapshot.get("market_odds_b", market_odds_b))

                model_probability = float(result["win_probability"])
                probability_b = 1 - model_probability
                fair_odds = int(result["fair_line"])
                fair_odds_b = probability_to_american(probability_b)
                no_vig_a, no_vig_b, sportsbook_hold = no_vig_probabilities(listed_a, listed_b)

                roi_a = (
                    model_probability * (american_to_decimal(listed_a) - 1)
                    - (1 - model_probability)
                )
                roi_b = (
                    probability_b * (american_to_decimal(listed_b) - 1)
                    - (1 - probability_b)
                )
                edge_a = model_probability - no_vig_a
                edge_b = probability_b - no_vig_b
                confidence = int(result["confidence"])

                considered_snapshot = str(
                    market_snapshot.get("considering_bet", "Just analyze")
                )
                if "Player A" in considered_snapshot:
                    considered_player = analyzed_a
                    considered_probability = model_probability
                    considered_fair_odds = fair_odds
                    considered_market_odds = listed_a
                    considered_no_vig = no_vig_a
                    considered_edge = edge_a
                    considered_roi = roi_a
                    opposite_player = analyzed_b
                    opposite_roi = roi_b
                elif "Player B" in considered_snapshot:
                    considered_player = analyzed_b
                    considered_probability = probability_b
                    considered_fair_odds = fair_odds_b
                    considered_market_odds = listed_b
                    considered_no_vig = no_vig_b
                    considered_edge = edge_b
                    considered_roi = roi_b
                    opposite_player = analyzed_a
                    opposite_roi = roi_a
                else:
                    considered_player = None
                    considered_probability = None
                    considered_fair_odds = None
                    considered_market_odds = None
                    considered_no_vig = None
                    considered_edge = None
                    considered_roi = None
                    opposite_player = None
                    opposite_roi = None

                minimum_price_a = minimum_acceptable_odds(
                    model_probability, required_roi=0.02
                )
                minimum_price_b = minimum_acceptable_odds(
                    probability_b, required_roi=0.02
                )
                if considered_player == analyzed_a:
                    minimum_price = minimum_price_a
                elif considered_player == analyzed_b:
                    minimum_price = minimum_price_b
                else:
                    minimum_price = None

                if considered_player:
                    decision, decision_reason = decision_label(
                        considered_roi, confidence
                    )
                else:
                    decision = "ANALYZE"
                    decision_reason = (
                        "No betting side was selected. Macabets is showing the matchup "
                        "objectively."
                    )

                # Preserve Player A fields for the existing archive structure.
                no_vig_edge = edge_a
                expected_roi = roi_a

                st.divider()
                st.markdown(f"### {analyzed_a} vs {analyzed_b}")

                st.markdown("#### Objective Match Price")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    f"{analyzed_a} probability",
                    f"{model_probability:.1%}",
                    f"Fair {format_american(fair_odds)}",
                )
                m2.metric(
                    f"{analyzed_b} probability",
                    f"{probability_b:.1%}",
                    f"Fair {format_american(fair_odds_b)}",
                )
                m3.metric("Sportsbook hold", f"{sportsbook_hold:.1%}")
                m4.metric("Model confidence", f"{confidence}/10")

                if considered_player:
                    st.markdown(f"#### Your Considered Bet: {considered_player}")
                    n1, n2, n3, n4 = st.columns(4)
                    n1.metric(
                        "Your market price",
                        format_american(considered_market_odds),
                    )
                    n2.metric(
                        "Macabets fair price",
                        format_american(considered_fair_odds),
                    )
                    n3.metric(
                        "Edge vs no-vig market",
                        f"{considered_edge:+.1%}",
                    )
                    n4.metric(
                        "Expected ROI",
                        f"{considered_roi:+.1%}",
                    )

                    d1, d2, d3 = st.columns(3)
                    d1.metric("Decision", decision)
                    d2.metric(
                        "Minimum acceptable price",
                        format_american(minimum_price),
                    )
                    d3.metric("Data quality", f"{int(result['data_quality'])}/10")

                    if decision == "BET":
                        st.success(
                            f"BET: Macabets prices {considered_player} at "
                            f"{format_american(considered_fair_odds)} versus your available "
                            f"price of {format_american(considered_market_odds)}. "
                            f"Estimated ROI is {considered_roi:+.1%}."
                        )
                    elif decision == "WATCH":
                        st.warning(
                            f"WATCH: {decision_reason} Macabets needs approximately "
                            f"{format_american(minimum_price)} or better for a 2% expected return."
                        )
                    else:
                        st.error(
                            f"PASS: {format_american(considered_market_odds)} is too expensive "
                            f"relative to Macabets' fair price of "
                            f"{format_american(considered_fair_odds)}. "
                            f"A price near {format_american(minimum_price)} or better is required."
                        )

                    if opposite_roi > considered_roi:
                        if opposite_roi > 0:
                            st.info(
                                f"Macabets currently sees more value on {opposite_player} "
                                f"({opposite_roi:+.1%} estimated ROI) than on your considered side."
                            )
                        else:
                            st.caption(
                                f"The opposite side grades better than your considered bet, "
                                f"but it still does not show positive expected value "
                                f"({opposite_roi:+.1%})."
                            )
                else:
                    st.info(
                        "No betting side selected. Macabets has priced both players objectively. "
                        "Select a player above and analyze again to receive a direct BET / WATCH / "
                        "PASS evaluation."
                    )
                    n1, n2, n3, n4 = st.columns(4)
                    n1.metric(f"{analyzed_a} ROI", f"{roi_a:+.1%}")
                    n2.metric(f"{analyzed_b} ROI", f"{roi_b:+.1%}")
                    n3.metric(f"{analyzed_a} no-vig edge", f"{edge_a:+.1%}")
                    n4.metric(f"{analyzed_b} no-vig edge", f"{edge_b:+.1%}")

                # Build a decision-focused explanation from the same neutral model factors.
                raw_factors = []
                for factor in result["factors"]:
                    raw_factors.append({
                        "name": str(factor["name"]),
                        "impact_a": float(factor["impact"]),
                        "reason": str(factor["reason"]),
                    })

                if considered_player:
                    considered_is_a = considered_player == analyzed_a
                    considered_factor_rows = []
                    for factor in raw_factors:
                        side_impact = (
                            factor["impact_a"]
                            if considered_is_a
                            else -factor["impact_a"]
                        )
                        considered_factor_rows.append({
                            "name": factor["name"],
                            "impact": side_impact,
                            "reason": factor["reason"],
                        })

                    support_factors = sorted(
                        [f for f in considered_factor_rows if f["impact"] > 0],
                        key=lambda item: item["impact"],
                        reverse=True,
                    )[:3]
                    opposition_factors = sorted(
                        [f for f in considered_factor_rows if f["impact"] < 0],
                        key=lambda item: item["impact"],
                    )[:3]

                    model_favors_considered = considered_probability > 0.50
                    market_favors_considered = considered_no_vig > 0.50
                    has_positive_value = considered_roi > 0
                    meaningful_value = considered_roi >= 0.05
                    price_gap = considered_probability - considered_no_vig

                    if model_favors_considered and not has_positive_value:
                        verdict_type = "Good player, bad price"
                        verdict_text = (
                            f"Macabets expects {considered_player} to win more often than lose, "
                            f"but {format_american(considered_market_odds)} is too expensive. "
                            f"The player and the bet are not the same decision."
                        )
                    elif not model_favors_considered and has_positive_value:
                        verdict_type = "Underdog value"
                        verdict_text = (
                            f"Macabets does not make {considered_player} the most likely winner, "
                            f"but the offered price is large enough to create positive expected value."
                        )
                    elif model_favors_considered and meaningful_value:
                        verdict_type = "Player and price align"
                        verdict_text = (
                            f"Macabets favors {considered_player} in the matchup and also believes "
                            f"your price is better than the model's fair price."
                        )
                    elif has_positive_value:
                        verdict_type = "Small price advantage"
                        verdict_text = (
                            f"The offered price is slightly better than Macabets' fair value, "
                            f"but the margin is not yet strong enough for a full BET grade."
                        )
                    else:
                        verdict_type = "No betting advantage"
                        verdict_text = (
                            f"Macabets does not see enough compensation at "
                            f"{format_american(considered_market_odds)} for the matchup risk."
                        )

                    st.markdown("#### Why Macabets Gave This Decision")
                    v1, v2, v3 = st.columns(3)
                    v1.metric(
                        "Player outlook",
                        "Favored" if model_favors_considered else "Underdog",
                        f"{considered_probability:.1%} win probability",
                    )
                    v2.metric(
                        "Price outlook",
                        "Positive value" if has_positive_value else "Negative value",
                        f"{price_gap:+.1%} vs no-vig market",
                    )
                    v3.metric("Bet diagnosis", verdict_type)

                    if decision == "BET":
                        st.success(verdict_text)
                    elif decision == "WATCH":
                        st.warning(verdict_text)
                    else:
                        st.error(verdict_text)

                    reason_col_a, reason_col_b = st.columns(2)
                    with reason_col_a:
                        st.markdown(f"**What supports {considered_player}**")
                        if support_factors:
                            for factor in support_factors:
                                st.markdown(
                                    f"- **{factor['name']}** "
                                    f"({factor['impact']:+.1%}): {factor['reason']}"
                                )
                        else:
                            st.caption(
                                "The current model does not identify a meaningful statistical "
                                "factor supporting this side."
                            )

                    with reason_col_b:
                        st.markdown(f"**What works against {considered_player}**")
                        if opposition_factors:
                            for factor in opposition_factors:
                                st.markdown(
                                    f"- **{factor['name']}** "
                                    f"({factor['impact']:+.1%}): {factor['reason']}"
                                )
                        else:
                            st.caption(
                                "The current model does not identify a meaningful statistical "
                                "factor working against this side."
                            )

                    if confidence < 6:
                        st.warning(
                            "Model confidence is limited. Treat the fair line as less stable "
                            "until the data sample or matchup context improves."
                        )
                    elif int(result["data_quality"]) < 6:
                        st.warning(
                            "The recommendation is being made with limited data quality. "
                            "The calculated edge may be less reliable than the headline number."
                        )

                st.markdown("#### Model Foundation")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric(
                    f"{analyzed_a} overall Elo",
                    f"{result['overall_elo'][0]:.0f}",
                )
                e2.metric(
                    f"{analyzed_b} overall Elo",
                    f"{result['overall_elo'][1]:.0f}",
                )
                e3.metric(
                    f"{analyzed_a} surface Elo",
                    f"{result['surface_elo'][0]:.0f}",
                )
                e4.metric(
                    f"{analyzed_b} surface Elo",
                    f"{result['surface_elo'][1]:.0f}",
                )

                factor_rows = []
                for factor in raw_factors:
                    impact = factor["impact_a"]
                    factor_rows.append({
                        "Factor": factor["name"],
                        "Probability impact": impact,
                        "Direction": analyzed_a if impact > 0 else analyzed_b if impact < 0 else "Neutral",
                        "Explanation": factor["reason"],
                    })
                factor_df = pd.DataFrame(factor_rows).sort_values(
                    "Probability impact",
                    ascending=False,
                )
                st.markdown("#### Why Macabets Made This Line")
                st.dataframe(
                    factor_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Probability impact": st.column_config.NumberColumn(format="%+.1%%")
                    },
                )

                profile_a = result["profile_a"]
                profile_b = result["profile_b"]
                profile_df = pd.DataFrame([
                    {
                        "Player": analyzed_a,
                        "Rank": profile_a["rank"],
                        "Last-10 win rate": profile_a["recent_win"],
                        f"{result['surface']} win rate": profile_a["surface_win"],
                        "Serve points won": profile_a["serve_points_won"],
                        "Return points won": profile_a["return_points_won"],
                        "Matches — 7 days": profile_a["matches_7"],
                        "Matches — 14 days": profile_a["matches_14"],
                        "Rest days": profile_a["rest_days"],
                        "Historical sample": profile_a["sample"],
                    },
                    {
                        "Player": analyzed_b,
                        "Rank": profile_b["rank"],
                        "Last-10 win rate": profile_b["recent_win"],
                        f"{result['surface']} win rate": profile_b["surface_win"],
                        "Serve points won": profile_b["serve_points_won"],
                        "Return points won": profile_b["return_points_won"],
                        "Matches — 7 days": profile_b["matches_7"],
                        "Matches — 14 days": profile_b["matches_14"],
                        "Rest days": profile_b["rest_days"],
                        "Historical sample": profile_b["sample"],
                    },
                ])
                st.markdown("#### Player Profiles")
                st.dataframe(
                    profile_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Last-10 win rate": st.column_config.NumberColumn(format="%.1%%"),
                        f"{result['surface']} win rate": st.column_config.NumberColumn(format="%.1%%"),
                        "Serve points won": st.column_config.NumberColumn(format="%.1%%"),
                        "Return points won": st.column_config.NumberColumn(format="%.1%%"),
                    },
                )

                simulation = result["simulation"]
                st.markdown("#### Outcome Simulation")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric(f"{analyzed_a} wins", f"{simulation['win_probability']:.1%}")
                s2.metric(f"{analyzed_a} straight sets", f"{simulation['straight_sets_a']:.1%}")
                s3.metric(f"{analyzed_b} straight sets", f"{simulation['straight_sets_b']:.1%}")
                s4.metric("Deciding set", f"{simulation['deciding_set']:.1%}")

                score_df = pd.DataFrame(
                    [
                        {"Set score": score, "Probability": probability}
                        for score, probability in simulation["set_scores"].items()
                    ]
                ).sort_values("Probability", ascending=False)
                st.dataframe(
                    score_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Probability": st.column_config.NumberColumn(format="%.1%%")
                    },
                )

                st.markdown("#### Pre-Match Decision Record")
                d1, d2 = st.columns(2)
                prediction = d1.text_area(
                    "Why Player A wins",
                    value=(
                        f"Macabets gives {analyzed_a} a {model_probability:.1%} win probability, "
                        f"with a fair line of {format_american(fair_odds)}."
                    ),
                    key="auto_prediction",
                )
                upset_path = d2.text_area(
                    "Why Player B wins",
                    key="auto_upset_path",
                )
                d3, d4 = st.columns(2)
                biggest_risk = d3.text_area("Biggest risk", key="auto_biggest_risk")
                assumptions = d4.text_area("Key assumptions", key="auto_assumptions")
                analysis_notes = st.text_area("Additional notes", key="auto_analysis_notes")

                if st.button(
                    "Save Automatic Analysis",
                    type="primary",
                    use_container_width=True,
                ):
                    analyses = st.session_state.analyses.copy()
                    next_analysis_id = int(analyses["analysis_id"].max()) + 1 if not analyses.empty else 1
                    row = {
                        "analysis_id": next_analysis_id,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "match_date": str(market_snapshot.get("match_date", date.today().isoformat())),
                        "tournament": result["tournament"],
                        "surface": result["surface"],
                        "round": result["round"],
                        "player_a": analyzed_a,
                        "player_b": analyzed_b,
                        "market_odds_a": listed_a,
                        "market_odds_b": listed_b,
                        "model_probability_a": model_probability,
                        "fair_odds_a": fair_odds,
                        "no_vig_probability_a": no_vig_a,
                        "no_vig_edge": no_vig_edge,
                        "decision": decision,
                        "minimum_acceptable_odds_a": minimum_price_a,
                        "estimated_roi": expected_roi,
                        "confidence": confidence,
                        "prediction": prediction.strip(),
                        "upset_path": upset_path.strip(),
                        "biggest_risk": biggest_risk.strip(),
                        "assumptions": assumptions.strip(),
                        "notes": (
                            (
                                f"Considering bet: {considered_player} at "
                                f"{format_american(considered_market_odds)}. "
                                f"Side-specific decision: {decision}. "
                                f"Side-specific estimated ROI: {considered_roi:+.1%}. "
                                if considered_player else
                                "No betting side selected. "
                            )
                            + analysis_notes.strip()
                        ).strip(),
                        "result": "Pending",
                        "closing_odds_a": np.nan,
                        "prediction_correct": "",
                        "closing_line_value": np.nan,
                        "review": "",
                        "lesson": "",
                    }
                    st.session_state.analyses = normalize_analyses(
                        pd.concat([analyses, pd.DataFrame([row])], ignore_index=True)
                    )
                    st.success("Automatic analysis saved to the archive.")

                st.caption(
                    f"Model base probability: {result['base_probability']:.1%}. "
                    f"Final pre-simulation model: {result['model_probability']:.1%}. "
                    f"Simulation count: {simulation['simulations']:,}."
                )

with tabs[4]:
    st.subheader("Analysis Archive")
    analyses = normalize_analyses(st.session_state.analyses)

    if analyses.empty:
        st.write("No pre-match analyses saved yet.")
    else:
        completed = analyses[analyses["result"].isin(["Player A Won", "Player B Won"])]
        completed_count = len(completed)
        correct_count = int((completed["prediction_correct"].astype(str) == "Yes").sum())
        accuracy = correct_count / completed_count if completed_count else 0.0
        avg_roi = pd.to_numeric(analyses["estimated_roi"], errors="coerce").mean()
        avg_clv = pd.to_numeric(completed["closing_line_value"], errors="coerce").mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Analyses Logged", len(analyses))
        k2.metric("Completed Reviews", completed_count)
        k3.metric("Prediction Accuracy", f"{accuracy:.1%}" if completed_count else "—")
        k4.metric("Average Closing Edge", f"{avg_clv:.1%}" if pd.notna(avg_clv) else "—")

        display_cols = [
            "analysis_id", "match_date", "tournament", "round", "player_a", "player_b",
            "market_odds_a", "fair_odds_a", "model_probability_a", "no_vig_edge",
            "decision", "estimated_roi", "confidence", "result", "prediction_correct"
        ]
        archive_view = analyses[display_cols].sort_values("analysis_id", ascending=False)
        st.dataframe(archive_view, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Post-Match Review")
        selected_analysis_id = st.selectbox(
            "Analysis ID",
            analyses["analysis_id"].astype(int).sort_values(ascending=False).tolist(),
            format_func=lambda x: (
                f"#{x} — "
                f"{analyses.loc[analyses['analysis_id'] == x, 'player_a'].iloc[0]} vs "
                f"{analyses.loc[analyses['analysis_id'] == x, 'player_b'].iloc[0]}"
            ),
        )
        selected_analysis = analyses[analyses["analysis_id"] == selected_analysis_id].iloc[0]

        st.markdown(
            f"**Original call:** {selected_analysis['player_a']} "
            f"{format_american(selected_analysis['market_odds_a']) if pd.notna(selected_analysis['market_odds_a']) else ''}  |  "
            f"Macabets {selected_analysis['model_probability_a']:.1%} "
            f"({format_american(selected_analysis['fair_odds_a']) if pd.notna(selected_analysis['fair_odds_a']) else '—'})"
        )

        a1, a2, a3 = st.columns(3)
        result_options = ["Pending", "Player A Won", "Player B Won", "Void"]
        current_result = selected_analysis["result"] if selected_analysis["result"] in result_options else "Pending"
        result = a1.selectbox("Result", result_options, index=result_options.index(current_result))
        closing_default = (
            int(selected_analysis["closing_odds_a"])
            if pd.notna(selected_analysis["closing_odds_a"]) and float(selected_analysis["closing_odds_a"]) != 0
            else int(selected_analysis["market_odds_a"])
        )
        closing_odds = a2.number_input(
            "Closing odds on Player A",
            value=closing_default,
            step=5,
        )

        predicted_a = float(selected_analysis["model_probability_a"]) >= 0.5
        actual_a = result == "Player A Won"
        auto_correct = "Yes" if result in ["Player A Won", "Player B Won"] and predicted_a == actual_a else (
            "No" if result in ["Player A Won", "Player B Won"] else ""
        )
        correctness_options = ["", "Yes", "No"]
        saved_correctness = str(selected_analysis["prediction_correct"])
        default_correctness = saved_correctness if saved_correctness in correctness_options and saved_correctness else auto_correct
        prediction_correct = a3.selectbox(
            "Was the prediction correct?",
            correctness_options,
            index=correctness_options.index(default_correctness) if default_correctness in correctness_options else 0,
        )

        review = st.text_area(
            "What happened?",
            value=str(selected_analysis["review"]),
            placeholder="Describe how the match actually unfolded—not just the final score.",
        )
        lesson = st.text_area(
            "What should Macabets learn?",
            value=str(selected_analysis["lesson"]),
            placeholder="Identify whether the model, assumptions, context, or price assessment needs adjustment.",
        )

        calculated_clv = closing_line_value(selected_analysis["model_probability_a"], closing_odds)
        if pd.notna(calculated_clv):
            st.caption(f"Model edge at closing price: {calculated_clv:+.1%}")

        save_col, delete_col = st.columns(2)
        if save_col.button("Save Post-Match Review", type="primary", use_container_width=True):
            idx = st.session_state.analyses.index[
                st.session_state.analyses["analysis_id"] == selected_analysis_id
            ][0]
            st.session_state.analyses.at[idx, "result"] = result
            st.session_state.analyses.at[idx, "closing_odds_a"] = int(closing_odds)
            st.session_state.analyses.at[idx, "prediction_correct"] = prediction_correct
            st.session_state.analyses.at[idx, "closing_line_value"] = float(calculated_clv)
            st.session_state.analyses.at[idx, "review"] = review.strip()
            st.session_state.analyses.at[idx, "lesson"] = lesson.strip()
            st.success("Post-match review saved.")
            st.rerun()

        if delete_col.button("Delete Analysis", use_container_width=True):
            st.session_state.analyses = st.session_state.analyses[
                st.session_state.analyses["analysis_id"] != selected_analysis_id
            ].reset_index(drop=True)
            st.success("Analysis deleted.")
            st.rerun()

        csv = normalize_analyses(st.session_state.analyses).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Analysis Archive CSV",
            csv,
            f"macabets_analysis_archive_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True,
        )

with tabs[5]:
    st.subheader("Matchup Lab")
    sport_lab = st.selectbox("Choose sport", SPORTS, key="lab_sport")

    if sport_lab in ["NFL", "College Football"]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Favorite")
            fav = st.text_input("Favorite team")
            fav_form = st.text_area("Recent form")
            fav_off = st.text_area("Offensive strengths / weaknesses")
            fav_def = st.text_area("Defensive strengths / weaknesses")
            fav_inj = st.text_area("Injuries / availability")
        with c2:
            st.markdown("#### Opponent")
            dog = st.text_input("Opponent")
            dog_form = st.text_area("Opponent recent form")
            dog_off = st.text_area("Opponent offensive profile")
            dog_def = st.text_area("Opponent defensive profile")
            situational = st.text_area("Venue, travel, rest, weather, rivalry")
        st.text_area("Where does the favorite have the clearest matchup advantage?")
        st.text_area("How can the opponent realistically upset the favorite?")
        st.slider("Overall confidence", 1, 10, 7, key="football_conf")

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
        st.slider("Overall confidence", 1, 10, 7, key="nba_conf")

    elif sport_lab == "Tennis":
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Favorite player")
            st.text_area("Recent form and workload")
            st.text_area("Serve / return advantages")
            st.text_area("Surface and conditions")
            st.text_area("Fitness / injury concerns")
        with c2:
            st.text_input("Opponent player")
            st.text_area("Opponent form")
            st.text_area("Opponent's upset weapons")
            st.text_area("Head-to-head context")
            st.text_area("Travel / scheduling / fatigue")
        st.text_area("How does the favorite lose this match?")
        st.slider("Overall confidence", 1, 10, 7, key="tennis_conf")

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
        st.slider("Overall confidence", 1, 10, 7, key="combat_conf")

    st.info("This page is a structured research worksheet. It does not automatically fetch current injuries, odds or statistics yet.")

with tabs[6]:
    st.subheader("Outcome Simulator")
    r1, r2, r3, r4 = st.columns(4)
    sim_bankroll = r1.number_input("Simulation bankroll", min_value=100.0, value=float(current_bankroll), step=1000.0)
    sim_odds = r2.number_input("Odds per bet", value=-250, step=5)
    model_prob_pct = r3.slider("Your estimated true win probability", 1.0, 99.0, 75.0, 0.5)
    number_bets = r4.number_input("Number of bets", min_value=1, max_value=500, value=50, step=1)

    s1, s2, s3 = st.columns(3)
    target_each = s1.number_input("Target profit per bet", min_value=1.0, value=float(st.session_state.target_profit), step=500.0)
    simulations = s2.number_input("Simulation runs", min_value=100, max_value=20000, value=3000, step=100)
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

with tabs[7]:
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

with tabs[9]:
    st.subheader("Daily Slate")
    st.caption(
        "Enter the full card, rank every matchup, and move the strongest opportunities "
        "into the Fair Line Engine for deeper analysis."
    )

    with st.expander("Add Matchup", expanded=True):
        s1, s2, s3, s4 = st.columns(4)
        slate_date = s1.date_input("Match date", value=date.today(), key="slate_date")
        slate_tournament = s2.text_input("Tournament", placeholder="Montreal", key="slate_tournament")
        slate_surface = s3.selectbox(
            "Surface", ["Hard", "Clay", "Grass", "Indoor Hard"], key="slate_surface"
        )
        slate_round = s4.selectbox(
            "Round",
            ["R128", "R64", "R32", "R16", "Quarterfinal", "Semifinal", "Final"],
            key="slate_round",
        )

        p1, p2 = st.columns(2)
        slate_player_a = p1.text_input("Player A", key="slate_player_a")
        slate_player_b = p2.text_input("Player B", key="slate_player_b")

        o1, o2, o3, o4 = st.columns(4)
        slate_odds_a = o1.number_input(
            "Sportsbook odds — A", value=-150, step=5, key="slate_odds_a"
        )
        slate_odds_b = o2.number_input(
            "Sportsbook odds — B", value=130, step=5, key="slate_odds_b"
        )
        slate_probability = o3.slider(
            "Macabets probability — A",
            1.0, 99.0, 60.0, 0.5,
            key="slate_probability",
        ) / 100
        slate_confidence = o4.slider(
            "Data confidence", 1, 10, 6, key="slate_confidence"
        )
        slate_notes = st.text_input(
            "Quick note",
            placeholder="Example: Better hard-court form, but fatigue needs verification.",
            key="slate_notes",
        )

        if st.button("Add to Daily Slate", type="primary", use_container_width=True):
            if not slate_player_a.strip() or not slate_player_b.strip():
                st.error("Enter both players.")
            else:
                slate = normalize_slate(st.session_state.daily_slate)
                next_id = int(slate["slate_id"].max()) + 1 if not slate.empty else 1
                row = {
                    "slate_id": next_id,
                    "match_date": slate_date.isoformat(),
                    "tournament": slate_tournament.strip(),
                    "surface": slate_surface,
                    "round": slate_round,
                    "player_a": slate_player_a.strip(),
                    "player_b": slate_player_b.strip(),
                    "market_odds_a": int(slate_odds_a),
                    "market_odds_b": int(slate_odds_b),
                    "model_probability_a": float(slate_probability),
                    "confidence": int(slate_confidence),
                    "notes": slate_notes.strip(),
                }
                st.session_state.daily_slate = normalize_slate(
                    pd.concat([slate, pd.DataFrame([row])], ignore_index=True)
                )
                st.success("Matchup added to the Daily Slate.")
                st.rerun()

    st.divider()
    slate_upload = st.file_uploader(
        "Import Daily Slate CSV",
        type=["csv"],
        key="daily_slate_upload",
    )
    if slate_upload is not None:
        try:
            st.session_state.daily_slate = normalize_slate(pd.read_csv(slate_upload))
            st.success(f"Loaded {len(st.session_state.daily_slate)} slate matchups.")
        except Exception as exc:
            st.error(f"Could not load Daily Slate CSV: {exc}")

    slate = score_daily_slate(st.session_state.daily_slate)

    if slate.empty:
        st.info("No matchups have been added to today's slate.")
    else:
        f1, f2, f3 = st.columns(3)
        tournament_options = sorted(
            [value for value in slate["tournament"].dropna().astype(str).unique() if value]
        )
        tournament_filter = f1.multiselect(
            "Tournament",
            tournament_options,
            default=tournament_options,
        )
        surface_options = sorted(slate["surface"].dropna().astype(str).unique().tolist())
        surface_filter = f2.multiselect(
            "Surface",
            surface_options,
            default=surface_options,
        )
        decision_filter = f3.multiselect(
            "Decision",
            ["BET", "WATCH", "PASS"],
            default=["BET", "WATCH", "PASS"],
        )

        filtered_slate = slate[
            slate["tournament"].astype(str).isin(tournament_filter)
            & slate["surface"].astype(str).isin(surface_filter)
            & slate["decision"].isin(decision_filter)
        ].copy()

        bet_count = int((slate["decision"] == "BET").sum())
        watch_count = int((slate["decision"] == "WATCH").sum())
        pass_count = int((slate["decision"] == "PASS").sum())
        best_score = float(slate["opportunity_score"].max())

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Slate Matches", len(slate))
        q2.metric("Bet Candidates", bet_count)
        q3.metric("Watch List", watch_count)
        q4.metric("Top Opportunity Score", f"{best_score:.1f}")

        display = filtered_slate[
            [
                "slate_id", "match_date", "tournament", "round", "player_a", "player_b",
                "market_odds_a", "market_odds_b", "model_probability_a",
                "fair_odds_a", "no_vig_edge", "estimated_roi", "confidence",
                "decision", "opportunity_score"
            ]
        ].copy()
        display["model_probability_a"] = display["model_probability_a"].map(lambda x: f"{x:.1%}")
        display["no_vig_edge"] = display["no_vig_edge"].map(lambda x: f"{x:+.1%}")
        display["estimated_roi"] = display["estimated_roi"].map(lambda x: f"{x:+.1%}")
        display["opportunity_score"] = display["opportunity_score"].map(lambda x: f"{x:.1f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("#### Open a Matchup for Deep Analysis")
        selected_slate_id = st.selectbox(
            "Slate matchup",
            slate["slate_id"].astype(int).tolist(),
            format_func=lambda x: (
                f"#{x} — "
                f"{slate.loc[slate['slate_id'] == x, 'player_a'].iloc[0]} vs "
                f"{slate.loc[slate['slate_id'] == x, 'player_b'].iloc[0]}"
            ),
        )
        selected_slate = slate[slate["slate_id"] == selected_slate_id].iloc[0]

        a1, a2, a3 = st.columns(3)
        a1.metric("Decision", selected_slate["decision"])
        a2.metric("Expected ROI", f"{selected_slate['estimated_roi']:+.1%}")
        a3.metric("Fair line — A", format_american(selected_slate["fair_odds_a"]))

        load_col, delete_col = st.columns(2)
        if load_col.button(
            "Load into Fair Line Engine",
            type="primary",
            use_container_width=True,
        ):
            surface_values = ["Hard", "Clay", "Grass", "Indoor Hard"]
            round_values = ["R128", "R64", "R32", "R16", "Quarterfinal", "Semifinal", "Final"]
            st.session_state.pending_fair_line_prefill = {
                "fle_date": date.fromisoformat(str(selected_slate["match_date"])),
                "fle_tournament": str(selected_slate["tournament"]),
                "fle_surface": (
                    str(selected_slate["surface"])
                    if str(selected_slate["surface"]) in surface_values else "Hard"
                ),
                "fle_round": (
                    str(selected_slate["round"])
                    if str(selected_slate["round"]) in round_values else "R32"
                ),
                "fle_favorite": str(selected_slate["player_a"]),
                "fle_opponent": str(selected_slate["player_b"]),
                "fle_market_a": int(selected_slate["market_odds_a"]),
                "fle_market_b": int(selected_slate["market_odds_b"]),
            }
            st.success("Matchup loaded. Open the Fair Line Engine tab.")
            st.rerun()

        if delete_col.button("Remove from Slate", use_container_width=True):
            st.session_state.daily_slate = st.session_state.daily_slate[
                st.session_state.daily_slate["slate_id"] != selected_slate_id
            ].reset_index(drop=True)
            st.success("Matchup removed.")
            st.rerun()

        export_slate = normalize_slate(st.session_state.daily_slate)
        st.download_button(
            "Download Daily Slate CSV",
            export_slate.to_csv(index=False).encode("utf-8"),
            f"macabets_daily_slate_{date.today().isoformat()}.csv",
            "text/csv",
            use_container_width=True,
        )


with tabs[8]:
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
    analysis_csv_data = normalize_analyses(st.session_state.analyses).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Analysis Archive CSV",
        data=analysis_csv_data,
        file_name=f"macabets_analysis_archive_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("#### Current Bet Data")
    st.dataframe(bets, use_container_width=True, hide_index=True)

    st.markdown("#### Current Analysis Data")
    st.dataframe(normalize_analyses(st.session_state.analyses), use_container_width=True, hide_index=True)
