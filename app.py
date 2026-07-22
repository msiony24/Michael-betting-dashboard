
import io
import math
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION = "Macabets Tennis v0.5"
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
    with st.expander("What's New in Macabets Tennis v0.5", expanded=True):
        st.markdown(
            """
            - Daily Slate workspace for ranking an entire card
            - Automatic Bet / Watch / Pass classification across all entered matches
            - Slate Opportunity Score combining edge, confidence, and model separation
            - Filters for tournament, surface, and decision
            - One-click matchup handoff into the Fair Line Engine
            - Daily Slate CSV import and export
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
    st.subheader("Fair Line Engine — Tennis v1")
    st.caption("Build an independent price from a structured matchup scorecard, then compare it with the sportsbook.")

    meta1, meta2, meta3, meta4 = st.columns(4)
    match_date = meta1.date_input("Match date", value=date.today(), key="fle_date")
    tournament = meta2.text_input("Tournament", placeholder="Montreal", key="fle_tournament")
    surface = meta3.selectbox("Surface", ["Hard", "Clay", "Grass", "Indoor Hard"], key="fle_surface")
    round_name = meta4.selectbox("Round", ["R128", "R64", "R32", "R16", "Quarterfinal", "Semifinal", "Final"], key="fle_round")

    top1, top2 = st.columns(2)
    favorite_name = top1.text_input("Player A", value="Favorite", key="fle_favorite")
    opponent_name = top2.text_input("Player B", value="Opponent", key="fle_opponent")

    market1, market2 = st.columns(2)
    market_odds = market1.number_input(
        "Sportsbook odds on Player A",
        value=-180,
        step=5,
        key="fle_market_a",
    )
    market_odds_b = market2.number_input(
        "Sportsbook odds on Player B",
        value=155,
        step=5,
        key="fle_market_b",
    )

    weights = {
        "Base quality": 0.24,
        "Surface and recent form": 0.20,
        "Serve/return matchup": 0.19,
        "Physical readiness": 0.14,
        "Conditions and scheduling": 0.11,
        "Pressure and experience": 0.12,
    }

    st.markdown("#### Matchup scorecard")
    st.caption("Rate each player from 0–10 using the information available before the match.")
    left, right = st.columns(2)
    favorite_scores = {}
    opponent_scores = {}
    with left:
        st.markdown(f"**{favorite_name}**")
        for factor in weights:
            favorite_scores[factor] = st.slider(
                factor, 0.0, 10.0, 7.0, 0.5, key=f"fav_{factor}"
            )
    with right:
        st.markdown(f"**{opponent_name}**")
        for factor in weights:
            opponent_scores[factor] = st.slider(
                factor, 0.0, 10.0, 5.5, 0.5, key=f"dog_{factor}"
            )

    confidence = st.slider(
        "Confidence in the available data", 1, 10, 7,
        help="Lower confidence pulls the model estimate toward 50% and prevents false precision.",
    )

    model_probability, weighted_difference = fair_line_probability(
        favorite_scores, opponent_scores, weights, confidence
    )
    fair_odds = probability_to_american(model_probability)
    fair_odds_b = probability_to_american(1 - model_probability)
    market_probability = implied_probability(int(market_odds))
    no_vig_a, no_vig_b, sportsbook_hold = no_vig_probabilities(market_odds, market_odds_b)
    probability_edge = model_probability - market_probability
    no_vig_edge = model_probability - no_vig_a
    expected_roi = model_probability * (american_to_decimal(int(market_odds)) - 1) - (1 - model_probability)
    minimum_price = minimum_acceptable_odds(model_probability, required_roi=0.02)
    decision, decision_reason = decision_label(expected_roi, confidence)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Macabets win probability", f"{model_probability:.1%}")
    m2.metric("Macabets fair line — A", format_american(fair_odds))
    m3.metric("Macabets fair line — B", format_american(fair_odds_b))
    m4.metric("Estimated ROI at listed price", f"{expected_roi:.1%}")

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("No-vig market — A", f"{no_vig_a:.1%}")
    n2.metric("No-vig market — B", f"{no_vig_b:.1%}")
    n3.metric("Macabets edge vs no-vig", f"{no_vig_edge:+.1%}")
    n4.metric("Sportsbook hold", f"{sportsbook_hold:.1%}")

    st.markdown("#### Decision Card")
    dcard1, dcard2, dcard3 = st.columns(3)
    dcard1.metric("Decision", decision)
    dcard2.metric("Current price", format_american(market_odds))
    dcard3.metric("Minimum acceptable price", format_american(minimum_price))
    st.caption(decision_reason)

    if decision == "BET":
        st.success(
            f"Potential value: Macabets prices {favorite_name} at {format_american(fair_odds)} "
            f"versus the market's {format_american(market_odds)}."
        )
    elif decision == "WATCH":
        st.warning(
            "The matchup is worth monitoring, but Macabets does not yet have a strong enough "
            "combination of edge and confidence."
        )
    else:
        st.error(
            f"Pass at the current price. Macabets needs approximately "
            f"{format_american(minimum_price)} or better to preserve a 2% expected return."
        )

    auto_brief = build_matchup_brief(
        favorite_name,
        opponent_name,
        favorite_scores,
        opponent_scores,
        weights,
    )
    with st.expander("Automatic Matchup Brief", expanded=True):
        st.write(auto_brief)

    with st.expander("See factor contribution"):
        rows = []
        for factor, weight in weights.items():
            difference = favorite_scores[factor] - opponent_scores[factor]
            rows.append({
                "Factor": factor,
                favorite_name: favorite_scores[factor],
                opponent_name: opponent_scores[factor],
                "Weight": weight,
                "Weighted advantage": difference * weight,
            })
        contribution_df = pd.DataFrame(rows).sort_values("Weighted advantage", ascending=False)
        st.dataframe(contribution_df, use_container_width=True, hide_index=True)
        st.caption(f"Total weighted matchup advantage: {weighted_difference:+.2f}")

    st.markdown("#### Pre-match decision record")
    d1, d2 = st.columns(2)
    prediction = d1.text_area("Why Player A wins", placeholder="The clearest winning path...")
    upset_path = d2.text_area("Why Player B wins", placeholder="The realistic upset path...")
    d3, d4 = st.columns(2)
    biggest_risk = d3.text_area("Biggest risk to the prediction")
    assumptions = d4.text_area("Key assumptions", placeholder="Example: Player A is physically healthy and serves near baseline...")
    analysis_notes = st.text_area("Additional notes")

    if st.button("Save Pre-Match Analysis", type="primary", use_container_width=True):
        analyses = st.session_state.analyses.copy()
        next_analysis_id = int(analyses["analysis_id"].max()) + 1 if not analyses.empty else 1
        row = {
            "analysis_id": next_analysis_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "match_date": match_date.isoformat(),
            "tournament": tournament.strip(),
            "surface": surface,
            "round": round_name,
            "player_a": favorite_name.strip(),
            "player_b": opponent_name.strip(),
            "market_odds_a": int(market_odds),
            "market_odds_b": int(market_odds_b),
            "model_probability_a": float(model_probability),
            "fair_odds_a": int(fair_odds),
            "no_vig_probability_a": float(no_vig_a),
            "no_vig_edge": float(no_vig_edge),
            "decision": decision,
            "minimum_acceptable_odds_a": int(minimum_price),
            "estimated_roi": float(expected_roi),
            "confidence": int(confidence),
            "prediction": prediction.strip(),
            "upset_path": upset_path.strip(),
            "biggest_risk": biggest_risk.strip(),
            "assumptions": assumptions.strip(),
            "notes": analysis_notes.strip(),
            "result": "Pending",
            "review": "",
        }
        st.session_state.analyses = normalize_analyses(
            pd.concat([analyses, pd.DataFrame([row])], ignore_index=True)
        )
        st.success("Pre-match analysis saved to the archive.")

    st.info("Current version uses manual pre-match ratings. This now creates a permanent, timestamped record so the model can be evaluated without hindsight.")

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
