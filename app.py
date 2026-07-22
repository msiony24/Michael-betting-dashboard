
from datetime import date
import math

import pandas as pd
import streamlit as st

from engine.data import load_matches
from engine.tennis import (
    analyze,
    implied_probability,
    player_names,
    tournament_names,
    tournament_surface,
)


st.set_page_config(
    page_title="Macabets",
    page_icon="M",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 14px;
    padding: 14px 16px;
}
[data-testid="stMetricValue"] {font-size: 2rem;}
div.stButton > button {
    min-height: 3.25rem;
    border-radius: 12px;
    font-weight: 700;
}
.macabets-kicker {
    letter-spacing: .14em;
    text-transform: uppercase;
    opacity: .62;
    font-size: .8rem;
}
.macabets-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: .4rem;
}
.sim-hero {
    border: 1px solid rgba(128,128,128,.24);
    border-radius: 18px;
    padding: 24px;
    margin: 14px 0 20px;
    text-align: center;
    background: rgba(128,128,128,.06);
}
.sim-label {font-size:.75rem; letter-spacing:.14em; opacity:.6; font-weight:700;}
.sim-main {font-size:1.25rem; font-weight:700; margin-top:10px;}
.sim-percent {font-size:4rem; line-height:1.05; font-weight:850; margin:8px 0;}
.sim-secondary {opacity:.7;}
.result-card {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 16px;
    padding: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="macabets-kicker">Betting Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="macabets-title">Macabets</div>', unsafe_allow_html=True)
st.caption("Search a matchup. Macabets handles the model, context and simulations.")

nav = st.radio(
    "Navigation",
    ["Analyze Matchup", "My Bets"],
    horizontal=True,
    label_visibility="collapsed",
)

if nav == "Analyze Matchup":
    sport = st.selectbox(
        "Sport",
        ["Tennis", "NFL", "College Football", "NBA", "UFC", "Boxing"],
        label_visibility="collapsed",
    )

    if sport != "Tennis":
        st.info(f"The automatic {sport} engine is next. Tennis is the first working simulation model.")
        st.stop()

    try:
        with st.spinner("Loading Macabets tennis database..."):
            matches, load_errors = load_matches()
    except Exception as exc:
        st.error("Macabets could not load its tennis database.")
        st.code(str(exc))
        st.caption("Open Manage app, choose Reboot app once, and return to this page.")
        st.stop()

    players = player_names(matches)
    tournaments = tournament_names(matches)

    st.markdown("## Search matchup")
    c1, vs, c2 = st.columns([1, .12, 1])
    with c1:
        player_a = st.selectbox(
            "Player 1",
            players,
            index=None,
            placeholder="Start typing a player",
        )
    with vs:
        st.markdown("<div style='text-align:center;padding-top:2.4rem;font-weight:700'>VS</div>",
                    unsafe_allow_html=True)
    with c2:
        player_b = st.selectbox(
            "Player 2",
            players,
            index=None,
            placeholder="Start typing a player",
        )

    e1, e2, e3, e4 = st.columns(4)
    tournament = e1.selectbox(
        "Tournament",
        tournaments,
        index=None,
        placeholder="Start typing a tournament",
    )
    round_label = e2.selectbox(
        "Round",
        ["Qualifying", "R128", "R64", "R32", "R16", "Quarterfinal", "Semifinal", "Final"],
        index=4,
    )
    default_surface = tournament_surface(matches, tournament) if tournament else "Hard"
    surfaces = ["Hard", "Clay", "Grass", "Carpet"]
    surface = e3.selectbox(
        "Surface",
        surfaces,
        index=surfaces.index(default_surface) if default_surface in surfaces else 0,
    )
    event_date = e4.date_input("Event date", value=date.today())

    with st.expander("Sportsbook price — optional"):
        o1, o2 = st.columns(2)
        odds_a_text = o1.text_input(f"{player_a or 'Player 1'} odds", placeholder="-180")
        odds_b_text = o2.text_input(f"{player_b or 'Player 2'} odds", placeholder="+155")

    run_disabled = not player_a or not player_b or player_a == player_b or not tournament
    run = st.button(
        "Run Macabets",
        type="primary",
        use_container_width=True,
        disabled=run_disabled,
    )

    if run:
        with st.spinner("Running 50,000 matchup simulations..."):
            st.session_state["analysis"] = analyze(
                matches,
                player_a,
                player_b,
                tournament,
                round_label,
                surface,
                event_date,
                simulations=50000,
            )

    result = st.session_state.get("analysis")
    if result and player_a == result["player_a"] and player_b == result["player_b"]:
        st.divider()
        st.markdown(f"## {result['player_a']} vs {result['player_b']}")
        wins_a = round(result["win_probability"] * result["simulation"]["simulations"])
        wins_b = result["simulation"]["simulations"] - wins_a
        st.markdown(
            f"""
            <div class="sim-hero">
                <div class="sim-label">MACABETS SIMULATION RESULT</div>
                <div class="sim-main">{result['player_a']} won {wins_a:,} of {result['simulation']['simulations']:,} simulations</div>
                <div class="sim-percent">{result['win_probability']:.1%}</div>
                <div class="sim-secondary">{result['player_b']} won {wins_b:,} simulations · {(1-result['win_probability']):.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"{result['tournament']} · {result['round']} · {result['surface']} · "
            f"{result['simulation']['simulations']:,} simulations"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Win probability", f"{result['win_probability']:.1%}")
        m2.metric("Fair line", f"{result['fair_line']:+d}")
        m3.metric("Confidence", f"{result['confidence']}/10")
        m4.metric("Data quality", f"{result['data_quality']}/10")

        odds_a = odds_b = None
        try:
            if odds_a_text.strip():
                odds_a = int(odds_a_text.replace("+", ""))
            if odds_b_text.strip():
                odds_b = int(odds_b_text.replace("+", ""))
        except ValueError:
            st.error("Odds must be entered as whole American numbers, such as -180 and +155.")

        if odds_a is not None and odds_b is not None:
            market_a = implied_probability(odds_a)
            market_b = implied_probability(odds_b)
            no_vig_a = market_a / (market_a + market_b)
            edge = result["win_probability"] - market_a
            no_vig_edge = result["win_probability"] - no_vig_a

            x1, x2, x3 = st.columns(3)
            x1.metric("Vegas implied", f"{market_a:.1%}")
            x2.metric("No-vig market", f"{no_vig_a:.1%}")
            x3.metric("Macabets edge", f"{edge:+.1%}", f"{no_vig_edge:+.1%} vs no-vig")

            if result["data_quality"] < 5:
                st.error("PASS — the available sample is not reliable enough.")
            elif edge >= .05 and result["confidence"] >= 7:
                st.success("GREEN LIGHT — meaningful edge with sufficient confidence.")
            elif edge >= .025 and result["confidence"] >= 6:
                st.warning("LEAN — positive price advantage, below Green Light standards.")
            else:
                st.error("PASS — the current price does not offer enough value.")
        else:
            st.info("Macabets produced a fair line. Add both sportsbook prices to calculate market edge.")

        st.markdown("### Simulation outcomes")
        s1, s2, s3 = st.columns(3)
        s1.metric(f"{player_a} straight sets", f"{result['simulation']['straight_sets_a']:.1%}")
        s2.metric("Deciding set", f"{result['simulation']['deciding_set']:.1%}")
        s3.metric(f"{player_b} straight sets", f"{result['simulation']['straight_sets_b']:.1%}")

        st.markdown("### Why Macabets landed here")
        factor_frame = pd.DataFrame([
            {
                "Factor": item["name"],
                "Impact": f"{item['impact']:+.2%}",
                "Reasoning": item["reason"],
            }
            for item in result["factors"]
        ])
        st.dataframe(factor_frame, use_container_width=True, hide_index=True)

        positive = sorted(
            [item for item in result["factors"] if item["impact"] > 0],
            key=lambda item: item["impact"],
            reverse=True,
        )
        negative = sorted(
            [item for item in result["factors"] if item["impact"] < 0],
            key=lambda item: item["impact"],
        )

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("#### Top reasons")
            if positive:
                for item in positive[:3]:
                    st.write(f"**{item['name']}** · {item['impact']:+.2%}")
            else:
                st.write("No positive contextual adjustment.")
        with r2:
            st.markdown("#### Main risks")
            if negative:
                for item in negative[:3]:
                    st.write(f"**{item['name']}** · {item['impact']:+.2%}")
            else:
                st.write("No negative contextual adjustment detected.")

        st.caption(
            "The model estimates fitness from match workload and rest. Psychological pressure is represented "
            "only through historical advanced-round and deciding-match performance. It does not claim access "
            "to private medical or mental-state information."
        )

        if load_errors:
            with st.expander("Data loading notes"):
                st.write(load_errors)

else:
    st.markdown("## My Bets")
    st.info(
        "The prior ledger, bankroll and performance tools will be reconnected after the core "
        "Analyze Matchup experience is confirmed."
    )
