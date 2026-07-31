"""Streamlit renderer for Macabets matchup intelligence.

This module is explanation-only. It displays the output from
engine.matchup_engine without changing probabilities, fair odds, ROI,
confidence, or betting verdicts.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from engine.matchup_engine import analyze_matchup


_STRENGTH_LABELS = {
    "Neutral": "Even",
    "Minor": "Slight edge",
    "Moderate": "Clear edge",
    "Major": "Major edge",
    "Match-Defining": "Match-defining edge",
}


def _render_edge(edge: dict[str, Any], player_a: str, player_b: str) -> None:
    category = str(edge.get("category", "Matchup factor"))
    strength = str(edge.get("strength", "Neutral"))
    winner = edge.get("winner")
    explanation = str(edge.get("explanation", "")).strip()

    st.markdown(f"**{category}**")
    if winner:
        st.write(f"{_STRENGTH_LABELS.get(strength, strength)}: **{winner}**")
    else:
        st.write("Even")
    if explanation:
        st.caption(explanation)


def render_matchup_engine_report(
    player_a: str,
    player_b: str,
    tournament: str,
    surface: str,
    environment: str = "Outdoor",
) -> dict[str, Any]:
    """Render the explanation-only matchup report and return its raw result."""

    try:
        report = analyze_matchup(
            player_a=player_a,
            player_b=player_b,
            tournament=tournament,
            surface=surface,
            environment=environment,
        )
    except Exception as exc:
        st.warning(f"Matchup intelligence is temporarily unavailable: {exc}")
        return {
            "status": "error",
            "error": str(exc),
            "explanation_only": True,
        }

    st.markdown("#### Matchup Intelligence")

    if report.get("status") != "ok":
        missing = report.get("missing_players", [])
        if missing:
            st.info(
                "Curated matchup intelligence is unavailable for: "
                + ", ".join(str(name) for name in missing)
                + ". The main statistical analysis remains unchanged."
            )
        else:
            st.info("Curated matchup intelligence is unavailable for this matchup.")
        return report

    st.info(str(report.get("tactical_read", "No clear tactical read is available.")))

    edges = list(report.get("edges", []))
    if edges:
        for index in range(0, len(edges), 2):
            columns = st.columns(2)
            for offset, column in enumerate(columns):
                edge_index = index + offset
                if edge_index >= len(edges):
                    continue
                with column:
                    _render_edge(edges[edge_index], player_a, player_b)

    st.markdown("##### Independent Paths to Victory")
    paths = report.get("paths_to_victory", {})
    counts = report.get("path_counts", {})
    path_col_a, path_col_b = st.columns(2)

    with path_col_a:
        st.markdown(f"**{player_a}: {counts.get(player_a, 0)}**")
        player_paths = list(paths.get(player_a, []))
        if player_paths:
            for path in player_paths:
                st.markdown(f"- {path}")
        else:
            st.caption("No independent moderate-or-strong matchup edge identified.")

    with path_col_b:
        st.markdown(f"**{player_b}: {counts.get(player_b, 0)}**")
        player_paths = list(paths.get(player_b, []))
        if player_paths:
            for path in player_paths:
                st.markdown(f"- {path}")
        else:
            st.caption("No independent moderate-or-strong matchup edge identified.")

    decisive_factors = list(report.get("decisive_factors", []))
    if decisive_factors:
        st.markdown("##### Decisive Factors")
        for factor in decisive_factors:
            winner = factor.get("winner")
            category = factor.get("category", "Matchup factor")
            explanation = factor.get("explanation", "")
            if winner:
                st.markdown(f"- **{winner} — {category}:** {explanation}")
            else:
                st.markdown(f"- **{category}:** {explanation}")

    st.caption(
        "This section explains player interaction only. It does not alter the "
        "probability model, fair odds, ROI, confidence score, or betting verdict."
    )
    return report
