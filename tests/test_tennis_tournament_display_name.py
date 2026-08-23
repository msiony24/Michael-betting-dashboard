from __future__ import annotations

import pandas as pd
import pytest

from engine.tennis import (
    resolve_tournament_display_name,
    tournament_category,
    tournament_category_for_display_name,
    tournament_surface_for_display_name,
)


def _matches(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_resolve_display_name_exact_match_short_circuits():
    matches = _matches([{"tourney_name": "Cincinnati", "tourney_date": "20260101", "tourney_level": "M", "surface": "Hard"}])
    assert resolve_tournament_display_name(matches, "Cincinnati") == "Cincinnati"


def test_resolve_display_name_bridges_sponsor_name_change():
    # The exact bug: the odds feed says "ATP Cincinnati Open", the historical
    # data (in an older season) calls it something else entirely because the
    # title sponsor changed. A plain exact/substring match on the current
    # display name would find nothing.
    matches = _matches([
        {"tourney_name": "Western & Southern Financial Group Masters", "tourney_date": "20230101", "tourney_level": "M", "surface": "Hard"},
        {"tourney_name": "Cincinnati", "tourney_date": "20260101", "tourney_level": "M", "surface": "Hard"},
    ])
    resolved = resolve_tournament_display_name(matches, "ATP Cincinnati Open")
    assert resolved == "Cincinnati"


def test_resolve_display_name_no_match_returns_input_unchanged():
    matches = _matches([{"tourney_name": "Cincinnati", "tourney_date": "20260101", "tourney_level": "M", "surface": "Hard"}])
    assert resolve_tournament_display_name(matches, "Some Random Challenger Event") == "Some Random Challenger Event"


def test_resolve_display_name_empty_matches_returns_input_unchanged():
    assert resolve_tournament_display_name(pd.DataFrame(), "ATP Cincinnati Open") == "ATP Cincinnati Open"


def test_category_for_display_name_cincinnati_is_masters_1000():
    matches = _matches([
        {"tourney_name": "Western & Southern Financial Group Masters", "tourney_date": "20230101", "tourney_level": "M", "surface": "Hard"},
    ])
    assert tournament_category_for_display_name(matches, "ATP Cincinnati Open") == "Masters 1000"


def test_category_for_display_name_falls_back_to_keyword_inference_with_no_data_at_all():
    # No historical data whatsoever -- still shouldn't silently call a
    # well-known Masters city an ATP 250 just because nothing matched.
    assert tournament_category_for_display_name(pd.DataFrame(), "ATP Cincinnati Open") == "Masters 1000"
    assert tournament_category_for_display_name(pd.DataFrame(), "ATP Shanghai Masters") == "Masters 1000"


def test_category_name_only_fallback_does_not_false_positive_on_substrings():
    # Regression: "Halle" (an ATP 500 city) must not match merely because its
    # letters appear inside an unrelated word like "Challenger".
    assert tournament_category_for_display_name(pd.DataFrame(), "Some Random Challenger Event") == "ATP 250"


def test_category_name_only_fallback_handles_apostrophes():
    assert tournament_category_for_display_name(pd.DataFrame(), "ATP Queen's Club Championships") == "ATP 500"


def test_surface_for_display_name_bridges_like_category_does():
    matches = _matches([
        {"tourney_name": "Cincinnati", "tourney_date": "20260101", "tourney_level": "M", "surface": "Hard"},
    ])
    assert tournament_surface_for_display_name(matches, "ATP Cincinnati Open") == "Hard"


def test_category_still_prefers_real_data_over_name_guessing():
    # If the real data disagrees with the name-based guess (e.g. a one-off
    # event downgraded/upgraded in level), the actual historical level wins.
    matches = _matches([
        {"tourney_name": "Some Small Event", "tourney_date": "20260101", "tourney_level": "G", "surface": "Grass"},
    ])
    assert tournament_category(matches, "Some Small Event") == "Grand Slam"
