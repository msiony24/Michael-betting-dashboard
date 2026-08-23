"""Tests for engine/nfl_coaching.py.

Includes a regression test for a real bug found in this audit: DEFAULT_COACHING_PATH
pointed at data/nfl/coaching_2026.csv, a path that doesn't exist, while the real
populated file has always lived at data/coaching_2026.csv. That silently flattened
every team's coaching rating to the neutral 70.0 fallback in production. Fixed by
correcting the default path; this test locks that fix in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.nfl_coaching import (
    DEFAULT_COACHING_PATH,
    NEUTRAL_COACHING_RATING,
    _record_parts,
    coaching_rating,
    load_coaching_priors,
)


def test_default_coaching_path_points_at_the_real_populated_file():
    # Regression guard: this file genuinely exists in the repo with real data
    # for all 32 teams. If this path ever drifts from the real file's location
    # again, every team's coaching rating silently flatlines to neutral.
    assert DEFAULT_COACHING_PATH.exists()
    priors = load_coaching_priors()
    assert len(priors) == 32


def test_record_parts_valid_record():
    assert _record_parts("8-9") == (8, 9)


def test_record_parts_placeholder_values_return_none():
    assert _record_parts("--") is None
    assert _record_parts("0-0") is None
    assert _record_parts("") is None
    assert _record_parts(None) is None


def test_record_parts_malformed_returns_none():
    assert _record_parts("not a record") is None


def test_coaching_rating_neutral_with_no_evidence():
    rating, components = coaching_rating(experience_years=0, prior_record="--")
    assert rating == NEUTRAL_COACHING_RATING
    assert components["experience_bonus"] == 0.0
    assert components["recent_record_adjustment"] == 0.0


def test_coaching_rating_experience_raises_rating_but_is_capped():
    rating, components = coaching_rating(experience_years=20, prior_record="--")
    assert rating > NEUTRAL_COACHING_RATING
    assert components["experience_bonus"] <= 5.0


def test_coaching_rating_winning_record_raises_rating():
    rating, components = coaching_rating(experience_years=1, prior_record="12-5")
    assert components["recent_record_adjustment"] > 0


def test_coaching_rating_losing_record_lowers_rating():
    rating, components = coaching_rating(experience_years=1, prior_record="4-13")
    assert components["recent_record_adjustment"] < 0


def test_coaching_rating_stays_within_64_to_80_bounds_even_at_extremes():
    high, _ = coaching_rating(experience_years=100, prior_record="17-0")
    low, _ = coaching_rating(experience_years=100, prior_record="0-17")
    assert 64.0 <= low <= 80.0
    assert 64.0 <= high <= 80.0


def test_load_coaching_priors_missing_file_returns_empty_dict(tmp_path: Path):
    assert load_coaching_priors(tmp_path / "does_not_exist.csv") == {}


def test_load_coaching_priors_parses_real_rows(tmp_path: Path):
    path = tmp_path / "coaching.csv"
    path.write_text(
        "team,head_coach,experience_years,record_2025,source_url,source_retrieved\n"
        "Buffalo Bills,Test Coach,5,10-7,https://example.com,2026-01-01\n",
        encoding="utf-8",
    )
    priors = load_coaching_priors(path)
    assert "Buffalo Bills" in priors
    assert priors["Buffalo Bills"]["head_coach"] == "Test Coach"
    assert priors["Buffalo Bills"]["status"] == "returning / 2025 record available"


def test_load_coaching_priors_no_record_gets_new_coach_status(tmp_path: Path):
    path = tmp_path / "coaching.csv"
    path.write_text(
        "team,head_coach,experience_years,record_2025,source_url,source_retrieved\n"
        "Buffalo Bills,New Coach,0,--,https://example.com,2026-01-01\n",
        encoding="utf-8",
    )
    priors = load_coaching_priors(path)
    assert priors["Buffalo Bills"]["status"] == "2026 coach prior / no 2025 team record"


def test_load_coaching_priors_skips_rows_without_a_team(tmp_path: Path):
    path = tmp_path / "coaching.csv"
    path.write_text(
        "team,head_coach,experience_years,record_2025,source_url,source_retrieved\n"
        ",No Team Coach,5,10-7,https://example.com,2026-01-01\n"
        "Buffalo Bills,Real Coach,5,10-7,https://example.com,2026-01-01\n",
        encoding="utf-8",
    )
    priors = load_coaching_priors(path)
    assert len(priors) == 1
    assert "Buffalo Bills" in priors
