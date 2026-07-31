"""Focused tests for abbreviated ATP player-name resolution."""

from __future__ import annotations

from engine.matchup_engine import resolve_player_profile
from engine.player_traits import PlayerTraitsDatabase


def test_surname_initial_resolves_sinner():
    database = PlayerTraitsDatabase()
    profile, resolution = resolve_player_profile(database, "Sinner J.")

    assert profile is not None
    assert profile["name"] == "Jannik Sinner"
    assert resolution["method"] == "surname_initial"


def test_surname_initial_resolves_zverev():
    database = PlayerTraitsDatabase()
    profile, resolution = resolve_player_profile(database, "Zverev A.")

    assert profile is not None
    assert profile["name"] == "Alexander Zverev"
    assert resolution["method"] == "surname_initial"


def test_multiword_surname_resolves():
    database = PlayerTraitsDatabase()
    profile, resolution = resolve_player_profile(database, "de Minaur A.")

    assert profile is not None
    assert profile["name"] == "Alex de Minaur"
    assert resolution["method"] == "surname_initial"


def test_initial_first_format_resolves():
    database = PlayerTraitsDatabase()
    profile, resolution = resolve_player_profile(database, "J. Sinner")

    assert profile is not None
    assert profile["name"] == "Jannik Sinner"


def test_unknown_name_remains_unresolved():
    database = PlayerTraitsDatabase()
    profile, resolution = resolve_player_profile(database, "NotAPlayer X.")

    assert profile is None
    assert resolution["method"] == "unresolved"
