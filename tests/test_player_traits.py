import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.player_traits import (
    DEFAULT_DATABASE_PATH,
    PlayerTraitsDatabase,
    PlayerTraitsError,
    normalize_player_name,
    validate_database,
)


def load_raw_database():
    with Path(DEFAULT_DATABASE_PATH).open("r", encoding="utf-8") as file:
        return json.load(file)


def test_database_loads_and_contains_initial_player_batch():
    database = PlayerTraitsDatabase()
    # Compare against the raw file's own metadata rather than a hardcoded
    # version string, so this doesn't go stale (and silently stop testing
    # anything) on the next version bump.
    assert database.version == load_raw_database()["_metadata"]["version"]
    # Player count grows as the curated database is expanded; assert against
    # the raw file instead of a hardcoded count so this doesn't go stale
    # (and silently stop testing anything) every time a player is added.
    assert len(database.all_players()) == len(load_raw_database()["players"])
    assert len(database.all_players()) > 0
    assert "Jannik Sinner" in database.all_players()
    assert "Carlos Alcaraz" in database.all_players()


def test_alias_lookup_is_case_and_punctuation_insensitive():
    database = PlayerTraitsDatabase()
    assert database.canonical_name("J. SINNER") == "Jannik Sinner"
    assert database.canonical_name("  a. de minaur ") == "Alex de Minaur"
    assert database.exists("Djokovic") is True


def test_get_returns_profile_with_canonical_name():
    database = PlayerTraitsDatabase()
    profile = database.require("Sinner")
    assert profile["name"] == "Jannik Sinner"
    assert profile["skills"]["return"] == 5


def test_get_returns_defensive_copy():
    database = PlayerTraitsDatabase()
    profile = database.require("Sinner")
    profile["skills"]["serve"] = 1
    assert database.require("Sinner")["skills"]["serve"] == 5


def test_unknown_player_behavior():
    database = PlayerTraitsDatabase()
    assert database.get("Unknown Player") is None
    assert database.exists("Unknown Player") is False
    with pytest.raises(KeyError):
        database.require("Unknown Player")


def test_rating_label():
    database = PlayerTraitsDatabase()
    assert database.rating_label(5) == "Elite"
    assert database.rating_label(1) == "Below Average"
    with pytest.raises(ValueError):
        database.rating_label(6)


def test_normalize_player_name():
    assert normalize_player_name("  J. Sinner  ") == "j sinner"


def test_validation_rejects_missing_required_skill():
    data = deepcopy(load_raw_database())
    del data["players"]["Jannik Sinner"]["skills"]["serve"]
    with pytest.raises(PlayerTraitsError, match="Invalid skill keys"):
        validate_database(data)


def test_validation_rejects_invalid_rating():
    data = deepcopy(load_raw_database())
    data["players"]["Jannik Sinner"]["skills"]["serve"] = 6
    with pytest.raises(PlayerTraitsError, match="integer from 1 to 5"):
        validate_database(data)


def test_validation_rejects_unknown_trait():
    data = deepcopy(load_raw_database())
    data["players"]["Jannik Sinner"]["signature_traits"].append("Imaginary Trait")
    with pytest.raises(PlayerTraitsError, match="Invalid signature traits"):
        validate_database(data)


def test_validation_rejects_duplicate_alias():
    data = deepcopy(load_raw_database())
    data["players"]["Carlos Alcaraz"]["aliases"].append("Sinner")
    with pytest.raises(PlayerTraitsError, match="Duplicate player name or alias"):
        validate_database(data)
