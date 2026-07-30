from engine.tournament_profiles import court_context_sentence, resolve_tournament_profile


def test_alias_resolution() -> None:
    profile = resolve_tournament_profile("French Open", "Clay")
    assert profile["matched_name"] == "roland garros"
    assert profile["speed_label"] == "medium-slow"


def test_sponsor_name_resolution() -> None:
    profile = resolve_tournament_profile("BNP Paribas Open", "Hard")
    assert profile["matched_name"] == "indian wells"
    assert profile["bounce"] == "high"


def test_unknown_event_falls_back_to_surface() -> None:
    profile = resolve_tournament_profile("Unknown Indoor Event", "Hard", "Indoor")
    assert profile["source"] == "surface_fallback"
    assert profile["speed_label"] == "medium-fast"
    assert profile["environment"] == "indoor"


def test_context_sentence_mentions_madrid_conditions() -> None:
    profile = resolve_tournament_profile("Madrid", "Clay")
    sentence = court_context_sentence("Madrid", profile)
    assert "medium-fast clay conditions" in sentence
    assert "high altitude" in sentence
