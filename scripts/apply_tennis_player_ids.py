from pathlib import Path


PATH = Path("update_tennis_data.py")
text = PATH.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match, found {count}: {old[:120]!r}")
    text = text.replace(old, new)


replace_once(
    'REFRESH_STATUS_PATH = DATA_DIR / "tennis_refresh_status.json"\n',
    'REFRESH_STATUS_PATH = DATA_DIR / "tennis_refresh_status.json"\n'
    'PLAYER_IDENTITY_PATH = DATA_DIR / "tennis_player_ids.csv"\n',
)

replace_once(
    '    "l_bpSaved", "w_bpFaced", "l_bpFaced",\n]\n\n\nSTAT_COLUMNS = [',
    '    "l_bpSaved", "w_bpFaced", "l_bpFaced",\n'
    '    "winner_player_key", "loser_player_key",\n'
    ']\n\n\nSTAT_COLUMNS = [',
)

replace_once(
    'STAT_COLUMNS = [\n'
    '    "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt",\n'
    '    "w_1stIn", "l_1stIn", "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon",\n'
    '    "w_SvGms", "l_SvGms", "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced",\n'
    ']\n',
    'STAT_COLUMNS = [\n'
    '    "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt",\n'
    '    "w_1stIn", "l_1stIn", "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon",\n'
    '    "w_SvGms", "l_SvGms", "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced",\n'
    ']\n'
    'PERSISTENT_API_COLUMNS = STAT_COLUMNS + ["winner_player_key", "loser_player_key"]\n',
)

anchor = '''def resolve_display_name(value: object, existing_names: dict[tuple[str, str], str]) -> str:\n    text = str(value or "").strip()\n    if not text:\n        return ""\n    return existing_names.get(player_signature(text), text)\n\n\n'''
registry_functions = '''def resolve_display_name(value: object, existing_names: dict[tuple[str, str], str]) -> str:\n    text = str(value or "").strip()\n    if not text:\n        return ""\n    return existing_names.get(player_signature(text), text)\n\n\ndef _normalized_player_key(value: object) -> str:\n    text = str(value or "").strip()\n    if text.casefold() in {"", "nan", "none", "<na>"}:\n        return ""\n    if re.fullmatch(r"\\d+\\.0", text):\n        text = text[:-2]\n    return text\n\n\ndef _read_player_identity_registry() -> pd.DataFrame:\n    if not PLAYER_IDENTITY_PATH.exists():\n        return pd.DataFrame(columns=["player_key", "alias", "canonical_name"])\n    try:\n        frame = pd.read_csv(PLAYER_IDENTITY_PATH, dtype={"player_key": str})\n    except Exception:\n        return pd.DataFrame(columns=["player_key", "alias", "canonical_name"])\n    for column in ("player_key", "alias", "canonical_name"):\n        if column not in frame:\n            frame[column] = ""\n    return frame[["player_key", "alias", "canonical_name"]]\n\n\ndef build_player_identity_registry(\n    fixtures: list[dict],\n    standings: list[dict],\n    *,\n    existing_names: dict[tuple[str, str], str],\n    existing_registry: pd.DataFrame | None = None,\n) -> pd.DataFrame:\n    """Build a durable API player-key to alias/canonical-name registry."""\n    aliases_by_key: dict[str, set[str]] = {}\n    previous_canonical: dict[str, str] = {}\n\n    prior = existing_registry if existing_registry is not None else _read_player_identity_registry()\n    if prior is not None and not prior.empty:\n        for _, row in prior.iterrows():\n            player_key = _normalized_player_key(row.get("player_key"))\n            if not player_key:\n                continue\n            alias = str(row.get("alias") or "").strip()\n            canonical = str(row.get("canonical_name") or "").strip()\n            if alias:\n                aliases_by_key.setdefault(player_key, set()).add(alias)\n            if canonical:\n                aliases_by_key.setdefault(player_key, set()).add(canonical)\n                previous_canonical.setdefault(player_key, canonical)\n\n    def add(player_key: object, name: object) -> None:\n        key = _normalized_player_key(player_key)\n        alias = str(name or "").strip()\n        if not key or not alias:\n            return\n        aliases_by_key.setdefault(key, set()).add(alias)\n\n    for row in standings:\n        add(row.get("player_key"), row.get("player"))\n    for event in fixtures:\n        add(event.get("first_player_key"), event.get("event_first_player"))\n        add(event.get("second_player_key"), event.get("event_second_player"))\n\n    records: list[dict] = []\n    for player_key, aliases in aliases_by_key.items():\n        historical_candidates: list[str] = []\n        for alias in aliases:\n            historical = existing_names.get(player_signature(alias))\n            if historical:\n                historical_candidates.append(historical)\n\n        if historical_candidates:\n            canonical = max(set(historical_candidates), key=historical_candidates.count)\n        elif previous_canonical.get(player_key):\n            canonical = previous_canonical[player_key]\n        else:\n            canonical = max(aliases, key=lambda value: (len(_name_tokens(value)), len(value)))\n\n        aliases = set(aliases)\n        aliases.add(canonical)\n        for alias in sorted(aliases):\n            records.append({\n                "player_key": player_key,\n                "alias": alias,\n                "canonical_name": canonical,\n            })\n\n    if not records:\n        return pd.DataFrame(columns=["player_key", "alias", "canonical_name"])\n    out = pd.DataFrame(records)\n    out["_alias_norm"] = out["alias"].map(lambda value: " ".join(_name_tokens(value)))\n    out = out.drop_duplicates(["player_key", "_alias_norm"], keep="last")\n    return out.drop(columns=["_alias_norm"]).sort_values(["canonical_name", "player_key", "alias"]).reset_index(drop=True)\n\n\n'''
replace_once(anchor, registry_functions)

replace_once(
    '            "score": _score_from_api(event, "first" if first_won else "second"),\n',
    '            "score": _score_from_api(event, "first" if first_won else "second"),\n'
    '            "winner_player_key": winner_key or pd.NA,\n'
    '            "loser_player_key": loser_key or pd.NA,\n',
)

replace_once(
    '    for column in STAT_COLUMNS:\n',
    '    for column in PERSISTENT_API_COLUMNS:\n',
)

replace_once(
    '    names = _existing_name_map(current_year_frames)\n'
    '    surfaces = _surface_lookup(current_year_frames)\n'
    '    ranks = _rank_map(standings_rows)\n'
    '    live = convert_api_fixtures(\n',
    '    names = _existing_name_map(current_year_frames)\n'
    '    surfaces = _surface_lookup(current_year_frames)\n'
    '    ranks = _rank_map(standings_rows)\n'
    '    identity_registry = build_player_identity_registry(\n'
    '        fixture_rows,\n'
    '        standings_rows,\n'
    '        existing_names=names,\n'
    '        existing_registry=_read_player_identity_registry(),\n'
    '    )\n'
    '    if not identity_registry.empty:\n'
    '        identity_registry.to_csv(PLAYER_IDENTITY_PATH, index=False)\n'
    '    live = convert_api_fixtures(\n',
)

replace_once(
    '        "standings_error": standings_error,\n'
    '    }\n',
    '        "standings_error": standings_error,\n'
    '        "player_identity_count": int(identity_registry["player_key"].nunique()) if not identity_registry.empty else 0,\n'
    '        "player_identity_alias_count": int(len(identity_registry)),\n'
    '    }\n',
)

PATH.write_text(text, encoding="utf-8")
print("Applied persistent tennis player ID patch.")
