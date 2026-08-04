from pathlib import Path


def test_nfl_report_v3_sections():
    source = Path("app.py").read_text(encoding="utf-8")
    assert '## Macabets Recommendation' in source
    assert '### Why Macabets Sees It This Way' in source
    assert '**Decisive factors**' in source
    assert '**Risk factors**' in source
    assert '### Matchup Advantages' in source
    assert '### NFL Brain' in source
    assert 'macabets-edge-card' in source
    assert 'Market Line' in source
    assert 'Macabets Fair Line' in source
    assert 'Market Edge' in source
    assert 'Projected Score' in source
    assert 'Win Probability' in source
    assert 'with st.expander("Market and line details"' not in source


def test_removed_repetitive_nfl_sections():
    source = Path("app.py").read_text(encoding="utf-8")
    assert '#### Bottom Line' not in source
    assert '## Betting Recommendation' not in source
    assert '#### What Could Go Wrong' not in source
    assert '#### Category Advantages' not in source
