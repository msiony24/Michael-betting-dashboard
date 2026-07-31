from pathlib import Path


def test_nfl_report_v2_sections():
    source = Path("app.py").read_text(encoding="utf-8")
    assert '## Game Prediction' in source
    assert '## Betting Recommendation' in source
    assert '#### What Could Go Wrong' in source
    assert '#### Decisive Factors' in source
    assert 'Spread value is the gap between those two lines' in source


def test_removed_repetitive_nfl_sections():
    source = Path("app.py").read_text(encoding="utf-8")
    assert '#### Why Each Team Can Win' not in source
    assert '#### Supporting Model Summary' not in source
    assert 'Edge: Arizona Cardinals by' not in source
