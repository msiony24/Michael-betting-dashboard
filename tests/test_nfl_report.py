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
    assert 'Spread Value' in source
    assert 'Projected Score' in source
    assert 'Win Probability' in source
    assert 'with st.expander("Market and line details"' not in source


def test_removed_repetitive_nfl_sections():
    source = Path("app.py").read_text(encoding="utf-8")
    assert '#### Bottom Line' not in source
    assert '## Betting Recommendation' not in source
    assert '#### What Could Go Wrong' not in source
    assert '#### Category Advantages' not in source


def test_primary_recommendation_stays_on_projected_winner_moneyline():
    source = Path('app.py').read_text(encoding='utf-8')
    assert 'actionable_moneyline_verdicts = {"Strong Bet", "Worth Betting", "Lean"}' in source
    assert 'f"{projected_nfl_winner} ML {format_american(winner_market_ml)}"' in source
    assert 'if price_report["verdict"] in actionable_moneyline_verdicts' in source
    # Spread value remains informational and must not drive the primary play.
    assert 'recommendation_text = (\n                    spread_value_text' not in source
