from datetime import date

from engine.nfl_weather import _climate_mismatch, _impact_from_conditions, get_nfl_weather


def test_normal_weather_has_no_material_adjustment():
    label, impact, total, penalty = _impact_from_conditions(
        temp=72, humidity=45, precip=0, snow=0, wind=8, gust=12
    )
    assert label == "Normal"
    assert impact == "None"
    assert total == 0
    assert penalty == 0


def test_severe_wind_moves_total_more_than_side():
    label, impact, total, penalty = _impact_from_conditions(
        temp=48, humidity=70, precip=0.05, snow=0, wind=26, gust=34
    )
    assert label == "High wind"
    assert impact == "High"
    assert total <= -4
    assert penalty >= 2


def test_miami_heat_can_create_small_home_familiarity_edge():
    points, message = _climate_mismatch(
        "Buffalo Bills", "Miami Dolphins", temp=92, humidity=72
    )
    assert 0 < points <= 0.75
    assert "Miami Dolphins" in message


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "hourly": {
                "time": ["2026-09-20T13:00"],
                "temperature_2m": [92],
                "relative_humidity_2m": [70],
                "precipitation": [0.0],
                "snowfall": [0.0],
                "wind_speed_10m": [9],
                "wind_gusts_10m": [14],
            }
        }


class _FakeSession:
    def get(self, *args, **kwargs):
        return _FakeResponse()


def test_weather_lookup_returns_model_adjustments(monkeypatch):
    monkeypatch.setattr(
        "engine.nfl_weather.find_scheduled_game",
        lambda *args, **kwargs: {"gameday": "2026-09-20", "gametime": "13:00", "stadium": "Hard Rock Stadium", "roof": "outdoors"},
    )
    result = get_nfl_weather(
        away_team="Buffalo Bills",
        home_team="Miami Dolphins",
        game_date=date(2026, 9, 20),
        session=_FakeSession(),
    )
    assert result["available"] is True
    assert result["home_margin_adjustment"] > 0
    assert result["source"] == "Open-Meteo"
