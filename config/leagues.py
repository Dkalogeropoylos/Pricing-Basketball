LEAGUES = {
    "WNBA": {
        "code": "WNBA",
        "minutes": 40,
        "primary_provider": "balldontlie",
        "fallback_provider": "nba_api",
        "league_id": "10",
        "enabled": True,
    },
    "NBA": {
        "code": "NBA",
        "minutes": 48,
        "primary_provider": "nba_api",
        "fallback_provider": None,
        "league_id": "00",
        "enabled": True,
    },
    "EuroLeague": {
        "code": "EUROLEAGUE",
        "minutes": 40,
        "primary_provider": "euroleague",
        "fallback_provider": None,
        "competition_code": "E",
        "enabled": False,
    },
    "EuroCup": {
        "code": "EUROCUP",
        "minutes": 40,
        "primary_provider": "euroleague",
        "fallback_provider": None,
        "competition_code": "U",
        "enabled": False,
    },
}
