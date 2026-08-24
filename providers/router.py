from providers.nba_stats import NBAStatsProvider
from providers.balldontlie_wnba import BDLWNBA


def get_provider(league: str, bdl_key: str | None = None):
    league=league.upper()
    if league=="WNBA":
        if bdl_key:
            return BDLWNBA(bdl_key), "BALLDONTLIE WNBA"
        return NBAStatsProvider("10"), "nba_api WNBA fallback"
    if league=="NBA":
        return NBAStatsProvider("00"), "nba_api NBA"
    raise ValueError(f"{league} is not enabled yet.")
