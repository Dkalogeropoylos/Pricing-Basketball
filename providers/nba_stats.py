from __future__ import annotations
import pandas as pd


class NBAStatsProvider:
    """nba_api adapter for NBA/WNBA basic player logs. Used as fallback."""
    def __init__(self, league_id="00"):
        self.league_id=str(league_id)

    def players(self, season):
        from nba_api.stats.endpoints import leaguedashplayerstats
        ep=leaguedashplayerstats.LeagueDashPlayerStats(
            season=str(season),
            season_type_all_star="Regular Season",
            league_id_nullable=self.league_id,
            per_mode_detailed="PerGame",
            timeout=20,
        )
        df=ep.get_data_frames()[0]
        out=df.rename(columns={"PLAYER_ID":"PLAYER_ID","PLAYER_NAME":"PLAYER_NAME","TEAM_ABBREVIATION":"TEAM_ABBR"})
        keep=[c for c in ["PLAYER_ID","PLAYER_NAME","TEAM_ABBR"] if c in out.columns]
        return out[keep].drop_duplicates()

    def player_game_log(self, player_id, season):
        from nba_api.stats.endpoints import playergamelog
        ep=playergamelog.PlayerGameLog(
            player_id=int(player_id),
            season=str(season),
            season_type_all_star="Regular Season",
            league_id_nullable=self.league_id,
            timeout=20,
        )
        x=ep.get_data_frames()[0].copy()
        x=x.rename(columns={
            "Game_ID":"GAME_ID","GAME_DATE":"GAME_DATE","MIN":"MIN",
            "FGM":"FGM","FGA":"FGA","FG3M":"FG3M","FG3A":"FG3A",
            "FTM":"FTM","FTA":"FTA","OREB":"OREB","DREB":"DREB","REB":"REB",
            "AST":"AST","STL":"STL","BLK":"BLK","TOV":"TOV","PF":"PF","PTS":"PTS",
        })
        if "MATCHUP" in x.columns:
            x["TEAM_ABBR"]=x["MATCHUP"].astype(str).str.split().str[0]
            x["OPP_ABBR"]=x["MATCHUP"].astype(str).str.extract(r"(?:vs\.|@)\s*([A-Z]{2,4})",expand=False)
        x["PLAYER_ID"]=int(player_id)
        return x
