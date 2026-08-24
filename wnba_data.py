
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd

try:
    from curl_cffi import requests
except ImportError as exc:
    raise ImportError(
        "curl_cffi is required for live WNBA Stats calls. "
        "Install with: pip install curl_cffi"
    ) from exc


WNBA_STATS_BASE = "https://stats.wnba.com/stats"
LEAGUE_ID = "10"

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.wnba.com",
    "Referer": "https://www.wnba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


class WNBAStatsError(RuntimeError):
    pass


def _parse_stats_response(payload: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Parse the standard stats.wnba.com / stats.nba.com resultSets shape.
    Returns {dataset_name: DataFrame}.
    """
    result_sets = payload.get("resultSets")
    if result_sets is None:
        one = payload.get("resultSet")
        if one is not None:
            result_sets = [one]

    if result_sets is None:
        raise WNBAStatsError(
            f"Unexpected response shape. Top-level keys: {list(payload.keys())[:20]}"
        )

    frames: Dict[str, pd.DataFrame] = {}
    for i, rs in enumerate(result_sets):
        name = rs.get("name") or f"result_{i}"
        headers = rs.get("headers", [])
        rows = rs.get("rowSet", [])
        frames[name] = pd.DataFrame(rows, columns=headers)
    return frames


def fetch_endpoint(
    endpoint: str,
    params: Dict[str, Any],
    timeout: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Generic live caller for WNBA Stats endpoints.

    curl_cffi is used because the Stats surface may reject ordinary requests/TLS
    fingerprints in some environments.
    """
    url = f"{WNBA_STATS_BASE}/{endpoint}"
    resp = requests.get(
        url,
        params=params,
        headers=DEFAULT_HEADERS,
        impersonate="chrome",
        timeout=timeout,
    )

    if resp.status_code != 200:
        raise WNBAStatsError(
            f"{endpoint}: HTTP {resp.status_code}. "
            f"Response starts: {resp.text[:300]}"
        )

    try:
        payload = resp.json()
    except Exception as exc:
        raise WNBAStatsError(
            f"{endpoint}: response was not valid JSON. "
            f"Response starts: {resp.text[:300]}"
        ) from exc

    return _parse_stats_response(payload)


def _game_log_params(season: int | str, season_type: str = "Regular Season") -> Dict[str, Any]:
    """
    The PlayerGameLogs/TeamGameLogs endpoints accept nullable filters.
    We send explicit blanks because that mirrors the Stats web requests.
    """
    return {
        "DateFrom": "",
        "DateTo": "",
        "GameSegment": "",
        "LastNGames": 0,
        "LeagueID": LEAGUE_ID,
        "Location": "",
        "MeasureType": "Base",
        "Month": 0,
        "Outcome": "",
        "PORound": 0,
        "PerMode": "Totals",
        "Period": 0,
        "PlayerID": "",
        "Season": str(season),
        "SeasonSegment": "",
        "SeasonType": season_type,
        "ShotClockRange": "",
        "TeamID": "",
        "VsConference": "",
        "VsDivision": "",
    }


def fetch_player_game_logs(
    season: int | str,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    params = _game_log_params(season, season_type)
    # PlayerGameLogs uses OpposingTeamID in its public URL.
    params["OpposingTeamID"] = ""
    frames = fetch_endpoint("playergamelogs", params)
    if not frames:
        return pd.DataFrame()
    df = next(iter(frames.values())).copy()
    return normalize_player_logs(df)


def fetch_team_game_logs(
    season: int | str,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    params = _game_log_params(season, season_type)
    params["OppTeamID"] = ""
    frames = fetch_endpoint("teamgamelogs", params)
    if not frames:
        return pd.DataFrame()
    df = next(iter(frames.values())).copy()
    return normalize_team_logs(df)


def normalize_player_logs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if "GAME_DATE" in out.columns:
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")

    numeric_cols = [
        "PLAYER_ID", "TEAM_ID", "MIN", "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK",
        "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if {"FGM", "FG3M"}.issubset(out.columns):
        out["FG2M"] = out["FGM"] - out["FG3M"]
    if {"FGA", "FG3A"}.issubset(out.columns):
        out["FG2A"] = out["FGA"] - out["FG3A"]

    if "MATCHUP" in out.columns:
        # Examples normally look like "SEA vs. DAL" or "SEA @ DAL".
        out["IS_HOME"] = out["MATCHUP"].astype(str).str.contains("vs.", regex=False)
        out["OPP_ABBREVIATION"] = (
            out["MATCHUP"].astype(str)
            .str.extract(r"(?:vs\.|@)\s+([A-Z]{2,4})", expand=False)
        )

    return out


def normalize_team_logs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if "GAME_DATE" in out.columns:
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")

    numeric_cols = [
        "TEAM_ID", "MIN", "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK",
        "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if {"FGM", "FG3M"}.issubset(out.columns):
        out["FG2M"] = out["FGM"] - out["FG3M"]
    if {"FGA", "FG3A"}.issubset(out.columns):
        out["FG2A"] = out["FGA"] - out["FG3A"]

    if "MATCHUP" in out.columns:
        out["IS_HOME"] = out["MATCHUP"].astype(str).str.contains("vs.", regex=False)
        out["OPP_ABBREVIATION"] = (
            out["MATCHUP"].astype(str)
            .str.extract(r"(?:vs\.|@)\s+([A-Z]{2,4})", expand=False)
        )

    return out
