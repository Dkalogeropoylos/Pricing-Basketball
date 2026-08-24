
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
import pandas as pd


BASIC_STATS = [
    "MIN", "PTS", "REB", "OREB", "DREB", "AST",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "TOV", "STL", "BLK", "PF", "PFD", "FG2M", "FG2A"
]


def safe_mean(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return float("nan")
    return float(pd.to_numeric(s, errors="coerce").mean())


def summarize_games(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    cols = [c for c in BASIC_STATS if c in df.columns]
    return df[cols].apply(pd.to_numeric, errors="coerce").mean()


def window_summary(
    player_logs: pd.DataFrame,
    player_name: str,
    last_n: Optional[int] = None,
    opponent: Optional[str] = None,
) -> pd.Series:
    d = player_logs[player_logs["PLAYER_NAME"] == player_name].copy()
    if "GAME_DATE" in d.columns:
        d = d.sort_values("GAME_DATE", ascending=False)

    if opponent:
        d = d[d["OPP_ABBREVIATION"] == opponent]

    if last_n:
        d = d.head(last_n)

    return summarize_games(d)


def selected_role_summary(df: pd.DataFrame, use_col: str = "USE") -> pd.Series:
    if use_col not in df.columns:
        return summarize_games(df)
    return summarize_games(df[df[use_col].fillna(False)])


def per_minute(summary: pd.Series, stat: str) -> float:
    minutes = float(summary.get("MIN", np.nan))
    value = float(summary.get(stat, np.nan))
    if not np.isfinite(minutes) or minutes <= 0 or not np.isfinite(value):
        return 0.0
    return value / minutes


def build_rate_triplet(
    season_summary: pd.Series,
    l10_summary: pd.Series,
    role_summary: pd.Series,
    stat: str,
):
    from wnba_prop_model import RateWindow
    return RateWindow(
        season=per_minute(season_summary, stat),
        last10=per_minute(l10_summary, stat),
        recent_role=per_minute(role_summary, stat),
    )


def opponent_team_allowed(
    team_logs: pd.DataFrame,
    opponent_abbreviation: str,
    stat: str,
) -> float:
    """
    Estimate how much the selected opponent allows by using the opposing
    teams' box scores in games where OPP_ABBREVIATION == opponent.

    Because each row is a team's own box score, rows whose opponent is the
    selected team are what that selected team allowed.
    """
    d = team_logs[team_logs["OPP_ABBREVIATION"] == opponent_abbreviation].copy()
    if d.empty or stat not in d.columns:
        return float("nan")
    return safe_mean(d[stat])


def league_team_average(team_logs: pd.DataFrame, stat: str) -> float:
    if stat not in team_logs.columns:
        return float("nan")
    return safe_mean(team_logs[stat])


def allowance_index(team_logs: pd.DataFrame, opponent: str, stat: str) -> float:
    allowed = opponent_team_allowed(team_logs, opponent, stat)
    league = league_team_average(team_logs, stat)
    if not np.isfinite(allowed) or not np.isfinite(league) or league == 0:
        return 1.0
    return float(allowed / league)


def infer_h2h_index(
    player_logs: pd.DataFrame,
    player_name: str,
    opponent: str,
    stat: str,
    current_rate_per_min: float,
    max_games: int = 3,
) -> float:
    """
    Lightweight H2H rate index. OT cleaning / rotation comparability must still
    be handled manually in the UI for the MVP.
    """
    d = player_logs[
        (player_logs["PLAYER_NAME"] == player_name)
        & (player_logs["OPP_ABBREVIATION"] == opponent)
    ].copy()

    if "GAME_DATE" in d.columns:
        d = d.sort_values("GAME_DATE", ascending=False)
    d = d.head(max_games)

    if d.empty or stat not in d.columns or "MIN" not in d.columns:
        return 1.0

    valid = d[(d["MIN"] > 0)].copy()
    if valid.empty:
        return 1.0

    rate = (valid[stat].sum() / valid["MIN"].sum())
    if current_rate_per_min <= 0:
        return 1.0
    return float(rate / current_rate_per_min)
