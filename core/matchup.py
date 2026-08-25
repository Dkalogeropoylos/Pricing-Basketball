from __future__ import annotations
import numpy as np
import pandas as pd


def estimate_possessions(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df["FGA"], errors="coerce").fillna(0)
        - pd.to_numeric(df["OREB"], errors="coerce").fillna(0)
        + pd.to_numeric(df["TOV"], errors="coerce").fillna(0)
        + 0.44 * pd.to_numeric(df["FTA"], errors="coerce").fillna(0)
    )


def _rates(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    x = df.copy()
    for c in ["PTS","REB","AST","FG3M","FG3A","FGM","FGA","FTA","TOV","OREB","PF","BLK"]:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    poss = float(estimate_possessions(x).sum())
    if poss <= 0:
        return {}
    a3 = float(x["FG3A"].sum())
    m3 = float(x["FG3M"].sum())
    fga = float(x["FGA"].sum())
    fgm = float(x["FGM"].sum())
    a2 = max(fga-a3, 0.0)
    m2 = max(fgm-m3, 0.0)
    return {
        "games": int(len(x)),
        "poss": poss,
        "PTS": float(x["PTS"].sum()) / poss,
        "REB": float(x["REB"].sum()) / poss,
        "AST": float(x["AST"].sum()) / poss,
        "3PA": a3 / poss,
        "2PA": a2 / poss,
        "FTA": float(x["FTA"].sum()) / poss,
        "TOV": float(x["TOV"].sum()) / poss,
        "OREB": float(x["OREB"].sum()) / poss,
        "PF": float(x["PF"].sum()) / poss,
        "BLK": float(x["BLK"].sum()) / poss if "BLK" in x.columns else np.nan,
        "3P_PCT": (m3/a3) if a3 > 0 else np.nan,
        "2P_PCT": (m2/a2) if a2 > 0 else np.nan,
        "3P_ATT": a3,
        "2P_ATT": a2,
    }


def _modifier_from_ratio(stat: str, ratio: float) -> float:
    """Stat-specific shrinkage: opponent context is a correction, never a second sample."""
    if stat in {"3P_PCT", "2P_PCT"}:
        ratio = float(np.clip(ratio, 0.88, 1.12))
        return float(np.clip(ratio ** 0.18, 0.96, 1.04))
    if stat in {"3PA", "2PA"}:
        ratio = float(np.clip(ratio, 0.78, 1.22))
        return float(np.clip(ratio ** 0.30, 0.91, 1.09))
    if stat == "BLK":
        # Opponent block-susceptibility is useful but noisy. It is a correction
        # to the team's own block rate, not a second full projection.
        ratio = float(np.clip(ratio, 0.75, 1.25))
        return float(np.clip(ratio ** 0.25, 0.94, 1.06))
    ratio = float(np.clip(ratio, 0.75, 1.25))
    return float(np.clip(ratio ** 0.35, 0.88, 1.12))


def opponent_allowed_profile(league_team_logs: pd.DataFrame, opponent_abbr: str):
    all_rows = league_team_logs.copy()
    opp_rows = all_rows[
        all_rows["OPP_ABBR"].astype(str).str.upper() == str(opponent_abbr).upper()
    ].copy()

    lg = _rates(all_rows)
    opp = _rates(opp_rows)

    keys = ["PTS","REB","AST","3PA","2PA","FTA","TOV","OREB","PF","BLK","3P_PCT","2P_PCT"]
    rows, ratios, auto = [], {}, {}
    for k in keys:
        l = lg.get(k, np.nan)
        o = opp.get(k, np.nan)
        ratio = (o/l) if np.isfinite(l) and l > 0 and np.isfinite(o) else 1.0
        mod = _modifier_from_ratio(k, ratio)
        ratios[k] = float(ratio)
        auto[k] = mod
        rows.append({
            "Stat": k,
            "Opponent allowed": o,
            "League avg": l,
            "Raw ratio": ratio,
            "Applied overall modifier": mod,
            "Opponent games": opp.get("games", 0),
        })
    return {"ratios": ratios, "modifiers": auto, "audit": pd.DataFrame(rows)}


def position_environment(
    player_df: pd.DataFrame,
    opponent_abbr: str,
    position_group: str,
):
    """
    Same-position opponent environment from the already-loaded player database.
    No extra API call. Broad G/F/C groups are intentionally used for stability.
    """
    x = player_df[
        player_df["POSITION_GROUP"].astype(str) == str(position_group)
    ].copy()
    vs = x[
        x["OPP_ABBR"].astype(str).str.upper() == str(opponent_abbr).upper()
    ].copy()
    return _position_summary(vs), _position_summary(x)


def _position_summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    mins = float(pd.to_numeric(df["MIN"], errors="coerce").fillna(0).sum())
    if mins <= 0:
        return {}
    out = {"sample_min": mins, "player_game_rows": int(len(df))}
    for source, key in [
        ("PTS","PTS"),("REB","REB"),("AST","AST"),
        ("FG3A","3PA"),("FTA","FTA"),("FGA","FGA"),
        ("OREB","OREB"),("TOV","TOV")
    ]:
        out[key] = float(pd.to_numeric(df[source], errors="coerce").fillna(0).sum()) / mins * 36.0

    a3 = float(pd.to_numeric(df["FG3A"], errors="coerce").fillna(0).sum())
    m3 = float(pd.to_numeric(df["FG3M"], errors="coerce").fillna(0).sum())
    fga = float(pd.to_numeric(df["FGA"], errors="coerce").fillna(0).sum())
    fgm = float(pd.to_numeric(df["FGM"], errors="coerce").fillna(0).sum())
    a2 = max(fga-a3, 0.0)
    m2 = max(fgm-m3, 0.0)
    out["3P_PCT"] = m3/a3 if a3 > 0 else np.nan
    out["2P_PCT"] = m2/a2 if a2 > 0 else np.nan
    out["3P_ATT"] = a3
    out["2P_ATT"] = a2
    return out


def combine_overall_and_position(overall_ratio, position_ratio, position_sample_min=0.0):
    """
    Overall defense is the base. Position modifies only the positional deviation
    from overall defense, avoiding double counting.
    """
    overall_ratio = float(np.clip(overall_ratio or 1.0, 0.75, 1.25))
    if position_ratio is None or not np.isfinite(position_ratio):
        return float(np.clip(overall_ratio ** 0.30, 0.90, 1.10)), 0.0

    position_ratio = float(np.clip(position_ratio, 0.78, 1.22))
    conf = float(np.clip(position_sample_min/(position_sample_min+900.0), 0.0, 1.0))
    relative_position = float(np.clip(position_ratio/overall_ratio, 0.84, 1.16))
    mod = (overall_ratio ** 0.30) * (relative_position ** (0.18*conf))
    return float(np.clip(mod, 0.88, 1.12)), conf


def combine_efficiency_overall_and_position(
    overall_ratio,
    position_ratio,
    position_attempts=0.0,
):
    """Very conservative shooting-efficiency opponent adjustment."""
    overall_ratio = float(np.clip(overall_ratio or 1.0, 0.88, 1.12))
    if position_ratio is None or not np.isfinite(position_ratio):
        return float(np.clip(overall_ratio ** 0.16, 0.97, 1.03)), 0.0
    position_ratio = float(np.clip(position_ratio, 0.88, 1.12))
    conf = float(np.clip(position_attempts/(position_attempts+180.0), 0.0, 1.0))
    relative = float(np.clip(position_ratio/overall_ratio, 0.92, 1.08))
    mod = (overall_ratio ** 0.16) * (relative ** (0.10*conf))
    return float(np.clip(mod, 0.965, 1.035)), conf


def player_matchup_modifiers(overall_profile, position_vs_opp=None, position_league=None):
    mapping = {"PTS":"PTS","REB":"REB","AST":"AST","3PA":"3PA","FTA":"FTA"}
    rows, final = [], {}
    for stat, poskey in mapping.items():
        overall_ratio = overall_profile.get("ratios", {}).get(stat, 1.0)
        pv = (position_vs_opp or {}).get(poskey, np.nan)
        pl = (position_league or {}).get(poskey, np.nan)
        pos_ratio = (pv/pl) if np.isfinite(pv) and np.isfinite(pl) and pl > 0 else None
        sample_min = float((position_vs_opp or {}).get("sample_min", 0.0))
        mod, conf = combine_overall_and_position(overall_ratio, pos_ratio, sample_min)
        final[stat] = mod
        rows.append({
            "Stat": stat,
            "Overall raw ratio": overall_ratio,
            "Position vs opp": pv,
            "Position league": pl,
            "Position raw ratio": pos_ratio,
            "Position sample MIN": sample_min,
            "Position confidence": conf,
            "Final auto modifier": mod,
        })

    # 3P/2P efficiency: use the same position concept but with much heavier shrinkage.
    for stat, att_key in [("3P_PCT","3P_ATT"),("2P_PCT","2P_ATT")]:
        overall_ratio = overall_profile.get("ratios", {}).get(stat, 1.0)
        pv = (position_vs_opp or {}).get(stat, np.nan)
        pl = (position_league or {}).get(stat, np.nan)
        pos_ratio = (pv/pl) if np.isfinite(pv) and np.isfinite(pl) and pl > 0 else None
        attempts = float((position_vs_opp or {}).get(att_key, 0.0))
        mod, conf = combine_efficiency_overall_and_position(overall_ratio, pos_ratio, attempts)
        final[stat] = mod
        rows.append({
            "Stat": stat,
            "Overall raw ratio": overall_ratio,
            "Position vs opp": pv,
            "Position league": pl,
            "Position raw ratio": pos_ratio,
            "Position attempts": attempts,
            "Position confidence": conf,
            "Final auto modifier": mod,
        })
    return final, pd.DataFrame(rows)


def team_matchup_modifiers(overall_profile):
    wanted = ["3PA","2PA","FTA","TOV","OREB","AST","PF","BLK","3P_PCT","2P_PCT"]
    return {k: overall_profile.get("modifiers", {}).get(k, 1.0) for k in wanted}
