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
    for c in ["PTS","REB","AST","FG3A","FGA","FTA","TOV","OREB","PF"]:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    poss = float(estimate_possessions(x).sum())
    if poss <= 0:
        return {}
    a3 = float(x["FG3A"].sum())
    fga = float(x["FGA"].sum())
    return {
        "games": int(len(x)),
        "poss": poss,
        "PTS": float(x["PTS"].sum()) / poss,
        "REB": float(x["REB"].sum()) / poss,
        "AST": float(x["AST"].sum()) / poss,
        "3PA": a3 / poss,
        "2PA": max(fga-a3, 0.0) / poss,
        "FTA": float(x["FTA"].sum()) / poss,
        "TOV": float(x["TOV"].sum()) / poss,
        "OREB": float(x["OREB"].sum()) / poss,
        "PF": float(x["PF"].sum()) / poss,
    }


def opponent_allowed_profile(league_team_logs: pd.DataFrame, opponent_abbr: str):
    all_rows = league_team_logs.copy()
    opp_rows = all_rows[
        all_rows["OPP_ABBR"].astype(str).str.upper() == str(opponent_abbr).upper()
    ].copy()

    lg = _rates(all_rows)
    opp = _rates(opp_rows)

    keys = ["PTS","REB","AST","3PA","2PA","FTA","TOV","OREB","PF"]
    rows, ratios, auto = [], {}, {}
    for k in keys:
        l = lg.get(k, np.nan)
        o = opp.get(k, np.nan)
        ratio = (o/l) if np.isfinite(l) and l > 0 and np.isfinite(o) else 1.0
        ratio = float(np.clip(ratio, 0.72, 1.28))
        mod = float(np.clip(ratio ** 0.45, 0.86, 1.14))
        ratios[k] = ratio
        auto[k] = mod
        rows.append({
            "Stat": k,
            "Opponent allowed / poss": o,
            "League avg / poss": l,
            "Raw ratio": ratio,
            "Applied overall modifier": mod,
            "Opponent games": opp.get("games", 0),
        })
    return {"ratios": ratios, "modifiers": auto, "audit": pd.DataFrame(rows)}


def combine_overall_and_position(overall_ratio, position_ratio, position_sample_min=0.0):
    """
    Overall defense is the base. Position modifies only the positional
    deviation from that overall defense, avoiding double counting.
    """
    overall_ratio = float(np.clip(overall_ratio or 1.0, 0.72, 1.28))
    if position_ratio is None or not np.isfinite(position_ratio):
        return float(np.clip(overall_ratio ** 0.45, 0.86, 1.14)), 0.0

    position_ratio = float(np.clip(position_ratio, 0.72, 1.28))
    conf = float(np.clip(position_sample_min/(position_sample_min+800.0), 0.0, 1.0))
    relative_position = float(np.clip(position_ratio/overall_ratio, 0.82, 1.18))
    mod = (overall_ratio ** 0.45) * (relative_position ** (0.25*conf))
    return float(np.clip(mod, 0.84, 1.16)), conf


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
            "Position vs opp /36": pv,
            "Position league /36": pl,
            "Position raw ratio": pos_ratio,
            "Position sample MIN": sample_min,
            "Position confidence": conf,
            "Final auto modifier": mod,
        })
    return final, pd.DataFrame(rows)


def team_matchup_modifiers(overall_profile):
    wanted = ["3PA","2PA","FTA","TOV","OREB","AST","PF"]
    return {k: overall_profile.get("modifiers", {}).get(k, 1.0) for k in wanted}
