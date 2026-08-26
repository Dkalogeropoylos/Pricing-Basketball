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


def _safe_div(a, b, default=np.nan):
    return float(a) / float(b) if np.isfinite(b) and float(b) > 0 else default


def _rates(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    x = df.copy()
    for c in [
        "PTS", "REB", "AST", "FG3M", "FG3A", "FGM", "FGA", "FTA",
        "TOV", "OREB", "PF", "BLK"
    ]:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    poss = float(estimate_possessions(x).sum())
    if poss <= 0:
        return {}

    a3 = float(x["FG3A"].sum())
    m3 = float(x["FG3M"].sum())
    fga = float(x["FGA"].sum())
    fgm = float(x["FGM"].sum())
    a2 = max(fga - a3, 0.0)
    m2 = max(fgm - m3, 0.0)
    tov = float(x["TOV"].sum())
    live = max(poss - tov, 1e-9)
    misses = max(fga - fgm, 0.0)

    return {
        "games": int(len(x)),
        "poss": poss,
        # Existing per-possession fields retained for Player Props/backward compatibility.
        "PTS": float(x["PTS"].sum()) / poss,
        "REB": float(x["REB"].sum()) / poss,
        "AST": float(x["AST"].sum()) / poss,
        "3PA": a3 / poss,
        "2PA": a2 / poss,
        "FTA": float(x["FTA"].sum()) / poss,
        "TOV": tov / poss,
        "OREB": float(x["OREB"].sum()) / poss,
        "PF": float(x["PF"].sum()) / poss,
        # For rows representing offenses playing AGAINST an opponent, BLK is the
        # number of blocks made by those offenses against the opponent's shots.
        # When opponent_allowed_profile filters OPP_ABBR==opponent, this therefore
        # measures how many blocks the opponent OFFENSE suffers from its opponents.
        "BLK": float(x["BLK"].sum()) / poss if "BLK" in x.columns else np.nan,
        "3P_PCT": (m3 / a3) if a3 > 0 else np.nan,
        "2P_PCT": (m2 / a2) if a2 > 0 else np.nan,
        "3P_ATT": a3,
        "2P_ATT": a2,
        # Team-market structural rates. These remove avoidable overlap:
        # FGA_LIVE is conditional on no turnover; 3P_SHARE allocates FGA between
        # 3PA/2PA; OREB_PER_MISS is conditional on a rebound opportunity;
        # AST_PER_MAKE is conditional on a made FG.
        "FGA_LIVE": fga / live,
        "3P_SHARE": a3 / fga if fga > 0 else np.nan,
        "OREB_PER_MISS": float(x["OREB"].sum()) / misses if misses > 0 else np.nan,
        "AST_PER_MAKE": float(x["AST"].sum()) / fgm if fgm > 0 else np.nan,
    }


def _modifier_from_ratio(stat: str, ratio: float) -> float:
    """Stat-specific shrinkage: opponent context is a correction, not a second sample."""
    if stat in {"3P_PCT", "2P_PCT"}:
        ratio = float(np.clip(ratio, 0.88, 1.12))
        return float(np.clip(ratio ** 0.18, 0.96, 1.04))
    if stat in {"3PA", "2PA"}:
        # Player-prop compatibility only; Team Markets use 3P_SHARE instead.
        ratio = float(np.clip(ratio, 0.78, 1.22))
        return float(np.clip(ratio ** 0.30, 0.91, 1.09))
    if stat == "FGA_LIVE":
        ratio = float(np.clip(ratio, 0.82, 1.18))
        return float(np.clip(ratio ** 0.20, 0.95, 1.05))
    if stat == "3P_SHARE":
        ratio = float(np.clip(ratio, 0.82, 1.18))
        return float(np.clip(ratio ** 0.35, 0.94, 1.06))
    if stat == "OREB_PER_MISS":
        ratio = float(np.clip(ratio, 0.78, 1.22))
        return float(np.clip(ratio ** 0.35, 0.90, 1.10))
    if stat == "AST_PER_MAKE":
        ratio = float(np.clip(ratio, 0.80, 1.20))
        return float(np.clip(ratio ** 0.30, 0.92, 1.08))
    if stat == "BLK":
        # BLK is opponent offensive susceptibility: how often this offense is
        # blocked by its opponents. This is deliberately more meaningful than
        # a 2PA proxy, but still only a correction to the defense's own BLK rate.
        ratio = float(np.clip(ratio, 0.72, 1.28))
        return float(np.clip(ratio ** 0.40, 0.90, 1.10))
    ratio = float(np.clip(ratio, 0.75, 1.25))
    return float(np.clip(ratio ** 0.35, 0.88, 1.12))


def opponent_allowed_profile(
    league_team_logs: pd.DataFrame,
    opponent_abbr: str,
    exclude_team_abbr: str | None = None,
):
    """Opponent-allowed profile, optionally excluding the current H2H opponent.

    exclude_team_abbr is used by Team Markets to keep opponent-allowed evidence
    DISJOINT from the explicit H2H sample. Player Props can continue using the
    default full opponent history.
    """
    all_rows = league_team_logs.copy()
    opp_rows = all_rows[
        all_rows["OPP_ABBR"].astype(str).str.upper() == str(opponent_abbr).upper()
    ].copy()
    if exclude_team_abbr:
        opp_rows = opp_rows[
            ~opp_rows["TEAM_ABBR"].astype(str).str.upper().eq(str(exclude_team_abbr).upper())
        ].copy()

    lg = _rates(all_rows)
    opp = _rates(opp_rows)

    keys = [
        "PTS", "REB", "AST", "3PA", "2PA", "FTA", "TOV", "OREB", "PF", "BLK",
        "3P_PCT", "2P_PCT", "FGA_LIVE", "3P_SHARE", "OREB_PER_MISS", "AST_PER_MAKE",
    ]
    rows, ratios, auto = [], {}, {}
    for k in keys:
        l = lg.get(k, np.nan)
        o = opp.get(k, np.nan)
        ratio = (o / l) if np.isfinite(l) and l > 0 and np.isfinite(o) else 1.0
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
            "H2H excluded team": str(exclude_team_abbr or "—"),
        })
    return {"ratios": ratios, "modifiers": auto, "audit": pd.DataFrame(rows)}


def position_environment(
    player_df: pd.DataFrame,
    opponent_abbr: str,
    position_group: str,
):
    """Same-position opponent environment for Player Props."""
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
        ("PTS", "PTS"), ("REB", "REB"), ("AST", "AST"),
        ("FG3A", "3PA"), ("FTA", "FTA"), ("FGA", "FGA"),
        ("OREB", "OREB"), ("TOV", "TOV")
    ]:
        out[key] = float(pd.to_numeric(df[source], errors="coerce").fillna(0).sum()) / mins * 36.0

    a3 = float(pd.to_numeric(df["FG3A"], errors="coerce").fillna(0).sum())
    m3 = float(pd.to_numeric(df["FG3M"], errors="coerce").fillna(0).sum())
    fga = float(pd.to_numeric(df["FGA"], errors="coerce").fillna(0).sum())
    fgm = float(pd.to_numeric(df["FGM"], errors="coerce").fillna(0).sum())
    a2 = max(fga - a3, 0.0)
    m2 = max(fgm - m3, 0.0)
    out["3P_PCT"] = m3 / a3 if a3 > 0 else np.nan
    out["2P_PCT"] = m2 / a2 if a2 > 0 else np.nan
    out["3P_ATT"] = a3
    out["2P_ATT"] = a2
    return out


def combine_overall_and_position(overall_ratio, position_ratio, position_sample_min=0.0):
    """Overall defense is base; position changes only the relative deviation."""
    overall_ratio = float(np.clip(overall_ratio or 1.0, 0.75, 1.25))
    if position_ratio is None or not np.isfinite(position_ratio):
        return float(np.clip(overall_ratio ** 0.30, 0.90, 1.10)), 0.0

    position_ratio = float(np.clip(position_ratio, 0.78, 1.22))
    conf = float(np.clip(position_sample_min / (position_sample_min + 900.0), 0.0, 1.0))
    relative_position = float(np.clip(position_ratio / overall_ratio, 0.84, 1.16))
    mod = (overall_ratio ** 0.30) * (relative_position ** (0.18 * conf))
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
    conf = float(np.clip(position_attempts / (position_attempts + 180.0), 0.0, 1.0))
    relative = float(np.clip(position_ratio / overall_ratio, 0.92, 1.08))
    mod = (overall_ratio ** 0.16) * (relative ** (0.10 * conf))
    return float(np.clip(mod, 0.965, 1.035)), conf


def player_matchup_modifiers(overall_profile, position_vs_opp=None, position_league=None):
    mapping = {"PTS": "PTS", "REB": "REB", "AST": "AST", "3PA": "3PA", "FTA": "FTA"}
    rows, final = [], {}
    for stat, poskey in mapping.items():
        overall_ratio = overall_profile.get("ratios", {}).get(stat, 1.0)
        pv = (position_vs_opp or {}).get(poskey, np.nan)
        pl = (position_league or {}).get(poskey, np.nan)
        pos_ratio = (pv / pl) if np.isfinite(pv) and np.isfinite(pl) and pl > 0 else None
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

    for stat, att_key in [("3P_PCT", "3P_ATT"), ("2P_PCT", "2P_ATT")]:
        overall_ratio = overall_profile.get("ratios", {}).get(stat, 1.0)
        pv = (position_vs_opp or {}).get(stat, np.nan)
        pl = (position_league or {}).get(stat, np.nan)
        pos_ratio = (pv / pl) if np.isfinite(pv) and np.isfinite(pl) and pl > 0 else None
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
    """Team-market modifiers with structurally non-overlapping denominators."""
    mods = overall_profile.get("modifiers", {})
    return {
        "FGA": mods.get("FGA_LIVE", 1.0),
        "3P_SHARE": mods.get("3P_SHARE", 1.0),
        "FTA": mods.get("FTA", 1.0),
        "TOV": mods.get("TOV", 1.0),
        "OREB": mods.get("OREB_PER_MISS", 1.0),
        "AST": mods.get("AST_PER_MAKE", 1.0),
        "PF": mods.get("PF", 1.0),
        "BLK": mods.get("BLK", 1.0),
        "3P_PCT": mods.get("3P_PCT", 1.0),
        "2P_PCT": mods.get("2P_PCT", 1.0),
    }


def block_position_susceptibility_modifier(
    player_df: pd.DataFrame,
    offense_abbr: str,
    current_pool: pd.DataFrame | None = None,
    out_players: list[str] | None = None,
):
    """Optional small positional block-susceptibility correction.

    Requires a player-level blocked-attempt column (BLKA / BA /
    BLOCKS_AGAINST). If the loaded dataset does not contain one, returns a
    neutral modifier rather than fabricating position information from 2PA.

    The modifier is RELATIVE to the offense's own overall blocked-attempt rate,
    so it does not double count the overall opponent-BLK-suffered factor.
    """
    if player_df is None or player_df.empty or "POSITION_GROUP" not in player_df.columns:
        return 1.0, pd.DataFrame([{"Available": False, "Reason": "No player/position data"}])

    ba_col = next((c for c in ["BLKA", "BA", "BLOCKS_AGAINST", "BLOCKED_ATTEMPTS"] if c in player_df.columns), None)
    if ba_col is None:
        return 1.0, pd.DataFrame([{
            "Available": False,
            "Reason": "No player-level blocked-attempt field in loaded data; positional BLK modifier kept neutral",
        }])

    x = player_df.copy()
    x[ba_col] = pd.to_numeric(x[ba_col], errors="coerce").fillna(0.0)
    x["FGA"] = pd.to_numeric(x["FGA"], errors="coerce").fillna(0.0)
    league = x[x["FGA"] > 0].copy()
    team = league[league["TEAM_ABBR"].astype(str).str.upper().eq(str(offense_abbr).upper())].copy()
    if team.empty:
        return 1.0, pd.DataFrame([{"Available": False, "Reason": "No offense rows for positional BLK"}])

    out_norm = {str(v).casefold() for v in (out_players or [])}
    # Current shot mix: recent FGA of active current players by broad position.
    current_mix = team.copy()
    if current_pool is not None and not current_pool.empty:
        active_names = set(
            current_pool[
                current_pool["TEAM_ABBR"].astype(str).str.upper().eq(str(offense_abbr).upper())
            ]["PLAYER_NAME"].astype(str).str.casefold().tolist()
        ) - out_norm
        current_mix = current_mix[current_mix["PLAYER_NAME"].astype(str).str.casefold().isin(active_names)].copy()
    current_mix = current_mix.sort_values("GAME_DATE").groupby("PLAYER_ID", group_keys=False).tail(5)

    rows = []
    weighted_ratio = 0.0
    weight_total = 0.0
    for pos in ["G", "F", "C"]:
        lp = league[league["POSITION_GROUP"].astype(str).eq(pos)]
        tp = team[team["POSITION_GROUP"].astype(str).eq(pos)]
        mp = current_mix[current_mix["POSITION_GROUP"].astype(str).eq(pos)]
        l_rate = _safe_div(lp[ba_col].sum(), lp["FGA"].sum(), np.nan)
        t_rate = _safe_div(tp[ba_col].sum(), tp["FGA"].sum(), np.nan)
        shot_weight = float(mp["FGA"].sum())
        ratio = (t_rate / l_rate) if np.isfinite(t_rate) and np.isfinite(l_rate) and l_rate > 0 else np.nan
        if np.isfinite(ratio) and shot_weight > 0:
            weighted_ratio += shot_weight * ratio
            weight_total += shot_weight
        rows.append({
            "Position": pos,
            "Team BLKA/FGA": t_rate,
            "League BLKA/FGA": l_rate,
            "Raw position ratio": ratio,
            "Current active recent FGA weight": shot_weight,
        })

    if weight_total <= 0:
        return 1.0, pd.DataFrame(rows)

    pos_ratio = weighted_ratio / weight_total
    overall_team = _safe_div(team[ba_col].sum(), team["FGA"].sum(), np.nan)
    overall_lg = _safe_div(league[ba_col].sum(), league["FGA"].sum(), np.nan)
    overall_ratio = (overall_team / overall_lg) if np.isfinite(overall_team) and np.isfinite(overall_lg) and overall_lg > 0 else 1.0
    relative = float(np.clip(pos_ratio / max(overall_ratio, 1e-9), 0.85, 1.15))
    mod = float(np.clip(relative ** 0.20, 0.97, 1.03))
    audit = pd.DataFrame(rows)
    audit.attrs["summary"] = {
        "Available": True,
        "Weighted position ratio": pos_ratio,
        "Overall offense ratio": overall_ratio,
        "Relative position ratio": relative,
        "Applied modifier": mod,
    }
    return mod, audit
