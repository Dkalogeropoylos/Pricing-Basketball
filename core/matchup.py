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
    return {
        "ratios": ratios,
        "modifiers": auto,
        "rates": {"league": lg, "opponent": opp},
        "audit": pd.DataFrame(rows),
    }


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
    out["2PA"] = a2 / mins * 36.0
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
    mapping = {"PTS": "PTS", "REB": "REB", "AST": "AST", "3PA": "3PA", "2PA": "2PA", "FTA": "FTA"}
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



def player_h2h_modifiers(
    player_log: pd.DataFrame,
    opponent_abbr: str,
    profile: dict,
    projected_minutes: float,
    rotation_similarity: float = 1.0,
    max_weight: float = 0.05,
):
    """Tiny, rotation/minute-aware same-season H2H opportunity correction.

    H2H is deliberately NOT a second sample and never changes shooting skill.
    It only nudges 2PA/3PA/REB/AST opportunity rates in log space. The maximum
    blend weight is 5%, and it shrinks further for sparse games, dissimilar
    rotations and H2Hs played at very different minutes.
    """
    neutral = {"2PA": 1.0, "3PA": 1.0, "REB": 1.0, "AST": 1.0}
    if player_log is None or player_log.empty:
        return neutral, pd.DataFrame()
    h = player_log[
        player_log["OPP_ABBR"].astype(str).str.upper().eq(str(opponent_abbr).upper())
    ].copy()
    if h.empty:
        return neutral, pd.DataFrame([{"H2H games": 0, "Applied H2H weight": 0.0}])

    mins = pd.to_numeric(h.get("MIN", 0), errors="coerce").fillna(0.0)
    h = h[mins > 0].copy()
    if h.empty:
        return neutral, pd.DataFrame([{"H2H games": 0, "Applied H2H weight": 0.0}])
    mins = pd.to_numeric(h["MIN"], errors="coerce").fillna(0.0)
    total_min = float(mins.sum())
    if total_min <= 0:
        return neutral, pd.DataFrame([{"H2H games": 0, "Applied H2H weight": 0.0}])

    fga = float(pd.to_numeric(h.get("FGA", 0), errors="coerce").fillna(0.0).sum())
    a3 = float(pd.to_numeric(h.get("FG3A", 0), errors="coerce").fillna(0.0).sum())
    a2 = max(fga - a3, 0.0)
    rates = {
        "2PA": a2 / total_min,
        "3PA": a3 / total_min,
        "REB": float(pd.to_numeric(h.get("REB", 0), errors="coerce").fillna(0.0).sum()) / total_min,
        "AST": float(pd.to_numeric(h.get("AST", 0), errors="coerce").fillna(0.0).sum()) / total_min,
    }
    base = {
        "2PA": float(profile.get("two_pa_pm", np.nan)),
        "3PA": float(profile.get("three_pa_pm", np.nan)),
        "REB": float(profile.get("reb_pm", np.nan)),
        "AST": float(profile.get("ast_pm", np.nan)),
    }
    n = int(len(h))
    rot = float(np.clip(rotation_similarity, 0.0, 1.0))
    minute_sim = float(np.mean(np.exp(-np.abs(mins.to_numpy(dtype=float) - float(projected_minutes)) / 10.0)))
    maturity = float(n / (n + 2.0))
    weight = float(min(float(max_weight), float(max_weight) * maturity * rot * minute_sim))

    out = dict(neutral)
    rows = []
    for stat in ("2PA", "3PA", "REB", "AST"):
        b = base[stat]
        hv = rates[stat]
        ratio = (hv / b) if np.isfinite(b) and b > 0 and np.isfinite(hv) else 1.0
        ratio = float(np.clip(ratio, 0.70, 1.30))
        mod = float(np.exp(weight * np.log(max(ratio, 1e-9)))) if weight > 0 else 1.0
        out[stat] = float(np.clip(mod, 0.97, 1.03))
        rows.append({
            "Stat": stat,
            "H2H games": n,
            "H2H rate/min": hv,
            "Baseline rate/min": b,
            "Raw ratio": ratio,
            "Rotation similarity": rot,
            "Minute similarity": minute_sim,
            "Applied H2H weight": weight,
            "Final H2H modifier": out[stat],
        })
    return out, pd.DataFrame(rows)

def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-5, 1 - 1e-5))
    return float(np.log(p / (1.0 - p)))


def _inv_logit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def _shrink_to_league(obs: float, league: float, games: float, k: float = 10.0) -> float:
    if not (np.isfinite(obs) and np.isfinite(league)):
        return float(league if np.isfinite(league) else obs)
    c = float(np.clip(float(games) / (float(games) + float(k)), 0.0, 1.0))
    return float(c * obs + (1.0 - c) * league)


def team_matchup_modifiers(overall_profile, own_profile=None, elasticities=None):
    """Opponent adjustment for Team Markets using league-calibrated elasticity.

    Conceptually:
        current offense identity
        + beta_stat * (opponent allowed - league average)

    where beta_stat is learned from historical pregame team-games by
    ``fit_opponent_elasticities``. Shares/percentages are adjusted in logit
    space; positive rates are adjusted in log space. This keeps opponent
    information meaningful without treating the opponent's raw allowed average
    as a second full sample or inventing a different hand-tuned exponent for
    every matchup.
    """
    mods = overall_profile.get("modifiers", {})
    out = {
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
    if not own_profile:
        return out

    elasticities = elasticities or {}
    # If the empirical calibrator is unavailable, use only the same small
    # structural floors used by fit_opponent_elasticities -- never the old
    # hand-tuned 30-55% response coefficients.
    defaults = {
        "FGA_LIVE": 0.04,
        "3P_SHARE": 0.08,
        "FTA": 0.08,
        "TOV": 0.08,
        "OREB_PER_MISS": 0.08,
        "AST_PER_MAKE": 0.06,
        "PF": 0.06,
        "3P_PCT": 0.03,
        "2P_PCT": 0.03,
    }
    rates = overall_profile.get("rates", {}) or {}
    lg = rates.get("league", {}) or {}
    opp = rates.get("opponent", {}) or {}
    games = float(opp.get("games", 0) or 0)

    def beta(name):
        return float(elasticities.get(name, defaults[name]))

    def rate_modifier(feature, own_key, lg_key=None, opp_key=None, bounds=(0.85, 1.15), k=10.0):
        lk = lg_key or feature
        ok = opp_key or feature
        own = float(own_profile.get(own_key, np.nan))
        lv = float(lg.get(lk, np.nan))
        ov = float(opp.get(ok, np.nan))
        if not (np.isfinite(own) and own > 0 and np.isfinite(lv) and lv > 0 and np.isfinite(ov) and ov > 0):
            return None
        ovs = _shrink_to_league(ov, lv, games, k=k)
        target = float(np.exp(np.log(own) + beta(feature) * (np.log(ovs) - np.log(lv))))
        return float(np.clip(target / own, bounds[0], bounds[1]))

    def prob_modifier(feature, own_key, lg_key, opp_key, bounds=(0.94, 1.06), k=14.0):
        own = float(own_profile.get(own_key, np.nan))
        lv = float(lg.get(lg_key, np.nan))
        ov = float(opp.get(opp_key, np.nan))
        if not (np.isfinite(own) and 0 < own < 1 and np.isfinite(lv) and 0 < lv < 1 and np.isfinite(ov) and 0 < ov < 1):
            return None
        ovs = _shrink_to_league(ov, lv, games, k=k)
        target = _inv_logit(_logit(own) + beta(feature) * (_logit(ovs) - _logit(lv)))
        return float(np.clip(target / own, bounds[0], bounds[1]))

    replacements = {
        "FGA": rate_modifier("FGA_LIVE", "fga_live", bounds=(0.94, 1.06)),
        "FTA": rate_modifier("FTA", "fta_pp", bounds=(0.84, 1.16)),
        "TOV": rate_modifier("TOV", "tov_pp", bounds=(0.86, 1.14)),
        "OREB": rate_modifier("OREB_PER_MISS", "oreb_per_miss", bounds=(0.86, 1.14)),
        "AST": rate_modifier("AST_PER_MAKE", "assist_per_make", bounds=(0.88, 1.12)),
        "PF": rate_modifier("PF", "pf_pp", bounds=(0.86, 1.14)),
        "3P_SHARE": prob_modifier("3P_SHARE", "three_share", "3P_SHARE", "3P_SHARE", bounds=(0.84, 1.16), k=10.0),
        # Shooting efficiency remains the most strongly shrunk opponent layer.
        "3P_PCT": prob_modifier("3P_PCT", "three_pct", "3P_PCT", "3P_PCT", bounds=(0.94, 1.06), k=18.0),
        "2P_PCT": prob_modifier("2P_PCT", "two_pct", "2P_PCT", "2P_PCT", bounds=(0.94, 1.06), k=18.0),
    }
    for key, value in replacements.items():
        if value is not None and np.isfinite(value):
            out[key] = float(value)

    # BLK keeps the v3 own-ability/opponent-suffered logic that has already
    # calibrated well in audits; do not overwrite it with the generic learner.
    return out


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


# ---------------------------------------------------------------------
# v2.13 empirical opponent-elasticity calibration
# ---------------------------------------------------------------------
def _single_game_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one-row-per-team-game structural features used by Team Markets."""
    x = df.copy()
    for c in ["FGA","FGM","FG3A","FG3M","FTA","TOV","OREB","AST","PF"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    x["POSS_CAL"] = estimate_possessions(x).clip(lower=1e-6)
    live = (x["POSS_CAL"] - x["TOV"]).clip(lower=1e-6)
    misses = (x["FGA"] - x["FGM"]).clip(lower=1e-6)
    a2 = (x["FGA"] - x["FG3A"]).clip(lower=1e-6)
    m2 = (x["FGM"] - x["FG3M"]).clip(lower=0.0)
    out = pd.DataFrame({
        "GAME_DATE": pd.to_datetime(x["GAME_DATE"], errors="coerce"),
        "GAME_ID": x["GAME_ID"].astype(str),
        "TEAM_ABBR": x["TEAM_ABBR"].astype(str),
        "OPP_ABBR": x["OPP_ABBR"].astype(str),
        "FGA_LIVE": x["FGA"] / live,
        "3P_SHARE": x["FG3A"] / x["FGA"].replace(0, np.nan),
        "FTA": x["FTA"] / x["POSS_CAL"],
        "TOV": x["TOV"] / x["POSS_CAL"],
        "OREB_PER_MISS": x["OREB"] / misses,
        "AST_PER_MAKE": x["AST"] / x["FGM"].replace(0, np.nan),
        "PF": x["PF"] / x["POSS_CAL"],
        "3P_PCT": x["FG3M"] / x["FG3A"].replace(0, np.nan),
        "2P_PCT": m2 / a2,
    })
    return out.replace([np.inf, -np.inf], np.nan)


def _transform_feature(name: str, s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    if name in {"3P_SHARE", "3P_PCT", "2P_PCT"}:
        v = v.clip(1e-4, 1 - 1e-4)
        return np.log(v / (1.0 - v))
    return np.log(v.clip(lower=1e-4))


def fit_opponent_elasticities(league_team_logs: pd.DataFrame):
    """Estimate stat-specific opponent response from historical WNBA team-games.

    There is deliberately NO universal "opponent weight".  For every derivative
    feature we build a pregame own baseline, a pregame opponent-allowed baseline
    and a pregame league baseline.  The coefficient is the predictive slope of
    the realized deviation from the team's own identity on the opponent's
    deviation from league average.

    v2.14 changes two things versus v2.13:
      * strong hand-picked priors (e.g. 0.55 for 3P share) are removed;
      * the learned slope is shrunk by both sample size and its statistical
        signal, with only a SMALL non-zero structural floor.  Thus opponent
        context always matters a little, but it only matters a lot when WNBA
        history actually supports it.

    Shares / shooting percentages are learned in logit space. Positive rates are
    learned in log space. The model remains league-level rather than team-specific
    because ~30-40 games are far too noisy for a separate elasticity per team.
    """
    if league_team_logs is None or league_team_logs.empty:
        return {}, pd.DataFrame()

    x = _single_game_structural_features(league_team_logs)
    x = x.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBR"]).reset_index(drop=True)
    features = [
        "FGA_LIVE", "3P_SHARE", "FTA", "TOV", "OREB_PER_MISS",
        "AST_PER_MAKE", "PF", "3P_PCT", "2P_PCT",
    ]

    # Small structural floors only. These are NOT target weights; they simply
    # prevent a noisy early-season fit from pretending the opponent is irrelevant.
    floors = {
        "FGA_LIVE": 0.04,
        "3P_SHARE": 0.08,
        "FTA": 0.08,
        "TOV": 0.08,
        "OREB_PER_MISS": 0.08,
        "AST_PER_MAKE": 0.06,
        "PF": 0.06,
        "3P_PCT": 0.03,
        "2P_PCT": 0.03,
    }
    caps = {
        "FGA_LIVE": 0.55,
        "3P_SHARE": 0.95,
        "FTA": 0.85,
        "TOV": 0.75,
        "OREB_PER_MISS": 0.75,
        "AST_PER_MAKE": 0.65,
        "PF": 0.70,
        # Opponent FG% is real, but raw opponent shooting percentage is noisy.
        "3P_PCT": 0.35,
        "2P_PCT": 0.35,
    }

    result, audit = {}, []
    for feat in features:
        tmp = x[["GAME_DATE", "GAME_ID", "TEAM_ABBR", "OPP_ABBR", feat]].copy()

        # Pregame own-offense mean.
        tmp["OWN_PRIOR"] = (
            tmp.groupby("TEAM_ABBR")[feat]
            .transform(lambda z: z.expanding(min_periods=5).mean().shift(1))
        )
        tmp["OWN_N"] = tmp.groupby("TEAM_ABBR").cumcount()

        # Pregame defensive allowed mean. Rows with OPP_ABBR=D are exactly the
        # offensive outcomes previously allowed by defense D.
        tmp["DEF_PRIOR"] = (
            tmp.groupby("OPP_ABBR")[feat]
            .transform(lambda z: z.expanding(min_periods=5).mean().shift(1))
        )
        tmp["DEF_N"] = tmp.groupby("OPP_ABBR").cumcount()

        # Pregame league baseline by GAME, preventing the other row of the same
        # game from leaking into the prior.
        game_means = (
            tmp.groupby(["GAME_DATE", "GAME_ID"], sort=True)[feat]
            .mean().reset_index().sort_values(["GAME_DATE", "GAME_ID"])
        )
        game_means["LG_PRIOR"] = game_means[feat].expanding(min_periods=10).mean().shift(1)
        tmp = tmp.merge(
            game_means[["GAME_DATE", "GAME_ID", "LG_PRIOR"]],
            on=["GAME_DATE", "GAME_ID"], how="left",
        )
        tmp = tmp.dropna(subset=[feat, "OWN_PRIOR", "DEF_PRIOR", "LG_PRIOR"])
        tmp = tmp[(tmp["OWN_N"] >= 5) & (tmp["DEF_N"] >= 5)].copy()

        floor = floors[feat]
        cap = caps[feat]
        raw = np.nan
        se_beta = np.nan
        base_rmse = adj_rmse = np.nan
        gain = 0.0
        signal_conf = 0.0
        sample_conf = 0.0

        if len(tmp) < 60:
            beta = floor
        else:
            y_actual = _transform_feature(feat, tmp[feat])
            own_t = _transform_feature(feat, tmp["OWN_PRIOR"])
            def_t = _transform_feature(feat, tmp["DEF_PRIOR"])
            lg_t = _transform_feature(feat, tmp["LG_PRIOR"])

            xv = (def_t - lg_t).clip(-0.75, 0.75).to_numpy(dtype=float)
            yv = (y_actual - own_t).clip(-0.90, 0.90).to_numpy(dtype=float)
            good = np.isfinite(xv) & np.isfinite(yv)
            xv, yv = xv[good], yv[good]

            denom = float(np.sum(xv * xv))
            if denom <= 1e-9 or len(xv) < 40:
                beta = floor
            else:
                raw_unclipped = float(np.sum(xv * yv) / denom)
                raw = float(np.clip(raw_unclipped, 0.0, 1.25))

                residual = yv - raw * xv
                dof = max(len(xv) - 1, 1)
                sigma2 = float(np.sum(residual * residual) / dof)
                se_beta = float(np.sqrt(max(sigma2 / denom, 0.0)))

                # How believable is the sign/magnitude of the slope?
                signal_conf = float(
                    (raw * raw) / (raw * raw + se_beta * se_beta + 1e-9)
                )
                sample_conf = float(len(xv) / (len(xv) + 120.0))

                beta = floor + (raw - floor) * signal_conf * sample_conf
                beta = float(np.clip(beta, floor, cap))

                base_rmse = float(np.sqrt(np.mean(yv ** 2))) if len(yv) else np.nan
                adj_rmse = float(np.sqrt(np.mean((yv - beta * xv) ** 2))) if len(yv) else np.nan
                if np.isfinite(base_rmse) and base_rmse > 0 and np.isfinite(adj_rmse):
                    gain = float((base_rmse - adj_rmse) / base_rmse)
                    # If the opponent layer does not improve historical prediction,
                    # do not let noisy correlation create a large adjustment.
                    if gain <= 0:
                        beta = floor
                        adj_rmse = float(np.sqrt(np.mean((yv - beta * xv) ** 2)))
                        gain = float((base_rmse - adj_rmse) / base_rmse)

        result[feat] = float(beta)
        audit.append({
            "Feature": feat,
            "Learned elasticity": float(beta),
            "Raw slope": raw,
            "Slope SE": se_beta,
            "Signal confidence": signal_conf,
            "Sample confidence": sample_conf,
            "Structural floor": floor,
            "Calibration rows": int(len(tmp)),
            "Base transformed RMSE": base_rmse,
            "Opponent-adjusted RMSE": adj_rmse,
            "Relative RMSE gain": gain,
        })

    return result, pd.DataFrame(audit)
