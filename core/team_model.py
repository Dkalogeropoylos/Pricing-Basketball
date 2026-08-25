from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.buckets import (
    WeightConfig,
    split_non_overlapping,
    active_weights,
    weighted_average_feature,
)


@dataclass
class TeamContext:
    projected_possessions: float
    possessions_sd: float = 3.0

    # Offense-vs-opponent interaction multipliers.
    # 1.00 = neutral. These are already shrinked contextual adjustments,
    # not extra samples.
    three_pa: float = 1.0
    three_pct: float = 1.0
    two_pa: float = 1.0
    two_pct: float = 1.0
    fta: float = 1.0
    tov: float = 1.0
    oreb: float = 1.0
    ast: float = 1.0
    pf: float = 1.0

    # Defensive opportunity conversion. In the coupled game simulator these
    # are applied to the OPPONENT'S turnovers / misses / 2P misses.
    dreb: float = 1.0
    stl: float = 1.0
    blk: float = 1.0


def estimate_possessions(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df["FGA"], errors="coerce").fillna(0)
        - pd.to_numeric(df["OREB"], errors="coerce").fillna(0)
        + pd.to_numeric(df["TOV"], errors="coerce").fillna(0)
        + 0.44 * pd.to_numeric(df["FTA"], errors="coerce").fillna(0)
    )


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a) / float(b) if np.isfinite(b) and b > 0 else default


def _weighted_sum(series: pd.Series, weights: np.ndarray) -> float:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return float(np.sum(values * weights))


def _row_weights(df: pd.DataFrame, game_weights: Optional[Dict[str, float]]) -> np.ndarray:
    if not game_weights or "GAME_ID" not in df.columns:
        return np.ones(len(df), dtype=float)
    return np.asarray(
        [float(game_weights.get(str(gid), 1.0)) for gid in df["GAME_ID"]],
        dtype=float,
    )


def _paired_opponent_rows(
    own_df: pd.DataFrame,
    league_team_logs: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Attach the opponent box-score row for each own-team game."""
    if league_team_logs is None or own_df.empty or "GAME_ID" not in own_df.columns:
        return pd.DataFrame()

    need = [
        "GAME_ID", "TEAM_ABBR", "FGM", "FGA", "FG3M", "FG3A",
        "OREB", "DREB", "REB", "TOV", "FTA", "PF",
    ]
    available = [c for c in need if c in league_team_logs.columns]
    if "GAME_ID" not in available or "TEAM_ABBR" not in available:
        return pd.DataFrame()

    opp = league_team_logs[available].copy()
    rename = {c: f"OPP_{c}" for c in available if c != "GAME_ID"}
    opp = opp.rename(columns=rename)

    own_cols = [c for c in ["GAME_ID", "TEAM_ABBR", "OPP_ABBR", "DREB", "STL", "BLK"] if c in own_df.columns]
    paired = own_df[own_cols].copy().merge(opp, on="GAME_ID", how="left")

    if "OPP_ABBR" in paired.columns and "OPP_TEAM_ABBR" in paired.columns:
        paired = paired[
            paired["OPP_ABBR"].astype(str).str.upper()
            == paired["OPP_TEAM_ABBR"].astype(str).str.upper()
        ].copy()
    elif "TEAM_ABBR" in paired.columns and "OPP_TEAM_ABBR" in paired.columns:
        paired = paired[
            paired["TEAM_ABBR"].astype(str).str.upper()
            != paired["OPP_TEAM_ABBR"].astype(str).str.upper()
        ].copy()

    return paired.drop_duplicates(subset=["GAME_ID"]).reset_index(drop=True)


def _feat(
    df: pd.DataFrame,
    league_team_logs: Optional[pd.DataFrame] = None,
    game_weights: Optional[Dict[str, float]] = None,
) -> dict:
    if df.empty:
        return {}

    x = df.copy()
    w = _row_weights(x, game_weights)
    poss_rows = estimate_possessions(x).to_numpy(dtype=float)
    poss = float(np.sum(poss_rows * w))

    fga = _weighted_sum(x["FGA"], w)
    fgm = _weighted_sum(x["FGM"], w)
    a3 = _weighted_sum(x["FG3A"], w)
    m3 = _weighted_sum(x["FG3M"], w)
    a2 = max(fga - a3, 0.0)
    m2 = max(fgm - m3, 0.0)
    fta = _weighted_sum(x["FTA"], w)
    ftm = _weighted_sum(x["FTM"], w)
    tov = _weighted_sum(x["TOV"], w)
    oreb = _weighted_sum(x["OREB"], w)
    dreb = _weighted_sum(x["DREB"], w) if "DREB" in x.columns else 0.0
    misses = max(fga - fgm, 0.0)
    live_poss = max(poss - tov, 1e-9)

    out = {
        "games": int(len(x)),
        "effective_games": float(np.sum(w)),
        "poss_pg": float(np.average(poss_rows, weights=w)) if len(w) else np.nan,
        "three_pa_pp": _safe_div(a3, poss),
        "two_pa_pp": _safe_div(a2, poss),
        "three_pa_live": _safe_div(a3, live_poss),
        "two_pa_live": _safe_div(a2, live_poss),
        "fta_pp": _safe_div(fta, poss),
        "tov_pp": _safe_div(tov, poss),
        "oreb_pp": _safe_div(oreb, poss),
        "dreb_pp": _safe_div(dreb, poss),
        "oreb_per_miss": _safe_div(oreb, misses),
        "ast_pp": _safe_div(_weighted_sum(x["AST"], w), poss),
        "stl_pp": _safe_div(_weighted_sum(x["STL"], w), poss),
        "blk_pp": _safe_div(_weighted_sum(x["BLK"], w), poss),
        "pf_pp": _safe_div(_weighted_sum(x["PF"], w), poss),
        "three_pct": _safe_div(m3, a3, np.nan),
        "two_pct": _safe_div(m2, a2, np.nan),
        "ft_pct": _safe_div(ftm, fta, np.nan),
        "three_att": a3,
        "two_att": a2,
        "ft_att": fta,
    }

    # Defensive stats must be driven by OPPONENT opportunities, never the
    # team's own TOV/2PA/misses. Attach the paired opponent row for each game.
    paired = _paired_opponent_rows(x, league_team_logs)
    if not paired.empty:
        # Recreate the same inner rotation-similarity weights on the paired rows.
        pw = _row_weights(paired, game_weights)
        opp_fga = _weighted_sum(paired["OPP_FGA"], pw)
        opp_fgm = _weighted_sum(paired["OPP_FGM"], pw)
        opp_a3 = _weighted_sum(paired["OPP_FG3A"], pw)
        opp_m3 = _weighted_sum(paired["OPP_FG3M"], pw)
        opp_a2 = max(opp_fga - opp_a3, 0.0)
        opp_m2 = max(opp_fgm - opp_m3, 0.0)
        opp_misses = max(opp_fga - opp_fgm, 0.0)
        opp_oreb = _weighted_sum(paired["OPP_OREB"], pw)
        opp_tov = _weighted_sum(paired["OPP_TOV"], pw)
        opp_2miss = max(opp_a2 - opp_m2, 0.0)

        own_dreb = _weighted_sum(paired["DREB"], pw)
        own_stl = _weighted_sum(paired["STL"], pw)
        own_blk = _weighted_sum(paired["BLK"], pw)

        # DREB is conditional on an opponent miss that was NOT recovered by
        # the offense. This prevents OREB + DREB double counting.
        dreb_chances = max(opp_misses - opp_oreb, 0.0)
        out["dreb_capture"] = _safe_div(own_dreb, dreb_chances, 0.94)

        # A steal is a subset of opponent turnovers.
        out["stl_per_opp_tov"] = _safe_div(own_stl, opp_tov, 0.55)

        # Learn block ability per opponent 2PA. Using opponent misses in the
        # denominator confounds rim protection with opponent shooting luck.
        out["blk_per_opp_2pa"] = _safe_div(own_blk, opp_a2, 0.075)
        out["opp_2pa_att"] = opp_a2
    else:
        # Conservative fallbacks only if paired game rows are unavailable.
        out["dreb_capture"] = 0.94
        out["stl_per_opp_tov"] = 0.55
        out["blk_per_opp_2pa"] = _safe_div(out["blk_pp"], max(out["two_pa_pp"], 0.05), 0.075)
        out["opp_2pa_att"] = out["two_att"]

    # Useful structural conversion rate for AST simulation.
    made_fg_pp = out["three_pa_pp"] * (out["three_pct"] if np.isfinite(out["three_pct"]) else 0.34) \
        + out["two_pa_pp"] * (out["two_pct"] if np.isfinite(out["two_pct"]) else 0.51)
    out["assist_per_make"] = _safe_div(out["ast_pp"], max(made_fg_pp, 0.05), 0.62)

    return out


def _shrink(obs: float, attempts: float, prior: float, prior_attempts: float) -> float:
    if not np.isfinite(obs):
        obs, attempts = prior, 0.0
    return float((obs * attempts + prior * prior_attempts) / (attempts + prior_attempts))


def build_team_profile(
    df: pd.DataFrame,
    cfg: WeightConfig,
    league_team_logs: Optional[pd.DataFrame] = None,
    game_weights: Optional[Dict[str, float]] = None,
) -> Tuple[dict, pd.DataFrame]:
    """
    Team profile with:
      - non-overlapping Old / G6-10 / L5 outer weights
      - optional current-rotation similarity INSIDE each bucket
      - opportunity rates from the weighted buckets
      - shooting ability from a larger-sample shrinkage model
      - defensive conversion rates tied to opponent opportunities
    """
    x = df.sort_values("GAME_DATE").copy()
    buckets = split_non_overlapping(x)
    weights = active_weights(buckets, cfg)
    feats = {
        k: _feat(v, league_team_logs=league_team_logs, game_weights=game_weights)
        for k, v in buckets.items()
    }

    p = {}
    for key in [
        "poss_pg",
        "three_pa_pp", "two_pa_pp", "three_pa_live", "two_pa_live",
        "fta_pp", "tov_pp", "oreb_pp", "dreb_pp", "oreb_per_miss",
        "ast_pp", "stl_pp", "blk_pp", "pf_pp",
        "assist_per_make", "dreb_capture", "stl_per_opp_tov",
        "blk_per_opp_2pa",
    ]:
        p[key] = weighted_average_feature(feats, weights, key)

    full = _feat(x, league_team_logs=league_team_logs, game_weights=None)

    # Same philosophy as the player engine: recent shooting is NOT accepted as
    # raw future truth. Use the larger sample and regress by actual attempts.
    p["three_pct"] = _shrink(
        full.get("three_pct", np.nan), full.get("three_att", 0.0), 0.340, 45.0
    )
    p["two_pct"] = _shrink(
        full.get("two_pct", np.nan), full.get("two_att", 0.0), 0.510, 70.0
    )
    p["ft_pct"] = _shrink(
        full.get("ft_pct", np.nan), full.get("ft_att", 0.0), 0.785, 40.0
    )

    # Defensive rates need sensible bounds after weighted blending.
    p["dreb_capture"] = float(np.clip(p.get("dreb_capture", 0.94), 0.78, 0.995))
    p["stl_per_opp_tov"] = float(np.clip(p.get("stl_per_opp_tov", 0.55), 0.20, 0.90))
    recent_blk = float(p.get("blk_per_opp_2pa", 0.075))
    stable_blk = _shrink(
        full.get("blk_per_opp_2pa", np.nan),
        full.get("opp_2pa_att", 0.0),
        0.075,
        120.0,
    )
    # Blocks are high-variance. Keep some role/recency signal but anchor the rate
    # heavily to the larger sample rather than opponent missed-shot percentage.
    p["blk_per_opp_2pa"] = float(np.clip(
        0.35 * recent_blk + 0.65 * stable_blk, 0.02, 0.18
    ))
    p["assist_per_make"] = float(np.clip(p.get("assist_per_make", 0.62), 0.25, 0.92))

    audit = []
    for k in ("old", "mid", "l5"):
        audit.append({"bucket": k, "weight": weights[k], **feats.get(k, {"games": 0})})
    return p, pd.DataFrame(audit)


def _home_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None

    if "IS_HOME" in df.columns:
        s = df["IS_HOME"]
        if s.dtype == bool:
            return s
        raw = s.astype(str).str.strip().str.lower()
        return raw.isin(["1", "true", "yes", "home", "h"])

    for col in ["HOME_AWAY", "LOCATION", "VENUE"]:
        if col in df.columns:
            raw = df[col].astype(str).str.strip().str.lower()
            return raw.str.startswith("h") | raw.eq("home")

    if "MATCHUP" in df.columns:
        raw = df["MATCHUP"].astype(str)
        return raw.str.contains(r"\bvs\.?\b", case=False, regex=True)

    return None


def team_location_modifiers(
    team_log: pd.DataFrame,
    is_home: bool,
    league_team_logs: Optional[pd.DataFrame] = None,
    min_games: int = 5,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Small home/away correction, explicitly shrinked so the location split does
    not become a second full sample. The non-overlapping profile remains the
    primary signal.
    """
    neutral = {
        "3PA": 1.0, "2PA": 1.0, "FTA": 1.0, "TOV": 1.0,
        "OREB": 1.0, "AST": 1.0, "PF": 1.0,
        "DREB": 1.0, "STL": 1.0, "BLK": 1.0,
    }
    mask = _home_mask(team_log)
    if mask is None:
        return neutral, pd.DataFrame([{"Location": "unavailable", "Games": 0}])

    split = team_log[mask if is_home else ~mask].copy()
    if len(split) < min_games:
        return neutral, pd.DataFrame([{
            "Location": "home" if is_home else "away",
            "Games": int(len(split)),
            "Note": f"< {min_games} games; neutral location modifier",
        }])

    base = _feat(team_log, league_team_logs=league_team_logs)
    loc = _feat(split, league_team_logs=league_team_logs)
    mapping = {
        "3PA": "three_pa_live",
        "2PA": "two_pa_live",
        "FTA": "fta_pp",
        "TOV": "tov_pp",
        "OREB": "oreb_per_miss",
        "AST": "assist_per_make",
        "PF": "pf_pp",
        "DREB": "dreb_capture",
        "STL": "stl_per_opp_tov",
        "BLK": "blk_per_opp_2pa",
    }
    rows = []
    out = dict(neutral)
    for market, key in mapping.items():
        b = base.get(key, np.nan)
        l = loc.get(key, np.nan)
        raw_ratio = l / b if np.isfinite(b) and b > 0 and np.isfinite(l) else 1.0
        raw_ratio = float(np.clip(raw_ratio, 0.75, 1.25))
        # Only 20% of the split deviation is used, capped at +/-6%.
        mod = float(np.clip(raw_ratio ** 0.20, 0.94, 1.06))
        out[market] = mod
        rows.append({
            "Stat": market,
            "Overall": b,
            "Location": l,
            "Raw ratio": raw_ratio,
            "Applied modifier": mod,
            "Games": int(len(split)),
        })
    return out, pd.DataFrame(rows)


def h2h_team_audit(
    league_team_logs: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
) -> pd.DataFrame:
    """Same-season H2H audit only. No extra numerical weight is applied."""
    x = league_team_logs.copy()
    mask = (
        x["TEAM_ABBR"].astype(str).str.upper().isin([str(home_abbr).upper(), str(away_abbr).upper()])
        & x["OPP_ABBR"].astype(str).str.upper().isin([str(home_abbr).upper(), str(away_abbr).upper()])
    )
    h = x[mask].copy()
    if h.empty:
        return pd.DataFrame()
    cols = [
        c for c in [
            "GAME_DATE", "GAME_ID", "TEAM_ABBR", "OPP_ABBR", "PTS", "FGA",
            "FG3A", "FG3M", "FTA", "FTM", "OREB", "DREB", "REB", "AST",
            "STL", "BLK", "TOV", "PF",
        ] if c in h.columns
    ]
    return h[cols].sort_values(["GAME_DATE", "TEAM_ABBR"], ascending=[False, True]).reset_index(drop=True)


def _simulate_offense(
    profile: dict,
    ctx: TeamContext,
    poss: np.ndarray,
    rng: np.random.Generator,
    z_style: np.ndarray,
    z_shoot: np.ndarray,
    z_foul: np.ndarray,
    z_tov: np.ndarray,
    z_reb: np.ndarray,
) -> Dict[str, np.ndarray]:
    poss_i = np.maximum(poss.astype(int), 1)

    tov_rate = np.clip(
        profile["tov_pp"] * ctx.tov * np.exp(0.08 * z_tov - 0.5 * 0.08**2),
        0.03, 0.30,
    )
    tov = rng.binomial(poss_i, tov_rate)
    live = np.maximum(poss_i - tov, 1)

    perimeter = np.exp(0.08 * z_style - 0.5 * 0.08**2)
    three_live = profile.get(
        "three_pa_live",
        profile["three_pa_pp"] / max(1.0 - profile["tov_pp"], 0.55),
    )
    two_live = profile.get(
        "two_pa_live",
        profile["two_pa_pp"] / max(1.0 - profile["tov_pp"], 0.55),
    )

    a3 = rng.poisson(np.clip(live * three_live * ctx.three_pa * perimeter, 0.001, None))
    a2 = rng.poisson(np.clip(live * two_live * ctx.two_pa / perimeter**0.35, 0.001, None))
    fta = rng.poisson(np.clip(
        poss * profile["fta_pp"] * ctx.fta * np.exp(0.12 * z_foul - 0.5 * 0.12**2),
        0.001, None,
    ))

    p3 = np.clip(profile["three_pct"] * ctx.three_pct + 0.03 * z_shoot, 0.10, 0.60)
    p2 = np.clip(profile["two_pct"] * ctx.two_pct + 0.025 * z_shoot, 0.25, 0.75)
    pft = np.clip(profile["ft_pct"] + 0.01 * z_shoot, 0.45, 0.98)

    m3 = rng.binomial(a3, p3)
    m2 = rng.binomial(a2, p2)
    ftm = rng.binomial(fta, pft)
    fgm = m3 + m2
    fga = a3 + a2
    pts = 3 * m3 + 2 * m2 + ftm

    misses3 = np.maximum(a3 - m3, 0)
    misses2 = np.maximum(a2 - m2, 0)
    misses = misses3 + misses2

    oreb_share = np.clip(
        profile.get("oreb_per_miss", 0.25)
        * ctx.oreb
        * np.exp(0.10 * z_reb - 0.5 * 0.10**2),
        0.08, 0.45,
    )
    oreb = rng.binomial(misses, oreb_share)

    assist_per_make = float(np.clip(profile.get("assist_per_make", 0.62), 0.25, 0.92))
    ast_prob = np.clip(assist_per_make * ctx.ast * (1.0 + 0.07 * z_shoot), 0.20, 0.95)
    ast = rng.binomial(fgm, ast_prob)

    pf = rng.poisson(np.clip(
        poss * profile["pf_pp"] * ctx.pf * np.exp(0.10 * z_foul - 0.5 * 0.10**2),
        0.001, None,
    ))

    return {
        "POSS": poss,
        "TOV": tov,
        "3PA": a3,
        "3PM": m3,
        "2PA": a2,
        "2PM": m2,
        "FGA": fga,
        "FGM": fgm,
        "FTA": fta,
        "FTM": ftm,
        "OREB": oreb,
        "AST": ast,
        "PF": pf,
        "PTS": pts,
        "MISSES": misses,
        "2MISS": misses2,
    }


def simulate_game(
    home_profile: dict,
    away_profile: dict,
    home_ctx: TeamContext,
    away_ctx: TeamContext,
    n: int = 100_000,
    seed: int = 3,
    opportunity_mult: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coupled two-team game simulation.

    Key conservation / dependency rules:
      - both teams share the same game-possession state
      - FGA = 2PA + 3PA
      - PTS = 2*2PM + 3*3PM + FTM
      - OREB is sampled from OWN misses
      - DREB is sampled from OPPONENT misses not already rebounded offensively
      - STL is a subset of OPPONENT turnovers
      - BLK rate is learned per OPPONENT 2PA, then capped by actual 2P misses
      - REB = OREB + DREB

    This removes the old same-team STL/TOV and BLK/2PA direction errors and
    enables coherent totals / "team with most" markets.
    """
    rng = np.random.default_rng(seed)

    central = 0.5 * (float(home_ctx.projected_possessions) + float(away_ctx.projected_possessions))
    sd = 0.5 * (float(home_ctx.possessions_sd) + float(away_ctx.possessions_sd))
    poss = np.rint(np.clip(rng.normal(central, sd, n), 55, 115) * opportunity_mult).astype(int)
    poss = np.maximum(poss, 1)

    # Shared game environments plus team-specific deviations.
    z_game_style = rng.normal(size=n)
    z_game_shoot = rng.normal(size=n)
    z_game_foul = rng.normal(size=n)
    z_game_tov = rng.normal(size=n)
    z_game_reb = rng.normal(size=n)

    def blend(shared, scale=0.70):
        return scale * shared + np.sqrt(max(1.0 - scale**2, 0.0)) * rng.normal(size=n)

    h = _simulate_offense(
        home_profile, home_ctx, poss, rng,
        blend(z_game_style), blend(z_game_shoot, 0.45), blend(z_game_foul),
        blend(z_game_tov, 0.55), blend(z_game_reb, 0.55),
    )
    a = _simulate_offense(
        away_profile, away_ctx, poss, rng,
        blend(z_game_style), blend(z_game_shoot, 0.45), blend(z_game_foul),
        blend(z_game_tov, 0.55), blend(z_game_reb, 0.55),
    )

    # Defensive rebound conversion from the OPPONENT'S remaining missed shots.
    h_dreb_chances = np.maximum(a["MISSES"] - a["OREB"], 0)
    a_dreb_chances = np.maximum(h["MISSES"] - h["OREB"], 0)

    h_dreb_p = np.clip(
        home_profile.get("dreb_capture", 0.94) * home_ctx.dreb
        * np.exp(0.035 * blend(z_game_reb, 0.50) - 0.5 * 0.035**2),
        0.72, 0.995,
    )
    a_dreb_p = np.clip(
        away_profile.get("dreb_capture", 0.94) * away_ctx.dreb
        * np.exp(0.035 * blend(z_game_reb, 0.50) - 0.5 * 0.035**2),
        0.72, 0.995,
    )
    h_dreb = rng.binomial(h_dreb_chances, h_dreb_p)
    a_dreb = rng.binomial(a_dreb_chances, a_dreb_p)

    # Steals are a subset of OPPONENT turnovers.
    h_stl_p = np.clip(
        home_profile.get("stl_per_opp_tov", 0.55) * home_ctx.stl
        * np.exp(0.05 * blend(z_game_tov, 0.45) - 0.5 * 0.05**2),
        0.10, 0.95,
    )
    a_stl_p = np.clip(
        away_profile.get("stl_per_opp_tov", 0.55) * away_ctx.stl
        * np.exp(0.05 * blend(z_game_tov, 0.45) - 0.5 * 0.05**2),
        0.10, 0.95,
    )
    h_stl = rng.binomial(a["TOV"], h_stl_p)
    a_stl = rng.binomial(h["TOV"], a_stl_p)

    # Blocks: estimate ability per opponent 2PA, not per opponent MISS.
    # The old miss-denominator confounded rim protection with opponent shooting luck.
    # Candidate blocks are generated from opponent 2PA and capped by actual misses.
    h_blk_p = np.clip(
        home_profile.get("blk_per_opp_2pa", 0.075) * home_ctx.blk,
        0.01, 0.22,
    )
    a_blk_p = np.clip(
        away_profile.get("blk_per_opp_2pa", 0.075) * away_ctx.blk,
        0.01, 0.22,
    )
    h_blk_candidate = rng.binomial(a["2PA"], h_blk_p)
    a_blk_candidate = rng.binomial(h["2PA"], a_blk_p)
    h_blk = np.minimum(h_blk_candidate, a["2MISS"])
    a_blk = np.minimum(a_blk_candidate, h["2MISS"])

    h["DREB"] = h_dreb
    a["DREB"] = a_dreb
    h["REB"] = h["OREB"] + h_dreb
    a["REB"] = a["OREB"] + a_dreb
    h["STL"] = h_stl
    a["STL"] = a_stl
    h["BLK"] = h_blk
    a["BLK"] = a_blk

    keep = [
        "POSS", "PTS", "FGM", "FGA", "3PM", "3PA", "2PM", "2PA",
        "FTM", "FTA", "REB", "OREB", "DREB", "AST", "STL", "BLK",
        "TOV", "PF",
    ]
    home = pd.DataFrame({k: h[k] for k in keep})
    away = pd.DataFrame({k: a[k] for k in keep})
    return home, away


def simulate_team(
    profile: dict,
    ctx: TeamContext,
    n: int = 100_000,
    seed: int = 3,
    opportunity_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Legacy single-team wrapper retained for backward compatibility.

    New Team Markets should use simulate_game(), because REB/DREB/STL/BLK and
    relative/total markets require the opponent side. Defensive outputs from
    this wrapper are intentionally omitted rather than fabricated from the
    team's own opportunities.
    """
    rng = np.random.default_rng(seed)
    poss = np.rint(
        np.clip(rng.normal(ctx.projected_possessions, ctx.possessions_sd, n), 55, 115)
        * opportunity_mult
    ).astype(int)
    z = [rng.normal(size=n) for _ in range(5)]
    o = _simulate_offense(profile, ctx, poss, rng, *z)
    keep = [
        "POSS", "PTS", "FGM", "FGA", "3PM", "3PA", "2PM", "2PA",
        "FTM", "FTA", "OREB", "AST", "TOV", "PF",
    ]
    return pd.DataFrame({k: o[k] for k in keep})
