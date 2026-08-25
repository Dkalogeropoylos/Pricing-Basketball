from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from core.buckets import WeightConfig, split_non_overlapping, active_weights, weighted_average_feature


@dataclass
class PlayerContext:
    projected_minutes: float
    minutes_sd: float = 2.0

    # Central matchup pace relative to the pace already embedded in the
    # player's historical per-minute rates. Random pace shock still sits
    # around this central multiplier.
    pace_multiplier: float = 1.00

    # opponent environment, 1.00 = neutral
    opp_pts: float = 1.00
    opp_reb: float = 1.00
    opp_ast: float = 1.00
    opp_3pa: float = 1.00
    opp_fta: float = 1.00

    # role redistribution
    usage: float = 1.00
    creation: float = 1.00
    reb_role: float = 1.00
    three_role: float = 1.00
    fta_role: float = 1.00

    # H2H context only: recommended range 0.90 - 1.10
    h2h_pts: float = 1.00
    h2h_reb: float = 1.00
    h2h_ast: float = 1.00
    h2h_3pa: float = 1.00


def _safe_div(a, b, default=0.0):
    return float(a) / float(b) if b and np.isfinite(b) and b > 0 else default


def _features(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    mins = float(df["MIN"].sum())
    fga = float(df["FGA"].sum())
    a3 = float(df["FG3A"].sum())
    m3 = float(df["FG3M"].sum())
    fgm = float(df["FGM"].sum())
    fta = float(df["FTA"].sum())
    ftm = float(df["FTM"].sum())
    a2 = max(fga - a3, 0.0)
    m2 = max(fgm - m3, 0.0)
    return {
        "games": len(df),
        "min_pg": float(df["MIN"].mean()),
        "two_pa_pm": _safe_div(a2, mins),
        "three_pa_pm": _safe_div(a3, mins),
        "fta_pm": _safe_div(fta, mins),
        "reb_pm": _safe_div(df["REB"].sum(), mins),
        "ast_pm": _safe_div(df["AST"].sum(), mins),
        "pts_pm": _safe_div(df["PTS"].sum(), mins),
        "three_pct": _safe_div(m3, a3, np.nan),
        "two_pct": _safe_div(m2, a2, np.nan),
        "ft_pct": _safe_div(ftm, fta, np.nan),
        "three_att": a3,
        "two_att": a2,
        "ft_att": fta,
    }


def _shrink(obs, attempts, prior, prior_attempts):
    if not np.isfinite(obs):
        obs, attempts = prior, 0.0
    return float((obs * attempts + prior * prior_attempts) / (attempts + prior_attempts))


def build_player_profile(df: pd.DataFrame, cfg: WeightConfig) -> Tuple[dict, pd.DataFrame]:
    x = df.sort_values("GAME_DATE").copy()
    buckets = split_non_overlapping(x)
    weights = active_weights(buckets, cfg)
    feats = {k: _features(v) for k, v in buckets.items()}

    profile = {}
    for key in ["min_pg", "two_pa_pm", "three_pa_pm", "fta_pm", "reb_pm", "ast_pm", "pts_pm"]:
        profile[key] = weighted_average_feature(feats, weights, key)

    full = _features(x)
    # Larger-sample ability + league priors; no raw L5 shooting truth.
    profile["three_pct"] = _shrink(full.get("three_pct", np.nan), full.get("three_att", 0), 0.340, 45)
    profile["two_pct"] = _shrink(full.get("two_pct", np.nan), full.get("two_att", 0), 0.510, 70)
    profile["ft_pct"] = _shrink(full.get("ft_pct", np.nan), full.get("ft_att", 0), 0.785, 40)

    audit = []
    for k in ("old", "mid", "l5"):
        audit.append({"bucket": k, "weight": weights[k], **feats.get(k, {"games": 0})})
    return profile, pd.DataFrame(audit)


def simulate_player(profile: dict, ctx: PlayerContext, n=100_000, seed=1, opportunity_mult=1.0):
    rng = np.random.default_rng(seed)

    z_min = rng.normal(size=n)
    z_pace = rng.normal(size=n)
    z_role = rng.normal(size=n)
    z_perim = rng.normal(size=n)
    z_foul = rng.normal(size=n)
    z_reb = rng.normal(size=n)
    z_ast = rng.normal(size=n)
    z_shoot = rng.normal(size=n)

    mins = np.clip(
        ctx.projected_minutes + ctx.minutes_sd * z_min,
        max(4.0, ctx.projected_minutes - 8.0),
        ctx.projected_minutes + 8.0,
    )
    pace = ctx.pace_multiplier * np.exp(0.035 * z_pace - 0.5 * 0.035**2)
    role = np.exp(0.05 * z_role - 0.5 * 0.05**2)
    perim = np.exp(0.07 * z_perim - 0.5 * 0.07**2)
    foul = np.exp(0.10 * z_foul - 0.5 * 0.10**2)
    reb_env = np.exp(0.10 * z_reb - 0.5 * 0.10**2)
    ast_env = np.exp(0.10 * z_ast - 0.5 * 0.10**2)

    base = pace * role * opportunity_mult

    lam3 = mins * profile["three_pa_pm"] * base * perim * ctx.usage * ctx.three_role * ctx.opp_3pa * ctx.h2h_3pa
    lam2 = mins * profile["two_pa_pm"] * base * ctx.usage * ctx.opp_pts * ctx.h2h_pts
    lamft = mins * profile["fta_pm"] * base * foul * ctx.usage * ctx.fta_role * ctx.opp_fta

    a3 = rng.poisson(np.clip(lam3, 0.001, None))
    a2 = rng.poisson(np.clip(lam2, 0.001, None))
    fta = rng.poisson(np.clip(lamft, 0.001, None))

    p3 = np.clip(profile["three_pct"] + 0.035 * z_shoot, 0.02, 0.98)
    p2 = np.clip(profile["two_pct"] + 0.026 * z_shoot, 0.02, 0.98)
    pft = np.clip(profile["ft_pct"] + 0.010 * z_shoot, 0.02, 0.98)

    m3 = rng.binomial(a3, p3)
    m2 = rng.binomial(a2, p2)
    ftm = rng.binomial(fta, pft)
    pts = 3*m3 + 2*m2 + ftm

    # Makes support assists; misses support rebound opportunity.
    ast_shot = np.clip(1 + 0.18*z_shoot, 0.70, 1.35)
    reb_shot = np.clip(1 - 0.10*z_shoot, 0.75, 1.30)

    lam_ast = mins * profile["ast_pm"] * pace * role * ast_env * ast_shot * opportunity_mult * ctx.creation * ctx.opp_ast * ctx.h2h_ast
    lam_reb = mins * profile["reb_pm"] * pace * reb_env * reb_shot * opportunity_mult * ctx.reb_role * ctx.opp_reb * ctx.h2h_reb

    ast = rng.poisson(np.clip(lam_ast, 0.001, None))
    reb = rng.poisson(np.clip(lam_reb, 0.001, None))

    out = pd.DataFrame({"MIN": mins, "PTS": pts, "REB": reb, "AST": ast, "3PM": m3, "3PA": a3, "FTA": fta})
    out["PRA"] = out.PTS + out.REB + out.AST
    out["PR"] = out.PTS + out.REB
    out["PA"] = out.PTS + out.AST
    out["AR"] = out.AST + out.REB
    return out
