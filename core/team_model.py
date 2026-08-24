from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from core.buckets import WeightConfig, split_non_overlapping, active_weights, weighted_average_feature


@dataclass
class TeamContext:
    projected_possessions: float
    possessions_sd: float = 3.0

    # Offense-vs-opponent interaction multipliers
    three_pa: float = 1.0
    three_pct: float = 1.0
    two_pa: float = 1.0
    two_pct: float = 1.0
    fta: float = 1.0
    tov: float = 1.0
    oreb: float = 1.0
    ast: float = 1.0
    stl: float = 1.0
    blk: float = 1.0
    pf: float = 1.0


def estimate_possessions(df):
    return df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]


def _feat(df):
    if df.empty:
        return {}
    poss = float(estimate_possessions(df).sum())
    fga = float(df.FGA.sum())
    a3 = float(df.FG3A.sum())
    m3 = float(df.FG3M.sum())
    a2 = max(fga-a3, 0)
    m2 = max(float(df.FGM.sum())-m3, 0)
    fta = float(df.FTA.sum())
    return {
        "games": len(df),
        "poss_pg": float(estimate_possessions(df).mean()),
        "three_pa_pp": a3/poss if poss else 0,
        "two_pa_pp": a2/poss if poss else 0,
        "fta_pp": fta/poss if poss else 0,
        "tov_pp": float(df.TOV.sum())/poss if poss else 0,
        "oreb_pp": float(df.OREB.sum())/poss if poss else 0,
        "ast_pp": float(df.AST.sum())/poss if poss else 0,
        "stl_pp": float(df.STL.sum())/poss if poss else 0,
        "blk_pp": float(df.BLK.sum())/poss if poss else 0,
        "pf_pp": float(df.PF.sum())/poss if poss else 0,
        "three_pct": m3/a3 if a3 else np.nan,
        "two_pct": m2/a2 if a2 else np.nan,
        "ft_pct": float(df.FTM.sum())/fta if fta else np.nan,
    }


def build_team_profile(df, cfg: WeightConfig):
    buckets = split_non_overlapping(df)
    weights = active_weights(buckets, cfg)
    feats = {k:_feat(v) for k,v in buckets.items()}

    p = {}
    for k in ["poss_pg","three_pa_pp","two_pa_pp","fta_pp","tov_pp","oreb_pp","ast_pp","stl_pp","blk_pp","pf_pp"]:
        p[k] = weighted_average_feature(feats, weights, k)

    full = _feat(df)
    # Simple larger-sample regression.
    p["three_pct"] = 0.75*(full.get("three_pct") if np.isfinite(full.get("three_pct",np.nan)) else .34) + 0.25*.34
    p["two_pct"] = 0.80*(full.get("two_pct") if np.isfinite(full.get("two_pct",np.nan)) else .51) + 0.20*.51
    p["ft_pct"] = 0.85*(full.get("ft_pct") if np.isfinite(full.get("ft_pct",np.nan)) else .785) + 0.15*.785

    audit=[]
    for k in ("old","mid","l5"):
        audit.append({"bucket":k,"weight":weights[k],**feats.get(k,{"games":0})})
    return p, pd.DataFrame(audit)


def simulate_team(profile, ctx: TeamContext, n=100_000, seed=3, opportunity_mult=1.0):
    rng=np.random.default_rng(seed)

    # common game state
    poss=np.clip(rng.normal(ctx.projected_possessions, ctx.possessions_sd, n), 55, 115)
    poss *= opportunity_mult
    z_style=rng.normal(size=n)
    z_shoot=rng.normal(size=n)
    z_foul=rng.normal(size=n)
    z_tov=rng.normal(size=n)
    z_reb=rng.normal(size=n)

    # Possession allocation.
    tov_rate=np.clip(profile["tov_pp"]*ctx.tov*np.exp(.08*z_tov-.5*.08**2), .03, .30)
    tov=rng.binomial(np.maximum(poss.astype(int),1), tov_rate)
    live=np.maximum(poss-tov, 1)

    perimeter=np.exp(.08*z_style-.5*.08**2)
    a3=rng.poisson(np.clip(live*profile["three_pa_pp"]*ctx.three_pa*perimeter, .001, None))
    a2=rng.poisson(np.clip(live*profile["two_pa_pp"]*ctx.two_pa/perimeter**0.35, .001, None))
    fta=rng.poisson(np.clip(poss*profile["fta_pp"]*ctx.fta*np.exp(.12*z_foul-.5*.12**2), .001, None))

    p3=np.clip(profile["three_pct"]*ctx.three_pct + .03*z_shoot, .10, .60)
    p2=np.clip(profile["two_pct"]*ctx.two_pct + .025*z_shoot, .25, .75)
    pft=np.clip(profile["ft_pct"] + .01*z_shoot, .45, .98)

    m3=rng.binomial(a3,p3)
    m2=rng.binomial(a2,p2)
    ftm=rng.binomial(fta,pft)
    fgm=m3+m2
    pts=3*m3+2*m2+ftm

    misses=np.maximum((a3-m3)+(a2-m2),0)
    # OREB conditional on misses, capped.
    oreb_share=np.clip(
        (profile["oreb_pp"] / max(profile["three_pa_pp"]+profile["two_pa_pp"], .1))
        * ctx.oreb * np.exp(.10*z_reb-.5*.10**2),
        .08,.45
    )
    oreb=rng.binomial(misses, oreb_share)

    # Assists from made FGs, with profile assisted-FG signal.
    assist_per_make = np.clip(profile["ast_pp"] / max((profile["three_pa_pp"]*profile["three_pct"] + profile["two_pa_pp"]*profile["two_pct"]), .05), .25, .90)
    ast_prob=np.clip(assist_per_make*ctx.ast*(1+.08*z_shoot), .20,.95)
    ast=rng.binomial(fgm, ast_prob)

    stl=rng.poisson(np.clip(tov*0.55*ctx.stl, .001, None))
    blk=rng.poisson(np.clip(a2*profile["blk_pp"]/max(profile["two_pa_pp"],.05)*ctx.blk, .001, None))
    pf=rng.poisson(np.clip(poss*profile["pf_pp"]*ctx.pf*np.exp(.10*z_foul-.5*.10**2), .001, None))

    out=pd.DataFrame({
        "POSS":poss,"TOV":tov,"3PA":a3,"3PM":m3,"2PA":a2,"2PM":m2,
        "FTA":fta,"FTM":ftm,"OREB":oreb,"AST":ast,"STL":stl,"BLK":blk,"PF":pf,"PTS":pts
    })
    return out
