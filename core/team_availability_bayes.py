from __future__ import annotations

"""Roster-informed empirical-Bayes availability bridge for Team Markets only.

The bridge is deliberately downstream of the player-prop engine.  It combines:

1) a mechanistic prior from the CURRENT 200-minute rotation versus a healthy
   counterfactual, using each player's historical per-minute event profile; and
2) opponent-residualized historical games that resemble the selected OUT state.

The shrinkage mass K is feature-specific and estimated league-wide from repeated
availability states in the structural-model residuals.  No fixed K is used by
this bridge.  Current explicit minute restrictions are treated as current-state
information and are added after the historical OUT-state posterior.
"""

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd


FEATURE_SPECS = {
    # output key: (structural feature, state-score key, team-profile key, bounds)
    "3P_SHARE": ("3P_SHARE", "3PA", "three_share", (0.06, 0.75)),
    "FTA": ("FTA", "FTA", "fta_pp", (0.05, 0.55)),
    "TOV": ("TOV", "TOV", "tov_pp", (0.03, 0.30)),
    "OREB": ("OREB_PER_MISS", "OREB", "oreb_per_miss", (0.05, 0.55)),
    "AST": ("AST_PER_MAKE", "AST", "assist_per_make", (0.20, 0.95)),
}


@dataclass(frozen=True)
class AvailabilityHyperParam:
    feature: str
    k: float
    raw_k: float
    within_var: float
    between_var: float
    state_groups: int
    state_rows: int


def _is_probability(feature: str) -> bool:
    return feature in {"3P_SHARE", "TOV", "OREB_PER_MISS", "AST_PER_MAKE"}


def _transform(feature: str, value: float) -> float:
    v = float(value)
    if _is_probability(feature):
        v = float(np.clip(v, 1e-4, 1.0 - 1e-4))
        return float(np.log(v / (1.0 - v)))
    return float(np.log(max(v, 1e-4)))


def _inverse(feature: str, z: float) -> float:
    z = float(np.clip(z, -12.0, 12.0))
    if _is_probability(feature):
        return float(1.0 / (1.0 + np.exp(-z)))
    return float(np.exp(z))


def _player_rate_table(player_db: pd.DataFrame, team_abbr: str) -> tuple[dict[str, dict], dict]:
    """Stable current-season per-minute player priors plus a pooled fallback."""
    cols = ["FGA", "FGM", "FG3A", "FTA", "TOV", "OREB", "AST"]
    x = player_db[
        player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
    ].copy()
    if x.empty:
        return {}, {c: 0.0 for c in cols}
    x["MIN"] = pd.to_numeric(x["MIN"], errors="coerce").fillna(0.0)
    for c in cols:
        if c not in x.columns:
            x[c] = 0.0
        else:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    # Very short appearances are poor estimates of a player's stable role/rate.
    active = x[x["MIN"] >= 4.0].copy()
    if active.empty:
        active = x[x["MIN"] > 0].copy()
    pooled_min = float(active["MIN"].sum())
    fallback = {
        c: float(active[c].sum()) / max(pooled_min, 1.0)
        for c in cols
    }
    rates: dict[str, dict] = {}
    for name, g in active.groupby(active["PLAYER_NAME"].astype(str)):
        mins = float(g["MIN"].sum())
        if mins <= 0:
            continue
        rates[str(name)] = {c: float(g[c].sum()) / mins for c in cols}
    return rates, fallback


def _rotation_rates(player_db: pd.DataFrame, team_abbr: str, board: pd.DataFrame) -> dict[str, float]:
    rates, fallback = _player_rate_table(player_db, team_abbr)
    totals = {c: 0.0 for c in ["FGA", "FGM", "FG3A", "FTA", "TOV", "OREB", "AST"]}
    if board is None or board.empty:
        return {
            "3P_SHARE": np.nan, "FTA": np.nan, "TOV": np.nan,
            "OREB_PER_MISS": np.nan, "AST_PER_MAKE": np.nan,
        }
    for _, row in board.iterrows():
        name = str(row.get("Player", ""))
        mins = float(row.get("Projected Min", 0.0) or 0.0)
        if mins <= 0:
            continue
        p = rates.get(name, fallback)
        for c in totals:
            totals[c] += mins * float(p.get(c, fallback.get(c, 0.0)) or 0.0)

    fga = max(totals["FGA"], 1e-6)
    fgm = max(totals["FGM"], 1e-6)
    misses = max(totals["FGA"] - totals["FGM"], 1e-6)
    poss = max(totals["FGA"] - totals["OREB"] + totals["TOV"] + 0.44 * totals["FTA"], 1e-6)
    return {
        "3P_SHARE": float(totals["FG3A"] / fga),
        "FTA": float(totals["FTA"] / poss),
        "TOV": float(totals["TOV"] / poss),
        "OREB_PER_MISS": float(totals["OREB"] / misses),
        "AST_PER_MAKE": float(totals["AST"] / fgm),
    }


def _historical_state_keys(player_db: pd.DataFrame, min_on: float = 10.0, min_core_games: int = 3):
    """Observed availability-state labels used only to learn league EB variance.

    A core player must have at least ``min_core_games`` appearances of >= min_on
    minutes.  Before her first appearance for that team she is not eligible to be
    called absent.  A game is labelled by core players with no appearance/minutes;
    short 0<MIN<min_on appearances are treated as ambiguous/ON rather than as OUT.
    """
    if player_db is None or player_db.empty:
        return {}
    x = player_db.copy()
    x["TEAM_ABBR"] = x["TEAM_ABBR"].astype(str).str.upper()
    x["PLAYER_NAME"] = x["PLAYER_NAME"].astype(str)
    x["GAME_ID"] = x["GAME_ID"].astype(str)
    x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
    x["MIN"] = pd.to_numeric(x["MIN"], errors="coerce").fillna(0.0)

    core = {}
    first = {}
    for (team, name), g in x.groupby(["TEAM_ABBR", "PLAYER_NAME"]):
        n_on = int((g["MIN"] >= float(min_on)).sum())
        if n_on >= int(min_core_games):
            core.setdefault(team, []).append(name)
            first[(team, name)] = g["GAME_DATE"].dropna().min()

    game_minutes = {
        (team, gid): dict(zip(g["PLAYER_NAME"], g["MIN"]))
        for (team, gid), g in x.groupby(["TEAM_ABBR", "GAME_ID"])
    }
    game_dates = {
        (team, gid): g["GAME_DATE"].dropna().min()
        for (team, gid), g in x.groupby(["TEAM_ABBR", "GAME_ID"])
    }
    keys = {}
    for tg, mins_map in game_minutes.items():
        team, gid = tg
        gdate = game_dates.get(tg)
        absent = []
        for name in core.get(team, []):
            start = first.get((team, name))
            if pd.isna(gdate) or start is None or pd.isna(start) or gdate < start:
                continue
            mins = float(mins_map.get(name, 0.0) or 0.0)
            if mins <= 0.0:
                absent.append(name)
        keys[(team, gid)] = tuple(sorted(absent, key=str.casefold))
    return keys


def fit_availability_hyperparams(
    player_db: pd.DataFrame,
    structural_models: Mapping[str, object],
    *,
    min_on: float = 10.0,
    min_core_games: int = 3,
    min_groups: int = 4,
    k_floor: float = 0.75,
    k_ceiling: float = 32.0,
) -> tuple[Dict[str, AvailabilityHyperParam], pd.DataFrame]:
    """Empirical-Bayes K = within-state variance / between-state variance.

    Structural residuals are already opponent-adjusted and strictly pregame on
    their own/opponent baselines.  Repeated observed availability states provide
    a league-wide random-effects estimate of how repeatable an availability
    residual is.  If the data cannot identify positive between-state variance,
    K=inf and current historical state games do not override the roster prior.
    """
    state_keys = _historical_state_keys(player_db, min_on=min_on, min_core_games=min_core_games)
    out: Dict[str, AvailabilityHyperParam] = {}
    audit_rows = []

    for feature in [s[0] for s in FEATURE_SPECS.values()]:
        model = structural_models.get(feature)
        tab = getattr(model, "training_table", None)
        beta = float(getattr(model, "opponent_beta", 0.0) or 0.0)
        if tab is None or tab.empty:
            hp = AvailabilityHyperParam(feature, np.inf, np.inf, np.nan, np.nan, 0, 0)
            out[feature] = hp
            audit_rows.append({"Feature": feature, "K": np.inf, "Reason": "no structural training rows"})
            continue

        z = tab.copy()
        z["TEAM_ABBR"] = z["TEAM_ABBR"].astype(str).str.upper()
        z["GAME_ID"] = z["GAME_ID"].astype(str)
        z["RESID"] = pd.to_numeric(z["Y"], errors="coerce") - (
            pd.to_numeric(z["OWN_DEV"], errors="coerce")
            + beta * pd.to_numeric(z["OPP_DEV"], errors="coerce")
        )
        z = z[np.isfinite(z["RESID"])].copy()
        if z.empty:
            hp = AvailabilityHyperParam(feature, np.inf, np.inf, np.nan, np.nan, 0, 0)
            out[feature] = hp
            audit_rows.append({"Feature": feature, "K": np.inf, "Reason": "no finite residuals"})
            continue

        # Remove persistent team residual bias before estimating state variance.
        z["RESID_C"] = z["RESID"] - z.groupby("TEAM_ABBR")["RESID"].transform("mean")
        z["STATE"] = [state_keys.get((t, g), tuple()) for t, g in zip(z["TEAM_ABBR"], z["GAME_ID"])]
        z = z[z["STATE"].map(len) > 0].copy()

        groups = []
        for (team, state), g in z.groupby(["TEAM_ABBR", "STATE"]):
            if len(g) < 2:
                continue
            vals = g["RESID_C"].to_numpy(float)
            if len(vals) < 2 or not np.all(np.isfinite(vals)):
                continue
            groups.append((team, state, len(vals), float(np.mean(vals)), float(np.var(vals, ddof=1))))

        if len(groups) < int(min_groups):
            hp = AvailabilityHyperParam(feature, np.inf, np.inf, np.nan, np.nan, len(groups), int(sum(g[2] for g in groups)))
            out[feature] = hp
            audit_rows.append({
                "Feature": feature, "K": np.inf, "Raw K": np.inf,
                "Repeated state groups": len(groups), "Rows": int(sum(g[2] for g in groups)),
                "Reason": "insufficient repeated availability states",
            })
            continue

        ns = np.asarray([g[2] for g in groups], dtype=float)
        means = np.asarray([g[3] for g in groups], dtype=float)
        vars_ = np.asarray([g[4] for g in groups], dtype=float)
        df_total = float(np.sum(np.maximum(ns - 1.0, 0.0)))
        within = float(np.sum((ns - 1.0) * vars_) / max(df_total, 1.0))
        mean_bar = float(np.average(means, weights=ns))
        between_raw = float(np.average((means - mean_bar) ** 2, weights=ns))
        sampling = float(np.average(within / np.maximum(ns, 1.0), weights=ns))
        between = max(between_raw - sampling, 0.0)

        if not np.isfinite(within) or within <= 1e-10 or not np.isfinite(between) or between <= 1e-10:
            raw_k = np.inf
            k = np.inf
            reason = "between-state variance not identified"
        else:
            raw_k = float(within / between)
            k = float(np.clip(raw_k, float(k_floor), float(k_ceiling)))
            reason = "league random-effects EB"
        hp = AvailabilityHyperParam(feature, k, raw_k, within, between, len(groups), int(np.sum(ns)))
        out[feature] = hp
        audit_rows.append({
            "Feature": feature,
            "K": k,
            "Raw K": raw_k,
            "Within-state residual var": within,
            "Between-state effect var": between,
            "Repeated state groups": len(groups),
            "Rows": int(np.sum(ns)),
            "Reason": reason,
        })

    return out, pd.DataFrame(audit_rows)


def availability_posterior_modifiers(
    *,
    player_db: pd.DataFrame,
    team_abbr: str,
    opponent_abbr: str,
    base_profile: Mapping[str, float],
    structural_modifiers: Mapping[str, float],
    structural_models: Mapping[str, object],
    state_scores_by_stat: Mapping[str, Mapping[str, float]],
    rotation_impact,
    hyperparams: Mapping[str, AvailabilityHyperParam],
) -> tuple[Dict[str, float], pd.DataFrame]:
    """Return Team-Market-only availability modifiers and posterior audit."""
    healthy = _rotation_rates(player_db, team_abbr, rotation_impact.healthy_minutes)
    out_only = _rotation_rates(player_db, team_abbr, rotation_impact.out_only_minutes)
    current = _rotation_rates(player_db, team_abbr, rotation_impact.current_minutes)

    mods: Dict[str, float] = {}
    rows = []
    team = str(team_abbr).upper()
    opp = str(opponent_abbr).upper()

    for out_key, (feature, state_key, profile_key, bounds) in FEATURE_SPECS.items():
        lo, hi = bounds
        healthy_rate = float(healthy.get(feature, np.nan))
        out_rate = float(out_only.get(feature, np.nan))
        current_rate = float(current.get(feature, np.nan))

        if not all(np.isfinite(v) and v > 0 for v in (healthy_rate, out_rate, current_rate)):
            mods[out_key] = 1.0
            rows.append({"Feature": feature, "Applied modifier": 1.0, "Reason": "rotation prior unavailable"})
            continue

        out_prior_delta = _transform(feature, out_rate) - _transform(feature, healthy_rate)
        restriction_delta = _transform(feature, current_rate) - _transform(feature, out_rate)

        model = structural_models.get(feature)
        tab = getattr(model, "training_table", None)
        beta = float(getattr(model, "opponent_beta", 0.0) or 0.0)
        score_map = dict(state_scores_by_stat.get(state_key, {}) or {})
        weighted_sum = 0.0
        evidence_mass = 0.0
        evidence_rows = 0
        residual_mean = np.nan
        if tab is not None and not tab.empty and score_map:
            h = tab[
                tab["TEAM_ABBR"].astype(str).str.upper().eq(team)
                & ~tab["OPP_ABBR"].astype(str).str.upper().eq(opp)
            ].copy()
            for _, r in h.iterrows():
                gid = str(r["GAME_ID"])
                s = float(score_map.get(gid, 0.0) or 0.0)
                if s <= 0:
                    continue
                w = s * s
                resid = float(r["Y"]) - (float(r["OWN_DEV"]) + beta * float(r["OPP_DEV"]))
                if not np.isfinite(resid):
                    continue
                weighted_sum += w * resid
                evidence_mass += w
                evidence_rows += 1
            if evidence_mass > 0:
                residual_mean = weighted_sum / evidence_mass

        hp = hyperparams.get(feature)
        k = float(getattr(hp, "k", np.inf)) if hp is not None else np.inf
        if evidence_mass > 0 and np.isfinite(k):
            out_post_delta = float((k * out_prior_delta + weighted_sum) / (k + evidence_mass))
            empirical_weight = float(evidence_mass / (k + evidence_mass))
        else:
            out_post_delta = float(out_prior_delta)
            empirical_weight = 0.0

        total_delta = float(out_post_delta + restriction_delta)
        base_rate = float(base_profile.get(profile_key, np.nan))
        structural_mult = float(structural_modifiers.get(out_key, 1.0) or 1.0)
        structural_rate = base_rate * structural_mult if np.isfinite(base_rate) else np.nan
        if not np.isfinite(structural_rate) or structural_rate <= 0:
            mod = 1.0
            final_rate = structural_rate
        else:
            final_rate = float(np.clip(_inverse(feature, _transform(feature, structural_rate) + total_delta), lo, hi))
            mod = float(final_rate / structural_rate)
        mods[out_key] = mod
        rows.append({
            "Feature": feature,
            "Healthy roster prior": healthy_rate,
            "OUT-only roster prior": out_rate,
            "Current roster prior": current_rate,
            "Roster OUT delta (transformed)": out_prior_delta,
            "Restriction delta (transformed)": restriction_delta,
            "EB K": k,
            "Historical effective mass": evidence_mass,
            "Historical matched rows": evidence_rows,
            "Historical residual mean": residual_mean,
            "Historical posterior weight": empirical_weight,
            "Posterior OUT delta (transformed)": out_post_delta,
            "Total availability delta (transformed)": total_delta,
            "Pre-availability structural rate": structural_rate,
            "Post-availability rate": final_rate,
            "Applied modifier": mod,
        })

    return mods, pd.DataFrame(rows)
