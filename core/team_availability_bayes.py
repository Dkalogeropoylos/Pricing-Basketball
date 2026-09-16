from __future__ import annotations

"""Rotation-local empirical-Bayes availability bridge for Team Markets only.

Production intent
-----------------
This module is deliberately downstream of the Player Props engine.  It does not
change player minutes, player role-state, opponent-by-position, player H2H, or
player Monte Carlo.

For each structural Team-Market rate it combines:

1) a mechanistic fallback prior from the CURRENT projected 200-minute rotation
   versus a healthy counterfactual; and
2) opponent/H2H-residualized historical TEAM games, weighted by how similar the
   *actual historical minute-share rotation* was to today's projected OUT-only
   rotation.  Recency is a secondary kernel dimension.

The local-kernel bandwidth, recency half-life scale, and prior-equivalent mass K
are selected feature-by-feature by chronological walk-forward validation.  A
local historical layer is activated only when it improves a genuinely later
holdout over the zero-residual structural baseline.  Thus there is no fixed
"important player" definition and no hand-set K=6 in this Team-Market bridge.

Similarity metric
-----------------
Historical/current rotations are probability vectors over player minute shares.
Distance is total-variation distance

    d = 0.5 * sum_i |p_i - q_i|

which has an intuitive basketball interpretation: d * 200 is the number of
regulation rotation minutes that would need to be reassigned to turn one
rotation into the other.  The local weight is

    w = exp(-d / h) * exp(-days / tau)

where h, tau, and K are learned chronologically.  tau=inf is an explicit
candidate, so the data may decide that recency adds no value beyond rotation
similarity.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Iterable

import numpy as np
import pandas as pd


FEATURE_SPECS = {
    # output key: (structural feature, team-profile key, physical bounds)
    "3P_SHARE": ("3P_SHARE", "three_share", (0.06, 0.75)),
    "FTA": ("FTA", "fta_pp", (0.05, 0.55)),
    "TOV": ("TOV", "tov_pp", (0.03, 0.30)),
    "OREB": ("OREB_PER_MISS", "oreb_per_miss", (0.05, 0.55)),
    "AST": ("AST_PER_MAKE", "assist_per_make", (0.20, 0.95)),
}


@dataclass(frozen=True)
class AvailabilityHyperParam:
    feature: str
    active: bool
    bandwidth: float
    recency_tau_days: float
    k: float
    tune_rmse: float
    baseline_tune_rmse: float
    holdout_rmse: float
    baseline_holdout_rmse: float
    holdout_rows: int
    eligible_prediction_rows: int


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mechanistic roster prior (fallback, not the primary local-history signal)
# ---------------------------------------------------------------------------

def _player_rate_table(player_db: pd.DataFrame, team_abbr: str) -> tuple[dict[str, dict], dict]:
    """Stable current-season per-minute player priors plus a pooled fallback.

    These rates are intentionally conservative.  They provide the prior mean
    when local historical rotations are sparse; they do not define which past
    games are relevant.
    """
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

    active = x[x["MIN"] >= 4.0].copy()
    if active.empty:
        active = x[x["MIN"] > 0].copy()
    pooled_min = float(active["MIN"].sum())
    fallback = {c: float(active[c].sum()) / max(pooled_min, 1.0) for c in cols}

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


# ---------------------------------------------------------------------------
# Rotation-state geometry
# ---------------------------------------------------------------------------

def _norm_name(v) -> str:
    return str(v).strip().casefold()


def _normalize_rotation(raw: Mapping[str, float]) -> dict[str, float]:
    x = {_norm_name(k): max(float(v or 0.0), 0.0) for k, v in raw.items() if str(k).strip()}
    den = float(sum(x.values()))
    if den <= 0:
        return {}
    return {k: float(v / den) for k, v in x.items() if v > 0}


def _board_rotation(board: pd.DataFrame) -> dict[str, float]:
    if board is None or board.empty:
        return {}
    raw = {}
    for _, r in board.iterrows():
        name = str(r.get("Player", ""))
        mins = float(r.get("Projected Min", 0.0) or 0.0)
        if name.strip() and mins > 0:
            raw[name] = raw.get(name, 0.0) + mins
    return _normalize_rotation(raw)


def _historical_rotations(player_db: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """Actual minute-share vector for every team-game.

    Shares are normalized within game, so overtime does not mechanically make a
    game less comparable merely because total team minutes exceeded 200.
    """
    if player_db is None or player_db.empty:
        return {}
    x = player_db.copy()
    x["TEAM_ABBR"] = x["TEAM_ABBR"].astype(str).str.upper()
    x["GAME_ID"] = x["GAME_ID"].astype(str)
    x["PLAYER_NAME"] = x["PLAYER_NAME"].astype(str)
    x["MIN"] = pd.to_numeric(x["MIN"], errors="coerce").fillna(0.0)
    x = x[x["MIN"] > 0].copy()
    out: dict[tuple[str, str], dict[str, float]] = {}
    for (team, gid), g in x.groupby(["TEAM_ABBR", "GAME_ID"], sort=False):
        raw = g.groupby(g["PLAYER_NAME"].map(_norm_name))["MIN"].sum().to_dict()
        out[(str(team).upper(), str(gid))] = _normalize_rotation(raw)
    return out


def _tv_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Total-variation distance between minute-share vectors, in [0,1]."""
    if not a or not b:
        return 1.0
    keys = set(a) | set(b)
    return float(np.clip(0.5 * sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys), 0.0, 1.0))


def _kernel_weight(distance: float, age_days: float, bandwidth: float, tau_days: float) -> float:
    if not np.isfinite(distance) or distance < 0 or not np.isfinite(bandwidth) or bandwidth <= 0:
        return 0.0
    w = float(np.exp(-float(distance) / float(bandwidth)))
    if np.isfinite(tau_days):
        if tau_days <= 0:
            return 0.0
        w *= float(np.exp(-max(float(age_days), 0.0) / float(tau_days)))
    return float(w)


# ---------------------------------------------------------------------------
# Structural residuals net of opponent and residualized H2H
# ---------------------------------------------------------------------------

def _residual_table(model) -> pd.DataFrame:
    tab = getattr(model, "training_table", None)
    if tab is None or tab.empty:
        return pd.DataFrame()
    z = tab.copy().sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBR"]).reset_index(drop=True)
    z["GAME_DATE"] = pd.to_datetime(z["GAME_DATE"], errors="coerce")
    z["TEAM_ABBR"] = z["TEAM_ABBR"].astype(str).str.upper()
    z["OPP_ABBR"] = z["OPP_ABBR"].astype(str).str.upper()
    z["GAME_ID"] = z["GAME_ID"].astype(str)
    model_active = bool(getattr(model, "active", True))
    beta = float(getattr(model, "opponent_beta", 0.0) or 0.0) if model_active else 0.0
    h2h_k = float(getattr(model, "h2h_prior_k", np.inf)) if model_active else np.inf

    pair_memory: dict[tuple[str, str], list[float]] = {}
    residuals = []
    for r in z.itertuples(index=False):
        base_no_h2h = float(r.OWN_DEV) + beta * float(r.OPP_DEV)
        key = (str(r.TEAM_ABBR).upper(), str(r.OPP_ABBR).upper())
        prior = pair_memory.get(key, [])
        if prior and np.isfinite(h2h_k):
            h2h_w = float(len(prior) / (len(prior) + max(h2h_k, 1e-9)))
            h2h_term = h2h_w * float(np.mean(prior))
        else:
            h2h_term = 0.0
        residuals.append(float(r.Y) - (base_no_h2h + h2h_term))
        # Match structural_calibration.py: pair memory stores residual against
        # the no-H2H expectation, not against a recursive corrected prediction.
        pair_memory.setdefault(key, []).append(float(r.Y) - base_no_h2h)
    z["LOCAL_RESID"] = residuals
    return z[np.isfinite(z["LOCAL_RESID"]) & z["GAME_DATE"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chronological learning of h, tau and K
# ---------------------------------------------------------------------------

def _candidate_history_for_row(
    rows: pd.DataFrame,
    rotations: Mapping[tuple[str, str], Mapping[str, float]],
    i: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Earlier same-team, non-H2H candidates for one target row."""
    r = rows.iloc[i]
    team = str(r["TEAM_ABBR"]).upper()
    opp = str(r["OPP_ABBR"]).upper()
    target_date = pd.Timestamp(r["GAME_DATE"])
    target_rot = rotations.get((team, str(r["GAME_ID"])), {})
    if not target_rot:
        return np.asarray([]), np.asarray([]), np.asarray([])

    dists, ages, vals = [], [], []
    for j in range(i):
        h = rows.iloc[j]
        if str(h["TEAM_ABBR"]).upper() != team:
            continue
        # H2H is modeled separately in the structural layer.  Do not let the
        # availability kernel reuse same-opponent rows for this target.
        if str(h["OPP_ABBR"]).upper() == opp:
            continue
        hist_rot = rotations.get((team, str(h["GAME_ID"])), {})
        if not hist_rot:
            continue
        age = float((target_date - pd.Timestamp(h["GAME_DATE"])).days)
        if age < 0:
            continue
        dists.append(_tv_distance(target_rot, hist_rot))
        ages.append(age)
        vals.append(float(h["LOCAL_RESID"]))
    return np.asarray(dists, dtype=float), np.asarray(ages, dtype=float), np.asarray(vals, dtype=float)


def _precompute_histories(rows: pd.DataFrame, rotations) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [_candidate_history_for_row(rows, rotations, i) for i in range(len(rows))]


def _predict_local_zero_prior(
    histories: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    bandwidth: float,
    tau_days: float,
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros(len(histories), dtype=float)
    mass = np.zeros(len(histories), dtype=float)
    if not np.isfinite(k):
        return pred, mass
    kk = max(float(k), 1e-9)
    for i, (d, a, y) in enumerate(histories):
        if len(y) == 0:
            continue
        w = np.exp(-d / float(bandwidth))
        if np.isfinite(tau_days):
            w = w * np.exp(-a / float(tau_days))
        m = float(np.sum(w))
        mass[i] = m
        if m > 0:
            pred[i] = float(np.sum(w * y) / (kk + m))
    return pred, mass


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(y - p))))


def _select_kernel_params(rows: pd.DataFrame, histories) -> AvailabilityHyperParam:
    feature = str(rows.attrs.get("feature", ""))
    n = len(rows)
    if n < 100:
        return AvailabilityHyperParam(feature, False, np.nan, np.inf, np.inf,
                                      np.nan, np.nan, np.nan, np.nan, 0, 0)

    y = rows["LOCAL_RESID"].to_numpy(dtype=float)
    # Structural model is the zero-residual baseline.
    zero = np.zeros(n, dtype=float)

    hold_start = max(int(n * 0.70), 70)
    tune_end = hold_start
    starts = sorted(set([max(55, int(tune_end * 0.55)), max(70, int(tune_end * 0.72))]))
    tune_idx = []
    block = max(15, int(tune_end * 0.12))
    for s in starts:
        e = min(s + block, tune_end)
        if e > s:
            tune_idx.extend(range(s, e))
    tune_idx = np.asarray(sorted(set(tune_idx)), dtype=int)
    if len(tune_idx) < 20:
        tune_idx = np.arange(max(55, int(tune_end * 0.60)), tune_end, dtype=int)

    # TV distance * 200 = shifted regulation minutes.  These h candidates span
    # roughly 10, 15, 20, 30, 40 and 60 shifted minutes.  The winning value is
    # selected from chronological predictive performance, not chosen for today's
    # matchup.
    h_grid = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
    tau_grid = [21.0, 42.0, 84.0, 168.0, np.inf]
    k_grid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]

    best = (np.inf, np.nan, np.inf, np.inf)
    for h in h_grid:
        for tau in tau_grid:
            # Cache kernel numerators/mass for all finite K because only the
            # denominator changes with K.
            numer = np.zeros(n, dtype=float)
            mass = np.zeros(n, dtype=float)
            for i, (d, a, vals) in enumerate(histories):
                if len(vals) == 0:
                    continue
                w = np.exp(-d / h)
                if np.isfinite(tau):
                    w = w * np.exp(-a / tau)
                mass[i] = float(np.sum(w))
                numer[i] = float(np.sum(w * vals))
            for k in k_grid:
                if np.isfinite(k):
                    p = numer / (float(k) + mass)
                else:
                    p = zero
                score = _rmse(y[tune_idx], p[tune_idx])
                # Tie-break toward simpler/more regularized specifications:
                # no local history, then larger K, then broader h, then tau=inf.
                complexity = (
                    0 if np.isinf(k) else 1,
                    0.0 if np.isinf(k) else -float(k),
                    -float(h),
                    0 if np.isinf(tau) else 1,
                )
                best_complexity = (
                    0 if np.isinf(best[3]) else 1,
                    0.0 if np.isinf(best[3]) else -float(best[3]),
                    -float(best[1]) if np.isfinite(best[1]) else 0.0,
                    0 if np.isinf(best[2]) else 1,
                )
                if score < best[0] - 1e-12 or (abs(score - best[0]) <= 1e-12 and complexity < best_complexity):
                    best = (score, float(h), float(tau), float(k))

    tune_rmse, h, tau, k = best
    baseline_tune = _rmse(y[tune_idx], zero[tune_idx])

    pred, mass = _predict_local_zero_prior(histories, h, tau, k)
    hold_idx = np.arange(hold_start, n, dtype=int)
    hold_rmse = _rmse(y[hold_idx], pred[hold_idx])
    baseline_hold = _rmse(y[hold_idx], zero[hold_idx])

    # Require true later-holdout improvement.  If not, history is disabled and
    # the live model falls back to the mechanistic roster prior for this feature.
    active = bool(
        np.isfinite(k)
        and np.isfinite(hold_rmse)
        and np.isfinite(baseline_hold)
        and hold_rmse < baseline_hold
    )
    if not active:
        k = np.inf

    eligible_rows = int(np.sum([len(hh[2]) > 0 for hh in histories]))
    return AvailabilityHyperParam(
        feature=feature,
        active=active,
        bandwidth=float(h),
        recency_tau_days=float(tau),
        k=float(k),
        tune_rmse=float(tune_rmse),
        baseline_tune_rmse=float(baseline_tune),
        holdout_rmse=float(hold_rmse),
        baseline_holdout_rmse=float(baseline_hold),
        holdout_rows=int(len(hold_idx)),
        eligible_prediction_rows=eligible_rows,
    )


def fit_availability_hyperparams(
    player_db: pd.DataFrame,
    structural_models: Mapping[str, object],
) -> tuple[Dict[str, AvailabilityHyperParam], pd.DataFrame]:
    """Learn local rotation-kernel h, recency tau and EB prior mass K.

    Learning is feature-specific and chronological.  Historical targets are
    predicted only from earlier same-team, different-opponent games.  This keeps
    the local availability layer disjoint from the explicit H2H residual layer.
    """
    rotations = _historical_rotations(player_db)
    out: Dict[str, AvailabilityHyperParam] = {}
    audit_rows = []

    for feature in [s[0] for s in FEATURE_SPECS.values()]:
        model = structural_models.get(feature)
        rows = _residual_table(model) if model is not None else pd.DataFrame()
        if rows.empty:
            hp = AvailabilityHyperParam(feature, False, np.nan, np.inf, np.inf,
                                        np.nan, np.nan, np.nan, np.nan, 0, 0)
        else:
            rows.attrs["feature"] = feature
            histories = _precompute_histories(rows, rotations)
            hp = _select_kernel_params(rows, histories)
        out[feature] = hp
        audit_rows.append({
            "Feature": feature,
            "Active local history": bool(hp.active),
            "Rotation bandwidth TV": hp.bandwidth,
            "Bandwidth shifted min": (hp.bandwidth * 200.0 if np.isfinite(hp.bandwidth) else np.nan),
            "Recency tau days": hp.recency_tau_days,
            "Prior-equivalent K": hp.k,
            "Tune RMSE": hp.tune_rmse,
            "Zero-residual tune RMSE": hp.baseline_tune_rmse,
            "Later holdout RMSE": hp.holdout_rmse,
            "Zero-residual holdout RMSE": hp.baseline_holdout_rmse,
            "Holdout rows": hp.holdout_rows,
            "Rows with prior same-team history": hp.eligible_prediction_rows,
            "Reason": (
                "walk-forward later-holdout improvement"
                if hp.active else
                "local history disabled; roster prior only"
            ),
        })

    return out, pd.DataFrame(audit_rows)


# ---------------------------------------------------------------------------
# Live posterior
# ---------------------------------------------------------------------------

def _live_local_evidence(
    *,
    model,
    rotations,
    team: str,
    opponent: str,
    target_rotation: Mapping[str, float],
    hp: AvailabilityHyperParam,
) -> tuple[float, float, int, float, str]:
    """Return weighted residual sum, mass, rows, avg distance and top-match audit."""
    if model is None or not hp.active or not np.isfinite(hp.k) or not target_rotation:
        return 0.0, 0.0, 0, np.nan, "—"
    rows = _residual_table(model)
    if rows.empty:
        return 0.0, 0.0, 0, np.nan, "—"

    team_rows = rows[
        rows["TEAM_ABBR"].astype(str).str.upper().eq(team)
        & ~rows["OPP_ABBR"].astype(str).str.upper().eq(opponent)
    ].copy()
    if team_rows.empty:
        return 0.0, 0.0, 0, np.nan, "—"

    # The live matchup occurs after the latest completed database game.  Only
    # relative ages matter; +1 day prevents the most recent completed game from
    # receiving age 0 while remaining agnostic to the exact scheduled tip date.
    latest = pd.to_datetime(rows["GAME_DATE"], errors="coerce").max()
    target_date = latest + pd.Timedelta(days=1) if pd.notna(latest) else pd.Timestamp.today().normalize()

    weighted_sum = 0.0
    mass = 0.0
    n = 0
    d_weighted = 0.0
    details = []
    for _, r in team_rows.iterrows():
        gid = str(r["GAME_ID"])
        hist_rot = rotations.get((team, gid), {})
        if not hist_rot:
            continue
        d = _tv_distance(target_rotation, hist_rot)
        age = max(float((target_date - pd.Timestamp(r["GAME_DATE"])).days), 0.0)
        w = _kernel_weight(d, age, hp.bandwidth, hp.recency_tau_days)
        if w <= 0 or not np.isfinite(w):
            continue
        resid = float(r["LOCAL_RESID"])
        if not np.isfinite(resid):
            continue
        weighted_sum += w * resid
        mass += w
        d_weighted += w * d
        n += 1
        details.append((w, gid, d, age, resid))

    avg_d = float(d_weighted / mass) if mass > 0 else np.nan
    top = sorted(details, key=lambda x: x[0], reverse=True)[:5]
    top_txt = ", ".join(
        f"{gid}:w={w:.2f},shift={d*200:.0f}m,age={age:.0f}d,res={resid:+.3f}"
        for w, gid, d, age, resid in top
    ) if top else "—"
    return float(weighted_sum), float(mass), int(n), avg_d, top_txt


def availability_posterior_modifiers(
    *,
    player_db: pd.DataFrame,
    team_abbr: str,
    opponent_abbr: str,
    base_profile: Mapping[str, float],
    structural_modifiers: Mapping[str, float],
    structural_models: Mapping[str, object],
    state_scores_by_stat: Mapping[str, Mapping[str, float]] | None = None,  # compatibility; intentionally unused
    rotation_impact=None,
    hyperparams: Mapping[str, AvailabilityHyperParam] | None = None,
) -> tuple[Dict[str, float], pd.DataFrame]:
    """Return Team-Market-only availability modifiers and full local-state audit.

    The live historical kernel is centered on the projected OUT-only 200-minute
    rotation.  Explicit minute restrictions remain a separate current-vs-OUT
    mechanistic delta so the same trader information is not counted twice.
    """
    if rotation_impact is None:
        return {k: 1.0 for k in FEATURE_SPECS}, pd.DataFrame()

    healthy = _rotation_rates(player_db, team_abbr, rotation_impact.healthy_minutes)
    out_only = _rotation_rates(player_db, team_abbr, rotation_impact.out_only_minutes)
    current = _rotation_rates(player_db, team_abbr, rotation_impact.current_minutes)
    target_rotation = _board_rotation(rotation_impact.out_only_minutes)
    rotations = _historical_rotations(player_db)

    mods: Dict[str, float] = {}
    rows = []
    team = str(team_abbr).upper()
    opp = str(opponent_abbr).upper()
    hyperparams = hyperparams or {}

    for out_key, (feature, profile_key, bounds) in FEATURE_SPECS.items():
        lo, hi = bounds
        healthy_rate = float(healthy.get(feature, np.nan))
        out_rate = float(out_only.get(feature, np.nan))
        current_rate = float(current.get(feature, np.nan))

        if not all(np.isfinite(v) and v > 0 for v in (healthy_rate, out_rate, current_rate)):
            mods[out_key] = 1.0
            rows.append({"Feature": feature, "Applied modifier": 1.0, "Reason": "rotation prior unavailable"})
            continue

        # Mechanistic prior mean from roster composition.  This is a fallback,
        # not the sole signal once comparable historical rotations exist.
        out_prior_delta = _transform(feature, out_rate) - _transform(feature, healthy_rate)
        restriction_delta = _transform(feature, current_rate) - _transform(feature, out_rate)

        hp = hyperparams.get(feature)
        if hp is None:
            hp = AvailabilityHyperParam(feature, False, np.nan, np.inf, np.inf,
                                        np.nan, np.nan, np.nan, np.nan, 0, 0)
        model = structural_models.get(feature)
        weighted_sum, evidence_mass, evidence_rows, avg_d, top_txt = _live_local_evidence(
            model=model,
            rotations=rotations,
            team=team,
            opponent=opp,
            target_rotation=target_rotation,
            hp=hp,
        )
        residual_mean = float(weighted_sum / evidence_mass) if evidence_mass > 0 else np.nan

        if hp.active and evidence_mass > 0 and np.isfinite(hp.k):
            k = max(float(hp.k), 1e-9)
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
            "Local history active": bool(hp.active),
            "Rotation bandwidth TV": hp.bandwidth,
            "Bandwidth shifted min": (hp.bandwidth * 200.0 if np.isfinite(hp.bandwidth) else np.nan),
            "Recency tau days": hp.recency_tau_days,
            "EB prior-equivalent K": hp.k,
            "Historical effective mass": evidence_mass,
            "Historical matched rows": evidence_rows,
            "Weighted avg rotation distance": avg_d,
            "Weighted avg shifted min": (avg_d * 200.0 if np.isfinite(avg_d) else np.nan),
            "Historical residual mean": residual_mean,
            "Historical posterior weight": empirical_weight,
            "Posterior OUT delta (transformed)": out_post_delta,
            "Total availability delta (transformed)": total_delta,
            "Pre-availability structural rate": structural_rate,
            "Post-availability rate": final_rate,
            "Applied modifier": mod,
            "Top local rotation matches": top_txt,
        })

    return mods, pd.DataFrame(rows)
