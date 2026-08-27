from __future__ import annotations

"""Walk-forward calibration for the main Team-Market structural rates.

v2.17 replaces fixed opponent/H2H shrinkage for the four most important
possession-allocation rates with a league-level empirical model:

    3P_SHARE      : 3PA / FGA
    FTA_RATE      : FTA / possessions
    TOV_RATE      : TOV / possessions
    AST_PER_MAKE  : AST / FGM   (WNBA/NBA official-stat convention)

The model is intentionally league-level rather than team-specific.  Every
historical team-game is converted into a *pregame* training row using only
information available before that game:

  - own Old / G6-10 / L5 (non-overlapping), excluding the current opponent;
  - opponent-allowed Old / G6-10 / L5, excluding the current offense;
  - prior same-season H2H, kept disjoint from both baselines;
  - the contemporaneous league baseline.

Coefficients are learned with ridge regression in transformed space.  The ridge
penalty is selected by expanding-window validation; the learned model is used
only when its walk-forward RMSE beats the existing Old/G6-10/L5 baseline.
There is therefore no manually chosen "opponent weight" or "H2H weight" for
these four rates.

Current-roster/OUT and location modifiers remain separate downstream layers.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from core.buckets import WeightConfig, split_non_overlapping, active_weights


FEATURES = ("3P_SHARE", "FTA", "TOV", "AST_PER_MAKE")
X_COLS = (
    "OWN_OLD", "OWN_MID", "OWN_L5",
    "OPP_OLD", "OPP_MID", "OPP_L5",
    "H2H_DEV", "H2H_DEV_LOGN",
)


@dataclass
class StructuralModel:
    feature: str
    active: bool
    beta: np.ndarray
    x_mean: np.ndarray
    x_sd: np.ndarray
    ridge_lambda: float
    rows: int
    cv_rmse: float
    baseline_cv_rmse: float
    coefficients: Dict[str, float]


def _estimate_possessions(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df["FGA"], errors="coerce").fillna(0.0)
        - pd.to_numeric(df["OREB"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["TOV"], errors="coerce").fillna(0.0)
        + 0.44 * pd.to_numeric(df["FTA"], errors="coerce").fillna(0.0)
    )


def _single_game_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in ["FGA", "FGM", "FG3A", "FTA", "TOV", "AST", "OREB"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    poss = _estimate_possessions(x).clip(lower=1e-6)
    out = pd.DataFrame({
        "GAME_DATE": pd.to_datetime(x["GAME_DATE"], errors="coerce"),
        "GAME_ID": x["GAME_ID"].astype(str),
        "TEAM_ABBR": x["TEAM_ABBR"].astype(str).str.upper(),
        "OPP_ABBR": x["OPP_ABBR"].astype(str).str.upper(),
        "3P_SHARE": x["FG3A"] / x["FGA"].replace(0, np.nan),
        "FTA": x["FTA"] / poss,
        "TOV": x["TOV"] / poss,
        # WNBA/NBA official assists are tied to made field goals.  FIBA/
        # EuroLeague has a different scoring-stat rule for passes leading to
        # shooting fouls; that requires a league-specific PBP event model and
        # must NOT be mixed into WNBA box-score AST.
        "AST_PER_MAKE": x["AST"] / x["FGM"].replace(0, np.nan),
    })
    return out.replace([np.inf, -np.inf], np.nan)


def _is_probability(feature: str) -> bool:
    return feature in {"3P_SHARE", "TOV", "AST_PER_MAKE"}


def _transform(feature: str, v):
    a = np.asarray(v, dtype=float)
    if _is_probability(feature):
        a = np.clip(a, 1e-4, 1.0 - 1e-4)
        return np.log(a / (1.0 - a))
    return np.log(np.clip(a, 1e-4, None))


def _inverse(feature: str, z):
    a = np.asarray(z, dtype=float)
    if _is_probability(feature):
        a = np.clip(a, -12.0, 12.0)
        return 1.0 / (1.0 + np.exp(-a))
    return np.exp(np.clip(a, -12.0, 12.0))


def _mean_feature(df: pd.DataFrame, feature: str, default=np.nan) -> float:
    if df is None or df.empty or feature not in df.columns:
        return float(default)
    v = pd.to_numeric(df[feature], errors="coerce").dropna()
    return float(v.mean()) if len(v) else float(default)


def _bucket_triplet(df: pd.DataFrame, feature: str, league: float) -> Tuple[float, float, float]:
    if df is None or df.empty:
        return float(league), float(league), float(league)
    b = split_non_overlapping(df.sort_values("GAME_DATE"))
    return tuple(_mean_feature(b[k], feature, league) for k in ("old", "mid", "l5"))


def _cfg_baseline(df: pd.DataFrame, feature: str, league: float, cfg: WeightConfig) -> float:
    """Existing outer-bucket baseline, used only as the benchmark/denominator."""
    if df is None or df.empty:
        return float(league)
    buckets = split_non_overlapping(df.sort_values("GAME_DATE"))
    weights = active_weights(buckets, cfg)
    vals, ws = [], []
    for k in ("old", "mid", "l5"):
        v = _mean_feature(buckets[k], feature, np.nan)
        if np.isfinite(v) and weights.get(k, 0.0) > 0:
            vals.append(v); ws.append(weights[k])
    return float(np.average(vals, weights=ws)) if vals else float(league)


def _x_from_histories(
    own_hist: pd.DataFrame,
    opp_allowed_hist: pd.DataFrame,
    h2h_hist: pd.DataFrame,
    feature: str,
    league: float,
    h2h_rotation_similarity: float = 1.0,
) -> np.ndarray:
    lg_t = float(_transform(feature, [league])[0])
    own = _bucket_triplet(own_hist, feature, league)
    opp = _bucket_triplet(opp_allowed_hist, feature, league)

    own_dev = [float(_transform(feature, [v])[0] - lg_t) for v in own]
    opp_dev = [float(_transform(feature, [v])[0] - lg_t) for v in opp]

    h_n = int(len(h2h_hist)) if h2h_hist is not None else 0
    h_val = _mean_feature(h2h_hist, feature, league) if h_n else float(league)
    h_dev = float(_transform(feature, [h_val])[0] - lg_t)
    # Current rotation similarity is a factual matchup-quality input, not a
    # fitted coefficient.  The *magnitude* of the H2H response is learned by
    # the league-wide regression below.
    h_dev *= float(np.clip(h2h_rotation_similarity, 0.0, 1.0))
    h_logn = h_dev * float(np.log1p(h_n))

    return np.asarray([*own_dev, *opp_dev, h_dev, h_logn], dtype=float)


def _training_table(team_logs: pd.DataFrame, feature: str) -> pd.DataFrame:
    g = _single_game_features(team_logs).dropna(subset=["GAME_DATE", feature]).copy()
    g = g.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBR"]).reset_index(drop=True)
    if g.empty:
        return pd.DataFrame()

    records = []
    # Process by game so neither team row from the target game can leak into the
    # other side's pregame feature set.
    games = (
        g[["GAME_DATE", "GAME_ID"]]
        .drop_duplicates()
        .sort_values(["GAME_DATE", "GAME_ID"])
        .itertuples(index=False, name=None)
    )
    prior = g.iloc[0:0].copy()
    for game_date, game_id in games:
        cur = g[(g["GAME_DATE"] == game_date) & (g["GAME_ID"] == game_id)]
        if len(prior) >= 80:
            league = _mean_feature(prior, feature, np.nan)
            if np.isfinite(league):
                for _, r in cur.iterrows():
                    team = str(r["TEAM_ABBR"]).upper()
                    opp = str(r["OPP_ABBR"]).upper()
                    own_hist = prior[prior["TEAM_ABBR"].eq(team) & ~prior["OPP_ABBR"].eq(opp)]
                    # Outcomes historically ALLOWED by today's opponent, with
                    # current-offense H2H removed to keep the layer disjoint.
                    opp_hist = prior[prior["OPP_ABBR"].eq(opp) & ~prior["TEAM_ABBR"].eq(team)]
                    h2h = prior[prior["TEAM_ABBR"].eq(team) & prior["OPP_ABBR"].eq(opp)]
                    if len(own_hist) < 8 or len(opp_hist) < 8:
                        continue
                    xv = _x_from_histories(own_hist, opp_hist, h2h, feature, league, 1.0)
                    actual = float(r[feature])
                    if not np.isfinite(actual):
                        continue
                    y = float(_transform(feature, [actual])[0] - _transform(feature, [league])[0])
                    stable = _cfg_baseline(own_hist, feature, league, WeightConfig.stable())
                    base_y = float(_transform(feature, [stable])[0] - _transform(feature, [league])[0])
                    records.append({
                        "GAME_DATE": game_date, "GAME_ID": str(game_id),
                        "TEAM_ABBR": team, "OPP_ABBR": opp,
                        **{k: float(v) for k, v in zip(X_COLS, xv)},
                        "Y": y, "BASE_Y": base_y,
                    })
        prior = pd.concat([prior, cur], ignore_index=True)
    return pd.DataFrame(records)


def _fit_ridge(X: np.ndarray, y: np.ndarray, lam: float):
    mean = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mean) / sd
    A = np.column_stack([np.ones(len(Z)), Z])
    pen = np.eye(A.shape[1]); pen[0, 0] = 0.0
    beta = np.linalg.pinv(A.T @ A + float(lam) * pen) @ (A.T @ y)
    return beta, mean, sd


def _predict(beta, mean, sd, X):
    Z = (X - mean) / sd
    A = np.column_stack([np.ones(len(Z)), Z])
    return A @ beta


def _walk_forward_lambda(df: pd.DataFrame) -> Tuple[float, float, float]:
    X = df[list(X_COLS)].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    base = df["BASE_Y"].to_numpy(dtype=float)
    n = len(df)
    if n < 90:
        return 1.0, np.nan, np.nan

    # Broad logarithmic grid; the DATA choose the penalty.  No one penalty is
    # hard-wired into the model.
    lambdas = np.logspace(-4, 2, 13)
    starts = sorted(set([int(n * q) for q in (0.55, 0.68, 0.81)]))
    val_size = max(20, int(n * 0.12))
    folds = [(s, min(s + val_size, n)) for s in starts if s >= 60 and s < n - 10]
    if not folds:
        return 1.0, np.nan, np.nan

    best_lam, best_rmse = None, np.inf
    all_val_idx = []
    for lam in lambdas:
        errs = []
        for s, e in folds:
            beta, mean, sd = _fit_ridge(X[:s], y[:s], lam)
            pred = _predict(beta, mean, sd, X[s:e])
            errs.extend((y[s:e] - pred).tolist())
            if lam == lambdas[0]:
                all_val_idx.extend(range(s, e))
        rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else np.inf
        if rmse < best_rmse:
            best_rmse, best_lam = rmse, float(lam)

    idx = np.asarray(sorted(set(all_val_idx)), dtype=int)
    baseline_rmse = float(np.sqrt(np.mean((y[idx] - base[idx]) ** 2))) if len(idx) else np.nan
    return float(best_lam), float(best_rmse), baseline_rmse


def fit_structural_rate_models(team_logs: pd.DataFrame):
    models: Dict[str, StructuralModel] = {}
    audit_rows = []
    for feature in FEATURES:
        df = _training_table(team_logs, feature)
        if len(df) < 90:
            model = StructuralModel(feature, False, np.zeros(1 + len(X_COLS)), np.zeros(len(X_COLS)),
                                    np.ones(len(X_COLS)), 1.0, len(df), np.nan, np.nan, {})
            models[feature] = model
            audit_rows.append({"Feature": feature, "Active": False, "Rows": len(df), "Reason": "insufficient walk-forward rows"})
            continue

        lam, cv_rmse, base_rmse = _walk_forward_lambda(df)
        X = df[list(X_COLS)].to_numpy(dtype=float)
        y = df["Y"].to_numpy(dtype=float)
        beta, mean, sd = _fit_ridge(X, y, lam)
        # Activate only when the learned structure improves genuinely unseen
        # games versus the existing stable Old/G6-10/L5 baseline.
        active = bool(np.isfinite(cv_rmse) and np.isfinite(base_rmse) and cv_rmse < base_rmse)
        coeff = {"Intercept": float(beta[0])}
        for j, name in enumerate(X_COLS):
            # Coefficient in standardized-X space; audit only.
            coeff[name] = float(beta[j + 1])
        model = StructuralModel(feature, active, beta, mean, sd, lam, len(df), cv_rmse, base_rmse, coeff)
        models[feature] = model
        audit_rows.append({
            "Feature": feature, "Active": active, "Rows": len(df),
            "Selected ridge lambda": lam, "Walk-forward RMSE": cv_rmse,
            "Existing baseline RMSE": base_rmse,
            "Relative RMSE gain": ((base_rmse - cv_rmse) / base_rmse) if np.isfinite(base_rmse) and base_rmse > 0 else np.nan,
            "Reason": "walk-forward improvement" if active else "no out-of-sample improvement",
        })
    return models, pd.DataFrame(audit_rows)


def predict_structural_modifiers(
    team_logs: pd.DataFrame,
    team_abbr: str,
    opponent_abbr: str,
    models: Dict[str, StructuralModel],
    cfg: WeightConfig,
    h2h_rotation_similarity: float = 1.0,
):
    """Return empirically learned structural modifiers for the current matchup.

    The modifier is learned_prediction / existing non-H2H outer-bucket baseline.
    Multiplying it by the live profile therefore preserves current availability /
    roster-similarity adjustments while replacing the fixed recent/opponent/H2H
    weights with the walk-forward learned structural signal.
    """
    g = _single_game_features(team_logs).dropna(subset=["GAME_DATE"]).copy()
    team = str(team_abbr).upper(); opp = str(opponent_abbr).upper()
    own_hist = g[g["TEAM_ABBR"].eq(team) & ~g["OPP_ABBR"].eq(opp)].copy()
    opp_hist = g[g["OPP_ABBR"].eq(opp) & ~g["TEAM_ABBR"].eq(team)].copy()
    h2h = g[g["TEAM_ABBR"].eq(team) & g["OPP_ABBR"].eq(opp)].copy()

    mods = {"3P_SHARE": 1.0, "FTA": 1.0, "TOV": 1.0, "AST": 1.0}
    audit = []
    mapping = {"3P_SHARE": "3P_SHARE", "FTA": "FTA", "TOV": "TOV", "AST_PER_MAKE": "AST"}

    for feature in FEATURES:
        model = models.get(feature)
        league = _mean_feature(g, feature, np.nan)
        baseline = _cfg_baseline(own_hist, feature, league, cfg) if np.isfinite(league) else np.nan
        pred = baseline
        if model is not None and model.active and np.isfinite(league) and len(own_hist) >= 8 and len(opp_hist) >= 8:
            xv = _x_from_histories(
                own_hist, opp_hist, h2h, feature, league,
                h2h_rotation_similarity=h2h_rotation_similarity,
            ).reshape(1, -1)
            dev = float(_predict(model.beta, model.x_mean, model.x_sd, xv)[0])
            pred = float(_inverse(feature, [_transform(feature, [league])[0] + dev])[0])
        # Physical simulator bounds only; no matchup-response cap.
        if feature == "3P_SHARE": pred = float(np.clip(pred, 0.06, 0.75))
        elif feature == "TOV": pred = float(np.clip(pred, 0.03, 0.30))
        elif feature == "FTA": pred = float(np.clip(pred, 0.05, 0.55))
        elif feature == "AST_PER_MAKE": pred = float(np.clip(pred, 0.20, 0.95))
        mod = float(pred / baseline) if np.isfinite(pred) and np.isfinite(baseline) and baseline > 0 else 1.0
        mods[mapping[feature]] = mod
        audit.append({
            "Feature": feature,
            "Model active": bool(model.active) if model else False,
            "Existing non-H2H baseline": baseline,
            "Learned current prediction": pred,
            "Applied modifier": mod,
            "H2H games": int(len(h2h)),
            "H2H rotation similarity": float(h2h_rotation_similarity),
            "Training rows": int(model.rows) if model else 0,
            "Walk-forward RMSE": float(model.cv_rmse) if model else np.nan,
            "Baseline RMSE": float(model.baseline_cv_rmse) if model else np.nan,
        })
    return mods, pd.DataFrame(audit)


def coefficient_audit(models: Dict[str, StructuralModel]) -> pd.DataFrame:
    rows = []
    for feature, model in models.items():
        for name, value in model.coefficients.items():
            rows.append({
                "Feature": feature, "Term": name, "Coefficient": value,
                "Active": model.active, "Rows": model.rows,
                "Ridge lambda": model.ridge_lambda,
            })
    return pd.DataFrame(rows)
