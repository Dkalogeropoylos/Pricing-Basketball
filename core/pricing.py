import numpy as np
import pandas as pd


def _fair_decimal(win_prob: float, push_prob: float = 0.0) -> float:
    """
    Fair decimal odds with push return handled correctly:
        EV = p_win * odds + p_push - 1 = 0
        odds = (1 - p_push) / p_win
    For half-lines p_push=0 and this reduces to 1/p.
    """
    return (1.0 - push_prob) / win_prob if win_prob > 0 else np.inf


def price(values, line, over_odds, under_odds):
    x = np.asarray(values)
    po = float(np.mean(x > line))
    pu = float(np.mean(x < line))
    pp = max(0.0, 1.0 - po - pu)
    return {
        "p_over": po,
        "p_under": pu,
        "p_push": pp,
        "fair_over": _fair_decimal(po, pp),
        "fair_under": _fair_decimal(pu, pp),
        "be_over": (1.0 - pp) / over_odds,
        "be_under": (1.0 - pp) / under_odds,
        "ev_over": po * over_odds + pp - 1.0,
        "ev_under": pu * under_odds + pp - 1.0,
    }


def model_line(values):
    """
    Convert a simulated distribution into a sportsbook-style HALF line.

    We do not simply round the mean. For discrete basketball stats the mean can
    sit away from the 50th percentile. Candidate x.5 lines are searched and the
    line with the most balanced model probabilities is selected.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "projection": np.nan,
            "median": np.nan,
            "line": np.nan,
            "p_over": np.nan,
            "p_under": np.nan,
            "fair_over": np.nan,
            "fair_under": np.nan,
        }

    mean = float(np.mean(x))
    median = float(np.median(x))
    lo = int(np.floor(np.quantile(x, 0.03))) - 1
    hi = int(np.ceil(np.quantile(x, 0.97))) + 1
    candidates = np.arange(lo, hi + 1, dtype=float) + 0.5

    best = None
    for line in candidates:
        po = float(np.mean(x > line))
        pu = float(np.mean(x < line))
        score = (
            abs(po - pu),
            abs(line - median),
            abs(line - mean),
        )
        if best is None or score < best[0]:
            best = (score, float(line), po, pu)

    _, line, po, pu = best
    return {
        "projection": mean,
        "median": median,
        "line": line,
        "p_over": po,
        "p_under": pu,
        "fair_over": 1.0 / po if po > 0 else np.inf,
        "fair_under": 1.0 / pu if pu > 0 else np.inf,
    }


def auto_market_table(sim, markets, stress_low=None, stress_high=None):
    rows = []
    for market in markets:
        if market not in sim.columns:
            continue
        ml = model_line(sim[market].to_numpy())
        row = {
            "Market": market,
            "Projection": ml["projection"],
            "Median": ml["median"],
            "Model line": ml["line"],
            "P Over": ml["p_over"],
            "Fair Over": ml["fair_over"],
            "P Under": ml["p_under"],
            "Fair Under": ml["fair_under"],
        }
        if stress_low is not None and market in stress_low.columns:
            row["Low projection"] = float(np.mean(stress_low[market]))
            row["Low P Over @ line"] = float(np.mean(stress_low[market].to_numpy() > ml["line"]))
        if stress_high is not None and market in stress_high.columns:
            row["High projection"] = float(np.mean(stress_high[market]))
            row["High P Over @ line"] = float(np.mean(stress_high[market].to_numpy() > ml["line"]))
        rows.append(row)
    return pd.DataFrame(rows)


def most_market(home_values, away_values):
    h = np.asarray(home_values)
    a = np.asarray(away_values)
    n = min(len(h), len(a))
    h, a = h[:n], a[:n]
    ph = float(np.mean(h > a))
    pa = float(np.mean(a > h))
    pt = float(np.mean(a == h))
    return {
        "p_home": ph,
        "p_tie": pt,
        "p_away": pa,
        "fair_home": 1.0 / ph if ph > 0 else np.inf,
        "fair_tie": 1.0 / pt if pt > 0 else np.inf,
        "fair_away": 1.0 / pa if pa > 0 else np.inf,
    }


def market_table(sim, stress_low, stress_high, markets):
    """Legacy compact projection table."""
    rows = []
    for market in markets:
        if market not in sim.columns:
            continue
        x = sim[market].to_numpy()
        rows.append({
            "Market": market,
            "Mean": float(np.mean(x)),
            "Median": float(np.median(x)),
            "Low mean": float(np.mean(stress_low[market])) if market in stress_low.columns else np.nan,
            "High mean": float(np.mean(stress_high[market])) if market in stress_high.columns else np.nan,
        })
    return pd.DataFrame(rows)
