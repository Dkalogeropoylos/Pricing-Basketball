import numpy as np
import pandas as pd


def price(values, line, over_odds, under_odds):
    x=np.asarray(values)
    po=float(np.mean(x>line))
    pu=float(np.mean(x<line))
    pp=max(0.0,1-po-pu)
    return {
        "p_over":po,"p_under":pu,"p_push":pp,
        "fair_over":1/po if po else np.inf,
        "fair_under":1/pu if pu else np.inf,
        "be_over":1/over_odds,
        "be_under":1/under_odds,
        "ev_over":po*over_odds+pp-1,
        "ev_under":pu*under_odds+pp-1,
    }


def market_table(sim, stress_low, stress_high, markets):
    rows=[]
    for market in markets:
        x=sim[market].to_numpy()
        rows.append({
            "Market":market,
            "Mean":float(np.mean(x)),
            "Median":float(np.median(x)),
            "Low mean":float(np.mean(stress_low[market])),
            "High mean":float(np.mean(stress_high[market])),
        })
    return pd.DataFrame(rows)
