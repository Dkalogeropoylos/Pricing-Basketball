"""Offline smoke test for the v2.8 coupled Team Markets engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.buckets import WeightConfig
from core.pricing import model_line, price
from core.team_model import TeamContext, build_team_profile, simulate_game


def synthetic_db(seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-05-01")
    for i in range(25):
        gid = str(9000 + i)
        date = start + pd.Timedelta(days=i)
        for tid, team, opp, base3, home in [
            (1, "AAA", "BBB", 25, True),
            (2, "BBB", "AAA", 31, False),
        ]:
            fga = max(50, int(rng.normal(68, 4)))
            a3 = min(fga - 15, max(10, int(rng.normal(base3, 3))))
            a2 = fga - a3
            m3 = rng.binomial(a3, 0.34)
            m2 = rng.binomial(a2, 0.51)
            fgm = m3 + m2
            fta = max(3, int(rng.normal(20, 4)))
            ftm = rng.binomial(fta, 0.79)
            tov = max(5, int(rng.normal(13, 2.5)))
            oreb = max(2, int(rng.normal(9, 2)))
            dreb = max(15, int(rng.normal(25, 3)))
            ast = min(fgm, max(7, int(rng.normal(21, 3))))
            stl = max(1, int(rng.normal(7, 1.5)))
            blk = max(0, int(rng.normal(4, 1.2)))
            pf = max(8, int(rng.normal(19, 3)))
            rows.append({
                "GAME_ID": gid,
                "GAME_DATE": date,
                "TEAM_ID": tid,
                "TEAM_ABBR": team,
                "OPP_ABBR": opp,
                "FGM": fgm,
                "FGA": fga,
                "FG3M": m3,
                "FG3A": a3,
                "FTM": ftm,
                "FTA": fta,
                "OREB": oreb,
                "DREB": dreb,
                "REB": oreb + dreb,
                "AST": ast,
                "STL": stl,
                "BLK": blk,
                "TOV": tov,
                "PF": pf,
                "PTS": 2 * m2 + 3 * m3 + ftm,
                "MATCHUP": f"{team} vs. {opp}" if home else f"{team} @ {opp}",
            })
    return pd.DataFrame(rows)


def main():
    db = synthetic_db()
    pa, _ = build_team_profile(
        db[db.TEAM_ABBR == "AAA"], WeightConfig.stable(), db
    )
    pb, _ = build_team_profile(
        db[db.TEAM_ABBR == "BBB"], WeightConfig.stable(), db
    )
    ctx = TeamContext(projected_possessions=80.0, possessions_sd=3.0)
    home, away = simulate_game(pa, pb, ctx, ctx, n=30_000, seed=22)

    assert ((home.FGA - home["2PA"] - home["3PA"]) == 0).all()
    assert ((away.FGA - away["2PA"] - away["3PA"]) == 0).all()
    assert ((home.REB - home.OREB - home.DREB) == 0).all()
    assert ((away.REB - away.OREB - away.DREB) == 0).all()
    assert (home.STL <= away.TOV).all()
    assert (away.STL <= home.TOV).all()
    assert (home.BLK <= (away["2PA"] - away["2PM"])).all()
    assert (away.BLK <= (home["2PA"] - home["2PM"])).all()
    assert (
        home.PTS == 3 * home["3PM"] + 2 * home["2PM"] + home.FTM
    ).all()

    ml = model_line(away["3PA"])
    assert ml["line"] % 1 == 0.5

    # Push-aware fair pricing check: 50% win, 20% push, 30% loss -> fair 1.60.
    x = np.asarray([2] * 5 + [1] * 2 + [0] * 3)
    p = price(x, 1, 1.90, 1.90)
    assert abs(p["fair_over"] - 1.60) < 1e-9

    print("v2.8 team engine smoke test: PASS")
    print("AAA mean PTS:", round(float(home.PTS.mean()), 2))
    print("BBB mean 3PA:", round(float(away["3PA"].mean()), 2))
    print("BBB model 3PA line:", ml["line"], "fair O/U:", round(ml["fair_over"], 3), round(ml["fair_under"], 3))


if __name__ == "__main__":
    main()
