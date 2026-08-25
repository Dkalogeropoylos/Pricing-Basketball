import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from core.team_model import TeamContext, simulate_game
from core.pricing import most_market_calibrated


def profile(blk_rate=0.05, two_rate=0.54):
    return {
        'tov_pp':0.16, 'three_pa_pp':0.28, 'two_pa_pp':0.45, 'three_pa_live':0.34, 'two_pa_live':0.52,
        'fta_pp':0.24, 'three_pct':0.34, 'two_pct':0.51, 'ft_pct':0.79,
        'oreb_per_miss':0.25, 'assist_per_make':0.62, 'pf_pp':0.24,
        'dreb_capture':0.94, 'stl_per_opp_tov':0.55,
        'blk_pp':blk_rate, 'blk_rate_pp':blk_rate,
        'league_two_pa_pp':two_rate,
    }

# BLK should respond strongly enough to opponent block susceptibility but only mildly to 2PA mix.
hp=profile(0.05); ap=profile(0.05)
neutral=TeamContext(80, blk=1.0, blk_h2h=1.0)
high_allow=TeamContext(80, blk=1.06, blk_h2h=1.0)

h0,a0=simulate_game(hp,ap,neutral,neutral,n=50000,seed=11)
h1,a1=simulate_game(hp,ap,high_allow,neutral,n=50000,seed=11)
assert h1['BLK'].mean() > h0['BLK'].mean()

# Most-market calibration synthetic league history.
rows=[]
for gid in range(80):
    # deliberately moderate difference spread with plenty of ties
    x = 8 + (gid % 3)
    y = 8 + ((gid+1) % 3)
    if gid % 5 == 0:
        y=x
    rows += [
        {'GAME_ID':str(gid),'FG3M':x,'FG3A':25,'FGM':30,'FGA':65,'FTM':15,'FTA':19,'REB':34,'OREB':8,'DREB':26,'AST':21,'STL':7,'BLK':4,'TOV':13,'PF':19},
        {'GAME_ID':str(gid),'FG3M':y,'FG3A':25,'FGM':30,'FGA':65,'FTM':15,'FTA':19,'REB':34,'OREB':8,'DREB':26,'AST':21,'STL':7,'BLK':4,'TOV':13,'PF':19},
    ]
league=pd.DataFrame(rows)
rng=np.random.default_rng(3)
h=rng.poisson(10,50000)
a=rng.poisson(8,50000)
pr=most_market_calibrated(h,a,league,'3PM')
assert pr['calibrated'] is True
assert abs(pr['p_home']+pr['p_tie']+pr['p_away']-1) < 1e-9
assert pr['calibration_games'] >= 30
print('v2.10 smoke: PASS')
print('BLK neutral/high-allowed', round(h0['BLK'].mean(),3), round(h1['BLK'].mean(),3))
print('Most tie raw/cal/league', round(pr['raw_p_tie'],3), round(pr['p_tie'],3), round(pr['league_tie_rate'],3))
