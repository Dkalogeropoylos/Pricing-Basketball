import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from core.availability import availability_state_weights
from core.team_model import TeamContext, simulate_game, h2h_profile_blend


def sim_profile(three_share=0.40, blk_rate=0.05):
    return {
        'tov_pp':0.15,
        'fga_live':0.96,
        'three_share':three_share,
        'three_pa_live':0.38,
        'two_pa_live':0.58,
        'fta_pp':0.24,
        'three_pct':0.34,
        'two_pct':0.51,
        'ft_pct':0.79,
        'oreb_per_miss':0.25,
        'assist_per_make':0.62,
        'pf_pp':0.24,
        'dreb_capture':0.94,
        'stl_per_opp_tov':0.55,
        'blk_pp':blk_rate,
        'blk_rate_pp':blk_rate,
        'league_two_pa_pp':0.54,
    }


# ------------------------------------------------------------------
# 1) Shot architecture: 3PA and 2PA must partition FGA exactly.
# ------------------------------------------------------------------
hp = sim_profile(0.46, 0.05)
ap = sim_profile(0.37, 0.05)
hctx = TeamContext(80, fga=1.0, three_share=1.0, blk=1.0)
actx = TeamContext(80, fga=1.0, three_share=1.0, blk=1.0)
h, a = simulate_game(hp, ap, hctx, actx, n=30000, seed=211)
assert ((h['FGA'] - h['3PA'] - h['2PA']) == 0).all()
assert ((a['FGA'] - a['3PA'] - a['2PA']) == 0).all()


# ------------------------------------------------------------------
# 2) BLK conservation uses total missed FGA, not only missed 2PA.
# ------------------------------------------------------------------
hp_blk = sim_profile(0.75, 0.18)
ap_3heavy = sim_profile(0.72, 0.05)
h2, a2 = simulate_game(
    hp_blk, ap_3heavy,
    TeamContext(82, blk=1.10), TeamContext(82),
    n=30000, seed=212,
)
home_missed_fga = a2['FGA'] - a2['FGM']
home_opp_missed_2pa = a2['2PA'] - a2['2PM']
assert (h2['BLK'] <= home_missed_fga).all()
# Proves the old missed-2PA cap is no longer binding in every simulation.
assert (h2['BLK'] > home_opp_missed_2pa).any()


# ------------------------------------------------------------------
# 3) Exact OUT-state weighting excludes the explicit H2H opponent.
# ------------------------------------------------------------------
dates = pd.date_range('2026-05-01', periods=10, freq='3D')
team_log = pd.DataFrame({
    'GAME_ID':[str(i) for i in range(1,11)],
    'GAME_DATE':dates,
    'TEAM_ABBR':['AAA']*10,
    'OPP_ABBR':['CCC','CCC','CCC','CCC','CCC','CCC','CCC','BBB','CCC','CCC'],
    'FGA':[65]*10,
})
# P1/P2 are present except exact-state games 6, 8, 10. Game 8 is H2H vs BBB
# and must be excluded from availability confidence when BBB is modeled separately.
rows=[]
for i, d in enumerate(dates, start=1):
    if i not in {6,8,10}:
        rows.append({'GAME_ID':str(i),'GAME_DATE':d,'TEAM_ABBR':'AAA','PLAYER_NAME':'P1','PLAYER_ID':1,'MIN':28})
        rows.append({'GAME_ID':str(i),'GAME_DATE':d,'TEAM_ABBR':'AAA','PLAYER_NAME':'P2','PLAYER_ID':2,'MIN':24})
# Establish first team appearances in game 1.
player_db = pd.DataFrame(rows)
w, audit, exact_ids = availability_state_weights(
    player_db, team_log, 'AAA', ['P1','P2'], k=6.0, exclude_opponent_abbr='BBB'
)
assert set(exact_ids) == {'6','10'}
assert int(audit.iloc[0]['Exact-state games']) == 2
assert '8' not in w
assert w['6'] > w['5']


# ------------------------------------------------------------------
# 4) Disjoint H2H blend should move a structural rate without touching
#    shooting percentages, and use a small bounded weight.
# ------------------------------------------------------------------
league_rows=[]
for gid, opp, fg3a, fga, blk in [
    ('h1','BBB',30,70,2), ('h2','BBB',28,68,3),
]:
    league_rows.append({
        'GAME_ID':gid,'GAME_DATE':'2026-07-01','TEAM_ABBR':'AAA','OPP_ABBR':opp,
        'FGM':30,'FGA':fga,'FG3M':10,'FG3A':fg3a,'FTM':15,'FTA':19,'OREB':8,'DREB':24,
        'REB':32,'AST':22,'STL':7,'BLK':blk,'TOV':13,'PF':19,'PTS':85,
    })
    league_rows.append({
        'GAME_ID':gid,'GAME_DATE':'2026-07-01','TEAM_ABBR':'BBB','OPP_ABBR':'AAA',
        'FGM':28,'FGA':66,'FG3M':8,'FG3A':22,'FTM':14,'FTA':18,'OREB':7,'DREB':25,
        'REB':32,'AST':18,'STL':6,'BLK':4,'TOV':14,'PF':20,'PTS':78,
    })
league = pd.DataFrame(league_rows)
base = sim_profile(0.34, 0.055)
base['fga_live'] = 0.90
base['three_share'] = 0.34
base['fta_pp'] = 0.22
base['tov_pp'] = 0.16
base['oreb_per_miss'] = 0.24
base['assist_per_make'] = 0.60
base['pf_pp'] = 0.23
base['dreb_capture'] = 0.92
base['stl_per_opp_tov'] = 0.50
base['three_pct'] = 0.40
blended, ha = h2h_profile_blend(league, 'AAA', 'BBB', base, rotation_similarity=0.8)
assert 0 < float(ha['Applied H2H weight'].max()) <= 0.15
assert blended['three_share'] != base['three_share']
assert blended['three_pct'] == base['three_pct']

print('v2.11 smoke: PASS')
print('shot means home 3PA/2PA/FGA', round(h['3PA'].mean(),2), round(h['2PA'].mean(),2), round(h['FGA'].mean(),2))
print('availability audit', audit.to_dict(orient='records')[0])
print('H2H weight', round(float(ha['Applied H2H weight'].max()),4))
