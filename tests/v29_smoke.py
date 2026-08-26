from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.buckets import WeightConfig
from core.pricing import model_line, required_odds_for_ev, line_ladder
from core.team_model import TeamContext, build_team_profile, simulate_game
from core.role_splits import same_role_game_weights
from core.player_model import build_player_profile, PlayerContext, simulate_player


def team_db(seed=1):
    rng=np.random.default_rng(seed); rows=[]
    for i in range(24):
        gid=str(1000+i); d=pd.Timestamp('2026-05-01')+pd.Timedelta(days=i)
        for team,opp,a3mu in [('AAA','BBB',24),('BBB','AAA',30)]:
            fga=max(55,int(rng.normal(69,3))); a3=max(10,min(fga-15,int(rng.normal(a3mu,2.5))))
            a2=fga-a3; m3=rng.binomial(a3,.35); m2=rng.binomial(a2,.50)
            fta=max(5,int(rng.normal(19,3))); ftm=rng.binomial(fta,.79)
            tov=max(6,int(rng.normal(13,2))); oreb=max(3,int(rng.normal(9,2))); dreb=max(18,int(rng.normal(25,3)))
            ast=max(8,int(rng.normal(20,3))); stl=max(1,int(rng.normal(7,1.5))); blk=max(0,int(rng.normal(4,1))); pf=max(10,int(rng.normal(19,2)))
            rows.append(dict(GAME_ID=gid,GAME_DATE=d,TEAM_ABBR=team,OPP_ABBR=opp,FGM=m3+m2,FGA=fga,FG3M=m3,FG3A=a3,FTM=ftm,FTA=fta,OREB=oreb,DREB=dreb,REB=oreb+dreb,AST=ast,STL=stl,BLK=blk,TOV=tov,PF=pf,PTS=3*m3+2*m2+ftm,MATCHUP=f'{team} vs. {opp}'))
    return pd.DataFrame(rows)


def player_db():
    rows=[]
    for i in range(15):
        gid=str(2000+i); d=pd.Timestamp('2026-06-01')+pd.Timedelta(days=i)
        # focal player; higher volume in games where teammate X is absent (last 6)
        out_state=i>=9
        rows.append(dict(GAME_ID=gid,GAME_DATE=d,PLAYER_ID=1,PLAYER_NAME='Focal',TEAM_ABBR='AAA',OPP_ABBR='BBB',MIN=32,FGM=7,FGA=15 if not out_state else 18,FG3M=2,FG3A=5 if not out_state else 7,FTM=4,FTA=5,REB=5,AST=4 if not out_state else 6,PTS=20 if not out_state else 24))
        if not out_state:
            rows.append(dict(GAME_ID=gid,GAME_DATE=d,PLAYER_ID=2,PLAYER_NAME='Teammate X',TEAM_ABBR='AAA',OPP_ABBR='BBB',MIN=30,FGM=5,FGA=11,FG3M=2,FG3A=5,FTM=2,FTA=2,REB=4,AST=3,PTS=14))
    return pd.DataFrame(rows)


def main():
    db=team_db()
    pa,_=build_team_profile(db[db.TEAM_ABBR=='AAA'],WeightConfig.stable(),db)
    pb,_=build_team_profile(db[db.TEAM_ABBR=='BBB'],WeightConfig.stable(),db)
    assert 'blk_per_opp_2pa' in pa and 'blk_per_opp_2miss' not in pa
    ctx=TeamContext(projected_possessions=80)
    h,a=simulate_game(pa,pb,ctx,ctx,n=20000,seed=3)
    assert (h.BLK <= (a['FGA']-a['FGM'])).all()
    assert (a.BLK <= (h['FGA']-h['FGM'])).all()
    assert ((h.FGA-h['2PA']-h['3PA'])==0).all()

    # Fair 1.70 with 6% target EV should require ~1.802.
    req=required_odds_for_ev(1/1.70,target_ev=.06)
    assert abs(req-1.802)<0.002

    pdb=player_db(); focal=pdb[pdb.PLAYER_NAME=='Focal'].copy()
    weights,audit=same_role_game_weights(focal,pdb,'AAA',['Teammate X'],True)
    assert int(audit.iloc[0]['Same-role games'])==6
    assert abs(np.mean(list(weights.values()))-1)<1e-9
    prof0,_=build_player_profile(focal,WeightConfig.role_change())
    prof1,_=build_player_profile(focal,WeightConfig.role_change(),game_weights=weights)
    assert prof1['three_pa_pm'] > prof0['three_pa_pm']

    sim=simulate_player(prof1,PlayerContext(projected_minutes=34),n=15000,seed=8)
    ladder=line_ladder(sim['PTS'],target_ev=.06)
    assert not ladder.empty and any('Play Over from' in c for c in ladder.columns)
    print('v2.9 smoke: PASS')
    print('block rates', round(pa['blk_per_opp_2pa'],4), round(pb['blk_per_opp_2pa'],4))
    print('same-role 3PA/min', round(prof0['three_pa_pm'],4), '->', round(prof1['three_pa_pm'],4))
    print('play-from example', round(req,3))

if __name__=='__main__': main()
