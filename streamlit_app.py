from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import streamlit as st

from config.leagues import LEAGUES
from providers.router import get_provider
from core.cleaning import clean_player_log, clean_team_log
from core.buckets import WeightConfig
from core.player_model import PlayerContext, build_player_profile, simulate_player
from core.team_model import TeamContext, build_team_profile, simulate_team
from core.pricing import price, market_table


st.set_page_config(page_title="Basketball Pricing Engine", page_icon="🏀", layout="wide")
st.title("🏀 Basketball Pricing Engine v2")
st.caption("Projection first • Odds second • No L5/L10 double counting • Joint Monte Carlo")

# ---------------- Security ----------------
def get_secret(name: str):
    try:
        return st.secrets.get(name, None)
    except Exception:
        return os.getenv(name)

bdl_key=get_secret("BALLDONTLIE_API_KEY")

with st.sidebar:
    st.header("Data / Security")
    st.write("BALLDONTLIE key:", "✅ configured" if bdl_key else "⚪ not configured")
    st.caption("API keys are never rendered. Store them only in Streamlit Secrets.")
    st.divider()
    league=st.selectbox("League", ["WNBA","NBA","EuroLeague","EuroCup"])
    season=st.number_input("Season", min_value=2000, max_value=2100, value=2026, step=1)

    if league in ("EuroLeague","EuroCup"):
        st.info("Adapter structure is ready. League will be enabled after NBA/WNBA validation.")
        st.stop()

# Cached provider calls keep free/trial API usage low.
@st.cache_data(ttl=3600, show_spinner=False)
def cached_players(league, season, has_bdl_key):
    provider,name=get_provider(league, bdl_key if has_bdl_key else None)
    if hasattr(provider,"players"):
        try:
            if league=="WNBA" and has_bdl_key:
                return provider.players(), name
            return provider.players(season), name
        except TypeError:
            return provider.players(), name
    return pd.DataFrame(),name

@st.cache_data(ttl=3600, show_spinner=False)
def cached_player_log(league, season, player_id, has_bdl_key):
    provider,name=get_provider(league, bdl_key if has_bdl_key else None)
    return provider.player_game_log(int(player_id),int(season)),name

@st.cache_data(ttl=3600, show_spinner=False)
def cached_team_log_wnba(season, team_id, has_bdl_key):
    provider,name=get_provider("WNBA", bdl_key if has_bdl_key else None)
    if not hasattr(provider,"team_game_log"):
        raise RuntimeError("Automatic team logs currently require BALLDONTLIE WNBA.")
    return provider.team_game_log(int(team_id),int(season)),name

@st.cache_data(ttl=3600, show_spinner=False)
def cached_advanced_player(season, player_id, measure, has_bdl_key):
    if not has_bdl_key:
        return []
    provider,_=get_provider("WNBA",bdl_key)
    return provider.player_season_advanced(int(season),int(player_id),measure)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_advanced_team(season, team_id, measure, has_bdl_key):
    if not has_bdl_key:
        return []
    provider,_=get_provider("WNBA",bdl_key)
    return provider.team_season_advanced(int(season),int(team_id),measure)

tab_game, tab_team, tab_player, tab_audit = st.tabs(
    ["🎯 Game Setup","🏟️ Team Markets","👤 Player Props","🔎 Data Audit"]
)

# ---------------- Game Context ----------------
with tab_game:
    st.subheader("Game context")
    st.write(
        "The API supplies historical data. You supply only pre-game information the API cannot reliably know: "
        "availability, projected minutes, starting role and role redistribution."
    )

    c1,c2=st.columns(2)
    with c1:
        home=st.text_input("Home team abbreviation", value="")
    with c2:
        away=st.text_input("Away team abbreviation", value="")

    st.markdown("#### Optional context JSON")
    st.caption("Use this only for injuries / minutes / role changes. Never put API keys in this JSON.")
    upload=st.file_uploader("Upload game_context.json",type=["json"])
    context={}
    if upload:
        context=json.load(upload)
        st.session_state["game_context"]=context
        st.success("Context loaded.")
    else:
        context=st.session_state.get("game_context",{})

    with st.expander("Current context"):
        st.json(context if context else {"message":"No manual context loaded."})

    st.markdown("#### Data source")
    try:
        _,provider_name=get_provider(league,bdl_key)
        st.success(f"Current primary path: {provider_name}")
    except Exception as e:
        st.warning(str(e))

# ---------------- Team Markets ----------------
with tab_team:
    st.subheader("Team markets")
    if league!="WNBA":
        st.info("Automatic Team Markets v2 is enabled for WNBA first. NBA adapter comes next.")
    elif not bdl_key:
        st.warning("Add BALLDONTLIE_API_KEY in Streamlit Secrets for automatic WNBA team logs.")
    else:
        try:
            players,_=cached_players("WNBA",int(season),True)
            if players.empty:
                st.warning("No WNBA player/team data returned.")
            else:
                teams=players[["TEAM_ID","TEAM_ABBR"]].dropna().drop_duplicates().sort_values("TEAM_ABBR")
                team_map={r.TEAM_ABBR:int(r.TEAM_ID) for _,r in teams.iterrows()}
                team_abbr=st.selectbox("Team",list(team_map.keys()),key="team_market_team")
                team_id=team_map[team_abbr]

                if st.button("Load team data",key="load_team"):
                    raw,src=cached_team_log_wnba(int(season),team_id,True)
                    st.session_state["team_log"]=clean_team_log(raw)
                    st.session_state["team_src"]=src

                tlog=st.session_state.get("team_log")
                if isinstance(tlog,pd.DataFrame) and not tlog.empty:
                    regime=st.radio(
                        "Sample weighting",
                        ["Stable 55/20/25","Role change 35/20/45"],
                        horizontal=True,key="team_regime"
                    )
                    cfg=WeightConfig.stable() if regime.startswith("Stable") else WeightConfig.role_change()
                    profile,audit=build_team_profile(tlog,cfg)

                    c1,c2=st.columns(2)
                    with c1:
                        default_poss=float(profile["poss_pg"]) if np.isfinite(profile["poss_pg"]) else 80.0
                        proj_poss=st.number_input("Projected possessions",60.0,110.0,round(default_poss,1),0.5)
                    with c2:
                        poss_sd=st.number_input("Possession SD",1.0,8.0,3.0,0.25)

                    st.markdown("**Opponent / current-role multipliers** — 1.00 = neutral")
                    cols=st.columns(5)
                    three_pa=cols[0].number_input("3PA",.70,1.30,1.00,.01,key="t3pa")
                    fta=cols[1].number_input("FTA",.70,1.30,1.00,.01,key="tfta")
                    tov=cols[2].number_input("TOV",.70,1.30,1.00,.01,key="ttov")
                    oreb=cols[3].number_input("OREB",.70,1.30,1.00,.01,key="toreb")
                    ast=cols[4].number_input("AST",.70,1.30,1.00,.01,key="tast")
                    cols2=st.columns(5)
                    two_pa=cols2[0].number_input("2PA",.70,1.30,1.00,.01,key="t2pa")
                    stl=cols2[1].number_input("STL",.70,1.30,1.00,.01,key="tstl")
                    blk=cols2[2].number_input("BLK",.70,1.30,1.00,.01,key="tblk")
                    pf=cols2[3].number_input("PF",.70,1.30,1.00,.01,key="tpf")
                    n=cols2[4].select_slider("Simulations",[25_000,50_000,100_000,250_000,500_000],100_000,key="tn")

                    ctx=TeamContext(
                        projected_possessions=proj_poss,possessions_sd=poss_sd,
                        three_pa=three_pa,fta=fta,tov=tov,oreb=oreb,ast=ast,
                        two_pa=two_pa,stl=stl,blk=blk,pf=pf
                    )
                    if st.button("Run team model",type="primary",key="run_team"):
                        sim=simulate_team(profile,ctx,int(n),seed=101)
                        sn=min(60_000,max(20_000,int(n)//4))
                        low=simulate_team(profile,ctx,sn,seed=102,opportunity_mult=.95)
                        high=simulate_team(profile,ctx,sn,seed=103,opportunity_mult=1.05)
                        st.session_state["team_sim"]=(sim,low,high)
                        st.session_state["team_audit"]=audit

                    pack=st.session_state.get("team_sim")
                    if pack:
                        sim,low,high=pack
                        markets=["PTS","3PA","3PM","2PA","2PM","FTA","TOV","OREB","AST","STL","BLK","PF"]
                        st.dataframe(market_table(sim,low,high,markets).round(2),use_container_width=True)

                        st.markdown("#### Price one team market")
                        c1,c2,c3,c4=st.columns(4)
                        market=c1.selectbox("Market",markets)
                        line=c2.number_input("Line",value=float(round(sim[market].mean()-.5,1)),step=.5,key="tml")
                        oo=c3.number_input("Over odds",1.01,20.0,1.90,.01,key="tmo")
                        uo=c4.number_input("Under odds",1.01,20.0,1.90,.01,key="tmu")
                        p=price(sim[market],line,oo,uo)
                        st.dataframe(pd.DataFrame([p]).round(4),use_container_width=True)
        except Exception as e:
            st.error(f"Team data error: {e}")

# ---------------- Player Props ----------------
with tab_player:
    st.subheader("Player props")
    try:
        pool,src=cached_players(league,int(season),bool(bdl_key))
        if pool.empty:
            st.warning("No player list returned. You can use the CSV fallback below.")
        else:
            st.caption(f"Source: {src}")
            pool=pool.dropna(subset=["PLAYER_ID","PLAYER_NAME"]).copy()
            labels=[]
            for _,r in pool.iterrows():
                team=r.get("TEAM_ABBR","")
                labels.append(f"{r.PLAYER_NAME} — {team}")
            selected=st.selectbox("Player",labels)
            idx=labels.index(selected)
            player_id=int(pool.iloc[idx].PLAYER_ID)
            player_name=str(pool.iloc[idx].PLAYER_NAME)

            if st.button("Load player data",key="load_player"):
                raw,source_name=cached_player_log(league,int(season),player_id,bool(bdl_key))
                st.session_state["player_log"]=clean_player_log(raw)
                st.session_state["player_source"]=source_name
                st.session_state["player_id"]=player_id
                st.session_state["player_name"]=player_name

        with st.expander("CSV fallback"):
            up=st.file_uploader("Upload normalized player game-log CSV",type=["csv"],key="player_csv")
            if up:
                st.session_state["player_log"]=clean_player_log(pd.read_csv(up))
                st.session_state["player_source"]="CSV"
                st.session_state["player_name"]="Uploaded player"

        plog=st.session_state.get("player_log")
        if isinstance(plog,pd.DataFrame) and not plog.empty:
            pname=st.session_state.get("player_name","Player")
            st.markdown(f"### {pname}")
            regime=st.radio(
                "Sample weighting",
                ["Stable 55/20/25","Role change 35/20/45"],
                horizontal=True,key="pregime"
            )
            cfg=WeightConfig.stable() if regime.startswith("Stable") else WeightConfig.role_change()
            profile,audit=build_player_profile(plog,cfg)

            recent_min=float(plog.tail(10).MIN.mean())
            c1,c2,c3=st.columns(3)
            pmin=c1.number_input("Projected minutes",5.0,45.0,float(round(recent_min,1)),.5)
            msd=c2.number_input("Minutes SD",.5,6.0,2.0,.25)
            n=c3.select_slider("Simulations",[25_000,50_000,100_000,250_000,500_000],100_000,key="pn")

            st.markdown("**Opponent multipliers**")
            ocols=st.columns(5)
            opp_pts=ocols[0].number_input("PTS/2PA env",.70,1.30,1.00,.01)
            opp_reb=ocols[1].number_input("REB env",.70,1.30,1.00,.01)
            opp_ast=ocols[2].number_input("AST env",.70,1.30,1.00,.01)
            opp_3pa=ocols[3].number_input("3PA env",.70,1.30,1.00,.01)
            opp_fta=ocols[4].number_input("FTA env",.70,1.30,1.00,.01)

            st.markdown("**Injury / role redistribution**")
            rcols=st.columns(5)
            usage=rcols[0].number_input("Usage",.70,1.50,1.00,.01)
            creation=rcols[1].number_input("Creation",.70,1.50,1.00,.01)
            reb_role=rcols[2].number_input("REB role",.70,1.50,1.00,.01)
            three_role=rcols[3].number_input("3PA role",.60,1.60,1.00,.01)
            fta_role=rcols[4].number_input("FTA role",.60,1.60,1.00,.01)

            st.markdown("**H2H contextual correction only (never a second sample)**")
            hcols=st.columns(4)
            hp=hcols[0].slider("PTS %",-10,10,0)
            hr=hcols[1].slider("REB %",-10,10,0)
            ha=hcols[2].slider("AST %",-10,10,0)
            h3=hcols[3].slider("3PA %",-10,10,0)

            ctx=PlayerContext(
                projected_minutes=pmin,minutes_sd=msd,
                opp_pts=opp_pts,opp_reb=opp_reb,opp_ast=opp_ast,opp_3pa=opp_3pa,opp_fta=opp_fta,
                usage=usage,creation=creation,reb_role=reb_role,three_role=three_role,fta_role=fta_role,
                h2h_pts=1+hp/100,h2h_reb=1+hr/100,h2h_ast=1+ha/100,h2h_3pa=1+h3/100,
            )
            if st.button("Run player model",type="primary",key="run_player"):
                sim=simulate_player(profile,ctx,int(n),seed=201)
                sn=min(60_000,max(20_000,int(n)//4))
                low=simulate_player(profile,ctx,sn,seed=202,opportunity_mult=.95)
                high=simulate_player(profile,ctx,sn,seed=203,opportunity_mult=1.05)
                st.session_state["player_sim"]=(sim,low,high)
                st.session_state["player_audit"]=audit

            pack=st.session_state.get("player_sim")
            if pack:
                sim,low,high=pack
                markets=["PTS","REB","AST","3PM","PRA","PR","PA","AR"]
                st.dataframe(market_table(sim,low,high,markets).round(2),use_container_width=True)

                st.markdown("#### Bookmaker pricing")
                price_rows=[]
                for m in markets:
                    with st.expander(m,expanded=m in ("PTS","REB","AST","3PM")):
                        c1,c2,c3=st.columns(3)
                        line=c1.number_input(f"{m} line",value=float(round(sim[m].mean()-.5,1)),step=.5,key=f"line_{m}")
                        oo=c2.number_input(f"{m} Over",1.01,20.0,1.90,.01,key=f"oo_{m}")
                        uo=c3.number_input(f"{m} Under",1.01,20.0,1.90,.01,key=f"uo_{m}")
                        p=price(sim[m],line,oo,uo)
                        pl=price(low[m],line,oo,uo)
                        ph=price(high[m],line,oo,uo)
                        price_rows.append({
                            "Market":m,"Line":line,
                            "P(O)":p["p_over"],"Fair O":p["fair_over"],"Odds O":oo,"EV O":p["ev_over"],
                            "Bear P(O)":pl["p_over"],"Bull P(O)":ph["p_over"],
                            "P(U)":p["p_under"],"Fair U":p["fair_under"],"Odds U":uo,"EV U":p["ev_under"],
                            "Bear P(U)":pl["p_under"],"Bull P(U)":ph["p_under"],
                        })
                st.dataframe(pd.DataFrame(price_rows).round(4),use_container_width=True)

            if league=="WNBA" and bdl_key and st.session_state.get("player_id"):
                with st.expander("BALLDONTLIE advanced season data"):
                    measure=st.selectbox("Measure",["advanced","usage","defense","four_factors","opponent","scoring","base"])
                    if st.button("Load advanced stats"):
                        try:
                            adv=cached_advanced_player(int(season),int(st.session_state["player_id"]),measure,True)
                            st.json(adv)
                        except Exception as e:
                            st.warning(f"Advanced endpoint unavailable for this key/tier: {e}")
    except Exception as e:
        st.error(f"Player data error: {e}")

# ---------------- Audit ----------------
with tab_audit:
    st.subheader("Model audit")
    st.markdown(
        """
**Hard rules**
- Season/L10/L5 are never used as three independent samples.
- H2H is never added on top of recent games; it is only a small contextual modifier.
- Pace/opportunity enters once.
- 3PM is simulated from attempts and regressed shooting efficiency.
- The market is priced after the projection exists.
- A central edge is not enough if a plausible adverse scenario destroys it.
        """
    )
    if st.session_state.get("player_audit") is not None:
        st.markdown("#### Player non-overlap audit")
        st.dataframe(st.session_state["player_audit"],use_container_width=True)
    if st.session_state.get("team_audit") is not None:
        st.markdown("#### Team non-overlap audit")
        st.dataframe(st.session_state["team_audit"],use_container_width=True)

    st.markdown("#### API status")
    st.write({
        "league":league,
        "BALLDONTLIE_key_configured":bool(bdl_key),
        "secrets_rendered_to_UI":False,
    })

st.caption(
    "Model-implied fair odds are not historically calibrated true odds. "
    "Use Data Audit + stress scenarios before qualifying a bet."
)
