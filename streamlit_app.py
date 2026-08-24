from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import streamlit as st

from providers.router import get_basic_provider, get_advanced_provider
from core.cleaning import clean_player_log, clean_team_log
from core.buckets import WeightConfig
from core.player_model import PlayerContext, build_player_profile, simulate_player
from core.team_model import TeamContext, build_team_profile, simulate_team
from core.pricing import price, market_table
from core.matchup import opponent_allowed_profile, player_matchup_modifiers, team_matchup_modifiers


st.set_page_config(page_title="Basketball Pricing Engine", page_icon="🏀", layout="wide")
st.title("🏀 Basketball Pricing Engine v2.2")
st.caption("Projection first • Odds second • No L5/L10 double counting • Automatic opponent context • Joint player Monte Carlo")


def get_secret(name):
    try:
        return st.secrets.get(name, None)
    except Exception:
        return os.getenv(name)


def ci_lookup(d, name, default=None):
    if not isinstance(d, dict):
        return default
    target = str(name).strip().casefold()
    for k, v in d.items():
        if str(k).strip().casefold() == target:
            return v
    return default


bdl_key = get_secret("BALLDONTLIE_API_KEY")

with st.sidebar:
    st.header("Data / Security")
    st.write("Basic data:", "✅ nba_api")
    st.write("BALLDONTLIE advanced key:", "✅ configured" if bdl_key else "⚪ not configured")
    st.caption("Keys are never rendered and never belong in context JSON.")
    st.divider()
    league = st.selectbox("League", ["WNBA","NBA","EuroLeague","EuroCup"])
    season = st.number_input("Season", min_value=2000, max_value=2100, value=2026, step=1)
    if league in ("EuroLeague","EuroCup"):
        st.info("Provider scaffold is ready; activation comes after NBA/WNBA validation.")
        st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_players(league, season):
    provider, name = get_basic_provider(league)
    return provider.players(season), name


@st.cache_data(ttl=3600, show_spinner=False)
def cached_league_team_logs(league, season):
    provider, name = get_basic_provider(league)
    return provider.league_team_game_logs(season), name


@st.cache_data(ttl=3600, show_spinner=False)
def cached_player_log(league, season, player_id):
    provider, name = get_basic_provider(league)
    return provider.player_game_log(int(player_id), int(season)), name


@st.cache_data(ttl=3600, show_spinner=False)
def cached_player_position(league, player_id):
    provider, _ = get_basic_provider(league)
    return provider.player_position(int(player_id))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_position_totals(league, season, position, opponent_team_id):
    provider, _ = get_basic_provider(league)
    return provider.position_totals(int(season), str(position), int(opponent_team_id or 0))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_advanced_player(season, player_id, measure):
    provider, _ = get_advanced_provider("WNBA", bdl_key)
    if provider is None:
        return []
    return provider.player_season_advanced(int(season), int(player_id), measure)


try:
    player_pool, basic_source = cached_players(league, int(season))
    player_pool = player_pool.dropna(subset=["PLAYER_ID","PLAYER_NAME"]).copy()
    team_pool = (
        player_pool[["TEAM_ID","TEAM_ABBR"]]
        .dropna()
        .drop_duplicates(subset=["TEAM_ID"])
        .sort_values("TEAM_ABBR")
        .reset_index(drop=True)
    )
except Exception as e:
    player_pool = pd.DataFrame()
    team_pool = pd.DataFrame()
    basic_source = "unavailable"
    st.error(f"Basic nba_api initialization error: {e}")


tab_game, tab_team, tab_player, tab_audit = st.tabs(
    ["🎯 Game Setup","🏟️ Team Markets","👤 Player Props","🔎 Data Audit"]
)

# ==================== GAME SETUP ====================
with tab_game:
    st.subheader("Game Setup")
    st.write(
        "Choose the matchup. Opponent allowed is calculated automatically. "
        "Manual context is only for pre-game information historical APIs cannot know."
    )

    if team_pool.empty:
        st.warning("Team list unavailable.")
    else:
        team_labels = team_pool["TEAM_ABBR"].astype(str).tolist()
        c1,c2 = st.columns(2)
        home_abbr = c1.selectbox("Home team", team_labels, key="setup_home")
        away_choices = [x for x in team_labels if x != home_abbr]
        away_abbr = c2.selectbox("Away team", away_choices, key="setup_away")
        team_id_map = {str(r.TEAM_ABBR): int(r.TEAM_ID) for _,r in team_pool.iterrows()}

        if st.button("Load matchup & opponent data", type="primary"):
            try:
                league_logs, src = cached_league_team_logs(league, int(season))
                league_logs = clean_team_log(league_logs)

                setup = {
                    "home_abbr": home_abbr, "home_id": team_id_map[home_abbr],
                    "away_abbr": away_abbr, "away_id": team_id_map[away_abbr],
                    "source": src
                }
                st.session_state["game_setup"] = setup
                st.session_state["league_team_logs"] = league_logs
                st.session_state["opp_profile_home"] = opponent_allowed_profile(league_logs, home_abbr)
                st.session_state["opp_profile_away"] = opponent_allowed_profile(league_logs, away_abbr)
                st.success(f"Loaded {away_abbr} @ {home_abbr}.")
            except Exception as e:
                st.error(f"Matchup load failed: {e}")

    st.markdown("### Pre-game context")
    st.caption("Upload or paste injuries / minutes / role changes. Matching players are prefilled automatically.")

    upload = st.file_uploader("Upload game_context.json", type=["json"])
    if upload is not None:
        try:
            st.session_state["game_context"] = json.load(upload)
            st.success("Context JSON loaded.")
        except Exception as e:
            st.error(f"Invalid JSON file: {e}")

    current_context = st.session_state.get("game_context", {})
    context_text = st.text_area(
        "Paste / edit context JSON",
        value=json.dumps(current_context, indent=2) if current_context else "",
        height=240,
        placeholder='{"injuries":{"Rae Burrell":{"status":"OUT"}},"projected_minutes":{"Ariel Atkins":31},"role_adjustments":{"Ariel Atkins":{"usage":1.08,"three_role":1.12}}}'
    )
    if st.button("Apply context JSON"):
        try:
            st.session_state["game_context"] = json.loads(context_text) if context_text.strip() else {}
            st.success("Context applied.")
        except Exception as e:
            st.error(f"JSON error: {e}")

    with st.expander("Current applied context"):
        st.json(st.session_state.get("game_context", {}) or {"message":"No manual context"})

    setup = st.session_state.get("game_setup")
    if setup:
        st.success(f"{setup['away_abbr']} @ {setup['home_abbr']} • source: {setup['source']}")


# ==================== TEAM MARKETS ====================
with tab_team:
    st.subheader("Team Markets")
    setup = st.session_state.get("game_setup")
    league_logs = st.session_state.get("league_team_logs")

    if not setup or not isinstance(league_logs, pd.DataFrame):
        st.info("Load a matchup in Game Setup first.")
    else:
        side = st.radio("Team to price", ["Away","Home"], horizontal=True, key="team_side")
        if side == "Away":
            team_abbr, team_id = setup["away_abbr"], setup["away_id"]
            opp_abbr, opp_id = setup["home_abbr"], setup["home_id"]
            opp_profile = st.session_state["opp_profile_home"]
        else:
            team_abbr, team_id = setup["home_abbr"], setup["home_id"]
            opp_abbr, opp_id = setup["away_abbr"], setup["away_id"]
            opp_profile = st.session_state["opp_profile_away"]

        team_log = clean_team_log(
            league_logs[pd.to_numeric(league_logs["TEAM_ID"], errors="coerce") == int(team_id)].copy()
        )
        opp_log = clean_team_log(
            league_logs[pd.to_numeric(league_logs["TEAM_ID"], errors="coerce") == int(opp_id)].copy()
        )

        regime = st.radio("Sample weighting", ["Stable 55/20/25","Role change 35/20/45"], horizontal=True, key="team_regime")
        cfg = WeightConfig.stable() if regime.startswith("Stable") else WeightConfig.role_change()
        profile, audit = build_team_profile(team_log, cfg)
        opp_off_profile, _ = build_team_profile(opp_log, cfg)
        auto = team_matchup_modifiers(opp_profile)

        default_poss = float(0.55*profile.get("poss_pg",80.0) + 0.45*opp_off_profile.get("poss_pg",80.0))

        st.markdown(f"### {team_abbr} vs {opp_abbr}")
        c1,c2,c3 = st.columns(3)
        proj_poss = c1.number_input("Projected possessions",60.0,110.0,round(default_poss,1),0.5,key=f"poss_{team_abbr}_{opp_abbr}")
        poss_sd = c2.number_input("Possession SD",1.0,8.0,3.0,0.25,key=f"posssd_{team_abbr}_{opp_abbr}")
        n = c3.select_slider("Simulations",[25_000,50_000,100_000,250_000,500_000],100_000,key=f"tn_{team_abbr}_{opp_abbr}")

        st.caption("Automatic opponent modifiers are prefilled but editable.")
        cols = st.columns(7)
        three_pa = cols[0].number_input("3PA",.70,1.30,float(auto["3PA"]),.01,key=f"t3pa_{team_abbr}_{opp_abbr}")
        two_pa = cols[1].number_input("2PA",.70,1.30,float(auto["2PA"]),.01,key=f"t2pa_{team_abbr}_{opp_abbr}")
        fta = cols[2].number_input("FTA",.70,1.30,float(auto["FTA"]),.01,key=f"tfta_{team_abbr}_{opp_abbr}")
        tov = cols[3].number_input("TOV",.70,1.30,float(auto["TOV"]),.01,key=f"ttov_{team_abbr}_{opp_abbr}")
        oreb = cols[4].number_input("OREB",.70,1.30,float(auto["OREB"]),.01,key=f"toreb_{team_abbr}_{opp_abbr}")
        ast = cols[5].number_input("AST",.70,1.30,float(auto["AST"]),.01,key=f"tast_{team_abbr}_{opp_abbr}")
        pf = cols[6].number_input("PF",.70,1.30,float(auto["PF"]),.01,key=f"tpf_{team_abbr}_{opp_abbr}")

        with st.expander("Opponent modifier audit"):
            st.dataframe(opp_profile["audit"].round(4), use_container_width=True)

        ctx = TeamContext(
            projected_possessions=proj_poss, possessions_sd=poss_sd,
            three_pa=three_pa, two_pa=two_pa, fta=fta, tov=tov,
            oreb=oreb, ast=ast, pf=pf
        )
        fingerprint = (team_abbr,opp_abbr,regime,proj_poss,poss_sd,three_pa,two_pa,fta,tov,oreb,ast,pf,int(n))

        if st.button("Run team model", type="primary", key=f"run_team_{team_abbr}_{opp_abbr}"):
            sim = simulate_team(profile,ctx,int(n),seed=101)
            sn = min(60_000,max(20_000,int(n)//4))
            low = simulate_team(profile,ctx,sn,seed=102,opportunity_mult=.95)
            high = simulate_team(profile,ctx,sn,seed=103,opportunity_mult=1.05)
            st.session_state["team_sim"] = (fingerprint,sim,low,high)
            st.session_state["team_audit"] = audit

        pack = st.session_state.get("team_sim")
        if pack and pack[0] == fingerprint:
            _,sim,low,high = pack
            markets = ["PTS","3PA","3PM","2PA","2PM","FTA","TOV","OREB","AST","STL","BLK","PF"]
            st.dataframe(market_table(sim,low,high,markets).round(2),use_container_width=True)
            a,b,c,d = st.columns(4)
            market = a.selectbox("Market",markets,key=f"tmkt_{team_abbr}")
            line = b.number_input("Line",value=float(round(sim[market].mean()-.5,1)),step=.5,key=f"tline_{team_abbr}_{market}")
            oo = c.number_input("Over odds",1.01,20.0,1.90,.01,key=f"to_{team_abbr}_{market}")
            uo = d.number_input("Under odds",1.01,20.0,1.90,.01,key=f"tu_{team_abbr}_{market}")
            p,pl,ph = price(sim[market],line,oo,uo),price(low[market],line,oo,uo),price(high[market],line,oo,uo)
            st.dataframe(pd.DataFrame([{**p,"bear_p_over":pl["p_over"],"bull_p_over":ph["p_over"],"bear_p_under":pl["p_under"],"bull_p_under":ph["p_under"]}]).round(4),use_container_width=True)
        elif pack:
            st.warning("Inputs changed since last simulation. Run again.")


# ==================== PLAYER PROPS ====================
with tab_player:
    st.subheader("Player Props")
    if player_pool.empty:
        st.warning("Player pool unavailable.")
    else:
        labels = [f"{r.PLAYER_NAME} — {r.TEAM_ABBR}" for _,r in player_pool.iterrows()]
        selected = st.selectbox("Player",labels,key="player_select")
        idx = labels.index(selected)
        prow = player_pool.iloc[idx]
        player_id = int(prow.PLAYER_ID)
        player_name = str(prow.PLAYER_NAME)
        player_team = str(prow.TEAM_ABBR)

        setup = st.session_state.get("game_setup")
        opp_abbr = opp_id = overall_profile = None
        if setup:
            if player_team == setup["home_abbr"]:
                opp_abbr,opp_id = setup["away_abbr"],setup["away_id"]
                overall_profile = st.session_state.get("opp_profile_away")
            elif player_team == setup["away_abbr"]:
                opp_abbr,opp_id = setup["home_abbr"],setup["home_id"]
                overall_profile = st.session_state.get("opp_profile_home")

        if opp_abbr:
            st.success(f"Detected: {player_name} ({player_team}) vs {opp_abbr}")
        else:
            st.warning("Player is not on the loaded matchup; automatic matchup modifiers are neutral.")

        if st.button("Load player data",key=f"load_player_{player_id}"):
            try:
                raw,src = cached_player_log(league,int(season),player_id)
                st.session_state["player_log"] = clean_player_log(raw)
                st.session_state["player_log_id"] = player_id
                st.session_state["player_source"] = src
            except Exception as e:
                st.error(f"Player game-log error: {e}")

        with st.expander("CSV fallback"):
            up = st.file_uploader("Upload normalized player game-log CSV",type=["csv"],key=f"player_csv_{player_id}")
            if up:
                st.session_state["player_log"] = clean_player_log(pd.read_csv(up))
                st.session_state["player_log_id"] = player_id
                st.session_state["player_source"] = "CSV"

        plog = st.session_state.get("player_log")
        if isinstance(plog,pd.DataFrame) and not plog.empty and st.session_state.get("player_log_id") == player_id:
            try:
                pos = cached_player_position(league,player_id)
            except Exception:
                pos = {"raw":"","broad":None}

            matchup_mods = {"PTS":1.0,"REB":1.0,"AST":1.0,"3PA":1.0,"FTA":1.0}
            matchup_audit = pd.DataFrame()

            if opp_id and overall_profile:
                try:
                    if pos.get("broad"):
                        pvo = cached_position_totals(league,int(season),pos["broad"],int(opp_id))
                        plg = cached_position_totals(league,int(season),pos["broad"],0)
                    else:
                        pvo,plg = {},{}
                    matchup_mods,matchup_audit = player_matchup_modifiers(overall_profile,pvo,plg)
                except Exception as e:
                    st.warning(f"Position matchup unavailable; using overall opponent profile: {e}")
                    matchup_mods,matchup_audit = player_matchup_modifiers(overall_profile,{},{})

            regime = st.radio("Sample weighting",["Stable 55/20/25","Role change 35/20/45"],horizontal=True,key=f"pregime_{player_id}")
            cfg = WeightConfig.stable() if regime.startswith("Stable") else WeightConfig.role_change()
            profile,audit = build_player_profile(plog,cfg)

            manual_context = st.session_state.get("game_context",{})
            injury_info = ci_lookup(manual_context.get("injuries",{}),player_name,{}) or {}
            context_minutes = ci_lookup(manual_context.get("projected_minutes",{}),player_name,None)
            role = ci_lookup(manual_context.get("role_adjustments",{}),player_name,{}) or {}

            recent_min = float(plog.tail(10).MIN.mean())
            default_minutes = float(context_minutes) if context_minutes is not None else recent_min

            if injury_info:
                status = injury_info.get("status","UNKNOWN") if isinstance(injury_info,dict) else injury_info
                if str(status).upper() == "OUT":
                    st.error(f"Context marks {player_name} OUT.")
                else:
                    st.info(f"Context injury status: {status}")

            st.markdown(f"**Position:** {pos.get('raw') or 'unknown'} → `{pos.get('broad') or 'N/A'}`")

            c1,c2,c3 = st.columns(3)
            pmin = c1.number_input("Projected minutes",5.0,45.0,float(round(default_minutes,1)),.5,key=f"pmin_{player_id}_{opp_abbr}")
            msd = c2.number_input("Minutes SD",.5,6.0,2.0,.25,key=f"msd_{player_id}_{opp_abbr}")
            n = c3.select_slider("Simulations",[25_000,50_000,100_000,250_000,500_000],100_000,key=f"pn_{player_id}_{opp_abbr}")

            st.markdown("### Automatic opponent / position modifiers")
            st.caption("Position only adjusts its deviation from overall defense, so opponent signal is not counted twice.")
            if not matchup_audit.empty:
                st.dataframe(matchup_audit.round(4),use_container_width=True)

            ocols = st.columns(5)
            opp_pts = ocols[0].number_input("PTS env",.70,1.30,float(matchup_mods["PTS"]),.01,key=f"oppts_{player_id}_{opp_abbr}")
            opp_reb = ocols[1].number_input("REB env",.70,1.30,float(matchup_mods["REB"]),.01,key=f"opreb_{player_id}_{opp_abbr}")
            opp_ast = ocols[2].number_input("AST env",.70,1.30,float(matchup_mods["AST"]),.01,key=f"opast_{player_id}_{opp_abbr}")
            opp_3pa = ocols[3].number_input("3PA env",.70,1.30,float(matchup_mods["3PA"]),.01,key=f"op3pa_{player_id}_{opp_abbr}")
            opp_fta = ocols[4].number_input("FTA env",.70,1.30,float(matchup_mods["FTA"]),.01,key=f"opfta_{player_id}_{opp_abbr}")

            st.markdown("### Injury / role redistribution")
            st.caption("Defaults are automatically read from Game Setup JSON.")
            rcols = st.columns(5)
            usage = rcols[0].number_input("Usage",.70,1.50,float(role.get("usage",1.0)),.01,key=f"usage_{player_id}")
            creation = rcols[1].number_input("Creation",.70,1.50,float(role.get("creation",1.0)),.01,key=f"creation_{player_id}")
            reb_role = rcols[2].number_input("REB role",.70,1.50,float(role.get("reb_role",role.get("reb",1.0))),.01,key=f"rebrole_{player_id}")
            three_role = rcols[3].number_input("3PA role",.60,1.60,float(role.get("three_role",role.get("three_pa",1.0))),.01,key=f"3role_{player_id}")
            fta_role = rcols[4].number_input("FTA role",.60,1.60,float(role.get("fta_role",1.0)),.01,key=f"ftarole_{player_id}")

            h2h = pd.DataFrame()
            if opp_abbr and "OPP_ABBR" in plog.columns:
                h2h = plog[plog["OPP_ABBR"].astype(str).str.upper() == str(opp_abbr).upper()].copy()

            st.markdown("### H2H audit — 0% extra weight by default")
            if h2h.empty:
                st.caption("No same-season H2H found.")
            else:
                show = [c for c in ["GAME_DATE","MIN","PTS","REB","AST","FG3M","FG3A","FTA"] if c in h2h.columns]
                hh = h2h[show].copy()
                if "MIN" in hh:
                    hh["OT flag"] = hh["MIN"] > 40
                st.dataframe(hh.sort_values("GAME_DATE",ascending=False),use_container_width=True)
                st.caption("These games already exist in Old/Mid/L5 where applicable and are not counted again.")

            hcols = st.columns(4)
            hp = hcols[0].slider("Optional PTS H2H %",-10,10,0,key=f"hp_{player_id}")
            hr = hcols[1].slider("Optional REB H2H %",-10,10,0,key=f"hr_{player_id}")
            ha = hcols[2].slider("Optional AST H2H %",-10,10,0,key=f"ha_{player_id}")
            h3 = hcols[3].slider("Optional 3PA H2H %",-10,10,0,key=f"h3_{player_id}")

            ctx = PlayerContext(
                projected_minutes=pmin,minutes_sd=msd,
                opp_pts=opp_pts,opp_reb=opp_reb,opp_ast=opp_ast,opp_3pa=opp_3pa,opp_fta=opp_fta,
                usage=usage,creation=creation,reb_role=reb_role,three_role=three_role,fta_role=fta_role,
                h2h_pts=1+hp/100,h2h_reb=1+hr/100,h2h_ast=1+ha/100,h2h_3pa=1+h3/100
            )
            fingerprint = (player_id,opp_abbr,regime,pmin,msd,opp_pts,opp_reb,opp_ast,opp_3pa,opp_fta,usage,creation,reb_role,three_role,fta_role,hp,hr,ha,h3,int(n))

            if st.button("Run player model",type="primary",key=f"run_player_{player_id}"):
                sim = simulate_player(profile,ctx,int(n),seed=201)
                sn = min(60_000,max(20_000,int(n)//4))
                low = simulate_player(profile,ctx,sn,seed=202,opportunity_mult=.95)
                high = simulate_player(profile,ctx,sn,seed=203,opportunity_mult=1.05)
                st.session_state["player_sim"] = (fingerprint,sim,low,high)
                st.session_state["player_audit"] = audit
                st.session_state["player_matchup_audit"] = matchup_audit

            pack = st.session_state.get("player_sim")
            if pack and pack[0] == fingerprint:
                _,sim,low,high = pack
                markets = ["PTS","REB","AST","3PM","PRA","PR","PA","AR"]
                st.dataframe(market_table(sim,low,high,markets).round(2),use_container_width=True)

                st.markdown("### Bookmaker pricing")
                rows = []
                for m in markets:
                    with st.expander(m,expanded=m in ("PTS","REB","AST","3PM")):
                        c1,c2,c3 = st.columns(3)
                        line = c1.number_input(f"{m} line",value=float(round(sim[m].mean()-.5,1)),step=.5,key=f"line_{player_id}_{m}")
                        oo = c2.number_input(f"{m} Over",1.01,20.0,1.90,.01,key=f"oo_{player_id}_{m}")
                        uo = c3.number_input(f"{m} Under",1.01,20.0,1.90,.01,key=f"uo_{player_id}_{m}")
                        p,pl,ph = price(sim[m],line,oo,uo),price(low[m],line,oo,uo),price(high[m],line,oo,uo)
                        rows.append({
                            "Market":m,"Line":line,
                            "P(O)":p["p_over"],"Fair O":p["fair_over"],"Odds O":oo,"EV O":p["ev_over"],
                            "Bear P(O)":pl["p_over"],"Bull P(O)":ph["p_over"],
                            "P(U)":p["p_under"],"Fair U":p["fair_under"],"Odds U":uo,"EV U":p["ev_under"],
                            "Bear P(U)":pl["p_under"],"Bull P(U)":ph["p_under"]
                        })
                st.dataframe(pd.DataFrame(rows).round(4),use_container_width=True)
                st.warning("Central EV alone does not qualify a bet. If a plausible adverse scenario kills the edge, PASS.")
            elif pack:
                st.warning("Inputs changed since last simulation. Run again.")

            if league == "WNBA" and bdl_key:
                with st.expander("Optional BALLDONTLIE advanced data (tier-dependent)"):
                    measure = st.selectbox("Measure",["advanced","usage","defense","four_factors","opponent","scoring","base"],key=f"adv_measure_{player_id}")
                    if st.button("Load optional advanced stats",key=f"adv_load_{player_id}"):
                        try:
                            st.json(cached_advanced_player(int(season),player_id,measure))
                        except Exception as e:
                            st.warning(f"Advanced endpoint unavailable for tier/rate limit. Core model unaffected. {e}")


# ==================== AUDIT ====================
with tab_audit:
    st.subheader("Data Audit")
    st.markdown("""
- Old season / Games 6–10 / L5 never overlap.
- H2H is displayed separately and gets 0% extra weight by default.
- Position modifies only the positional deviation from overall opponent defense.
- Game-context JSON auto-prefills matching minutes and role changes.
- Pace/opportunity enters once.
- 3PM comes from 3PA and regressed shooting efficiency.
- Projection exists before bookmaker odds are entered.
    """)

    setup = st.session_state.get("game_setup")
    if setup:
        st.markdown("### Current matchup")
        st.json(setup)
        hp = st.session_state.get("opp_profile_home")
        ap = st.session_state.get("opp_profile_away")
        if hp:
            st.markdown(f"#### {setup['home_abbr']} opponent-allowed profile")
            st.dataframe(hp["audit"].round(4),use_container_width=True)
        if ap:
            st.markdown(f"#### {setup['away_abbr']} opponent-allowed profile")
            st.dataframe(ap["audit"].round(4),use_container_width=True)

    st.markdown("### Applied context JSON")
    st.json(st.session_state.get("game_context",{}) or {"message":"No manual context"})

    if st.session_state.get("player_audit") is not None:
        st.markdown("### Player non-overlap audit")
        st.dataframe(st.session_state["player_audit"],use_container_width=True)
    if st.session_state.get("player_matchup_audit") is not None:
        st.markdown("### Player opponent/position audit")
        st.dataframe(st.session_state["player_matchup_audit"],use_container_width=True)
    if st.session_state.get("team_audit") is not None:
        st.markdown("### Team non-overlap audit")
        st.dataframe(st.session_state["team_audit"],use_container_width=True)

    st.write({
        "league":league,
        "basic_provider":"nba_api",
        "BALLDONTLIE_advanced_key_configured":bool(bdl_key),
        "keys_rendered_to_UI":False
    })

st.caption("Model-implied fair odds are not historically calibrated true odds. Use audit + stress before qualifying a bet.")
