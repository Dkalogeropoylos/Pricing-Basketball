from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import streamlit as st

from providers.sportsdataverse_wnba import SportsDataverseWNBA
from providers.router import get_advanced_provider
from core.cleaning import clean_player_log, clean_team_log
from core.buckets import WeightConfig
from core.player_model import PlayerContext, build_player_profile, simulate_player
from core.team_model import TeamContext, build_team_profile, simulate_team
from core.pricing import price, market_table
from core.matchup import (
    opponent_allowed_profile,
    player_matchup_modifiers,
    team_matchup_modifiers,
)


st.set_page_config(
    page_title="Basketball Pricing Engine",
    page_icon="🏀",
    layout="wide"
)
st.title("🏀 Basketball Pricing Engine v2.4.1")
st.caption(
    "SportsDataverse historical database • No stats.nba.com dependency • "
    "No L5/L10 overlap • Automatic opponent context • Joint Monte Carlo"
)


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


def reset_context_driven_player_widgets():
    """
    Streamlit number_input widgets keep their old state even when a new default
    is supplied. Clear only context-driven player inputs so uploaded/applied
    metadata can prefill them on the next rerun.
    """
    prefixes = (
        "pmin_",
        "usage_",
        "creation_",
        "rebrole_",
        "3role_",
        "ftarole_",
    )
    for key in list(st.session_state.keys()):
        if any(str(key).startswith(prefix) for prefix in prefixes):
            del st.session_state[key]


bdl_key = get_secret("BALLDONTLIE_API_KEY")


with st.sidebar:
    st.header("Data / Security")
    st.write("WNBA historical:", "✅ SportsDataverse")
    st.write("stats.nba.com:", "🚫 not used")
    st.write(
        "BALLDONTLIE advanced:",
        "✅ configured" if bdl_key else "⚪ optional"
    )
    st.caption("No API token is required for the core WNBA database.")
    st.divider()

    league = st.selectbox(
        "League",
        ["WNBA", "NBA (later)", "EuroLeague (later)", "EuroCup (later)"]
    )
    season = st.number_input(
        "Season", min_value=2002, max_value=2100, value=2026, step=1
    )

    if league != "WNBA":
        st.info(
            "v2.4 activates the database architecture for WNBA first. "
            "The same provider interface will be extended to the other leagues."
        )
        st.stop()


@st.cache_data(ttl=21600, show_spinner=False)
def load_sportsdataverse_season(season):
    provider = SportsDataverseWNBA(timeout=30)
    return provider.load_season(int(season))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_advanced_player(season, player_id, measure):
    provider, _ = get_advanced_provider("WNBA", bdl_key)
    if provider is None:
        return []
    return provider.player_season_advanced(
        int(season), int(player_id), measure
    )


# No network call at startup.
data_pack = st.session_state.get("sdv_data")
player_db = data_pack["player"] if data_pack else pd.DataFrame()
team_db = data_pack["team"] if data_pack else pd.DataFrame()

tab_game, tab_team, tab_player, tab_audit = st.tabs(
    ["🎯 Game Setup", "🏟️ Team Markets", "👤 Player Props", "🔎 Data Audit"]
)


# =====================================================================
# GAME SETUP
# =====================================================================
with tab_game:
    st.subheader("Game Setup")

    if data_pack is None:
        st.info(
            "Load the season database once. The app downloads two static "
            "SportsDataverse release files and caches them for 6 hours."
        )
        if st.button(
            f"Load WNBA {int(season)} database",
            type="primary"
        ):
            try:
                with st.spinner(
                    "Downloading player + team boxscore databases from GitHub Releases..."
                ):
                    pack = load_sportsdataverse_season(int(season))
                st.session_state["sdv_data"] = pack
                st.session_state.pop("game_setup", None)
                st.session_state.pop("player_sim", None)
                st.session_state.pop("team_sim", None)
                st.rerun()
            except Exception as e:
                st.error(
                    "SportsDataverse database download failed. "
                    f"Details: {e}"
                )
    else:
        provider = SportsDataverseWNBA()
        teams = provider.teams(team_db)

        dmin = player_db["GAME_DATE"].min()
        dmax = player_db["GAME_DATE"].max()

        a,b,c,d = st.columns(4)
        a.metric("Player-game rows", f"{len(player_db):,}")
        b.metric("Team-game rows", f"{len(team_db):,}")
        c.metric("Players", f"{player_db['PLAYER_ID'].nunique():,}")
        d.metric("Through", str(pd.Timestamp(dmax).date()))

        with st.expander("Database sources"):
            st.json(data_pack["sources"])

        team_names = teams["TEAM_NAME"].astype(str).tolist()
        c1,c2 = st.columns(2)
        home_name = c1.selectbox("Home team", team_names)
        away_choices = [x for x in team_names if x != home_name]
        away_name = c2.selectbox("Away team", away_choices)

        team_lookup = teams.set_index("TEAM_NAME")

        if st.button("Set matchup & calculate opponent profiles", type="primary"):
            hr = team_lookup.loc[home_name]
            ar = team_lookup.loc[away_name]

            setup = {
                "home_name": home_name,
                "home_id": int(hr["TEAM_ID"]),
                "home_abbr": str(hr["TEAM_ABBR"]),
                "away_name": away_name,
                "away_id": int(ar["TEAM_ID"]),
                "away_abbr": str(ar["TEAM_ABBR"]),
                "season": int(season),
                "source": "SportsDataverse ESPN WNBA boxscores",
            }

            st.session_state["game_setup"] = setup
            st.session_state["opp_profile_home"] = opponent_allowed_profile(
                team_db, setup["home_abbr"]
            )
            st.session_state["opp_profile_away"] = opponent_allowed_profile(
                team_db, setup["away_abbr"]
            )
            st.success(
                f"Loaded {setup['away_abbr']} @ {setup['home_abbr']}"
            )

    st.markdown("### Pre-game context")
    st.caption(
        "Historical stats are automatic. This is only for OUT/GTD, "
        "projected minutes and role redistribution."
    )

    upload = st.file_uploader(
        "Upload game_context.json",
        type=["json"],
        key="context_uploader"
    )

    # Process a newly uploaded file only once; otherwise Streamlit would
    # re-apply it on every rerun while the uploader remains populated.
    if upload is not None:
        try:
            raw_bytes = upload.getvalue()
            upload_sig = (upload.name, len(raw_bytes), hash(raw_bytes))
            if st.session_state.get("_context_upload_sig") != upload_sig:
                parsed = json.loads(raw_bytes.decode("utf-8"))
                st.session_state["game_context"] = parsed
                st.session_state["context_editor"] = json.dumps(
                    parsed, indent=2, ensure_ascii=False
                )
                st.session_state["_context_upload_sig"] = upload_sig
                reset_context_driven_player_widgets()
                st.success("Context JSON loaded and player inputs reset.")
                st.rerun()
        except Exception as e:
            st.error(f"Invalid JSON file: {e}")

    if "context_editor" not in st.session_state:
        current_context = st.session_state.get("game_context", {})
        st.session_state["context_editor"] = (
            json.dumps(current_context, indent=2, ensure_ascii=False)
            if current_context else ""
        )

    st.text_area(
        "Paste / edit context JSON",
        height=230,
        key="context_editor",
        placeholder=(
            '{"injuries":{"Rae Burrell":{"status":"OUT"}},'
            '"projected_minutes":{"Ariel Atkins":31},'
            '"role_adjustments":{"Ariel Atkins":{"usage":1.08,'
            '"three_role":1.12}}}'
        ),
    )

    if st.button("Apply context JSON"):
        try:
            context_text = st.session_state.get("context_editor", "")
            st.session_state["game_context"] = (
                json.loads(context_text) if context_text.strip() else {}
            )
            reset_context_driven_player_widgets()
            st.success("Context applied. Player inputs will now refresh from metadata.")
            st.rerun()
        except Exception as e:
            st.error(f"JSON error: {e}")

    setup = st.session_state.get("game_setup")
    if setup:
        st.success(
            f"Current matchup: {setup['away_abbr']} @ {setup['home_abbr']}"
        )


# =====================================================================
# TEAM MARKETS
# =====================================================================
with tab_team:
    st.subheader("Team Markets")
    setup = st.session_state.get("game_setup")

    if data_pack is None:
        st.info("Load the season database in Game Setup first.")
    elif not setup:
        st.info("Set the matchup in Game Setup first.")
    else:
        side = st.radio(
            "Team to price",
            ["Away", "Home"],
            horizontal=True,
            key="team_side"
        )

        if side == "Away":
            team_abbr, team_id = setup["away_abbr"], setup["away_id"]
            opp_abbr, opp_id = setup["home_abbr"], setup["home_id"]
            opp_profile = st.session_state["opp_profile_home"]
        else:
            team_abbr, team_id = setup["home_abbr"], setup["home_id"]
            opp_abbr, opp_id = setup["away_abbr"], setup["away_id"]
            opp_profile = st.session_state["opp_profile_away"]

        team_log = clean_team_log(
            team_db[team_db["TEAM_ID"] == team_id].copy()
        )
        opp_log = clean_team_log(
            team_db[team_db["TEAM_ID"] == opp_id].copy()
        )

        regime = st.radio(
            "Sample weighting",
            ["Stable 55/20/25", "Role change 35/20/45"],
            horizontal=True,
            key="team_regime"
        )
        cfg = (
            WeightConfig.stable()
            if regime.startswith("Stable")
            else WeightConfig.role_change()
        )

        profile, audit = build_team_profile(team_log, cfg)
        opp_off_profile, _ = build_team_profile(opp_log, cfg)
        auto = team_matchup_modifiers(opp_profile)

        default_poss = float(
            0.55 * profile.get("poss_pg", 80.0)
            + 0.45 * opp_off_profile.get("poss_pg", 80.0)
        )

        st.markdown(f"### {team_abbr} vs {opp_abbr}")

        c1,c2,c3 = st.columns(3)
        proj_poss = c1.number_input(
            "Projected possessions",
            60.0,110.0,round(default_poss,1),0.5,
            key=f"poss_{team_abbr}_{opp_abbr}"
        )
        poss_sd = c2.number_input(
            "Possession SD",1.0,8.0,3.0,0.25,
            key=f"posssd_{team_abbr}_{opp_abbr}"
        )
        n = c3.select_slider(
            "Simulations",
            [25_000,50_000,100_000,250_000,500_000],
            100_000,
            key=f"tn_{team_abbr}_{opp_abbr}"
        )

        st.caption(
            "These opponent modifiers are calculated automatically from "
            "opponent allowed per possession vs WNBA league average."
        )
        cols = st.columns(7)
        three_pa = cols[0].number_input(
            "3PA",.70,1.30,float(auto["3PA"]),.01,
            key=f"t3pa_{team_abbr}_{opp_abbr}"
        )
        two_pa = cols[1].number_input(
            "2PA",.70,1.30,float(auto["2PA"]),.01,
            key=f"t2pa_{team_abbr}_{opp_abbr}"
        )
        fta = cols[2].number_input(
            "FTA",.70,1.30,float(auto["FTA"]),.01,
            key=f"tfta_{team_abbr}_{opp_abbr}"
        )
        tov = cols[3].number_input(
            "TOV",.70,1.30,float(auto["TOV"]),.01,
            key=f"ttov_{team_abbr}_{opp_abbr}"
        )
        oreb = cols[4].number_input(
            "OREB",.70,1.30,float(auto["OREB"]),.01,
            key=f"toreb_{team_abbr}_{opp_abbr}"
        )
        ast = cols[5].number_input(
            "AST",.70,1.30,float(auto["AST"]),.01,
            key=f"tast_{team_abbr}_{opp_abbr}"
        )
        pf = cols[6].number_input(
            "PF",.70,1.30,float(auto["PF"]),.01,
            key=f"tpf_{team_abbr}_{opp_abbr}"
        )

        with st.expander("Opponent modifier audit"):
            st.dataframe(
                opp_profile["audit"].round(4),
                use_container_width=True
            )

        ctx = TeamContext(
            projected_possessions=proj_poss,
            possessions_sd=poss_sd,
            three_pa=three_pa,
            two_pa=two_pa,
            fta=fta,
            tov=tov,
            oreb=oreb,
            ast=ast,
            pf=pf,
        )

        fingerprint = (
            team_abbr,opp_abbr,regime,proj_poss,poss_sd,
            three_pa,two_pa,fta,tov,oreb,ast,pf,int(n)
        )

        if st.button(
            "Run team Monte Carlo",
            type="primary",
            key=f"run_team_{team_abbr}_{opp_abbr}"
        ):
            sim = simulate_team(profile,ctx,int(n),seed=101)
            sn = min(60_000,max(20_000,int(n)//4))
            low = simulate_team(
                profile,ctx,sn,seed=102,opportunity_mult=.95
            )
            high = simulate_team(
                profile,ctx,sn,seed=103,opportunity_mult=1.05
            )
            st.session_state["team_sim"] = (
                fingerprint,sim,low,high
            )
            st.session_state["team_audit"] = audit

        pack = st.session_state.get("team_sim")
        if pack and pack[0] == fingerprint:
            _,sim,low,high = pack
            markets = [
                "PTS","3PA","3PM","2PA","2PM","FTA",
                "TOV","OREB","AST","STL","BLK","PF"
            ]
            st.dataframe(
                market_table(sim,low,high,markets).round(2),
                use_container_width=True
            )

            a,b,c,d = st.columns(4)
            market = a.selectbox(
                "Market",markets,key=f"tmkt_{team_abbr}"
            )
            line = b.number_input(
                "Line",
                value=float(round(sim[market].mean()-.5,1)),
                step=.5,
                key=f"tline_{team_abbr}_{market}"
            )
            oo = c.number_input(
                "Over odds",1.01,20.0,1.90,.01,
                key=f"to_{team_abbr}_{market}"
            )
            uo = d.number_input(
                "Under odds",1.01,20.0,1.90,.01,
                key=f"tu_{team_abbr}_{market}"
            )

            p = price(sim[market],line,oo,uo)
            pl = price(low[market],line,oo,uo)
            ph = price(high[market],line,oo,uo)

            st.dataframe(
                pd.DataFrame([{
                    **p,
                    "bear_p_over":pl["p_over"],
                    "bull_p_over":ph["p_over"],
                    "bear_p_under":pl["p_under"],
                    "bull_p_under":ph["p_under"],
                }]).round(4),
                use_container_width=True
            )


# =====================================================================
# PLAYER PROPS
# =====================================================================
with tab_player:
    st.subheader("Player Props")
    setup = st.session_state.get("game_setup")

    if data_pack is None:
        st.info("Load the season database first.")
    elif not setup:
        st.info("Set the matchup first.")
    else:
        provider = SportsDataverseWNBA()
        pool = provider.current_player_pool(player_db)

        side = st.radio(
            "Team",
            [
                f"Away — {setup['away_abbr']}",
                f"Home — {setup['home_abbr']}"
            ],
            horizontal=True,
            key="player_team_side"
        )

        if side.startswith("Away"):
            selected_team = setup["away_abbr"]
            opp_abbr = setup["home_abbr"]
            overall_profile = st.session_state["opp_profile_home"]
        else:
            selected_team = setup["home_abbr"]
            opp_abbr = setup["away_abbr"]
            overall_profile = st.session_state["opp_profile_away"]

        tp = pool[pool["TEAM_ABBR"] == selected_team].copy()
        if tp.empty:
            st.warning("No current players found for this team.")
        else:
            labels = tp["PLAYER_NAME"].astype(str).tolist()
            player_name = st.selectbox("Player", labels)
            prow = tp[tp["PLAYER_NAME"] == player_name].iloc[0]
            player_id = prow["PLAYER_ID"]

            plog = clean_player_log(
                player_db[player_db["PLAYER_ID"] == player_id].copy()
            )

            pos_group = prow.get("POSITION_GROUP")
            pos_abbr = prow.get("POSITION_ABBR")

            pvo, plg = {}, {}
            if pos_group:
                pvo, plg = provider.position_environment(
                    player_db, opp_abbr, pos_group
                )

            matchup_mods, matchup_audit = player_matchup_modifiers(
                overall_profile, pvo, plg
            )

            regime = st.radio(
                "Sample weighting",
                ["Stable 55/20/25","Role change 35/20/45"],
                horizontal=True,
                key=f"pregime_{player_id}"
            )
            cfg = (
                WeightConfig.stable()
                if regime.startswith("Stable")
                else WeightConfig.role_change()
            )

            profile,audit = build_player_profile(plog,cfg)

            manual_context = st.session_state.get("game_context",{})
            injury_info = (
                ci_lookup(
                    manual_context.get("injuries",{}),
                    player_name,{}
                ) or {}
            )
            context_minutes = ci_lookup(
                manual_context.get("projected_minutes",{}),
                player_name,None
            )
            role = (
                ci_lookup(
                    manual_context.get("role_adjustments",{}),
                    player_name,{}
                ) or {}
            )

            # Explicit audit: show exactly what the context file contributed.
            if context_minutes is not None or injury_info or role:
                with st.expander("✅ Metadata applied to this player", expanded=True):
                    st.json({
                        "player": player_name,
                        "injury": injury_info or None,
                        "projected_minutes_from_context": context_minutes,
                        "role_adjustments_from_context": role or {},
                    })
            else:
                st.caption("No player-specific metadata found in the applied context JSON.")

            recent_min = float(plog.tail(10)["MIN"].mean())
            default_minutes = (
                float(context_minutes)
                if context_minutes is not None
                else recent_min
            )

            if injury_info:
                status = (
                    injury_info.get("status","UNKNOWN")
                    if isinstance(injury_info,dict)
                    else injury_info
                )
                if str(status).upper() == "OUT":
                    st.error(
                        f"Context marks {player_name} OUT. "
                        "Do not price unless testing a counterfactual."
                    )
                else:
                    st.info(f"Context injury status: {status}")

            st.markdown(
                f"**Position:** {pos_abbr or 'unknown'} "
                f"→ bucket `{pos_group or 'N/A'}`"
            )

            c1,c2,c3 = st.columns(3)
            pmin = c1.number_input(
                "Projected minutes",
                5.0,45.0,float(round(default_minutes,1)),.5,
                key=f"pmin_{player_id}_{opp_abbr}"
            )
            msd = c2.number_input(
                "Minutes SD",.5,6.0,2.0,.25,
                key=f"msd_{player_id}_{opp_abbr}"
            )
            n = c3.select_slider(
                "Simulations",
                [25_000,50_000,100_000,250_000,500_000],
                100_000,
                key=f"pn_{player_id}_{opp_abbr}"
            )

            st.markdown("### Automatic opponent / position modifiers")
            st.caption(
                "Overall opponent defense is the base. Position adds only "
                "its deviation from that overall profile, so we do not "
                "double-count the opponent."
            )
            st.dataframe(
                matchup_audit.round(4),
                use_container_width=True
            )

            ocols = st.columns(5)
            opp_pts = ocols[0].number_input(
                "PTS env",.70,1.30,float(matchup_mods["PTS"]),.01,
                key=f"oppts_{player_id}_{opp_abbr}"
            )
            opp_reb = ocols[1].number_input(
                "REB env",.70,1.30,float(matchup_mods["REB"]),.01,
                key=f"opreb_{player_id}_{opp_abbr}"
            )
            opp_ast = ocols[2].number_input(
                "AST env",.70,1.30,float(matchup_mods["AST"]),.01,
                key=f"opast_{player_id}_{opp_abbr}"
            )
            opp_3pa = ocols[3].number_input(
                "3PA env",.70,1.30,float(matchup_mods["3PA"]),.01,
                key=f"op3pa_{player_id}_{opp_abbr}"
            )
            opp_fta = ocols[4].number_input(
                "FTA env",.70,1.30,float(matchup_mods["FTA"]),.01,
                key=f"opfta_{player_id}_{opp_abbr}"
            )

            st.markdown("### Injury / role redistribution")
            st.caption(
                "These defaults are automatically read from Game Setup JSON."
            )
            rcols = st.columns(5)
            usage = rcols[0].number_input(
                "Usage",.70,1.50,float(role.get("usage",1.0)),.01,
                key=f"usage_{player_id}"
            )
            creation = rcols[1].number_input(
                "Creation",.70,1.50,float(role.get("creation",1.0)),.01,
                key=f"creation_{player_id}"
            )
            reb_role = rcols[2].number_input(
                "REB role",.70,1.50,
                float(role.get("reb_role",role.get("reb",1.0))),.01,
                key=f"rebrole_{player_id}"
            )
            three_role = rcols[3].number_input(
                "3PA role",.60,1.60,
                float(role.get("three_role",role.get("three_pa",1.0))),.01,
                key=f"3role_{player_id}"
            )
            fta_role = rcols[4].number_input(
                "FTA role",.60,1.60,
                float(role.get("fta_role",1.0)),.01,
                key=f"ftarole_{player_id}"
            )

            # H2H already lives inside this same game-log sample.
            h2h = plog[
                plog["OPP_ABBR"].astype(str).str.upper()
                == str(opp_abbr).upper()
            ].copy()

            st.markdown("### H2H audit — zero extra weight by default")
            if h2h.empty:
                st.caption("No same-season H2H found.")
            else:
                show = [
                    c for c in [
                        "GAME_DATE","OT_FLAG","MIN","PTS","REB",
                        "AST","FG3M","FG3A","FTA"
                    ]
                    if c in h2h.columns
                ]
                st.dataframe(
                    h2h[show].sort_values(
                        "GAME_DATE",ascending=False
                    ),
                    use_container_width=True
                )
                st.caption(
                    "These games are already in Old/Mid/L5 where applicable "
                    "and are NOT counted again."
                )

            hcols = st.columns(4)
            hp = hcols[0].slider(
                "Optional PTS H2H %",-10,10,0,
                key=f"hp_{player_id}"
            )
            hr = hcols[1].slider(
                "Optional REB H2H %",-10,10,0,
                key=f"hr_{player_id}"
            )
            ha = hcols[2].slider(
                "Optional AST H2H %",-10,10,0,
                key=f"ha_{player_id}"
            )
            h3 = hcols[3].slider(
                "Optional 3PA H2H %",-10,10,0,
                key=f"h3_{player_id}"
            )

            ctx = PlayerContext(
                projected_minutes=pmin,
                minutes_sd=msd,
                opp_pts=opp_pts,
                opp_reb=opp_reb,
                opp_ast=opp_ast,
                opp_3pa=opp_3pa,
                opp_fta=opp_fta,
                usage=usage,
                creation=creation,
                reb_role=reb_role,
                three_role=three_role,
                fta_role=fta_role,
                h2h_pts=1+hp/100,
                h2h_reb=1+hr/100,
                h2h_ast=1+ha/100,
                h2h_3pa=1+h3/100,
            )

            fingerprint = (
                player_id,opp_abbr,regime,pmin,msd,
                opp_pts,opp_reb,opp_ast,opp_3pa,opp_fta,
                usage,creation,reb_role,three_role,fta_role,
                hp,hr,ha,h3,int(n)
            )

            if st.button(
                "Run player Monte Carlo",
                type="primary",
                key=f"run_player_{player_id}"
            ):
                sim = simulate_player(profile,ctx,int(n),seed=201)
                sn = min(60_000,max(20_000,int(n)//4))
                low = simulate_player(
                    profile,ctx,sn,seed=202,opportunity_mult=.95
                )
                high = simulate_player(
                    profile,ctx,sn,seed=203,opportunity_mult=1.05
                )

                st.session_state["player_sim"] = (
                    fingerprint,sim,low,high
                )
                st.session_state["player_audit"] = audit
                st.session_state["player_matchup_audit"] = matchup_audit

            pack = st.session_state.get("player_sim")
            if pack and pack[0] == fingerprint:
                _,sim,low,high = pack
                markets = [
                    "PTS","REB","AST","3PM","PRA","PR","PA","AR"
                ]
                st.dataframe(
                    market_table(
                        sim,low,high,markets
                    ).round(2),
                    use_container_width=True
                )

                st.markdown("### Bookmaker pricing")
                rows = []
                for m in markets:
                    with st.expander(
                        m,
                        expanded=m in ("PTS","REB","AST","3PM")
                    ):
                        c1,c2,c3 = st.columns(3)
                        line = c1.number_input(
                            f"{m} line",
                            value=float(round(sim[m].mean()-.5,1)),
                            step=.5,
                            key=f"line_{player_id}_{m}"
                        )
                        oo = c2.number_input(
                            f"{m} Over",
                            1.01,20.0,1.90,.01,
                            key=f"oo_{player_id}_{m}"
                        )
                        uo = c3.number_input(
                            f"{m} Under",
                            1.01,20.0,1.90,.01,
                            key=f"uo_{player_id}_{m}"
                        )

                        p = price(sim[m],line,oo,uo)
                        pl = price(low[m],line,oo,uo)
                        ph = price(high[m],line,oo,uo)

                        rows.append({
                            "Market":m,
                            "Line":line,
                            "P(O)":p["p_over"],
                            "Fair O":p["fair_over"],
                            "Odds O":oo,
                            "EV O":p["ev_over"],
                            "Bear P(O)":pl["p_over"],
                            "Bull P(O)":ph["p_over"],
                            "P(U)":p["p_under"],
                            "Fair U":p["fair_under"],
                            "Odds U":uo,
                            "EV U":p["ev_under"],
                            "Bear P(U)":pl["p_under"],
                            "Bull P(U)":ph["p_under"],
                        })

                st.dataframe(
                    pd.DataFrame(rows).round(4),
                    use_container_width=True
                )
                st.warning(
                    "Central EV alone does not qualify a bet. "
                    "If a plausible adverse scenario kills the edge: PASS."
                )


# =====================================================================
# DATA AUDIT
# =====================================================================
with tab_audit:
    st.subheader("Data Audit")

    st.markdown("""
**Data architecture**
- SportsDataverse ESPN WNBA player boxscores = player historical database.
- SportsDataverse ESPN WNBA team boxscores = team historical database.
- No `stats.nba.com` call is required.
- Files are cached for 6 hours.

**Model protections**
- Old season / Games 6–10 / L5 do not overlap.
- H2H receives 0% extra weight by default.
- Opponent overall is normalized per possession vs league average.
- Position changes only the positional deviation from overall defense.
- Game-context JSON supplies only pre-game information.
- Pace/opportunity enters once.
- Bookmaker price is entered after projection.
    """)

    if data_pack:
        a,b,c,d = st.columns(4)
        a.metric("Player rows", f"{len(player_db):,}")
        b.metric("Team rows", f"{len(team_db):,}")
        c.metric(
            "Games",
            f"{team_db['GAME_ID'].nunique():,}"
        )
        d.metric(
            "Last date",
            str(pd.Timestamp(player_db["GAME_DATE"].max()).date())
        )

        st.markdown("### Source assets")
        st.json(data_pack["sources"])

    setup = st.session_state.get("game_setup")
    if setup:
        st.markdown("### Matchup")
        st.json(setup)

        hp = st.session_state.get("opp_profile_home")
        ap = st.session_state.get("opp_profile_away")

        if hp:
            st.markdown(
                f"#### {setup['home_abbr']} opponent allowed"
            )
            st.dataframe(
                hp["audit"].round(4),
                use_container_width=True
            )
        if ap:
            st.markdown(
                f"#### {setup['away_abbr']} opponent allowed"
            )
            st.dataframe(
                ap["audit"].round(4),
                use_container_width=True
            )

    st.markdown("### Applied manual context")
    st.json(
        st.session_state.get("game_context",{})
        or {"message":"No manual context"}
    )

    if st.session_state.get("player_audit") is not None:
        st.markdown("### Player non-overlap buckets")
        st.dataframe(
            st.session_state["player_audit"],
            use_container_width=True
        )

    if st.session_state.get("player_matchup_audit") is not None:
        st.markdown("### Player opponent / position calculation")
        st.dataframe(
            st.session_state["player_matchup_audit"],
            use_container_width=True
        )

    if st.session_state.get("team_audit") is not None:
        st.markdown("### Team non-overlap buckets")
        st.dataframe(
            st.session_state["team_audit"],
            use_container_width=True
        )

    st.write({
        "WNBA_historical_provider":
            "SportsDataverse GitHub Releases",
        "stats_nba_com_used": False,
        "BALLDONTLIE_key_configured": bool(bdl_key),
        "keys_rendered_to_UI": False,
    })


st.caption(
    "Model-implied fair odds are not yet historically calibrated true odds. "
    "Use Data Audit + stress scenarios before qualification."
)
