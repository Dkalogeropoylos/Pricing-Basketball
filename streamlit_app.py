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
from core.minutes_engine import project_team_minutes
from core.pace_engine import (
    project_game_pace,
    player_historical_pace_environment,
)


st.set_page_config(
    page_title="Basketball Pricing Engine",
    page_icon="🏀",
    layout="wide",
)
st.title("🏀 Basketball Pricing Engine v2.7.2")
st.caption(
    "Trader overrides • Learned role-aware redistribution • Rotation-similarity minutes • Shared fitted pace • "
    "No L5/L10 overlap • Existing value model preserved"
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


def reset_context_widgets():
    prefixes = (
        "deep_min_",
        "usage_",
        "creation_",
        "rebrole_",
        "3role_",
        "ftarole_",
        "sel_override_",
    )
    for key in list(st.session_state.keys()):
        if any(str(key).startswith(p) for p in prefixes):
            del st.session_state[key]
    st.session_state.pop("selected_player_board", None)


def context_role(manual_context, player_name):
    return ci_lookup(
        manual_context.get("role_adjustments", {}),
        player_name,
        {},
    ) or {}


def concat_without_attrs(frames, **kwargs):
    """
    pandas propagates/compares DataFrame.attrs during concat. v2.7 stores
    audit DataFrames in attrs, and pandas 2.x/3.x can raise ValueError while
    comparing those nested objects. Strip attrs only on temporary concat
    copies; the original frames keep their audit metadata.
    """
    clean = []
    for frame in frames:
        tmp = frame.copy()
        tmp.attrs = {}
        clean.append(tmp)
    return pd.concat(clean, **kwargs)


def current_game_pace():
    pp = st.session_state.get("pace_projection")
    if not pp:
        return None
    mode = st.session_state.get("pace_mode", "AUTO")
    if mode == "TRADER OVERRIDE":
        return float(st.session_state.get("pace_override", pp.central))
    return float(pp.central)


bdl_key = get_secret("BALLDONTLIE_API_KEY")

with st.sidebar:
    st.header("Data / Security")
    st.write("WNBA historical:", "✅ SportsDataverse")
    st.write("stats.nba.com:", "🚫 not used")
    st.write(
        "BALLDONTLIE advanced:",
        "✅ configured" if bdl_key else "⚪ optional"
    )
    st.divider()
    league = st.selectbox(
        "League",
        ["WNBA", "NBA (later)", "EuroLeague (later)", "EuroCup (later)"]
    )
    season = st.number_input(
        "Season", min_value=2002, max_value=2100, value=2026, step=1
    )
    if league != "WNBA":
        st.info("v2.6 is activated for WNBA first.")
        st.stop()


@st.cache_data(ttl=21600, show_spinner=False)
def load_sportsdataverse_season(season):
    return SportsDataverseWNBA(timeout=30).load_season(int(season))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_advanced_player(season, player_id, measure):
    provider, _ = get_advanced_provider("WNBA", bdl_key)
    if provider is None:
        return []
    return provider.player_season_advanced(
        int(season), int(player_id), measure
    )


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
            "Load the season database once. Two static release files are cached "
            "for 6 hours."
        )
        if st.button(f"Load WNBA {int(season)} database", type="primary"):
            try:
                with st.spinner("Loading WNBA historical database..."):
                    pack = load_sportsdataverse_season(int(season))
                st.session_state["sdv_data"] = pack
                for k in [
                    "game_setup", "opp_profile_home", "opp_profile_away",
                    "pace_projection", "team_sim", "player_sim",
                    "selected_player_board",
                ]:
                    st.session_state.pop(k, None)
                st.rerun()
            except Exception as e:
                st.error(f"Database load failed: {e}")
    else:
        provider = SportsDataverseWNBA()
        teams = provider.teams(team_db)

        a,b,c,d = st.columns(4)
        a.metric("Player-game rows", f"{len(player_db):,}")
        b.metric("Team-game rows", f"{len(team_db):,}")
        c.metric("Players", f"{player_db['PLAYER_ID'].nunique():,}")
        d.metric(
            "Through",
            str(pd.Timestamp(player_db["GAME_DATE"].max()).date())
        )

        team_names = teams["TEAM_NAME"].astype(str).tolist()
        c1,c2 = st.columns(2)
        home_name = c1.selectbox("Home team", team_names, key="home_team")
        away_name = c2.selectbox(
            "Away team",
            [x for x in team_names if x != home_name],
            key="away_team",
        )

        lookup = teams.set_index("TEAM_NAME")

        if st.button(
            "Set matchup & build game inputs",
            type="primary",
        ):
            hr, ar = lookup.loc[home_name], lookup.loc[away_name]
            setup = {
                "home_name": home_name,
                "home_id": int(hr["TEAM_ID"]),
                "home_abbr": str(hr["TEAM_ABBR"]),
                "away_name": away_name,
                "away_id": int(ar["TEAM_ID"]),
                "away_abbr": str(ar["TEAM_ABBR"]),
                "season": int(season),
            }
            st.session_state["game_setup"] = setup
            st.session_state["opp_profile_home"] = opponent_allowed_profile(
                team_db, setup["home_abbr"]
            )
            st.session_state["opp_profile_away"] = opponent_allowed_profile(
                team_db, setup["away_abbr"]
            )
            st.session_state["pace_projection"] = project_game_pace(
                team_db,
                setup["home_abbr"],
                setup["away_abbr"],
                WeightConfig.stable(),
            )
            st.session_state["pace_mode"] = "AUTO"
            st.session_state.pop("team_sim", None)
            st.session_state.pop("selected_player_board", None)
            st.success(
                f"Loaded {setup['away_abbr']} @ {setup['home_abbr']}"
            )

    setup = st.session_state.get("game_setup")
    pace = st.session_state.get("pace_projection")

    if setup and pace:
        st.markdown("### Shared Pace / Possessions Engine")
        st.caption(
            "The same central possessions feed Team Markets and Player Props. "
            "Fast/slow control weights are fitted from completed WNBA games; "
            "they are not a fixed 50/50 midpoint."
        )

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric(f"{setup['home_abbr']} pace", f"{pace.home_pace:.2f}")
        m2.metric(f"{setup['away_abbr']} pace", f"{pace.away_pace:.2f}")
        m3.metric("Auto projection", f"{pace.central:.2f}")
        m4.metric(
            "Fast / Slow weight",
            f"{pace.fast_weight:.2f} / {pace.slow_weight:.2f}"
        )
        m5.metric("Calibration games", f"{pace.calibration_games}")

        p1,p2 = st.columns(2)
        mode = p1.radio(
            "Pace source",
            ["AUTO", "TRADER OVERRIDE"],
            horizontal=True,
            key="pace_mode",
        )
        if mode == "TRADER OVERRIDE":
            p2.number_input(
                "Trader projected possessions",
                min_value=60.0,
                max_value=105.0,
                value=float(round(pace.central, 1)),
                step=0.5,
                key="pace_override",
            )
        else:
            p2.write(
                f"Stress band: **{pace.low:.1f} – {pace.high:.1f}** "
                f"(fit RMSE {pace.rmse:.2f})"
            )

        shared = current_game_pace()
        st.success(f"Shared game possessions used by models: **{shared:.2f}**")

        with st.expander("Market total / handicap — audit only"):
            st.caption(
                "These are NOT fed into pace or projections in v2.6. "
                "Using the sportsbook total to create the same team projections "
                "would be circular. Spread/total remain external cross-checks until "
                "we calibrate a separate market-prior/blowout layer."
            )
            c1,c2 = st.columns(2)
            c1.number_input(
                "Market total (optional)",
                min_value=0.0,
                max_value=250.0,
                value=0.0,
                step=0.5,
                key="market_total_audit",
            )
            c2.number_input(
                "Home handicap (optional)",
                min_value=-40.0,
                max_value=40.0,
                value=0.0,
                step=0.5,
                key="market_spread_audit",
            )

    st.markdown("### Trader context")
    st.caption(
        "The model does not guess injuries. Trader marks OUT/GTD and can "
        "override minutes/roles. If minutes are omitted, the Minutes Engine "
        "projects them automatically."
    )

    upload = st.file_uploader(
        "Upload game_context.json",
        type=["json"],
        key="context_uploader",
    )
    if upload is not None:
        try:
            raw = upload.getvalue()
            sig = (upload.name, len(raw), hash(raw))
            if st.session_state.get("_context_sig") != sig:
                parsed = json.loads(raw.decode("utf-8"))
                st.session_state["game_context"] = parsed
                st.session_state["context_editor"] = json.dumps(
                    parsed, indent=2, ensure_ascii=False
                )
                st.session_state["_context_sig"] = sig
                reset_context_widgets()
                st.rerun()
        except Exception as e:
            st.error(f"Invalid context JSON: {e}")

    if "context_editor" not in st.session_state:
        st.session_state["context_editor"] = json.dumps(
            st.session_state.get("game_context", {}),
            indent=2,
            ensure_ascii=False,
        ) if st.session_state.get("game_context") else ""

    st.text_area(
        "Paste / edit trader context",
        height=230,
        key="context_editor",
        placeholder=(
            '{"injuries":{"Player X":{"status":"OUT"}},'
            '"projected_minutes":{"Player Y":31},'
            '"rotation_regime":{"LAS":"role_change"}}'
        ),
    )

    if st.button("Apply trader context"):
        try:
            txt = st.session_state.get("context_editor", "")
            st.session_state["game_context"] = (
                json.loads(txt) if txt.strip() else {}
            )
            reset_context_widgets()
            st.success("Trader context applied.")
            st.rerun()
        except Exception as e:
            st.error(f"JSON error: {e}")


# =====================================================================
# TEAM MARKETS
# =====================================================================
with tab_team:
    st.subheader("Team Markets")

    setup = st.session_state.get("game_setup")
    shared_pace = current_game_pace()

    if data_pack is None:
        st.info("Load the database first.")
    elif not setup or shared_pace is None:
        st.info("Set the matchup in Game Setup first.")
    else:
        side = st.radio(
            "Team to price",
            ["Away", "Home"],
            horizontal=True,
            key="team_side",
        )

        if side == "Away":
            team_abbr, team_id = setup["away_abbr"], setup["away_id"]
            opp_abbr = setup["home_abbr"]
            opp_profile = st.session_state["opp_profile_home"]
        else:
            team_abbr, team_id = setup["home_abbr"], setup["home_id"]
            opp_abbr = setup["away_abbr"]
            opp_profile = st.session_state["opp_profile_away"]

        team_log = clean_team_log(
            team_db[team_db["TEAM_ID"] == team_id].copy()
        )

        regime = st.radio(
            "Team sample weighting",
            ["Stable 55/20/25", "Role change 35/20/45"],
            horizontal=True,
            key="team_regime",
        )
        cfg = (
            WeightConfig.stable()
            if regime.startswith("Stable")
            else WeightConfig.role_change()
        )

        profile, audit = build_team_profile(team_log, cfg)
        auto = team_matchup_modifiers(opp_profile)

        st.markdown(f"### {team_abbr} vs {opp_abbr}")
        st.info(
            f"Projected possessions = **{shared_pace:.2f}** "
            "(shared with Player Props)"
        )

        c1,c2 = st.columns(2)
        poss_sd = c1.number_input(
            "Possession SD",
            1.0,8.0,
            float(st.session_state["pace_projection"].sd),
            0.25,
            key=f"poss_sd_{team_abbr}",
        )
        n = c2.select_slider(
            "Simulations",
            [25_000,50_000,100_000,250_000,500_000],
            100_000,
            key=f"team_n_{team_abbr}",
        )

        cols = st.columns(7)
        three_pa = cols[0].number_input("3PA",.70,1.30,float(auto["3PA"]),.01,key=f"t3pa_{team_abbr}")
        two_pa = cols[1].number_input("2PA",.70,1.30,float(auto["2PA"]),.01,key=f"t2pa_{team_abbr}")
        fta = cols[2].number_input("FTA",.70,1.30,float(auto["FTA"]),.01,key=f"tfta_{team_abbr}")
        tov = cols[3].number_input("TOV",.70,1.30,float(auto["TOV"]),.01,key=f"ttov_{team_abbr}")
        oreb = cols[4].number_input("OREB",.70,1.30,float(auto["OREB"]),.01,key=f"toreb_{team_abbr}")
        ast = cols[5].number_input("AST",.70,1.30,float(auto["AST"]),.01,key=f"tast_{team_abbr}")
        pf = cols[6].number_input("PF",.70,1.30,float(auto["PF"]),.01,key=f"tpf_{team_abbr}")

        with st.expander("Opponent modifier audit"):
            st.dataframe(
                opp_profile["audit"].round(4),
                use_container_width=True,
            )

        ctx = TeamContext(
            projected_possessions=shared_pace,
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
            team_abbr, shared_pace, poss_sd, regime,
            three_pa,two_pa,fta,tov,oreb,ast,pf,int(n)
        )

        if st.button(
            "Run team Monte Carlo",
            type="primary",
            key=f"run_team_{team_abbr}",
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
                use_container_width=True,
            )

            a,b,c,d = st.columns(4)
            market = a.selectbox(
                "Market", markets, key=f"team_market_{team_abbr}"
            )
            line = b.number_input(
                "Book line",
                value=float(round(sim[market].mean() - 0.5, 1)),
                step=.5,
                key=f"team_line_{team_abbr}_{market}"
            )
            oo = c.number_input(
                "Over odds",1.01,20.0,1.90,.01,
                key=f"team_oo_{team_abbr}_{market}"
            )
            uo = d.number_input(
                "Under odds",1.01,20.0,1.90,.01,
                key=f"team_uo_{team_abbr}_{market}"
            )

            p = price(sim[market],line,oo,uo)
            pl = price(low[market],line,oo,uo)
            ph = price(high[market],line,oo,uo)

            st.dataframe(pd.DataFrame([{
                **p,
                "bear_p_over":pl["p_over"],
                "bull_p_over":ph["p_over"],
                "bear_p_under":pl["p_under"],
                "bull_p_under":ph["p_under"],
            }]).round(4), use_container_width=True)


# =====================================================================
# PLAYER PROPS
# =====================================================================
with tab_player:
    st.subheader("Player Props")

    setup = st.session_state.get("game_setup")
    shared_pace = current_game_pace()

    if data_pack is None:
        st.info("Load the database first.")
    elif not setup or shared_pace is None:
        st.info("Set the matchup first.")
    else:
        provider = SportsDataverseWNBA()
        pool = provider.current_player_pool(player_db)
        manual_context = st.session_state.get("game_context", {})

        # Background minutes engine always projects the whole rotation so the
        # team stays at 200 regulation minutes.
        base_away = project_team_minutes(
            player_db, team_db, pool,
            setup["away_abbr"], setup["away_name"],
            manual_context,
        )
        base_home = project_team_minutes(
            player_db, team_db, pool,
            setup["home_abbr"], setup["home_name"],
            manual_context,
        )
        base_minutes = concat_without_attrs(
            [base_away, base_home], ignore_index=True
        )

        st.markdown("### 1. Select players")
        all_options = []
        for _, r in base_minutes.sort_values(
            ["Team","Projected Min"], ascending=[True,False]
        ).iterrows():
            all_options.append(f"{r['Player']} — {r['Team']}")

        selected_labels = st.multiselect(
            "Click only the players you want to price",
            all_options,
            default=st.session_state.get("last_selected_players", []),
            key="selected_players",
        )
        st.session_state["last_selected_players"] = selected_labels

        selected_names = [
            label.rsplit(" — ", 1)[0]
            for label in selected_labels
        ]

        if selected_names:
            st.markdown("### 2. Minutes: AUTO or trader override")
            st.caption(
                "AUTO uses rotation similarity, starter history, OT/blowout "
                "downweighting, non-overlapping buckets and a 200-minute team "
                "constraint. Enter 0 to keep AUTO; any positive value is a "
                "trader override and the rest of that team's minutes rebalance."
            )

            override_cols = st.columns(
                min(4, max(1, len(selected_names)))
            )
            ui_overrides = {}
            for i, pname in enumerate(selected_names):
                current_row = base_minutes[
                    base_minutes["Player"] == pname
                ].iloc[0]
                with override_cols[i % len(override_cols)]:
                    val = st.number_input(
                        f"{pname} override (0=AUTO)",
                        min_value=0.0,
                        max_value=40.0,
                        value=0.0,
                        step=0.5,
                        key=f"sel_override_{pname}",
                        help=(
                            f"AUTO currently {current_row['Projected Min']:.1f} "
                            f"({current_row['Source']})"
                        ),
                    )
                    if val > 0:
                        ui_overrides[pname] = val

            # Rebalance each whole team after selected trader overrides.
            away_overrides = {
                k:v for k,v in ui_overrides.items()
                if not base_away[base_away["Player"] == k].empty
            }
            home_overrides = {
                k:v for k,v in ui_overrides.items()
                if not base_home[base_home["Player"] == k].empty
            }

            away_min = project_team_minutes(
                player_db, team_db, pool,
                setup["away_abbr"], setup["away_name"],
                manual_context, away_overrides,
            )
            home_min = project_team_minutes(
                player_db, team_db, pool,
                setup["home_abbr"], setup["home_name"],
                manual_context, home_overrides,
            )
            final_minutes = concat_without_attrs(
                [away_min, home_min], ignore_index=True
            )

            selected_min = final_minutes[
                final_minutes["Player"].isin(selected_names)
            ][[
                "Team","Player","Status","Auto Baseline Min",
                "Projected Min","Override Delta","Low Min",
                "High Min","Minutes SD","Source","Starter P",
                "Rotation Similarity","Regime"
            ]].copy()

            st.dataframe(
                selected_min.round({
                    "Projected Min":1,
                    "Low Min":1,
                    "High Min":1,
                    "Minutes SD":2,
                    "Starter P":2,
                    "Rotation Similarity":2,
                }),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Full rotation minute audit"):
                st.caption(
                    "Each team is constrained to 200 regulation minutes. "
                    "OUT/trader/metadata minutes are fixed first; AUTO minutes "
                    "absorb the remaining allocation."
                )
                st.dataframe(
                    final_minutes[[
                        "Team","Player","Status","Auto Baseline Min",
                        "Projected Min","Override Delta",
                        "Low Min","High Min","Source","Regime"
                    ]].round(2),
                    use_container_width=True,
                    hide_index=True,
                )
                sums = final_minutes.groupby("Team")["Projected Min"].sum()
                st.write({
                    team: round(float(total), 2)
                    for team,total in sums.items()
                })

                st.markdown("#### Role-aware override impact")
                impacts = []
                for team_frame in [away_min, home_min]:
                    imp = team_frame.attrs.get("override_impact")
                    if isinstance(imp, pd.DataFrame) and not imp.empty:
                        impacts.append(imp)
                if impacts:
                    st.dataframe(
                        concat_without_attrs(impacts, ignore_index=True).round(3),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No explicit minute override is active.")

                st.markdown("#### Learned replacement weights")
                selected_focus = st.selectbox(
                    "Show replacement matrix for",
                    selected_names,
                    key="replacement_focus",
                )
                matrix_parts = []
                for team_frame in [away_min, home_min]:
                    aud = team_frame.attrs.get(
                        "redistribution_matrix_audit"
                    )
                    if isinstance(aud, pd.DataFrame) and not aud.empty:
                        part = aud[
                            aud["Focal"] == selected_focus
                        ].copy()
                        if not part.empty:
                            matrix_parts.append(part)
                if matrix_parts:
                    matrix_view = concat_without_attrs(
                        matrix_parts, ignore_index=True
                    ).sort_values("Weight", ascending=False)
                    st.dataframe(
                        matrix_view[[
                            "Focal","Replacement","Weight","games",
                            "neg_slope","onoff","confidence",
                            "role_prior","Focal Pos","Replacement Pos"
                        ]].round(3),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Weight = historically learned inverse-minutes relation "
                        "+ on/off lift, shrunk toward a small positional prior "
                        "when the sample is thin."
                    )

            st.markdown("### 3. Shared pace")
            st.info(
                f"Game possessions: **{shared_pace:.2f}**. "
                "Each player's historical pace environment is calculated "
                "separately; only the relative pace difference is applied."
            )

            n = st.select_slider(
                "Simulations per selected player",
                [10_000,25_000,50_000,100_000,250_000],
                50_000,
                key="selected_sims",
            )

            if st.button(
                "Run selected players",
                type="primary",
                key="run_selected",
            ):
                board_rows = []
                detail_store = {}

                for pname in selected_names:
                    mr = final_minutes[
                        final_minutes["Player"] == pname
                    ].iloc[0]

                    if str(mr["Status"]).upper() == "OUT":
                        continue

                    pool_row = pool[
                        pool["PLAYER_NAME"] == pname
                    ]
                    if pool_row.empty:
                        continue
                    prow = pool_row.iloc[0]
                    pid = prow["PLAYER_ID"]
                    team_abbr = str(prow["TEAM_ABBR"])

                    if team_abbr == setup["away_abbr"]:
                        opp_abbr = setup["home_abbr"]
                        overall_profile = st.session_state["opp_profile_home"]
                    else:
                        opp_abbr = setup["away_abbr"]
                        overall_profile = st.session_state["opp_profile_away"]

                    plog = clean_player_log(
                        player_db[player_db["PLAYER_ID"] == pid].copy()
                    )

                    # Preserve fixed protocol. Team/player regime comes from
                    # trader rotation context or explicit role redistribution.
                    role = context_role(manual_context, pname)
                    regime = str(mr["Regime"])
                    cfg = (
                        WeightConfig.role_change()
                        if regime == "role_change" or role
                        else WeightConfig.stable()
                    )
                    profile,audit = build_player_profile(plog,cfg)

                    pos_group = prow.get("POSITION_GROUP")
                    pvo,plg = {},{}
                    if pos_group:
                        pvo,plg = provider.position_environment(
                            player_db,opp_abbr,pos_group
                        )
                    matchup_mods, matchup_audit = player_matchup_modifiers(
                        overall_profile,pvo,plg
                    )

                    historical_pace = player_historical_pace_environment(
                        plog, team_db, cfg
                    )
                    pace_mult = float(
                        np.clip(shared_pace / max(historical_pace, 1.0), .88, 1.12)
                    )

                    ctx = PlayerContext(
                        projected_minutes=float(mr["Projected Min"]),
                        minutes_sd=float(mr["Minutes SD"]),
                        pace_multiplier=pace_mult,
                        opp_pts=float(matchup_mods["PTS"]),
                        opp_reb=float(matchup_mods["REB"]),
                        opp_ast=float(matchup_mods["AST"]),
                        opp_3pa=float(matchup_mods["3PA"]),
                        opp_fta=float(matchup_mods["FTA"]),
                        usage=float(role.get("usage",1.0)),
                        creation=float(role.get("creation",1.0)),
                        reb_role=float(
                            role.get("reb_role",role.get("reb",1.0))
                        ),
                        three_role=float(
                            role.get("three_role",role.get("three_pa",1.0))
                        ),
                        fta_role=float(role.get("fta_role",1.0)),
                    )

                    seed = (abs(hash(str(pid))) % 100000) + 1000
                    sim = simulate_player(
                        profile,ctx,int(n),seed=seed
                    )

                    board_rows.append({
                        "Team":team_abbr,
                        "Player":pname,
                        "Min":float(mr["Projected Min"]),
                        "Min Low":float(mr["Low Min"]),
                        "Min High":float(mr["High Min"]),
                        "Min Source":mr["Source"],
                        "Hist Pace":historical_pace,
                        "Game Pace":shared_pace,
                        "Pace Mult":pace_mult,
                        "PTS":float(sim["PTS"].mean()),
                        "REB":float(sim["REB"].mean()),
                        "AST":float(sim["AST"].mean()),
                        "3PM":float(sim["3PM"].mean()),
                        "PRA":float(sim["PRA"].mean()),
                        "PR":float(sim["PR"].mean()),
                        "PA":float(sim["PA"].mean()),
                        "AR":float(sim["AR"].mean()),
                    })

                    detail_store[pname] = {
                        "sim":sim,
                        "profile_audit":audit,
                        "matchup_audit":matchup_audit,
                        "ctx":ctx,
                        "plog":plog,
                        "opp_abbr":opp_abbr,
                    }

                st.session_state["selected_player_board"] = pd.DataFrame(
                    board_rows
                )
                st.session_state["selected_player_details"] = detail_store

            board = st.session_state.get("selected_player_board")
            if isinstance(board,pd.DataFrame) and not board.empty:
                st.markdown("### Selected-player projections")
                st.dataframe(
                    board.round({
                        "Min":1,"Min Low":1,"Min High":1,
                        "Hist Pace":2,"Game Pace":2,"Pace Mult":3,
                        "PTS":2,"REB":2,"AST":2,"3PM":2,
                        "PRA":2,"PR":2,"PA":2,"AR":2,
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("### 4. Price one selected player")
                pname = st.selectbox(
                    "Deep-dive player",
                    board["Player"].tolist(),
                    key="deep_player",
                )
                detail = st.session_state["selected_player_details"][pname]
                sim = detail["sim"]

                with st.expander("Model audit", expanded=False):
                    st.write({
                        "opponent": detail["opp_abbr"],
                        "projected_minutes":
                            detail["ctx"].projected_minutes,
                        "minutes_sd":
                            detail["ctx"].minutes_sd,
                        "pace_multiplier":
                            detail["ctx"].pace_multiplier,
                    })
                    st.markdown("**Non-overlapping sample audit**")
                    st.dataframe(
                        detail["profile_audit"],
                        use_container_width=True,
                    )
                    st.markdown("**Opponent / position audit**")
                    st.dataframe(
                        detail["matchup_audit"].round(4),
                        use_container_width=True,
                    )

                    h2h = detail["plog"][
                        detail["plog"]["OPP_ABBR"].astype(str).str.upper()
                        == str(detail["opp_abbr"]).upper()
                    ]
                    st.markdown("**H2H — zero extra weight**")
                    if h2h.empty:
                        st.caption("No same-season H2H.")
                    else:
                        cols = [
                            c for c in [
                                "GAME_DATE","OT_FLAG","MIN","PTS","REB",
                                "AST","FG3M","FG3A","FTA"
                            ] if c in h2h.columns
                        ]
                        st.dataframe(
                            h2h[cols].sort_values(
                                "GAME_DATE",ascending=False
                            ),
                            use_container_width=True,
                        )

                markets = ["PTS","REB","AST","3PM","PRA","PR","PA","AR"]
                rows = []
                for m in markets:
                    with st.expander(m, expanded=m in ["PTS","REB","AST","3PM"]):
                        c1,c2,c3 = st.columns(3)
                        line = c1.number_input(
                            f"{pname} {m} book line",
                            value=float(round(sim[m].mean() - .5, 1)),
                            step=.5,
                            key=f"deep_line_{pname}_{m}",
                        )
                        oo = c2.number_input(
                            f"{m} Over odds",
                            1.01,20.0,1.90,.01,
                            key=f"deep_oo_{pname}_{m}",
                        )
                        uo = c3.number_input(
                            f"{m} Under odds",
                            1.01,20.0,1.90,.01,
                            key=f"deep_uo_{pname}_{m}",
                        )
                        p = price(sim[m],line,oo,uo)

                        # Stress minutes and game opportunity around the same
                        # core projection rather than changing the model thesis.
                        ctx0 = detail["ctx"]
                        low_ctx = PlayerContext(**{
                            **ctx0.__dict__,
                            "projected_minutes": max(
                                float(ctx0.projected_minutes)
                                - float(ctx0.minutes_sd),
                                1.0
                            ),
                        })
                        high_ctx = PlayerContext(**{
                            **ctx0.__dict__,
                            "projected_minutes": min(
                                float(ctx0.projected_minutes)
                                + float(ctx0.minutes_sd),
                                40.0
                            ),
                        })
                        sn = min(35_000,max(10_000,int(n)//3))
                        low_sim = simulate_player(
                            build_player_profile(
                                detail["plog"],
                                WeightConfig.role_change()
                                if str(final_minutes[
                                    final_minutes["Player"] == pname
                                ].iloc[0]["Regime"]) == "role_change"
                                else WeightConfig.stable()
                            )[0],
                            low_ctx,
                            sn,
                            seed=3101,
                            opportunity_mult=.97,
                        )
                        high_sim = simulate_player(
                            build_player_profile(
                                detail["plog"],
                                WeightConfig.role_change()
                                if str(final_minutes[
                                    final_minutes["Player"] == pname
                                ].iloc[0]["Regime"]) == "role_change"
                                else WeightConfig.stable()
                            )[0],
                            high_ctx,
                            sn,
                            seed=3102,
                            opportunity_mult=1.03,
                        )
                        pl = price(low_sim[m],line,oo,uo)
                        ph = price(high_sim[m],line,oo,uo)

                        rows.append({
                            "Market":m,"Line":line,
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
                    use_container_width=True,
                )
                st.warning(
                    "Projection/model structure is unchanged. v2.6 only improves "
                    "minutes and pace feeding. Central EV alone still does not "
                    "qualify a bet."
                )
        else:
            st.info("Select at least one player.")


# =====================================================================
# DATA AUDIT
# =====================================================================
with tab_audit:
    st.subheader("Data Audit")

    st.markdown("""
**Unchanged core protocol**
- Old season / Games 6–10 / L5 are non-overlapping.
- Stable = 55/20/25; role-change = 35/20/45.
- H2H receives 0% extra weight by default.
- Opponent overall + opponent-by-position avoid double counting.
- 3PM is generated from volume and regressed efficiency.
- Joint Monte Carlo / stress logic remains intact.

**New feeding layer**
- Trader declares availability; injuries are not guessed automatically.
- Minutes are similarity-weighted inside the existing buckets.
- OT and large blowout minute games are downweighted, not deleted.
- Full team rotation is constrained to 200 regulation minutes.
- Explicit minute overrides are redistributed through a historically learned teammate replacement matrix, not proportional roster scaling.
- OUT availability is not hit with that full matrix a second time; it is already handled by current-rotation similarity + the 200-minute constraint.
- Trader/metadata minutes override AUTO and the remaining rotation rebalances.
- Pace control is fitted from completed WNBA games, with a mild ridge prior
  rather than fixed 50/50.
- The exact same projected possessions feed Team Markets and Player Props.
- Player pace adjustment is today's game pace / that player's historical
  pace environment, so pace is applied once rather than double-counted.
- Market total/handicap are audit-only in v2.6.
    """)

    setup = st.session_state.get("game_setup")
    pace = st.session_state.get("pace_projection")
    if setup:
        st.markdown("### Matchup")
        st.json(setup)

    if pace:
        st.markdown("### Pace fit")
        st.json({
            "home_pace":pace.home_pace,
            "away_pace":pace.away_pace,
            "league_pace":pace.league_pace,
            "fast_weight":pace.fast_weight,
            "slow_weight":pace.slow_weight,
            "auto_central":pace.central,
            "shared_used":current_game_pace(),
            "sd":pace.sd,
            "calibration_games":pace.calibration_games,
        })

    st.markdown("### Trader context")
    st.json(
        st.session_state.get("game_context", {})
        or {"message":"No trader context"}
    )

    if setup:
        hp = st.session_state.get("opp_profile_home")
        ap = st.session_state.get("opp_profile_away")
        if hp:
            st.markdown(f"### {setup['home_abbr']} opponent allowed")
            st.dataframe(hp["audit"].round(4),use_container_width=True)
        if ap:
            st.markdown(f"### {setup['away_abbr']} opponent allowed")
            st.dataframe(ap["audit"].round(4),use_container_width=True)

    st.write({
        "historical_provider":"SportsDataverse",
        "stats_nba_used":False,
        "injuries_guessed_by_model":False,
        "market_total_used_as_model_input":False,
        "shared_pace_engine":True,
        "team_minutes_constraint":200,
    })


st.caption(
    "Model-implied fair odds are not yet historically calibrated true odds. "
    "v2.6 preserves the existing value model and upgrades only the input layer."
)
