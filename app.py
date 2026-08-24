
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from storage import WNBADatabase
from wnba_data import fetch_player_game_logs, fetch_team_game_logs
from features import (
    summarize_games,
    window_summary,
    selected_role_summary,
    build_rate_triplet,
    allowance_index,
)
from wnba_prop_model import (
    WNBAPropModel,
    PlayerProfile,
    MatchupContext,
    EfficiencyWindow,
)


st.set_page_config(
    page_title="WNBA Prop Model",
    page_icon="🏀",
    layout="wide",
)

DB_PATH = Path("data/wnba_props.duckdb")
db = WNBADatabase(DB_PATH)
db.init_schema()

st.title("WNBA Prop Model")
st.caption(
    "MVP: Official WNBA game logs → DuckDB → same-role selection → "
    "matchup adjustments → Monte Carlo → fair odds."
)

tabs = st.tabs(["1 · Data Sync", "2 · Player Lab", "3 · Market Pricing", "4 · Run History"])


# ---------------------------------------------------------------------
# Data Sync
# ---------------------------------------------------------------------
with tabs[0]:
    st.subheader("Live WNBA Stats sync")

    c1, c2 = st.columns(2)
    with c1:
        season = st.number_input("Season", min_value=1997, max_value=2100, value=2026, step=1)
    with c2:
        season_type = st.selectbox(
            "Season type",
            ["Regular Season", "Playoffs"],
            index=0,
        )

    st.info(
        "Primary source: stats.wnba.com (LeagueID 10). "
        "The app stores local copies so model/backtest work does not repeatedly hit the live endpoint."
    )

    if st.button("Sync player + team game logs", type="primary"):
        with st.spinner("Fetching WNBA player game logs..."):
            players = fetch_player_game_logs(int(season), season_type)
        with st.spinner("Fetching WNBA team game logs..."):
            teams = fetch_team_game_logs(int(season), season_type)

        if players.empty or teams.empty:
            st.error("The endpoint returned no data.")
        else:
            db.replace_table("player_game_logs", players)
            db.replace_table("team_game_logs", teams)
            st.success(
                f"Saved {len(players):,} player-game rows and {len(teams):,} team-game rows."
            )

    if db.table_exists("player_game_logs"):
        p = db.read_table("player_game_logs")
        st.metric("Player-game rows in local DB", f"{len(p):,}")
        st.dataframe(p.sort_values("GAME_DATE", ascending=False).head(20), use_container_width=True)
    else:
        st.warning("No player data in DuckDB yet.")

    if db.table_exists("team_game_logs"):
        t = db.read_table("team_game_logs")
        st.metric("Team-game rows in local DB", f"{len(t):,}")


# ---------------------------------------------------------------------
# Player Lab
# ---------------------------------------------------------------------
with tabs[1]:
    st.subheader("Player Lab — build the pre-market projection")

    if not (db.table_exists("player_game_logs") and db.table_exists("team_game_logs")):
        st.warning("Run Data Sync first.")
    else:
        player_logs = db.read_table("player_game_logs")
        team_logs = db.read_table("team_game_logs")

        player_names = sorted(player_logs["PLAYER_NAME"].dropna().unique().tolist())
        team_abbrs = sorted(team_logs["TEAM_ABBREVIATION"].dropna().unique().tolist())

        top1, top2, top3 = st.columns(3)
        with top1:
            player_name = st.selectbox("Player", player_names)
        with top2:
            opponent = st.selectbox("Opponent", team_abbrs)
        with top3:
            position = st.selectbox("Position / role bucket", ["G", "W", "F", "C"])

        p_logs = player_logs[player_logs["PLAYER_NAME"] == player_name].copy()
        p_logs = p_logs.sort_values("GAME_DATE", ascending=False)

        season_s = summarize_games(p_logs)
        l10_s = summarize_games(p_logs.head(10))
        l5_s = summarize_games(p_logs.head(5))
        h2h = p_logs[p_logs["OPP_ABBREVIATION"] == opponent].head(5)

        st.markdown("#### Raw context")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.write("Season")
            st.dataframe(season_s.to_frame("AVG").T, use_container_width=True)
        with s2:
            st.write("Last 10")
            st.dataframe(l10_s.to_frame("AVG").T, use_container_width=True)
        with s3:
            st.write("Last 5")
            st.dataframe(l5_s.to_frame("AVG").T, use_container_width=True)

        st.markdown("#### H2H — inspect before weighting")
        if h2h.empty:
            st.caption("No current-season H2H rows found.")
        else:
            show_cols = [
                c for c in [
                    "GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST",
                    "FGA", "FG3A", "FTA", "FG3M", "OREB", "DREB"
                ] if c in h2h.columns
            ]
            st.dataframe(h2h[show_cols], use_container_width=True)
            st.caption(
                "MVP rule: OT / different-role H2Hs are not automatically trusted. "
                "Set H2H weight manually below."
            )

        st.markdown("#### Same-role games")
        st.caption(
            "This is intentionally manual in v0.1. Tick only the games whose rotation/minutes/absence "
            "context is comparable to today's role. Later the injury/rotation layer will automate this."
        )

        role_cols = [
            c for c in [
                "GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST",
                "FGA", "FG3A", "FTA", "FG3M", "OREB", "DREB"
            ] if c in p_logs.columns
        ]
        role_df = p_logs.head(12)[role_cols].copy()
        role_df.insert(0, "USE", [True] * min(5, len(role_df)) + [False] * max(0, len(role_df) - 5))
        edited = st.data_editor(
            role_df,
            use_container_width=True,
            hide_index=True,
            disabled=[c for c in role_df.columns if c != "USE"],
        )
        role_s = selected_role_summary(edited)

        st.markdown("#### Minutes / uncertainty")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            minutes_floor = st.number_input("Minutes floor", value=float(max(0, season_s.get("MIN", 20) - 5)), step=0.5)
        with m2:
            minutes_ceiling = st.number_input("Minutes ceiling", value=float(min(40, max(l10_s.get("MIN", 25), role_s.get("MIN", 25)) + 5)), step=0.5)
        with m3:
            minutes_sd = st.number_input("Minutes SD", value=2.5, min_value=0.5, max_value=10.0, step=0.25)
        with m4:
            uncertainty_mode = st.selectbox("Information quality", ["Normal", "High uncertainty"])

        st.markdown("#### Matchup modifiers")
        st.caption(
            "Overall opponent indices are auto-computed from team game logs. "
            "Position indices remain manual in v0.1 until we complete the position table."
        )

        # Auto opponent overall indices from team-game logs.
        auto_overall = {
            "fga": allowance_index(team_logs, opponent, "FGA"),
            "3pa": allowance_index(team_logs, opponent, "FG3A"),
            "fta": allowance_index(team_logs, opponent, "FTA"),
            "reb": allowance_index(team_logs, opponent, "REB"),
            "ast": allowance_index(team_logs, opponent, "AST"),
        }

        st.json({k: round(v, 3) for k, v in auto_overall.items()})

        pos1, pos2, pos3, pos4, pos5 = st.columns(5)
        with pos1:
            pos_fga = st.number_input("Pos FGA index", value=1.00, min_value=0.70, max_value=1.30, step=0.01)
        with pos2:
            pos_3pa = st.number_input("Pos 3PA index", value=1.00, min_value=0.70, max_value=1.30, step=0.01)
        with pos3:
            pos_fta = st.number_input("Pos FTA index", value=1.00, min_value=0.70, max_value=1.30, step=0.01)
        with pos4:
            pos_reb = st.number_input("Pos REB index", value=1.00, min_value=0.70, max_value=1.30, step=0.01)
        with pos5:
            pos_ast = st.number_input("Pos AST index", value=1.00, min_value=0.70, max_value=1.30, step=0.01)

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            team_pace = st.number_input("Team pace", value=80.0, step=0.1)
        with a2:
            opp_pace = st.number_input("Opponent pace", value=80.0, step=0.1)
        with a3:
            league_pace = st.number_input("League pace", value=80.0, step=0.1)
        with a4:
            h2h_weight = st.slider("H2H weight", 0.0, 0.10, 0.03, 0.01)

        st.markdown("#### Absence / rotation adjustment")
        st.caption(
            "Only enter extra opportunities if the checked same-role games do NOT already include "
            "today's absence setup. This is the anti-double-counting rule."
        )
        x1, x2, x3, x4, x5 = st.columns(5)
        with x1:
            extra_fga = st.number_input("Extra FGA", value=0.0, step=0.25)
        with x2:
            extra_3pa = st.number_input("Extra 3PA", value=0.0, step=0.25)
        with x3:
            extra_fta = st.number_input("Extra FTA", value=0.0, step=0.25)
        with x4:
            extra_reb = st.number_input("Extra REB", value=0.0, step=0.25)
        with x5:
            extra_ast = st.number_input("Extra AST", value=0.0, step=0.25)

        # Efficiency inputs: current MVP derives raw percentages from season/L5,
        # while opponent-position / league values are editable.
        def pct(s, made, att, fallback):
            a = float(s.get(att, 0) or 0)
            m = float(s.get(made, 0) or 0)
            return fallback if a <= 0 else m / a

        season_2p = (
            (season_s.get("FG2M", 0) / season_s.get("FG2A", 1))
            if season_s.get("FG2A", 0) and season_s.get("FG2A", 0) > 0 else 0.50
        )
        recent_2p = (
            (l5_s.get("FG2M", 0) / l5_s.get("FG2A", 1))
            if l5_s.get("FG2A", 0) and l5_s.get("FG2A", 0) > 0 else season_2p
        )
        season_3p = pct(season_s, "FG3M", "FG3A", 0.33)
        recent_3p = pct(l5_s, "FG3M", "FG3A", season_3p)
        season_ft = pct(season_s, "FTM", "FTA", 0.80)
        recent_ft = pct(l5_s, "FTM", "FTA", season_ft)

        with st.expander("Efficiency regression inputs"):
            e1, e2, e3 = st.columns(3)
            with e1:
                opp_2p_pct = st.number_input("Opponent-position 2P%", value=0.50, step=0.01)
                league_2p_pct = st.number_input("League 2P%", value=0.50, step=0.01)
            with e2:
                opp_3p_pct = st.number_input("Opponent-position 3P%", value=0.33, step=0.01)
                league_3p_pct = st.number_input("League 3P%", value=0.33, step=0.01)
            with e3:
                opp_ft_pct = st.number_input("Opponent-position FT%", value=0.79, step=0.01)
                league_ft_pct = st.number_input("League FT%", value=0.79, step=0.01)

        if st.button("Build frozen projection", type="primary"):
            # Fallback: if user selected no role rows, use L5.
            if edited["USE"].sum() == 0:
                role_s = l5_s

            player = PlayerProfile(
                name=player_name,
                minutes_season=float(season_s.get("MIN", 0)),
                minutes_last10=float(l10_s.get("MIN", 0)),
                minutes_recent_role=float(role_s.get("MIN", 0)),
                minutes_floor=float(minutes_floor),
                minutes_ceiling=float(minutes_ceiling),
                minutes_sd=float(minutes_sd) * (1.35 if uncertainty_mode == "High uncertainty" else 1.0),

                fga_per_min=build_rate_triplet(season_s, l10_s, role_s, "FGA"),
                three_pa_per_min=build_rate_triplet(season_s, l10_s, role_s, "FG3A"),
                fta_per_min=build_rate_triplet(season_s, l10_s, role_s, "FTA"),
                reb_per_min=build_rate_triplet(season_s, l10_s, role_s, "REB"),
                ast_per_min=build_rate_triplet(season_s, l10_s, role_s, "AST"),

                two_pt_pct=EfficiencyWindow(
                    season=float(season_2p),
                    recent=float(recent_2p),
                    opponent_position_allowed=float(opp_2p_pct),
                    league_average=float(league_2p_pct),
                ),
                three_pt_pct=EfficiencyWindow(
                    season=float(season_3p),
                    recent=float(recent_3p),
                    opponent_position_allowed=float(opp_3p_pct),
                    league_average=float(league_3p_pct),
                ),
                ft_pct=EfficiencyWindow(
                    season=float(season_ft),
                    recent=float(recent_ft),
                    opponent_position_allowed=float(opp_ft_pct),
                    league_average=float(league_ft_pct),
                ),

                extra_fga=float(extra_fga),
                extra_three_pa=float(extra_3pa),
                extra_fta=float(extra_fta),
                extra_reb=float(extra_reb),
                extra_ast=float(extra_ast),
            )

            positional = {
                "fga": pos_fga,
                "3pa": pos_3pa,
                "fta": pos_fta,
                "reb": pos_reb,
                "ast": pos_ast,
            }

            context = MatchupContext(
                team_pace=float(team_pace),
                opponent_pace=float(opp_pace),
                league_pace=float(league_pace),
                overall_indices=auto_overall,
                positional_indices=positional,
                h2h_indices={},
                h2h_weight=float(h2h_weight),
            )

            model = WNBAPropModel(seed=20260823)
            projection = model.project(player, context)

            st.session_state["current_player"] = player
            st.session_state["current_context"] = context
            st.session_state["current_projection"] = projection
            st.session_state["current_player_name"] = player_name
            st.session_state["current_opponent"] = opponent

            st.success("Frozen projection built.")
            proj_df = pd.DataFrame([{
                "MIN": projection.minutes,
                "FGA": projection.fga,
                "3PA": projection.three_pa,
                "FTA": projection.fta,
                "PTS": projection.points,
                "REB": projection.rebounds,
                "AST": projection.assists,
                "3PM": projection.three_pm,
            }]).round(2)
            st.dataframe(proj_df, use_container_width=True)


# ---------------------------------------------------------------------
# Market Pricing
# ---------------------------------------------------------------------
with tabs[2]:
    st.subheader("Monte Carlo market pricing")

    if "current_player" not in st.session_state:
        st.warning("Build a frozen projection in Player Lab first.")
    else:
        player = st.session_state["current_player"]
        context = st.session_state["current_context"]
        proj = st.session_state["current_projection"]

        st.write(
            f"Frozen input: **{st.session_state['current_player_name']}** "
            f"vs **{st.session_state['current_opponent']}**"
        )

        mc1, mc2 = st.columns(2)
        with mc1:
            n_sims = st.selectbox(
                "Monte Carlo simulations",
                [25_000, 50_000, 100_000, 250_000],
                index=2,
            )
        with mc2:
            reliability = st.slider(
                "Model reliability multiplier",
                0.50, 1.00, 0.85, 0.05,
                help="Use lower values when injuries/rotation are uncertain.",
            )

        model = WNBAPropModel(seed=20260823)
        with st.spinner(f"Running {n_sims:,} simulations..."):
            sims = model.simulate_player(player, context, n_sims=int(n_sims))

        means = {
            k: round(float(np.mean(v)), 2)
            for k, v in sims.items()
            if k in ["PTS", "REB", "AST", "3PM", "P+R", "P+A", "A+R", "PRA"]
        }
        st.write("Simulation means")
        st.json(means)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            market = st.selectbox("Market", ["PTS", "REB", "AST", "3PM", "P+R", "P+A", "A+R", "PRA"])
        with p2:
            line = st.number_input("Line", value=10.5, step=0.5)
        with p3:
            side = st.selectbox("Side", ["over", "under"])
        with p4:
            max_units = st.number_input("Max units", value=1.25, min_value=0.0, max_value=5.0, step=0.25)

        o1, o2 = st.columns(2)
        with o1:
            selected_odds = st.number_input("Selected side odds", value=1.90, min_value=1.01, step=0.01)
        with o2:
            opposite_odds = st.number_input("Opposite side odds", value=1.90, min_value=1.01, step=0.01)

        correlation_multiplier = st.slider(
            "Portfolio correlation multiplier",
            0.40, 1.00, 1.00, 0.05,
            help="Reduce if this market overlaps heavily with bets already on the card.",
        )

        result = model.price_market(
            simulations=sims,
            market=market,
            line=float(line),
            side=side,
            bookmaker_odds=float(selected_odds),
            opposite_odds=float(opposite_odds),
            reliability_multiplier=float(reliability),
            correlation_multiplier=float(correlation_multiplier),
            max_units=float(max_units),
        )

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Model probability", f"{result.model_probability:.1%}")
        r2.metric("Fair odds", f"{result.fair_odds:.2f}")
        if result.no_vig_market_probability is not None:
            r3.metric("Market no-vig", f"{result.no_vig_market_probability:.1%}")
        if result.probability_edge_pp is not None:
            r4.metric("Edge", f"{result.probability_edge_pp:+.1f} pp")
        if result.ev_per_unit is not None:
            r5.metric("EV / unit", f"{result.ev_per_unit:+.1%}")

        st.metric("Suggested units", f"{result.suggested_units:.2f}u")

        with st.expander("Distribution quantiles"):
            q = np.quantile(sims[market], [0.10, 0.25, 0.50, 0.75, 0.90])
            st.write({
                "P10": float(q[0]),
                "P25": float(q[1]),
                "Median": float(q[2]),
                "P75": float(q[3]),
                "P90": float(q[4]),
            })

        if st.button("Save market run"):
            input_payload = {
                "player": st.session_state["current_player_name"],
                "opponent": st.session_state["current_opponent"],
                "projection": {
                    "minutes": proj.minutes,
                    "fga": proj.fga,
                    "three_pa": proj.three_pa,
                    "fta": proj.fta,
                    "points": proj.points,
                    "rebounds": proj.rebounds,
                    "assists": proj.assists,
                    "three_pm": proj.three_pm,
                },
                "n_sims": int(n_sims),
                "reliability_multiplier": float(reliability),
                "correlation_multiplier": float(correlation_multiplier),
            }

            db.add_model_run({
                "game_key": f"{st.session_state['current_player_name']} vs {st.session_state['current_opponent']}",
                "player_name": st.session_state["current_player_name"],
                "opponent": st.session_state["current_opponent"],
                "market": market,
                "line": float(line),
                "side": side,
                "book_odds": float(selected_odds),
                "model_probability": result.model_probability,
                "fair_odds": result.fair_odds,
                "no_vig_probability": result.no_vig_market_probability,
                "edge_pp": result.probability_edge_pp,
                "ev": result.ev_per_unit,
                "units": result.suggested_units,
                "input_json": json.dumps(input_payload),
            })
            st.success("Saved to model_runs.")


# ---------------------------------------------------------------------
# Run History
# ---------------------------------------------------------------------
with tabs[3]:
    st.subheader("Saved model runs")
    history = db.query("SELECT * FROM model_runs ORDER BY run_ts DESC")
    if history.empty:
        st.caption("No saved runs yet.")
    else:
        st.dataframe(history, use_container_width=True)
