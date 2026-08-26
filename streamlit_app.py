from __future__ import annotations

import copy
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
from core.team_model import (
    TeamContext, build_team_profile, simulate_game,
    team_location_modifiers, h2h_team_audit, h2h_profile_blend,
)
from core.pricing import (
    price, auto_market_table, model_line, most_market, most_market_calibrated, line_ladder,
    required_odds_for_ev,
)
from core.matchup import (
    opponent_allowed_profile,
    player_matchup_modifiers,
    team_matchup_modifiers,
    position_environment,
    block_position_susceptibility_modifier,
)
from core.minutes_engine import (
    project_team_minutes, rotation_regime_for_team, rotation_similarity_weights,
    residual_rotation_similarity_weights, h2h_rotation_similarity,
)
from core.pace_engine import (
    project_game_pace,
    player_historical_pace_environment,
)
from core.role_splits import current_out_teammates, same_role_game_weights
from core.availability import (
    confirmed_out_players, availability_state_weights, combine_game_weights,
)


st.set_page_config(
    page_title="Basketball Pricing Engine",
    page_icon="🏀",
    layout="wide",
)
st.title("🏀 Basketball Pricing Engine v2.11.0")
st.caption(
    "Confirmed-OUT state weighting • Non-overlap H2H • FGA→3P-share shot model • BLK v3 • "
    "Coupled two-team markets • Auto line ladders/play-from prices • No duplicate injury/H2H samples"
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
        "home_team_mod_",
        "away_team_mod_",
        "confirmed_out_",
    )
    for key in list(st.session_state.keys()):
        if any(str(key).startswith(p) for p in prefixes):
            del st.session_state[key]
    st.session_state.pop("selected_player_board", None)
    st.session_state.pop("team_game_sim", None)
    st.session_state.pop("home_team_weighting_mode", None)
    st.session_state.pop("away_team_weighting_mode", None)


def context_role(manual_context, player_name):
    return ci_lookup(
        manual_context.get("role_adjustments", {}),
        player_name,
        {},
    ) or {}


def _context_out_names_for_team(manual_context, pool, team_abbr):
    return confirmed_out_players(manual_context or {}, pool, team_abbr)


def _apply_confirmed_out_selection(manual_context, pool, team_abbr, selected_names):
    """Update only CONFIRMED OUT state for current-team players.

    GTD/questionable entries may remain in JSON for notes, but Team Markets do
    not probabilistically model them. A player affects availability only after
    the trader explicitly selects OUT.
    """
    ctx = copy.deepcopy(manual_context or {})
    injuries = ctx.setdefault("injuries", {})
    team_names = set(
        pool[pool["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())]
        ["PLAYER_NAME"].astype(str).tolist()
    )
    selected_cf = {str(x).casefold() for x in selected_names}
    # Remove stale OUT flags for this current team when user deselects them.
    for name in list(injuries.keys()):
        if str(name) not in team_names:
            continue
        info = injuries.get(name, {})
        status = str(info.get("status", "") if isinstance(info, dict) else info).upper()
        if status == "OUT" and str(name).casefold() not in selected_cf:
            injuries.pop(name, None)
    for name in selected_names:
        old = injuries.get(str(name), {})
        note = old.get("note") if isinstance(old, dict) else None
        injuries[str(name)] = {"status": "OUT", "team": str(team_abbr).upper()}
        if note:
            injuries[str(name)]["note"] = note
    return ctx


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
    st.subheader("Pricing discipline")
    target_ev_pct = st.select_slider(
        "Minimum model EV before calling a price playable",
        options=[4,5,6,7,8,10],
        value=6,
        help="Fair odds are break-even. Play-from odds include this extra EV buffer.",
    )
    reference_odds = st.selectbox(
        "Reference price for automatic line thresholds",
        [1.80,1.85,1.90,1.95,2.00],
        index=2,
    )
    target_ev = float(target_ev_pct) / 100.0
    st.caption(
        f"Example: fair 1.70 becomes play-from {1.70*(1+target_ev):.2f} at {target_ev:.0%} target EV."
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
                    "team_opp_profile_home", "team_opp_profile_away",
                    "pace_projection", "team_sim", "team_game_sim", "player_sim",
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
            reset_context_widgets()
            # Full opponent profiles remain available for Player Props.
            st.session_state["opp_profile_home"] = opponent_allowed_profile(
                team_db, setup["home_abbr"]
            )
            st.session_state["opp_profile_away"] = opponent_allowed_profile(
                team_db, setup["away_abbr"]
            )
            # Team Markets keep opponent-allowed evidence disjoint from the
            # explicit same-season H2H sample.
            st.session_state["team_opp_profile_home"] = opponent_allowed_profile(
                team_db, setup["home_abbr"], exclude_team_abbr=setup["away_abbr"]
            )
            st.session_state["team_opp_profile_away"] = opponent_allowed_profile(
                team_db, setup["away_abbr"], exclude_team_abbr=setup["home_abbr"]
            )
            st.session_state["pace_projection"] = project_game_pace(
                team_db,
                setup["home_abbr"],
                setup["away_abbr"],
                WeightConfig.stable(),
            )
            st.session_state["pace_mode"] = "AUTO"
            st.session_state.pop("team_sim", None)
            st.session_state.pop("team_game_sim", None)
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
    st.subheader("Team Markets — coupled game engine")

    setup = st.session_state.get("game_setup")
    shared_pace = current_game_pace()

    if data_pack is None:
        st.info("Load the database first.")
    elif not setup or shared_pace is None:
        st.info("Set the matchup in Game Setup first.")
    else:
        manual_context = st.session_state.get("game_context", {})
        provider = SportsDataverseWNBA()
        pool = provider.current_player_pool(player_db)

        home_log = clean_team_log(
            team_db[team_db["TEAM_ID"] == setup["home_id"]].copy()
        )
        away_log = clean_team_log(
            team_db[team_db["TEAM_ID"] == setup["away_id"]].copy()
        )

        st.markdown(
            f"### {setup['away_abbr']} @ {setup['home_abbr']}"
        )
        st.info(
            f"Shared projected possessions: **{shared_pace:.2f}**. "
            "Both teams are simulated in the SAME game state, so totals, "
            "rebounds, steals, blocks and 'team with most' markets are coherent."
        )

        # -----------------------------------------------------------------
        # CONFIRMED OUT state: explicit trader input only. No Q/GTD guessing.
        # -----------------------------------------------------------------
        with st.expander("Confirmed OUT availability — exact historical state", expanded=True):
            st.caption(
                "Select only confirmed OUT players. QUESTIONABLE/GTD is ignored until you explicitly mark OUT. "
                "The model looks for historical games where ALL selected OUT players were absent together. "
                "Those games are reweighted inside Old/G6–10/L5; they are NOT added as a fourth sample."
            )
            ac1, ac2, ac3 = st.columns([1, 1, 0.7])
            away_opts = sorted(
                pool[pool["TEAM_ABBR"].astype(str).str.upper().eq(setup["away_abbr"].upper())]
                ["PLAYER_NAME"].astype(str).unique().tolist(),
                key=str.casefold,
            )
            home_opts = sorted(
                pool[pool["TEAM_ABBR"].astype(str).str.upper().eq(setup["home_abbr"].upper())]
                ["PLAYER_NAME"].astype(str).unique().tolist(),
                key=str.casefold,
            )
            away_default = [x for x in _context_out_names_for_team(manual_context, pool, setup["away_abbr"]) if x in away_opts]
            home_default = [x for x in _context_out_names_for_team(manual_context, pool, setup["home_abbr"]) if x in home_opts]
            away_selected_out = ac1.multiselect(
                f"{setup['away_abbr']} confirmed OUT", away_opts, default=away_default,
                key=f"confirmed_out_{setup['away_abbr']}",
            )
            home_selected_out = ac2.multiselect(
                f"{setup['home_abbr']} confirmed OUT", home_opts, default=home_default,
                key=f"confirmed_out_{setup['home_abbr']}",
            )
            availability_k = ac3.number_input(
                "State shrink K", min_value=3.0, max_value=15.0, value=6.0, step=1.0,
                help=(
                    "Provisional partial-pooling strength. Exact-state confidence = N/(N+K). "
                    "K=6 is intentionally conservative and should later be tuned by rolling out-of-sample backtest."
                ),
                key="availability_state_k",
            )
            manual_context = _apply_confirmed_out_selection(
                manual_context, pool, setup["away_abbr"], away_selected_out
            )
            manual_context = _apply_confirmed_out_selection(
                manual_context, pool, setup["home_abbr"], home_selected_out
            )
            # Persist so the Player Props minutes/role engines see the same confirmed OUT state.
            st.session_state["game_context"] = manual_context
            st.caption(
                "Overlap guard: OUT identity is handled by the exact-state engine. Residual rotation similarity removes "
                "those OUT names before Jaccard, so the same absence is not counted twice."
            )

        # AUTO regime is separate from confirmed OUT state. Use role_change only
        # for a broader structural role/rotation change, not merely because the
        # same OUT players were selected above.
        home_regime_auto = rotation_regime_for_team(
            manual_context, setup["home_name"], setup["home_abbr"]
        )
        away_regime_auto = rotation_regime_for_team(
            manual_context, setup["away_name"], setup["away_abbr"]
        )

        with st.expander("Team sample weighting / trader override", expanded=False):
            st.caption(
                "AUTO reads rotation_regime from game_context.json. Stable = "
                "55/20/25; role_change = 35/20/45. Confirmed OUT state is a separate INNER-bucket relevance layer. "
                "Use role_change only for residual structural change so the same absence is not counted twice."
            )
            c1, c2 = st.columns(2)
            home_mode = c1.selectbox(
                f"{setup['home_abbr']} weighting",
                ["AUTO", "Stable 55/20/25", "Role change 35/20/45"],
                key="home_team_weighting_mode",
            )
            away_mode = c2.selectbox(
                f"{setup['away_abbr']} weighting",
                ["AUTO", "Stable 55/20/25", "Role change 35/20/45"],
                key="away_team_weighting_mode",
            )
            rotation_similarity_enabled = st.toggle(
                "AUTO: residual rotation similarity after OUT-state matching",
                value=True,
                help=(
                    "Confirmed OUT players are removed from this Jaccard comparison because their effect is already "
                    "handled by exact availability-state weighting. Only the residual rotation receives a weak 0.85–1.00 inner weight."
                ),
                key="team_rotation_similarity_toggle",
            )

        def resolve_cfg(mode, auto_regime, has_confirmed_out):
            if mode.startswith("Stable"):
                return WeightConfig.stable(), "stable"
            if mode.startswith("Role"):
                return WeightConfig.role_change(), "role_change"
            # Overlap guard: when confirmed OUT identity is already modeled by
            # exact-state historical matching, AUTO does not ALSO switch to the
            # 35/20/45 injury-style recency profile. If there is a broader role
            # change (trade/coaching shift), trader can explicitly choose Role change.
            if has_confirmed_out:
                return WeightConfig.stable(), "stable (OUT-state guard)"
            return (
                (WeightConfig.role_change(), "role_change")
                if auto_regime == "role_change"
                else (WeightConfig.stable(), "stable")
            )

        home_cfg, home_regime = resolve_cfg(home_mode, home_regime_auto, bool(home_selected_out))
        away_cfg, away_regime = resolve_cfg(away_mode, away_regime_auto, bool(away_selected_out))
        if (home_selected_out and home_regime_auto == "role_change" and home_mode == "AUTO") or \
           (away_selected_out and away_regime_auto == "role_change" and away_mode == "AUTO"):
            st.caption(
                "OUT-state overlap guard active: AUTO keeps Stable 55/20/25 when confirmed OUT is selected. "
                "Choose Role change explicitly only if there is an additional structural rotation change beyond those OUTs."
            )

        # -----------------------------------------------------------------
        # INNER historical relevance: exact OUT state + weak residual rotation.
        # These two layers are deliberately separated to avoid injury overlap.
        # -----------------------------------------------------------------
        home_out = confirmed_out_players(manual_context, pool, setup["home_abbr"])
        away_out = confirmed_out_players(manual_context, pool, setup["away_abbr"])

        home_avail_w, home_avail_audit, home_exact_ids = availability_state_weights(
            player_db, home_log, setup["home_abbr"], home_out, k=float(availability_k),
            exclude_opponent_abbr=setup["away_abbr"],
        )
        away_avail_w, away_avail_audit, away_exact_ids = availability_state_weights(
            player_db, away_log, setup["away_abbr"], away_out, k=float(availability_k),
            exclude_opponent_abbr=setup["home_abbr"],
        )

        home_rot_w = (
            residual_rotation_similarity_weights(
                player_db, pool, setup["home_abbr"], manual_context,
                out_players=home_out, residual_strength=0.15,
            ) if rotation_similarity_enabled else {}
        )
        away_rot_w = (
            residual_rotation_similarity_weights(
                player_db, pool, setup["away_abbr"], manual_context,
                out_players=away_out, residual_strength=0.15,
            ) if rotation_similarity_enabled else {}
        )
        home_game_weights = combine_game_weights(home_avail_w, home_rot_w)
        away_game_weights = combine_game_weights(away_avail_w, away_rot_w)

        # Baseline excludes current-opponent H2H INSIDE the actual Old/G6-10/L5
        # buckets. H2H is then added back once, with a small rotation-aware weight.
        home_profile, home_audit = build_team_profile(
            home_log, home_cfg,
            league_team_logs=team_db,
            game_weights=home_game_weights,
            exclude_opponent_abbr=setup["away_abbr"],
        )
        away_profile, away_audit = build_team_profile(
            away_log, away_cfg,
            league_team_logs=team_db,
            game_weights=away_game_weights,
            exclude_opponent_abbr=setup["home_abbr"],
        )

        h2h_all = h2h_team_audit(team_db, setup["home_abbr"], setup["away_abbr"])
        home_h2h_ids = (
            h2h_all[h2h_all["TEAM_ABBR"].astype(str).str.upper().eq(setup["home_abbr"].upper())]["GAME_ID"].astype(str).tolist()
            if not h2h_all.empty else []
        )
        away_h2h_ids = (
            h2h_all[h2h_all["TEAM_ABBR"].astype(str).str.upper().eq(setup["away_abbr"].upper())]["GAME_ID"].astype(str).tolist()
            if not h2h_all.empty else []
        )
        home_h2h_sim = h2h_rotation_similarity(
            player_db, pool, setup["home_abbr"], manual_context, home_h2h_ids
        )
        away_h2h_sim = h2h_rotation_similarity(
            player_db, pool, setup["away_abbr"], manual_context, away_h2h_ids
        )
        home_profile, home_h2h_audit = h2h_profile_blend(
            team_db, setup["home_abbr"], setup["away_abbr"], home_profile,
            rotation_similarity=home_h2h_sim,
        )
        away_profile, away_h2h_audit = h2h_profile_blend(
            team_db, setup["away_abbr"], setup["home_abbr"], away_profile,
            rotation_similarity=away_h2h_sim,
        )

        # Opponent allowance for Team Markets excludes the same H2H rows, so the
        # explicit H2H layer above is genuinely non-overlapping.
        home_opp = st.session_state.get("team_opp_profile_away") or opponent_allowed_profile(
            team_db, setup["away_abbr"], exclude_team_abbr=setup["home_abbr"]
        )
        away_opp = st.session_state.get("team_opp_profile_home") or opponent_allowed_profile(
            team_db, setup["home_abbr"], exclude_team_abbr=setup["away_abbr"]
        )
        home_auto = team_matchup_modifiers(home_opp)
        away_auto = team_matchup_modifiers(away_opp)

        # Small location correction with the current H2H opponent excluded as well.
        home_loc, home_loc_audit = team_location_modifiers(
            home_log, True, league_team_logs=team_db,
            exclude_opponent_abbr=setup["away_abbr"],
        )
        away_loc, away_loc_audit = team_location_modifiers(
            away_log, False, league_team_logs=team_db,
            exclude_opponent_abbr=setup["home_abbr"],
        )

        def combined_mods(auto, loc):
            # v2.11 removes independent 3PA and 2PA contextual multipliers.
            # Total FGA opportunity and the 3P share are distinct conditional layers.
            return {
                "FGA": float(np.clip(auto.get("FGA", 1.0) * loc.get("FGA", 1.0), 0.94, 1.06)),
                "3P_SHARE": float(np.clip(auto.get("3P_SHARE", 1.0) * loc.get("3P_SHARE", 1.0), 0.92, 1.08)),
                "FTA": float(auto.get("FTA", 1.0) * loc.get("FTA", 1.0)),
                "TOV": float(auto.get("TOV", 1.0) * loc.get("TOV", 1.0)),
                "OREB": float(auto.get("OREB", 1.0) * loc.get("OREB", 1.0)),
                "AST": float(auto.get("AST", 1.0) * loc.get("AST", 1.0)),
                "PF": float(auto.get("PF", 1.0) * loc.get("PF", 1.0)),
                "DREB": float(loc.get("DREB", 1.0)),
                "STL": float(loc.get("STL", 1.0)),
                "BLK": float(np.clip(auto.get("BLK", 1.0) * loc.get("BLK", 1.0), 0.88, 1.12)),
                "3P_PCT": float(auto.get("3P_PCT", 1.0)),
                "2P_PCT": float(auto.get("2P_PCT", 1.0)),
            }

        home_mod = combined_mods(home_auto, home_loc)
        away_mod = combined_mods(away_auto, away_loc)

        # Optional positional BLK susceptibility. It is neutral unless the
        # loaded player data contains an actual blocked-attempt field.
        home_blk_pos, home_blk_pos_audit = block_position_susceptibility_modifier(
            player_db, setup["away_abbr"], current_pool=pool, out_players=away_out
        )
        away_blk_pos, away_blk_pos_audit = block_position_susceptibility_modifier(
            player_db, setup["home_abbr"], current_pool=pool, out_players=home_out
        )

        with st.expander("Automatic matchup/location modifiers — optional trader override", expanded=False):
            st.caption(
                "Leave these untouched for full AUTO. FGA and 3P_SHARE replace independent 3PA/2PA multipliers. "
                "OREB is per miss; AST is per made FG. This keeps the main opportunity layers from being counted twice."
            )

            def modifier_editor(prefix, team_abbr, mods):
                cols = st.columns(5)
                out = {}
                keys = ["FGA","3P_SHARE","FTA","TOV","OREB","AST","PF","DREB","STL","BLK","3P_PCT","2P_PCT"]
                for i, key in enumerate(keys):
                    out[key] = cols[i % 5].number_input(
                        f"{team_abbr} {key}",
                        min_value=0.70, max_value=1.30,
                        value=float(mods[key]), step=0.01,
                        key=f"{prefix}_{key}",
                    )
                return out

            st.markdown(f"**{setup['away_abbr']}**")
            away_mod = modifier_editor("away_team_mod", setup["away_abbr"], away_mod)
            st.markdown(f"**{setup['home_abbr']}**")
            home_mod = modifier_editor("home_team_mod", setup["home_abbr"], home_mod)

        pace_obj = st.session_state["pace_projection"]
        c1, c2 = st.columns(2)
        poss_sd = c1.number_input(
            "Possession SD",
            min_value=1.0, max_value=8.0,
            value=float(pace_obj.sd), step=0.25,
            key="game_team_poss_sd",
        )
        n = c2.select_slider(
            "Game simulations",
            [25_000, 50_000, 100_000, 250_000, 500_000],
            100_000,
            key="game_team_sims",
        )

        def make_ctx(mod, blk_position=1.0):
            return TeamContext(
                projected_possessions=float(shared_pace),
                possessions_sd=float(poss_sd),
                fga=float(mod["FGA"]),
                three_share=float(mod["3P_SHARE"]),
                three_pct=float(mod.get("3P_PCT", 1.0)),
                two_pct=float(mod.get("2P_PCT", 1.0)),
                fta=float(mod["FTA"]),
                tov=float(mod["TOV"]),
                oreb=float(mod["OREB"]),
                ast=float(mod["AST"]),
                pf=float(mod["PF"]),
                dreb=float(mod["DREB"]),
                stl=float(mod["STL"]),
                blk=float(mod["BLK"]),
                blk_position=float(blk_position),
                blk_h2h=1.0,
            )

        home_ctx = make_ctx(home_mod, home_blk_pos)
        away_ctx = make_ctx(away_mod, away_blk_pos)

        fingerprint = (
            setup["home_abbr"], setup["away_abbr"], float(shared_pace),
            float(poss_sd), home_regime, away_regime, bool(rotation_similarity_enabled),
            tuple(home_out), tuple(away_out), float(availability_k),
            round(home_h2h_sim, 4), round(away_h2h_sim, 4),
            round(home_blk_pos, 4), round(away_blk_pos, 4),
            tuple(round(home_mod[k], 4) for k in sorted(home_mod)),
            tuple(round(away_mod[k], 4) for k in sorted(away_mod)),
            int(n),
        )

        if st.button("Run full-game Monte Carlo", type="primary", key="run_full_team_game"):
            home_sim, away_sim = simulate_game(
                home_profile, away_profile, home_ctx, away_ctx,
                int(n), seed=801,
            )
            sn = min(80_000, max(25_000, int(n)//4))
            low_mult = float(pace_obj.low / shared_pace) if shared_pace else 0.96
            high_mult = float(pace_obj.high / shared_pace) if shared_pace else 1.04
            home_low, away_low = simulate_game(
                home_profile, away_profile, home_ctx, away_ctx,
                sn, seed=802, opportunity_mult=low_mult,
            )
            home_high, away_high = simulate_game(
                home_profile, away_profile, home_ctx, away_ctx,
                sn, seed=803, opportunity_mult=high_mult,
            )
            st.session_state["team_game_sim"] = {
                "fingerprint": fingerprint,
                "home": home_sim, "away": away_sim,
                "home_low": home_low, "away_low": away_low,
                "home_high": home_high, "away_high": away_high,
            }

        pack = st.session_state.get("team_game_sim")
        if pack and pack.get("fingerprint") == fingerprint:
            home_sim = pack["home"]
            away_sim = pack["away"]
            home_low = pack["home_low"]
            away_low = pack["away_low"]
            home_high = pack["home_high"]
            away_high = pack["away_high"]

            markets = [
                "PTS","FGA","FGM","3PA","3PM","2PA","2PM",
                "FTA","FTM","REB","OREB","DREB","AST","STL",
                "BLK","TOV","PF",
            ]

            total_sim = home_sim[markets].add(away_sim[markets], fill_value=0)
            total_low = home_low[markets].add(away_low[markets], fill_value=0)
            total_high = home_high[markets].add(away_high[markets], fill_value=0)

            st.markdown("### Automatic model lines + fair prices")
            st.caption(
                "Example: a projection such as 30.7 is converted from the full "
                "simulated distribution to the half-point line that is closest "
                "to a 50/50 model market. Fair prices come from the simulation, "
                "not from rounding 1 / mean."
            )

            subtabs = st.tabs([
                setup["away_abbr"], setup["home_abbr"], "GAME TOTALS", "TEAM WITH MOST"
            ])
            with subtabs[0]:
                st.dataframe(
                    auto_market_table(away_sim, markets, away_low, away_high, target_ev=target_ev, reference_odds=reference_odds).round(3),
                    use_container_width=True, hide_index=True,
                )
            with subtabs[1]:
                st.dataframe(
                    auto_market_table(home_sim, markets, home_low, home_high, target_ev=target_ev, reference_odds=reference_odds).round(3),
                    use_container_width=True, hide_index=True,
                )
            with subtabs[2]:
                st.dataframe(
                    auto_market_table(total_sim, markets, total_low, total_high, target_ev=target_ev, reference_odds=reference_odds).round(3),
                    use_container_width=True, hide_index=True,
                )
            with subtabs[3]:
                most_rows = []
                most_audit_rows = []
                st.caption(
                    "PTS / match-winner is intentionally omitted. Team-with-most keeps the simulated mean difference, "
                    "but calibrates the DIFFERENCE spread to actual same-game league history; bookmaker prices are never used."
                )
                for m in [
                    "3PM","3PA","2PM","2PA","FTM","FTA",
                    "REB","OREB","DREB","AST","STL","BLK","TOV","PF",
                ]:
                    pr = most_market_calibrated(
                        home_sim[m], away_sim[m], team_logs=team_db, market=m,
                        calibration_strength=0.60,
                    )
                    most_rows.append({
                        "Market": m,
                        f"P {setup['home_abbr']}": pr["p_home"],
                        f"Fair {setup['home_abbr']}": pr["fair_home"],
                        f"Play {setup['home_abbr']} from": required_odds_for_ev(pr["p_home"], 0.0, target_ev),
                        "P Tie": pr["p_tie"],
                        "Fair Tie": pr["fair_tie"],
                        f"P {setup['away_abbr']}": pr["p_away"],
                        f"Fair {setup['away_abbr']}": pr["fair_away"],
                        f"Play {setup['away_abbr']} from": required_odds_for_ev(pr["p_away"], 0.0, target_ev),
                    })
                    most_audit_rows.append({
                        "Market": m,
                        "Raw tie": pr.get("raw_p_tie", pr["p_tie"]),
                        "Calibrated tie": pr["p_tie"],
                        "League tie": pr.get("league_tie_rate", np.nan),
                        "Raw diff SD": pr.get("raw_diff_sd", np.nan),
                        "League target diff SD": pr.get("league_target_diff_sd", np.nan),
                        "Applied diff SD": pr.get("applied_diff_sd", np.nan),
                        "Calibration games": pr.get("calibration_games", 0),
                    })
                st.dataframe(
                    pd.DataFrame(most_rows).round(3),
                    use_container_width=True, hide_index=True,
                )
                with st.expander("Team-with-most calibration audit", expanded=False):
                    st.dataframe(
                        pd.DataFrame(most_audit_rows).round(3),
                        use_container_width=True, hide_index=True,
                    )

            st.caption("No bookmaker input is required. Compare the market later against Model line / Fair / Play-from columns above.")

            with st.expander("Model audit: buckets / location / H2H / conservation"):
                st.markdown(f"**{setup['away_abbr']} buckets — {away_regime}**")
                st.dataframe(away_audit.round(4), use_container_width=True, hide_index=True)
                st.markdown(f"**{setup['home_abbr']} buckets — {home_regime}**")
                st.dataframe(home_audit.round(4), use_container_width=True, hide_index=True)

                st.markdown("**Confirmed OUT exact-state + residual rotation — overlap audit**")
                st.caption(
                    "Exact OUT-state relevance and residual rotation are both INNER-bucket weights. "
                    "Selected OUT names are removed from residual Jaccard, so the same absence is not counted twice. "
                    "GTD/Q is not modeled unless you explicitly select OUT."
                )
                st.dataframe(
                    pd.concat([away_avail_audit, home_avail_audit], ignore_index=True).round(3),
                    use_container_width=True, hide_index=True,
                )

                def _inner_weight_summary(team_abbr, game_weights, regime, out_names):
                    ws = np.asarray(list(game_weights.values()), dtype=float) if game_weights else np.asarray([])
                    return {
                        "Team": team_abbr,
                        "Regime": regime,
                        "Confirmed OUT": ", ".join(out_names) if out_names else "—",
                        "Residual rotation": "ON 0.85–1.00" if rotation_similarity_enabled else "OFF",
                        "Historical games weighted": int(len(ws)),
                        "Min combined inner weight": float(ws.min()) if ws.size else 1.0,
                        "Mean combined inner weight": float(ws.mean()) if ws.size else 1.0,
                        "Max combined inner weight": float(ws.max()) if ws.size else 1.0,
                    }

                st.dataframe(pd.DataFrame([
                    _inner_weight_summary(setup["away_abbr"], away_game_weights, away_regime, away_out),
                    _inner_weight_summary(setup["home_abbr"], home_game_weights, home_regime, home_out),
                ]).round(3), use_container_width=True, hide_index=True)

                st.markdown("**Shot architecture audit — FGA → 3P share → 3PA/2PA**")
                st.dataframe(pd.DataFrame([
                    {
                        "Team": setup["away_abbr"],
                        "Base FGA/live": away_profile.get("fga_live", np.nan),
                        "Base 3P share": away_profile.get("three_share", np.nan),
                        "Opponent+location FGA mod": away_mod.get("FGA", 1.0),
                        "Opponent+location 3P-share mod": away_mod.get("3P_SHARE", 1.0),
                    },
                    {
                        "Team": setup["home_abbr"],
                        "Base FGA/live": home_profile.get("fga_live", np.nan),
                        "Base 3P share": home_profile.get("three_share", np.nan),
                        "Opponent+location FGA mod": home_mod.get("FGA", 1.0),
                        "Opponent+location 3P-share mod": home_mod.get("3P_SHARE", 1.0),
                    },
                ]).round(4), use_container_width=True, hide_index=True)

                st.markdown("**BLK v3 audit — own ability / opponent suffered / H2H / position / tiny 2PA**")
                def _h2h_blk_weight(audit):
                    if audit is None or audit.empty:
                        return 0.0
                    hit = audit[audit["Feature"].astype(str).eq("blk_rate_pp")]
                    return float(hit.iloc[0]["Applied H2H weight"]) if not hit.empty else 0.0

                blk_audit = pd.DataFrame([
                    {
                        "Team": setup["away_abbr"],
                        "Base non-H2H BLK/poss": away_profile.get("blk_pp", np.nan),
                        "Final own BLK/poss after H2H": away_profile.get("blk_rate_pp", np.nan),
                        "Opponent BLK-suffered modifier": away_auto.get("BLK", 1.0),
                        "Location BLK modifier": away_loc.get("BLK", 1.0),
                        "Positional relative modifier": away_blk_pos,
                        "H2H weight": _h2h_blk_weight(away_h2h_audit),
                        "2PA effect cap": "±2%",
                    },
                    {
                        "Team": setup["home_abbr"],
                        "Base non-H2H BLK/poss": home_profile.get("blk_pp", np.nan),
                        "Final own BLK/poss after H2H": home_profile.get("blk_rate_pp", np.nan),
                        "Opponent BLK-suffered modifier": home_auto.get("BLK", 1.0),
                        "Location BLK modifier": home_loc.get("BLK", 1.0),
                        "Positional relative modifier": home_blk_pos,
                        "H2H weight": _h2h_blk_weight(home_h2h_audit),
                        "2PA effect cap": "±2%",
                    },
                ])
                st.dataframe(blk_audit.round(4), use_container_width=True, hide_index=True)

                hb1, hb2 = st.columns(2)
                with hb1:
                    st.caption(f"{setup['away_abbr']} disjoint H2H structural blend")
                    st.dataframe(away_h2h_audit.round(4), use_container_width=True, hide_index=True)
                with hb2:
                    st.caption(f"{setup['home_abbr']} disjoint H2H structural blend")
                    st.dataframe(home_h2h_audit.round(4), use_container_width=True, hide_index=True)

                with st.expander("Positional BLK susceptibility audit", expanded=False):
                    st.caption(
                        "Applied only if the loaded player dataset contains a real BLKA/blocked-attempt field. "
                        "Otherwise the modifier is exactly 1.00; the model never infers positional blocks from 2PA."
                    )
                    pb1, pb2 = st.columns(2)
                    pb1.dataframe(away_blk_pos_audit.round(4), use_container_width=True, hide_index=True)
                    pb2.dataframe(home_blk_pos_audit.round(4), use_container_width=True, hide_index=True)

                st.markdown("**Location correction (small shrink only)**")
                ca, ch = st.columns(2)
                ca.dataframe(away_loc_audit.round(4), use_container_width=True, hide_index=True)
                ch.dataframe(home_loc_audit.round(4), use_container_width=True, hide_index=True)

                st.markdown("**H2H raw rows — separated from baseline and opponent-allowed samples**")
                if h2h_all.empty:
                    st.caption("No same-season H2H rows found.")
                else:
                    st.dataframe(h2h_all, use_container_width=True, hide_index=True)

                # Explicit conservation checks from the actual simulations.
                checks = {
                    "Away FGA = 2PA + 3PA": bool(((away_sim["FGA"] - away_sim["2PA"] - away_sim["3PA"]) == 0).all()),
                    "Home FGA = 2PA + 3PA": bool(((home_sim["FGA"] - home_sim["2PA"] - home_sim["3PA"]) == 0).all()),
                    "Away REB = OREB + DREB": bool(((away_sim["REB"] - away_sim["OREB"] - away_sim["DREB"]) == 0).all()),
                    "Home REB = OREB + DREB": bool(((home_sim["REB"] - home_sim["OREB"] - home_sim["DREB"]) == 0).all()),
                    "Away STL <= Home TOV": bool((away_sim["STL"] <= home_sim["TOV"]).all()),
                    "Home STL <= Away TOV": bool((home_sim["STL"] <= away_sim["TOV"]).all()),
                    "Away BLK <= Home missed FGA": bool((away_sim["BLK"] <= (home_sim["FGA"]-home_sim["FGM"])).all()),
                    "Home BLK <= Away missed FGA": bool((home_sim["BLK"] <= (away_sim["FGA"]-away_sim["FGM"])).all()),
                }
                st.json(checks)


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
            same_role_enabled = st.toggle(
                "AUTO: weight historical games toward the current teammate-absence state",
                value=True,
                help=(
                    "Example: if Fudd is confirmed OUT, Bueckers games in which Fudd did not play "
                    "receive a regularized inner weight. This does NOT create a fourth sample."
                ),
                key="same_role_absence_toggle",
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
                    out_teammates = current_out_teammates(
                        manual_context, pool, team_abbr, pname
                    )
                    role_game_weights, same_role_audit = same_role_game_weights(
                        plog, player_db, team_abbr, out_teammates,
                        enabled=same_role_enabled,
                    )
                    profile,audit = build_player_profile(
                        plog,cfg,game_weights=role_game_weights
                    )

                    pos_group = prow.get("POSITION_GROUP")
                    pvo,plg = {},{}
                    if pos_group:
                        pvo,plg = position_environment(
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
                        opp_three_pct=float(matchup_mods.get("3P_PCT",1.0)),
                        opp_two_pct=float(matchup_mods.get("2P_PCT",1.0)),
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
                        "3PA":float(sim["3PA"].mean()),
                        "FTA":float(sim["FTA"].mean()),
                        "PRA":float(sim["PRA"].mean()),
                        "PR":float(sim["PR"].mean()),
                        "PA":float(sim["PA"].mean()),
                        "AR":float(sim["AR"].mean()),
                    })

                    detail_store[pname] = {
                        "sim":sim,
                        "profile_audit":audit,
                        "matchup_audit":matchup_audit,
                        "same_role_audit":same_role_audit,
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
                        "PTS":2,"REB":2,"AST":2,"3PM":2,"3PA":2,"FTA":2,
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
                markets = ["PTS","REB","AST","3PM","3PA","FTA","PRA","PR","PA","AR"]

                st.markdown("#### Automatic model lines + fair prices")
                st.dataframe(
                    auto_market_table(sim, markets, target_ev=target_ev, reference_odds=reference_odds).round(3),
                    use_container_width=True, hide_index=True,
                )

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
                    st.markdown("**Current teammate-absence / same-role audit**")
                    st.dataframe(
                        detail["same_role_audit"].round(3),
                        use_container_width=True,
                        hide_index=True,
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

                st.markdown("#### Price ladder — no bookmaker input required")
                ladder_market = st.selectbox(
                    "Market ladder", markets, key="player_ladder_market"
                )
                st.caption(
                    f"Play-from price = the minimum decimal price that gives at least "
                    f"{target_ev:.0%} model EV. Fair price alone is NOT labeled value. "
                    f"The {reference_odds:.2f} columns above also show the line needed at a common market price."
                )
                st.dataframe(
                    line_ladder(
                        sim[ladder_market],
                        center_line=model_line(sim[ladder_market])["line"],
                        radius=3,
                        target_ev=target_ev,
                    ).round(3),
                    use_container_width=True,
                    hide_index=True,
                )
                st.info(
                    "Use the sportsbook only as a comparison after the model is frozen: "
                    "match its offered line to this ladder, then require at least the displayed play-from price."
                )
        else:
            st.info("Select at least one player.")


# =====================================================================
# DATA AUDIT
# =====================================================================
with tab_audit:
    st.subheader("Data Audit")

    st.markdown("""
**v2.11 overlap protocol**
- Old season / Games 6–10 / L5 stay non-overlapping.
- Stable = 55/20/25; role-change = 35/20/45.
- Confirmed OUT is explicit trader input only; QUESTIONABLE/GTD is not probability-modeled.
- Exact OUT-state games are reweighted INSIDE the existing buckets, never added as a fourth sample.
- Residual Jaccard removes those selected OUT names first and is only 0.85–1.00, so the same absence is not counted twice.
- Same-season H2H rows are removed from baseline buckets, opponent-allowed profiles and location splits before H2H is added back once with a small rotation-aware weight.
- Team shot generation is conditional: possessions → TOV → FGA → 3P share → 3PA/2PA. Independent 3PA/2PA Poisson draws are gone.
- Team OREB matchup uses OREB per miss; AST matchup uses AST per made FG; TOV remains per possession.
- Shooting percentages remain heavily regressed and are not set by L5 hot/cold results.
- Team BLK = own ability × opponent blocks-suffered × disjoint H2H × optional relative positional susceptibility × tiny 2PA opportunity nudge (max ±2%).
- BLK conservation is against total missed FGA, not only missed 2PA, because three-point attempts can also be blocked.
- Positional BLK is applied only when a real player-level BLKA/blocked-attempt field exists; otherwise it is exactly neutral rather than inferred from 2PA.
- REB still uses available misses; no extra 3PA→REB multiplier is added, avoiding overlap. A 2P/3P rebound split waits for real play-by-play attribution.
- Fair price is break-even only; Play-from price adds the selected target-EV buffer.
- Team-with-most calibration remains downstream of the joint simulation and does not use bookmaker prices.

**Minutes / pace**
- Full team rotation is constrained to 200 regulation minutes.
- Explicit minute overrides use the historically learned replacement matrix.
- Player Props continue to use the same confirmed OUT context.
- Pace control is fitted from completed WNBA games with a mild ridge prior.
- The exact same projected possessions feed Team Markets and Player Props.
- Market total/handicap remain audit-only.
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
        thp = st.session_state.get("team_opp_profile_home")
        tap = st.session_state.get("team_opp_profile_away")
        if thp:
            st.markdown(f"### {setup['home_abbr']} TEAM-MARKET opponent allowed (current H2H excluded)")
            st.dataframe(thp["audit"].round(4), use_container_width=True)
        if tap:
            st.markdown(f"### {setup['away_abbr']} TEAM-MARKET opponent allowed (current H2H excluded)")
            st.dataframe(tap["audit"].round(4), use_container_width=True)

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
    "v2.11 changes the team opportunity architecture and availability/H2H handling; re-backtest before treating model fair odds as calibrated true probabilities."
)
