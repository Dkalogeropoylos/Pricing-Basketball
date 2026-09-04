
import json
import pandas as pd
import streamlit as st


PAGE_SIZE = 1000


# ==========================================
# LOAD ALL ANALYSIS DATA
# ==========================================

def load_analysis_data(
    supabase
):

    rows = []
    start = 0

    try:

        while True:

            response = (
                supabase
                .table("bets")
                .select("*")
                .eq(
                    "is_deleted",
                    False
                )
                .eq(
                    "needs_review",
                    False
                )
                .neq(
                    "result",
                    "Pending"
                )
                .order(
                    "bet_date",
                    desc=False
                )
                .order(
                    "bet_number",
                    desc=False
                )
                .range(
                    start,
                    start + PAGE_SIZE - 1
                )
                .execute()
            )

            page = (
                response.data
                or []
            )

            rows.extend(page)

            if len(page) < PAGE_SIZE:
                break

            start += PAGE_SIZE


    except Exception as e:

        st.error(
            f"Could not load analysis data: {e}"
        )

        return pd.DataFrame()


    if not rows:

        return pd.DataFrame()


    return pd.DataFrame(rows)


# ==========================================
# DISPLAY SELECTION
# ==========================================

def format_selection(row):

    scope = row.get("scope")

    subject = (
        row.get("subject")
        or ""
    )

    selection_2 = (
        row.get("selection_2")
        or ""
    )

    market = (
        row.get("market")
        or ""
    )

    side = (
        row.get("side")
        or ""
    )

    line = row.get("line")


    if scope == "OUTRIGHT":

        if market == "Final Matchup":

            if selection_2:

                return (
                    f"{subject} vs "
                    f"{selection_2}"
                )

            return subject


        if market == "Straight Forecast":

            if selection_2:

                return (
                    f"1st: {subject} | "
                    f"2nd: {selection_2}"
                )

            return (
                f"1st: {subject}"
            )


        if (
            market.startswith("Top ")
            and market.endswith(" - Team")
        ):

            if selection_2:

                return (
                    f"{subject} "
                    f"({selection_2})"
                )

            return subject


        return subject


    result = side


    if pd.notna(line):

        result = (
            f"{result} "
            f"{float(line):g}"
        ).strip()


    return result



# ==========================================
# COMBO HELPERS
# ==========================================

def _normalise_combo_legs(value):
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    return []


def _combo_flat_selections(value):
    combo_legs = _normalise_combo_legs(value)
    selections = []

    for component in combo_legs:
        if not isinstance(component, dict):
            continue

        if component.get("kind") in ["SINGLE", "OUTRIGHT"]:
            selections.append(component)

        elif component.get("kind") == "BET_BUILDER":
            for selection in component.get("selections", []) or []:
                if isinstance(selection, dict):
                    selections.append(selection)

    return selections


def _combo_total_selections(value):
    return len(_combo_flat_selections(value))


def _combo_component_count(value):
    return len(_normalise_combo_legs(value))


def _combo_bb_sizes(value):
    sizes = []
    for component in _normalise_combo_legs(value):
        if (
            isinstance(component, dict)
            and component.get("kind") == "BET_BUILDER"
        ):
            sizes.append(
                len(component.get("selections", []) or [])
            )
    return sizes


def _combo_profile(value):
    for component in _normalise_combo_legs(value):
        if isinstance(component, dict):
            profile = (
                component.get("combo_profile")
                or component.get("parlay_profile")
            )
            if profile:
                profile = str(profile)
                if profile == "Combo Values":
                    return "Value"
                return profile
    return None


def _combo_parlay_profile(value):
    return _combo_profile(value)


def _combo_has_outright_leg(value):
    return any(
        isinstance(component, dict)
        and str(component.get("kind") or "").upper() == "OUTRIGHT"
        for component in _normalise_combo_legs(value)
    )


def _bet_format_from_row(row):
    market = row.get("market")
    is_outright_parlay = (
        market == "Parlay"
        and _combo_has_outright_leg(
            row.get("combo_legs")
        )
    )
    if row.get("scope") == "OUTRIGHT" or is_outright_parlay:
        return "Outright"
    if market == "Bet Builder":
        return "Bet Builder"
    if market == "Parlay":
        return "Parlay"
    if market == "Abuse":
        return "Abuse"
    return "Single"


def _outright_type_from_row(row):
    is_outright_parlay = (
        row.get("market") == "Parlay"
        and _combo_has_outright_leg(
            row.get("combo_legs")
        )
    )
    if is_outright_parlay:
        return "Outright Parlay"
    if row.get("scope") == "OUTRIGHT":
        return "Single Outright"
    return None


# ==========================================
# ANALYSIS PAGE
# ==========================================

def analysis_page(
    supabase,
    load_tipsters
):

    st.header(
        "📊 Analysis"
    )


    df = load_analysis_data(
        supabase
    )


    if df.empty:

        st.info(
            "No settled bets available "
            "for analysis yet."
        )

        return


    # ======================================
    # TIPSTERS
    # ======================================

    tipsters = load_tipsters()


    tipster_map = {
        t["id"]:
            t["name"]
        for t in tipsters
    }


    df["tipster_name"] = (
        df["tipster_id"]
        .map(
            tipster_map
        )
    )

    # Shared picks keep the identity of the user who originally shared them.
    # Keep this separate from tipsters so Analysis can distinguish
    # "TIPSTER = Chat GPT" from "SHARED BY = another app user".
    df["shared_by"] = None

    if "shared_from_email" in df.columns:
        shared_mask = df["origin"].eq("SHARED")
        shared_names = (
            df["shared_from_email"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.split("@")
            .str[0]
        )
        shared_names = shared_names.replace("", "Unknown user")
        df.loc[shared_mask, "shared_by"] = shared_names[shared_mask]


    # ======================================
    # NUMERIC DATA
    # ======================================

    numeric_columns = [
        "stake",
        "profit",
        "market_odds",
        "my_odds",
        "p_market",
        "p_you",
        "edge_pp",
        "ev_pct",
        "cashout_return"
    ]


    for column in numeric_columns:

        if column in df.columns:

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                )
            )


    # ======================================
    # BET TIMING
    # ======================================

    if "is_live" not in df.columns:
        df["is_live"] = False

    df["is_live"] = (
        df["is_live"]
        .fillna(False)
        .astype(bool)
    )

    if "combo_legs" not in df.columns:
        df["combo_legs"] = None

    legacy_outright_parlay_mask = (
        df["market"].eq("Parlay")
        & df["combo_legs"].apply(
            _combo_has_outright_leg
        )
    )
    df.loc[
        legacy_outright_parlay_mask,
        "scope"
    ] = "OUTRIGHT"

    df["bet_format"] = df.apply(
        _bet_format_from_row,
        axis=1
    )

    df["outright_type"] = df.apply(
        _outright_type_from_row,
        axis=1
    )


    # ======================================
    # FILTERS — MULTI SELECT
    # ======================================

    st.subheader(
        "Filters"
    )

    st.caption(
        "Select one or more values in each filter. "
        "Leave a filter empty to include all values."
    )


    multi_keys = [
        "analysis_sports_multi",
        "analysis_scopes_multi",
        "analysis_formats_multi",
        "analysis_outright_types_multi",
        "analysis_leagues_multi",
        "analysis_markets_multi",
        "analysis_sides_multi",
        "analysis_origins_multi",
        "analysis_confidences_multi",
        "analysis_tipsters_multi",
        "analysis_shared_by_multi",
        "analysis_results_multi",
        "analysis_timings_multi",
        "analysis_reasons_multi",
        "analysis_players_multi",
        "analysis_combo_selection_count",
        "analysis_parlay_component_count"
    ]


    if st.button(
        "↺ Clear Analysis Filters",
        key="analysis_clear_multi_filters"
    ):

        for key in multi_keys:
            st.session_state.pop(
                key,
                None
            )

        st.rerun()


    def clean_options(series):

        values = []
        seen = set()

        for value in (
            series
            .dropna()
            .tolist()
        ):

            value = str(value).strip()

            if not value:
                continue

            folded = value.casefold()

            if folded in seen:
                continue

            seen.add(folded)
            values.append(value)

        return sorted(
            values,
            key=lambda x: x.casefold()
        )


    def apply_multi_filter(
        frame,
        column,
        selected
    ):

        if not selected:
            return frame

        return frame[
            frame[column].isin(selected)
        ]


    # ------------------------------
    # SPORT / BET TYPE
    # ------------------------------

    c1, c2 = st.columns(2)


    with c1:

        sport_options = clean_options(
            df["sport"]
        )

        sports_selected = st.multiselect(
            "Sport",
            sport_options,
            key="analysis_sports_multi",
            placeholder="All sports"
        )


    sport_source = apply_multi_filter(
        df,
        "sport",
        sports_selected
    )


    with c2:

        scope_options = clean_options(
            sport_source["scope"]
        )

        scopes_selected = st.multiselect(
            "Bet Type",
            scope_options,
            key="analysis_scopes_multi",
            placeholder="All bet types"
        )


    scope_source = apply_multi_filter(
        sport_source,
        "scope",
        scopes_selected
    )


    format_options = clean_options(
        scope_source["bet_format"]
    )

    formats_selected = st.multiselect(
        "Bet Structure",
        format_options,
        key="analysis_formats_multi",
        placeholder="Singles + Bet Builders + Parlays"
    )

    format_source = apply_multi_filter(
        scope_source,
        "bet_format",
        formats_selected
    )

    outright_type_options = clean_options(
        format_source["outright_type"]
    )

    outright_types_selected = st.multiselect(
        "Outright Type",
        outright_type_options,
        key="analysis_outright_types_multi",
        placeholder="Single Outright + Outright Parlay"
    )

    outright_source = apply_multi_filter(
        format_source,
        "outright_type",
        outright_types_selected
    )


    # ------------------------------
    # LEAGUE / MARKET
    # ------------------------------

    c1, c2 = st.columns(2)


    with c1:

        league_options = clean_options(
            outright_source["league"]
        )

        leagues_selected = st.multiselect(
            "League / Tour",
            league_options,
            key="analysis_leagues_multi",
            placeholder="All leagues / tours"
        )


    league_source = apply_multi_filter(
        outright_source,
        "league",
        leagues_selected
    )


    with c2:

        market_options = clean_options(
            league_source["market"]
        )

        markets_selected = st.multiselect(
            "Market",
            market_options,
            key="analysis_markets_multi",
            placeholder="All markets"
        )


    market_source = apply_multi_filter(
        league_source,
        "market",
        markets_selected
    )


    # ------------------------------
    # SIDE / ORIGIN
    # ------------------------------

    c1, c2 = st.columns(2)


    with c1:

        side_options = clean_options(
            market_source["side"]
        )

        sides_selected = st.multiselect(
            "Side / Selection",
            side_options,
            key="analysis_sides_multi",
            placeholder="All sides / selections"
        )


    side_source = apply_multi_filter(
        market_source,
        "side",
        sides_selected
    )


    with c2:

        origin_options = clean_options(
            side_source["origin"]
        )

        origins_selected = st.multiselect(
            "Origin",
            origin_options,
            key="analysis_origins_multi",
            placeholder="All origins"
        )


    origin_source = apply_multi_filter(
        side_source,
        "origin",
        origins_selected
    )


    # ------------------------------
    # CONFIDENCE / TIPSTER / SHARED BY
    # ------------------------------

    c1, c2, c3 = st.columns(3)


    with c1:

        confidence_options = clean_options(
            origin_source["confidence"]
        )

        confidences_selected = st.multiselect(
            "Confidence",
            confidence_options,
            key="analysis_confidences_multi",
            placeholder="All confidence levels"
        )


    confidence_source = apply_multi_filter(
        origin_source,
        "confidence",
        confidences_selected
    )


    with c2:

        tipster_options = clean_options(
            confidence_source[
                "tipster_name"
            ]
        )

        tipsters_selected = st.multiselect(
            "Specific Tipster",
            tipster_options,
            key="analysis_tipsters_multi",
            placeholder="All tipsters"
        )


    tipster_source = apply_multi_filter(
        confidence_source,
        "tipster_name",
        tipsters_selected
    )


    with c3:

        shared_options = clean_options(
            tipster_source.loc[
                tipster_source["origin"].eq("SHARED"),
                "shared_by"
            ]
        )

        shared_by_selected = st.multiselect(
            "Shared By",
            shared_options,
            key="analysis_shared_by_multi",
            placeholder="All shared users"
        )


    shared_source = apply_multi_filter(
        tipster_source,
        "shared_by",
        shared_by_selected
    )


    # ------------------------------
    # RESULT / TIMING
    # ------------------------------

    c1, c2 = st.columns(2)


    with c1:

        result_options = [
            result
            for result in [
                "Win",
                "Loss",
                "Cashout",
                "Void"
            ]
            if result in set(
                shared_source[
                    "result"
                ]
                .dropna()
                .tolist()
            )
        ]

        results_selected = st.multiselect(
            "Result",
            result_options,
            key="analysis_results_multi",
            placeholder="All results"
        )


    result_source = apply_multi_filter(
        shared_source,
        "result",
        results_selected
    )


    with c2:

        timings_selected = st.multiselect(
            "Bet Timing",
            [
                "Pre-live",
                "Live"
            ],
            key="analysis_timings_multi",
            placeholder="All timings"
        )


    timing_source = result_source.copy()


    if timings_selected:

        timing_mask = pd.Series(
            False,
            index=timing_source.index
        )

        if "Live" in timings_selected:

            timing_mask = (
                timing_mask
                | timing_source[
                    "is_live"
                ]
            )

        if "Pre-live" in timings_selected:

            timing_mask = (
                timing_mask
                | ~timing_source[
                    "is_live"
                ]
            )

        timing_source = timing_source[
            timing_mask
        ]


    # ------------------------------
    # REASON / PLAYER
    # ------------------------------

    c1, c2 = st.columns(2)


    with c1:

        reason_options = clean_options(
            timing_source[
                "primary_reason"
            ]
        )

        reasons_selected = st.multiselect(
            "Primary Reason",
            reason_options,
            key="analysis_reasons_multi",
            placeholder="All primary reasons"
        )


    reason_source = apply_multi_filter(
        timing_source,
        "primary_reason",
        reasons_selected
    )


    with c2:

        player_source = reason_source[
            reason_source["scope"]
            == "PLAYER"
        ]

        player_options = clean_options(
            player_source["subject"]
        )

        players_selected = st.multiselect(
            "Player",
            player_options,
            key="analysis_players_multi",
            placeholder="All players"
        )


    # ======================================
    # APPLY FILTERS
    # ======================================

    filtered = reason_source.copy()


    if players_selected:

        filtered = filtered[
            filtered["subject"]
            .isin(players_selected)
        ]


    if filtered.empty:

        st.warning(
            "No bets match these filters."
        )

        return


    # ======================================
    # FINANCIAL PERFORMANCE
    # ======================================

    performance = filtered[
        filtered["result"]
        .isin([
            "Win",
            "Loss",
            "Cashout"
        ])
    ].copy()


    st.divider()

    st.subheader(
        "💰 Financial Performance"
    )


    if not performance.empty:

        total_bets = len(
            performance
        )


        total_stake = (
            performance["stake"]
            .fillna(0)
            .sum()
        )


        total_profit = (
            performance["profit"]
            .fillna(0)
            .sum()
        )


        total_return = (
            total_stake
            + total_profit
        )


        realized_roi = (
            total_profit
            / total_stake
            * 100
            if total_stake > 0
            else 0
        )


        decisions = performance[
            performance["result"]
            .isin([
                "Win",
                "Loss"
            ])
        ]


        if not decisions.empty:

            wins = (
                decisions["result"]
                .eq("Win")
                .sum()
            )


            win_rate = (
                wins
                / len(decisions)
                * 100
            )


            win_rate_text = (
                f"{win_rate:.2f}%"
            )


        else:

            win_rate_text = "—"


        c1, c2, c3, c4 = (
            st.columns(4)
        )


        with c1:

            st.metric(
                "Bets",
                total_bets
            )


        with c2:

            st.metric(
                "Total Stake",
                f"{total_stake:.2f}"
            )


        with c3:

            st.metric(
                "Total Return",
                f"{total_return:.2f}"
            )


        with c4:

            st.metric(
                "Net Profit",
                f"{total_profit:+.2f}"
            )


        c1, c2 = st.columns(2)


        with c1:

            st.metric(
                "Realized ROI",
                f"{realized_roi:+.2f}%"
            )


        with c2:

            st.metric(
                "Win Rate",
                win_rate_text
            )


    # ======================================
    # CASHOUT PERFORMANCE
    # ======================================

    cashouts = filtered[
        filtered["result"]
        == "Cashout"
    ].copy()


    if not cashouts.empty:

        cashout_stake = (
            cashouts["stake"]
            .fillna(0)
            .sum()
        )


        cashout_return = (
            cashouts[
                "cashout_return"
            ]
            .fillna(0)
            .sum()
        )


        cashout_profit = (
            cashouts["profit"]
            .fillna(0)
            .sum()
        )


        cashout_roi = (
            cashout_profit
            / cashout_stake
            * 100
            if cashout_stake > 0
            else 0
        )


        st.subheader(
            "💰 Cashout Performance"
        )


        c1, c2, c3, c4 = (
            st.columns(4)
        )


        with c1:

            st.metric(
                "Cashouts",
                len(cashouts)
            )


        with c2:

            st.metric(
                "Cashout Return",
                f"{cashout_return:.2f}"
            )


        with c3:

            st.metric(
                "Cashout Profit",
                f"{cashout_profit:+.2f}"
            )


        with c4:

            st.metric(
                "Cashout ROI",
                f"{cashout_roi:+.2f}%"
            )


    # ======================================
    # EXPECTED VS ACTUAL
    # ======================================

    value_sample = filtered[
        filtered["ev_pct"]
        .notna()
        & filtered["result"]
        .isin([
            "Win",
            "Loss",
            "Cashout"
        ])
    ].copy()


    if not value_sample.empty:

        value_stake = (
            value_sample["stake"]
            .fillna(0)
            .sum()
        )


        value_sample[
            "expected_profit"
        ] = (
            value_sample["stake"]
            * value_sample["ev_pct"]
            / 100
        )


        expected_profit = (
            value_sample[
                "expected_profit"
            ]
            .fillna(0)
            .sum()
        )


        expected_roi = (
            expected_profit
            / value_stake
            * 100
            if value_stake > 0
            else 0
        )


        actual_profit = (
            value_sample["profit"]
            .fillna(0)
            .sum()
        )


        actual_roi = (
            actual_profit
            / value_stake
            * 100
            if value_stake > 0
            else 0
        )


        roi_difference = (
            actual_roi
            - expected_roi
        )


        st.divider()

        st.subheader(
            "📐 Expected vs Actual"
        )


        st.caption(
            "Compares the value estimated "
            "at entry with the money "
            "actually made."
        )


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:

            st.metric(
                "Expected ROI at Entry",
                f"{expected_roi:+.2f}%"
            )


        with c2:

            st.metric(
                "Realized ROI",
                f"{actual_roi:+.2f}%"
            )


        with c3:

            st.metric(
                "ROI Difference",
                f"{roi_difference:+.2f} pp"
            )


        c1, c2 = st.columns(2)


        with c1:

            st.metric(
                "Expected Profit",
                f"{expected_profit:+.2f}"
            )


        with c2:

            st.metric(
                "Actual Profit",
                f"{actual_profit:+.2f}"
            )


    # ======================================
    # PROBABILITY CALIBRATION
    # ======================================

    calibration = filtered[
        filtered["p_you"]
        .notna()
        & filtered["result"]
        .isin([
            "Win",
            "Loss"
        ])
    ].copy()


    if not calibration.empty:

        calibration[
            "actual_win"
        ] = (
            calibration["result"]
            .eq("Win")
            .astype(int)
        )


        avg_probability = (
            calibration["p_you"]
            .mean()
            * 100
        )


        actual_hit_rate = (
            calibration[
                "actual_win"
            ]
            .mean()
            * 100
        )


        calibration_difference = (
            actual_hit_rate
            - avg_probability
        )


        st.divider()

        st.subheader(
            "🎯 Probability Calibration"
        )


        st.caption(
            "Cashouts are excluded. "
            "This checks whether your "
            "estimated probabilities "
            "match actual Win/Loss outcomes."
        )


        c1, c2, c3, c4 = (
            st.columns(4)
        )


        with c1:

            st.metric(
                "Completed Bets",
                len(calibration)
            )


        with c2:

            st.metric(
                "Avg Your Probability",
                f"{avg_probability:.2f}%"
            )


        with c3:

            st.metric(
                "Actual Hit Rate",
                f"{actual_hit_rate:.2f}%"
            )


        with c4:

            st.metric(
                "Calibration Difference",
                f"{calibration_difference:+.2f} pp"
            )


        calibration[
            "probability_pct"
        ] = (
            calibration["p_you"]
            * 100
        )


        bins = [
            0,
            40,
            50,
            55,
            60,
            65,
            70,
            80,
            100.01
        ]


        labels = [
            "<40%",
            "40-50%",
            "50-55%",
            "55-60%",
            "60-65%",
            "65-70%",
            "70-80%",
            "80%+"
        ]


        calibration[
            "Probability Band"
        ] = pd.cut(
            calibration[
                "probability_pct"
            ],
            bins=bins,
            labels=labels,
            right=False
        )


        calibration_table = (
            calibration
            .groupby(
                "Probability Band",
                observed=True
            )
            .agg(
                Bets=(
                    "id",
                    "count"
                ),
                Your_Probability=(
                    "probability_pct",
                    "mean"
                ),
                Actual_Hit_Rate=(
                    "actual_win",
                    "mean"
                )
            )
            .reset_index()
        )


        calibration_table[
            "Actual_Hit_Rate"
        ] = (
            calibration_table[
                "Actual_Hit_Rate"
            ]
            * 100
        )


        calibration_table[
            "Difference_pp"
        ] = (
            calibration_table[
                "Actual_Hit_Rate"
            ]
            - calibration_table[
                "Your_Probability"
            ]
        )


        calibration_table = (
            calibration_table
            .rename(
                columns={
                    "Your_Probability":
                        "Your Probability %",
                    "Actual_Hit_Rate":
                        "Actual Hit Rate %",
                    "Difference_pp":
                        "Difference pp"
                }
            )
        )


        for column in [
            "Your Probability %",
            "Actual Hit Rate %",
            "Difference pp"
        ]:

            calibration_table[
                column
            ] = (
                calibration_table[
                    column
                ]
                .round(2)
            )


        st.subheader(
            "Calibration by Probability Range"
        )


        st.dataframe(
            calibration_table,
            use_container_width=True,
            hide_index=True
        )


    # ======================================
    # CUMULATIVE PROFIT
    # ======================================

    if not performance.empty:

        chart_df = (
            performance
            .copy()
            .sort_values(
                [
                    "bet_date",
                    "bet_number"
                ]
            )
        )


        chart_df[
            "Cumulative Profit"
        ] = (
            chart_df["profit"]
            .fillna(0)
            .cumsum()
        )


        chart_df["Bet"] = range(
            1,
            len(chart_df) + 1
        )


        chart_df = (
            chart_df
            .set_index("Bet")
        )


        st.divider()

        st.subheader(
            "📈 Cumulative Profit"
        )


        st.line_chart(
            chart_df[
                ["Cumulative Profit"]
            ]
        )


    # ======================================
    # PERFORMANCE BY MARKET
    # ======================================

    if not performance.empty:

        market_summary = (
            performance
            .groupby(
                "market",
                dropna=False
            )
            .agg(
                Bets=(
                    "id",
                    "count"
                ),
                Stake=(
                    "stake",
                    "sum"
                ),
                Profit=(
                    "profit",
                    "sum"
                )
            )
            .reset_index()
        )


        market_summary[
            "ROI %"
        ] = (
            market_summary["Profit"]
            / market_summary["Stake"]
            * 100
        )


        market_summary = (
            market_summary
            .sort_values(
                "ROI %",
                ascending=False
            )
        )


        for column in [
            "Stake",
            "Profit",
            "ROI %"
        ]:

            market_summary[
                column
            ] = (
                market_summary[
                    column
                ]
                .round(2)
            )


        st.subheader(
            "Performance by Market"
        )


        st.dataframe(
            market_summary,
            use_container_width=True,
            hide_index=True
        )



    # ======================================
    # OUTRIGHT ANALYSIS
    # ======================================

    outright_sample = filtered[
        filtered["scope"].eq("OUTRIGHT")
    ].copy()

    if not outright_sample.empty:

        st.divider()
        st.subheader("🏆 Outright Analysis")

        outright_perf = outright_sample[
            outright_sample["result"].isin([
                "Win",
                "Loss",
                "Cashout"
            ])
        ].copy()

        if not outright_perf.empty:

            outright_summary = (
                outright_perf
                .groupby(
                    "outright_type",
                    dropna=False
                )
                .agg(
                    Bets=("id", "count"),
                    Stake=("stake", "sum"),
                    Profit=("profit", "sum"),
                    Wins=("result", lambda s: (s == "Win").sum()),
                    Losses=("result", lambda s: (s == "Loss").sum())
                )
                .reset_index()
                .rename(
                    columns={
                        "outright_type": "Outright Type"
                    }
                )
            )

            outright_summary["ROI %"] = (
                outright_summary["Profit"]
                / outright_summary["Stake"]
                * 100
            )

            decisions = (
                outright_summary["Wins"]
                + outright_summary["Losses"]
            )

            outright_summary["Win Rate %"] = (
                outright_summary["Wins"]
                / decisions.where(decisions > 0)
                * 100
            )

            for column in [
                "Stake",
                "Profit",
                "ROI %",
                "Win Rate %"
            ]:
                outright_summary[column] = (
                    outright_summary[column]
                    .round(2)
                )

            st.dataframe(
                outright_summary,
                use_container_width=True,
                hide_index=True
            )




    # ======================================
    # BET BUILDER / PARLAY ANALYSIS
    # ======================================

    combo_sample = filtered[
        filtered["bet_format"].eq("Bet Builder")
        | filtered["market"].eq("Parlay")
    ].copy()

    combo_sample["combo_structure"] = combo_sample.apply(
        lambda row: (
            "Outright Parlay"
            if (
                row.get("scope") == "OUTRIGHT"
                and row.get("market") == "Parlay"
            )
            else row.get("bet_format")
        ),
        axis=1
    )

    if not combo_sample.empty:

        st.divider()
        st.subheader("🧩 Bet Builder / Parlay Analysis")

        combo_sample["Selections"] = (
            combo_sample["combo_legs"]
            .apply(_combo_total_selections)
        )
        combo_sample["Components"] = (
            combo_sample["combo_legs"]
            .apply(_combo_component_count)
        )
        combo_sample["Combo Profile"] = (
            combo_sample["combo_legs"]
            .apply(_combo_profile)
        )

        selection_count_options = sorted(
            int(value)
            for value in combo_sample["Selections"].dropna().unique()
            if int(value) > 0
        )

        parlay_component_options = sorted(
            int(value)
            for value in combo_sample.loc[
                combo_sample["market"] == "Parlay",
                "Components"
            ].dropna().unique()
            if int(value) > 0
        )

        parlay_sport_options = sorted(
            str(value)
            for value in combo_sample.loc[
                combo_sample["market"] == "Parlay",
                "sport"
            ].dropna().unique()
            if str(value).strip()
        )

        combo_profile_options = sorted(
            str(value)
            for value in combo_sample["Combo Profile"].dropna().unique()
            if str(value).strip()
        )

        c1, c2 = st.columns(2)

        with c1:
            combo_size_selected = st.multiselect(
                "Number of underlying selections",
                selection_count_options,
                key="analysis_combo_selection_count",
                placeholder="All combo sizes"
            )

        with c2:
            parlay_components_selected = st.multiselect(
                "Parlay legs / components",
                parlay_component_options,
                key="analysis_parlay_component_count",
                placeholder="All parlay leg counts"
            )

        c3, c4 = st.columns(2)

        with c3:
            parlay_sport_selected = st.multiselect(
                "Parlay Sport",
                parlay_sport_options,
                key="analysis_parlay_sport",
                placeholder="All parlay sports"
            )

        with c4:
            combo_profile_selected = st.multiselect(
                "BB / Parlay Type",
                combo_profile_options,
                key="analysis_combo_profile",
                placeholder="Value + Τζόγος"
            )

        combo_view = combo_sample.copy()

        if combo_size_selected:
            combo_view = combo_view[
                combo_view["Selections"].isin(combo_size_selected)
            ]

        if parlay_components_selected:
            combo_view = combo_view[
                (combo_view["market"] != "Parlay")
                | combo_view["Components"].isin(
                    parlay_components_selected
                )
            ]

        if parlay_sport_selected:
            combo_view = combo_view[
                (combo_view["market"] != "Parlay")
                | combo_view["sport"].isin(parlay_sport_selected)
            ]

        if combo_profile_selected:
            combo_view = combo_view[
                combo_view["Combo Profile"].isin(
                    combo_profile_selected
                )
            ]

        if not combo_view.empty:
            combo_perf = combo_view[
                combo_view["result"].isin([
                    "Win",
                    "Loss",
                    "Cashout"
                ])
            ].copy()

            if not combo_perf.empty:
                combo_stake = combo_perf["stake"].fillna(0).sum()
                combo_profit = combo_perf["profit"].fillna(0).sum()
                combo_roi = (
                    combo_profit / combo_stake * 100
                    if combo_stake > 0
                    else 0
                )
                combo_decisions = combo_perf[
                    combo_perf["result"].isin(["Win", "Loss"])
                ]
                combo_win_rate = (
                    combo_decisions["result"].eq("Win").mean() * 100
                    if not combo_decisions.empty
                    else None
                )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Combo Bets", len(combo_perf))
                with c2:
                    st.metric("Stake", f"{combo_stake:.2f}")
                with c3:
                    st.metric("Profit", f"{combo_profit:+.2f}")
                with c4:
                    st.metric("ROI", f"{combo_roi:+.2f}%")

                if combo_win_rate is not None:
                    st.metric("Combo Win Rate", f"{combo_win_rate:.2f}%")

                combo_summary = (
                    combo_perf
                    .groupby(
                        ["combo_structure", "Selections", "Components"],
                        dropna=False
                    )
                    .agg(
                        Bets=("id", "count"),
                        Stake=("stake", "sum"),
                        Profit=("profit", "sum")
                    )
                    .reset_index()
                )
                combo_summary["ROI %"] = (
                    combo_summary["Profit"]
                    / combo_summary["Stake"]
                    * 100
                )
                combo_summary = combo_summary.rename(
                    columns={"combo_structure": "Structure"}
                )
                for col in ["Stake", "Profit", "ROI %"]:
                    combo_summary[col] = combo_summary[col].round(2)

                st.subheader("Performance by combo size")
                st.dataframe(
                    combo_summary.sort_values(
                        ["Structure", "Selections"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            # ----------------------------------
            # FLAT-BET SIMULATION OF SELECTIONS
            # ----------------------------------

            leg_rows = []

            for _, parent in combo_view.iterrows():
                for selection in _combo_flat_selections(
                    parent.get("combo_legs")
                ):
                    result = selection.get("result") or "Pending"
                    odds = pd.to_numeric(
                        selection.get("odds"),
                        errors="coerce"
                    )

                    if (
                        result not in ["Win", "Loss", "Void"]
                        or pd.isna(odds)
                        or float(odds) <= 1
                    ):
                        continue

                    flat_profit = 0.0
                    if result == "Win":
                        flat_profit = float(odds) - 1.0
                    elif result == "Loss":
                        flat_profit = -1.0

                    leg_rows.append({
                        "Parent Structure": parent.get("combo_structure") or parent["bet_format"],
                        "Parent Date": parent["bet_date"],
                        "Selection": selection.get("label") or "",
                        "Odds": float(odds),
                        "Result": result,
                        "Flat Stake": 1.0,
                        "Flat Profit": flat_profit
                    })

            st.subheader("🧮 Flat-bet simulation — individual selections")
            st.caption(
                "Assumes 1 unit on every individually settled selection at "
                "the standalone odds you entered. Pending legs are excluded."
            )

            if leg_rows:
                leg_df = pd.DataFrame(leg_rows)
                flat_stake = leg_df["Flat Stake"].sum()
                flat_profit = leg_df["Flat Profit"].sum()
                flat_roi = (
                    flat_profit / flat_stake * 100
                    if flat_stake > 0
                    else 0
                )
                flat_decisions = leg_df[
                    leg_df["Result"].isin(["Win", "Loss"])
                ]
                flat_hit = (
                    flat_decisions["Result"].eq("Win").mean() * 100
                    if not flat_decisions.empty
                    else None
                )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Settled selections", len(leg_df))
                with c2:
                    st.metric("Flat Stake", f"{flat_stake:.2f}u")
                with c3:
                    st.metric("Flat P/L", f"{flat_profit:+.2f}u")
                with c4:
                    st.metric("Flat ROI", f"{flat_roi:+.2f}%")

                if flat_hit is not None:
                    st.metric("Selection Hit Rate", f"{flat_hit:.2f}%")

                st.dataframe(
                    leg_df.sort_values(
                        "Parent Date",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(
                    "No individual combo selections have been settled yet. "
                    "Set their Win/Loss/Void results from Pending or Manage."
                )



    # ======================================
    # ABUSE / PROMO PERFORMANCE
    # ======================================

    abuse_sample = filtered[
        filtered["market"].eq("Abuse")
    ].copy()

    if not abuse_sample.empty:
        st.divider()
        st.subheader("🧪 Abuse / Promo Performance")
        st.caption(
            "Free-bet / bonus face value is tracked separately and is not "
            "treated as cash profit. Profit only includes the qualifying/casino "
            "cash result plus cash actually realized from the promo."
        )

        abuse_rows = []
        for _, row in abuse_sample.iterrows():
            raw = row.get("abuse_data")
            data = raw
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                except Exception:
                    data = {}
            if not isinstance(data, dict):
                data = {}

            category = data.get("category") or "SPORTS"
            reward_face = pd.to_numeric(
                data.get("reward_face_value"), errors="coerce"
            )
            reward_face = 0.0 if pd.isna(reward_face) else float(reward_face)

            if category == "CASINO":
                qualifying_cost = 0.0
                realized_promo = 0.0
                turnover = pd.to_numeric(
                    data.get("required_turnover"), errors="coerce"
                )
                turnover = 0.0 if pd.isna(turnover) else float(turnover)
                abuse_kind = "Casino"
                mechanic = data.get("reward_type") or "Casino Promo"
            else:
                qualifying_cost = pd.to_numeric(
                    data.get("worst_case_qualifying_pl"), errors="coerce"
                )
                qualifying_cost = 0.0 if pd.isna(qualifying_cost) else float(qualifying_cost)
                realized_promo = pd.to_numeric(
                    data.get("promo_realized_cash"), errors="coerce"
                )
                realized_promo = 0.0 if pd.isna(realized_promo) else float(realized_promo)
                turnover = 0.0
                abuse_kind = data.get("match_format") or "Sports"
                mechanic = data.get("promo_mechanic") or "Promo"

            abuse_rows.append({
                "Date": row.get("bet_date"),
                "Type": abuse_kind,
                "Mechanic": mechanic,
                "Event": row.get("event"),
                "Cash Exposure": float(row.get("stake") or 0),
                "Promo Face Value": reward_face,
                "Promo Cash Realized": realized_promo,
                "Qualifying P/L": qualifying_cost,
                "Casino Turnover": turnover,
                "Profit": float(row.get("profit") or 0),
                "Result": row.get("result")
            })

        abuse_df = pd.DataFrame(abuse_rows)
        abuse_profit = abuse_df["Profit"].sum()
        abuse_exposure = abuse_df["Cash Exposure"].sum()
        abuse_roi = (
            abuse_profit / abuse_exposure * 100
            if abuse_exposure > 0 else 0.0
        )
        face_total = abuse_df["Promo Face Value"].sum()
        realized_total = abuse_df["Promo Cash Realized"].sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Completed abuses", len(abuse_df))
        with c2:
            st.metric("Cash exposure", f"€{abuse_exposure:.2f}")
        with c3:
            st.metric("Net P/L", f"€{abuse_profit:+.2f}")
        with c4:
            st.metric("ROI", f"{abuse_roi:+.2f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Promo face value tracked", f"€{face_total:.2f}")
        with c2:
            st.metric("Promo cash realized", f"€{realized_total:.2f}")

        if not abuse_df.empty:
            summary = (
                abuse_df
                .groupby(["Type", "Mechanic"], dropna=False)
                .agg(
                    Abuses=("Event", "count"),
                    Exposure=("Cash Exposure", "sum"),
                    Profit=("Profit", "sum"),
                    Promo_Face=("Promo Face Value", "sum"),
                    Promo_Cash=("Promo Cash Realized", "sum")
                )
                .reset_index()
            )
            summary["ROI %"] = summary.apply(
                lambda r: (r["Profit"] / r["Exposure"] * 100)
                if r["Exposure"] > 0 else 0.0,
                axis=1
            )
            for col in ["Exposure", "Profit", "Promo_Face", "Promo_Cash", "ROI %"]:
                summary[col] = summary[col].round(2)

            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True
            )

            with st.expander("Abuse details"):
                st.dataframe(
                    abuse_df.sort_values("Date", ascending=False),
                    hide_index=True,
                    use_container_width=True
                )

    # ======================================
    # SHARED PICKS PERFORMANCE
    # ======================================

    shared_perf = filtered[
        filtered["origin"].eq("SHARED")
    ].copy()

    if not shared_perf.empty:

        st.subheader("👥 Shared Picks Performance")
        st.caption(
            "Performance of picks you copied from each user who shared them."
        )

        shared_perf["shared_by"] = (
            shared_perf["shared_by"]
            .fillna("Unknown user")
        )

        shared_summary = (
            shared_perf
            .groupby("shared_by", dropna=False)
            .agg(
                Bets=("result", "count"),
                Stake=("stake", "sum"),
                Profit=("profit", "sum"),
                Wins=("result", lambda x: (x == "Win").sum()),
                Losses=("result", lambda x: (x == "Loss").sum()),
                Cashouts=("result", lambda x: (x == "Cashout").sum()),
                Voids=("result", lambda x: (x == "Void").sum())
            )
            .reset_index()
            .rename(columns={"shared_by": "Shared By"})
        )

        shared_summary["ROI %"] = shared_summary.apply(
            lambda r: (r["Profit"] / r["Stake"] * 100)
            if r["Stake"] > 0 else 0.0,
            axis=1
        )

        shared_summary["Win Rate %"] = shared_summary.apply(
            lambda r: (r["Wins"] / (r["Wins"] + r["Losses"]) * 100)
            if (r["Wins"] + r["Losses"]) > 0 else 0.0,
            axis=1
        )

        for col in ["Stake", "Profit", "ROI %", "Win Rate %"]:
            shared_summary[col] = shared_summary[col].round(2)

        shared_summary = shared_summary.sort_values(
            ["Profit", "Bets"],
            ascending=[False, False]
        )

        st.dataframe(
            shared_summary,
            hide_index=True,
            use_container_width=True
        )


    # ======================================
    # FILTERED BETS
    # ======================================

    st.subheader(
        "Filtered Bets"
    )


    filtered = (
        filtered.copy()
    )


    filtered[
        "selection_display"
    ] = filtered.apply(
        format_selection,
        axis=1
    )

    filtered[
        "bet_timing"
    ] = filtered["is_live"].map({
        True: "Live",
        False: "Pre-live"
    })


    columns_to_show = [
        "bet_date",
        "sport",
        "bet_format",
        "outright_type",
        "bet_timing",
        "league",
        "scope",
        "event",
        "subject",
        "market",
        "selection_display",
        "market_odds",
        "my_odds",
        "origin",
        "shared_by",
        "tipster_name",
        "confidence",
        "primary_reason",
        "result",
        "cashout_return",
        "ev_pct",
        "profit"
    ]


    visible_columns = [
        column
        for column
        in columns_to_show
        if column
        in filtered.columns
    ]


    display_df = (
        filtered[
            visible_columns
        ]
        .sort_values(
            "bet_date",
            ascending=False
        )
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
