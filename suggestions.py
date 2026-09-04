
import pandas as pd
import streamlit as st
from collections import Counter


PAGE_SIZE = 1000


# ==========================================
# LOAD DATA
# ==========================================

def load_suggestion_data(
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


            rows.extend(
                page
            )


            if len(page) < PAGE_SIZE:

                break


            start += PAGE_SIZE


    except Exception as e:

        st.error(
            f"Could not load suggestion data: {e}"
        )

        return pd.DataFrame()


    if not rows:

        return pd.DataFrame()


    df = pd.DataFrame(
        rows
    )

    # Combo bets have their own analysis and should not
    # contaminate single-bet Suggestions patterns.
    if "market" in df.columns:
        df = df[
            ~df["market"].isin([
                "Bet Builder",
                "Parlay",
                "Abuse"
            ])
        ].copy()

    return df


# ==========================================
# CREATE CANDIDATE GROUPS
# ==========================================

def create_candidate_groups(
    df,
    group_columns,
    pattern_name,
    priority,
    min_bets
):

    work = df.copy()


    # --------------------------------------
    # REMOVE EMPTY GROUP VALUES
    # --------------------------------------

    for column in group_columns:

        if column not in work.columns:

            return []

        work = work[
            work[column]
            .notna()
        ]

        work = work[
            work[column]
            .astype(str)
            .str.strip()
            .ne("")
        ]


    if work.empty:

        return []


    candidates = []


    grouped = work.groupby(
        group_columns,
        dropna=False
    )


    for values, group in grouped:

        if len(group) < min_bets:

            continue


        if not isinstance(
            values,
            tuple
        ):

            values = (
                values,
            )


        category_parts = [
            str(value)
            for value in values
        ]


        category = (
            " → ".join(
                category_parts
            )
        )


        member_ids = set(
            group["id"]
            .astype(str)
            .tolist()
        )


        key = (
            pattern_name
            + "||"
            + category
        )


        candidates.append({

            "key":
                key,

            "Pattern":
                pattern_name,

            "Category":
                category,

            "group_columns":
                group_columns,

            "specificity":
                len(
                    group_columns
                ),

            "priority":
                priority,

            "base_count":
                len(
                    group
                ),

            "member_ids":
                member_ids
        })


    return candidates


# ==========================================
# ASSIGN EACH BET TO ONE BUCKET
# ==========================================

def unique_bucket_assignment(
    df,
    candidates,
    min_bets
):

    if not candidates:

        return {}, {}


    candidate_map = {
        candidate["key"]:
            candidate
        for candidate in candidates
    }


    # --------------------------------------
    # ALL POSSIBLE BUCKETS FOR EACH BET
    # --------------------------------------

    choices_by_bet = {}


    for candidate in candidates:

        key = candidate[
            "key"
        ]


        for bet_id in candidate[
            "member_ids"
        ]:

            choices_by_bet.setdefault(
                bet_id,
                []
            ).append(
                key
            )


    # --------------------------------------
    # PRIORITY
    #
    # 1. More specific pattern
    # 2. More useful/actionable definition
    # 3. Larger original sample
    # --------------------------------------

    def candidate_rank(
        key
    ):

        candidate = (
            candidate_map[
                key
            ]
        )


        return (

            -candidate[
                "specificity"
            ],

            candidate[
                "priority"
            ],

            -candidate[
                "base_count"
            ],

            candidate[
                "Category"
            ]
        )


    # Sort possible choices once

    for bet_id in choices_by_bet:

        choices_by_bet[
            bet_id
        ] = sorted(
            choices_by_bet[
                bet_id
            ],
            key=candidate_rank
        )


    active = set(
        candidate_map.keys()
    )


    # --------------------------------------
    # ITERATIVE ASSIGNMENT
    #
    # If a bucket ends up below min_bets
    # after unique assignment, remove it
    # and reassign those bets.
    # --------------------------------------

    while True:

        assignment = {}

        assigned_counts = Counter()


        for bet_id, choices in (
            choices_by_bet.items()
        ):

            valid_choices = [
                key
                for key in choices
                if key in active
            ]


            if not valid_choices:

                continue


            best_key = (
                valid_choices[0]
            )


            assignment[
                bet_id
            ] = best_key


            assigned_counts[
                best_key
            ] += 1


        too_small = {
            key
            for key in active
            if assigned_counts[
                key
            ] < min_bets
        }


        if not too_small:

            break


        active -= too_small


        if not active:

            assignment = {}
            break


    # --------------------------------------
    # GROUP FINAL IDS
    # --------------------------------------

    bucket_ids = {}


    for bet_id, key in (
        assignment.items()
    ):

        bucket_ids.setdefault(
            key,
            set()
        ).add(
            bet_id
        )


    return (
        bucket_ids,
        candidate_map
    )


# ==========================================
# FINAL BUCKET METRICS
# ==========================================

def build_final_results(
    df,
    bucket_ids,
    candidate_map
):

    results = []


    if not bucket_ids:

        return results


    df = df.copy()

    df["_bet_id_string"] = (
        df["id"]
        .astype(str)
    )


    for key, member_ids in (
        bucket_ids.items()
    ):

        candidate = (
            candidate_map[
                key
            ]
        )


        subset = df[
            df[
                "_bet_id_string"
            ]
            .isin(
                member_ids
            )
        ].copy()


        if subset.empty:

            continue


        bets = len(
            subset
        )


        stake = float(
            subset["stake"]
            .fillna(0)
            .sum()
        )


        profit = float(
            subset["profit"]
            .fillna(0)
            .sum()
        )


        roi = (
            profit
            / stake
            * 100
            if stake > 0
            else 0
        )


        wins = int(
            subset["result"]
            .eq("Win")
            .sum()
        )


        losses = int(
            subset["result"]
            .eq("Loss")
            .sum()
        )


        cashouts = int(
            subset["result"]
            .eq("Cashout")
            .sum()
        )


        decisions = (
            wins
            + losses
        )


        win_rate = (
            wins
            / decisions
            * 100
            if decisions > 0
            else None
        )


        avg_odds = (
            subset[
                "market_odds"
            ]
            .mean()
        )


        avg_ev = (
            subset[
                "ev_pct"
            ]
            .mean()
        )


        results.append({

            "Pattern":
                candidate[
                    "Pattern"
                ],

            "Category":
                candidate[
                    "Category"
                ],

            "Bets":
                bets,

            "Stake":
                round(
                    stake,
                    2
                ),

            "Profit":
                round(
                    profit,
                    2
                ),

            "ROI %":
                round(
                    roi,
                    2
                ),

            "Win Rate %":
                (
                    round(
                        win_rate,
                        2
                    )
                    if win_rate
                    is not None
                    else None
                ),

            "Avg Odds":
                (
                    round(
                        float(
                            avg_odds
                        ),
                        2
                    )
                    if pd.notna(
                        avg_odds
                    )
                    else None
                ),

            "Avg EV %":
                (
                    round(
                        float(
                            avg_ev
                        ),
                        2
                    )
                    if pd.notna(
                        avg_ev
                    )
                    else None
                ),

            "Cashouts":
                cashouts,

            "_key":
                key,

            "_bet_ids":
                member_ids
        })


    return results


# ==========================================
# SUGGESTIONS PAGE
# ==========================================

def suggestions_page(
    supabase,
    load_tipsters
):

    st.header(
        "💡 Suggestions"
    )


    st.caption(
        "Each bet is assigned to one "
        "suggestion category only."
    )


    df = load_suggestion_data(
        supabase
    )


    if df.empty:

        st.info(
            "No settled bets yet."
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


    df[
        "tipster_name"
    ] = (
        df["tipster_id"]
        .map(
            tipster_map
        )
    )


    # ======================================
    # NUMERIC
    # ======================================

    numeric_columns = [
        "stake",
        "profit",
        "market_odds",
        "ev_pct"
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
    # REAL MONEY RESULTS
    # ======================================

    df = df[
        df["result"]
        .isin([
            "Win",
            "Loss",
            "Cashout"
        ])
    ].copy()


    if df.empty:

        st.info(
            "No financial results yet."
        )

        return


    # ======================================
    # SOURCE
    # ======================================

    source = st.radio(
        "Source",
        [
            "SELF",
            "TIPSTER",
            "SHARED",
            "ALL"
        ],
        index=0,
        horizontal=True
    )


    filtered = df.copy()


    if source != "ALL":

        filtered = filtered[
            filtered["origin"]
            == source
        ]


    # ======================================
    # SPORT
    # Suggestions are intentionally
    # evaluated inside one sport at a time.
    # ======================================

    sport_options = sorted(
        filtered["sport"]
        .dropna()
        .unique()
        .tolist()
    )


    if not sport_options:

        st.warning(
            "No sports available "
            "for the selected source."
        )

        return


    default_sport_index = (
        sport_options.index(
            "Basketball"
        )
        if "Basketball"
        in sport_options
        else 0
    )


    selected_sport = (
        st.selectbox(
            "Sport",
            sport_options,
            index=default_sport_index
        )
    )


    filtered = filtered[
        filtered["sport"]
        == selected_sport
    ].copy()


    # ======================================
    # SPECIFIC TIPSTER
    # ======================================

    if source == "TIPSTER":

        names = sorted(
            [
                t["name"]
                for t in tipsters
                if t.get("name")
            ]
        )


        specific_tipster = (
            st.selectbox(
                "Specific Tipster",
                ["All"] + names
            )
        )


        if specific_tipster != "All":

            filtered = filtered[
                filtered[
                    "tipster_name"
                ]
                == specific_tipster
            ]


    # ======================================
    # CONTROLS
    # ======================================

    c1, c2 = st.columns(2)


    with c1:

        min_bets = st.slider(
            "Minimum Bets",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )


    with c2:

        ranking = st.selectbox(
            "Rank Categories By",
            [
                "Net Profit",
                "ROI"
            ]
        )


    if filtered.empty:

        st.warning(
            "No bets match "
            "the selected source."
        )

        return


    # ======================================
    # CANDIDATE PATTERNS
    #
    # Lower priority number = preferred
    # when specificity is equal.
    # ======================================

    definitions = [

        (
            [
                "league",
                "primary_reason",
                "market",
                "side"
            ],
            "League → Reason → Market → Side",
            1
        ),

        (
            [
                "primary_reason",
                "market",
                "side"
            ],
            "Reason → Market → Side",
            1
        ),

        (
            [
                "subject",
                "market",
                "side"
            ],
            "Player → Market → Side",
            2
        ),

        (
            [
                "confidence",
                "primary_reason",
                "market"
            ],
            "Confidence → Reason → Market",
            3
        ),

        (
            [
                "league",
                "market",
                "side"
            ],
            "League → Market → Side",
            4
        ),

        (
            [
                "primary_reason",
                "market"
            ],
            "Reason → Market",
            1
        ),

        (
            [
                "subject",
                "market"
            ],
            "Player → Market",
            2
        ),

        (
            [
                "market",
                "side"
            ],
            "Market → Side",
            3
        ),

        (
            [
                "league",
                "market"
            ],
            "League → Market",
            4
        ),

        (
            [
                "confidence",
                "market"
            ],
            "Confidence → Market",
            5
        ),

        (
            [
                "scope",
                "market"
            ],
            "Bet Type → Market",
            6
        ),

        (
            [
                "primary_reason"
            ],
            "Reason",
            1
        ),

        (
            [
                "market"
            ],
            "Market",
            2
        )
    ]


    candidates = []


    for (
        columns,
        pattern,
        priority
    ) in definitions:

        # Player patterns should only
        # use actual PLAYER bets.

        if columns[0] == "subject":

            pattern_df = filtered[
                filtered["scope"]
                == "PLAYER"
            ].copy()

        else:

            pattern_df = (
                filtered
            )


        candidates.extend(
            create_candidate_groups(
                pattern_df,
                columns,
                pattern,
                priority,
                min_bets
            )
        )


    # ======================================
    # UNIQUE ASSIGNMENT
    # ======================================

    (
        bucket_ids,
        candidate_map
    ) = unique_bucket_assignment(
        filtered,
        candidates,
        min_bets
    )


    results = build_final_results(
        filtered,
        bucket_ids,
        candidate_map
    )


    if not results:

        st.info(
            "Not enough unique data "
            "for the current "
            "Minimum Bets setting.\n\n"
            "For testing, try "
            "Minimum Bets = 1."
        )

        return


    result_df = pd.DataFrame(
        results
    )


    # ======================================
    # SAFETY CHECK:
    # NO BET MAY APPEAR TWICE
    # ======================================

    all_assigned_ids = []

    for ids in result_df[
        "_bet_ids"
    ]:

        all_assigned_ids.extend(
            list(ids)
        )


    unique_count = len(
        set(
            all_assigned_ids
        )
    )


    total_assigned = len(
        all_assigned_ids
    )


    if unique_count != total_assigned:

        st.error(
            "Duplicate assignment detected."
        )

        return


    # ======================================
    # SUMMARY
    # ======================================

    classified_bets = (
        unique_count
    )


    total_source_bets = (
        len(filtered)
    )


    st.success(
        "✅ Unique assignment active — "
        "no bet appears in more than "
        "one suggestion."
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Bets in Sample",
            total_source_bets
        )


    with c2:

        st.metric(
            "Bets Classified",
            classified_bets
        )


    with c3:

        st.metric(
            "Unique Categories",
            len(
                result_df
            )
        )


    ranking_column = (
        "Profit"
        if ranking
        == "Net Profit"
        else "ROI %"
    )


    # ======================================
    # WINNING / LOSING BUCKETS
    # ======================================

    winning = result_df[
        result_df["Profit"]
        > 0
    ].copy()


    losing = result_df[
        result_df["Profit"]
        < 0
    ].copy()


    top = (
        winning
        .sort_values(
            [
                ranking_column,
                "Bets"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(10)
        .copy()
    )


    worst = (
        losing
        .sort_values(
            [
                ranking_column,
                "Bets"
            ],
            ascending=[
                True,
                False
            ]
        )
        .head(10)
        .copy()
    )


    display_columns = [
        "Pattern",
        "Category",
        "Bets",
        "Stake",
        "Profit",
        "ROI %",
        "Win Rate %",
        "Avg Odds",
        "Avg EV %",
        "Cashouts"
    ]


    st.divider()


    # ======================================
    # TOP
    # ======================================

    st.subheader(
        f"🔥 Top {len(top)} Categories"
    )


    if top.empty:

        st.info(
            "No profitable categories "
            "meet the current criteria."
        )

    else:

        st.dataframe(
            top[
                display_columns
            ],
            use_container_width=True,
            hide_index=True
        )


    # ======================================
    # WORST
    # ======================================

    st.subheader(
        f"🧊 Worst {len(worst)} Categories"
    )


    if worst.empty:

        st.info(
            "No losing categories "
            "meet the current criteria."
        )

    else:

        st.dataframe(
            worst[
                display_columns
            ],
            use_container_width=True,
            hide_index=True
        )


    st.caption(
        "A bet cannot contribute to "
        "multiple visible categories. "
        "The engine first prefers the "
        "most specific category that "
        "still has enough sample size."
    )


    # ======================================
    # INSPECT SUGGESTION
    # ======================================

    surfaced_frames = []


    if not top.empty:

        surfaced_frames.append(
            top
        )


    if not worst.empty:

        surfaced_frames.append(
            worst
        )


    if not surfaced_frames:

        return


    surfaced = (
        pd.concat(
            surfaced_frames,
            ignore_index=True
        )
    )


    st.divider()

    st.subheader(
        "🔎 Inspect Suggestion"
    )


    label_map = {}


    for _, row in (
        surfaced.iterrows()
    ):

        label = (
            f"{row['Pattern']} | "
            f"{row['Category']} | "
            f"{row['Bets']} bets | "
            f"{row['Profit']:+.2f} | "
            f"{row['ROI %']:+.2f}%"
        )


        label_map[
            label
        ] = row


    selected_label = (
        st.selectbox(
            "Select surfaced category",
            list(
                label_map.keys()
            )
        )
    )


    selected = (
        label_map[
            selected_label
        ]
    )


    selected_ids = (
        selected[
            "_bet_ids"
        ]
    )


    inspect_df = (
        filtered[
            filtered["id"]
            .astype(str)
            .isin(
                selected_ids
            )
        ]
        .copy()
    )


    columns_to_show = [
        "bet_date",
        "sport",
        "league",
        "scope",
        "event",
        "subject",
        "market",
        "side",
        "line",
        "market_odds",
        "my_odds",
        "origin",
        "tipster_name",
        "primary_reason",
        "confidence",
        "result",
        "cashout_return",
        "profit"
    ]


    visible_columns = [
        column
        for column
        in columns_to_show
        if column
        in inspect_df.columns
    ]


    st.write(
        f"**{len(inspect_df)} unique bets "
        f"in this category**"
    )


    st.dataframe(
        inspect_df[
            visible_columns
        ]
        .sort_values(
            "bet_date",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        "Suggestions describe historical "
        "performance, not future "
        "predictions. Keep Minimum Bets "
        "high enough before drawing "
        "strong conclusions."
    )
