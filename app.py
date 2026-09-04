
import os
import copy
import streamlit as st
import pandas as pd

from datetime import date, datetime, timezone
from supabase import create_client

from options import (
    DEFAULT_SPORT,
    SPORTS,
    BOOKMAKERS,
    get_leagues,
    get_scope_options,
    get_default_markets,
    get_periods,
    get_reasons,
    get_market_style,
    get_winner_side_options
)

from analytics import analysis_page
from suggestions import suggestions_page


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Bet Tracker",
    page_icon="🎯",
    layout="centered"
)


# ==========================================
# SUPABASE CONNECTION
# ==========================================

def get_secret(name):

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name)


SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_PUBLISHABLE_KEY")


if not SUPABASE_URL or not SUPABASE_KEY:

    st.error(
        "Supabase credentials are missing."
    )

    st.stop()


supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ==========================================
# SESSION STATE
# ==========================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "delete_confirm_id" not in st.session_state:
    st.session_state.delete_confirm_id = None


# Restore auth after Streamlit rerun
if (
    st.session_state.logged_in
    and st.session_state.access_token
    and st.session_state.refresh_token
):

    try:

        session_response = (
            supabase.auth.set_session(
                st.session_state.access_token,
                st.session_state.refresh_token
            )
        )

        if session_response.session:

            st.session_state.access_token = (
                session_response
                .session
                .access_token
            )

            st.session_state.refresh_token = (
                session_response
                .session
                .refresh_token
            )

    except Exception:

        st.session_state.logged_in = False


# ==========================================
# HELPERS
# ==========================================

def now_utc():

    return (
        datetime
        .now(timezone.utc)
        .isoformat()
    )


def safe_index(
    options,
    value,
    default=0
):

    try:
        return options.index(value)

    except Exception:
        return default


def get_market_options(
    scope,
    sport=DEFAULT_SPORT
):
    return get_default_markets(
        sport,
        scope
    )



def outright_needs_second_selection(
    market,
    sport=DEFAULT_SPORT
):
    if market in [
        "Final Matchup",
        "Straight Forecast"
    ]:
        return True

    if (
        sport == "Basketball"
        and market.startswith("Top ")
        and market.endswith(" - Team")
    ):
        return True

    return False



def outright_selection_labels(
    market,
    sport=DEFAULT_SPORT
):
    if sport == "Tennis":
        if market == "Final Matchup":
            return ("Player 1", "Player 2")
        if market == "Straight Forecast":
            return ("Winner", "Runner-up")
        return ("Player", None)

    if sport == "Football":
        if market == "Final Matchup":
            return ("Team 1", "Team 2")
        if market == "Straight Forecast":
            return ("Winner", "Runner-up")
        if market in ["Top Goalscorer", "Top Assists"]:
            return ("Player", None)
        return ("Team", None)

    if market == "Final Matchup":
        return ("Team 1", "Team 2")

    if market == "Straight Forecast":
        return ("1st Place", "2nd Place")

    if (
        market.startswith("Top ")
        and market.endswith(" - Team")
    ):
        return ("Player", "Team")

    if market.startswith("Top "):
        return ("Player", None)

    return ("Team", None)



def format_bet_selection(
    bet
):

    scope = bet.get("scope")

    market = (
        bet.get("market")
        or ""
    )

    subject = (
        bet.get("subject")
        or ""
    )

    selection_2 = (
        bet.get("selection_2")
        or ""
    )

    side = bet.get("side")
    line = bet.get("line")


    if _combo_is_bet(bet):
        combo_legs = bet.get("combo_legs") or []
        selection_count = _combo_selection_count(combo_legs)
        if bet.get("market") == "Parlay":
            component_count = _combo_component_count(combo_legs)
            return (
                f"{component_count} components | "
                f"{selection_count} selections"
            )
        return f"{selection_count} selections"


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
            and market.endswith(
                " - Team"
            )
        ):

            if selection_2:

                return (
                    f"{subject} "
                    f"({selection_2})"
                )

            return subject


        return subject


    text_value = side or ""


    if line is not None:

        text_value = (
            f"{text_value} "
            f"{float(line):g}"
        ).strip()


    return text_value


def calculate_metrics(
    market_odds,
    my_odds=None,
    tipster_posted_odds=None
):

    p_market = (
        1 / float(market_odds)
    )

    p_you = None
    edge_pp = None
    ev_pct = None

    if my_odds:

        p_you = (
            1 / float(my_odds)
        )

        edge_pp = (
            p_you - p_market
        ) * 100

        ev_pct = (
            p_you
            * float(market_odds)
            - 1
        ) * 100


    price_deterioration_pp = None

    if tipster_posted_odds:

        posted_probability = (
            1
            / float(
                tipster_posted_odds
            )
        )

        price_deterioration_pp = (
            p_market
            - posted_probability
        ) * 100


    return {

        "p_market":
            p_market,

        "p_you":
            p_you,

        "edge_pp":
            edge_pp,

        "ev_pct":
            ev_pct,

        "price_deterioration_pp":
            price_deterioration_pp
    }


def calculate_profit(
    result,
    stake,
    market_odds
):

    stake = float(stake)
    odds = float(market_odds)

    if result == "Win":

        return round(
            stake * (odds - 1),
            2
        )

    if result == "Loss":

        return round(
            -stake,
            2
        )

    return 0.0


# ==========================================
# LOGIN
# ==========================================

def login_page():

    st.title(
        "🎯 Bet Tracker"
    )

    st.caption(
        "Sign in to your personal tracker"
    )


    with st.form(
        "login_form"
    ):

        email = st.text_input(
            "Email",
            autocomplete="username"
        )

        password = st.text_input(
            "Password",
            type="password",
            autocomplete="current-password"
        )

        submitted = (
            st.form_submit_button(
                "Login",
                use_container_width=True
            )
        )


    st.caption(
        "💾 Your browser can offer to save the login "
        "for this device."
    )


    if submitted:

        if not email or not password:

            st.warning(
                "Enter email and password."
            )

            return


        try:

            response = (
                supabase
                .auth
                .sign_in_with_password({
                    "email":
                        email,

                    "password":
                        password
                })
            )


            if (
                response.session
                and response.user
            ):

                st.session_state.logged_in = (
                    True
                )

                st.session_state.access_token = (
                    response
                    .session
                    .access_token
                )

                st.session_state.refresh_token = (
                    response
                    .session
                    .refresh_token
                )

                st.session_state.user_id = (
                    response.user.id
                )

                st.session_state.user_email = (
                    response.user.email
                )

                st.rerun()


        except Exception as e:

            st.error(
                f"Login failed: {e}"
            )


def logout():

    try:

        supabase.auth.sign_out()

    except Exception:
        pass

    st.session_state.clear()

    st.rerun()


# ==========================================
# TIPSTERS
# ==========================================

def load_tipsters():

    try:

        response = (
            supabase
            .table("tipsters")
            .select("id,name")
            .order("name")
            .execute()
        )

        return response.data or []

    except Exception:

        return []


def create_tipster(name):

    name = name.strip()

    if not name:
        return None


    response = (
        supabase
        .table("tipsters")
        .insert({

            "user_id":
                st.session_state.user_id,

            "name":
                name
        })
        .execute()
    )


    if response.data:

        return response.data[0]

    return None


# ==========================================
# COUNTERS
# ==========================================

def get_total_bets_count():

    response = (
        supabase
        .table("bets")
        .select(
            "id",
            count="exact"
        )
        .eq(
            "is_deleted",
            False
        )
        .eq(
            "needs_review",
            False
        )
        .execute()
    )

    return response.count or 0


def get_pending_bets_count():

    response = (
        supabase
        .table("bets")
        .select(
            "id",
            count="exact"
        )
        .eq(
            "is_deleted",
            False
        )
        .eq(
            "needs_review",
            False
        )
        .eq(
            "result",
            "Pending"
        )
        .execute()
    )

    return response.count or 0


def get_settled_bets_count():

    response = (
        supabase
        .table("bets")
        .select(
            "id",
            count="exact"
        )
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
        .execute()
    )

    return response.count or 0


# ==========================================
# LOAD BETS
# ==========================================

def load_pending_bets():

    try:

        response = (
            supabase
            .table("bets")
            .select("*")
            .eq(
                "is_deleted",
                False
            )
            .eq(
                "result",
                "Pending"
            )
            .order(
                "bet_date",
                desc=False
            )
            .order(
                "bet_number",
                desc=True
            )
            .execute()
        )

        return response.data or []


    except Exception as e:

        st.error(
            f"Could not load pending bets: {e}"
        )

        return []


def load_history_bets():

    try:

        response = (
            supabase
            .table("bets")
            .select("*")
            .eq(
                "is_deleted",
                False
            )
            .neq(
                "result",
                "Pending"
            )
            .order(
                "bet_date",
                desc=True
            )
            .order(
                "bet_number",
                desc=True
            )
            .execute()
        )

        return response.data or []


    except Exception as e:

        st.error(
            f"Could not load history: {e}"
        )

        return []


def load_active_bets():

    try:

        response = (
            supabase
            .table("bets")
            .select("*")
            .eq(
                "is_deleted",
                False
            )
            .order(
                "bet_date",
                desc=True
            )
            .order(
                "bet_number",
                desc=True
            )
            .execute()
        )

        return response.data or []


    except Exception as e:

        st.error(
            f"Could not load bets: {e}"
        )

        return []


def load_deleted_bets():

    try:

        response = (
            supabase
            .table("bets")
            .select("*")
            .eq(
                "is_deleted",
                True
            )
            .order(
                "deleted_at",
                desc=True
            )
            .execute()
        )

        return response.data or []


    except Exception as e:

        st.error(
            f"Could not load Trash: {e}"
        )

        return []


# ==========================================
# SHARED PICKS
# ==========================================

SHARED_PICK_SNAPSHOT_FIELDS = [
    "bet_date",
    "sport",
    "league",
    "is_live",
    "event",
    "scope",
    "subject",
    "selection_2",
    "market",
    "side",
    "line",
    "period",
    "bookmaker",
    "market_odds",
    "my_odds",
    "confidence",
    "primary_reason",
    "secondary_reason",
    "combo_legs"
]


def _shared_display_name(email):
    email = (email or "").strip()
    if not email:
        return "Another user"
    return email.split("@", 1)[0]


def _reset_combo_results_for_copy(combo_legs):
    copied = copy.deepcopy(combo_legs or [])

    for component in copied:
        if not isinstance(component, dict):
            continue

        if component.get("kind") in ["SINGLE", "OUTRIGHT"]:
            component["result"] = "Pending"

        elif component.get("kind") == "BET_BUILDER":
            for selection in component.get("selections", []) or []:
                if isinstance(selection, dict):
                    selection["result"] = "Pending"

    return copied


def build_shared_pick_snapshot(bet, tipster_name=None):
    snapshot = {
        field: copy.deepcopy(bet.get(field))
        for field in SHARED_PICK_SNAPSHOT_FIELDS
    }

    # Personal bankroll/result fields are deliberately not shared.
    snapshot["source_origin"] = bet.get("origin")
    snapshot["source_tipster_name"] = tipster_name
    snapshot["source_bookmaker"] = bet.get("bookmaker")
    snapshot["source_market_odds"] = bet.get("market_odds")

    if snapshot.get("combo_legs"):
        snapshot["combo_legs"] = _reset_combo_results_for_copy(
            snapshot.get("combo_legs")
        )

    return snapshot


def load_my_active_shared_source_ids():
    try:
        response = (
            supabase
            .table("shared_picks")
            .select("source_bet_id")
            .eq("owner_user_id", st.session_state.user_id)
            .eq("is_active", True)
            .execute()
        )

        return {
            str(row.get("source_bet_id"))
            for row in (response.data or [])
            if row.get("source_bet_id") is not None
        }

    except Exception:
        # This keeps the rest of Pending usable if the migration
        # has not been run yet.
        return set()


def share_pending_bet(bet, tipster_name=None):
    if _is_abuse_bet(bet):
        raise ValueError("Abuse entries are personal promo records and cannot be shared.")

    source_bet_id = str(bet.get("id"))
    snapshot = build_shared_pick_snapshot(
        bet,
        tipster_name=tipster_name
    )

    existing = (
        supabase
        .table("shared_picks")
        .select("id")
        .eq("owner_user_id", st.session_state.user_id)
        .eq("source_bet_id", source_bet_id)
        .limit(1)
        .execute()
    )

    payload = {
        "owner_user_id": st.session_state.user_id,
        "owner_email": st.session_state.user_email,
        "source_bet_id": source_bet_id,
        "bet_snapshot": snapshot,
        "is_active": True,
        "updated_at": now_utc()
    }

    if existing.data:
        response = (
            supabase
            .table("shared_picks")
            .update(payload)
            .eq("id", existing.data[0]["id"])
            .eq("owner_user_id", st.session_state.user_id)
            .execute()
        )
    else:
        payload["created_at"] = now_utc()
        response = (
            supabase
            .table("shared_picks")
            .insert(payload)
            .execute()
        )

    return response.data or []


def unshare_pending_bet(bet_id):
    try:
        response = (
            supabase
            .table("shared_picks")
            .update({
                "is_active": False,
                "updated_at": now_utc()
            })
            .eq("owner_user_id", st.session_state.user_id)
            .eq("source_bet_id", str(bet_id))
            .execute()
        )
        return response.data or []
    except Exception:
        return []


def deactivate_shared_pick_for_source(bet_id):
    # Settling/deleting the owner's original bet removes it from the
    # group feed, but copies already taken by other users remain theirs.
    try:
        (
            supabase
            .table("shared_picks")
            .update({
                "is_active": False,
                "updated_at": now_utc()
            })
            .eq("owner_user_id", st.session_state.user_id)
            .eq("source_bet_id", str(bet_id))
            .execute()
        )
    except Exception:
        pass


def load_shared_picks():
    try:
        response = (
            supabase
            .table("shared_picks")
            .select("*")
            .eq("is_active", True)
            .order("created_at", desc=True)
            .limit(300)
            .execute()
        )
        return response.data or []
    except Exception as e:
        st.error(
            "Could not load Shared Picks. If this is the first deploy, "
            f"run the supplied Supabase migration first. Details: {e}"
        )
        return []


def load_my_copied_shared_pick_ids():
    try:
        response = (
            supabase
            .table("bets")
            .select("shared_pick_id")
            .eq("is_deleted", False)
            .execute()
        )

        return {
            str(row.get("shared_pick_id"))
            for row in (response.data or [])
            if row.get("shared_pick_id")
        }
    except Exception:
        return set()


def add_shared_pick_to_my_pending(
    shared_pick,
    bookmaker,
    market_odds,
    stake,
    confidence
):
    snapshot = copy.deepcopy(
        shared_pick.get("bet_snapshot") or {}
    )

    if snapshot.get("market") == "Abuse":
        raise ValueError("Abuse entries cannot be copied as shared picks.")

    combo_legs = _reset_combo_results_for_copy(
        snapshot.get("combo_legs") or []
    )

    metrics = calculate_metrics(
        float(market_odds),
        None,
        None
    )

    shared_by_email = shared_pick.get("owner_email")
    # Do not copy the sender's private notes into another user's tracker.
    notes = f"Shared by {shared_by_email or 'another user'}"

    record = {
        "user_id": st.session_state.user_id,
        "bet_date": snapshot.get("bet_date") or date.today().isoformat(),
        "sport": snapshot.get("sport") or DEFAULT_SPORT,
        "league": snapshot.get("league") or "Other",
        "is_live": bool(snapshot.get("is_live", False)),
        "event": snapshot.get("event") or "Shared Pick",
        "scope": snapshot.get("scope") or "MATCH",
        "subject": snapshot.get("subject"),
        "selection_2": snapshot.get("selection_2"),
        "market": snapshot.get("market") or "Shared Pick",
        "side": snapshot.get("side"),
        "line": snapshot.get("line"),
        "period": snapshot.get("period") or "Full Game",
        "bookmaker": bookmaker,
        "market_odds": float(market_odds),
        "my_odds": None,
        "origin": "SHARED",
        "tipster_id": None,
        "tipster_posted_odds": None,
        "confidence": confidence,
        "has_own_reasoning": False,
        "primary_reason": None,
        "secondary_reason": None,
        "stake": float(stake),
        "p_market": metrics["p_market"],
        "p_you": None,
        "edge_pp": None,
        "ev_pct": None,
        "price_deterioration_pp": None,
        "result": "Pending",
        "profit": 0,
        "notes": notes,
        "is_deleted": False,
        "needs_review": False,
        "combo_legs": combo_legs or None,
        "shared_pick_id": shared_pick.get("id"),
        "shared_from_user_id": shared_pick.get("owner_user_id"),
        "shared_from_email": shared_by_email
    }

    response = (
        supabase
        .table("bets")
        .insert(record)
        .execute()
    )

    return response.data or []


# ==========================================
# SETTLE
# ==========================================

def settle_bet(
    bet_id,
    result,
    stake,
    market_odds
):

    profit = calculate_profit(
        result,
        stake,
        market_odds
    )


    response = (
        supabase
        .table("bets")
        .update({

            "result":
                result,

            "profit":
                profit,

            "settled_at":
                now_utc(),

            "updated_at":
                now_utc()
        })
        .eq(
            "id",
            bet_id
        )
        .eq(
            "user_id",
            st.session_state.user_id
        )
        .execute()
    )

    deactivate_shared_pick_for_source(bet_id)

    return response.data



# ==========================================
# CASHOUT
# ==========================================

def settle_cashout(
    bet_id,
    stake,
    cashout_return
):

    stake = float(stake)
    cashout_return = float(
        cashout_return
    )

    profit = round(
        cashout_return - stake,
        2
    )

    timestamp = now_utc()

    response = (
        supabase
        .table("bets")
        .update({

            "result":
                "Cashout",

            "cashout_return":
                cashout_return,

            "profit":
                profit,

            "cashout_at":
                timestamp,

            "settled_at":
                timestamp,

            "updated_at":
                timestamp
        })
        .eq(
            "id",
            bet_id
        )
        .eq(
            "user_id",
            st.session_state.user_id
        )
        .execute()
    )

    deactivate_shared_pick_for_source(bet_id)

    return response.data


# ==========================================
# SOFT DELETE / RESTORE
# ==========================================

def soft_delete_bet(bet_id):

    response = (
        supabase
        .table("bets")
        .update({

            "is_deleted":
                True,

            "deleted_at":
                now_utc(),

            "updated_at":
                now_utc()
        })
        .eq(
            "id",
            bet_id
        )
        .eq(
            "user_id",
            st.session_state.user_id
        )
        .execute()
    )

    deactivate_shared_pick_for_source(bet_id)

    return response.data


def restore_bet(bet_id):

    response = (
        supabase
        .table("bets")
        .update({

            "is_deleted":
                False,

            "deleted_at":
                None,

            "updated_at":
                now_utc()
        })
        .eq(
            "id",
            bet_id
        )
        .eq(
            "user_id",
            st.session_state.user_id
        )
        .execute()
    )

    return response.data



# ==========================================
# ENTRY AUTOCOMPLETE
# ==========================================

def load_entry_suggestions(
    sport
):
    rows = []
    page_size = 1000
    start = 0

    try:
        while True:
            response = (
                supabase
                .table("bets")
                .select(
                    "event,scope,subject,"
                    "selection_2,market,sport"
                )
                .eq("is_deleted", False)
                .eq("needs_review", False)
                .eq("sport", sport)
                .range(
                    start,
                    start + page_size - 1
                )
                .execute()
            )

            page = response.data or []
            rows.extend(page)

            if len(page) < page_size:
                break

            start += page_size

    except Exception:
        rows = []

    regular_events = []
    outright_events = []
    players = []
    teams = []

    for bet in rows:
        scope = bet.get("scope")
        event = (bet.get("event") or "").strip()
        subject = (bet.get("subject") or "").strip()
        selection_2 = (bet.get("selection_2") or "").strip()
        market = bet.get("market") or ""

        if event:
            if scope == "OUTRIGHT":
                outright_events.append(event)
            else:
                regular_events.append(event)

        if scope == "PLAYER":
            if subject:
                players.append(subject)

        elif scope == "TEAM":
            if subject:
                teams.append(subject)

        elif scope == "OUTRIGHT":
            if sport == "Tennis":
                if subject:
                    players.append(subject)
                if selection_2:
                    players.append(selection_2)

            elif (
                sport == "Football"
                and market in ["Top Goalscorer", "Top Assists"]
            ):
                if subject:
                    players.append(subject)

            elif (
                sport == "Basketball"
                and market.startswith("Top ")
            ):
                if subject:
                    players.append(subject)
                if market.endswith(" - Team") and selection_2:
                    teams.append(selection_2)

            elif market in ["Final Matchup", "Straight Forecast"]:
                if subject:
                    teams.append(subject)
                if selection_2:
                    teams.append(selection_2)

            else:
                if subject:
                    teams.append(subject)

    def clean(values):
        unique = {}
        for value in values:
            value = str(value).strip()
            if not value:
                continue
            key = value.casefold()
            if key not in unique:
                unique[key] = value
        return sorted(
            unique.values(),
            key=lambda x: x.casefold()
        )

    return {
        "regular_events": clean(regular_events),
        "outright_events": clean(outright_events),
        "players": clean(players),
        "teams": clean(teams)
    }



# ==========================================
# ADD BET
# ==========================================


# ==========================================
# STICKY ENTRY / CUSTOM OPTIONS
# ==========================================

def _remember_entry_value(
    bucket,
    value,
    sport=None,
    scope=None
):
    if value is None:
        return

    value = str(value).strip()

    if not value:
        return

    memory_key = "::".join([
        str(sport or "ALL"),
        str(scope or "ALL"),
        str(bucket)
    ])

    if "_recent_entry_suggestions" not in st.session_state:
        st.session_state["_recent_entry_suggestions"] = {}

    recent = st.session_state["_recent_entry_suggestions"]
    values = recent.get(memory_key, [])
    existing = {str(v).casefold() for v in values}

    if value.casefold() not in existing:
        values.append(value)

    recent[memory_key] = values



def _merge_recent_entry_options(
    bucket,
    values,
    sport=None,
    scope=None
):
    values = list(values or [])

    memory_key = "::".join([
        str(sport or "ALL"),
        str(scope or "ALL"),
        str(bucket)
    ])

    recent = (
        st.session_state
        .get("_recent_entry_suggestions", {})
        .get(memory_key, [])
    )

    combined = []
    seen = set()

    for value in values + recent:
        if value is None:
            continue

        value = str(value).strip()

        if not value:
            continue

        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)
        combined.append(value)

    return combined



def load_user_league_options(
    sport
):
    saved_leagues = []
    page_size = 1000
    start = 0

    try:
        while True:
            response = (
                supabase
                .table("bets")
                .select("league")
                .eq("is_deleted", False)
                .eq("needs_review", False)
                .eq("sport", sport)
                .range(
                    start,
                    start + page_size - 1
                )
                .execute()
            )

            page = response.data or []

            for row in page:
                league = (row.get("league") or "").strip()
                if league:
                    saved_leagues.append(league)

            if len(page) < page_size:
                break

            start += page_size

    except Exception:
        pass

    return _merge_recent_entry_options(
        "leagues",
        get_leagues(sport) + saved_leagues,
        sport=sport
    )


def load_user_market_options(
    sport,
    scope
):
    saved_markets = []
    page_size = 1000
    start = 0

    try:
        while True:
            response = (
                supabase
                .table("bets")
                .select("market")
                .eq("is_deleted", False)
                .eq("needs_review", False)
                .eq("sport", sport)
                .eq("scope", scope)
                .range(
                    start,
                    start + page_size - 1
                )
                .execute()
            )

            page = response.data or []

            for row in page:
                market = (row.get("market") or "").strip()
                if market:
                    saved_markets.append(market)

            if len(page) < page_size:
                break

            start += page_size

    except Exception:
        pass

    return _merge_recent_entry_options(
        "markets",
        get_default_markets(
            sport,
            scope
        ) + saved_markets,
        sport=sport,
        scope=scope
    )


def _include_session_option(
    options,
    key
):
    options = list(options or [])
    current = st.session_state.get(key)

    if current is None:
        return options

    current = str(current).strip()

    if not current:
        return options

    existing = {
        str(value).casefold()
        for value in options
    }

    if current.casefold() not in existing:
        options.append(current)

    return options


def infer_saved_custom_market_format(
    sport,
    scope,
    market
):
    try:
        response = (
            supabase
            .table("bets")
            .select("side,line")
            .eq("is_deleted", False)
            .eq("sport", sport)
            .eq("scope", scope)
            .eq("market", market)
            .limit(1)
            .execute()
        )

        rows = response.data or []

        if not rows:
            return "Over / Under"

        row = rows[0]
        side = row.get("side") or ""
        line = row.get("line")

        if side in ["Yes", "No"]:
            return "Yes / No"

        winner_sides = get_winner_side_options(
            sport,
            market
        )

        if side in winner_sides:
            if line is None:
                return "Winner / Selection"
            return "Handicap / Spread"

    except Exception:
        pass

    return "Over / Under"





# ==========================================
# COMBO BETS (BET BUILDER / PARLAY)
# ==========================================

COMBO_MARKETS = ["Bet Builder", "Parlay"]


def _combo_is_bet(bet):
    return (
        (bet.get("market") in COMBO_MARKETS)
        and bool(bet.get("combo_legs"))
    )


def _combo_has_outright_leg(combo_legs):
    return any(
        isinstance(component, dict)
        and str(component.get("kind") or "").upper() == "OUTRIGHT"
        for component in (combo_legs or [])
    )


def _is_outright_parlay(bet):
    return (
        bet.get("market") == "Parlay"
        and _combo_is_bet(bet)
        and _combo_has_outright_leg(
            bet.get("combo_legs") or []
        )
    )


def _combo_flat_selections(combo_legs):
    """Return every underlying selection from a BB/parlay payload."""
    selections = []

    for component in (combo_legs or []):
        if not isinstance(component, dict):
            continue

        kind = component.get("kind")

        if kind in ["SINGLE", "OUTRIGHT"]:
            selections.append(component)
            continue

        if kind == "BET_BUILDER":
            for selection in component.get("selections", []) or []:
                if isinstance(selection, dict):
                    selections.append(selection)

    return selections


def _combo_selection_count(combo_legs):
    return len(_combo_flat_selections(combo_legs))


def _combo_component_count(combo_legs):
    return len(combo_legs or [])


def _combo_mark_all_results(combo_legs, result):
    """Return a copy with every underlying combo selection set to result."""
    updated = copy.deepcopy(combo_legs or [])

    for component in updated:
        if not isinstance(component, dict):
            continue

        if component.get("kind") in ["SINGLE", "OUTRIGHT"]:
            component["result"] = result
        elif component.get("kind") == "BET_BUILDER":
            for selection in component.get("selections", []) or []:
                if isinstance(selection, dict):
                    selection["result"] = result

    return updated


def _combo_has_pending_results(combo_legs):
    for selection in _combo_flat_selections(combo_legs):
        if (selection.get("result") or "Pending") == "Pending":
            return True
    return False


def _parlay_component_odds(component):
    """Odds contribution of one parlay component.

    Singles/outrights use their standalone odds. A nested BB uses the BB
    combined price entered by the user. Older rows fall back to multiplying
    the stored standalone selection odds so they remain readable.
    """
    if not isinstance(component, dict):
        return 1.0

    kind = component.get("kind")

    if kind in ["SINGLE", "OUTRIGHT"]:
        return float(component.get("odds") or 1.0)

    if kind == "BET_BUILDER":
        if component.get("component_odds") is not None:
            return float(component.get("component_odds") or 1.0)

        value = 1.0
        found = False
        for selection in component.get("selections", []) or []:
            odds = float(selection.get("odds") or 1.0)
            value *= odds
            found = True
        return value if found else 1.0

    return 1.0


def _calculate_parlay_odds(combo_legs):
    value = 1.0
    found = False

    for component in combo_legs or []:
        component_odds = _parlay_component_odds(component)
        if component_odds > 1.0:
            value *= component_odds
            found = True

    return round(value if found else 1.01, 2)


def _combo_bb_sizes(combo_legs):
    sizes = []
    for component in (combo_legs or []):
        if (
            isinstance(component, dict)
            and component.get("kind") == "BET_BUILDER"
        ):
            sizes.append(
                len(component.get("selections", []) or [])
            )
    return sizes


def _combo_profile(combo_legs):
    """Return Value / Τζόγος for either a Bet Builder or Parlay.

    Older parlays stored this as parlay_profile, so keep backward compatibility.
    """
    for component in (combo_legs or []):
        if isinstance(component, dict):
            value = (
                component.get("combo_profile")
                or component.get("parlay_profile")
            )
            if value:
                value = str(value)
                if value == "Combo Values":
                    return "Value"
                return value
    return None


def _combo_parlay_profile(combo_legs):
    # Backwards-compatible alias used by older UI code.
    return _combo_profile(combo_legs)


def _combo_parlay_sport(combo_legs):
    for component in (combo_legs or []):
        if isinstance(component, dict):
            value = component.get("parlay_sport")
            if value:
                return str(value)
    return None


def update_combo_legs(bet_id, combo_legs):
    response = (
        supabase
        .table("bets")
        .update({
            "combo_legs": combo_legs,
            "updated_at": now_utc()
        })
        .eq("id", bet_id)
        .eq("user_id", st.session_state.user_id)
        .execute()
    )
    return response.data


def _render_combo_origin_fields(reason_sport):
    origin = st.radio(
        "Origin",
        ["SELF", "TIPSTER"],
        horizontal=True,
        key="combo_origin"
    )

    my_odds = None
    tipster_id = None
    tipster_posted_odds = None
    has_own_reasoning = False
    primary_reason = None
    secondary_reason = None
    confidence = None

    reasons = (
        get_reasons(reason_sport)
        if reason_sport in SPORTS
        else [
            "Projection Edge",
            "Price / Odds",
            "Correlation",
            "Matchup",
            "Injury / Availability",
            "Other"
        ]
    )

    if "Correlation" not in reasons:
        reasons = list(reasons) + ["Correlation"]

    if origin == "SELF":
        my_odds = st.number_input(
            "My Fair Odds (final bet)",
            min_value=1.01,
            value=1.80,
            step=0.01,
            format="%.2f",
            key="combo_my_odds"
        )

        confidence = st.radio(
            "Confidence",
            ["Low", "Medium", "High"],
            index=1,
            horizontal=True,
            key="combo_confidence"
        )

        reason_options = ["Select reason..."] + reasons
        primary_reason = st.selectbox(
            "Primary Reason",
            reason_options,
            index=(
                reason_options.index("Projection Edge")
                if "Projection Edge" in reason_options
                else 0
            ),
            key="combo_primary_reason"
        )

        secondary_options = ["None"] + [
            reason for reason in reasons
            if reason != primary_reason
        ]
        secondary_reason = st.selectbox(
            "Secondary Reason",
            secondary_options,
            key="combo_secondary_reason"
        )
        has_own_reasoning = True

    else:
        tipsters = load_tipsters()
        tipster_map = {
            tipster["name"]: tipster["id"]
            for tipster in tipsters
        }
        tipster_options = ["+ Add new tipster"] + list(tipster_map.keys())

        tipster_choice = st.selectbox(
            "Tipster",
            tipster_options,
            key="combo_tipster_choice"
        )

        if tipster_choice == "+ Add new tipster":
            new_tipster = st.text_input(
                "New Tipster Name",
                key="combo_new_tipster"
            )
            if st.button(
                "Save Tipster",
                key="combo_save_tipster"
            ):
                try:
                    record = create_tipster(new_tipster)
                    if record:
                        st.success("Tipster saved.")
                        st.rerun()
                except Exception as e:
                    st.error(str(e))
        else:
            tipster_id = tipster_map[tipster_choice]

        add_posted_odds = st.checkbox(
            "I know the tipster's posted odds",
            key="combo_has_tipster_odds"
        )

        if add_posted_odds:
            tipster_posted_odds = st.number_input(
                "Tipster Posted Odds",
                min_value=1.01,
                value=1.90,
                step=0.01,
                format="%.2f",
                key="combo_tipster_odds"
            )

        confidence = st.radio(
            "Your Confidence",
            ["N/A", "Low", "Medium", "High"],
            index=2,
            horizontal=True,
            key="combo_tipster_confidence"
        )

        has_own_reasoning = st.checkbox(
            "I also have my own reasoning for this bet",
            key="combo_own_reasoning"
        )

        if has_own_reasoning:
            reason_options = ["Select reason..."] + reasons
            primary_reason = st.selectbox(
                "Primary Reason",
                reason_options,
                index=(
                    reason_options.index("Projection Edge")
                    if "Projection Edge" in reason_options
                    else 0
                ),
                key="combo_tipster_primary_reason"
            )

            secondary_options = ["None"] + [
                reason for reason in reasons
                if reason != primary_reason
            ]
            secondary_reason = st.selectbox(
                "Secondary Reason",
                secondary_options,
                key="combo_tipster_secondary_reason"
            )

    return {
        "origin": origin,
        "my_odds": my_odds,
        "tipster_id": tipster_id,
        "tipster_posted_odds": tipster_posted_odds,
        "has_own_reasoning": has_own_reasoning,
        "primary_reason": primary_reason,
        "secondary_reason": secondary_reason,
        "confidence": confidence
    }



# ==========================================
# ABUSE / PROMO TRACKING
# ==========================================

ABUSE_MARKET = "Abuse"


def _is_abuse_bet(bet):
    return (bet or {}).get("market") == ABUSE_MARKET


def _abuse_result_from_profit(profit):
    profit = round(float(profit or 0), 2)
    if profit > 0.004:
        return "Win"
    if profit < -0.004:
        return "Loss"
    return "Void"


def _abuse_equalized_outcomes(outcomes, qualifying_index, qualifying_stake):
    """Calculate hedge stakes around one fixed qualifying/promo stake.

    All outcomes target the same gross return. Stakes are rounded to cents,
    so the outcome P/L can differ by a few cents in practice.
    """
    outcomes = copy.deepcopy(outcomes or [])
    if not outcomes:
        return outcomes, 0.0, []

    qualifying_index = max(0, min(int(qualifying_index), len(outcomes) - 1))
    qualifying_stake = round(float(qualifying_stake or 0), 2)
    qualifying_odds = float(outcomes[qualifying_index].get("odds") or 0)

    if qualifying_stake <= 0 or qualifying_odds <= 1:
        return outcomes, 0.0, []

    target_return = qualifying_stake * qualifying_odds

    for idx, outcome in enumerate(outcomes):
        odds = float(outcome.get("odds") or 0)
        if odds <= 1:
            outcome["stake"] = 0.0
            continue
        if idx == qualifying_index:
            stake = qualifying_stake
        else:
            stake = target_return / odds
        outcome["stake"] = round(stake, 2)

    total_outlay = round(sum(float(x.get("stake") or 0) for x in outcomes), 2)
    outcome_pls = []
    for outcome in outcomes:
        odds = float(outcome.get("odds") or 0)
        stake = float(outcome.get("stake") or 0)
        pnl = round(stake * odds - total_outlay, 2) if odds > 1 else -total_outlay
        outcome["qualifying_pl"] = pnl
        outcome_pls.append(pnl)

    return outcomes, total_outlay, outcome_pls


def _abuse_sports_profit(data):
    outcomes = data.get("outcomes") or []
    winning_index = data.get("winning_outcome_index")
    base_pl = None

    if winning_index is not None:
        try:
            winning_index = int(winning_index)
            if 0 <= winning_index < len(outcomes):
                base_pl = float(outcomes[winning_index].get("qualifying_pl") or 0)
        except Exception:
            base_pl = None

    if base_pl is None:
        # Conservative fallback while the winner has not been selected.
        pls = [float(x.get("qualifying_pl") or 0) for x in outcomes]
        base_pl = min(pls) if pls else 0.0

    promo_cash = float(data.get("promo_realized_cash") or 0)
    return round(base_pl + promo_cash, 2)


def _abuse_casino_profit(data):
    cash_in = float(data.get("cash_in") or 0)
    final_cash_out = float(data.get("final_cash_out") or 0)
    return round(final_cash_out - cash_in, 2)


def _abuse_profit(data):
    if (data or {}).get("category") == "CASINO":
        return _abuse_casino_profit(data or {})
    return _abuse_sports_profit(data or {})


def _save_abuse_progress(bet_id, abuse_data, complete=False, event=None, notes=None):
    abuse_data = copy.deepcopy(abuse_data or {})
    abuse_data["completed"] = bool(complete)

    payload = {
        "abuse_data": abuse_data,
        "updated_at": now_utc()
    }

    if event is not None:
        payload["event"] = str(event).strip() or "Abuse"
    if notes is not None:
        payload["notes"] = str(notes).strip() or None

    if abuse_data.get("category") == "CASINO":
        payload["stake"] = float(abuse_data.get("cash_in") or 0)
        if abuse_data.get("operator"):
            payload["bookmaker"] = abuse_data.get("operator")
    else:
        payload["stake"] = float(abuse_data.get("total_cash_outlay") or 0)
        promo_idx = int(abuse_data.get("qualifying_outcome_index") or 0)
        outcomes = abuse_data.get("outcomes") or []
        if outcomes and 0 <= promo_idx < len(outcomes):
            if outcomes[promo_idx].get("bookmaker"):
                payload["bookmaker"] = outcomes[promo_idx]["bookmaker"]

    if complete:
        profit = _abuse_profit(abuse_data)
        payload.update({
            "profit": profit,
            "result": _abuse_result_from_profit(profit),
            "settled_at": now_utc()
        })
    else:
        payload.update({
            "profit": 0.0,
            "result": "Pending",
            "settled_at": None
        })

    response = (
        supabase
        .table("bets")
        .update(payload)
        .eq("id", bet_id)
        .eq("user_id", st.session_state.user_id)
        .execute()
    )
    return response.data


def abuse_bet_page():
    st.caption(
        "Track promo / matched-betting abuse separately from normal bets. "
        "Free-bet or bonus face value is NOT booked as profit; only the cash "
        "you actually realize from it is counted."
    )

    abuse_category = st.radio(
        "Abuse Type",
        ["Sports Abuse", "Casino Abuse"],
        horizontal=True,
        key="abuse_category"
    )

    bet_date = st.date_input(
        "Date",
        value=date.today(),
        key="abuse_date"
    )

    abuse_confidence = st.radio(
        "Confidence",
        ["Low", "Medium", "High"],
        index=1,
        horizontal=True,
        key="abuse_confidence"
    )

    notes = ""

    if abuse_category == "Sports Abuse":
        sport_options = list(SPORTS) + ["Other"]
        sport = st.selectbox(
            "Sport",
            sport_options,
            key="abuse_sport"
        )

        event = st.text_input(
            "Event / Match",
            placeholder="e.g. Arsenal - Chelsea",
            key="abuse_event"
        )

        match_format = st.radio(
            "Match Winner Format",
            ["2-Way Match Winner", "3-Way Match Winner"],
            horizontal=True,
            key="abuse_match_format"
        )

        promo_mechanic = st.selectbox(
            "Promo / Abuse Mechanic",
            ["Free Bet", "Bonus", "Double Win", "Cashback", "Other"],
            key="abuse_promo_mechanic"
        )

        outcome_count = 2 if match_format.startswith("2-Way") else 3
        default_labels = (
            ["Selection 1", "Selection 2"]
            if outcome_count == 2
            else ["Home", "Draw", "Away"]
        )

        raw_outcomes = []
        st.write("**Match-winner prices**")
        for idx in range(outcome_count):
            c1, c2, c3 = st.columns([2.4, 1, 1.8])
            with c1:
                label = st.text_input(
                    f"Outcome {idx + 1}",
                    value=default_labels[idx],
                    key=f"abuse_outcome_label_{idx}"
                )
            with c2:
                odds = st.number_input(
                    "Odds",
                    min_value=1.01,
                    value=2.00 if outcome_count == 2 else 3.00,
                    step=0.01,
                    format="%.2f",
                    key=f"abuse_outcome_odds_{idx}"
                )
            with c3:
                bookmaker = st.selectbox(
                    "Bookmaker",
                    BOOKMAKERS,
                    key=f"abuse_outcome_bookmaker_{idx}"
                )

            raw_outcomes.append({
                "label": label.strip() or default_labels[idx],
                "odds": float(odds),
                "bookmaker": bookmaker
            })

        outcome_labels = [x["label"] for x in raw_outcomes]
        qualifying_label = st.selectbox(
            "Qualifying / promo bet is on",
            outcome_labels,
            key="abuse_qualifying_outcome"
        )
        qualifying_index = outcome_labels.index(qualifying_label)

        qualifying_stake = st.number_input(
            "Qualifying stake (€)",
            min_value=0.01,
            value=100.00,
            step=5.00,
            format="%.2f",
            key="abuse_qualifying_stake"
        )

        outcomes, total_outlay, outcome_pls = _abuse_equalized_outcomes(
            raw_outcomes,
            qualifying_index,
            qualifying_stake
        )

        st.subheader("Recommended equal-loss staking")
        hedge_rows = []
        for idx, outcome in enumerate(outcomes):
            hedge_rows.append({
                "Outcome": outcome.get("label"),
                "Bookmaker": outcome.get("bookmaker"),
                "Odds": round(float(outcome.get("odds") or 0), 2),
                "Stake €": round(float(outcome.get("stake") or 0), 2),
                "P/L if wins €": round(float(outcome.get("qualifying_pl") or 0), 2),
                "Promo bet": "Yes" if idx == qualifying_index else ""
            })
        st.dataframe(pd.DataFrame(hedge_rows), hide_index=True, use_container_width=True)

        worst_pl = min(outcome_pls) if outcome_pls else 0.0
        best_pl = max(outcome_pls) if outcome_pls else 0.0
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total cash staked", f"€{total_outlay:.2f}")
        with c2:
            st.metric("Worst-case qualifying P/L", f"€{worst_pl:+.2f}")
        with c3:
            st.metric("P/L range after rounding", f"€{worst_pl:+.2f} to €{best_pl:+.2f}")

        reward_face_value = st.number_input(
            f"{promo_mechanic} face value / reward (€)",
            min_value=0.00,
            value=100.00,
            step=5.00,
            format="%.2f",
            key="abuse_reward_face_value",
            help="Tracked for promo value only. It is not counted as cash profit."
        )

        promo_realized_cash = st.number_input(
            "Cash actually realized from the promo (€)",
            min_value=0.00,
            value=0.00,
            step=1.00,
            format="%.2f",
            key="abuse_promo_realized_cash",
            help=(
                "Example: you spent about €10 to unlock a €100 free bet. "
                "If that free bet later produces €65 cash, enter €65 here."
            )
        )

        completed_now = st.checkbox(
            "Complete this abuse now",
            value=False,
            key="abuse_complete_now",
            help="Leave off if the promo/free bet has not been converted yet."
        )

        winning_outcome_index = None
        if completed_now:
            winning_label = st.selectbox(
                "Actual match winner / settled outcome",
                outcome_labels,
                key="abuse_winning_outcome"
            )
            winning_outcome_index = outcome_labels.index(winning_label)
            exact_base_pl = outcomes[winning_outcome_index]["qualifying_pl"]
            final_profit = round(float(exact_base_pl) + float(promo_realized_cash), 2)
            st.metric("Final abuse P/L", f"€{final_profit:+.2f}")

        notes = st.text_area(
            "Notes",
            placeholder="Optional promo details",
            key="abuse_notes"
        )

        abuse_data = {
            "category": "SPORTS",
            "match_format": match_format,
            "promo_mechanic": promo_mechanic,
            "qualifying_outcome_index": qualifying_index,
            "qualifying_stake": float(qualifying_stake),
            "outcomes": outcomes,
            "total_cash_outlay": float(total_outlay),
            "worst_case_qualifying_pl": float(worst_pl),
            "reward_face_value": float(reward_face_value),
            "promo_realized_cash": float(promo_realized_cash),
            "winning_outcome_index": winning_outcome_index,
            "completed": bool(completed_now)
        }

        if st.button(
            "💾 SAVE SPORTS ABUSE",
            type="primary",
            use_container_width=True,
            key="save_sports_abuse"
        ):
            if not event.strip():
                st.error("Event / Match is required.")
                return

            profit = _abuse_profit(abuse_data) if completed_now else 0.0
            result = _abuse_result_from_profit(profit) if completed_now else "Pending"
            promo_bookmaker = outcomes[qualifying_index].get("bookmaker") or BOOKMAKERS[0]

            record = {
                "user_id": st.session_state.user_id,
                "bet_date": bet_date.isoformat(),
                "is_live": False,
                "sport": sport,
                "league": "Promo Abuse",
                "event": event.strip(),
                "scope": "MATCH",
                "subject": None,
                "selection_2": None,
                "market": ABUSE_MARKET,
                "period": "Promo",
                "side": None,
                "line": None,
                "bookmaker": promo_bookmaker,
                "market_odds": 1.01,
                "my_odds": None,
                "origin": "SELF",
                "tipster_id": None,
                "tipster_posted_odds": None,
                "confidence": abuse_confidence,
                "has_own_reasoning": False,
                "primary_reason": None,
                "secondary_reason": None,
                "stake": float(total_outlay),
                "result": result,
                "p_market": None,
                "p_you": None,
                "edge_pp": None,
                "ev_pct": None,
                "price_deterioration_pp": None,
                "cashout_return": None,
                "profit": float(profit),
                "notes": notes.strip() or None,
                "abuse_data": abuse_data
            }
            if completed_now:
                record["settled_at"] = now_utc()

            try:
                response = supabase.table("bets").insert(record).execute()
                if response.data:
                    st.success("✅ Sports abuse saved.")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not save abuse: {e}")

    else:
        sport = "Casino"
        operator_options = list(BOOKMAKERS)
        operator = st.selectbox(
            "Casino / Operator",
            operator_options,
            accept_new_options=True,
            key="casino_abuse_operator"
        )

        promo_name = st.text_input(
            "Promo / Abuse name",
            placeholder="e.g. Deposit bonus / Free Spins campaign",
            key="casino_abuse_name"
        )

        reward_type = st.selectbox(
            "Reward Type",
            ["Bonus", "Free Spins", "Cashback", "Other"],
            key="casino_abuse_reward_type"
        )

        c1, c2 = st.columns(2)
        with c1:
            cash_in = st.number_input(
                "Own cash committed (€)",
                min_value=0.00,
                value=100.00,
                step=5.00,
                format="%.2f",
                key="casino_abuse_cash_in"
            )
        with c2:
            required_turnover = st.number_input(
                "Required turnover / wagering (€)",
                min_value=0.00,
                value=1000.00,
                step=50.00,
                format="%.2f",
                key="casino_abuse_turnover"
            )

        reward_face_value = st.number_input(
            f"{reward_type} face value (€)",
            min_value=0.00,
            value=100.00,
            step=5.00,
            format="%.2f",
            key="casino_abuse_reward_value"
        )

        completed_now = st.checkbox(
            "Casino abuse completed",
            value=False,
            key="casino_abuse_complete"
        )

        final_cash_out = 0.0
        if completed_now:
            final_cash_out = st.number_input(
                "Final cash withdrawn / returned (€)",
                min_value=0.00,
                value=float(cash_in),
                step=5.00,
                format="%.2f",
                key="casino_abuse_final_cash"
            )
            st.metric(
                "Final casino abuse P/L",
                f"€{float(final_cash_out) - float(cash_in):+.2f}"
            )

        notes = st.text_area(
            "Notes",
            placeholder="Optional casino/promo details",
            key="casino_abuse_notes"
        )

        abuse_data = {
            "category": "CASINO",
            "operator": operator,
            "promo_name": promo_name.strip(),
            "reward_type": reward_type,
            "cash_in": float(cash_in),
            "required_turnover": float(required_turnover),
            "reward_face_value": float(reward_face_value),
            "final_cash_out": float(final_cash_out),
            "completed": bool(completed_now)
        }

        if st.button(
            "💾 SAVE CASINO ABUSE",
            type="primary",
            use_container_width=True,
            key="save_casino_abuse"
        ):
            event_name = promo_name.strip() or f"{operator} Casino Abuse"
            profit = _abuse_profit(abuse_data) if completed_now else 0.0
            result = _abuse_result_from_profit(profit) if completed_now else "Pending"

            record = {
                "user_id": st.session_state.user_id,
                "bet_date": bet_date.isoformat(),
                "is_live": False,
                "sport": "Casino",
                "league": "Casino Abuse",
                "event": event_name,
                "scope": "MATCH",
                "subject": None,
                "selection_2": None,
                "market": ABUSE_MARKET,
                "period": "Promo",
                "side": None,
                "line": None,
                "bookmaker": operator,
                "market_odds": 1.01,
                "my_odds": None,
                "origin": "SELF",
                "tipster_id": None,
                "tipster_posted_odds": None,
                "confidence": abuse_confidence,
                "has_own_reasoning": False,
                "primary_reason": None,
                "secondary_reason": None,
                "stake": float(cash_in),
                "result": result,
                "p_market": None,
                "p_you": None,
                "edge_pp": None,
                "ev_pct": None,
                "price_deterioration_pp": None,
                "cashout_return": None,
                "profit": float(profit),
                "notes": notes.strip() or None,
                "abuse_data": abuse_data
            }
            if completed_now:
                record["settled_at"] = now_utc()

            try:
                response = supabase.table("bets").insert(record).execute()
                if response.data:
                    st.success("✅ Casino abuse saved.")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not save casino abuse: {e}")


def render_pending_abuse(bet):
    data = copy.deepcopy(bet.get("abuse_data") or {})
    category = data.get("category") or "SPORTS"

    st.subheader(f"🧪 {bet.get('event') or 'Abuse'}")
    st.caption(f"{bet.get('bet_date')} | {category.title()} Abuse")

    if category == "CASINO":
        cash_in = float(data.get("cash_in") or 0)
        turnover = float(data.get("required_turnover") or 0)
        reward_face = float(data.get("reward_face_value") or 0)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Own cash", f"€{cash_in:.2f}")
        with c2:
            st.metric("Required turnover", f"€{turnover:.2f}")
        with c3:
            st.metric("Reward face value", f"€{reward_face:.2f}")

        final_cash = st.number_input(
            "Final cash withdrawn / returned (€)",
            min_value=0.00,
            value=float(data.get("final_cash_out") or 0),
            step=5.00,
            format="%.2f",
            key=f"pending_abuse_casino_final_{bet['id']}"
        )
        data["final_cash_out"] = float(final_cash)
        projected_profit = _abuse_casino_profit(data)
        st.metric("Final P/L if completed now", f"€{projected_profit:+.2f}")

    else:
        outcomes = data.get("outcomes") or []
        st.write(
            f"**{data.get('match_format') or 'Match Winner'}** | "
            f"{data.get('promo_mechanic') or 'Promo'}"
        )
        rows = []
        for outcome in outcomes:
            rows.append({
                "Outcome": outcome.get("label"),
                "Odds": outcome.get("odds"),
                "Stake €": outcome.get("stake"),
                "P/L if wins €": outcome.get("qualifying_pl")
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        reward_face = float(data.get("reward_face_value") or 0)
        st.caption(
            f"Promo face value: €{reward_face:.2f} (tracked, not booked as profit)"
        )
        labels = [x.get("label") or f"Outcome {i+1}" for i, x in enumerate(outcomes)]
        if labels:
            current_index = data.get("winning_outcome_index")
            current_index = int(current_index) if current_index is not None else 0
            winner = st.selectbox(
                "Actual settled outcome",
                labels,
                index=max(0, min(current_index, len(labels)-1)),
                key=f"pending_abuse_winner_{bet['id']}"
            )
            data["winning_outcome_index"] = labels.index(winner)

        promo_cash = st.number_input(
            "Cash actually realized from free bet / bonus / promo (€)",
            min_value=0.00,
            value=float(data.get("promo_realized_cash") or 0),
            step=1.00,
            format="%.2f",
            key=f"pending_abuse_promo_cash_{bet['id']}"
        )
        data["promo_realized_cash"] = float(promo_cash)
        projected_profit = _abuse_sports_profit(data)
        st.metric("Final P/L if completed now", f"€{projected_profit:+.2f}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "💾 Save abuse progress",
            use_container_width=True,
            key=f"save_abuse_progress_{bet['id']}"
        ):
            _save_abuse_progress(bet["id"], data, complete=False)
            st.success("Abuse progress saved.")
            st.rerun()
    with c2:
        if st.button(
            "✅ Complete abuse",
            type="primary",
            use_container_width=True,
            key=f"complete_abuse_{bet['id']}"
        ):
            _save_abuse_progress(bet["id"], data, complete=True)
            st.rerun()


def render_manage_abuse(bet):
    bet_id = bet["id"]
    data = copy.deepcopy(bet.get("abuse_data") or {})
    category = data.get("category") or "SPORTS"

    st.divider()
    st.subheader("🧪 Edit Abuse")

    edit_date = st.date_input(
        "Date",
        value=datetime.strptime(bet["bet_date"], "%Y-%m-%d").date(),
        key=f"edit_abuse_date_{bet_id}"
    )
    edit_event = st.text_input(
        "Name / Event",
        value=bet.get("event") or "Abuse",
        key=f"edit_abuse_event_{bet_id}"
    )
    abuse_conf_options = ["Low", "Medium", "High"]
    current_abuse_conf = bet.get("confidence")
    if current_abuse_conf not in abuse_conf_options:
        current_abuse_conf = "Medium"
    edit_abuse_confidence = st.radio(
        "Confidence",
        abuse_conf_options,
        index=abuse_conf_options.index(current_abuse_conf),
        horizontal=True,
        key=f"edit_abuse_confidence_{bet_id}"
    )

    if category == "CASINO":
        operator = st.selectbox(
            "Casino / Operator",
            list(dict.fromkeys([data.get("operator"), bet.get("bookmaker")] + list(BOOKMAKERS))),
            index=0,
            accept_new_options=True,
            key=f"edit_abuse_operator_{bet_id}"
        )
        cash_in = st.number_input(
            "Own cash committed (€)",
            min_value=0.00,
            value=float(data.get("cash_in") or bet.get("stake") or 0),
            step=5.00,
            key=f"edit_abuse_cash_in_{bet_id}"
        )
        turnover = st.number_input(
            "Required turnover / wagering (€)",
            min_value=0.00,
            value=float(data.get("required_turnover") or 0),
            step=50.00,
            key=f"edit_abuse_turnover_{bet_id}"
        )
        reward_face = st.number_input(
            "Reward face value (€)",
            min_value=0.00,
            value=float(data.get("reward_face_value") or 0),
            step=5.00,
            key=f"edit_abuse_reward_face_{bet_id}"
        )
        final_cash = st.number_input(
            "Final cash withdrawn / returned (€)",
            min_value=0.00,
            value=float(data.get("final_cash_out") or 0),
            step=5.00,
            key=f"edit_abuse_final_cash_{bet_id}"
        )
        completed = st.checkbox(
            "Completed",
            value=bet.get("result") != "Pending",
            key=f"edit_abuse_completed_{bet_id}"
        )
        data.update({
            "operator": operator,
            "cash_in": float(cash_in),
            "required_turnover": float(turnover),
            "reward_face_value": float(reward_face),
            "final_cash_out": float(final_cash)
        })
    else:
        st.caption(
            f"{data.get('match_format') or 'Match Winner'} | "
            f"{data.get('promo_mechanic') or 'Promo'}"
        )
        outcomes = data.get("outcomes") or []
        edited_raw = []
        for idx, outcome in enumerate(outcomes):
            c1, c2, c3 = st.columns([2.4, 1, 1.8])
            with c1:
                label = st.text_input(
                    f"Outcome {idx + 1}",
                    value=outcome.get("label") or f"Outcome {idx+1}",
                    key=f"edit_abuse_label_{bet_id}_{idx}"
                )
            with c2:
                odds = st.number_input(
                    "Odds",
                    min_value=1.01,
                    value=float(outcome.get("odds") or 2.0),
                    step=0.01,
                    format="%.2f",
                    key=f"edit_abuse_odds_{bet_id}_{idx}"
                )
            with c3:
                bm_options = list(BOOKMAKERS)
                if outcome.get("bookmaker") not in bm_options:
                    bm_options.append(outcome.get("bookmaker"))
                bookmaker = st.selectbox(
                    "Bookmaker",
                    bm_options,
                    index=safe_index(bm_options, outcome.get("bookmaker")),
                    key=f"edit_abuse_bm_{bet_id}_{idx}"
                )
            edited_raw.append({"label": label, "odds": float(odds), "bookmaker": bookmaker})

        labels = [x["label"] for x in edited_raw]
        current_q = int(data.get("qualifying_outcome_index") or 0)
        q_label = st.selectbox(
            "Qualifying / promo bet",
            labels,
            index=max(0, min(current_q, len(labels)-1)) if labels else 0,
            key=f"edit_abuse_q_{bet_id}"
        ) if labels else None
        q_index = labels.index(q_label) if q_label in labels else 0
        q_stake = st.number_input(
            "Qualifying stake (€)",
            min_value=0.01,
            value=float(data.get("qualifying_stake") or 100),
            step=5.00,
            key=f"edit_abuse_q_stake_{bet_id}"
        )
        edited_outcomes, total_outlay, pls = _abuse_equalized_outcomes(
            edited_raw, q_index, q_stake
        )
        data["outcomes"] = edited_outcomes
        data["qualifying_outcome_index"] = q_index
        data["qualifying_stake"] = float(q_stake)
        data["total_cash_outlay"] = float(total_outlay)
        data["worst_case_qualifying_pl"] = float(min(pls) if pls else 0)

        reward_face = st.number_input(
            "Promo face value (€)",
            min_value=0.00,
            value=float(data.get("reward_face_value") or 0),
            step=5.00,
            key=f"edit_abuse_reward_face_{bet_id}"
        )
        data["reward_face_value"] = float(reward_face)

        if labels:
            current_w = data.get("winning_outcome_index")
            current_w = int(current_w) if current_w is not None else 0
            winner = st.selectbox(
                "Actual settled outcome",
                labels,
                index=max(0, min(current_w, len(labels)-1)),
                key=f"edit_abuse_winner_{bet_id}"
            )
            data["winning_outcome_index"] = labels.index(winner)

        promo_cash = st.number_input(
            "Cash actually realized from promo (€)",
            min_value=0.00,
            value=float(data.get("promo_realized_cash") or 0),
            step=1.00,
            key=f"edit_abuse_promo_cash_{bet_id}"
        )
        data["promo_realized_cash"] = float(promo_cash)
        completed = st.checkbox(
            "Completed",
            value=bet.get("result") != "Pending",
            key=f"edit_abuse_completed_{bet_id}"
        )

    edit_notes = st.text_area(
        "Notes",
        value=bet.get("notes") or "",
        key=f"edit_abuse_notes_{bet_id}"
    )

    projected = _abuse_profit(data)
    st.metric("P/L if completed", f"€{projected:+.2f}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "💾 SAVE ABUSE",
            type="primary",
            use_container_width=True,
            key=f"save_abuse_manage_{bet_id}"
        ):
            data["completed"] = bool(completed)
            _save_abuse_progress(
                bet_id,
                data,
                complete=bool(completed),
                event=edit_event,
                notes=edit_notes
            )
            supabase.table("bets").update({
                "bet_date": edit_date.isoformat(),
                "confidence": edit_abuse_confidence,
                "updated_at": now_utc()
            }).eq("id", bet_id).eq("user_id", st.session_state.user_id).execute()
            st.success("Abuse updated.")
            st.rerun()
    with c2:
        if st.button(
            "🗑️ MOVE TO TRASH",
            use_container_width=True,
            key=f"delete_abuse_{bet_id}"
        ):
            soft_delete_bet(bet_id)
            st.rerun()

def combo_bet_page(combo_type, force_outright=False):
    """Fast entry for Bet Builders and Parlays using one JSONB column."""

    is_bb = combo_type == "Bet Builder"
    force_outright = bool(force_outright and not is_bb)

    st.caption(
        "Enter the individual selections and their standalone odds. "
        "The final odds below are the actual combined odds you took."
    )

    bet_date = st.date_input(
        "Bet Date",
        value=date.today(),
        key="combo_bet_date"
    )

    is_live = st.checkbox(
        "🔴 Live Bet",
        value=False,
        key="combo_is_live"
    )

    combo_legs = []

    if is_bb:
        sport = st.selectbox(
            "Sport",
            SPORTS,
            key="combo_bb_sport"
        )

        combo_profile = st.radio(
            "BB Type",
            ["Value", "Τζόγος"],
            index=0,
            horizontal=True,
            key="combo_bb_profile"
        )

        league_options = load_user_league_options(sport)
        league_key = f"combo_bb_league_{sport.lower()}"
        league_options = _include_session_option(
            league_options,
            league_key
        )
        league = st.selectbox(
            "League / Tour",
            league_options,
            accept_new_options=True,
            key=league_key
        )

        suggestions = load_entry_suggestions(sport)
        event_options = _include_session_option(
            suggestions.get("regular_events", []),
            "combo_bb_event"
        )
        event = st.selectbox(
            "Event / Match / Fight",
            event_options,
            index=None,
            placeholder="Search or enter event...",
            accept_new_options=True,
            key="combo_bb_event"
        )

        selection_count = int(
            st.number_input(
                "Number of selections",
                min_value=2,
                max_value=20,
                value=2,
                step=1,
                key="combo_bb_selection_count"
            )
        )

        selections = []

        for index in range(selection_count):
            c1, c2 = st.columns([3, 1])
            with c1:
                label = st.text_input(
                    f"Selection {index + 1}",
                    placeholder="e.g. Sinner ML / Giannis O10.5 rebounds",
                    key=f"combo_bb_selection_{index}"
                )
            with c2:
                odds = st.number_input(
                    "Standalone odds",
                    min_value=1.01,
                    value=1.50,
                    step=0.01,
                    format="%.2f",
                    key=f"combo_bb_selection_odds_{index}"
                )

            selections.append({
                "label": label.strip(),
                "odds": float(odds),
                "result": "Pending"
            })

        combo_legs = [{
            "kind": "BET_BUILDER",
            "label": (event or "Bet Builder"),
            "selections": selections,
            "combo_profile": combo_profile
        }]

        reason_sport = sport

    else:
        parlay_sport_options = list(SPORTS) + ["Combo Sports"]

        sport = st.selectbox(
            "Parlay Sport",
            parlay_sport_options,
            key="combo_parlay_sport",
            help=(
                "Choose one sport when every component is from the same sport. "
                "Use Combo Sports when the parlay mixes sports."
            )
        )

        combo_profile = st.radio(
            "Parlay Type",
            ["Value", "Τζόγος"],
            index=0,
            horizontal=True,
            key="combo_parlay_profile"
        )

        if force_outright:
            parlay_content = "Outright"
            st.caption(
                "🏆 Outright Parlay — every leg is an outright/future. "
                "This bet will be stored under Outrights."
            )
        else:
            parlay_content = st.radio(
                "Parlay Content",
                ["Regular", "Outright", "Mixed"],
                index=0,
                horizontal=True,
                key="combo_parlay_content",
                help=(
                    "Regular = match/player/team selections and BBs. "
                    "Outright = every leg is an outright/future and will be stored under Outrights. "
                    "Mixed = combine regular selections, BBs and outrights in one parlay."
                )
            )

        league = "Multiple"
        event = st.text_input(
            "Outright parlay name (optional)" if parlay_content == "Outright" else "Parlay name (optional)",
            placeholder=(
                "e.g. Season Outrights"
                if parlay_content == "Outright"
                else "e.g. Sunday Parlay"
            ),
            key="combo_parlay_name"
        )
        event = event.strip() or "Parlay"

        component_count = int(
            st.number_input(
                "Number of parlay legs / components",
                min_value=2,
                max_value=20,
                value=2,
                step=1,
                key="combo_parlay_component_count"
            )
        )

        for component_index in range(component_count):
            with st.expander(
                f"Leg {component_index + 1}",
                expanded=True
            ):
                if parlay_content == "Outright":
                    component_kind = "Outright"
                    st.caption("🏆 Outright leg")
                else:
                    leg_type_options = (
                        ["Single selection", "Bet Builder", "Outright"]
                        if parlay_content == "Mixed"
                        else ["Single selection", "Bet Builder"]
                    )
                    component_kind = st.radio(
                        "Leg type",
                        leg_type_options,
                        horizontal=True,
                        key=f"combo_component_kind_{component_index}"
                    )

                if component_kind == "Single selection":
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        label = st.text_input(
                            "Selection",
                            placeholder="e.g. Sinner ML",
                            key=f"combo_component_label_{component_index}"
                        )
                    with c2:
                        odds = st.number_input(
                            "Standalone odds",
                            min_value=1.01,
                            value=1.50,
                            step=0.01,
                            format="%.2f",
                            key=f"combo_component_odds_{component_index}"
                        )

                    combo_legs.append({
                        "kind": "SINGLE",
                        "label": label.strip(),
                        "odds": float(odds),
                        "result": "Pending",
                        "parlay_sport": sport,
                        "combo_profile": combo_profile,
                        "parlay_profile": combo_profile,
                        "parlay_content": parlay_content
                    })

                elif component_kind == "Outright":
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        outright_label = st.text_input(
                            "Outright selection",
                            placeholder=(
                                "e.g. Sinner Wimbledon Winner / "
                                "Arsenal Premier League Winner"
                            ),
                            key=f"combo_component_outright_{component_index}"
                        )
                    with c2:
                        outright_odds = st.number_input(
                            "Standalone odds",
                            min_value=1.01,
                            value=2.00,
                            step=0.01,
                            format="%.2f",
                            key=f"combo_component_outright_odds_{component_index}"
                        )

                    combo_legs.append({
                        "kind": "OUTRIGHT",
                        "label": outright_label.strip(),
                        "odds": float(outright_odds),
                        "result": "Pending",
                        "parlay_sport": sport,
                        "combo_profile": combo_profile,
                        "parlay_profile": combo_profile,
                        "parlay_content": parlay_content
                    })

                else:
                    bb_label = st.text_input(
                        "BB / event label",
                        placeholder="e.g. Arsenal - Chelsea BB",
                        key=f"combo_component_bb_label_{component_index}"
                    )

                    bb_count = int(
                        st.number_input(
                            "Selections inside this BB",
                            min_value=2,
                            max_value=20,
                            value=2,
                            step=1,
                            key=f"combo_component_bb_count_{component_index}"
                        )
                    )

                    bb_selections = []
                    for selection_index in range(bb_count):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            selection_label = st.text_input(
                                f"BB selection {selection_index + 1}",
                                key=(
                                    f"combo_component_{component_index}_"
                                    f"selection_{selection_index}"
                                )
                            )
                        with c2:
                            selection_odds = st.number_input(
                                "Standalone odds",
                                min_value=1.01,
                                value=1.50,
                                step=0.01,
                                format="%.2f",
                                key=(
                                    f"combo_component_{component_index}_"
                                    f"odds_{selection_index}"
                                )
                            )

                        bb_selections.append({
                            "label": selection_label.strip(),
                            "odds": float(selection_odds),
                            "result": "Pending"
                        })

                    bb_component_odds = st.number_input(
                        "BB combined odds (as one parlay leg)",
                        min_value=1.01,
                        value=2.00,
                        step=0.01,
                        format="%.2f",
                        key=f"combo_component_bb_price_{component_index}",
                        help=(
                            "Enter the actual combined BB price shown by the bookmaker. "
                            "The tracker uses this one price when calculating the parlay."
                        )
                    )

                    combo_legs.append({
                        "kind": "BET_BUILDER",
                        "label": bb_label.strip() or f"BB {component_index + 1}",
                        "selections": bb_selections,
                        "component_odds": float(bb_component_odds),
                        "parlay_sport": sport,
                        "combo_profile": combo_profile,
                        "parlay_profile": combo_profile,
                        "parlay_content": parlay_content
                    })

        reason_sport = sport

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        bookmaker = st.selectbox(
            "Bookmaker",
            BOOKMAKERS,
            key="combo_bookmaker"
        )

    with c2:
        if is_bb:
            final_odds = st.number_input(
                "Final odds taken",
                min_value=1.01,
                value=2.00,
                step=0.01,
                format="%.2f",
                key="combo_final_odds"
            )
        else:
            calculated_parlay_odds = _calculate_parlay_odds(combo_legs)
            st.metric(
                "Calculated parlay odds",
                f"{calculated_parlay_odds:.2f}"
            )

    if not is_bb:
        use_calculated_odds = st.checkbox(
            "Use calculated parlay odds",
            value=True,
            key="combo_use_calculated_parlay_odds",
            help=(
                "Turn this off only if the bookmaker's final price differs, "
                "for example because of a boost."
            )
        )

        if use_calculated_odds:
            final_odds = calculated_parlay_odds
            st.caption(
                "Final odds will be filled automatically from the parlay legs."
            )
        else:
            final_odds = st.number_input(
                "Final odds taken (override)",
                min_value=1.01,
                value=float(calculated_parlay_odds),
                step=0.01,
                format="%.2f",
                key="combo_final_odds_override"
            )

    origin_data = _render_combo_origin_fields(reason_sport)

    stake = st.number_input(
        "Stake",
        min_value=0.01,
        value=10.00,
        step=1.00,
        key="combo_stake"
    )

    notes = st.text_area(
        "Notes",
        placeholder="Optional",
        key="combo_notes"
    )

    if origin_data["origin"] == "SELF":
        preview = calculate_metrics(
            final_odds,
            origin_data["my_odds"]
        )
        st.info(
            f"Market probability: {preview['p_market']*100:.2f}% | "
            f"My probability: {preview['p_you']*100:.2f}% | "
            f"EV: {preview['ev_pct']:.2f}%"
        )

    if st.button(
        f"💾 SAVE {combo_type.upper()}",
        type="primary",
        use_container_width=True,
        key="combo_save_bet"
    ):
        flat_selections = _combo_flat_selections(combo_legs)

        if not flat_selections:
            st.error("Add at least one selection.")
            return

        missing_labels = [
            item for item in flat_selections
            if not (item.get("label") or "").strip()
        ]
        if missing_labels:
            st.error("Every selection needs a short description.")
            return

        if is_bb and not (event and str(event).strip()):
            st.error("Event is required for a Bet Builder.")
            return

        if (
            origin_data["origin"] == "TIPSTER"
            and origin_data["tipster_id"] is None
        ):
            st.error("Select or create a tipster.")
            return

        if (
            origin_data["primary_reason"] == "Select reason..."
            and (
                origin_data["origin"] == "SELF"
                or origin_data["has_own_reasoning"]
            )
        ):
            st.error("Select a Primary Reason.")
            return

        metrics = calculate_metrics(
            final_odds,
            origin_data["my_odds"],
            origin_data["tipster_posted_odds"]
        )

        contains_outright = (
            (not is_bb)
            and _combo_has_outright_leg(combo_legs)
        )

        record = {
            "user_id": st.session_state.user_id,
            "bet_date": bet_date.isoformat(),
            "is_live": bool(is_live),
            "sport": sport,
            "league": league,
            "event": str(event).strip(),
            "scope": (
                "OUTRIGHT"
                if contains_outright
                else "MATCH"
            ),
            "subject": None,
            "selection_2": None,
            "market": combo_type,
            "period": (
                "Full Competition"
                if contains_outright
                else "Combined"
            ),
            "side": None,
            "line": None,
            "bookmaker": bookmaker,
            "market_odds": float(final_odds),
            "my_odds": origin_data["my_odds"],
            "origin": origin_data["origin"],
            "tipster_id": origin_data["tipster_id"],
            "tipster_posted_odds": origin_data["tipster_posted_odds"],
            "confidence": origin_data["confidence"],
            "has_own_reasoning": origin_data["has_own_reasoning"],
            "primary_reason": (
                None
                if origin_data["primary_reason"] == "Select reason..."
                else origin_data["primary_reason"]
            ),
            "secondary_reason": (
                None
                if origin_data["secondary_reason"] in [None, "None"]
                else origin_data["secondary_reason"]
            ),
            "stake": float(stake),
            "result": "Pending",
            "p_market": metrics["p_market"],
            "p_you": metrics["p_you"],
            "edge_pp": metrics["edge_pp"],
            "ev_pct": metrics["ev_pct"],
            "price_deterioration_pp": metrics["price_deterioration_pp"],
            "profit": 0,
            "notes": notes.strip() or None,
            "combo_legs": combo_legs
        }

        try:
            response = (
                supabase
                .table("bets")
                .insert(record)
                .execute()
            )
            if response.data:
                st.success(
                    f"✅ {combo_type} saved successfully!"
                )
                st.write(
                    f"Selections: {_combo_selection_count(combo_legs)} | "
                    f"Final odds: {float(final_odds):.2f}"
                )
        except Exception as e:
            st.error(f"Could not save {combo_type}: {e}")


def add_bet_page():

    st.header("➕ Add Bet")

    bet_structure = st.radio(
        "Bet Structure",
        ["Single", "Bet Builder", "Parlay", "Outright", "Abuse"],
        horizontal=True,
        key="add_bet_structure"
    )

    if bet_structure == "Abuse":
        abuse_bet_page()
        return

    if bet_structure == "Bet Builder":
        combo_bet_page("Bet Builder")
        return

    if bet_structure == "Parlay":
        combo_bet_page("Parlay")
        return

    force_outright_single = False

    if bet_structure == "Outright":
        outright_entry_type = st.radio(
            "Outright Type",
            ["Single Outright", "Outright Parlay"],
            index=0,
            horizontal=True,
            key="add_outright_entry_type",
            help=(
                "Single Outright = one future/outright. "
                "Outright Parlay = two or more outright selections combined."
            )
        )

        if outright_entry_type == "Outright Parlay":
            combo_bet_page("Parlay", force_outright=True)
            return

        force_outright_single = True

    def ensure_valid(
        key,
        options,
        default=None
    ):
        if key not in st.session_state:
            return

        current = st.session_state[key]

        if current in options:
            return

        if default is not None:
            st.session_state[key] = default
        else:
            st.session_state.pop(key, None)

    ensure_valid(
        "add_sport",
        SPORTS,
        DEFAULT_SPORT
    )

    sport = st.selectbox(
        "Sport",
        SPORTS,
        key="add_sport"
    )

    previous_sport = (
        st.session_state
        .get("_add_last_sport")
    )

    if previous_sport is None:
        st.session_state[
            "_add_last_sport"
        ] = sport

    elif previous_sport != sport:
        dependent_keys = [
            "add_scope",
            "add_regular_event",
            "add_outright_event",
            "add_player",
            "add_team",
            "add_outright_market",
            "add_outright_subject",
            "add_outright_selection_2",
            "add_market_player",
            "add_market_team",
            "add_market_match",
            "add_period",
            "add_side",
            "add_line",
            "add_self_primary_reason",
            "add_self_secondary_reason",
            "add_tipster_primary_reason",
            "add_tipster_secondary_reason"
        ]

        for key in dependent_keys:
            st.session_state.pop(
                key,
                None
            )

        st.session_state[
            "_add_last_sport"
        ] = sport

        st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        bet_date = st.date_input(
            "Bet Date",
            value=date.today(),
            key="add_bet_date"
        )

    with col2:
        league_options = (
            load_user_league_options(
                sport
            )
        )

        league_key = (
            f"add_league_"
            f"{sport.lower()}"
        )

        league_options = (
            _include_session_option(
                league_options,
                league_key
            )
        )

        league = st.selectbox(
            "League / Tour",
            league_options,
            accept_new_options=True,
            key=league_key
        )

    is_live = st.checkbox(
        "🔴 Live Bet",
        value=False,
        key="add_is_live",
        help=(
            "Leave unchecked for a pre-live bet. "
            "Check it only if the bet was placed live."
        )
    )

    scope_options = (
        get_scope_options(
            sport
        )
    )

    if force_outright_single:
        scope = "OUTRIGHT"
        st.caption("🏆 Outright")
    else:
        ensure_valid(
            "add_scope",
            scope_options,
            scope_options[0]
        )

        scope = st.radio(
            "Bet Type",
            scope_options,
            horizontal=True,
            key="add_scope"
        )

    entry_suggestions = (
        load_entry_suggestions(
            sport
        )
    )

    for _bucket in [
        "regular_events",
        "outright_events",
        "players",
        "teams"
    ]:
        entry_suggestions[_bucket] = (
            _merge_recent_entry_options(
                _bucket,
                entry_suggestions.get(
                    _bucket,
                    []
                ),
                sport=sport
            )
        )

    event_options = (
        entry_suggestions["outright_events"]
        if scope == "OUTRIGHT"
        else entry_suggestions["regular_events"]
    )

    event_key = (
        "add_outright_event"
        if scope == "OUTRIGHT"
        else "add_regular_event"
    )

    event_options = (
        _include_session_option(
            event_options,
            event_key
        )
    )

    event = st.selectbox(
        (
            "Tournament / Event"
            if (
                sport == "Tennis"
                and scope == "OUTRIGHT"
            )
            else (
                "Competition / Event"
                if scope == "OUTRIGHT"
                else ("Fight" if sport == "UFC" else "Event")
            )
        ),
        event_options,
        index=None,
        placeholder=(
            "Search or enter tournament..."
            if (
                sport == "Tennis"
                and scope == "OUTRIGHT"
            )
            else (
                "Search or enter competition..."
                if scope == "OUTRIGHT"
                else (
                    "Search or enter fight..."
                    if sport == "UFC"
                    else "Search or enter matchup..."
                )
            )
        ),
        accept_new_options=True,
        key=event_key
    )

    st.divider()

    subject = None
    selection_2 = None
    line = None
    side = None

    if scope == "OUTRIGHT":
        market_options = (
            load_user_market_options(
                sport,
                scope
            )
        )

        market_options = (
            _include_session_option(
                market_options,
                "add_outright_market"
            )
        )

        market = st.selectbox(
            "Outright Market",
            market_options,
            accept_new_options=True,
            key="add_outright_market"
        )

        label_1, label_2 = (
            outright_selection_labels(
                market,
                sport
            )
        )

        if (
            sport == "Tennis"
            or label_1 == "Player"
            or label_1.startswith("Player ")
        ):
            outright_options_1 = (
                entry_suggestions["players"]
            )
        else:
            outright_options_1 = (
                entry_suggestions["teams"]
            )

        outright_options_1 = (
            _include_session_option(
                outright_options_1,
                "add_outright_subject"
            )
        )

        subject = st.selectbox(
            label_1,
            outright_options_1,
            index=None,
            placeholder=(
                f"Search or enter "
                f"{label_1.lower()}..."
            ),
            accept_new_options=True,
            key="add_outright_subject"
        )

        if label_2:
            second_options = (
                entry_suggestions["players"]
                if sport == "Tennis"
                else entry_suggestions["teams"]
            )

            second_options = (
                _include_session_option(
                    second_options,
                    "add_outright_selection_2"
                )
            )

            selection_2 = st.selectbox(
                label_2,
                second_options,
                index=None,
                placeholder=(
                    f"Search or enter "
                    f"{label_2.lower()}..."
                ),
                accept_new_options=True,
                key="add_outright_selection_2"
            )

        period = "Full Competition"

        st.caption(
            "🏆 This bet will be stored "
            "separately from regular "
            "pending bets."
        )

    else:
        if scope == "PLAYER":
            player_options = (
                _include_session_option(
                    entry_suggestions["players"],
                    "add_player"
                )
            )

            subject = st.selectbox(
                ("Fighter" if sport == "UFC" else "Player"),
                player_options,
                index=None,
                placeholder=(
                    "Search or enter player..."
                ),
                accept_new_options=True,
                key="add_player"
            )

        elif scope == "TEAM":
            team_options = (
                _include_session_option(
                    entry_suggestions["teams"],
                    "add_team"
                )
            )

            subject = st.selectbox(
                "Team",
                team_options,
                index=None,
                placeholder=(
                    "Search or enter team..."
                ),
                accept_new_options=True,
                key="add_team"
            )

        market_key = (
            f"add_market_"
            f"{scope.lower()}"
        )

        market_options = (
            load_user_market_options(
                sport,
                scope
            )
        )

        market_options = (
            _include_session_option(
                market_options,
                market_key
            )
        )

        market = st.selectbox(
            "Market",
            market_options,
            accept_new_options=True,
            key=market_key
        )

        periods = get_periods(
            sport
        )

        ensure_valid(
            "add_period",
            periods,
            periods[0]
        )

        period = st.selectbox(
            "Period",
            periods,
            key="add_period"
        )

        default_markets = (
            get_default_markets(
                sport,
                scope
            )
        )

        is_custom_market = (
            market not in default_markets
        )

        if is_custom_market:
            format_options = [
                "Over / Under",
                "Winner / Selection",
                "Handicap / Spread",
                "Yes / No"
            ]

            format_key = (
                "add_custom_market_format_"
                + sport
                + "_"
                + scope
                + "_"
                + market
            )

            if format_key not in st.session_state:
                st.session_state[
                    format_key
                ] = (
                    infer_saved_custom_market_format(
                        sport,
                        scope,
                        market
                    )
                )

            custom_market_format = (
                st.selectbox(
                    "Market Format",
                    format_options,
                    key=format_key
                )
            )

            style_map = {
                "Over / Under": "total",
                "Winner / Selection": "winner",
                "Handicap / Spread": "handicap",
                "Yes / No": "yes_no"
            }

            market_style = (
                style_map[
                    custom_market_format
                ]
            )
        else:
            market_style = (
                get_market_style(
                    sport,
                    scope,
                    market
                )
            )

        if market_style == "winner":
            side_options = (
                get_winner_side_options(
                    sport,
                    market
                )
            )

            ensure_valid(
                "add_side",
                side_options,
                side_options[0]
            )

            side = st.radio(
                "Selection",
                side_options,
                horizontal=True,
                key="add_side"
            )

        elif market_style == "handicap":
            side_options = (
                get_winner_side_options(
                    sport,
                    market
                )
            )

            ensure_valid(
                "add_side",
                side_options,
                side_options[0]
            )

            side = st.radio(
                "Selection",
                side_options,
                horizontal=True,
                key="add_side"
            )

            line = st.number_input(
                "Line",
                step=0.5,
                format="%.1f",
                key="add_line"
            )

        elif market_style == "yes_no":
            side_options = [
                "Yes",
                "No"
            ]

            ensure_valid(
                "add_side",
                side_options,
                "Yes"
            )

            side = st.radio(
                "Selection",
                side_options,
                horizontal=True,
                key="add_side"
            )

        else:
            side_options = [
                "Over",
                "Under"
            ]

            ensure_valid(
                "add_side",
                side_options,
                "Over"
            )

            side = st.radio(
                "Side",
                side_options,
                horizontal=True,
                key="add_side"
            )

            line = st.number_input(
                "Line",
                step=0.5,
                format="%.1f",
                key="add_line"
            )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        ensure_valid(
            "add_bookmaker",
            BOOKMAKERS,
            BOOKMAKERS[0]
        )

        bookmaker = st.selectbox(
            "Bookmaker",
            BOOKMAKERS,
            key="add_bookmaker"
        )

    with col2:
        market_odds = st.number_input(
            "Odds Taken",
            min_value=1.01,
            value=1.90,
            step=0.01,
            format="%.2f",
            key="add_market_odds"
        )

    origin = st.radio(
        "Origin",
        ["SELF", "TIPSTER"],
        horizontal=True,
        key="add_origin"
    )

    my_odds = None
    tipster_id = None
    tipster_posted_odds = None
    has_own_reasoning = False
    primary_reason = None
    secondary_reason = None
    confidence = None

    reasons = get_reasons(
        sport
    )

    if origin == "SELF":
        my_odds = st.number_input(
            "My Fair Odds",
            min_value=1.01,
            value=1.80,
            step=0.01,
            format="%.2f",
            key="add_self_fair_odds"
        )

        confidence_options = [
            "Low",
            "Medium",
            "High"
        ]

        ensure_valid(
            "add_self_confidence",
            confidence_options,
            "Medium"
        )

        confidence = st.radio(
            "Confidence",
            confidence_options,
            horizontal=True,
            key="add_self_confidence"
        )

        reason_options = (
            ["Select reason..."]
            + reasons
        )

        ensure_valid(
            "add_self_primary_reason",
            reason_options,
            "Projection Edge"
        )

        primary_reason = st.selectbox(
            "Primary Reason",
            reason_options,
            key="add_self_primary_reason"
        )

        secondary_options = (
            ["None"]
            + [
                reason
                for reason in reasons
                if reason != primary_reason
            ]
        )

        ensure_valid(
            "add_self_secondary_reason",
            secondary_options,
            "None"
        )

        secondary_reason = st.selectbox(
            "Secondary Reason",
            secondary_options,
            key="add_self_secondary_reason"
        )

        has_own_reasoning = True

    else:
        tipsters = load_tipsters()

        tipster_map = {
            t["name"]: t["id"]
            for t in tipsters
        }

        existing_names = list(
            tipster_map.keys()
        )

        tipster_options = (
            ["+ Add new tipster"]
            + existing_names
        )

        ensure_valid(
            "add_tipster_choice",
            tipster_options,
            tipster_options[0]
        )

        tipster_choice = st.selectbox(
            "Tipster",
            tipster_options,
            key="add_tipster_choice"
        )

        if tipster_choice == "+ Add new tipster":
            new_tipster = st.text_input(
                "New Tipster Name",
                key="add_new_tipster"
            )

            if st.button(
                "Save Tipster",
                key="save_tipster_button"
            ):
                try:
                    record = create_tipster(
                        new_tipster
                    )

                    if record:
                        st.success(
                            "Tipster saved."
                        )
                        st.rerun()

                except Exception as e:
                    st.error(str(e))

        else:
            tipster_id = (
                tipster_map[
                    tipster_choice
                ]
            )

        add_posted_odds = st.checkbox(
            "I know the tipster's "
            "posted odds",
            key="add_tipster_has_posted_odds"
        )

        if add_posted_odds:
            tipster_posted_odds = (
                st.number_input(
                    "Tipster Posted Odds",
                    min_value=1.01,
                    value=1.90,
                    step=0.01,
                    format="%.2f",
                    key="add_tipster_posted_odds"
                )
            )

        tipster_confidence_options = [
            "N/A",
            "Low",
            "Medium",
            "High"
        ]

        ensure_valid(
            "add_tipster_confidence",
            tipster_confidence_options,
            "Medium"
        )

        confidence = st.radio(
            "Your Confidence",
            tipster_confidence_options,
            horizontal=True,
            key="add_tipster_confidence"
        )

        has_own_reasoning = st.checkbox(
            "I also have my own "
            "reasoning for this bet",
            key="add_tipster_own_reasoning"
        )

        if has_own_reasoning:
            reason_options = (
                ["Select reason..."]
                + reasons
            )

            ensure_valid(
                "add_tipster_primary_reason",
                reason_options,
                "Projection Edge"
            )

            primary_reason = st.selectbox(
                "Primary Reason",
                reason_options,
                key="add_tipster_primary_reason"
            )

            secondary_options = (
                ["None"]
                + [
                    reason
                    for reason in reasons
                    if reason != primary_reason
                ]
            )

            ensure_valid(
                "add_tipster_secondary_reason",
                secondary_options,
                "None"
            )

            secondary_reason = st.selectbox(
                "Secondary Reason",
                secondary_options,
                key="add_tipster_secondary_reason"
            )

    st.divider()

    stake = st.number_input(
        "Stake",
        min_value=0.01,
        value=10.00,
        step=1.00,
        key="add_stake"
    )

    notes = st.text_area(
        "Notes",
        placeholder="Optional",
        key="add_notes"
    )

    if origin == "SELF":
        preview = calculate_metrics(
            market_odds,
            my_odds
        )

        st.info(
            f"Market probability: "
            f"{preview['p_market']*100:.2f}% | "
            f"My probability: "
            f"{preview['p_you']*100:.2f}% | "
            f"Probability Edge: "
            f"{preview['edge_pp']:.2f} pp | "
            f"EV: "
            f"{preview['ev_pct']:.2f}%"
        )

    if st.button(
        "💾 SAVE BET",
        type="primary",
        use_container_width=True,
        key="save_bet_button"
    ):
        if not (
            event
            and str(event).strip()
        ):
            st.error(
                "Event / Competition "
                "is required."
            )
            return

        if (
            scope in ["PLAYER", "TEAM"]
            and not (
                subject
                and str(subject).strip()
            )
        ):
            st.error(
                "Player / Team is required."
            )
            return

        if scope == "OUTRIGHT":
            if not (
                subject
                and str(subject).strip()
            ):
                st.error(
                    "Outright selection "
                    "is required."
                )
                return

            if (
                outright_needs_second_selection(
                    market,
                    sport
                )
                and not (
                    selection_2
                    and str(selection_2).strip()
                )
            ):
                st.error(
                    "The second selection "
                    "is required."
                )
                return

        if (
            origin == "TIPSTER"
            and tipster_id is None
        ):
            st.error(
                "Select or create a tipster."
            )
            return

        if (
            primary_reason == "Select reason..."
            and (
                origin == "SELF"
                or (
                    origin == "TIPSTER"
                    and has_own_reasoning
                )
            )
        ):
            st.error(
                "Select a Primary Reason."
            )
            return

        metrics = calculate_metrics(
            market_odds,
            my_odds,
            tipster_posted_odds
        )

        record = {
            "user_id": st.session_state.user_id,
            "bet_date": bet_date.isoformat(),
            "is_live": bool(is_live),
            "sport": sport,
            "league": league,
            "event": str(event).strip(),
            "scope": scope,
            "subject": (
                str(subject).strip()
                if subject
                else None
            ),
            "selection_2": (
                str(selection_2).strip()
                if selection_2
                else None
            ),
            "market": market,
            "period": period,
            "side": side,
            "line": line,
            "bookmaker": bookmaker,
            "market_odds": market_odds,
            "my_odds": my_odds,
            "origin": origin,
            "tipster_id": tipster_id,
            "tipster_posted_odds": tipster_posted_odds,
            "confidence": confidence,
            "has_own_reasoning": has_own_reasoning,
            "primary_reason": (
                None
                if primary_reason == "Select reason..."
                else primary_reason
            ),
            "secondary_reason": (
                None
                if secondary_reason in [None, "None"]
                else secondary_reason
            ),
            "stake": stake,
            "result": "Pending",
            "p_market": metrics["p_market"],
            "p_you": metrics["p_you"],
            "edge_pp": metrics["edge_pp"],
            "ev_pct": metrics["ev_pct"],
            "price_deterioration_pp": (
                metrics[
                    "price_deterioration_pp"
                ]
            ),
            "profit": 0,
            "notes": (
                notes.strip()
                if notes.strip()
                else None
            )
        }

        try:
            response = (
                supabase
                .table("bets")
                .insert(record)
                .execute()
            )

            if response.data:
                _remember_entry_value(
                    "leagues",
                    league,
                    sport=sport
                )

                _remember_entry_value(
                    "markets",
                    market,
                    sport=sport,
                    scope=scope
                )

                if scope == "OUTRIGHT":
                    _remember_entry_value(
                        "outright_events",
                        event,
                        sport=sport
                    )

                    if sport == "Tennis":
                        _remember_entry_value(
                            "players",
                            subject,
                            sport=sport
                        )
                        _remember_entry_value(
                            "players",
                            selection_2,
                            sport=sport
                        )

                    elif (
                        sport == "Football"
                        and market in [
                            "Top Goalscorer",
                            "Top Assists"
                        ]
                    ):
                        _remember_entry_value(
                            "players",
                            subject,
                            sport=sport
                        )

                    elif (
                        sport == "Basketball"
                        and market.startswith("Top ")
                    ):
                        _remember_entry_value(
                            "players",
                            subject,
                            sport=sport
                        )

                        if market.endswith(" - Team"):
                            _remember_entry_value(
                                "teams",
                                selection_2,
                                sport=sport
                            )

                    else:
                        _remember_entry_value(
                            "teams",
                            subject,
                            sport=sport
                        )
                        _remember_entry_value(
                            "teams",
                            selection_2,
                            sport=sport
                        )

                else:
                    _remember_entry_value(
                        "regular_events",
                        event,
                        sport=sport
                    )

                    if scope == "PLAYER":
                        _remember_entry_value(
                            "players",
                            subject,
                            sport=sport
                        )

                    elif scope == "TEAM":
                        _remember_entry_value(
                            "teams",
                            subject,
                            sport=sport
                        )

                st.success(
                    "✅ Bet saved successfully!"
                )

                st.write(
                    f"Total Bets: "
                    f"{get_total_bets_count()}"
                )

        except Exception as e:
            st.error(
                f"Could not save bet: {e}"
            )





# ==========================================
# PENDING
# ==========================================

def render_pending_group(
    bets,
    tipster_map,
    shared_source_ids=None
):

    shared_source_ids = shared_source_ids or set()

    if not bets:

        st.info(
            "No pending bets "
            "in this category."
        )

        return


    for bet in bets:

        st.divider()

        if _is_abuse_bet(bet):
            render_pending_abuse(bet)
            continue


        scope = bet.get("scope")

        event_text = (
            bet.get("event")
            or (
                "Outright"
                if scope == "OUTRIGHT"
                else "Unknown Event"
            )
        )

        subject_text = (
            bet.get("subject")
            or ""
        )

        selection_text = (
            format_bet_selection(
                bet
            )
        )


        # Always show the event first so a player/team
        # pending bet is immediately identifiable.
        st.subheader(
            event_text
        )


        if scope == "PLAYER":

            st.write(
                f"👤 **Player:** "
                f"{subject_text or '—'}"
            )


        elif scope == "TEAM":

            st.write(
                f"🛡️ **Team:** "
                f"{subject_text or '—'}"
            )


        elif scope == "OUTRIGHT":

            if selection_text:

                st.write(
                    f"🏆 **Selection:** "
                    f"{selection_text}"
                )


        market_text = (
            f"📈 **Market:** "
            f"{bet.get('market') or '—'}"
        )


        if (
            scope != "OUTRIGHT"
            and selection_text
        ):

            market_text += (
                f" | {selection_text}"
            )


        st.write(
            market_text
        )


        if bet["scope"] == "OUTRIGHT":

            st.caption(
                f"{bet.get('sport') or DEFAULT_SPORT} | "
                f"🏆 {bet['league']} | "
                f"{bet['bookmaker']} | "
                f"{'🔴 Live' if bet.get('is_live') else 'Pre-live'}"
            )

        else:

            st.caption(
                f"{bet.get('sport') or DEFAULT_SPORT} | "
                f"{bet['league']} | "
                f"{bet['period']} | "
                f"{bet['bookmaker']} | "
                f"{'🔴 Live' if bet.get('is_live') else 'Pre-live'}"
            )


        # Explicit sharing only: personal bets remain private unless the
        # owner presses Share. A copied shared pick is already visible to the
        # group, so resharing it is unnecessary.
        if bet.get("origin") != "SHARED":
            source_id = str(bet.get("id"))
            is_shared = source_id in shared_source_ids

            share_col, unshare_col = st.columns(2)

            with share_col:
                share_label = (
                    "🔄 Update shared pick"
                    if is_shared
                    else "📤 Share Pick"
                )
                if st.button(
                    share_label,
                    key=f"share_pending_{bet['id']}",
                    use_container_width=True
                ):
                    try:
                        source_tipster_name = (
                            tipster_map.get(bet.get("tipster_id"))
                            if bet.get("origin") == "TIPSTER"
                            else None
                        )
                        share_pending_bet(
                            bet,
                            tipster_name=source_tipster_name
                        )
                        st.success("Shared with the group.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not share pick: {e}")

            with unshare_col:
                if is_shared:
                    if st.button(
                        "🙈 Unshare",
                        key=f"unshare_pending_{bet['id']}",
                        use_container_width=True
                    ):
                        unshare_pending_bet(bet["id"])
                        st.rerun()
                else:
                    st.caption("Private until you share it.")

        else:
            shared_name = _shared_display_name(
                bet.get("shared_from_email")
            )
            st.caption(f"👥 Added from {shared_name}'s shared picks")


        if _combo_is_bet(bet):

            combo_legs = bet.get("combo_legs") or []
            flat_count = _combo_selection_count(combo_legs)

            combo_caption = f"🧩 {flat_count} underlying selections"
            profile = _combo_profile(combo_legs)
            if profile:
                combo_caption += f" | {profile}"
            st.caption(combo_caption)

            with st.expander(
                "🧩 Selections / Flat-bet results"
            ):

                edited_combo = copy.deepcopy(combo_legs)
                result_options = [
                    "Pending",
                    "Win",
                    "Loss",
                    "Void"
                ]

                for component_index, component in enumerate(edited_combo):

                    if component.get("kind") in ["SINGLE", "OUTRIGHT"]:
                        component_title = (
                            "Outright"
                            if component.get("kind") == "OUTRIGHT"
                            else "Leg"
                        )
                        st.write(
                            f"**{component_title} {component_index + 1}:** "
                            f"{component.get('label') or '—'} "
                            f"@{float(component.get('odds') or 0):.2f}"
                        )

                        current_result = (
                            component.get("result")
                            or "Pending"
                        )

                        component["result"] = st.selectbox(
                            "Result",
                            result_options,
                            index=safe_index(
                                result_options,
                                current_result
                            ),
                            key=(
                                f"combo_leg_result_{bet['id']}_"
                                f"{component_index}"
                            )
                        )

                    elif component.get("kind") == "BET_BUILDER":
                        st.write(
                            f"**BB {component_index + 1}:** "
                            f"{component.get('label') or 'Bet Builder'}"
                        )

                        for selection_index, selection in enumerate(
                            component.get("selections", []) or []
                        ):
                            c1, c2 = st.columns([3, 2])

                            with c1:
                                st.write(
                                    f"{selection_index + 1}. "
                                    f"{selection.get('label') or '—'} "
                                    f"@{float(selection.get('odds') or 0):.2f}"
                                )

                            with c2:
                                current_result = (
                                    selection.get("result")
                                    or "Pending"
                                )

                                selection["result"] = st.selectbox(
                                    "Result",
                                    result_options,
                                    index=safe_index(
                                        result_options,
                                        current_result
                                    ),
                                    key=(
                                        f"combo_leg_result_{bet['id']}_"
                                        f"{component_index}_{selection_index}"
                                    ),
                                    label_visibility="collapsed"
                                )

                save_col, all_win_col = st.columns(2)

                with save_col:
                    if st.button(
                        "💾 Save leg results",
                        key=f"save_combo_results_{bet['id']}",
                        use_container_width=True
                    ):
                        update_combo_legs(
                            bet["id"],
                            edited_combo
                        )
                        st.success("Leg results saved.")
                        st.rerun()

                with all_win_col:
                    if st.button(
                        "✅ Mark all Win",
                        key=f"combo_all_win_{bet['id']}",
                        use_container_width=True
                    ):
                        all_win_combo = copy.deepcopy(combo_legs)

                        for component in all_win_combo:
                            if component.get("kind") in ["SINGLE", "OUTRIGHT"]:
                                component["result"] = "Win"
                            elif component.get("kind") == "BET_BUILDER":
                                for selection in component.get(
                                    "selections", []
                                ) or []:
                                    selection["result"] = "Win"

                        update_combo_legs(
                            bet["id"],
                            all_win_combo
                        )
                        st.success("All leg results marked Win.")
                        st.rerun()


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Odds",
                f"{float(bet['market_odds']):.2f}"
            )


        with c2:

            st.metric(
                "Stake",
                f"{float(bet['stake']):.2f}"
            )


        with c3:

            st.metric(
                "Date",
                bet["bet_date"]
            )


        if bet["origin"] == "SELF":

            if bet["my_odds"] is not None:

                st.caption(
                    f"My Fair Odds: "
                    f"{float(bet['my_odds']):.2f} | "
                    f"EV: "
                    f"{float(bet['ev_pct']):.2f}%"
                )


        elif bet["origin"] == "TIPSTER":

            tipster_name = (
                tipster_map.get(
                    bet["tipster_id"],
                    "Unknown Tipster"
                )
            )

            st.caption(
                f"Tipster: "
                f"{tipster_name}"
            )


        elif bet["origin"] == "SHARED":

            st.caption(
                "Shared by: "
                f"{_shared_display_name(bet.get('shared_from_email'))}"
            )


        # ==================================
        # RESULT BUTTONS
        # ==================================

        win_col, loss_col, void_col = (
            st.columns(3)
        )


        with win_col:

            if st.button(
                "✅ WIN",
                key=f"win_{bet['id']}",
                use_container_width=True
            ):

                if _combo_is_bet(bet):
                    all_win_combo = _combo_mark_all_results(
                        bet.get("combo_legs") or [],
                        "Win"
                    )
                    update_combo_legs(
                        bet["id"],
                        all_win_combo
                    )

                settle_bet(
                    bet["id"],
                    "Win",
                    bet["stake"],
                    bet["market_odds"]
                )

                st.session_state.pop(
                    f"combo_loss_review_{bet['id']}",
                    None
                )
                st.rerun()


        with loss_col:

            if st.button(
                "❌ LOSS",
                key=f"loss_{bet['id']}",
                use_container_width=True
            ):

                if (
                    _combo_is_bet(bet)
                    and _combo_has_pending_results(
                        bet.get("combo_legs") or []
                    )
                ):
                    st.session_state[
                        f"combo_loss_review_{bet['id']}"
                    ] = True
                else:
                    settle_bet(
                        bet["id"],
                        "Loss",
                        bet["stake"],
                        bet["market_odds"]
                    )
                    st.session_state.pop(
                        f"combo_loss_review_{bet['id']}",
                        None
                    )
                    st.rerun()


        with void_col:

            if st.button(
                "↩️ PUSH / VOID",
                key=f"void_{bet['id']}",
                use_container_width=True
            ):

                settle_bet(
                    bet["id"],
                    "Void",
                    bet["stake"],
                    bet["market_odds"]
                )

                st.rerun()


        if (
            _combo_is_bet(bet)
            and st.session_state.get(
                f"combo_loss_review_{bet['id']}",
                False
            )
        ):
            current_combo_legs = bet.get("combo_legs") or []

            if _combo_has_pending_results(current_combo_legs):
                st.warning(
                    "⚠️ Before settling this BB/Parlay as LOSS, open "
                    "'Selections / Flat-bet results' above, mark which "
                    "underlying selections won/lost/voided, and save them. "
                    "This keeps the flat-bet post-analysis correct."
                )
            else:
                st.success(
                    "✅ All underlying selections have results. "
                    "You can now confirm the combo loss."
                )
                if st.button(
                    "❌ CONFIRM COMBO LOSS",
                    key=f"confirm_combo_loss_{bet['id']}",
                    use_container_width=True,
                    type="primary"
                ):
                    settle_bet(
                        bet["id"],
                        "Loss",
                        bet["stake"],
                        bet["market_odds"]
                    )
                    st.session_state.pop(
                        f"combo_loss_review_{bet['id']}",
                        None
                    )
                    st.rerun()


        # ==================================
        # CASHOUT
        # ==================================

        with st.expander(
            "💰 Cash Out"
        ):

            cashout_return = (
                st.number_input(
                    "Cashout Return",
                    min_value=0.00,
                    value=float(
                        bet["stake"]
                    ),
                    step=0.50,
                    format="%.2f",
                    key=(
                        f"cashout_return_"
                        f"{bet['id']}"
                    )
                )
            )


            cashout_profit = (
                float(cashout_return)
                - float(bet["stake"])
            )


            st.caption(
                f"Cashout P/L: "
                f"{cashout_profit:+.2f}"
            )


            if st.button(
                "💰 CONFIRM CASH OUT",
                key=f"cashout_{bet['id']}",
                use_container_width=True,
                type="primary"
            ):

                try:

                    settle_cashout(
                        bet["id"],
                        bet["stake"],
                        cashout_return
                    )

                    st.rerun()


                except Exception as e:

                    st.error(
                        f"Could not cash out "
                        f"bet: {e}"
                    )


def pending_bets_page():

    bets = load_pending_bets()


    regular_bets = [
        bet
        for bet in bets
        if (
            bet.get("scope") != "OUTRIGHT"
            and not _is_outright_parlay(bet)
        )
    ]


    outright_bets = [
        bet
        for bet in bets
        if (
            bet.get("scope") == "OUTRIGHT"
            or _is_outright_parlay(bet)
        )
    ]

    single_outright_bets = [
        bet for bet in outright_bets
        if not _is_outright_parlay(bet)
    ]

    outright_parlay_bets = [
        bet for bet in outright_bets
        if _is_outright_parlay(bet)
    ]


    st.header(
        f"⏳ Pending Bets "
        f"({len(bets)})"
    )


    if not bets:

        st.success(
            "No pending bets 🎉"
        )

        return


    tipsters = load_tipsters()

    shared_source_ids = load_my_active_shared_source_ids()


    tipster_map = {
        t["id"]:
            t["name"]
        for t in tipsters
    }


    regular_tab, outright_tab = (
        st.tabs([
            (
                f"🎯 Regular "
                f"({len(regular_bets)})"
            ),
            (
                f"🏆 Outrights "
                f"({len(outright_bets)})"
            )
        ])
    )


    with regular_tab:

        render_pending_group(
            regular_bets,
            tipster_map,
            shared_source_ids
        )


    with outright_tab:

        single_out_tab, parlay_out_tab = st.tabs([
            f"Single Outrights ({len(single_outright_bets)})",
            f"Outright Parlays ({len(outright_parlay_bets)})"
        ])

        with single_out_tab:
            render_pending_group(
                single_outright_bets,
                tipster_map,
                shared_source_ids
            )

        with parlay_out_tab:
            render_pending_group(
                outright_parlay_bets,
                tipster_map,
                shared_source_ids
            )


# ==========================================
# SHARED PICKS PAGE
# ==========================================

def _render_shared_combo_preview(combo_legs):
    for component_index, component in enumerate(combo_legs or []):
        if not isinstance(component, dict):
            continue

        kind = component.get("kind")

        if kind == "SINGLE":
            st.write(
                f"• {component.get('label') or 'Selection'} "
                f"@{float(component.get('odds') or 0):.2f}"
            )

        elif kind == "OUTRIGHT":
            st.write(
                f"• 🏆 {component.get('label') or 'Outright'} "
                f"@{float(component.get('odds') or 0):.2f}"
            )

        elif kind == "BET_BUILDER":
            label = component.get("label") or f"Bet Builder {component_index + 1}"
            st.write(f"• 🧩 **{label}**")
            for selection in component.get("selections", []) or []:
                st.caption(
                    f"   ↳ {selection.get('label') or 'Selection'} "
                    f"@{float(selection.get('odds') or 0):.2f}"
                )


def shared_picks_page():
    st.header("👥 Shared Picks")
    st.caption(
        "Only bets that a user explicitly shares appear here. "
        "Adding one creates your own independent Pending bet; "
        "the original user's bet stays private and unchanged."
    )

    picks = load_shared_picks()

    if not picks:
        st.info("No active shared picks right now.")
        return

    copied_ids = load_my_copied_shared_pick_ids()

    from_others = [
        pick for pick in picks
        if pick.get("owner_user_id") != st.session_state.user_id
    ]
    mine = [
        pick for pick in picks
        if pick.get("owner_user_id") == st.session_state.user_id
    ]

    others_tab, mine_tab = st.tabs([
        f"📥 From others ({len(from_others)})",
        f"📤 My shared ({len(mine)})"
    ])

    with others_tab:
        if not from_others:
            st.info("Nobody else has shared a pending pick yet.")

        for pick in from_others:
            snapshot = pick.get("bet_snapshot") or {}
            st.divider()

            event_text = snapshot.get("event") or "Shared Pick"
            st.subheader(event_text)

            selection_text = format_bet_selection(snapshot)
            market_text = snapshot.get("market") or "—"
            if snapshot.get("scope") == "OUTRIGHT":
                st.write(f"🏆 **{selection_text or snapshot.get('subject') or 'Outright'}**")
                st.write(f"📈 **Market:** {market_text}")
            else:
                st.write(
                    f"📈 **Market:** {market_text}"
                    + (f" | {selection_text}" if selection_text else "")
                )
                if snapshot.get("subject"):
                    st.write(f"👤/🛡️ **Subject:** {snapshot.get('subject')}")

            shared_name = _shared_display_name(pick.get("owner_email"))
            original_odds = float(snapshot.get("market_odds") or 1.01)
            st.caption(
                f"Shared by {shared_name} | "
                f"{snapshot.get('sport') or DEFAULT_SPORT} | "
                f"{snapshot.get('league') or '—'} | "
                f"Shared odds {original_odds:.2f} | "
                f"Confidence {snapshot.get('confidence') or 'Medium'}"
            )

            combo_legs = snapshot.get("combo_legs") or []
            if combo_legs:
                with st.expander("🧩 View selections"):
                    _render_shared_combo_preview(combo_legs)

            pick_id = str(pick.get("id"))
            already_added = pick_id in copied_ids

            if already_added:
                st.success("✅ Already added to your tracker.")
                continue

            c1, c2 = st.columns(2)
            with c1:
                bookmaker_options = list(BOOKMAKERS)
                if "Stoiximan" not in bookmaker_options:
                    bookmaker_options.insert(0, "Stoiximan")
                chosen_bookmaker = st.selectbox(
                    "My bookmaker",
                    bookmaker_options,
                    index=safe_index(bookmaker_options, "Stoiximan", 0),
                    key=f"shared_bookmaker_{pick_id}"
                )

                chosen_odds = st.number_input(
                    "My odds taken",
                    min_value=1.01,
                    value=original_odds,
                    step=0.01,
                    format="%.2f",
                    key=f"shared_odds_{pick_id}"
                )

            with c2:
                chosen_stake = st.number_input(
                    "My stake",
                    min_value=0.01,
                    value=10.00,
                    step=1.00,
                    format="%.2f",
                    key=f"shared_stake_{pick_id}"
                )

                confidence_options = ["Low", "Medium", "High"]
                chosen_confidence = st.selectbox(
                    "My confidence",
                    confidence_options,
                    index=1,
                    key=f"shared_confidence_{pick_id}"
                )

            if st.button(
                "➕ Add to my Pending",
                key=f"copy_shared_{pick_id}",
                type="primary",
                use_container_width=True
            ):
                try:
                    add_shared_pick_to_my_pending(
                        pick,
                        chosen_bookmaker,
                        chosen_odds,
                        chosen_stake,
                        chosen_confidence
                    )
                    st.success("Added to your Pending bets.")
                    st.rerun()
                except Exception as e:
                    message = str(e)
                    if "duplicate" in message.lower() or "unique" in message.lower():
                        st.info("You have already added this shared pick.")
                    else:
                        st.error(f"Could not add shared pick: {e}")

    with mine_tab:
        if not mine:
            st.info("You have not shared any pending picks.")

        for pick in mine:
            snapshot = pick.get("bet_snapshot") or {}
            st.divider()
            st.write(
                f"**{snapshot.get('event') or 'Shared Pick'}** — "
                f"{snapshot.get('market') or '—'}"
            )
            st.caption(
                f"{snapshot.get('sport') or DEFAULT_SPORT} | "
                f"{snapshot.get('league') or '—'} | "
                f"Odds {float(snapshot.get('market_odds') or 1.01):.2f}"
            )

            if st.button(
                "🙈 Unshare",
                key=f"shared_page_unshare_{pick.get('id')}",
                use_container_width=True
            ):
                try:
                    (
                        supabase
                        .table("shared_picks")
                        .update({
                            "is_active": False,
                            "updated_at": now_utc()
                        })
                        .eq("id", pick.get("id"))
                        .eq("owner_user_id", st.session_state.user_id)
                        .execute()
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not unshare pick: {e}")


# ==========================================
# HISTORY
# ==========================================

def history_page():

    st.header(
        "📜 Bet History"
    )


    history = load_history_bets()


    if not history:

        st.info(
            "No settled bets yet."
        )

        return


    sports = sorted(
        list(
            set(
                (
                    bet.get("sport")
                    or DEFAULT_SPORT
                )
                for bet in history
            )
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        sport_filter = st.selectbox(
            "Sport",
            ["All"] + sports,
            key="history_sport"
        )


    with col2:

        result_filter = st.selectbox(
            "Result",
            [
                "All",
                "Win",
                "Loss",
                "Cashout",
                "Void"
            ],
            key="history_result"
        )


    timing_filter = st.selectbox(
        "Bet Timing",
        [
            "All",
            "Pre-live",
            "Live"
        ],
        key="history_timing"
    )


    scope_source = [
        bet
        for bet in history
        if (
            sport_filter == "All"
            or (
                bet.get("sport")
                or DEFAULT_SPORT
            )
            == sport_filter
        )
    ]


    scopes = sorted(
        list(
            set(
                bet["scope"]
                for bet in scope_source
                if bet.get("scope")
            )
        )
    )


    scope_filter = st.selectbox(
        "Bet Type",
        ["All"] + scopes,
        key="history_scope"
    )


    league_source = [
        bet
        for bet in scope_source
        if (
            scope_filter == "All"
            or bet["scope"]
            == scope_filter
        )
    ]


    leagues = sorted(
        list(
            set(
                bet["league"]
                for bet in league_source
                if bet["league"]
            )
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        league_filter = st.selectbox(
            "League",
            ["All"] + leagues,
            key="history_league"
        )


    with col2:

        origin_filter = st.selectbox(
            "Origin",
            [
                "All",
                "SELF",
                "TIPSTER",
                "SHARED"
            ],
            key="history_origin"
        )


    filtered = history.copy()


    if sport_filter != "All":

        filtered = [
            bet
            for bet in filtered
            if (
                bet.get("sport")
                or DEFAULT_SPORT
            )
            == sport_filter
        ]


    if scope_filter != "All":

        filtered = [
            bet
            for bet in filtered
            if bet["scope"]
            == scope_filter
        ]


    if result_filter != "All":

        filtered = [
            bet
            for bet in filtered
            if bet["result"]
            == result_filter
        ]


    if league_filter != "All":

        filtered = [
            bet
            for bet in filtered
            if bet["league"]
            == league_filter
        ]


    if origin_filter != "All":

        filtered = [
            bet
            for bet in filtered
            if bet["origin"]
            == origin_filter
        ]


    if timing_filter == "Live":

        filtered = [
            bet
            for bet in filtered
            if bool(
                bet.get("is_live", False)
            )
        ]


    elif timing_filter == "Pre-live":

        filtered = [
            bet
            for bet in filtered
            if not bool(
                bet.get("is_live", False)
            )
        ]


    performance = [
        bet
        for bet in filtered
        if bet["result"]
        in [
            "Win",
            "Loss",
            "Cashout"
        ]
    ]


    total_profit = sum(
        float(
            bet["profit"]
            or 0
        )
        for bet in performance
    )


    total_stake = sum(
        float(
            bet["stake"]
            or 0
        )
        for bet in performance
    )


    roi = (
        total_profit
        / total_stake
        * 100
        if total_stake
        else 0
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "Bets",
            len(filtered)
        )


    with c2:

        st.metric(
            "Stake",
            f"{total_stake:.2f}"
        )


    with c3:

        st.metric(
            "Profit",
            f"{total_profit:+.2f}"
        )


    with c4:

        st.metric(
            "ROI",
            f"{roi:+.2f}%"
        )


    rows = []


    for bet in filtered:

        selection = (
            format_bet_selection(
                bet
            )
        )


        if bet["scope"] == "OUTRIGHT":

            subject_display = (
                selection
            )

        elif bet["scope"] == "MATCH":

            subject_display = (
                bet["event"]
            )

        else:

            subject_display = (
                bet["subject"]
            )


        rows.append({

            "Date":
                bet["bet_date"],

            "Sport":
                (
                    bet.get("sport")
                    or DEFAULT_SPORT
                ),

            "Timing":
                (
                    "Live"
                    if bool(
                        bet.get(
                            "is_live",
                            False
                        )
                    )
                    else "Pre-live"
                ),

            "League":
                bet["league"],

            "Type":
                bet["scope"],

            "Event / Competition":
                bet["event"],

            "Subject / Selection":
                subject_display,

            "Market":
                bet["market"],

            "Selection":
                (
                    selection
                    if bet["scope"]
                    != "OUTRIGHT"
                    else ""
                ),

            "Odds":
                (
                    None
                    if _is_abuse_bet(bet)
                    else float(bet["market_odds"])
                ),

            "Stake":
                float(
                    bet["stake"]
                ),

            "Origin":
                bet["origin"],

            "Confidence":
                bet["confidence"],

            "Result":
                bet["result"],

            "Cashout Return":
                (
                    float(
                        bet[
                            "cashout_return"
                        ]
                    )
                    if bet.get(
                        "cashout_return"
                    ) is not None
                    else None
                ),

            "Profit":
                float(
                    bet["profit"]
                )
        })


    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# MANAGE / EDIT
# ==========================================

def manage_bets_page():

    st.header(
        "✏️ Manage Bets"
    )


    bets = load_active_bets()


    if not bets:

        st.info(
            "No active bets."
        )

        return


    # Keep Manage Bets usable even with a large history: choose the date first.
    available_dates = sorted(
        {bet.get("bet_date") for bet in bets if bet.get("bet_date")},
        reverse=True
    )

    date_options = ["All dates"] + available_dates
    default_date_index = 1 if available_dates else 0

    selected_manage_date = st.selectbox(
        "Bet Date",
        date_options,
        index=default_date_index,
        key="manage_bet_date_filter"
    )

    if selected_manage_date != "All dates":
        bets = [
            bet for bet in bets
            if bet.get("bet_date") == selected_manage_date
        ]

    # Optional structure filter so Manage Bets does not become cluttered.
    # Outrights are kept separate from normal Singles, while BB/Parlay are
    # identified by their top-level combo market. Multiple types can be viewed
    # together when useful.
    manage_type_options = ["Single", "Bet Builder", "Parlay", "Outright", "Abuse"]
    selected_manage_types = st.multiselect(
        "Bet Type",
        manage_type_options,
        default=manage_type_options,
        key="manage_bet_type_filter"
    )

    def _manage_bet_types(bet):
        if _is_abuse_bet(bet):
            return {"Abuse"}
        if bet.get("market") == "Bet Builder" and _combo_is_bet(bet):
            return {"Bet Builder"}
        if bet.get("market") == "Parlay" and _combo_is_bet(bet):
            types = {"Parlay"}
            combo_legs = bet.get("combo_legs") or []
            if any(
                isinstance(component, dict)
                and str(component.get("kind") or "").upper() == "OUTRIGHT"
                for component in combo_legs
            ):
                types.add("Outright")
            return types
        if bet.get("scope") == "OUTRIGHT":
            return {"Outright"}
        return {"Single"}

    if selected_manage_types:
        selected_manage_types = set(selected_manage_types)
        bets = [
            bet for bet in bets
            if _manage_bet_types(bet) & selected_manage_types
        ]
    else:
        bets = []

    if "Outright" in selected_manage_types:
        outright_subtype_options = [
            "Single Outright",
            "Outright Parlay"
        ]
        selected_outright_subtypes = st.multiselect(
            "Outright Type",
            outright_subtype_options,
            default=outright_subtype_options,
            key="manage_outright_type_filter"
        )

        def _manage_outright_subtype(bet):
            if _is_outright_parlay(bet):
                return "Outright Parlay"
            if bet.get("scope") == "OUTRIGHT":
                return "Single Outright"
            return None

        if selected_outright_subtypes:
            selected_outright_subtypes = set(
                selected_outright_subtypes
            )
            bets = [
                bet for bet in bets
                if (
                    _manage_outright_subtype(bet) is None
                    or _manage_outright_subtype(bet)
                    in selected_outright_subtypes
                )
            ]
        else:
            bets = [
                bet for bet in bets
                if _manage_outright_subtype(bet) is None
            ]

    st.caption(
        f"Showing {len(bets)} bet{'s' if len(bets) != 1 else ''}"
        + (
            f" from {selected_manage_date}."
            if selected_manage_date != "All dates"
            else " across all dates."
        )
    )

    if not bets:
        st.info("No bets match the selected date/type filters.")
        return


    label_map = {}


    for bet in bets:

        if _is_abuse_bet(bet):
            abuse_data = bet.get("abuse_data") or {}
            abuse_kind = (
                "Casino"
                if abuse_data.get("category") == "CASINO"
                else abuse_data.get("match_format") or "Sports"
            )
            label = (
                f"{bet['bet_date']} | 🧪 {bet.get('event') or 'Abuse'} | "
                f"{abuse_kind} | {bet.get('result') or 'Pending'} | "
                f"P/L {float(bet.get('profit') or 0):+.2f}"
            )

        elif bet["scope"] == "OUTRIGHT":

            description = (
                format_bet_selection(
                    bet
                )
            )

            label = (
                f"{bet['bet_date']} | "
                f"🏆 {bet['event']} | "
                f"{bet['market']} | "
                f"{description} | "
                f"@{float(bet['market_odds']):.2f} | "
                f"{bet['result']}"
            )


        else:

            subject = (
                bet["event"]
                if bet["scope"] == "MATCH"
                else bet["subject"]
            )


            selection = (
                format_bet_selection(
                    bet
                )
            )


            label = (
                f"{bet['bet_date']} | "
                f"{subject} | "
                f"{bet['market']} "
                f"{selection} | "
                f"@{float(bet['market_odds']):.2f} | "
                f"{bet['result']}"
            )


        label_map[label] = bet


    selected_label = (
        st.selectbox(
            "Choose Bet",
            list(
                label_map.keys()
            )
        )
    )


    bet = label_map[
        selected_label
    ]


    bet_id = bet["id"]

    # ======================================
    # ABUSE EDITOR
    # ======================================

    if _is_abuse_bet(bet):
        render_manage_abuse(bet)
        return

    # ======================================
    # SIMPLE COMBO EDITOR
    # ======================================

    if _combo_is_bet(bet):

        st.divider()
        st.subheader(
            f"Edit {bet.get('market') or 'Combo Bet'}"
        )

        edit_date = st.date_input(
            "Bet Date",
            value=datetime.strptime(
                bet["bet_date"],
                "%Y-%m-%d"
            ).date(),
            key=f"edit_combo_date_{bet_id}"
        )

        edit_live = st.checkbox(
            "🔴 Live Bet",
            value=bool(bet.get("is_live", False)),
            key=f"edit_combo_live_{bet_id}"
        )

        bookmaker_options = list(BOOKMAKERS)
        if bet.get("bookmaker") not in bookmaker_options:
            bookmaker_options.append(bet.get("bookmaker"))

        edit_bookmaker = st.selectbox(
            "Bookmaker",
            bookmaker_options,
            index=safe_index(
                bookmaker_options,
                bet.get("bookmaker")
            ),
            key=f"edit_combo_bookmaker_{bet_id}"
        )

        c1, c2 = st.columns(2)
        with c1:
            edit_final_odds = st.number_input(
                "Final odds taken",
                min_value=1.01,
                value=float(bet.get("market_odds") or 1.01),
                step=0.01,
                format="%.2f",
                key=f"edit_combo_odds_{bet_id}"
            )
        with c2:
            edit_stake = st.number_input(
                "Stake",
                min_value=0.01,
                value=float(bet.get("stake") or 0.01),
                step=1.00,
                key=f"edit_combo_stake_{bet_id}"
            )

        edit_notes = st.text_area(
            "Notes",
            value=bet.get("notes") or "",
            key=f"edit_combo_notes_{bet_id}"
        )

        # Origin / Tipster can be changed for Bet Builders and Parlays.
        origin_options = ["SELF", "TIPSTER", "SHARED"]
        edit_origin = st.radio(
            "Origin",
            origin_options,
            index=safe_index(origin_options, bet.get("origin") or "SELF"),
            horizontal=True,
            key=f"edit_combo_origin_{bet_id}"
        )

        edit_tipster_id = None
        edit_tipster_posted_odds = None
        edit_my_odds = None

        if edit_origin == "TIPSTER":
            tipsters = load_tipsters()
            tipster_map = {t["name"]: t["id"] for t in tipsters}
            tipster_names = list(tipster_map.keys())
            current_tipster_name = next(
                (name for name, tid in tipster_map.items() if tid == bet.get("tipster_id")),
                None
            )

            if not tipster_names:
                st.warning("No saved tipsters found. Add one from a normal Add Bet entry first.")
            else:
                edit_tipster_name = st.selectbox(
                    "Tipster",
                    tipster_names,
                    index=safe_index(tipster_names, current_tipster_name, 0),
                    key=f"edit_combo_tipster_{bet_id}"
                )
                edit_tipster_id = tipster_map[edit_tipster_name]

            know_posted = st.checkbox(
                "I know the tipster's posted odds",
                value=bet.get("tipster_posted_odds") is not None,
                key=f"edit_combo_has_tipster_odds_{bet_id}"
            )
            if know_posted:
                edit_tipster_posted_odds = st.number_input(
                    "Tipster Posted Odds",
                    min_value=1.01,
                    value=float(bet.get("tipster_posted_odds") or edit_final_odds),
                    step=0.01,
                    format="%.2f",
                    key=f"edit_combo_tipster_odds_{bet_id}"
                )
        elif edit_origin == "SHARED":
            st.info(
                "This bet came from "
                f"{_shared_display_name(bet.get('shared_from_email'))}'s shared picks. "
                "You can leave it as SHARED or convert it to SELF/TIPSTER."
            )
        else:
            edit_my_odds = st.number_input(
                "My Fair Odds",
                min_value=1.01,
                value=float(bet.get("my_odds") or 1.80),
                step=0.01,
                format="%.2f",
                key=f"edit_combo_my_odds_{bet_id}"
            )

        combo_legs = bet.get("combo_legs") or []
        st.caption(
            f"Components: {_combo_component_count(combo_legs)} | "
            f"Underlying selections: {_combo_selection_count(combo_legs)}"
        )

        with st.expander("🧩 Edit leg results"):
            edited_combo = copy.deepcopy(combo_legs)
            result_options = ["Pending", "Win", "Loss", "Void"]

            for component_index, component in enumerate(edited_combo):
                if component.get("kind") in ["SINGLE", "OUTRIGHT"]:
                    component_title = (
                        "Outright"
                        if component.get("kind") == "OUTRIGHT"
                        else "Leg"
                    )
                    st.write(
                        f"**{component_title} {component_index + 1}:** "
                        f"{component.get('label') or '—'} "
                        f"@{float(component.get('odds') or 0):.2f}"
                    )
                    current_result = component.get("result") or "Pending"
                    component["result"] = st.selectbox(
                        "Result",
                        result_options,
                        index=safe_index(result_options, current_result),
                        key=f"manage_combo_result_{bet_id}_{component_index}"
                    )
                elif component.get("kind") == "BET_BUILDER":
                    st.write(
                        f"**BB {component_index + 1}:** "
                        f"{component.get('label') or 'Bet Builder'}"
                    )
                    for selection_index, selection in enumerate(
                        component.get("selections", []) or []
                    ):
                        c1, c2 = st.columns([3, 2])
                        with c1:
                            st.write(
                                f"{selection_index + 1}. "
                                f"{selection.get('label') or '—'} "
                                f"@{float(selection.get('odds') or 0):.2f}"
                            )
                        with c2:
                            current_result = selection.get("result") or "Pending"
                            selection["result"] = st.selectbox(
                                "Result",
                                result_options,
                                index=safe_index(result_options, current_result),
                                key=(
                                    f"manage_combo_result_{bet_id}_"
                                    f"{component_index}_{selection_index}"
                                ),
                                label_visibility="collapsed"
                            )

            if st.button(
                "💾 Save leg results",
                key=f"manage_save_combo_results_{bet_id}",
                use_container_width=True
            ):
                update_combo_legs(bet_id, edited_combo)
                st.success("Leg results saved.")
                st.rerun()

        save_col, delete_col = st.columns(2)

        with save_col:
            if st.button(
                "💾 SAVE CHANGES",
                type="primary",
                use_container_width=True,
                key=f"save_combo_changes_{bet_id}"
            ):
                if edit_origin == "TIPSTER" and edit_tipster_id is None:
                    st.error("Select an existing tipster before saving.")
                    return

                metrics = calculate_metrics(
                    edit_final_odds,
                    edit_my_odds,
                    edit_tipster_posted_odds
                )

                new_profit = float(bet.get("profit") or 0)
                if bet.get("result") in ["Win", "Loss"]:
                    new_profit = calculate_profit(
                        bet.get("result"),
                        edit_stake,
                        edit_final_odds
                    )

                response = (
                    supabase
                    .table("bets")
                    .update({
                        "bet_date": edit_date.isoformat(),
                        "is_live": bool(edit_live),
                        "bookmaker": edit_bookmaker,
                        "market_odds": float(edit_final_odds),
                        "stake": float(edit_stake),
                        "notes": edit_notes.strip() or None,
                        "origin": edit_origin,
                        "tipster_id": edit_tipster_id if edit_origin == "TIPSTER" else None,
                        "tipster_posted_odds": (
                            float(edit_tipster_posted_odds)
                            if edit_tipster_posted_odds is not None
                            else None
                        ),
                        "my_odds": (
                            float(edit_my_odds)
                            if edit_my_odds is not None
                            else None
                        ),
                        "p_market": metrics["p_market"],
                        "p_you": metrics["p_you"],
                        "edge_pp": metrics["edge_pp"],
                        "ev_pct": metrics["ev_pct"],
                        "price_deterioration_pp": metrics[
                            "price_deterioration_pp"
                        ],
                        "profit": new_profit,
                        "updated_at": now_utc()
                    })
                    .eq("id", bet_id)
                    .eq("user_id", st.session_state.user_id)
                    .execute()
                )

                if response.data:
                    st.success("Combo bet updated.")
                    st.rerun()

        with delete_col:
            if st.button(
                "🗑️ MOVE TO TRASH",
                use_container_width=True,
                key=f"delete_combo_{bet_id}"
            ):
                soft_delete_bet(bet_id)
                st.rerun()

        return

    edit_sport = (
        bet.get("sport")
        or DEFAULT_SPORT
    )


    st.divider()

    st.caption(
        f"Sport: {edit_sport}"
    )

    st.subheader(
        "Edit Bet"
    )


    # ======================================
    # DATE / LEAGUE
    # ======================================

    col1, col2 = st.columns(2)


    with col1:

        edit_date = st.date_input(
            "Bet Date",
            value=datetime.strptime(
                bet["bet_date"],
                "%Y-%m-%d"
            ).date(),
            key=f"edit_date_{bet_id}"
        )


    with col2:

        edit_league_options = (
            load_user_league_options(
                edit_sport
            )
        )

        if (
            bet["league"]
            not in edit_league_options
        ):
            edit_league_options.append(
                bet["league"]
            )

        edit_league = st.selectbox(
            "League / Tour",
            edit_league_options,
            index=safe_index(
                edit_league_options,
                bet["league"]
            ),
            accept_new_options=True,
            key=f"edit_league_{bet_id}"
        )


    edit_is_live = st.checkbox(
        "🔴 Live Bet",
        value=bool(
            bet.get("is_live", False)
        ),
        key=f"edit_is_live_{bet_id}"
    )


    scope_options = (
        get_scope_options(
            edit_sport
        )
    )


    edit_scope = st.radio(
        "Bet Type",
        scope_options,
        index=safe_index(
            scope_options,
            bet["scope"]
        ),
        horizontal=True,
        key=f"edit_scope_{bet_id}"
    )


    edit_event = st.text_input(
        (
            "Competition / Event"
            if edit_scope
            == "OUTRIGHT"
            else "Event"
        ),
        value=bet["event"] or "",
        key=f"edit_event_{bet_id}"
    )


    edit_subject = None
    edit_selection_2 = None
    edit_line = None
    edit_side = None


    # ======================================
    # OUTRIGHT
    # ======================================

    if edit_scope == "OUTRIGHT":

        edit_market_options = (
            load_user_market_options(
                edit_sport,
                "OUTRIGHT"
            )
        )

        if (
            bet["market"]
            not in edit_market_options
        ):
            edit_market_options.append(
                bet["market"]
            )

        edit_market = st.selectbox(
            "Outright Market",
            edit_market_options,
            index=safe_index(
                edit_market_options,
                bet["market"]
            ),
            accept_new_options=True,
            key=f"edit_market_{bet_id}"
        )


        label_1, label_2 = (
            outright_selection_labels(
                edit_market,
                edit_sport
            )
        )


        edit_subject = (
            st.text_input(
                label_1,
                value=(
                    bet["subject"]
                    or ""
                ),
                key=(
                    f"edit_subject_"
                    f"{bet_id}"
                )
            )
        )


        if label_2:

            edit_selection_2 = (
                st.text_input(
                    label_2,
                    value=(
                        bet.get(
                            "selection_2"
                        )
                        or ""
                    ),
                    key=(
                        f"edit_selection2_"
                        f"{bet_id}"
                    )
                )
            )


        edit_period = (
            "Full Competition"
        )


    # ======================================
    # REGULAR
    # ======================================

    else:

        if edit_scope == "PLAYER":

            edit_subject = (
                st.text_input(
                    "Player",
                    value=(
                        bet["subject"]
                        or ""
                    ),
                    key=(
                        f"edit_player_"
                        f"{bet_id}"
                    )
                )
            )


        elif edit_scope == "TEAM":

            edit_subject = (
                st.text_input(
                    "Team",
                    value=(
                        bet["subject"]
                        or ""
                    ),
                    key=(
                        f"edit_team_"
                        f"{bet_id}"
                    )
                )
            )


        edit_market_options = (
            load_user_market_options(
                edit_sport,
                edit_scope
            )
        )

        if (
            bet["market"]
            not in edit_market_options
        ):
            edit_market_options.append(
                bet["market"]
            )


        edit_market = st.selectbox(
            "Market",
            edit_market_options,
            index=safe_index(
                edit_market_options,
                bet["market"]
            ),
            accept_new_options=True,
            key=f"edit_market_{bet_id}"
        )


        edit_period_options = (
            get_periods(
                edit_sport
            )
        )

        if (
            bet["period"]
            and bet["period"]
            not in edit_period_options
        ):
            edit_period_options.append(
                bet["period"]
            )


        edit_period = st.selectbox(
            "Period",
            edit_period_options,
            index=safe_index(
                edit_period_options,
                bet["period"]
            ),
            key=f"edit_period_{bet_id}"
        )


        default_edit_markets = (
            get_default_markets(
                edit_sport,
                edit_scope
            )
        )


        if (
            edit_market
            in default_edit_markets
        ):

            edit_market_style = (
                get_market_style(
                    edit_sport,
                    edit_scope,
                    edit_market
                )
            )

        else:

            edit_market_style = (
                infer_saved_custom_market_format(
                    edit_sport,
                    edit_scope,
                    edit_market
                )
            )

            edit_market_style = {
                "Over / Under":
                    "total",
                "Winner / Selection":
                    "winner",
                "Handicap / Spread":
                    "handicap",
                "Yes / No":
                    "yes_no"
            }.get(
                edit_market_style,
                "total"
            )


        if edit_market_style == "winner":

            side_options = (
                get_winner_side_options(
                    edit_sport,
                    edit_market
                )
            )


            edit_side = st.radio(
                "Selection",
                side_options,
                index=safe_index(
                    side_options,
                    bet["side"]
                ),
                horizontal=True,
                key=f"edit_side_{bet_id}"
            )


        elif edit_market_style == "handicap":

            side_options = (
                get_winner_side_options(
                    edit_sport,
                    edit_market
                )
            )


            edit_side = st.radio(
                "Selection",
                side_options,
                index=safe_index(
                    side_options,
                    bet["side"]
                ),
                horizontal=True,
                key=f"edit_side_{bet_id}"
            )


            edit_line = (
                st.number_input(
                    "Line",
                    value=float(
                        bet["line"]
                        or 0
                    ),
                    step=0.5,
                    format="%.1f",
                    key=(
                        f"edit_line_"
                        f"{bet_id}"
                    )
                )
            )


        elif edit_market_style == "yes_no":

            side_options = [
                "Yes",
                "No"
            ]


            edit_side = st.radio(
                "Selection",
                side_options,
                index=safe_index(
                    side_options,
                    bet["side"]
                ),
                horizontal=True,
                key=f"edit_side_{bet_id}"
            )


        else:

            side_options = [
                "Over",
                "Under"
            ]


            edit_side = st.radio(
                "Side",
                side_options,
                index=safe_index(
                    side_options,
                    bet["side"]
                ),
                horizontal=True,
                key=f"edit_side_{bet_id}"
            )


            edit_line = (
                st.number_input(
                    "Line",
                    value=float(
                        bet["line"]
                        or 0
                    ),
                    step=0.5,
                    format="%.1f",
                    key=(
                        f"edit_line_"
                        f"{bet_id}"
                    )
                )
            )


    # ======================================
    # BOOK / ODDS
    # ======================================

    col1, col2 = st.columns(2)


    with col1:

        edit_bookmaker = (
            st.selectbox(
                "Bookmaker",
                BOOKMAKERS,
                index=safe_index(
                    BOOKMAKERS,
                    bet["bookmaker"]
                ),
                key=f"edit_book_{bet_id}"
            )
        )


    with col2:

        edit_market_odds = (
            st.number_input(
                "Odds Taken",
                min_value=1.01,
                value=float(
                    bet["market_odds"]
                ),
                step=0.01,
                format="%.2f",
                key=f"edit_odds_{bet_id}"
            )
        )


    # ======================================
    # ORIGIN
    # ======================================

    origin_options = [
        "SELF",
        "TIPSTER",
        "SHARED"
    ]


    edit_origin = st.radio(
        "Origin",
        origin_options,
        index=safe_index(
            origin_options,
            bet["origin"]
        ),
        horizontal=True,
        key=f"edit_origin_{bet_id}"
    )


    edit_my_odds = None
    edit_tipster_id = None
    edit_tipster_posted_odds = None
    edit_has_reasoning = False
    edit_primary = None
    edit_secondary = None

    edit_reasons = (
        get_reasons(
            edit_sport
        )
    )


    if edit_origin == "SELF":

        edit_my_odds = (
            st.number_input(
                "My Fair Odds",
                min_value=1.01,
                value=float(
                    bet["my_odds"]
                    or 1.80
                ),
                step=0.01,
                format="%.2f",
                key=(
                    f"edit_myodds_"
                    f"{bet_id}"
                )
            )
        )


        confidence_options = [
            "Low",
            "Medium",
            "High"
        ]


        edit_confidence = (
            st.radio(
                "Confidence",
                confidence_options,
                index=safe_index(
                    confidence_options,
                    bet["confidence"],
                    1
                ),
                horizontal=True,
                key=f"edit_conf_{bet_id}"
            )
        )


        reason_options = (
            ["Select reason..."]
            + edit_reasons
        )


        edit_primary = (
            st.selectbox(
                "Primary Reason",
                reason_options,
                index=safe_index(
                    reason_options,
                    bet["primary_reason"]
                ),
                key=(
                    f"edit_primary_"
                    f"{bet_id}"
                )
            )
        )


        secondary_options = (
            ["None"]
            + [
                reason
                for reason in edit_reasons
                if reason
                != edit_primary
            ]
        )


        edit_secondary = (
            st.selectbox(
                "Secondary Reason",
                secondary_options,
                index=safe_index(
                    secondary_options,
                    (
                        bet[
                            "secondary_reason"
                        ]
                        or "None"
                    )
                ),
                key=(
                    f"edit_secondary_"
                    f"{bet_id}"
                )
            )
        )


        edit_has_reasoning = True


    elif edit_origin == "SHARED":

        edit_confidence = st.radio(
            "Confidence",
            ["Low", "Medium", "High"],
            index=safe_index(
                ["Low", "Medium", "High"],
                bet.get("confidence") or "Medium",
                1
            ),
            horizontal=True,
            key=f"edit_shared_confidence_{bet_id}"
        )

        st.info(
            "This bet came from "
            f"{_shared_display_name(bet.get('shared_from_email'))}'s shared picks. "
            "Leave Origin as SHARED to keep that attribution, or change it to SELF/TIPSTER."
        )

    else:

        tipsters = load_tipsters()


        if not tipsters:

            st.warning(
                "No tipsters saved."
            )

            edit_tipster_id = None


        else:

            tipster_names = [
                t["name"]
                for t in tipsters
            ]


            tipster_ids = {
                t["name"]:
                    t["id"]
                for t in tipsters
            }


            current_tipster = None


            for tipster in tipsters:

                if (
                    tipster["id"]
                    == bet["tipster_id"]
                ):

                    current_tipster = (
                        tipster["name"]
                    )


            selected_tipster = (
                st.selectbox(
                    "Tipster",
                    tipster_names,
                    index=safe_index(
                        tipster_names,
                        current_tipster
                    ),
                    key=(
                        f"edit_tipster_"
                        f"{bet_id}"
                    )
                )
            )


            edit_tipster_id = (
                tipster_ids[
                    selected_tipster
                ]
            )


        has_posted = (
            bet[
                "tipster_posted_odds"
            ]
            is not None
        )


        edit_has_posted = (
            st.checkbox(
                "I know the tipster's "
                "posted odds",
                value=has_posted,
                key=(
                    f"edit_hasposted_"
                    f"{bet_id}"
                )
            )
        )


        if edit_has_posted:

            edit_tipster_posted_odds = (
                st.number_input(
                    "Tipster Posted Odds",
                    min_value=1.01,
                    value=float(
                        bet[
                            "tipster_posted_odds"
                        ]
                        or 1.90
                    ),
                    step=0.01,
                    format="%.2f",
                    key=(
                        f"edit_posted_"
                        f"{bet_id}"
                    )
                )
            )


        confidence_options = [
            "N/A",
            "Low",
            "Medium",
            "High"
        ]


        edit_confidence = (
            st.radio(
                "Your Confidence",
                confidence_options,
                index=safe_index(
                    confidence_options,
                    bet["confidence"]
                ),
                horizontal=True,
                key=f"edit_conf_{bet_id}"
            )
        )


        edit_has_reasoning = (
            st.checkbox(
                "I also have my own "
                "reasoning for this bet",
                value=bool(
                    bet[
                        "has_own_reasoning"
                    ]
                ),
                key=(
                    f"edit_reasoning_"
                    f"{bet_id}"
                )
            )
        )


        if edit_has_reasoning:

            reason_options = (
                ["Select reason..."]
                + edit_reasons
            )


            edit_primary = (
                st.selectbox(
                    "Primary Reason",
                    reason_options,
                    index=safe_index(
                        reason_options,
                        bet[
                            "primary_reason"
                        ]
                    ),
                    key=(
                        f"edit_primary_"
                        f"{bet_id}"
                    )
                )
            )


            secondary_options = (
                ["None"]
                + [
                    reason
                    for reason in edit_reasons
                    if reason
                    != edit_primary
                ]
            )


            edit_secondary = (
                st.selectbox(
                    "Secondary Reason",
                    secondary_options,
                    index=safe_index(
                        secondary_options,
                        (
                            bet[
                                "secondary_reason"
                            ]
                            or "None"
                        )
                    ),
                    key=(
                        f"edit_secondary_"
                        f"{bet_id}"
                    )
                )
            )


    # ======================================
    # STAKE / RESULT
    # ======================================

    edit_stake = st.number_input(
        "Stake",
        min_value=0.01,
        value=float(
            bet["stake"]
        ),
        step=1.00,
        key=f"edit_stake_{bet_id}"
    )


    result_options = [
        "Pending",
        "Win",
        "Loss",
        "Cashout",
        "Void"
    ]


    edit_result = st.selectbox(
        "Result",
        result_options,
        index=safe_index(
            result_options,
            bet["result"]
        ),
        key=f"edit_result_{bet_id}"
    )


    edit_cashout_return = None


    if edit_result == "Cashout":

        edit_cashout_return = (
            st.number_input(
                "Cashout Return",
                min_value=0.00,
                value=float(
                    bet[
                        "cashout_return"
                    ]
                    if bet.get(
                        "cashout_return"
                    ) is not None
                    else bet["stake"]
                ),
                step=0.50,
                format="%.2f",
                key=(
                    f"edit_cashout_"
                    f"{bet_id}"
                )
            )
        )


        st.caption(
            f"Cashout P/L: "
            f"{float(edit_cashout_return) - float(edit_stake):+.2f}"
        )


    edit_notes = st.text_area(
        "Notes",
        value=bet["notes"] or "",
        key=f"edit_notes_{bet_id}"
    )


    # ======================================
    # SAVE CHANGES
    # ======================================

    if st.button(
        "💾 SAVE CHANGES",
        type="primary",
        use_container_width=True,
        key=f"save_edit_{bet_id}"
    ):


        if not edit_event.strip():

            st.error(
                "Event / Competition "
                "is required."
            )

            return


        if (
            edit_scope
            in [
                "PLAYER",
                "TEAM",
                "OUTRIGHT"
            ]
            and not (
                edit_subject
                and edit_subject.strip()
            )
        ):

            st.error(
                "Selection is required."
            )

            return


        if (
            edit_scope == "OUTRIGHT"
            and outright_needs_second_selection(
                edit_market,
                edit_sport
            )
            and not (
                edit_selection_2
                and edit_selection_2.strip()
            )
        ):

            st.error(
                "Second selection "
                "is required."
            )

            return


        if (
            edit_origin == "SELF"
            and edit_primary
            == "Select reason..."
        ):

            st.error(
                "Select a Primary Reason."
            )

            return


        if (
            edit_origin == "TIPSTER"
            and edit_tipster_id is None
        ):

            st.error(
                "Select a valid tipster."
            )

            return


        if (
            edit_origin == "TIPSTER"
            and edit_has_reasoning
            and edit_primary
            == "Select reason..."
        ):

            st.error(
                "Select a Primary Reason."
            )

            return


        metrics = calculate_metrics(
            edit_market_odds,
            edit_my_odds,
            edit_tipster_posted_odds
        )


        if edit_result == "Cashout":

            edit_profit = round(
                float(
                    edit_cashout_return
                )
                - float(edit_stake),
                2
            )

        else:

            edit_profit = (
                calculate_profit(
                    edit_result,
                    edit_stake,
                    edit_market_odds
                )
            )


        if edit_result == "Pending":

            new_settled_at = None
            new_cashout_at = None


        else:

            new_settled_at = (
                bet["settled_at"]
                or now_utc()
            )


            if (
                edit_result
                == "Cashout"
            ):

                new_cashout_at = (
                    bet.get(
                        "cashout_at"
                    )
                    or now_utc()
                )

            else:

                new_cashout_at = None


        update_record = {

            "sport":
                edit_sport,

            "is_live":
                bool(edit_is_live),

            "bet_date":
                edit_date.isoformat(),

            "league":
                edit_league,

            "event":
                edit_event.strip(),

            "scope":
                edit_scope,

            "subject":
                (
                    edit_subject.strip()
                    if edit_subject
                    else None
                ),

            "selection_2":
                (
                    edit_selection_2.strip()
                    if edit_selection_2
                    else None
                ),

            "market":
                edit_market,

            "period":
                edit_period,

            "side":
                edit_side,

            "line":
                edit_line,

            "bookmaker":
                edit_bookmaker,

            "market_odds":
                edit_market_odds,

            "my_odds":
                edit_my_odds,

            "origin":
                edit_origin,

            "tipster_id":
                edit_tipster_id,

            "tipster_posted_odds":
                edit_tipster_posted_odds,

            "confidence":
                edit_confidence,

            "has_own_reasoning":
                edit_has_reasoning,

            "primary_reason":
                (
                    None
                    if edit_primary
                    == "Select reason..."
                    else edit_primary
                ),

            "secondary_reason":
                (
                    None
                    if edit_secondary
                    in [
                        None,
                        "None"
                    ]
                    else edit_secondary
                ),

            "stake":
                edit_stake,

            "result":
                edit_result,

            "settled_at":
                new_settled_at,

            "cashout_return":
                (
                    edit_cashout_return
                    if edit_result
                    == "Cashout"
                    else None
                ),

            "cashout_at":
                new_cashout_at,

            "p_market":
                metrics["p_market"],

            "p_you":
                metrics["p_you"],

            "edge_pp":
                metrics["edge_pp"],

            "ev_pct":
                metrics["ev_pct"],

            "price_deterioration_pp":
                metrics[
                    "price_deterioration_pp"
                ],

            "profit":
                edit_profit,

            "notes":
                (
                    edit_notes.strip()
                    if edit_notes.strip()
                    else None
                ),

            "updated_at":
                now_utc()
        }


        try:

            (
                supabase
                .table("bets")
                .update(
                    update_record
                )
                .eq(
                    "id",
                    bet_id
                )
                .eq(
                    "user_id",
                    st.session_state.user_id
                )
                .execute()
            )


            st.success(
                "✅ Bet updated successfully."
            )

            st.rerun()


        except Exception as e:

            st.error(
                f"Could not update bet: {e}"
            )


    # ======================================
    # SOFT DELETE
    # ======================================

    st.divider()

    st.subheader(
        "Delete Bet"
    )


    st.caption(
        "The bet will move to Trash. "
        "It will not be permanently deleted."
    )


    if (
        st.session_state.delete_confirm_id
        != bet_id
    ):

        if st.button(
            "🗑️ MOVE TO TRASH",
            use_container_width=True,
            key=f"delete_{bet_id}"
        ):

            st.session_state.delete_confirm_id = (
                bet_id
            )

            st.rerun()


    else:

        st.warning(
            "Are you sure you want "
            "to move this bet to Trash?"
        )


        c1, c2 = st.columns(2)


        with c1:

            if st.button(
                "Cancel",
                use_container_width=True,
                key=(
                    f"cancel_delete_"
                    f"{bet_id}"
                )
            ):

                st.session_state.delete_confirm_id = (
                    None
                )

                st.rerun()


        with c2:

            if st.button(
                "Yes, move to Trash",
                type="primary",
                use_container_width=True,
                key=(
                    f"confirm_delete_"
                    f"{bet_id}"
                )
            ):

                try:

                    soft_delete_bet(
                        bet_id
                    )

                    st.session_state.delete_confirm_id = (
                        None
                    )

                    st.rerun()


                except Exception as e:

                    st.error(
                        f"Could not delete "
                        f"bet: {e}"
                    )





# ==========================================
# TEMPORARY STOIXIMAN QUICK IMPORT REVIEW
# ==========================================

QUICK_IMPORT_BATCH = "STOIXIMAN_FIBA_WWC_SEP_2026_V1"
QUICK_IMPORT_FILE = "stoiximan_sep_2026_quick_import.json"


def _quick_resolve_chatgpt_tipster():
    """Use the user's EXISTING tipster named exactly Chat GPT. Never create one."""
    tipsters = load_tipsters()
    matches = [
        t for t in tipsters
        if (t.get("name") or "").strip() == "Chat GPT"
    ]
    if len(matches) != 1:
        return None, (
            "Expected exactly one existing tipster named 'Chat GPT'. "
            f"Found {len(matches)}. Import stopped; no tipster was created."
        )
    return matches[0]["id"], None


def _quick_import_payload():
    import json
    from pathlib import Path
    path = Path(__file__).with_name(QUICK_IMPORT_FILE)
    return json.loads(path.read_text(encoding="utf-8"))


def _quick_import_batch_count():
    try:
        response = (
            supabase.table("bets")
            .select("id", count="exact")
            .eq("user_id", st.session_state.user_id)
            .eq("import_batch_id", QUICK_IMPORT_BATCH)
            .execute()
        )
        return response.count or 0
    except Exception:
        return 0


def _quick_review_rows():
    try:
        response = (
            supabase.table("bets")
            .select("*")
            .eq("user_id", st.session_state.user_id)
            .eq("needs_review", True)
            .eq("import_batch_id", QUICK_IMPORT_BATCH)
            .order("bet_date", desc=True)
            .order("bet_number", desc=True)
            .execute()
        )
        return response.data or []
    except Exception as e:
        st.error(f"Could not load quick-import rows: {e}")
        return []


def _quick_import_batch():
    if _quick_import_batch_count() > 0:
        raise RuntimeError("This screenshot batch already exists. Duplicate import blocked.")

    chatgpt_tipster_id, error = _quick_resolve_chatgpt_tipster()
    if error:
        raise RuntimeError(error)

    payload = _quick_import_payload()
    records = []

    for item in payload:
        odds = float(item["market_odds"])
        combo_legs = item.get("combo_legs")
        record = {
            "user_id": st.session_state.user_id,
            "bet_date": item["bet_date"],
            "is_live": False,
            "sport": item.get("sport") or "Basketball",
            "league": item.get("league") or "FIBA Women's Basketball World Cup",
            "event": item.get("event") or "FIBA Women's Basketball World Cup 2026",
            "scope": item.get("scope") or "MATCH",
            "subject": item.get("subject"),
            "selection_2": None,
            "market": item.get("market") or "Other",
            "period": item.get("period") or "Full Game",
            "side": item.get("side"),
            "line": item.get("line"),
            "bookmaker": "Stoiximan",
            "market_odds": odds,
            "my_odds": None,
            "origin": "TIPSTER",
            "tipster_id": chatgpt_tipster_id,
            "tipster_posted_odds": None,
            "confidence": "Medium",
            "has_own_reasoning": False,
            "primary_reason": None,
            "secondary_reason": None,
            "stake": float(item["stake"]),
            "result": "Pending",
            "p_market": 1 / odds,
            "p_you": None,
            "edge_pp": None,
            "ev_pct": None,
            "price_deterioration_pp": None,
            "profit": 0,
            "notes": (
                "Stoiximan screenshot quick import"
                + (f" | {item.get('review_note')}" if item.get("review_note") else "")
            ),
            "combo_legs": combo_legs,
            "needs_review": True,
            "import_batch_id": QUICK_IMPORT_BATCH,
            "import_confidence": item.get("import_confidence", "HIGH"),
            "import_review_note": item.get("review_note"),
            "import_return": None,
        }
        records.append(record)

    for start in range(0, len(records), 50):
        supabase.table("bets").insert(records[start:start + 50]).execute()

    return len(records)


def _quick_tipster_maps():
    tipsters = load_tipsters()
    by_name = {(t.get("name") or "").strip(): t.get("id") for t in tipsters if t.get("id")}
    by_id = {str(v): k for k, v in by_name.items()}
    return by_name, by_id


def _quick_optional_float(text):
    text = str(text or "").strip().replace(",", ".")
    if not text:
        return None
    return float(text)


def quick_import_page():
    st.header("⚡ Stoiximan Quick Import")
    st.caption(
        "Temporary helper for the Sep 2026 screenshot batch. Rows stay OUT of Pending, "
        "History, Analysis and Manage until you approve them. Default source is the EXISTING "
        "tipster 'Chat GPT'; this helper never creates a new tipster."
    )

    total_existing = _quick_import_batch_count()
    if total_existing == 0:
        st.info(
            "The batch is ready. Exact same selections on the SAME DATE were merged where useful; "
            "different odds were converted to an effective weighted price so total stake/payout is preserved."
        )
        if st.button("📥 Load screenshot batch", type="primary", use_container_width=True):
            try:
                count = _quick_import_batch()
                st.success(f"Loaded {count} review rows.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        return

    rows = _quick_review_rows()
    st.metric("Still to approve", len(rows))

    if not rows:
        st.success("All screenshot bets are approved. The temporary helper can now be removed.")
        return

    tipster_map, tipster_reverse = _quick_tipster_maps()
    tipster_names = list(tipster_map.keys())

    high_rows = [r for r in rows if (r.get("import_confidence") or "HIGH") == "HIGH"]
    if high_rows:
        if st.button(
            f"✅ Approve all HIGH confidence ({len(high_rows)})",
            use_container_width=True,
            help="Use this only after a quick visual check; the CHECK row(s) stay here."
        ):
            try:
                for r in high_rows:
                    supabase.table("bets").update({
                        "needs_review": False,
                        "updated_at": now_utc(),
                    }).eq("id", r["id"]).eq("user_id", st.session_state.user_id).execute()
                st.rerun()
            except Exception as e:
                st.error(f"Bulk approve failed: {e}")

    show = st.radio(
        "Show",
        ["Needs check", "Outrights", "Combos", "All"],
        horizontal=True,
        key="quick_import_show"
    )

    def is_combo_row(r):
        return r.get("market") in ["Bet Builder", "Parlay"] and bool(r.get("combo_legs"))

    if show == "Needs check":
        rows = [r for r in rows if (r.get("import_confidence") or "HIGH") != "HIGH"]
    elif show == "Outrights":
        rows = [r for r in rows if r.get("scope") == "OUTRIGHT"]
    elif show == "Combos":
        rows = [r for r in rows if is_combo_row(r)]

    for bet in rows:
        is_combo = is_combo_row(bet)
        badge = "⚠️ CHECK" if (bet.get("import_confidence") or "HIGH") != "HIGH" else "✅ HIGH"
        current_tipster = tipster_reverse.get(str(bet.get("tipster_id")), "Chat GPT")

        if is_combo:
            if bet.get("scope") == "OUTRIGHT":
                type_label = "Outright Parlay"
            elif bet.get("market") == "Bet Builder":
                type_label = "Bet Builder"
            else:
                type_label = "Parlay"
            title = bet.get("event") or type_label
        else:
            type_label = "Outright" if bet.get("scope") == "OUTRIGHT" else "Single"
            title = bet.get("subject") or bet.get("event") or "Imported bet"

        with st.expander(
            f"{badge} | {bet.get('bet_date')} | {type_label} | {title}",
            expanded=(badge.startswith("⚠️"))
        ):
            if bet.get("import_review_note"):
                st.caption(f"ℹ️ {bet.get('import_review_note')}")

            top1, top2, top3 = st.columns(3)
            with top1:
                edit_date = st.date_input(
                    "Bet Date",
                    value=datetime.strptime(bet.get("bet_date"), "%Y-%m-%d").date(),
                    key=f"qimp_date_{bet['id']}"
                )
            with top2:
                bookmaker_options = list(BOOKMAKERS)
                current_bookmaker = bet.get("bookmaker") or "Stoiximan"
                if current_bookmaker not in bookmaker_options:
                    bookmaker_options = [current_bookmaker] + bookmaker_options
                edit_bookmaker = st.selectbox(
                    "Bookmaker",
                    bookmaker_options,
                    index=safe_index(bookmaker_options, current_bookmaker),
                    key=f"qimp_book_{bet['id']}"
                )
            with top3:
                edit_confidence = st.selectbox(
                    "Confidence",
                    ["Low", "Medium", "High"],
                    index=safe_index(["Low", "Medium", "High"], bet.get("confidence") or "Medium"),
                    key=f"qimp_conf_{bet['id']}"
                )

            oc1, oc2 = st.columns(2)
            with oc1:
                edit_origin = st.radio(
                    "Origin",
                    ["TIPSTER", "SELF"],
                    index=0 if bet.get("origin") != "SELF" else 1,
                    horizontal=True,
                    key=f"qimp_origin_{bet['id']}"
                )
            with oc2:
                edit_tipster_name = None
                if edit_origin == "TIPSTER":
                    if not tipster_names:
                        st.error("No existing tipsters found.")
                    else:
                        default_tipster = current_tipster if current_tipster in tipster_names else (
                            "Chat GPT" if "Chat GPT" in tipster_names else tipster_names[0]
                        )
                        edit_tipster_name = st.selectbox(
                            "Tipster",
                            tipster_names,
                            index=safe_index(tipster_names, default_tipster),
                            key=f"qimp_tipster_{bet['id']}",
                            help="Existing tipsters only — this helper cannot create a new one."
                        )

            g1, g2 = st.columns(2)
            with g1:
                edit_sport = st.selectbox(
                    "Sport", SPORTS,
                    index=safe_index(SPORTS, bet.get("sport") or DEFAULT_SPORT),
                    key=f"qimp_sport_{bet['id']}"
                )
                edit_league = st.text_input(
                    "League / Tour", value=bet.get("league") or "",
                    key=f"qimp_league_{bet['id']}"
                )
                edit_event = st.text_input(
                    "Event / Competition", value=bet.get("event") or "",
                    key=f"qimp_event_{bet['id']}"
                )
            with g2:
                edit_odds = st.number_input(
                    "Odds Taken", min_value=1.001,
                    value=float(bet.get("market_odds") or 1.01),
                    step=0.01, format="%.4f",
                    key=f"qimp_odds_{bet['id']}"
                )
                edit_stake = st.number_input(
                    "Stake", min_value=0.01,
                    value=float(bet.get("stake") or 0.01),
                    step=1.0, format="%.2f",
                    key=f"qimp_stake_{bet['id']}"
                )
                st.text_input(
                    "Result", value="Pending", disabled=True,
                    key=f"qimp_result_{bet['id']}"
                )

            edited_legs = copy.deepcopy(bet.get("combo_legs") or [])

            if is_combo:
                structure_options = ["Bet Builder", "Parlay", "Outright Parlay"]
                current_structure = (
                    "Outright Parlay" if bet.get("scope") == "OUTRIGHT"
                    else bet.get("market")
                )
                edit_structure = st.selectbox(
                    "Classification",
                    structure_options,
                    index=safe_index(structure_options, current_structure),
                    key=f"qimp_structure_{bet['id']}"
                )

                st.caption("Selections / components")
                for ci, component in enumerate(edited_legs):
                    kind = str(component.get("kind") or "SINGLE").upper()
                    if kind == "BET_BUILDER":
                        st.write(f"**Component {ci + 1}: Bet Builder**")
                        component["label"] = st.text_input(
                            "BB label",
                            value=component.get("label") or "",
                            key=f"qimp_bb_label_{bet['id']}_{ci}"
                        )
                        comp_odds_text = st.text_input(
                            "BB combined odds",
                            value="" if component.get("component_odds") is None else str(component.get("component_odds")),
                            key=f"qimp_bb_comp_odds_{bet['id']}_{ci}"
                        )
                        component["component_odds"] = _quick_optional_float(comp_odds_text)
                        for si, selection in enumerate(component.get("selections") or []):
                            lc1, lc2 = st.columns([3, 1])
                            with lc1:
                                selection["label"] = st.text_input(
                                    f"BB selection {si + 1}",
                                    value=selection.get("label") or "",
                                    key=f"qimp_bb_sel_{bet['id']}_{ci}_{si}"
                                )
                            with lc2:
                                sodds = st.text_input(
                                    "Standalone odds",
                                    value="" if selection.get("odds") is None else str(selection.get("odds")),
                                    key=f"qimp_bb_sel_odds_{bet['id']}_{ci}_{si}"
                                )
                                selection["odds"] = _quick_optional_float(sodds)
                    else:
                        lc1, lc2, lc3 = st.columns([1, 3, 1])
                        with lc1:
                            kind_choice = st.selectbox(
                                "Leg type",
                                ["SINGLE", "OUTRIGHT"],
                                index=safe_index(["SINGLE", "OUTRIGHT"], kind),
                                key=f"qimp_leg_kind_{bet['id']}_{ci}"
                            )
                        with lc2:
                            component["label"] = st.text_input(
                                f"Leg {ci + 1}",
                                value=component.get("label") or "",
                                key=f"qimp_leg_label_{bet['id']}_{ci}"
                            )
                        with lc3:
                            lodds = st.text_input(
                                "Odds",
                                value="" if component.get("odds") is None else str(component.get("odds")),
                                key=f"qimp_leg_odds_{bet['id']}_{ci}"
                            )
                            component["odds"] = _quick_optional_float(lodds)
                        component["kind"] = kind_choice

            else:
                single_type_options = ["Single", "Outright"]
                current_single_type = "Outright" if bet.get("scope") == "OUTRIGHT" else "Single"
                edit_single_type = st.selectbox(
                    "Classification",
                    single_type_options,
                    index=safe_index(single_type_options, current_single_type),
                    key=f"qimp_single_type_{bet['id']}"
                )

                c1, c2 = st.columns(2)
                with c1:
                    if edit_single_type == "Outright":
                        edit_scope = "OUTRIGHT"
                        st.text_input("Scope", value="OUTRIGHT", disabled=True, key=f"qimp_scope_display_{bet['id']}")
                    else:
                        scope_opts = ["PLAYER", "TEAM", "MATCH"]
                        edit_scope = st.selectbox(
                            "Scope", scope_opts,
                            index=safe_index(scope_opts, bet.get("scope") if bet.get("scope") in scope_opts else "PLAYER"),
                            key=f"qimp_scope_{bet['id']}"
                        )
                    edit_subject = st.text_input(
                        "Player / Team / Selection",
                        value=bet.get("subject") or "",
                        key=f"qimp_subject_{bet['id']}"
                    )
                    edit_market = st.text_input(
                        "Market", value=bet.get("market") or "",
                        key=f"qimp_market_{bet['id']}"
                    )
                with c2:
                    side_options = ["", "Over", "Under", "Home", "Away", "Draw", "Yes", "No"]
                    edit_side = st.selectbox(
                        "Side", side_options,
                        index=safe_index(side_options, bet.get("side") or ""),
                        key=f"qimp_side_{bet['id']}"
                    )
                    edit_line_text = st.text_input(
                        "Line", value="" if bet.get("line") is None else str(bet.get("line")),
                        key=f"qimp_line_{bet['id']}"
                    )
                    edit_period = (
                        "Full Competition" if edit_single_type == "Outright"
                        else st.text_input(
                            "Period", value=bet.get("period") or "Full Game",
                            key=f"qimp_period_{bet['id']}"
                        )
                    )

            def save_current(approve=False):
                if edit_origin == "TIPSTER":
                    if not edit_tipster_name or edit_tipster_name not in tipster_map:
                        raise RuntimeError("Choose an existing tipster.")
                    tipster_id = tipster_map[edit_tipster_name]
                else:
                    tipster_id = None

                update = {
                    "bet_date": edit_date.isoformat(),
                    "sport": edit_sport,
                    "league": edit_league.strip(),
                    "event": edit_event.strip(),
                    "bookmaker": edit_bookmaker,
                    "market_odds": float(edit_odds),
                    "stake": float(edit_stake),
                    "origin": edit_origin,
                    "tipster_id": tipster_id,
                    "confidence": edit_confidence,
                    "p_market": 1 / float(edit_odds),
                    "updated_at": now_utc(),
                    "needs_review": (not approve),
                }

                if is_combo:
                    if edit_structure == "Bet Builder":
                        update.update({"market": "Bet Builder", "scope": "MATCH", "period": "Combined"})
                    elif edit_structure == "Outright Parlay":
                        update.update({"market": "Parlay", "scope": "OUTRIGHT", "period": "Full Competition"})
                        for component in edited_legs:
                            if str(component.get("kind") or "").upper() != "BET_BUILDER":
                                component["kind"] = "OUTRIGHT"
                                component["parlay_content"] = "Outright"
                    else:
                        update.update({"market": "Parlay", "scope": "MATCH", "period": "Combined"})
                    update["combo_legs"] = edited_legs
                else:
                    line_value = _quick_optional_float(edit_line_text)
                    update.update({
                        "scope": edit_scope,
                        "subject": edit_subject.strip() or None,
                        "market": edit_market.strip(),
                        "side": edit_side or None,
                        "line": line_value,
                        "period": edit_period,
                    })

                supabase.table("bets").update(update).eq("id", bet["id"]).eq("user_id", st.session_state.user_id).execute()

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("💾 SAVE", key=f"qimp_save_{bet['id']}", use_container_width=True):
                    try:
                        save_current(False)
                        st.success("Saved.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            with b2:
                if st.button("✅ SAVE + APPROVE", key=f"qimp_approve_{bet['id']}", type="primary", use_container_width=True):
                    try:
                        save_current(True)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            with b3:
                if st.button("🗑️ REMOVE", key=f"qimp_remove_{bet['id']}", use_container_width=True):
                    try:
                        supabase.table("bets").delete().eq("id", bet["id"]).eq("user_id", st.session_state.user_id).eq("import_batch_id", QUICK_IMPORT_BATCH).execute()
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))


# ==========================================
# TRASH
# ==========================================

def trash_page():

    deleted = (
        load_deleted_bets()
    )


    st.header(
        f"🗑️ Trash ({len(deleted)})"
    )


    st.caption(
        "Deleted bets are excluded from "
        "counters, History and Analysis."
    )


    if not deleted:

        st.success(
            "Trash is empty."
        )

        return


    for bet in deleted:

        st.divider()


        subject = (

            bet["event"]

            if bet["scope"] == "MATCH"

            else bet["subject"]
        )


        st.subheader(
            subject
        )


        market_text = (
            f"{bet['market']} | "
            f"{bet['side']}"
        )


        if bet["line"] is not None:

            market_text += (
                f" {float(bet['line']):g}"
            )


        st.write(
            market_text
        )


        st.caption(
            f"{bet.get('sport') or DEFAULT_SPORT} | "
            f"{bet['league']} | "
            f"Odds {float(bet['market_odds']):.2f} | "
            f"Result: {bet['result']}"
        )


        st.caption(
            f"Deleted: "
            f"{bet['deleted_at'] or 'Unknown'}"
        )


        if st.button(
            "♻️ RESTORE",
            key=f"restore_{bet['id']}",
            use_container_width=True
        ):

            try:

                restore_bet(
                    bet["id"]
                )

                st.rerun()


            except Exception as e:

                st.error(
                    f"Could not restore bet: {e}"
                )


# ==========================================
# MAIN
# ==========================================

if not st.session_state.logged_in:

    login_page()

    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.write(
        f"👤 "
        f"{st.session_state.user_email}"
    )


    if st.button(
        "Logout",
        use_container_width=True
    ):

        logout()


# ==========================================
# HEADER
# ==========================================

st.title(
    "🎯 Bet Tracker"
)

st.caption(
    "Personal betting & analytics tracker"
)


# ==========================================
# COUNTERS
# ==========================================

try:

    total_bets = (
        get_total_bets_count()
    )

    pending_count = (
        get_pending_bets_count()
    )

    settled_count = (
        get_settled_bets_count()
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Total Bets",
            total_bets
        )


    with c2:

        st.metric(
            "Pending",
            pending_count
        )


    with c3:

        st.metric(
            "Settled",
            settled_count
        )


except Exception as e:

    st.warning(
        f"Could not load counters: {e}"
    )



# ==========================================
# NAVIGATION
# ==========================================

(
    add_tab,
    pending_tab,
    shared_tab,
    history_tab,
    analysis_tab,
    suggestions_tab,
    manage_tab,
    quick_import_tab,
    trash_tab
) = st.tabs([

    "➕ Add Bet",
    "⏳ Pending",
    "👥 Shared Picks",
    "📜 History",
    "📊 Analysis",
    "💡 Suggestions",
    "✏️ Manage",
    "⚡ Quick Import",
    "🗑️ Trash"
])


with add_tab:

    add_bet_page()


with pending_tab:

    pending_bets_page()


with shared_tab:

    shared_picks_page()


with history_tab:

    history_page()


with analysis_tab:

    analysis_page(
        supabase,
        load_tipsters
    )


with suggestions_tab:

    suggestions_page(
        supabase,
        load_tipsters
    )



with manage_tab:

    manage_bets_page()


with quick_import_tab:

    quick_import_page()


with trash_tab:

    trash_page()
