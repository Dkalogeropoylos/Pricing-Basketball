from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from core.buckets import WeightConfig, active_weights, split_non_overlapping
from core.minutes_engine import project_team_minutes


STAT_COLS = [
    "PTS", "FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM", "TOV",
    "OREB", "DREB", "REB", "AST", "PF", "STL", "BLK",
]


def _norm(v) -> str:
    return str(v).strip().casefold()


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    return float(a) / float(b) if np.isfinite(b) and float(b) > 0 else default


def _weighted_bucket_rate(bucket: pd.DataFrame, stat: str) -> float:
    if bucket is None or bucket.empty or stat not in bucket.columns:
        return np.nan
    mins = pd.to_numeric(bucket.get("MIN", 0), errors="coerce").fillna(0.0)
    vals = pd.to_numeric(bucket.get(stat, 0), errors="coerce").fillna(0.0)
    den = float(mins.sum())
    return float(vals.sum() / den) if den > 0 else np.nan


def _player_rate_profile(log: pd.DataFrame) -> dict:
    """Stable non-overlapping per-minute player rates for roster synthesis.

    This is intentionally simpler than the Player Props engine: it is used only
    to estimate how a changed 200-minute rotation changes aggregate team style.
    It never becomes a fourth historical sample.
    """
    x = log.sort_values("GAME_DATE").copy()
    if x.empty:
        return {}
    buckets = split_non_overlapping(x)
    outer = active_weights(buckets, WeightConfig.stable())
    out = {}
    for stat in STAT_COLS:
        vals, ws = [], []
        for k in ("old", "mid", "l5"):
            r = _weighted_bucket_rate(buckets.get(k, pd.DataFrame()), stat)
            w = float(outer.get(k, 0.0))
            if w > 0 and np.isfinite(r):
                vals.append(r)
                ws.append(w)
        if vals:
            ws = np.asarray(ws, dtype=float)
            ws = ws / ws.sum()
            out[stat] = float(np.sum(np.asarray(vals) * ws))
        else:
            out[stat] = 0.0
    return out


def recent_team_player_names(
    player_db: pd.DataFrame,
    team_abbr: str,
    lookback_days: int = 70,
    min_recent_minutes: float = 1.0,
) -> list[str]:
    """Recent team names, including players no longer returned by current-pool APIs.

    This lets the trader explicitly mark a just-traded/bought-out player OUT so
    the historical team baseline can be adjusted away from that player.
    """
    if player_db is None or player_db.empty:
        return []
    x = player_db[
        player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
    ].copy()
    if x.empty:
        return []
    x["_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
    max_date = x["_DATE"].max()
    if pd.notna(max_date):
        x = x[x["_DATE"] >= max_date - pd.Timedelta(days=int(lookback_days))].copy()
    x["_MIN"] = pd.to_numeric(x.get("MIN", 0), errors="coerce").fillna(0.0)
    g = x.groupby("PLAYER_NAME", as_index=False)["_MIN"].sum()
    names = g[g["_MIN"] >= float(min_recent_minutes)]["PLAYER_NAME"].astype(str).tolist()
    return sorted(set(names), key=str.casefold)


def augment_current_pool(
    current_pool: pd.DataFrame,
    player_db: pd.DataFrame,
    team_abbr: str,
    extra_names: Iterable[str],
) -> pd.DataFrame:
    """Add explicitly referenced recent players to the provider's current pool.

    Needed for newly unavailable players (buyout/trade/season-ending removal)
    who may disappear from a current-roster endpoint before the historical model
    has any games without them.
    """
    pool = current_pool.copy() if current_pool is not None else pd.DataFrame()
    extras = [str(x) for x in extra_names if str(x).strip()]
    if not extras or player_db is None or player_db.empty:
        return pool

    existing = set(pool.get("PLAYER_NAME", pd.Series(dtype=str)).astype(str).map(_norm).tolist())
    add_rows = []
    for name in extras:
        if _norm(name) in existing:
            continue
        hist = player_db[
            player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
            & player_db["PLAYER_NAME"].astype(str).map(_norm).eq(_norm(name))
        ].copy()
        if hist.empty:
            continue
        latest = hist.sort_values("GAME_DATE").iloc[-1]
        row = {}
        # Preserve the provider-pool schema where possible.
        cols = list(pool.columns) if len(pool.columns) else [
            "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBR", "POSITION_ABBR", "POSITION_GROUP"
        ]
        for c in cols:
            if c == "PLAYER_NAME":
                row[c] = str(latest.get(c, name))
            elif c == "TEAM_ABBR":
                row[c] = str(team_abbr).upper()
            else:
                row[c] = latest.get(c, np.nan)
        if "PLAYER_ID" not in row:
            row["PLAYER_ID"] = latest.get("PLAYER_ID")
        if "PLAYER_NAME" not in row:
            row["PLAYER_NAME"] = name
        if "TEAM_ABBR" not in row:
            row["TEAM_ABBR"] = str(team_abbr).upper()
        add_rows.append(row)
        existing.add(_norm(name))
    if add_rows:
        pool = pd.concat([pool, pd.DataFrame(add_rows)], ignore_index=True, sort=False)
    return pool.drop_duplicates(subset=["PLAYER_ID"], keep="last").reset_index(drop=True)


def _set_selected_out_in(ctx: dict, out_players: Iterable[str]) -> dict:
    out = copy.deepcopy(ctx or {})
    injuries = out.setdefault("injuries", {})
    targets = {_norm(x) for x in out_players}
    for name, info in list(injuries.items()):
        if _norm(name) not in targets:
            continue
        if isinstance(info, dict):
            info = dict(info)
            info["status"] = "IN"
            injuries[name] = info
        else:
            injuries[name] = {"status": "IN"}
    return out


def _without_team_minute_overrides(ctx: dict, team_names: Iterable[str]) -> dict:
    out = copy.deepcopy(ctx or {})
    block = dict(out.get("projected_minutes", {}) or {})
    targets = {_norm(x) for x in team_names}
    block = {k: v for k, v in block.items() if _norm(k) not in targets}
    out["projected_minutes"] = block
    return out


def _profiles_for_team(player_db: pd.DataFrame, team_abbr: str, names: Iterable[str]) -> Dict[str, dict]:
    profiles = {}
    for name in names:
        log = player_db[
            player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
            & player_db["PLAYER_NAME"].astype(str).map(_norm).eq(_norm(name))
        ].copy()
        p = _player_rate_profile(log)
        if p:
            profiles[str(name)] = p
    return profiles


def _synthetic_from_minutes(board: pd.DataFrame, profiles: Dict[str, dict]) -> Tuple[dict, dict]:
    totals = {s: 0.0 for s in STAT_COLS}
    by_player = {}
    if board is None or board.empty:
        return totals, by_player
    for _, row in board.iterrows():
        name = str(row["Player"])
        mins = float(row.get("Projected Min", 0.0) or 0.0)
        p = profiles.get(name, {})
        pt = {}
        for stat in STAT_COLS:
            val = mins * float(p.get(stat, 0.0) or 0.0)
            totals[stat] += val
            pt[stat] = val
        pt["MIN"] = mins
        by_player[name] = pt
    fga, fgm = totals["FGA"], totals["FGM"]
    a3, m3 = totals["FG3A"], totals["FG3M"]
    a2, m2 = max(fga - a3, 0.0), max(fgm - m3, 0.0)
    misses = max(fga - fgm, 0.0)
    feats = {
        "FGA": fga,
        "3P_SHARE": _safe_div(a3, fga, 0.35),
        "FTA": totals["FTA"],
        "TOV": totals["TOV"],
        "OREB": _safe_div(totals["OREB"], misses, 0.25),
        "AST": _safe_div(totals["AST"], fgm, 0.62),
        "PF": totals["PF"],
        "DREB": totals["DREB"],
        "STL": totals["STL"],
        "BLK": totals["BLK"],
        "3P_PCT": _safe_div(m3, a3, np.nan),
        "2P_PCT": _safe_div(m2, a2, np.nan),
    }
    return feats, by_player


def _ratio(a, b, default=1.0):
    return _safe_div(a, b, default) if np.isfinite(a) and np.isfinite(b) else default


def _pow_clip(ratio: float, exponent: float, lo: float, hi: float) -> float:
    if not np.isfinite(ratio) or ratio <= 0 or exponent <= 0:
        return 1.0
    return float(np.clip(float(ratio) ** float(exponent), lo, hi))


TEAM_STAT_POWER = {
    # FGA count is mostly possession-driven, so roster identity only nudges it.
    "FGA": 0.25,
    # Shot selection / foul drawing / creation can move more when a role changes.
    "3P_SHARE": 0.65,
    "FTA": 0.60,
    "TOV": 0.45,
    "OREB": 0.45,
    "AST": 0.55,
    "PF": 0.40,
    "DREB": 0.35,
    "STL": 0.35,
    "BLK": 0.55,
    # Shooter-mix changes matter, but efficiency stays heavily regularized.
    "3P_PCT": 0.45,
    "2P_PCT": 0.40,
}

TEAM_CAPS = {
    "FGA": (0.96, 1.04),
    "3P_SHARE": (0.90, 1.10),
    "FTA": (0.88, 1.12),
    "TOV": (0.90, 1.10),
    "OREB": (0.90, 1.10),
    "AST": (0.90, 1.10),
    "PF": (0.92, 1.08),
    "DREB": (0.94, 1.06),
    "STL": (0.92, 1.08),
    "BLK": (0.88, 1.12),
    "3P_PCT": (0.96, 1.04),
    "2P_PCT": (0.96, 1.04),
}


@dataclass
class RotationStateImpact:
    modifiers: Dict[str, float]
    team_audit: pd.DataFrame
    current_minutes: pd.DataFrame
    out_only_minutes: pd.DataFrame
    healthy_minutes: pd.DataFrame
    raw_player_role_modifiers: pd.DataFrame


def build_rotation_state_impact(
    player_db: pd.DataFrame,
    team_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    team_name: str,
    manual_context: dict,
    out_players: Iterable[str],
    exact_state_confidence: float = 0.0,
) -> RotationStateImpact:
    """Roster/minute-state bridge for Team Markets and Player Props.

    Three states are simulated with the SAME player per-minute priors:
      healthy  = selected OUT players restored; no team minute overrides
      out_only = confirmed OUT state; no team minute overrides
      current  = confirmed OUT + explicit projected-minute restrictions/returns

    Therefore:
      - exact historical OUT games remain the primary empirical layer;
      - the synthetic OUT modifier is strongest only when exact-state evidence
        is sparse, avoiding duplicate injury information;
      - minute restrictions are a separate trader-information layer and are not
        mistaken for another historical sample.
    """
    out_players = [str(x) for x in out_players if str(x).strip()]
    # Only explicitly selected unavailable players are reintroduced into the
    # calculation pool. Other recent ex-players must never receive today's minutes merely because they appear in historical logs.
    pool = augment_current_pool(current_pool, player_db, team_abbr, out_players)
    team_names = pool[
        pool["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
    ]["PLAYER_NAME"].astype(str).tolist()

    current_ctx = copy.deepcopy(manual_context or {})
    out_only_ctx = _without_team_minute_overrides(current_ctx, team_names)
    healthy_ctx = _set_selected_out_in(out_only_ctx, out_players)

    current_board = project_team_minutes(
        player_db, team_db, pool, team_abbr, team_name, current_ctx
    )
    out_only_board = project_team_minutes(
        player_db, team_db, pool, team_abbr, team_name, out_only_ctx
    )
    healthy_board = project_team_minutes(
        player_db, team_db, pool, team_abbr, team_name, healthy_ctx
    )

    names = sorted(set(team_names), key=str.casefold)
    profiles = _profiles_for_team(player_db, team_abbr, names)
    current_feat, current_by = _synthetic_from_minutes(current_board, profiles)
    out_feat, out_by = _synthetic_from_minutes(out_only_board, profiles)
    healthy_feat, healthy_by = _synthetic_from_minutes(healthy_board, profiles)

    conf = float(np.clip(exact_state_confidence, 0.0, 1.0))
    # Exact historical state already moves the team profile. Synthetic OUT
    # evidence fills only the remaining information gap.
    out_evidence = 0.65 * (1.0 - conf)

    # Explicit projected_minutes are trader information and are not represented
    # by exact historical OUT matching, so they receive a separate strong but
    # still shrinked bridge.
    def _shifted_minutes(a: pd.DataFrame, b: pd.DataFrame) -> float:
        if a is None or a.empty or b is None or b.empty:
            return 0.0
        aa = a.set_index("Player")["Projected Min"]
        bb = b.set_index("Player")["Projected Min"]
        idx = aa.index.union(bb.index)
        return 0.5 * float((aa.reindex(idx, fill_value=0.0) - bb.reindex(idx, fill_value=0.0)).abs().sum())

    out_shift = _shifted_minutes(out_only_board, healthy_board)
    restriction_shift = _shifted_minutes(current_board, out_only_board)
    restriction_evidence = 0.80 if restriction_shift >= 0.5 else 0.0

    mods = {}
    rows = []
    for stat, power in TEAM_STAT_POWER.items():
        lo, hi = TEAM_CAPS[stat]
        r_out = _ratio(out_feat.get(stat, np.nan), healthy_feat.get(stat, np.nan), 1.0)
        r_rest = _ratio(current_feat.get(stat, np.nan), out_feat.get(stat, np.nan), 1.0)
        m_out = _pow_clip(r_out, power * out_evidence, lo, hi)
        m_rest = _pow_clip(r_rest, power * restriction_evidence, lo, hi)
        final = float(np.clip(m_out * m_rest, lo, hi))
        mods[stat] = final
        rows.append({
            "Stat": stat,
            "Healthy synthetic": healthy_feat.get(stat, np.nan),
            "OUT-only synthetic": out_feat.get(stat, np.nan),
            "Current synthetic": current_feat.get(stat, np.nan),
            "OUT raw ratio": r_out,
            "Exact-state confidence": conf,
            "OUT bridge strength": power * out_evidence,
            "Restriction raw ratio": r_rest,
            "Restriction bridge strength": power * restriction_evidence,
            "Applied roster-state modifier": final,
        })

    # Player-role fallback: minutes are ALREADY modeled directly, so this layer
    # only allocates a fraction of vacated event volume that would otherwise
    # disappear because replacement players have lower historical per-minute rates.
    # It is later shrunk again by each focal player's exact-state confidence.
    player_rows = []
    events = {
        "FGA": ("usage", 0.90),
        "FG3A": ("three_rate", 0.85),
        "FTA": ("fta_rate", 0.75),
        "AST": ("creation", 0.80),
        "REB": ("reb_role", 0.95),
    }

    def _stage_adjust(
        base_by, reference_by, stage_current_board, preserve_scale=1.0,
        reference_board=None, exclude_minute_decliners=False,
    ):
        adj = {name: {s: float(base_by.get(name, {}).get(s, 0.0)) for s in events} for name in names}
        active = {
            str(r["Player"]): float(r.get("Projected Min", 0.0) or 0.0)
            for _, r in stage_current_board.iterrows()
            if float(r.get("Projected Min", 0.0) or 0.0) > 0.1
        }
        ref_min = {}
        if reference_board is not None and not reference_board.empty:
            ref_min = {
                str(r["Player"]): float(r.get("Projected Min", 0.0) or 0.0)
                for _, r in reference_board.iterrows()
            }
        for stat, (_, preserve) in events.items():
            base_total = sum(float(base_by.get(n, {}).get(stat, 0.0)) for n in names)
            ref_total = sum(float(reference_by.get(n, {}).get(stat, 0.0)) for n in names)
            lost = max(ref_total - base_total, 0.0)
            residual = lost * preserve * preserve_scale
            if residual <= 0 or not active:
                continue
            raw_w = {}
            for n, mins in active.items():
                if exclude_minute_decliners and mins < ref_min.get(n, mins) - 0.25:
                    raw_w[n] = 0.0
                    continue
                contrib = max(float(base_by.get(n, {}).get(stat, 0.0)), 0.0)
                raw_w[n] = max(contrib, 0.02 * mins) ** 1.10
            den = sum(raw_w.values())
            if den <= 0:
                continue
            for n, w in raw_w.items():
                adj.setdefault(n, {})[stat] = adj.get(n, {}).get(stat, 0.0) + residual * w / den
        return adj

    # OUT residual is reduced by exact-state evidence; restriction residual is
    # separate and full-strength because it is trader-specified current context.
    out_adjusted = _stage_adjust(
        out_by, healthy_by, out_only_board, preserve_scale=(1.0 - conf),
        reference_board=healthy_board, exclude_minute_decliners=False,
    )
    current_adjusted = _stage_adjust(
        current_by, out_by, current_board, preserve_scale=1.0,
        reference_board=out_only_board, exclude_minute_decliners=True,
    )

    current_min_map = {
        str(r["Player"]): float(r.get("Projected Min", 0.0) or 0.0)
        for _, r in current_board.iterrows()
    }
    for name in names:
        mins = current_min_map.get(name, 0.0)
        p = profiles.get(name, {})
        if mins <= 0.1 or not p:
            continue
        base_fga_pm = max(float(p.get("FGA", 0.0)), 1e-6)
        base_3pa_pm = max(float(p.get("FG3A", 0.0)), 1e-6)
        base_fta_pm = max(float(p.get("FTA", 0.0)), 1e-6)
        base_ast_pm = max(float(p.get("AST", 0.0)), 1e-6)
        base_reb_pm = max(float(p.get("REB", 0.0)), 1e-6)

        # Combine OUT-stage and restriction-stage residual additions. Each stage
        # starts from its own current synthetic contribution; we take the extra
        # above that stage baseline and add it to today's raw contribution.
        def extra(stage_adj, stage_base, stat):
            return max(
                float(stage_adj.get(name, {}).get(stat, 0.0))
                - float(stage_base.get(name, {}).get(stat, 0.0)), 0.0
            )

        fga_total = float(current_by.get(name, {}).get("FGA", 0.0)) \
            + extra(out_adjusted, out_by, "FGA") + extra(current_adjusted, current_by, "FGA")
        three_total = float(current_by.get(name, {}).get("FG3A", 0.0)) \
            + extra(out_adjusted, out_by, "FG3A") + extra(current_adjusted, current_by, "FG3A")
        fta_total = float(current_by.get(name, {}).get("FTA", 0.0)) \
            + extra(out_adjusted, out_by, "FTA") + extra(current_adjusted, current_by, "FTA")
        ast_total = float(current_by.get(name, {}).get("AST", 0.0)) \
            + extra(out_adjusted, out_by, "AST") + extra(current_adjusted, current_by, "AST")
        reb_total = float(current_by.get(name, {}).get("REB", 0.0)) \
            + extra(out_adjusted, out_by, "REB") + extra(current_adjusted, current_by, "REB")

        usage = float(np.clip((fga_total / mins) / base_fga_pm, 0.85, 1.20))
        three_rate = float(np.clip((three_total / mins) / base_3pa_pm, 0.80, 1.25))
        fta_rate = float(np.clip((fta_total / mins) / base_fta_pm, 0.80, 1.25))
        creation = float(np.clip((ast_total / mins) / base_ast_pm, 0.80, 1.25))
        reb_role = float(np.clip((reb_total / mins) / base_reb_pm, 0.85, 1.15))
        # simulate_player multiplies 3PA and FTA by usage as well; use relative
        # modifiers so the same vacated shot is not counted twice.
        three_role = float(np.clip(three_rate / max(usage, 1e-6), 0.80, 1.25))
        fta_role = float(np.clip(fta_rate / max(usage, 1e-6), 0.80, 1.25))
        player_rows.append({
            "Player": name,
            "Usage fallback": usage,
            "Three-role fallback": three_role,
            "FTA-role fallback": fta_role,
            "Creation fallback": creation,
            "Rebound-role fallback": reb_role,
            "Current Min": mins,
            "OUT minutes shifted": out_shift,
            "Restriction minutes shifted": restriction_shift,
        })

    audit = pd.DataFrame(rows)
    audit.attrs["out_minutes_shifted"] = out_shift
    audit.attrs["restriction_minutes_shifted"] = restriction_shift
    return RotationStateImpact(
        modifiers=mods,
        team_audit=audit,
        current_minutes=current_board,
        out_only_minutes=out_only_board,
        healthy_minutes=healthy_board,
        raw_player_role_modifiers=pd.DataFrame(player_rows),
    )
