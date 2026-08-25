from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd

from core.buckets import WeightConfig, split_non_overlapping, active_weights
from core.redistribution import learn_redistribution_matrix, apply_role_aware_overrides


@dataclass
class MinutesProjection:
    player_id: object
    player_name: str
    team_abbr: str
    central_raw: float
    sd_raw: float
    low_raw: float
    high_raw: float
    starter_probability: float
    rotation_similarity: float
    regime: str


def _truthy_starter(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y", "starter", "start"}


def _case_lookup(d: dict, name: str, default=None):
    if not isinstance(d, dict):
        return default
    target = str(name).strip().casefold()
    for k, v in d.items():
        if str(k).strip().casefold() == target:
            return v
    return default


def _status(manual_context: dict, player_name: str) -> str:
    info = _case_lookup(manual_context.get("injuries", {}), player_name, {})
    if isinstance(info, dict):
        return str(info.get("status", "")).upper()
    return str(info or "").upper()


def _rotation_regime(manual_context: dict, team_name: str, team_abbr: str) -> str:
    # Accept either:
    # {"rotation_regime":{"LAS":"stable"}}
    # or the older nested game.rotation_regime structure.
    top = manual_context.get("rotation_regime", {})
    nested = manual_context.get("game", {}).get("rotation_regime", {})
    value = (
        _case_lookup(top, team_abbr, None)
        or _case_lookup(top, team_name, None)
        or _case_lookup(nested, team_abbr, None)
        or _case_lookup(nested, team_name, None)
    )
    v = str(value or "stable").strip().lower()
    return "role_change" if v in {"role_change", "change", "changed", "35/20/45"} else "stable"


def _projected_starters(manual_context: dict, team_name: str, team_abbr: str) -> set[str]:
    block = manual_context.get("projected_starters", {})
    vals = (
        _case_lookup(block, team_abbr, None)
        or _case_lookup(block, team_name, None)
        or []
    )
    return {str(x).strip().casefold() for x in vals}


def _team_game_margin(team_db: pd.DataFrame, game_id, team_abbr: str) -> float:
    rows = team_db[
        team_db["GAME_ID"].astype(str).eq(str(game_id))
    ].copy()
    if rows.empty or "PTS" not in rows.columns:
        return 0.0
    own = rows[rows["TEAM_ABBR"].astype(str).str.upper() == str(team_abbr).upper()]
    opp = rows[rows["TEAM_ABBR"].astype(str).str.upper() != str(team_abbr).upper()]
    if own.empty or opp.empty:
        return 0.0
    return float(own.iloc[0]["PTS"] - opp.iloc[0]["PTS"])


def _current_rotation_set(
    player_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    manual_context: dict,
    min_rotation_avg: float = 4.0,
) -> set[str]:
    names = []
    team_pool = current_pool[
        current_pool["TEAM_ABBR"].astype(str).str.upper() == str(team_abbr).upper()
    ]
    for _, row in team_pool.iterrows():
        pname = str(row["PLAYER_NAME"])
        if _status(manual_context, pname) == "OUT":
            continue
        pid = row["PLAYER_ID"]
        log = player_db[player_db["PLAYER_ID"] == pid].sort_values("GAME_DATE")
        if log.empty:
            continue
        recent = pd.to_numeric(log.tail(5)["MIN"], errors="coerce").dropna()
        avg = float(recent.mean()) if len(recent) else 0.0
        override = _case_lookup(
            manual_context.get("projected_minutes", {}), pname, None
        )
        if avg >= min_rotation_avg or override is not None:
            names.append(pname.casefold())
    return set(names)


def _historical_rotation_sets(player_db: pd.DataFrame, team_abbr: str) -> Dict[str, set[str]]:
    x = player_db[
        player_db["TEAM_ABBR"].astype(str).str.upper() == str(team_abbr).upper()
    ].copy()
    out: Dict[str, set[str]] = {}
    for gid, g in x.groupby("GAME_ID"):
        # Rotation similarity should focus on players who actually occupied
        # meaningful rotation minutes, not 1-minute garbage-time appearances.
        meaningful = g[pd.to_numeric(g["MIN"], errors="coerce").fillna(0) >= 4.0]
        out[str(gid)] = set(
            meaningful["PLAYER_NAME"].astype(str).str.casefold().tolist()
        )
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _starter_probability(player_log: pd.DataFrame) -> float:
    if "STARTER" not in player_log.columns or player_log.empty:
        return 0.5
    recent = player_log.sort_values("GAME_DATE").tail(5)
    vals = recent["STARTER"].map(_truthy_starter).astype(float)
    return float(vals.mean()) if len(vals) else 0.5


def _weighted_bucket_minutes(
    bucket: pd.DataFrame,
    current_rotation: set[str],
    historical_rotations: Dict[str, set[str]],
    today_starter: Optional[bool],
    team_db: pd.DataFrame,
    team_abbr: str,
):
    if bucket.empty:
        return np.nan, np.nan, np.nan

    vals, weights, sims = [], [], []

    for _, row in bucket.iterrows():
        gid = str(row["GAME_ID"])
        hist_rotation = historical_rotations.get(gid, set())
        sim = _jaccard(current_rotation, hist_rotation)

        # Similarity is a modifier inside the already non-overlapping bucket.
        # It does NOT create another sample or change the bucket's outer weight.
        w = 0.55 + 0.45 * sim

        if today_starter is not None and "STARTER" in row:
            historical_starter = _truthy_starter(row["STARTER"])
            w *= 1.10 if historical_starter == today_starter else 0.88

        # OT and large blowouts can distort minutes without representing normal
        # rotation expectations. Downweight rather than delete.
        if bool(row.get("OT_FLAG", False)):
            w *= 0.78

        margin = abs(_team_game_margin(team_db, gid, team_abbr))
        if margin >= 20:
            w *= 0.82
        elif margin >= 15:
            w *= 0.92

        m = float(row["MIN"])
        vals.append(m)
        weights.append(max(w, 0.05))
        sims.append(sim)

    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = float(np.average(vals, weights=weights))
    var = float(np.average((vals - mean) ** 2, weights=weights))
    similarity = float(np.average(np.asarray(sims), weights=weights))
    return mean, float(np.sqrt(max(var, 0.0))), similarity


def project_player_minutes_raw(
    player_log: pd.DataFrame,
    player_db: pd.DataFrame,
    team_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    team_name: str,
    player_name: str,
    player_id,
    manual_context: dict,
) -> MinutesProjection:
    x = player_log.sort_values("GAME_DATE").copy()

    regime = _rotation_regime(manual_context, team_name, team_abbr)
    cfg = WeightConfig.role_change() if regime == "role_change" else WeightConfig.stable()

    buckets = split_non_overlapping(x)
    outer = active_weights(buckets, cfg)

    current_rotation = _current_rotation_set(
        player_db, current_pool, team_abbr, manual_context
    )
    historical_rotations = _historical_rotation_sets(player_db, team_abbr)

    manual_starters = _projected_starters(manual_context, team_name, team_abbr)
    if manual_starters:
        today_starter = player_name.casefold() in manual_starters
        starter_prob = 1.0 if today_starter else 0.0
    else:
        starter_prob = _starter_probability(x)
        today_starter = starter_prob >= 0.60

    bucket_stats = {}
    for key in ("old", "mid", "l5"):
        bucket_stats[key] = _weighted_bucket_minutes(
            buckets[key],
            current_rotation,
            historical_rotations,
            today_starter,
            team_db,
            team_abbr,
        )

    means, sds, sims, ws = [], [], [], []
    for key in ("old", "mid", "l5"):
        mean, sd, sim = bucket_stats[key]
        w = outer.get(key, 0.0)
        if w > 0 and np.isfinite(mean):
            means.append(mean)
            sds.append(sd if np.isfinite(sd) else 2.0)
            sims.append(sim if np.isfinite(sim) else 0.5)
            ws.append(w)

    if not means:
        recent = pd.to_numeric(x.tail(5)["MIN"], errors="coerce").dropna()
        mean = float(recent.mean()) if len(recent) else 0.0
        sd = float(recent.std(ddof=0)) if len(recent) > 1 else 2.0
        similarity = 0.5
    else:
        ws = np.asarray(ws, dtype=float)
        ws = ws / ws.sum()
        mean = float(np.sum(np.asarray(means) * ws))
        # combine within-bucket variability + disagreement between buckets
        within = float(np.sum((np.asarray(sds) ** 2) * ws))
        between = float(np.sum(((np.asarray(means) - mean) ** 2) * ws))
        sd = float(np.sqrt(within + between))
        similarity = float(np.sum(np.asarray(sims) * ws))

    # Robust stabilizer against a single odd rotation game.
    recent5 = pd.to_numeric(x.tail(5)["MIN"], errors="coerce").dropna()
    if len(recent5):
        median5 = float(recent5.median())
        mean = 0.82 * mean + 0.18 * median5

    mean = float(np.clip(mean, 0.0, 40.0))
    sd = float(np.clip(max(sd, 1.25), 1.25, 5.5))
    low = float(np.clip(mean - 1.20 * sd, 0.0, 40.0))
    high = float(np.clip(mean + 1.20 * sd, 0.0, 40.0))

    return MinutesProjection(
        player_id=player_id,
        player_name=player_name,
        team_abbr=team_abbr,
        central_raw=mean,
        sd_raw=sd,
        low_raw=low,
        high_raw=high,
        starter_probability=starter_prob,
        rotation_similarity=similarity,
        regime=regime,
    )


def _allocate_capped(
    raw: Dict[str, float],
    fixed: Dict[str, float],
    total_minutes: float = 200.0,
    cap: float = 40.0,
) -> Dict[str, float]:
    """
    Allocate remaining team minutes proportionally while honoring fixed trader
    overrides and the WNBA 200-minute regulation constraint.
    """
    out = {k: float(np.clip(v, 0.0, cap)) for k, v in fixed.items()}
    fixed_sum = sum(out.values())
    remaining = max(total_minutes - fixed_sum, 0.0)

    free = {k: max(float(v), 0.0) for k, v in raw.items() if k not in out}
    if not free:
        return out

    active = dict(free)
    allocated = {k: 0.0 for k in free}

    for _ in range(20):
        if remaining <= 1e-8 or not active:
            break
        denom = sum(active.values())
        if denom <= 0:
            share = remaining / len(active)
            proposals = {k: share for k in active}
        else:
            proposals = {
                k: remaining * v / denom
                for k, v in active.items()
            }

        hit_cap = []
        used = 0.0
        for k, proposal in proposals.items():
            room = cap - allocated[k]
            add = min(proposal, room)
            allocated[k] += add
            used += add
            if room - add <= 1e-8:
                hit_cap.append(k)

        remaining -= used
        for k in hit_cap:
            active.pop(k, None)

        if used <= 1e-10:
            break

    out.update(allocated)
    return out



def project_team_minutes(
    player_db: pd.DataFrame,
    team_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    team_name: str,
    manual_context: dict,
    ui_overrides: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Context-aware AUTO rotation + 200-minute constraint.

    Important separation:
    - OUT availability is handled inside the AUTO rotation-similarity engine.
      We do not apply a second full pairwise redistribution for the same OUT.
    - Explicit metadata/trader minute targets are deviations from AUTO and use
      the historically learned role-aware redistribution matrix.
    """
    ui_overrides = ui_overrides or {}

    team_pool = current_pool[
        current_pool["TEAM_ABBR"].astype(str).str.upper()
        == str(team_abbr).upper()
    ].copy()

    rows = []
    raw = {}
    out_players = set()

    # First create context-aware AUTO raw minutes. Status OUT is fixed at zero
    # and excluded from current rotation similarity.
    for _, prow in team_pool.iterrows():
        pname = str(prow["PLAYER_NAME"])
        pid = prow["PLAYER_ID"]
        status = _status(manual_context, pname)

        log = player_db[player_db["PLAYER_ID"] == pid].copy()
        if log.empty:
            continue

        proj = project_player_minutes_raw(
            log, player_db, team_db, current_pool,
            team_abbr, team_name, pname, pid, manual_context
        )

        recent = pd.to_numeric(
            log.sort_values("GAME_DATE").tail(5)["MIN"],
            errors="coerce"
        ).dropna()
        recent_avg = float(recent.mean()) if len(recent) else 0.0

        if status == "OUT":
            out_players.add(pname)
            raw[pname] = 0.0
            source = "OUT"
        else:
            auto_raw = max(proj.central_raw, 0.0)
            if recent_avg < 2.5 and proj.central_raw < 3.0:
                auto_raw = min(auto_raw, 0.75)
            raw[pname] = auto_raw
            source = "AUTO"

        rows.append({
            "PLAYER_ID": pid,
            "Player": pname,
            "Team": team_abbr,
            "Pos": prow.get("POSITION_ABBR"),
            "Status": status or "ACTIVE/UNK",
            "Raw Auto Min": proj.central_raw,
            "Raw SD": proj.sd_raw,
            "Raw Low": proj.low_raw,
            "Raw High": proj.high_raw,
            "Starter P": proj.starter_probability,
            "Rotation Similarity": proj.rotation_similarity,
            "Regime": proj.regime,
            "Source": source,
        })

    # Context-aware AUTO baseline constrained to 200, with OUT fixed at zero.
    non_out_raw = {
        p: v for p, v in raw.items()
        if p not in out_players
    }
    base_alloc = _allocate_capped(
        non_out_raw,
        {p: 0.0 for p in out_players},
        total_minutes=200.0,
        cap=40.0,
    )

    # Explicit minute targets are trader-like deviations from AUTO.
    metadata = manual_context.get("projected_minutes", {}) or {}
    metadata_targets = {}
    for pname in base_alloc:
        v = _case_lookup(metadata, pname, None)
        if v is not None and pname not in out_players:
            metadata_targets[pname] = float(v)

    # UI override supersedes metadata.
    explicit_targets = dict(metadata_targets)
    for pname, val in ui_overrides.items():
        if pname in base_alloc and pname not in out_players:
            explicit_targets[pname] = float(val)

    matrix, matrix_audit = learn_redistribution_matrix(
        player_db, team_db, current_pool, team_abbr
    )

    final_alloc, impact = apply_role_aware_overrides(
        base_alloc,
        explicit_targets,
        matrix,
        out_players=out_players,
        total_minutes=200.0,
    )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Auto Baseline Min"] = out["Player"].map(base_alloc).fillna(0.0)
    out["Projected Min"] = out["Player"].map(final_alloc).fillna(0.0)
    out["Override Delta"] = out["Projected Min"] - out["Auto Baseline Min"]

    # Source labels
    def source_for(name):
        if name in out_players:
            return "OUT"
        if name in ui_overrides and float(ui_overrides[name]) > 0:
            return "TRADER"
        if name in metadata_targets:
            return "METADATA"
        return "AUTO"

    out["Source"] = out["Player"].map(source_for)

    # Preserve AUTO uncertainty shape around the final constrained central.
    ratio = np.where(
        pd.to_numeric(out["Raw Auto Min"], errors="coerce")
        .fillna(0).to_numpy() > 0.5,
        out["Projected Min"].to_numpy()
        / np.maximum(out["Raw Auto Min"].to_numpy(), 0.5),
        1.0,
    )
    out["Minutes SD"] = np.clip(
        out["Raw SD"].to_numpy()
        * np.sqrt(np.clip(ratio, 0.6, 1.6)),
        1.0, 5.5
    )
    out["Low Min"] = np.clip(
        out["Projected Min"] - 1.20 * out["Minutes SD"],
        0, 40
    )
    out["High Min"] = np.clip(
        out["Projected Min"] + 1.20 * out["Minutes SD"],
        0, 40
    )

    fixed_mask = out["Source"].isin(["TRADER", "METADATA"])
    out.loc[fixed_mask, "Minutes SD"] = np.minimum(
        out.loc[fixed_mask, "Minutes SD"],
        2.25
    )
    out.loc[fixed_mask, "Low Min"] = np.clip(
        out.loc[fixed_mask, "Projected Min"]
        - 1.20 * out.loc[fixed_mask, "Minutes SD"],
        0, 40
    )
    out.loc[fixed_mask, "High Min"] = np.clip(
        out.loc[fixed_mask, "Projected Min"]
        + 1.20 * out.loc[fixed_mask, "Minutes SD"],
        0, 40
    )
    out.loc[
        out["Source"] == "OUT",
        ["Minutes SD", "Low Min", "High Min"]
    ] = 0.0

    # Store compact audit objects in dataframe attrs so the Streamlit layer can
    # expose them without changing the model interface.
    out.attrs["redistribution_matrix_audit"] = matrix_audit
    out.attrs["override_impact"] = impact

    return out.sort_values(
        "Projected Min", ascending=False
    ).reset_index(drop=True)



# ---------------------------------------------------------------------
# Public team-context helpers used by Team Markets
# ---------------------------------------------------------------------
def rotation_regime_for_team(
    manual_context: dict,
    team_name: str,
    team_abbr: str,
) -> str:
    """Public wrapper so Team Markets and Player Props use the same regime parser."""
    return _rotation_regime(manual_context, team_name, team_abbr)


def rotation_similarity_weights(
    player_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    manual_context: dict,
) -> Dict[str, float]:
    """
    Current-rotation similarity modifier by historical GAME_ID.

    This is an INNER-bucket modifier only. It does not create another sample
    and therefore does not double count Old/G6-10/L5 outer weights.
    The same 0.55 + 0.45*Jaccard structure used by the minutes engine is reused
    here so team-market history is softly tilted toward games with a more
    comparable rotation, especially after fresh OUT/IN changes.
    """
    current = _current_rotation_set(
        player_db, current_pool, team_abbr, manual_context
    )
    historical = _historical_rotation_sets(player_db, team_abbr)
    out: Dict[str, float] = {}
    for gid, hist in historical.items():
        sim = _jaccard(current, hist)
        out[str(gid)] = 0.55 + 0.45 * sim
    return out
