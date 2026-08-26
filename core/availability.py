from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AvailabilityAudit:
    team_abbr: str
    out_players: tuple[str, ...]
    eligibility_start: Optional[pd.Timestamp]
    eligible_games: int
    exact_games: int
    natural_share: float
    confidence: float
    target_share: float
    exact_weight: float
    other_weight: float
    k: float

    def as_dict(self) -> dict:
        return {
            "Team": self.team_abbr,
            "Confirmed OUT state": ", ".join(self.out_players) if self.out_players else "—",
            "Eligibility start": self.eligibility_start.date().isoformat() if self.eligibility_start is not None else "—",
            "Eligible games": self.eligible_games,
            "Exact-state games": self.exact_games,
            "Natural exact share": self.natural_share,
            "State confidence": self.confidence,
            "Target exact share": self.target_share,
            "Exact-game inner weight": self.exact_weight,
            "Other eligible inner weight": self.other_weight,
            "Shrink K": self.k,
        }


def _norm_name(v) -> str:
    return str(v).strip().casefold()


def _status(info) -> str:
    if isinstance(info, dict):
        return str(info.get("status", "")).strip().upper()
    return str(info or "").strip().upper()


def _player_team_from_pool(current_pool: pd.DataFrame, player_name: str) -> str:
    if current_pool is None or current_pool.empty:
        return ""
    hit = current_pool[
        current_pool["PLAYER_NAME"].astype(str).str.casefold() == _norm_name(player_name)
    ]
    if hit.empty:
        return ""
    return str(hit.iloc[0].get("TEAM_ABBR", "")).upper()


def confirmed_out_players(
    manual_context: dict,
    current_pool: pd.DataFrame,
    team_abbr: str,
) -> list[str]:
    """Return only CONFIRMED OUT players for the requested team.

    QUESTIONABLE/GTD/DOUBTFUL are intentionally ignored by the availability-state
    engine. The user explicitly decides when a player becomes OUT.
    """
    injuries = (manual_context or {}).get("injuries", {}) or {}
    out = []
    for name, info in injuries.items():
        if _status(info) != "OUT":
            continue
        info_team = str(info.get("team", "") if isinstance(info, dict) else "").upper()
        pool_team = _player_team_from_pool(current_pool, str(name))
        if info_team == str(team_abbr).upper() or pool_team == str(team_abbr).upper():
            out.append(str(name))
    # Stable order for audit/session reproducibility.
    return sorted(set(out), key=str.casefold)


def _first_team_appearance(
    player_db: pd.DataFrame,
    team_abbr: str,
    player_name: str,
) -> Optional[pd.Timestamp]:
    if player_db is None or player_db.empty:
        return None
    x = player_db[
        player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
        & player_db["PLAYER_NAME"].astype(str).str.casefold().eq(_norm_name(player_name))
    ].copy()
    if x.empty:
        return None
    dates = pd.to_datetime(x["GAME_DATE"], errors="coerce").dropna()
    return dates.min() if len(dates) else None


def _historical_presence(
    player_db: pd.DataFrame,
    team_abbr: str,
    player_names: Iterable[str],
    min_minutes: float = 1.0,
) -> Dict[str, set[str]]:
    names = {_norm_name(x) for x in player_names}
    if not names or player_db is None or player_db.empty:
        return {}
    x = player_db[
        player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
        & player_db["PLAYER_NAME"].astype(str).str.casefold().isin(names)
    ].copy()
    mins = pd.to_numeric(x.get("MIN", 0), errors="coerce").fillna(0.0)
    x = x[mins >= float(min_minutes)].copy()
    out: Dict[str, set[str]] = {}
    for gid, g in x.groupby("GAME_ID"):
        out[str(gid)] = set(g["PLAYER_NAME"].astype(str).str.casefold().tolist())
    return out


def availability_state_weights(
    player_db: pd.DataFrame,
    team_log: pd.DataFrame,
    team_abbr: str,
    out_players: Iterable[str],
    k: float = 6.0,
    max_target_share: float = 0.70,
    exclude_opponent_abbr: str | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame, set[str]]:
    """Exact confirmed-OUT state reweighting with disjoint evidence groups.

    The historical games are divided into two DISJOINT groups once every selected
    OUT player had first appeared for this team:
      exact: all selected OUT players absent from that game
      other: at least one selected OUT player played

    No fourth sample is created. Instead, exact/other rows receive inner weights
    whose total exact-state share is driven by N/(N+K), but is never lower than
    the natural historical exact-state share. This is transparent partial pooling.

    Games before all selected players had first appeared for the team are neutral
    (weight 1.0) rather than falsely classified as 'OUT' games.
    """
    out_players = tuple(sorted({str(x) for x in out_players if str(x).strip()}, key=str.casefold))
    neutral_audit = AvailabilityAudit(
        team_abbr=str(team_abbr).upper(), out_players=out_players,
        eligibility_start=None, eligible_games=0, exact_games=0,
        natural_share=0.0, confidence=0.0, target_share=0.0,
        exact_weight=1.0, other_weight=1.0, k=float(k),
    )
    if not out_players or team_log is None or team_log.empty:
        return {}, pd.DataFrame([neutral_audit.as_dict()]), set()

    starts = [_first_team_appearance(player_db, team_abbr, p) for p in out_players]
    if any(s is None for s in starts):
        # If we cannot establish that every selected player was actually on this
        # team historically, do not treat pre-roster games as injury matches.
        return {}, pd.DataFrame([neutral_audit.as_dict()]), set()

    eligibility_start = max(starts)
    t = team_log.copy()
    t["_DATE"] = pd.to_datetime(t["GAME_DATE"], errors="coerce")
    eligible = t[t["_DATE"] >= eligibility_start].copy()
    if exclude_opponent_abbr and "OPP_ABBR" in eligible.columns:
        eligible = eligible[
            ~eligible["OPP_ABBR"].astype(str).str.upper().eq(str(exclude_opponent_abbr).upper())
        ].copy()
    if eligible.empty:
        audit = neutral_audit
        audit.eligibility_start = eligibility_start
        return {}, pd.DataFrame([audit.as_dict()]), set()

    selected_norm = {_norm_name(x) for x in out_players}
    presence = _historical_presence(player_db, team_abbr, out_players, min_minutes=1.0)

    exact_ids: set[str] = set()
    eligible_ids = [str(g) for g in eligible["GAME_ID"].astype(str).tolist()]
    for gid in eligible_ids:
        played = presence.get(str(gid), set())
        if selected_norm.isdisjoint(played):
            exact_ids.add(str(gid))

    n = len(eligible_ids)
    n_exact = len(exact_ids)
    n_other = n - n_exact
    if n_exact == 0 or n_other == 0:
        audit = AvailabilityAudit(
            team_abbr=str(team_abbr).upper(), out_players=out_players,
            eligibility_start=eligibility_start, eligible_games=n,
            exact_games=n_exact, natural_share=(n_exact / n if n else 0.0),
            # No contrast group => no identifiable exact-state effect. Keep
            # confidence neutral so the roster-synthetic fallback is not muted.
            confidence=0.0,
            target_share=(n_exact / n if n else 0.0),
            exact_weight=1.0, other_weight=1.0, k=float(k),
        )
        return {}, pd.DataFrame([audit.as_dict()]), exact_ids

    natural = n_exact / n
    confidence = n_exact / (n_exact + float(k))
    target = min(float(max_target_share), max(float(natural), float(confidence)))

    # Choose per-row weights so exact-state rows sum to target*n and other rows
    # sum to (1-target)*n. Mean eligible inner weight therefore stays exactly 1.
    w_exact = target * n / n_exact
    w_other = (1.0 - target) * n / n_other

    weights: Dict[str, float] = {}
    for gid in eligible_ids:
        weights[str(gid)] = float(w_exact if str(gid) in exact_ids else w_other)

    audit = AvailabilityAudit(
        team_abbr=str(team_abbr).upper(), out_players=out_players,
        eligibility_start=eligibility_start, eligible_games=n,
        exact_games=n_exact, natural_share=natural, confidence=confidence,
        target_share=target, exact_weight=float(w_exact),
        other_weight=float(w_other), k=float(k),
    )
    return weights, pd.DataFrame([audit.as_dict()]), exact_ids


def combine_game_weights(*weight_maps: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Multiply independent INNER-bucket relevance weights.

    This function deliberately does not introduce any outer sample weight. The
    Old/G6-10/L5 weights remain unchanged. Missing ids are neutral (1.0).
    """
    maps = [m for m in weight_maps if m]
    if not maps:
        return {}
    ids = set().union(*(set(m.keys()) for m in maps))
    out = {}
    for gid in ids:
        w = 1.0
        for m in maps:
            w *= float(m.get(str(gid), 1.0))
        out[str(gid)] = float(np.clip(w, 0.20, 4.00))
    return out
