"""
WNBA Prop Projection + Monte Carlo Engine
=========================================
Starter framework for automating the process we discussed:

1) Project minutes / role
2) Project opportunities (FGA, 3PA, FTA, REB chances/rates, AST creation)
3) Apply pace, opponent overall, positional and H2H adjustments
4) Regress shooting efficiency
5) Simulate player outcomes with Monte Carlo
6) Price single props and combo props
7) Compare with bookmaker no-vig probability
8) Calculate EV and fractional-Kelly stake

IMPORTANT
---------
The coefficients below are MODEL DEFAULTS, not coefficients fitted on historical WNBA
prop data yet. The automation project should later estimate/calibrate them on a backtest.

The most important design principle is:
    Minutes / Role / Opportunities >>> Recent box-score outcomes.

So a recent scoring slump does NOT automatically lower a points projection if
FGA / 3PA / FTA and minutes remain healthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import math
import numpy as np


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

EPS = 1e-9


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def weighted_mean(values: List[Tuple[float, float]]) -> float:
    """
    values = [(value, weight), ...]
    Ignores NaN/None-like values.
    """
    good = [(v, w) for v, w in values if v is not None and np.isfinite(v) and w > 0]
    if not good:
        raise ValueError("No valid values supplied to weighted_mean.")
    denom = sum(w for _, w in good)
    return sum(v * w for v, w in good) / denom


def no_vig_prob(over_odds: float, under_odds: float, side: str = "over") -> float:
    """Remove two-way bookmaker margin."""
    p_over_raw = 1.0 / over_odds
    p_under_raw = 1.0 / under_odds
    denom = p_over_raw + p_under_raw
    if side.lower() == "over":
        return p_over_raw / denom
    if side.lower() == "under":
        return p_under_raw / denom
    raise ValueError("side must be 'over' or 'under'")


def fair_odds(prob: float) -> float:
    return math.inf if prob <= 0 else 1.0 / prob


def expected_value(prob: float, decimal_odds: float) -> float:
    """Expected return per 1 unit staked."""
    return prob * decimal_odds - 1.0


def full_kelly_fraction(prob: float, decimal_odds: float) -> float:
    """
    Kelly fraction of bankroll.
    Returns 0 for negative Kelly.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    k = (b * prob - q) / b
    return max(0.0, k)


def fractional_kelly_units(
    prob: float,
    decimal_odds: float,
    kelly_fraction: float = 0.25,
    bankroll_units: float = 100.0,
    max_units: float = 1.25,
    reliability_multiplier: float = 1.0,
    correlation_multiplier: float = 1.0,
) -> float:
    """
    Converts fractional Kelly to units.
    Example: 100 bankroll units, 1% bankroll = 1 unit.
    """
    full_k = full_kelly_fraction(prob, decimal_odds)
    raw_units = full_k * kelly_fraction * bankroll_units
    adjusted = raw_units * reliability_multiplier * correlation_multiplier
    return clamp(adjusted, 0.0, max_units)


def gamma_poisson_sample(
    rng: np.random.Generator,
    mean: np.ndarray | float,
    dispersion: float,
) -> np.ndarray:
    """
    Gamma-Poisson mixture = Negative Binomial style overdispersed counts.

    Var[X] = mean + mean^2 / dispersion

    Higher dispersion -> closer to Poisson.
    """
    mean_arr = np.asarray(mean, dtype=float)
    mean_arr = np.maximum(mean_arr, EPS)
    shape = max(dispersion, EPS)
    scale = mean_arr / shape
    lam = rng.gamma(shape=shape, scale=scale)
    return rng.poisson(lam)


# ---------------------------------------------------------------------
# Input structures
# ---------------------------------------------------------------------

@dataclass
class RateWindow:
    """
    Per-minute or percentage statistics from different windows.

    The model intentionally keeps 'recent same-role' separate from raw recent
    games. This is crucial when rotation / injuries changed.
    """
    season: float
    last10: float
    recent_role: float


@dataclass
class EfficiencyWindow:
    season: float
    recent: float
    opponent_position_allowed: float
    league_average: float


@dataclass
class PlayerProfile:
    name: str

    # Minutes
    minutes_season: float
    minutes_last10: float
    minutes_recent_role: float
    minutes_floor: float
    minutes_ceiling: float
    minutes_sd: float = 2.5

    # Opportunity rates per minute
    fga_per_min: RateWindow = field(default_factory=lambda: RateWindow(0, 0, 0))
    three_pa_per_min: RateWindow = field(default_factory=lambda: RateWindow(0, 0, 0))
    fta_per_min: RateWindow = field(default_factory=lambda: RateWindow(0, 0, 0))
    reb_per_min: RateWindow = field(default_factory=lambda: RateWindow(0, 0, 0))
    ast_per_min: RateWindow = field(default_factory=lambda: RateWindow(0, 0, 0))

    # Optional more granular rebound inputs
    oreb_per_min: Optional[RateWindow] = None
    dreb_per_min: Optional[RateWindow] = None

    # Shooting efficiency
    two_pt_pct: EfficiencyWindow = field(
        default_factory=lambda: EfficiencyWindow(.50, .50, .50, .50)
    )
    three_pt_pct: EfficiencyWindow = field(
        default_factory=lambda: EfficiencyWindow(.33, .33, .33, .33)
    )
    ft_pct: EfficiencyWindow = field(
        default_factory=lambda: EfficiencyWindow(.80, .80, .80, .80)
    )

    # Optional "vacated opportunity" bonuses.
    # Set to zero if recent_role ALREADY reflects the current absence/rotation.
    extra_fga: float = 0.0
    extra_three_pa: float = 0.0
    extra_fta: float = 0.0
    extra_reb: float = 0.0
    extra_ast: float = 0.0


@dataclass
class MatchupContext:
    # Pace
    team_pace: float
    opponent_pace: float
    league_pace: float

    # Opponent overall allowance indices:
    # 1.00 = league average; 1.05 = allows 5% more than average.
    overall_indices: Dict[str, float]

    # Opponent positional allowance indices.
    positional_indices: Dict[str, float]

    # H2H rate indices AFTER:
    # - removing OT impact
    # - filtering non-comparable rotations
    # - converting to same role/minutes basis
    h2h_indices: Dict[str, float] = field(default_factory=dict)

    # H2H confidence 0 to 0.10 by design.
    h2h_weight: float = 0.05

    # Role / lineup modifiers.
    starter_minutes_delta: float = 0.0
    injury_minutes_delta: float = 0.0
    rotation_minutes_delta: float = 0.0
    blowout_minutes_delta: float = 0.0

    # Shared game-environment uncertainty for Monte Carlo.
    pace_sd_pct: float = 0.035
    offense_environment_sd_pct: float = 0.06
    rebound_environment_sd_pct: float = 0.07
    creation_environment_sd_pct: float = 0.07
    foul_environment_sd_pct: float = 0.09


@dataclass
class ModelWeights:
    # Minutes
    minutes_recent_role: float = 0.50
    minutes_last10: float = 0.30
    minutes_season: float = 0.20

    # Opportunity rates
    rate_recent_role: float = 0.45
    rate_last10: float = 0.30
    rate_season: float = 0.25

    # Shooting regression
    efficiency_season: float = 0.60
    efficiency_recent: float = 0.15
    efficiency_opponent_position: float = 0.15
    efficiency_league: float = 0.10

    # Matchup shrinkage
    pace_strength: float = 0.50
    overall_strength: float = 0.30
    positional_strength: float = 0.35

    # Monte Carlo dispersion: higher = less variance
    fga_dispersion: float = 20.0
    fta_dispersion: float = 10.0
    reb_dispersion: float = 8.0
    ast_dispersion: float = 7.0


@dataclass
class Projection:
    minutes: float
    fga: float
    three_pa: float
    two_pa: float
    fta: float
    two_pt_pct: float
    three_pt_pct: float
    ft_pct: float
    points: float
    rebounds: float
    assists: float
    three_pm: float


@dataclass
class MarketResult:
    market: str
    line: float
    side: str
    model_probability: float
    fair_odds: float
    bookmaker_odds: Optional[float] = None
    no_vig_market_probability: Optional[float] = None
    probability_edge_pp: Optional[float] = None
    ev_per_unit: Optional[float] = None
    suggested_units: Optional[float] = None


# ---------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------

class WNBAPropModel:
    def __init__(self, weights: Optional[ModelWeights] = None, seed: int = 42):
        self.w = weights or ModelWeights()
        self.seed = seed

    # ----------------------------
    # Deterministic projection
    # ----------------------------

    def project_minutes(self, p: PlayerProfile, c: MatchupContext) -> float:
        base = weighted_mean([
            (p.minutes_recent_role, self.w.minutes_recent_role),
            (p.minutes_last10, self.w.minutes_last10),
            (p.minutes_season, self.w.minutes_season),
        ])

        minutes = (
            base
            + c.starter_minutes_delta
            + c.injury_minutes_delta
            + c.rotation_minutes_delta
            + c.blowout_minutes_delta
        )
        return clamp(minutes, p.minutes_floor, p.minutes_ceiling)

    def blend_rate(self, r: RateWindow) -> float:
        return weighted_mean([
            (r.recent_role, self.w.rate_recent_role),
            (r.last10, self.w.rate_last10),
            (r.season, self.w.rate_season),
        ])

    def blend_efficiency(self, e: EfficiencyWindow) -> float:
        return weighted_mean([
            (e.season, self.w.efficiency_season),
            (e.recent, self.w.efficiency_recent),
            (e.opponent_position_allowed, self.w.efficiency_opponent_position),
            (e.league_average, self.w.efficiency_league),
        ])

    def pace_adjustment(self, c: MatchupContext) -> float:
        expected_pace = 0.5 * (c.team_pace + c.opponent_pace)
        raw_factor = expected_pace / max(c.league_pace, EPS)
        return 1.0 + self.w.pace_strength * (raw_factor - 1.0)

    def matchup_adjustment(self, stat: str, c: MatchupContext) -> float:
        overall = c.overall_indices.get(stat, 1.0)
        positional = c.positional_indices.get(stat, 1.0)
        h2h_index = c.h2h_indices.get(stat, 1.0)

        overall_adj = 1.0 + self.w.overall_strength * (overall - 1.0)
        positional_adj = 1.0 + self.w.positional_strength * (positional - 1.0)

        h2h_w = clamp(c.h2h_weight, 0.0, 0.10)
        h2h_adj = 1.0 + h2h_w * (h2h_index - 1.0)

        return overall_adj * positional_adj * h2h_adj

    def opportunity_projection(
        self,
        rate: RateWindow,
        minutes: float,
        extra: float,
        stat: str,
        c: MatchupContext,
    ) -> float:
        base = minutes * self.blend_rate(rate)
        raw = base + extra
        return max(
            0.0,
            raw
            * self.pace_adjustment(c)
            * self.matchup_adjustment(stat, c)
        )

    def project(self, p: PlayerProfile, c: MatchupContext) -> Projection:
        minutes = self.project_minutes(p, c)

        fga = self.opportunity_projection(
            p.fga_per_min, minutes, p.extra_fga, "fga", c
        )
        three_pa = self.opportunity_projection(
            p.three_pa_per_min, minutes, p.extra_three_pa, "3pa", c
        )
        fta = self.opportunity_projection(
            p.fta_per_min, minutes, p.extra_fta, "fta", c
        )

        # Keep 3PA logically inside FGA.
        three_pa = min(three_pa, fga)
        two_pa = max(0.0, fga - three_pa)

        p2 = clamp(self.blend_efficiency(p.two_pt_pct), 0.0, 1.0)
        p3 = clamp(self.blend_efficiency(p.three_pt_pct), 0.0, 1.0)
        pft = clamp(self.blend_efficiency(p.ft_pct), 0.0, 1.0)

        points = 2.0 * two_pa * p2 + 3.0 * three_pa * p3 + fta * pft
        three_pm = three_pa * p3

        if p.oreb_per_min is not None and p.dreb_per_min is not None:
            oreb = self.opportunity_projection(
                p.oreb_per_min, minutes, 0.0, "oreb", c
            )
            dreb = self.opportunity_projection(
                p.dreb_per_min, minutes, 0.0, "dreb", c
            )
            rebounds = oreb + dreb + p.extra_reb
        else:
            rebounds = self.opportunity_projection(
                p.reb_per_min, minutes, p.extra_reb, "reb", c
            )

        assists = self.opportunity_projection(
            p.ast_per_min, minutes, p.extra_ast, "ast", c
        )

        return Projection(
            minutes=minutes,
            fga=fga,
            three_pa=three_pa,
            two_pa=two_pa,
            fta=fta,
            two_pt_pct=p2,
            three_pt_pct=p3,
            ft_pct=pft,
            points=points,
            rebounds=rebounds,
            assists=assists,
            three_pm=three_pm,
        )

    # ----------------------------
    # Monte Carlo
    # ----------------------------

    def simulate_player(
        self,
        p: PlayerProfile,
        c: MatchupContext,
        n_sims: int = 100_000,
    ) -> Dict[str, np.ndarray]:
        """
        Simulates:
          MIN, FGA, 3PA, FTA, 2PM, 3PM, FTM, PTS, REB, AST,
          P+R, P+A, A+R, PRA.

        Shared latent factors introduce realistic positive correlation:
          pace factor
          offense factor
          rebound factor
          creation factor
          foul factor
        """
        base = self.project(p, c)
        rng = np.random.default_rng(self.seed)

        # Shared game factors, clipped to avoid nonsensical tails.
        pace_factor = np.clip(
            rng.normal(1.0, c.pace_sd_pct, n_sims), 0.88, 1.14
        )
        offense_factor = np.clip(
            rng.normal(1.0, c.offense_environment_sd_pct, n_sims), 0.80, 1.22
        )
        rebound_factor = np.clip(
            rng.normal(1.0, c.rebound_environment_sd_pct, n_sims), 0.78, 1.25
        )
        creation_factor = np.clip(
            rng.normal(1.0, c.creation_environment_sd_pct, n_sims), 0.78, 1.25
        )
        foul_factor = np.clip(
            rng.normal(1.0, c.foul_environment_sd_pct, n_sims), 0.72, 1.32
        )

        # Minutes
        minutes = np.clip(
            rng.normal(base.minutes, p.minutes_sd, n_sims),
            p.minutes_floor,
            p.minutes_ceiling,
        )
        minute_ratio = minutes / max(base.minutes, EPS)

        # Attempts
        fga_mean = np.maximum(
            base.fga * minute_ratio * pace_factor * offense_factor, 0.05
        )
        fga = gamma_poisson_sample(rng, fga_mean, self.w.fga_dispersion)

        # 3PA share conditional on FGA.
        base_three_share = clamp(base.three_pa / max(base.fga, EPS), 0.0, 1.0)
        # Small game-to-game shot-profile fluctuation.
        three_share = np.clip(
            rng.normal(base_three_share, 0.045, n_sims), 0.0, 0.90
        )
        three_pa = rng.binomial(fga, three_share)
        two_pa = fga - three_pa

        fta_mean = np.maximum(
            base.fta * minute_ratio * pace_factor * offense_factor * foul_factor,
            0.02,
        )
        fta = gamma_poisson_sample(rng, fta_mean, self.w.fta_dispersion)

        # Shooting:
        # Add modest per-game efficiency uncertainty around regressed means.
        p2_game = np.clip(rng.normal(base.two_pt_pct, 0.045, n_sims), 0.15, 0.80)
        p3_game = np.clip(rng.normal(base.three_pt_pct, 0.050, n_sims), 0.05, 0.65)
        pft_game = np.clip(rng.normal(base.ft_pct, 0.035, n_sims), 0.40, 0.98)

        two_pm = rng.binomial(two_pa, p2_game)
        three_pm = rng.binomial(three_pa, p3_game)
        ftm = rng.binomial(fta, pft_game)

        points = 2 * two_pm + 3 * three_pm + ftm

        # Rebounds / Assists:
        reb_mean = np.maximum(
            base.rebounds * minute_ratio * pace_factor * rebound_factor, 0.02
        )
        rebounds = gamma_poisson_sample(rng, reb_mean, self.w.reb_dispersion)

        ast_mean = np.maximum(
            base.assists * minute_ratio * pace_factor * creation_factor, 0.02
        )
        assists = gamma_poisson_sample(rng, ast_mean, self.w.ast_dispersion)

        return {
            "MIN": minutes,
            "FGA": fga,
            "3PA": three_pa,
            "FTA": fta,
            "2PM": two_pm,
            "3PM": three_pm,
            "FTM": ftm,
            "PTS": points,
            "REB": rebounds,
            "AST": assists,
            "P+R": points + rebounds,
            "P+A": points + assists,
            "A+R": assists + rebounds,
            "PRA": points + rebounds + assists,
        }

    # ----------------------------
    # Market pricing
    # ----------------------------

    @staticmethod
    def probability_from_sims(
        sims: np.ndarray,
        line: float,
        side: str,
    ) -> float:
        side = side.lower()
        if side == "over":
            return float(np.mean(sims > line))
        if side == "under":
            return float(np.mean(sims < line))
        raise ValueError("side must be 'over' or 'under'")

    def price_market(
        self,
        simulations: Dict[str, np.ndarray],
        market: str,
        line: float,
        side: str,
        bookmaker_odds: Optional[float] = None,
        opposite_odds: Optional[float] = None,
        reliability_multiplier: float = 1.0,
        correlation_multiplier: float = 1.0,
        max_units: float = 1.25,
    ) -> MarketResult:
        if market not in simulations:
            raise KeyError(f"Market '{market}' not found in simulations.")

        prob = self.probability_from_sims(simulations[market], line, side)
        result = MarketResult(
            market=market,
            line=line,
            side=side,
            model_probability=prob,
            fair_odds=fair_odds(prob),
            bookmaker_odds=bookmaker_odds,
        )

        if bookmaker_odds is not None:
            result.ev_per_unit = expected_value(prob, bookmaker_odds)
            result.suggested_units = fractional_kelly_units(
                prob=prob,
                decimal_odds=bookmaker_odds,
                kelly_fraction=0.25,
                bankroll_units=100.0,
                max_units=max_units,
                reliability_multiplier=reliability_multiplier,
                correlation_multiplier=correlation_multiplier,
            )

        if bookmaker_odds is not None and opposite_odds is not None:
            market_prob = no_vig_prob(
                bookmaker_odds if side == "over" else opposite_odds,
                opposite_odds if side == "over" else bookmaker_odds,
                side=side,
            )
            result.no_vig_market_probability = market_prob
            result.probability_edge_pp = 100.0 * (prob - market_prob)

        return result

    # ----------------------------
    # Stress testing
    # ----------------------------

    def stress_test_projection(
        self,
        projection: Projection,
        points_mean_delta: float = -0.75,
        rebounds_mean_delta: float = -0.50,
        assists_mean_delta: float = -0.30,
        three_pa_multiplier: float = 0.96,
        fta_multiplier: float = 0.96,
    ) -> Dict[str, float]:
        """
        Simple transparent stress scenario.
        This is intentionally conservative and can later be replaced with
        parameter-bootstrap / posterior uncertainty.
        """
        return {
            "PTS": max(0.0, projection.points + points_mean_delta),
            "REB": max(0.0, projection.rebounds + rebounds_mean_delta),
            "AST": max(0.0, projection.assists + assists_mean_delta),
            "3PA": max(0.0, projection.three_pa * three_pa_multiplier),
            "FTA": max(0.0, projection.fta * fta_multiplier),
            "P+R": max(0.0, projection.points + points_mean_delta
                       + projection.rebounds + rebounds_mean_delta),
            "P+A": max(0.0, projection.points + points_mean_delta
                       + projection.assists + assists_mean_delta),
            "PRA": max(0.0, projection.points + points_mean_delta
                       + projection.rebounds + rebounds_mean_delta
                       + projection.assists + assists_mean_delta),
        }


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # IMPORTANT:
    # These are synthetic/example values only.
    # Replace with automatically collected WNBA data later.

    player = PlayerProfile(
        name="Example Player",
        minutes_season=28.0,
        minutes_last10=31.0,
        minutes_recent_role=34.0,
        minutes_floor=27.0,
        minutes_ceiling=37.0,
        minutes_sd=2.2,

        fga_per_min=RateWindow(
            season=10.0 / 28.0,
            last10=11.5 / 31.0,
            recent_role=12.2 / 34.0,
        ),
        three_pa_per_min=RateWindow(
            season=3.2 / 28.0,
            last10=4.3 / 31.0,
            recent_role=4.8 / 34.0,
        ),
        fta_per_min=RateWindow(
            season=2.5 / 28.0,
            last10=2.8 / 31.0,
            recent_role=3.1 / 34.0,
        ),
        reb_per_min=RateWindow(
            season=4.1 / 28.0,
            last10=4.4 / 31.0,
            recent_role=4.8 / 34.0,
        ),
        ast_per_min=RateWindow(
            season=3.5 / 28.0,
            last10=4.1 / 31.0,
            recent_role=5.0 / 34.0,
        ),

        two_pt_pct=EfficiencyWindow(
            season=.48,
            recent=.43,
            opponent_position_allowed=.50,
            league_average=.50,
        ),
        three_pt_pct=EfficiencyWindow(
            season=.33,
            recent=.24,  # recent slump is deliberately LOW weighted
            opponent_position_allowed=.34,
            league_average=.34,
        ),
        ft_pct=EfficiencyWindow(
            season=.80,
            recent=.77,
            opponent_position_allowed=.79,
            league_average=.79,
        ),

        # Leave these at 0 if recent_role already includes the current injury setup.
        extra_fga=0.0,
        extra_three_pa=0.0,
        extra_fta=0.0,
        extra_reb=0.0,
        extra_ast=0.0,
    )

    matchup = MatchupContext(
        team_pace=80.5,
        opponent_pace=79.0,
        league_pace=80.0,

        overall_indices={
            "fga": 1.01,
            "3pa": 1.04,
            "fta": 1.06,
            "reb": 1.03,
            "ast": 0.98,
        },
        positional_indices={
            "fga": 1.03,
            "3pa": 1.06,
            "fta": 1.04,
            "reb": 1.02,
            "ast": 1.01,
        },
        h2h_indices={
            "fga": 1.02,
            "3pa": 1.00,
            "fta": 1.05,
            "reb": 1.00,
            "ast": 0.98,
        },
        h2h_weight=0.05,
    )

    model = WNBAPropModel(seed=20260823)

    proj = model.project(player, matchup)
    print("\nDETERMINISTIC PROJECTION")
    print(proj)

    sims = model.simulate_player(player, matchup, n_sims=100_000)

    result = model.price_market(
        simulations=sims,
        market="P+A",
        line=16.5,
        side="over",
        bookmaker_odds=1.95,
        opposite_odds=1.80,
        reliability_multiplier=0.90,
        correlation_multiplier=1.00,
    )

    print("\nMARKET RESULT")
    print(result)

    print("\nSIMULATION MEANS")
    for key in ["PTS", "REB", "AST", "3PM", "P+R", "P+A", "PRA"]:
        print(f"{key:>4}: {np.mean(sims[key]):.2f}")

    print("\nSTRESS TEST CENTRAL MEANS")
    print(model.stress_test_projection(proj))
