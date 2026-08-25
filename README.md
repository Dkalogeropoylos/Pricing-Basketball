# Basketball Pricing Engine v2.6

v2.6 preserves the existing value model and upgrades the **feeding layer**.

## Core model intentionally unchanged

The following remain fixed:
- Old / Games 6-10 / L5 non-overlap
- Stable 55/20/25
- Role-change 35/20/45
- H2H = zero extra weight by default
- opponent overall + positional normalization
- efficiency regression
- joint Monte Carlo
- stress testing
- model-implied fair odds / EV

## Trader layer

The model does **not** guess injuries.

Trader context can declare:
- OUT / GTD
- projected starters
- projected minute overrides
- role redistribution
- rotation regime (`stable` or `role_change`)

If no minute override is supplied, AUTO minutes are used.

## Minutes Engine

AUTO player minutes use:
1. The same non-overlapping Old / G6-10 / L5 buckets.
2. Rotation similarity **inside** each bucket.
3. Historical starter match.
4. OT downweighting.
5. Large-blowout downweighting.
6. Recent median stabilization.
7. Low / central / high minute uncertainty.

The full team rotation is then constrained to:

`5 players × 40 regulation minutes = 200 team-minutes`

OUT players are fixed at zero.
Trader or metadata overrides are fixed first.
The remaining AUTO players absorb the remaining team minutes proportionally.

This means a trader can override one player's minutes without creating an
impossible 215-minute team rotation.

## Selected players UI

The app projects the entire rotation in the background but lets the trader
multiselect only the players they want to simulate.

For selected players:
- 0 minute override = AUTO
- any positive override = TRADER
- metadata projected minutes are used if no UI override exists

## Pace Engine

Team pace is first estimated using the existing non-overlapping weighting.

Historical WNBA games are then used to fit relative **fast-side vs slow-side**
pace control. The fit is ridge-shrunk toward a mild fast-side prior, instead of
hardcoding a 50/50 midpoint or a fixed 60/40 rule.

The output is:
- home pace
- away pace
- fitted fast/slow weights
- central possessions
- low/high band
- empirical fit RMSE

The exact same central possessions feed:
- Team Markets
- Player Props

### Player pace adjustment

Player history is per-minute, so each player receives:

`today projected possessions / player's historical possession environment`

This applies matchup pace once. It avoids adding a second pace modifier on top
of historical rates that already came from fast or slow games.

## Total / handicap

Sportsbook total and spread are **audit-only** in v2.6.

Why:
- a high total can come from pace, efficiency, or both;
- our team projections already create their own scoring expectation;
- feeding the sportsbook total directly back into the same model would be circular.

A separate calibrated market-prior / blowout layer can be added later without
contaminating the core projection.

## Historical provider

WNBA historical data remains SportsDataverse GitHub release data.
No stats.nba.com call is required from Streamlit Cloud.


## v2.7 — historically learned role-aware minute redistribution

v2.7 removes proportional roster-wide scaling for explicit minute overrides.

### Learned replacement matrix

For every current-roster teammate pair A -> B, the engine learns whether B
historically gains minutes when A loses minutes.

The score combines:

1. Negative continuous minute slope between A and B.
2. On/off-like lift: how much B gains when A is at the low end of A's minute
   distribution versus A's normal/high-minute games.
3. Sample-size confidence.
4. A small G/F/C positional prior only as a shrinkage fallback when history is
   thin.

Each focal player's teammate scores are normalized to a row that sums to 1.

Example interpretation:

`Canada -> Backup Guard = 0.48`

means that, among eligible non-fixed teammates, the historical rotation model
expects roughly 48% of a Canada minute override to be absorbed by that player,
subject to 0-40 minute constraints.

### Explicit override flow

1. Build the context-aware AUTO rotation.
2. Apply the 200-minute team constraint.
3. Treat trader/metadata projected-minute targets as deviations from AUTO.
4. Transfer the delta through the learned replacement row.
5. Only if the learned recipients hit 0/40 minute bounds does the engine use a
   broader constraint fallback.
6. Team minutes remain exactly 200.

### Avoiding injury double counting

OUT availability is already part of the current-rotation similarity engine:
OUT players are removed from today's rotation signature, and historical games
with similar rotations receive more weight.

Therefore v2.7 does **not** apply a second full pairwise redistribution on top
of the same OUT absence. That would double count the injury effect.

The learned matrix is used primarily for explicit trader/metadata minute
targets relative to the context-aware AUTO rotation.

### Audit

The Streamlit UI now shows:
- Auto Baseline Min
- final Projected Min
- Override Delta
- exact teammate minute impact of each override
- learned replacement weights
- negative-slope signal
- on/off signal
- sample confidence
- positional fallback prior

The underlying prop/value model is unchanged.
