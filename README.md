# Basketball Pricing Engine v2.8.0 — Coupled Team Markets + Auto Fair Lines

Apply this cumulative patch **on top of the working v2.7.2 project** (the project that already contains `providers/sportsdataverse_wnba.py`).

## What v2.8 fixes

### 1. Team Markets are now a coupled two-team game simulation

The previous engine simulated one team at a time. That was acceptable for basic offensive counts, but it was structurally wrong for defensive/rebound and relative markets.

v2.8 simulates HOME and AWAY inside the same game state and enforces:

- `FGA = 2PA + 3PA`
- `PTS = 2*2PM + 3*3PM + FTM`
- `REB = OREB + DREB`
- OREB comes from the team's own missed FGs
- DREB comes from opponent misses not recovered by opponent OREB
- STL is a subset of opponent TOV
- BLK is a subset of opponent missed 2PA

This fixes the v2.7.2 direction errors where STL was linked to the same team's TOV and BLK to the same team's 2PA.

### 2. No new recent-sample overlap

The outer protocol remains:

- Stable: Old / G6-10 / L5 = `55 / 20 / 25`
- Role change: Old / G6-10 / L5 = `35 / 20 / 45`

Current-rotation Jaccard similarity is now an **inner weight inside each existing bucket**. It does not create a second sample and does not change the outer bucket weights.

OUT players are already removed from today's rotation signature, so v2.8 does not add a second full injury penalty.

### 3. Shooting efficiency is regressed, not copied from L5

Team 3P%, 2P% and FT% use larger-sample attempt-based shrinkage, matching the player-engine philosophy. Recent hot/cold shooting is not treated as future true shooting ability.

### 4. Small automatic home/away correction

If location metadata exists (`IS_HOME`, `HOME_AWAY`, `LOCATION`, or `MATCHUP`), a small shrinked home/away modifier is applied. Only 20% of the split deviation is used and the modifier is capped at +/-6%, so the location split cannot dominate the main sample.

If location metadata is unavailable or the split has fewer than five games, the modifier stays neutral.

### 5. Automatic model line + fair price

For every simulated market the app now reports:

- Projection mean
- Median
- **Model line** (half-point sportsbook line chosen from the simulated distribution, not by blindly rounding the mean)
- P(Over) / fair Over
- P(Under) / fair Under
- low/high pace stress projections

So a projection such as `30.7` is automatically translated into a practical `x.5` line and fair prices from the full distribution.

The same automatic line/fair table is shown for selected PLAYER props, including PTS, using the existing minutes engine.

### 6. Push-aware fair pricing fixed

For integer bookmaker lines, fair odds now correctly account for pushes:

`fair odds = (1 - p_push) / p_win`

The old `1 / p_win` formula was only correct when push probability was zero.

### 7. New automatically priced team scopes

One Monte Carlo run now creates:

- Away team markets
- Home team markets
- Game totals
- Team-with-most 3-way probabilities/fair odds

Supported team outputs include:

`PTS, FGA, FGM, 3PA, 3PM, 2PA, 2PM, FTA, FTM, REB, OREB, DREB, AST, STL, BLK, TOV, PF`

### 8. H2H remains audit-only

Same-season H2H is displayed in the model audit but receives **zero extra numerical weight**, so those games are not counted again on top of the historical buckets.

## Files changed

- `streamlit_app.py`
- `core/team_model.py`
- `core/pricing.py`
- `core/minutes_engine.py`

Optional offline check:

```bash
python tests/team_engine_smoke.py
```

## Important

Sportsbook total/spread remain audit-only. They are not fed back into the projection, so the model does not circularly reproduce the market it is trying to price.
