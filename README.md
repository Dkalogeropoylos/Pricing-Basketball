# Basketball Pricing Engine v2.11.0

This patch is intentionally structural rather than a collection of market-specific tweaks.

## Files to upload

Upload these **four files into `core/`**:

- `availability.py` **NEW**
- `minutes_engine.py`
- `matchup.py`
- `team_model.py`

Then upload this at repo root:

- `streamlit_app.py`

`tests/` is optional.

---

## 1. Confirmed OUT availability state

Team Markets now have two explicit multiselects:

- away confirmed OUT
- home confirmed OUT

Only `OUT` is modeled. QUESTIONABLE/GTD is ignored until the trader explicitly marks the player OUT.

For selected OUT set `O = {p1, p2, ...}`, a historical game is a selected-state match when **all selected players were absent**. Games before every selected player had first appeared for that team are neutral and are not falsely classified as injury matches.

Current-opponent H2H games are excluded from the availability-state sample because H2H has its own separate layer.

For eligible games:

- `N` = eligible games
- `Ns` = selected-state matches
- natural state share = `p = Ns/N`
- confidence = `c = Ns/(Ns + K)`
- default `K = 6` (provisional; expose in UI and later tune with rolling out-of-sample backtest)
- target exact-state share = `min(0.70, max(p, c))`

Per-game weights are chosen so the selected-state games collectively receive the target share while mean eligible inner weight remains 1.

### Overlap guard

Confirmed OUT identity is **removed from residual Jaccard rotation similarity**. Residual rotation similarity is only:

`0.85 + 0.15 * Jaccard`

so OUT state and rotation similarity do not price the same absence twice.

Also, when confirmed OUT is selected, AUTO uses Stable 55/20/25 outer weights. If there is an additional structural role change beyond the OUTs, trader can explicitly choose Role change 35/20/45.

---

## 2. Non-overlapping H2H

The actual chronological Old / G6-10 / L5 buckets are formed first. Current-opponent H2H rows are then removed **inside those buckets**, rather than replacing them with older games.

The same H2H rows are also excluded from:

- Team Market opponent-allowed profile
- location split
- availability-state confidence

Then H2H is added back once as a separate structural sample.

H2H weight:

`wH = min(0.15, 0.20 * N_H2H/(N_H2H + 2) * rotation_similarity)`

Typical two-game comparable H2H = roughly 7-10% weight.

H2H blends structural rates such as FGA/live, 3P share, FTA/poss, TOV/poss, OREB/miss, AST/made-FG, PF/poss, DREB capture, steals/opponent-TOV and BLK/poss.

Shooting percentages are **not** H2H blended.

---

## 3. Shot architecture

Old model:

- `3PA ~ Poisson(...)`
- `2PA ~ Poisson(...)`

independently.

v2.11:

1. possessions
2. turnovers
3. live possessions
4. total FGA
5. 3P share
6. `3PA ~ Binomial(FGA, 3P_share)`
7. `2PA = FGA - 3PA`

Therefore `FGA = 3PA + 2PA` by construction.

Team opponent context now uses:

- `FGA_LIVE = FGA / (possessions - TOV)`
- `3P_SHARE = 3PA / FGA`

rather than independent opponent 3PA/poss and 2PA/poss multipliers.

---

## 4. OREB / AST denominators

To reduce structural overlap:

- Team OREB opponent matchup uses `OREB / missed FGA`
- Team AST opponent matchup uses `AST / FGM`
- TOV remains `TOV / possessions`

The current rebound simulator remains miss-opportunity based. v2.11 **does not invent** a separate 2P-miss / 3P-miss rebound multiplier from box scores. That split should only be activated when real play-by-play attribution is available.

---

## 5. BLK v3

Core block expectation:

`own BLK rate × opponent blocks-suffered modifier × location × optional positional relative modifier × tiny 2PA opportunity modifier × possessions`

### Own BLK ability

Old/G6-10/L5 + exact availability weighting + weak residual rotation, with current H2H excluded. A light league stabilization remains.

### Opponent blocks suffered

For an offense, opponent-suffered BLK is measured from how many blocks its opponents record against it. Raw opponent susceptibility ratio is shrunk with exponent `0.40`, capped to 0.90-1.10.

Example: susceptibility ratio `1.129` becomes about `1.05`, not `1.03` and not the full `1.129`.

### H2H

H2H is already blended once into own BLK rate via the disjoint H2H layer above. No separate tiny H2H multiplier remains in the active v2.11 path.

### Position

If player data contains a real blocked-attempt field (`BLKA`, `BA`, etc.), a relative G/F/C susceptibility correction is calculated and capped to 0.97-1.03.

If such a field does not exist, positional BLK modifier is exactly `1.00`. The app explicitly says this in audit; it does **not** infer position blocks from 2PA.

### 2PA

Opponent 2PA is only a small opportunity nudge:

`(projected opponent 2PA/poss / league 2PA/poss)^0.07`, cap 0.98-1.02.

### Conservation fix

Old incorrect cap:

`BLK <= opponent missed 2PA`

v2.11:

`BLK <= opponent missed FGA`

because three-point attempts can also be blocked.

---

## 6. Team With Most

v2.10 historical difference/tie calibration is retained. No bookmaker odds enter that calibration.

v2.11 first fixes the base joint distributions; Team With Most should be re-evaluated after running historical backtests on the new base model.

---

## Validation

Included tests:

- `tests/v29_smoke.py`
- `tests/v210_smoke.py`
- `tests/v211_smoke.py`

v2.11 smoke verifies:

- FGA exactly partitions into 3PA + 2PA
- BLK is capped by total missed FGA, not missed 2PA
- selected-OUT state can exclude current H2H from its confidence sample
- H2H structural blend is small, rotation-aware and does not modify shooting percentages
