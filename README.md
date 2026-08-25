# Basketball Pricing Engine v2.9.0 patch

## What changed

### 1. No bookmaker input required for core pricing
- Every simulated market now shows:
  - Projection
  - Model line
  - Fair Over / Fair Under
  - **Play Over from / Play Under from** using the selected target EV buffer (default 6%)
  - Highest Over line still playable at a reference price (default 1.90)
  - Lowest Under line still playable at the same reference price
- Player props also get a half-line ladder around the model line.
- The old Bear/Bull columns and mandatory manual bookmaker entry flow are removed from the player pricing view.
- Fair price is explicitly treated as break-even, not as an automatic bet trigger.

### 2. Same-role teammate-absence weighting for player props
- New ON/OFF control is ON by default.
- Confirmed OUT teammates are read from `game_context.json`.
- Example: if Fudd is OUT, Bueckers historical games without Fudd are detected automatically.
- Matching games are regularized and reweighted **inside** Old / G6-10 / L5 only.
- No fourth sample is created, so recent games are not double-counted.
- Model audit shows how many matching games exist and their raw MIN/PTS/REB/AST/3PA/FTA averages.

### 3. Position matchup improved
- Opponent-by-position remains G/F/C for stability.
- Position effects already influence PTS, REB, AST, 3PA and FTA.
- v2.9 also adds heavily-shrunk positional 3P% / 2P% context.
- Overall opponent defense remains the base; position only modifies the relative positional deviation, avoiding double counting.

### 4. Three-point treatment
- Player 3P ability still uses larger-sample regression rather than raw L5 makes.
- The 3P prior is slightly lighter for established shooters (32 pseudo-attempts instead of 45).
- Team 3PA/2PA opponent + location corrections are capped so the opponent layer cannot overpower the team's own weighted shot profile.
- No blind +X 3PA correction was added; the change addresses over-shrinkage rather than fitting one game to the market.

### 5. Blocks repaired
Old issue:
- BLK ability was expressed as `BLK / opponent missed 2PA`.
- This confounded shot-blocking ability with opponent shooting luck/efficiency and could reverse team rankings.

v2.9:
- learns BLK rate per **opponent 2PA**
- shrinks the rate strongly toward a league-style prior because blocks are noisy
- simulates candidates from opponent 2PA
- caps final blocks by actual opponent missed 2PA so the box-score identity remains logical

### 6. Team shooting defense
- Opponent allowed 3P% and 2P% are now fed into team scoring with very conservative shrinkage.
- `TEAM WITH MOST -> PTS` remains hidden until a dedicated score/winner layer is backtested.

## Files to overwrite
Copy these into the matching paths in the existing repo:

- `streamlit_app.py`
- `core/pricing.py`
- `core/player_model.py`
- `core/matchup.py`
- `core/team_model.py`
- `core/role_splits.py` (new)

Do not upload the `core/*.py` files to the repository root.

## Validation
`tests/v29_smoke.py` checks:
- FGA identity
- block logical cap
- new block-rate field
- 6% play-from price math
- same-role weighting stays inside existing sample scale
- same-role higher-volume sample can move opportunity rates
- player price ladder generation
