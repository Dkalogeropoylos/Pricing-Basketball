# Basketball Pricing Engine v2.10.0

This patch is intentionally narrow: it changes **team BLK**, **TEAM WITH MOST calibration**, and makes **current IN/OUT rotation weighting visible and switchable**. Other working team markets are left unchanged.

## 1) New BLK model

The previous BLK-per-opponent-2PA estimator is no longer used to price blocks.

For team A vs team B:

1. **Own block ability** comes from the existing non-overlapping weighted history:
   - stable: Old/G6-10/L5 = 55% / 20% / 25%
   - role change: 35% / 20% / 45%
   - current-rotation similarity acts only inside these buckets.

   `r_A = weighted BLK / weighted possessions`

2. Because blocks are noisy, only a light league prior is used:

   `r_A* = 0.85*r_A + 0.15*r_league`

3. **Opponent block susceptibility** is how many blocks opponents make against team B per possession relative to league average. It is a small contextual correction:

   `M_allow = clip((B_BLK_allowed / league_BLK)^0.25, 0.94, 1.06)`

4. **Home/away** is a small split correction only:

   `M_loc = clip((location_rate / overall_rate)^0.20, 0.94, 1.06)`

   Combined opponent/location BLK correction is capped to 0.90–1.10.

5. **Same-season H2H** is deliberately tiny because those games are already inside the historical buckets:

   `ratio_H2H = H2H_BLK_rate / base_BLK_rate`

   `confidence = n_H2H / (n_H2H + 3)`

   `M_H2H = clip(ratio_H2H^(0.08*confidence), 0.97, 1.03)`

6. **Opponent 2PA opportunity** is only a small nudge, not the block denominator:

   `M_2PA = clip((opponent_simulated_2PA_per_poss / league_2PA_per_poss)^0.15, 0.96, 1.04)`

7. For each Monte Carlo iteration:

   `lambda_BLK = POSS * r_A* * M_allow/location * M_H2H * M_2PA * block_noise`

   where `block_noise = exp(0.10*z - 0.5*0.10^2)`.

   Candidate blocks are Poisson(`lambda_BLK`) and then capped by the opponent's actual missed 2PA **only as a box-score conservation rule**.

So the main information is now exactly: **what the team does + what the opponent allows + pace**, with H2H and 2PA only small logical corrections.

## 2) TEAM WITH MOST calibration

The old table directly counted `home > away`, `tie`, `away > home` from the raw joint simulation. The team means could be sensible while residual difference variance was too wide, suppressing tie probabilities.

v2.10 keeps the model's expected team difference, but calibrates the **spread of the difference** to actual same-game league history for each stat. No bookmaker prices are used.

For market X:

- raw simulation difference: `D = Home_X - Away_X`
- raw mean: `mu = mean(D)`
- raw SD: `s_raw = sd(D)`
- league historical paired-game difference SD: `s_league`
- league historical mean total for the stat: `m_league`
- current simulated mean total: `m_current`

Scale league spread to the current stat environment:

`target_sd = s_league * sqrt(m_current / m_league)`

Blend 60% toward historical calibration:

`applied_sd = 0.40*s_raw + 0.60*target_sd`

The SD ratio is capped to 0.80–1.20 so calibration cannot rewrite the model.

`D* = mu + (D-mu) * clip(applied_sd/s_raw, 0.80, 1.20)`

`D*` is converted back to an integer-difference distribution with expectation-preserving linear floor/ceil allocation. Then:

- Home most: `P(D*>0)`
- Tie: `P(D*=0)`
- Away most: `P(D*<0)`

The UI includes a hidden audit with raw tie, calibrated tie, historical league tie, raw difference SD and target difference SD.

PTS/match-winner remains omitted.

## 3) Current IN/OUT rotation visibility

### Team Markets
Under **Team sample weighting / trader override** there is now an ON/OFF toggle:

`AUTO: weight team history toward the current IN/OUT rotation`

When ON, each historical game receives the already-existing current-rotation inner weight:

`Jaccard = |current rotation ∩ historical rotation| / |current rotation ∪ historical rotation|`

`game_weight = 0.55 + 0.45*Jaccard`

This is inside Old/G6-10/L5 only and is not a fourth sample.

The Model Audit now shows OUT/GTD players and min/mean/max historical inner weights.

### Player Props
The existing player toggle remains:

`AUTO: weight historical games toward the current teammate-absence state`

For a focal player with confirmed OUT teammate(s):

- find games where those teammate(s) did not play;
- if matching games < 2: neutral;
- otherwise `confidence = n_match/(n_match+6)`;
- matching-game weight = `1 + 0.40*confidence`;
- mismatch-game weight = `1 - 0.18*confidence`;
- normalize weights to mean 1;
- apply only inside the existing Old/G6-10/L5 buckets.

The player Model Audit displays absent teammate names, same-role game count, confidence and same-role MIN/PTS/REB/AST/3PA/FTA.

## Files changed

- `core/team_model.py`
- `core/matchup.py`
- `core/pricing.py`
- `streamlit_app.py`
- `tests/v210_smoke.py`
