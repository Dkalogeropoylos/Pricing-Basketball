# Basketball Pricing Engine v2.8.1 cumulative fix

This patch fixes the v2.8 deployment mismatch where `streamlit_app.py` could be updated while `core/team_model.py` remained on v2.7.2, causing:

`ImportError: cannot import name 'simulate_game' from 'core.team_model'`

## IMPORTANT
Copy the **whole contents** of this zip into the project root and overwrite existing files while preserving folders.

Required overwrites:
- `streamlit_app.py`
- `core/team_model.py`
- `core/pricing.py`
- `core/minutes_engine.py`
- `core/player_model.py`
- `core/pace_engine.py`
- `core/redistribution.py`

The project must also keep the already-installed:
- `providers/sportsdataverse_wnba.py`

Do not put `team_model.py` in the root. It must stay under `core/team_model.py`.

## Verification
The v2.8 team model must contain these functions:
- `simulate_game`
- `team_location_modifiers`
- `h2h_team_audit`

Optional local check:
```bash
python tests/team_engine_smoke.py
```
