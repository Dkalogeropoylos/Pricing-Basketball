# Basketball Pricing Engine v2.11.1 hotfix

This hotfix changes only `streamlit_app.py` and assumes v2.11.0 core files are already installed.

## Changes
- One shared Confirmed OUT selector in Game Setup feeds Team Markets and Player Props.
- Player Props now visibly show the confirmed OUT state they are using.
- Player Props use the v2.11 exact JOINT absence-state weighting (`availability_state_weights`) rather than the older same-role helper.
- Eligibility-start guard prevents games before an absent teammate joined the team from being mislabeled as same-state games.
- Player injury overlap guard: confirmed OUT state changes projected minutes via the 200-minute engine and per-minute rates via exact-state inner weighting; the same OUT state no longer also auto-switches the player profile to 35/20/45.
- Explicit trader role multipliers can still be used intentionally as a separate override.

## Upload
Replace only the root `streamlit_app.py` in the GitHub repo.
