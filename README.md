# Basketball Pricing Engine v2.2

## New in v2.2
- Real Home/Away team selection from nba_api.
- Automatic opponent-allowed profiles normalized per possession vs league average.
- Automatic G/F/C opponent-by-position modifiers using OpponentTeamID + PlayerPosition.
- Position adjusts only the deviation from overall defense, avoiding double counting.
- Game-context JSON now auto-prefills projected minutes and role multipliers.
- Same-season H2H is shown automatically and receives 0% extra weight by default.

## Providers
Core/basic:
- WNBA: nba_api, LeagueID 10
- NBA: nba_api, LeagueID 00

Optional:
- BALLDONTLIE WNBA advanced endpoints, tier/rate-limit dependent.

Future:
- EuroLeague (E)
- EuroCup (U)

## Security
Use Streamlit Community Cloud Secrets only:

```toml
BALLDONTLIE_API_KEY = "your-key"
```

Never commit keys or place them in context JSON.

## Protocol
- Stable weights: 55 / 20 / 25
- Role-change weights: 35 / 20 / 45
- Old / Games 6–10 / L5 are non-overlapping
- H2H is context only
- Pace/opportunity enters once
- 3PM = 3PA x regressed shooting ability
- Joint player MC for PTS/REB/AST/3PM + PRA/PR/PA/AR
- Stress scenarios before qualification
- Bookmaker price entered after projection

## Current team-model note
Team Markets v2.2 uses automatic opponent interaction but still simulates one team's
distribution at a time. A fully coupled two-team latent game engine is the next major upgrade.


## v2.3 — Cloud hang protection

v2.3 makes no `stats.nba.com` request during Streamlit startup.

- WNBA team names are loaded from BALLDONTLIE `/teams` only after a button click.
- `nba_api` is invoked only after explicit user action.
- nba_api timeouts were reduced to 8 seconds.
- If stats.nba.com blocks or times out from Streamlit Cloud, the UI stays responsive and reports the failure.
- This is necessary because BALLDONTLIE Free provides Teams/Players/Games, but WNBA Player Stats and Team Stats require GOAT.
