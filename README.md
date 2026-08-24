# Basketball Pricing Engine v2

Deploy-ready Streamlit project for NBA/WNBA player props and WNBA team markets.

## Security

**Never put API keys in code, JSON, or GitHub.**

For Streamlit Community Cloud:
1. Deploy the GitHub repository.
2. Open **App settings -> Secrets**.
3. Add:

```toml
BALLDONTLIE_API_KEY = "your-key"
```

The app never renders the key. It only displays whether a key is configured.

The repository `.gitignore` explicitly excludes `.streamlit/secrets.toml`.

## Data providers

### WNBA
Primary when a key is available:
- BALLDONTLIE WNBA API
  - players
  - games
  - player game stats
  - team game stats
  - player/team season advanced endpoints

Fallback:
- `nba_api` with WNBA `LeagueID=10`

### NBA
- `nba_api` with NBA `LeagueID=00`

### EuroLeague / EuroCup
Adapter structure already exists under `providers/euroleague.py`.
Not enabled yet. The intended competition codes are:
- `E` EuroLeague
- `U` EuroCup

## Fixed protocol encoded

- Old season / Games 6–10 / L5 are non-overlapping.
- Stable rotation weights: 55 / 20 / 25.
- Role/injury-change weights: 35 / 20 / 45.
- H2H is context only and capped in the UI at +/-10%.
- Pace/opportunity enters once.
- 3PM comes from simulated attempts and regressed shooting efficiency.
- Joint Monte Carlo provides PTS / REB / AST / 3PM and combos.
- Team engine provides PTS / 3PA / 3PM / 2PA / 2PM / FTA / TOV / OREB / AST / STL / BLK / PF.
- Low / central / high opportunity stress scenarios.
- Bookmaker odds are entered after the projection is produced.

## Manual input: what the user supplies

Historical stats should come from APIs. Manual input is only for pre-game context that no historical API can reliably infer:
- OUT / questionable / available
- projected minutes
- starting/bench role
- role redistribution due to absences
- optional small H2H contextual modifier
- bookmaker line and price

A small optional `game_context.json` is supported for this purpose.

## Streamlit Cloud deployment

Repository files needed:
- `streamlit_app.py`
- `core/`
- `providers/`
- `config/`
- `requirements.txt`
- `.streamlit/config.toml`

Entry point:
`streamlit_app.py`

No local run is required for deployment.

## Current v2 limitations

1. Automatic opponent-by-position data is not yet normalized. The UI exposes explicit opponent multipliers so it is never silently assumed.
2. Advanced BALLDONTLIE endpoints may depend on API tier. They fail gracefully.
3. WNBA Team Markets currently require BALLDONTLIE basic team game logs.
4. NBA team-market adapter and EuroLeague/EuroCup normalization are the next modules.
5. Model-implied fair prices are not yet historically calibrated probabilities.
