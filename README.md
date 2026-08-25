# Basketball Pricing Engine v2.4

WNBA core data no longer depends on `stats.nba.com`.

## Historical data

Primary WNBA historical provider:
**SportsDataverse GitHub Releases**

The app downloads and caches:

- `espn_wnba_player_boxscores/player_box_{season}.parquet`
- `espn_wnba_team_boxscores/team_box_{season}.parquet`

If parquet is unavailable, it automatically tries the corresponding CSV.

The source datasets are published by SportsDataverse/wehoop and are refreshed
during the WNBA season.

## What the two files provide

Player boxscores:
- game/date/team/opponent
- minutes
- FGM/FGA
- 3PM/3PA
- FTM/FTA
- OREB/DREB/REB
- AST/STL/BLK/TOV/PF/PTS
- starter
- position
- home/away

Team boxscores:
- team/opponent/game/date
- FGM/FGA
- 3PM/3PA
- FTM/FTA
- OREB/DREB/REB
- AST/STL/BLK/TOV/PF/PTS

## Automatic calculations

From those files the app calculates:
- current team and player lists
- Old season / Games 6-10 / L5
- team pace / possession profiles
- opponent allowed per possession vs league average
- G/F/C opponent-by-position per-36 vs league positional baseline
- same-season H2H
- OT flag when a game contains any player above 40 minutes

No extra H2H weight is applied automatically.

## Pregame manual context

Only information a historical database cannot know should be manual:

```json
{
  "injuries": {
    "Rae Burrell": {"status": "OUT"}
  },
  "projected_minutes": {
    "Ariel Atkins": 31
  },
  "role_adjustments": {
    "Ariel Atkins": {
      "usage": 1.08,
      "three_role": 1.12,
      "creation": 1.03
    }
  }
}
```

## Providers retained for later

- BALLDONTLIE: optional advanced / injuries depending on tier.
- nba_api: retained in the repository as an optional local/fallback adapter,
  but v2.4 WNBA Streamlit core does not call stats.nba.com.
- NBA, EuroLeague and EuroCup adapters can use the same normalized model layer.

## Model protocol

- Stable weighting: 55 / 20 / 25
- Role-change weighting: 35 / 20 / 45
- Old / Games 6-10 / L5 never overlap
- H2H is context only
- 3PM generated from projected 3PA and regressed 3P%
- player joint MC: PTS / REB / AST / 3PM and combos
- team MC: PTS / 3PA / 3PM / 2PA / 2PM / FTA / TOV / OREB / AST / STL / BLK / PF
- bear / central / bull opportunity stress
- bookmaker odds entered after projection

## Important

Model-implied fair odds are not yet historically calibrated probabilities.
