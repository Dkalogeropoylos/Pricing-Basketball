# WNBA Prop Model — Streamlit MVP

## What this version does

- Pulls **player game logs** from the WNBA Stats surface (`LeagueID=10`)
- Pulls **team game logs**
- Stores local data in **DuckDB**
- Lets you inspect:
  - Season
  - Last 10
  - Last 5
  - H2H
- Lets you manually select **same-role games**
- Builds the frozen pre-market projection using:
  - minutes
  - FGA / 3PA / FTA
  - REB / AST rates
  - opponent overall allowances
  - editable positional modifiers
  - editable injury/rotation redistribution
  - efficiency regression
- Runs Monte Carlo
- Prices:
  - PTS
  - REB
  - AST
  - 3PM
  - P+R
  - P+A
  - A+R
  - PRA
- Calculates:
  - model probability
  - fair odds
  - bookmaker no-vig probability
  - probability edge
  - EV
  - fractional-Kelly units
- Saves model runs for later backtesting

## Why DuckDB

For the first local version DuckDB is ideal:
- no server to install
- one local `.duckdb` file
- SQL + Pandas work easily
- very fast for analytical queries
- later we can migrate to PostgreSQL/Supabase if the app becomes cloud/multi-user

## Data architecture

Primary live data:
- WNBA Stats / official Stats surface
- LeagueID = `10`

Local warehouse:
- `data/wnba_props.duckdb`

Planned additional feeds:
1. Official WNBA injury report
2. Player positions / roster snapshots
3. Schedule and future-game table
4. Rotation / starter history
5. Historical WNBA Stats parquet backup
6. Bookmaker market snapshots

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## MVP workflow

1. Open **Data Sync**
2. Sync the current WNBA season
3. Open **Player Lab**
4. Select player + opponent
5. Inspect Season / L10 / L5 / H2H
6. Tick only comparable **same-role** games
7. Enter current injury/rotation adjustments
8. Set matchup / positional inputs
9. Click **Build frozen projection**
10. Open **Market Pricing**
11. Enter bookmaker line + both prices
12. Run / save the market

## Important modeling rule

If your `same-role` games already contain the current absence setup, do **not**
also add the full vacated FGA / AST / REB again. That is double counting.

## Next development step

Automate these fields that are manual in v0.1:

- current injuries
- expected starting five
- position buckets
- positional opponent allowances
- same-role game tagging
- expected minutes
- H2H OT cleanup
- bookmaker market import

The mathematical engine should remain separate from the AI/context layer.
