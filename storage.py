
from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd

DEFAULT_DB_PATH = Path("data/wnba_props.duckdb")


class WNBADatabase:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self):
        return duckdb.connect(str(self.path))

    def init_schema(self) -> None:
        with self.connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS injury_snapshots (
                    snapshot_ts TIMESTAMP,
                    game_date DATE,
                    team_abbreviation VARCHAR,
                    player_name VARCHAR,
                    status VARCHAR,
                    reason VARCHAR,
                    source VARCHAR,
                    notes VARCHAR
                )
            """)

            con.execute("""
                CREATE TABLE IF NOT EXISTS player_positions (
                    player_id BIGINT,
                    player_name VARCHAR,
                    position VARCHAR,
                    source VARCHAR,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            con.execute("""
                CREATE TABLE IF NOT EXISTS model_runs (
                    run_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    game_key VARCHAR,
                    player_name VARCHAR,
                    opponent VARCHAR,
                    market VARCHAR,
                    line DOUBLE,
                    side VARCHAR,
                    book_odds DOUBLE,
                    model_probability DOUBLE,
                    fair_odds DOUBLE,
                    no_vig_probability DOUBLE,
                    edge_pp DOUBLE,
                    ev DOUBLE,
                    units DOUBLE,
                    input_json VARCHAR
                )
            """)

    def replace_table(self, table: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        with self.connect() as con:
            con.register("_incoming_df", df)
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _incoming_df")
            con.unregister("_incoming_df")

    def table_exists(self, table: str) -> bool:
        with self.connect() as con:
            n = con.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = ?
                """,
                [table],
            ).fetchone()[0]
        return bool(n)

    def read_table(self, table: str) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(f"SELECT * FROM {table}").df()

    def query(self, sql: str, params=None) -> pd.DataFrame:
        with self.connect() as con:
            if params is None:
                return con.execute(sql).df()
            return con.execute(sql, params).df()

    def add_model_run(self, row: dict) -> None:
        cols = [
            "game_key", "player_name", "opponent", "market", "line", "side",
            "book_odds", "model_probability", "fair_odds",
            "no_vig_probability", "edge_pp", "ev", "units", "input_json"
        ]
        vals = [row.get(c) for c in cols]
        placeholders = ", ".join(["?"] * len(cols))
        with self.connect() as con:
            con.execute(
                f"""
                INSERT INTO model_runs ({", ".join(cols)})
                VALUES ({placeholders})
                """,
                vals,
            )
