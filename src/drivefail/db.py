"""DuckDB 連線：把已轉好的季度 Parquet 註冊成 `daily` view。"""

import duckdb

from drivefail.paths import DERIVED_DIR, PARQUET_DIR, TMP_DIR

# Backblaze 開機碟裡的 SSD 型號特徵（其餘一律視為 HDD）
SSD_PATTERN = r"SSD|ZA250|WDS\d|SA510|MTFDDAV|DELLBOSS|MZ7|SSDSC|^CT\d+MX"


def done_quarters() -> list[str]:
    return sorted(p.parent.name for p in PARQUET_DIR.glob("*/_DONE"))


def connect(quarters: list[str] | None = None, memory_limit: str = "6GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{(TMP_DIR / 'duckdb_spill').as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    quarters = quarters or done_quarters()
    if not quarters:
        raise RuntimeError("data/parquet 底下沒有轉好的季度，先跑 python -m drivefail.ingest")
    globs = ", ".join(f"'{(PARQUET_DIR / q).as_posix()}/*.parquet'" for q in quarters)
    con.execute(
        f"""CREATE VIEW raw_daily AS
        SELECT * FROM read_parquet([{globs}], union_by_name=true)"""
    )
    register_daily(con, "raw_daily")
    drives = DERIVED_DIR / "drives.parquet"
    if drives.exists():
        con.execute(f"CREATE VIEW drives AS SELECT * FROM read_parquet('{drives.as_posix()}')")
    return con


def register_daily(con: duckdb.DuckDBPyConnection, source: str) -> None:
    """在 source 上加正規化型號與 HDD/SSD 分類，建立 `daily` view。測試也用這個。"""
    con.execute(
        f"""CREATE OR REPLACE VIEW daily AS
        SELECT *,
            -- 'WUH721816ALE6L4' 和 'WDC WUH721816ALE6L4' 是同一型號
            regexp_replace(trim(model), '^WDC\\s+', '') AS model_norm,
            CASE WHEN regexp_matches(model, '{SSD_PATTERN}') THEN 'SSD' ELSE 'HDD' END AS media
        FROM {source}"""
    )
