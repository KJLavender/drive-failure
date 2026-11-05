"""特徵表：每顆（抽樣）HDD 每天一列。

抽樣：依「硬碟」而非「天」抽樣，才能算出每顆硬碟每月的誤報率。
- 出過故障的硬碟：全部保留，weight = 1
- 健康硬碟：hash(serial) 取 1/HEALTHY_RATE，weight = HEALTHY_RATE
- 故障後又復活的序號：排除（標籤不可信）
- 第一次故障之後的紀錄：丟掉

標籤不在這裡定，只存距離欄位，讓 N=7/14/30 共用同一張表：
- days_to_fail：距離故障幾天（健康硬碟為 NULL）
- days_to_last：距離這顆硬碟最後一筆紀錄幾天（用來處理「消失但沒標故障」）
- days_to_end：距離整份資料最後一天幾天（右設限）
"""

import argparse

import duckdb

from drivefail.db import connect
from drivefail.paths import DERIVED_DIR

HEALTHY_RATE = 10

# 累積型計數器：算變化量與是否首次出現
COUNTERS = [1, 5, 7, 10, 12, 183, 184, 187, 188, 189, 192, 193, 197, 198, 199, 240, 241, 242]
# 算 30 天斜率的關鍵指標
SLOPE_ATTRS = [5, 187, 197, 198]
# Seagate 的 1／7 raw 是複合編碼，normalized 比較有意義
NORMALIZED = [1, 3, 7]
# Backblaze 官方用來判斷故障的 5 個指標（標籤循環消融用）
BACKBLAZE_FIVE = [5, 187, 188, 197, 198]


def _feature_select(available: set[str]) -> str:
    cols = []
    w = "PARTITION BY serial_number ORDER BY date RANGE BETWEEN INTERVAL {n} DAYS PRECEDING AND CURRENT ROW"
    for a in COUNTERS:
        c = f"smart_{a}_raw"
        if c not in available:
            continue
        cols += [
            f"{c} AS s{a}",
            f"{c} - min({c}) OVER ({w.format(n=7)}) AS s{a}_d7",
            f"{c} - min({c}) OVER ({w.format(n=30)}) AS s{a}_d30",
            # 7 天前還是 0（或沒紀錄），現在 > 0
            f"""CASE WHEN {c} > 0 AND coalesce(max({c}) OVER (PARTITION BY serial_number ORDER BY date
                RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 7 DAYS PRECEDING), 0) = 0
                THEN 1 ELSE 0 END AS s{a}_new7""",
        ]
    for a in SLOPE_ATTRS:
        c = f"smart_{a}_raw"
        if c in available:
            cols.append(f"regr_slope({c}, epoch(date) / 86400) OVER ({w.format(n=30)}) AS s{a}_slope30")
    for a in NORMALIZED:
        c = f"smart_{a}_normalized"
        if c in available:
            cols += [f"{c} AS n{a}", f"{c} - max({c}) OVER ({w.format(n=30)}) AS n{a}_d30"]
    if "smart_194_raw" in available:
        cols += [
            "smart_194_raw AS temp",
            f"avg(smart_194_raw) OVER ({w.format(n=30)}) AS temp_avg30",
            f"max(smart_194_raw) OVER ({w.format(n=30)}) AS temp_max30",
        ]
    if "smart_9_raw" in available:
        cols.append("smart_9_raw / 24.0 AS power_on_days")
    return ",\n        ".join(cols)


def build_features(con: duckdb.DuckDBPyConnection, healthy_rate: int = HEALTHY_RATE) -> None:
    available = {r[0] for r in con.execute("DESCRIBE raw_daily").fetchall()}
    data_end = con.execute("SELECT max(last_date) FROM drives").fetchone()[0]
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE sample_drives AS
        SELECT serial_number, model, failed, first_fail_date, last_date,
               CASE WHEN failed THEN 1 ELSE {healthy_rate} END AS weight
        FROM drives
        WHERE media = 'HDD' AND n_models = 1 AND days_after_fail = 0
          AND (failed OR hash(serial_number) % {healthy_rate} = 0)"""
    )
    con.execute(
        f"""CREATE OR REPLACE TABLE features AS
        WITH base AS (
            -- 直接讀 raw_daily、型號從 drives 拿：避免對上億筆逐筆跑 regex
            SELECT d.*, s.model AS model_norm FROM raw_daily d JOIN sample_drives s USING (serial_number)
            WHERE s.first_fail_date IS NULL OR d.date <= s.first_fail_date
        ),
        feat AS (
            SELECT serial_number, date, model_norm AS model,
                capacity_bytes / 1e12 AS capacity_tb,
                {_feature_select(available)}
            FROM base
        )
        SELECT f.*,
            s.failed, s.weight,
            datediff('day', f.date, s.first_fail_date) AS days_to_fail,
            datediff('day', f.date, s.last_date) AS days_to_last,
            datediff('day', f.date, DATE '{data_end}') AS days_to_end,
        FROM feat f JOIN sample_drives s USING (serial_number)"""
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--healthy-rate", type=int, default=HEALTHY_RATE)
    args = ap.parse_args()
    con = connect(memory_limit="7GB")
    build_features(con, args.healthy_rate)
    out = DERIVED_DIR / "features.parquet"
    con.execute(f"COPY features TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    print(con.execute(
        "SELECT count(*) AS rows, count(DISTINCT serial_number) AS drives, "
        "count(DISTINCT serial_number) FILTER (WHERE failed) AS failed_drives FROM features"
    ).df().to_string(index=False))


if __name__ == "__main__":
    main()
