"""每顆硬碟一列的彙總表 + 標籤品質檢查。

標籤問題（Backblaze 的「故障」= 從名單消失，不一定是真的壞）：
- 復活：failure=1 之後同一序號又出現 → Backblaze 的「消失」不一定是真故障
- 消失但沒標故障：遷移／汰換，視為在 last_date 右設限
- 右設限：資料尾端仍健康的硬碟，未來是否故障未知
- 序號對到多個型號：序號碰撞，排除
"""

import argparse

import duckdb

from drivefail.db import connect
from drivefail.paths import DERIVED_DIR, REPORTS_DIR

DRIVES_SQL = """
WITH per_drive AS (
    SELECT
        serial_number,
        any_value(model_norm) AS model,
        count(DISTINCT model_norm) AS n_models,
        any_value(media) AS media,
        max(capacity_bytes) AS capacity_bytes,
        min(date) AS first_date,
        max(date) AS last_date,
        count(*) AS n_days,
        min(date) FILTER (WHERE failure = 1) AS first_fail_date,
        count(*) FILTER (WHERE failure = 1) AS n_fail_events,
        -- 進出觀測時的通電時數（存活分析的左截斷用）
        arg_min(smart_9_raw, date) FILTER (WHERE smart_9_raw IS NOT NULL) AS poh_first,
        arg_max(smart_9_raw, date) FILTER (WHERE smart_9_raw IS NOT NULL) AS poh_last,
    FROM daily
    GROUP BY serial_number
)
SELECT *,
    first_fail_date IS NOT NULL AS failed,
    -- 第一次故障之後還出現多久（日曆天）；只掃一次資料，不再 join 回 daily
    coalesce(datediff('day', first_fail_date, last_date), 0) AS days_after_fail,
    datediff('day', first_date, last_date) + 1 - n_days AS missing_days,
FROM per_drive
"""


def build_drives(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP VIEW IF EXISTS drives")  # connect() 可能已掛上舊的 drives.parquet
    con.execute(f"CREATE OR REPLACE TABLE drives AS {DRIVES_SQL}")


def quality_summary(con: duckdb.DuckDBPyConnection, censor_days: int = 30) -> dict:
    """回傳標籤品質統計。censor_days：資料最後幾天內仍健康的硬碟視為右設限。"""
    data_end = con.execute("SELECT max(last_date) FROM drives").fetchone()[0]
    row = con.execute(
        f"""SELECT
            count(*) AS drives,
            count(*) FILTER (WHERE failed) AS failed,
            count(*) FILTER (WHERE failed AND days_after_fail > 0) AS resurrected,
            count(*) FILTER (WHERE n_fail_events > 1) AS multi_fail_events,
            count(*) FILTER (WHERE n_models > 1) AS serial_collisions,
            count(*) FILTER (WHERE NOT failed AND last_date < DATE '{data_end}' - {censor_days})
                AS vanished_without_failure,
            count(*) FILTER (WHERE NOT failed AND last_date >= DATE '{data_end}' - {censor_days})
                AS right_censored,
            count(*) FILTER (WHERE missing_days > 0) AS drives_with_gaps,
        FROM drives"""
    ).fetchone()
    cols = [d[0] for d in con.description]
    out = dict(zip(cols, row))
    out["data_end"] = str(data_end)
    out["censor_days"] = censor_days
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--censor-days", type=int, default=30)
    args = ap.parse_args()
    con = connect()
    build_drives(con)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY drives TO '{(DERIVED_DIR / 'drives.parquet').as_posix()}' (FORMAT parquet)")
    q = quality_summary(con, args.censor_days)
    REPORTS_DIR.mkdir(exist_ok=True)
    lines = ["# 標籤品質檢查", "", "| 項目 | 數量 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in q.items()]
    (REPORTS_DIR / "label_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in q.items():
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main()
