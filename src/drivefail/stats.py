"""各型號年化故障率（AFR）。

Backblaze 公式：AFR = 故障數 / (drive_days / 365) × 100%
另算 afr_strict：排除「故障後又復活」的序號，量化標籤雜訊的影響。
門檻沿用 Backblaze 慣例：drive_days ≥ 10,000 且 ≥ 100 顆。
"""

import duckdb

from drivefail.db import connect
from drivefail.paths import DERIVED_DIR, REPORTS_DIR

AFR_SQL = """
WITH dd AS (
    SELECT model_norm AS model, media, count(*) AS drive_days,
           count(DISTINCT serial_number) AS n_drives
    FROM daily GROUP BY ALL
),
f AS (
    SELECT model, count(*) FILTER (WHERE failed) AS failures,
           count(*) FILTER (WHERE failed AND days_after_fail = 0) AS failures_strict
    FROM drives GROUP BY model
)
SELECT dd.model, dd.media, dd.n_drives, dd.drive_days,
    coalesce(f.failures, 0) AS failures,
    round(coalesce(f.failures, 0) / (dd.drive_days / 365.0) * 100, 2) AS afr_pct,
    round(coalesce(f.failures_strict, 0) / (dd.drive_days / 365.0) * 100, 2) AS afr_strict_pct,
FROM dd LEFT JOIN f USING (model)
WHERE dd.drive_days >= {min_days} AND dd.n_drives >= {min_drives}
ORDER BY dd.n_drives DESC
"""


def afr_by_model(con: duckdb.DuckDBPyConnection, min_days: int = 10_000, min_drives: int = 100):
    return con.execute(AFR_SQL.format(min_days=min_days, min_drives=min_drives)).df()


def afr_overall(con: duckdb.DuckDBPyConnection):
    return con.execute(
        """SELECT media, count(DISTINCT serial_number) AS n_drives, count(*) AS drive_days,
               sum(failure) AS failures,
               round(sum(failure) / (count(*) / 365.0) * 100, 2) AS afr_pct
           FROM daily GROUP BY media ORDER BY media"""
    ).df()


def afr_by_quarter(con: duckdb.DuckDBPyConnection):
    return con.execute(
        """SELECT year(date) || '-Q' || quarter(date) AS quarter, media,
               sum(failure) AS failures,
               round(sum(failure) / (count(*) / 365.0) * 100, 2) AS afr_pct
           FROM daily GROUP BY ALL ORDER BY ALL"""
    ).df()


def main() -> None:
    con = connect()
    if not (DERIVED_DIR / "drives.parquet").exists():
        raise RuntimeError("先跑 python -m drivefail.drives 產生 drives.parquet")
    REPORTS_DIR.mkdir(exist_ok=True)
    by_model = afr_by_model(con)
    by_model.to_csv(REPORTS_DIR / "afr_by_model.csv", index=False)
    overall = afr_overall(con)
    by_q = afr_by_quarter(con)
    md = [
        "# 年化故障率（AFR）",
        "",
        "## 整體（HDD／SSD）",
        overall.to_markdown(index=False),
        "",
        "## 各季",
        by_q.to_markdown(index=False),
        "",
        "## 各型號（drive_days ≥ 10,000 且 ≥ 100 顆）",
        "`afr_strict_pct` 排除故障後又復活的序號。",
        "",
        by_model.to_markdown(index=False),
    ]
    (REPORTS_DIR / "afr.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(overall.to_string(index=False))
    print(by_model.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
