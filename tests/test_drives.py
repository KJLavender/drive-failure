from datetime import date, timedelta

import duckdb
import pandas as pd

from drivefail.db import register_daily
from drivefail.drives import build_drives, quality_summary

D0 = date(2025, 1, 1)


def _rows(serial, model, days, fail_day=None, skip=()):
    return [
        {"date": D0 + timedelta(d), "serial_number": serial, "model": model,
         "capacity_bytes": 1, "failure": int(d == fail_day), "smart_9_raw": 24.0 * (100 + d)}
        for d in days if d not in skip
    ]


def _con(rows):
    con = duckdb.connect()
    con.register("src", pd.DataFrame(rows))
    register_daily(con, "src")
    build_drives(con)
    return con


def test_labels_and_quality():
    rows = (
        _rows("healthy", "ST8000DM002", range(100))                      # 到資料尾端都健康 → 右設限
        + _rows("dead", "ST8000DM002", range(50), fail_day=49)           # 正常故障
        + _rows("zombie", "ST8000DM002", range(60), fail_day=40)         # 故障後又出現 19 天
        + _rows("gone", "ST8000DM002", range(20))                        # 消失沒標故障
        + _rows("gappy", "ST8000DM002", range(100), skip={10, 11})       # 中間缺 2 天
        + _rows("dup", "WDC WUH721816ALE6L4", range(3))
        + _rows("dup", "WUH721816ALE6L4", range(3, 6))                   # 同型號不同寫法，不算碰撞
        + _rows("ssd", "CT250MX500SSD1", range(100))
    )
    con = _con(rows)
    d = con.execute("SELECT * FROM drives").df().set_index("serial_number")

    assert d.loc["dead", "failed"] and d.loc["dead", "days_after_fail"] == 0
    assert d.loc["zombie", "days_after_fail"] == 19
    assert d.loc["gappy", "missing_days"] == 2
    assert d.loc["dead", "poh_first"] == 2400 and d.loc["dead", "poh_last"] == 24 * 149
    assert d.loc["dup", "n_models"] == 1
    assert d.loc["ssd", "media"] == "SSD" and d.loc["healthy", "media"] == "HDD"

    q = quality_summary(con, censor_days=30)
    assert q["failed"] == 2
    assert q["resurrected"] == 1
    assert q["vanished_without_failure"] == 2        # gone, dup
    assert q["right_censored"] == 3                  # healthy, gappy, ssd
    assert q["serial_collisions"] == 0
