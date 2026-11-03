from datetime import date, timedelta

import duckdb
import pandas as pd

from drivefail.db import register_daily
from drivefail.drives import build_drives
from drivefail.features import build_features

D0 = date(2025, 1, 1)


def _rows(serial, n_days, s5=lambda d: 0, fail_day=None):
    return [
        {"date": D0 + timedelta(d), "serial_number": serial, "model": "ST8000DM002",
         "capacity_bytes": 8_000_000_000_000, "failure": int(d == fail_day),
         "smart_5_raw": float(s5(d)), "smart_9_raw": 24.0 * (1000 + d), "smart_194_raw": 30.0}
        for d in range(n_days)
    ]


def test_window_features_and_labels():
    rows = (
        # 第 40 天起 smart 5 從 0 開始每天 +2，第 59 天故障（最後一筆）
        _rows("dying", 60, s5=lambda d: max(0, d - 39) * 2, fail_day=59)
        + _rows("zombie", 70, fail_day=30)                     # 故障後又出現 → 整顆排除
        + _rows("ok", 70)
    )
    con = duckdb.connect()
    con.register("raw_daily", pd.DataFrame(rows))
    register_daily(con, "raw_daily")
    build_drives(con)
    build_features(con, healthy_rate=1)  # rate=1 → 全部健康硬碟都留下

    f = con.execute("SELECT * FROM features ORDER BY serial_number, date").df()
    f["date"] = f.date.dt.date
    dying = f[f.serial_number == "dying"].set_index("date")

    assert "zombie" not in set(f.serial_number)
    assert dying.index.max() == D0 + timedelta(59)
    assert dying.loc[D0 + timedelta(59), "days_to_fail"] == 0
    assert dying.loc[D0 + timedelta(52), "days_to_fail"] == 7

    d45 = dying.loc[D0 + timedelta(45)]                       # s5 = 12，7 天前（第 38 天）= 0
    assert d45.s5 == 12 and d45.s5_d7 == 12 and d45.s5_new7 == 1
    d55 = dying.loc[D0 + timedelta(55)]                       # s5 = 32，第 48 天 = 18
    assert d55.s5_d7 == 32 - 18 and d55.s5_new7 == 0          # 已不是「首次出現」
    assert 0.5 < d55.s5_slope30 < 2.0                         # 視窗前半為 0、後半每天 +2，迴歸斜率介於兩者

    ok = f[f.serial_number == "ok"]
    assert ok.failed.eq(False).all() and ok.days_to_fail.isna().all()
    assert ok.power_on_days.iloc[0] == 1000
