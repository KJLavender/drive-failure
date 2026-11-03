from datetime import date, timedelta

import numpy as np
import pandas as pd

from drivefail.evaluate import detection, false_alarm_rate, label, threshold_for_budget


def _drive(serial, n, fail=False, weight=1, end_pad=100):
    d0 = date(2025, 1, 1)
    return pd.DataFrame({
        "serial_number": serial,
        "date": [d0 + timedelta(i) for i in range(n)],
        "failed": fail,
        "weight": weight,
        "days_to_fail": [n - 1 - i for i in range(n)] if fail else [np.nan] * n,
        "days_to_last": [n - 1 - i for i in range(n)],
        "days_to_end": [n - 1 - i + end_pad for i in range(n)],
    })


def test_label_excludes_failure_day_and_uncertain_tail():
    df = pd.concat([_drive("f", 40, fail=True), _drive("h", 40, end_pad=0)], ignore_index=True)
    y = label(df, horizon=7)
    f = y[df.serial_number == "f"].to_numpy()
    assert np.isnan(f[-1])                       # 故障當天不算
    assert (f[-8:-1] == 1).all()                 # 前 1..7 天
    assert (f[:-8] == 0).all()
    h = y[df.serial_number == "h"].to_numpy()
    assert np.isnan(h[-7:]).all() and (h[:-7] == 0).all()   # 資料尾端 7 天不確定


def test_false_alarm_rate_is_weighted_per_drive_month():
    # 兩顆健康硬碟、權重 10、各 31 天（1 月）；只有一顆被警告一次
    df = pd.concat([_drive("a", 31, weight=10), _drive("b", 31, weight=10)], ignore_index=True)
    alarm = np.zeros(len(df), bool)
    alarm[3] = True
    assert false_alarm_rate(df, alarm) == 500.0   # 1/2 顆-月 → 每 1000 顆 500 次


def test_detection_and_lead_time():
    df = pd.concat([_drive("f1", 40, fail=True), _drive("f2", 40, fail=True)], ignore_index=True)
    alarm = np.zeros(len(df), bool)
    alarm[40 - 1 - 5] = True                     # f1 在故障前 5 天被警告
    rec, lead, n = detection(df, alarm, horizon=7)
    assert (rec, lead, n) == (0.5, 5.0, 2)


def test_threshold_respects_budget():
    rng = np.random.default_rng(0)
    df = pd.concat([_drive(f"h{i}", 30) for i in range(200)], ignore_index=True)
    score = rng.random(len(df))
    thr = threshold_for_budget(df, score, budget=100)
    assert false_alarm_rate(df, score >= thr) <= 100


def test_detection_accepts_nullable_int_from_duckdb():
    df = pd.concat([_drive("f", 20, fail=True), _drive("h", 20)], ignore_index=True)
    df["days_to_fail"] = df["days_to_fail"].astype("Int64")   # 和 DuckDB .df() 一樣
    df["days_to_fail"] = df["days_to_fail"].astype("float64")  # train.load_split 的轉換
    alarm = np.ones(len(df), bool)
    assert detection(df, alarm, horizon=7)[0] == 1.0
