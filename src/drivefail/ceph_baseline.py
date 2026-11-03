"""與 Ceph diskprediction_local（Red Hat 模型）比較。

步驟：
1. extract：抽出測試期保留組硬碟的原始 SMART（多抓 5 天湊滿 6 天視窗）
2. scripts/ceph_score.py：在舊 sklearn 環境替每顆硬碟每天打分（見該檔說明）
3. compare：Ceph 的「Bad」對比 N=14、「Bad 或 Warning」對比 N=30；
   我們的 LightGBM 調到和 Ceph 相同的誤報率，再比偵測率。
只比 Ceph 有模型的硬碟：Seagate（ST…）與 HGST 開頭的型號。
"""

import argparse
import json
from datetime import timedelta

import duckdb
import numpy as np
import pandas as pd

from drivefail.db import connect
from drivefail.evaluate import false_alarm_rate, operating_point, threshold_for_budget
from drivefail.paths import DERIVED_DIR, REPORTS_DIR
from drivefail.train import (HOLDOUT_MOD, TEST_START, TRAIN_NEG_RATE, VALID_START, _labeled,
                             feature_cols, fit_lgbm, load_split)

CEPH_ATTRS = sorted({a for attrs in json.loads((DERIVED_DIR.parent / "ceph" / "config.json").read_text()).values()
                     for a in attrs if a.startswith("smart_")}, key=lambda a: (int(a.split("_")[1]), a))


def covered(model: pd.Series) -> pd.Series:
    return model.str.startswith("ST") | model.str.startswith("HGST")


def extract() -> None:
    feats = (DERIVED_DIR / "features.parquet").as_posix()
    con = connect()
    cols = ", ".join(f"d.{a}" for a in CEPH_ATTRS)
    con.execute(
        f"""COPY (
            WITH ids AS (
                SELECT DISTINCT serial_number FROM read_parquet('{feats}')
                WHERE hash(serial_number || ':split') % {HOLDOUT_MOD} = 0 AND date >= DATE '{TEST_START}'
            )
            SELECT d.serial_number, d.date, d.model, d.capacity_bytes AS user_capacity, {cols}
            FROM raw_daily d JOIN ids USING (serial_number)
            WHERE d.date >= DATE '{TEST_START - timedelta(days=5)}'
              AND (d.model LIKE 'ST%' OR d.model LIKE 'HGST%')
        ) TO '{(DERIVED_DIR / "ceph_input.parquet").as_posix()}' (FORMAT parquet)"""
    )
    print(con.execute(f"SELECT count(*), count(DISTINCT serial_number) FROM '{(DERIVED_DIR / 'ceph_input.parquet').as_posix()}'").fetchone())


def compare() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    te = load_split(con, TEST_START, None, holdout=True)
    va = load_split(con, VALID_START, TEST_START, holdout=True)
    te = te[covered(te.model)].reset_index(drop=True)
    pred = pd.read_parquet(DERIVED_DIR / "ceph_pred.parquet")
    pred["date"] = pd.to_datetime(pred["date"])
    te["date"] = pd.to_datetime(te["date"])
    te = te.merge(pred, on=["serial_number", "date"], how="left")
    te["ceph_class"] = te["ceph_class"].fillna(-1)
    tr_all = load_split(con, None, VALID_START, holdout=False, neg_rate=TRAIN_NEG_RATE)

    rows = []
    for horizon, ceph_alarm, label in [(14, te.ceph_class == 2, "Bad"),
                                       (30, te.ceph_class >= 1, "Bad or Warning")]:
        ceph_op = operating_point(te, ceph_alarm.astype(float).to_numpy(), 1.0, horizon)
        rows.append({"model": f"Ceph RH ({label})", "horizon": horizon, **ceph_op})
        tr, y = _labeled(tr_all, horizon)
        vl, yv = _labeled(va, horizon)
        m = fit_lgbm(tr, y, vl, yv, feature_cols(tr))
        va_cov = va[covered(va.model)].reset_index(drop=True)
        s_va, s_te = m.predict(va_cov), m.predict(te)
        budget = ceph_op["false_alarms_per_1k_month"]
        for b, tag in [(budget, "matched FA"), (2.0, "budget 2")]:
            rows.append({"model": f"LightGBM ({tag})", "horizon": horizon,
                         **operating_point(te, s_te, threshold_for_budget(va_cov, s_va, b), horizon)})
    res = pd.DataFrame(rows)
    res["unknown_share"] = round(float((te.ceph_class == -1).mean()), 4)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["extract", "compare"])
    args = ap.parse_args()
    if args.step == "extract":
        extract()
    else:
        res = compare()
        REPORTS_DIR.mkdir(exist_ok=True)
        res.to_csv(REPORTS_DIR / "ceph_comparison.csv", index=False)
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
