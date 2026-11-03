"""Week 3 實驗：滾動重訓（Q5）、跨廠牌（Q3）、錯誤案例分析、信賴區間。

全部用 N = 30 天、LightGBM（Week 2 的正則化設定）。
"""

import argparse
import json
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from drivefail.evaluate import (bootstrap_ci, detection, false_alarm_rate, operating_point,
                                pr_auc, threshold_for_budget)
from drivefail.features import BACKBLAZE_FIVE
from drivefail.paths import REPORTS_DIR
from drivefail.train import (TEST_START, TRAIN_NEG_RATE, VALID_START, _labeled, feature_cols,
                             fit_lgbm, load_split)

H = 30
BUDGET = 2.0
QUARTERS = [date(2025, 1, 1), date(2025, 4, 1), date(2025, 7, 1), date(2025, 10, 1),
            date(2026, 1, 1), date(2026, 4, 1), None]  # None = 資料結尾
QNAME = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]
# 真正的錯誤計數器。排除正常使用就會增加的（240–242 使用量、192/193 載入次數、12 開機次數）
# 與 Seagate 複合編碼的 1/7
ERROR_COUNTERS = [5, 10, 183, 184, 187, 188, 189, 197, 198, 199]


def _con():
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    return con


def vendor(model: pd.Series) -> pd.Series:
    m = model.str.upper()
    return np.select(
        [m.str.startswith("ST") | m.str.startswith("SEAGATE"), m.str.startswith("TOSHIBA"),
         m.str.match(r"^(HGST|WUH|HUH|HMS|WDC)")],
        ["Seagate", "Toshiba", "WDC/HGST"], default="other")


def _fit(con, train_start, train_end, valid_start, valid_end, filt=None):
    tr = load_split(con, train_start, train_end, holdout=False, neg_rate=TRAIN_NEG_RATE)
    va = load_split(con, valid_start, valid_end, holdout=True)
    if filt is not None:
        tr, va = tr[filt(tr)], va[filt(va)]
    t, y = _labeled(tr, H)
    v, yv = _labeled(va, H)
    model = fit_lgbm(t, y, v, yv, feature_cols(t))
    return model, va


def _point(name, quarter, te, s, thr, extra=None):
    op = operating_point(te, s, thr, H)
    return {"strategy": name, "test_quarter": quarter, **op,
            "pr_auc": round(pr_auc(te, s, H), 4), **(extra or {})}


# ---------- Q5：滾動重訓 ----------

def rolling(con) -> pd.DataFrame:
    rows = []
    static, sva = _fit(con, QUARTERS[0], QUARTERS[2], QUARTERS[2], QUARTERS[3])
    static_thr = threshold_for_budget(sva, static.predict(sva), BUDGET)
    for k in range(3, 6):  # 測試 2025Q4、2026Q1、2026Q2
        te = load_split(con, QUARTERS[k], QUARTERS[k + 1], holdout=True)
        s = static.predict(te)
        rows.append(_point("static", QNAME[k], te, s, static_thr))
        # 只重選門檻：模型不動，門檻用前一季
        if k > 3:
            pva = load_split(con, QUARTERS[k - 1], QUARTERS[k], holdout=True)
            rows.append(_point("static_rethreshold", QNAME[k], te, s,
                               threshold_for_budget(pva, static.predict(pva), BUDGET)))
        for name, start in [("expanding", QUARTERS[0]), ("sliding_2q", QUARTERS[max(0, k - 3)])]:
            if k == 3:
                continue  # 2025Q4 時三種策略相同
            m, va = _fit(con, start, QUARTERS[k - 1], QUARTERS[k - 1], QUARTERS[k])
            s = m.predict(te)
            rows.append(_point(name, QNAME[k], te, s, threshold_for_budget(va, m.predict(va), BUDGET)))
        print(f"[rolling] {QNAME[k]} done")
    return pd.DataFrame(rows)


# ---------- Q3：跨廠牌 ----------

def cross_vendor(con) -> pd.DataFrame:
    te_all = load_split(con, TEST_START, None, holdout=True)
    va_all = load_split(con, VALID_START, TEST_START, holdout=True)
    te_all["vendor"] = vendor(te_all["model"])
    va_all["vendor"] = vendor(va_all["model"])
    targets = ["Seagate", "Toshiba", "WDC/HGST"]
    rows = []
    for src in ["all", *targets]:
        filt = None if src == "all" else (lambda d, v=src: vendor(d["model"]) == v)
        m, sva = _fit(con, None, VALID_START, VALID_START, TEST_START, filt)
        src_thr = threshold_for_budget(sva, m.predict(sva), BUDGET)
        for tgt in targets:
            te = te_all[te_all.vendor == tgt].reset_index(drop=True)
            va = va_all[va_all.vendor == tgt].reset_index(drop=True)
            s = m.predict(te)
            tgt_thr = threshold_for_budget(va, m.predict(va), BUDGET)
            src_op = operating_point(te, s, src_thr, H)
            tgt_op = operating_point(te, s, tgt_thr, H)
            rows.append({
                "train_on": src, "test_on": tgt, "n_failures": src_op["n_failures"],
                "pr_auc": round(pr_auc(te, s, H), 4),
                "recall@src_thr": src_op["recall"], "fa@src_thr": src_op["false_alarms_per_1k_month"],
                "recall@tgt_thr": tgt_op["recall"], "fa@tgt_thr": tgt_op["false_alarms_per_1k_month"],
            })
        print(f"[cross] train_on={src} done")
    return pd.DataFrame(rows)


# ---------- 錯誤案例 + 信賴區間 ----------

def error_analysis(con) -> dict:
    m, va = _fit(con, None, VALID_START, VALID_START, TEST_START)
    te = load_split(con, TEST_START, None, holdout=True)
    s_va, s = m.predict(va), m.predict(te)
    thr = threshold_for_budget(va, s_va, BUDGET)
    out = {"operating_point": operating_point(te, s, thr, H), **bootstrap_ci(te, s, thr, H)}

    te = te.assign(score=s, alarm=s >= thr, vendor=vendor(te["model"]))
    win = te[te.days_to_fail.between(1, H)]
    failed_ids = te.loc[te.days_to_fail == 0, "serial_number"].unique()
    win = win[win.serial_number.isin(failed_ids)]
    d30 = [f"s{a}_d30" for a in ERROR_COUNTERS if f"s{a}_d30" in te]
    bb5 = [f"s{a}" for a in BACKBLAZE_FIVE]
    per = win.groupby("serial_number").agg(
        detected=("alarm", "max"),
        vendor=("vendor", "first"),
        model=("model", "first"),
        max_score=("score", "max"),
    )
    per["any_bb5"] = win.groupby("serial_number")[bb5].max().fillna(0).gt(0).any(axis=1)
    per["any_counter_rise"] = win.groupby("serial_number")[d30].max().fillna(0).gt(0).any(axis=1)
    # 沉默故障：故障前 30 天 5 指標全 0，且沒有任何錯誤計數器上升
    per["silent"] = ~per.any_bb5 & ~per.any_counter_rise
    missed = per[~per.detected]
    out["failures"] = int(len(per))
    out["silent_share_all_failures"] = round(float(per.silent.mean()), 4)
    out["silent_share_missed"] = round(float(missed.silent.mean()), 4)
    out["recall_excluding_silent"] = round(float(per.loc[~per.silent, "detected"].mean()), 4)
    out["by_vendor"] = (per.groupby("vendor")
                        .agg(failures=("detected", "size"), recall=("detected", "mean"),
                             silent_share=("silent", "mean"))
                        .round(3).reset_index().to_dict("records"))

    # 誤報：被警告過的健康硬碟
    h = te[~te.failed]
    fa_ids = h.loc[h.alarm, "serial_number"].unique()
    fa = h[h.serial_number.isin(fa_ids)].groupby("serial_number").agg(
        vendor=("vendor", "first"), **{f"max_{c}": (c, "max") for c in bb5})
    out["false_alarm_drives"] = int(len(fa))
    out["false_alarm_with_bb5_nonzero"] = round(float(fa.filter(like="max_s").fillna(0).gt(0).any(axis=1).mean()), 4) if len(fa) else None
    healthy_bb5 = h.groupby("serial_number")[bb5].max().fillna(0).gt(0).any(axis=1)
    out["healthy_drives_with_bb5_nonzero"] = round(float(healthy_bb5.mean()), 4)
    out["false_alarm_vendor_mix"] = fa.vendor.value_counts(normalize=True).round(3).to_dict()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parts", nargs="*", default=["rolling", "cross", "errors"])
    args = ap.parse_args()
    con = _con()
    REPORTS_DIR.mkdir(exist_ok=True)
    if "rolling" in args.parts:
        r = rolling(con)
        r.to_csv(REPORTS_DIR / "week3_rolling.csv", index=False)
        print(r.to_string(index=False))
    if "cross" in args.parts:
        c = cross_vendor(con)
        c.to_csv(REPORTS_DIR / "week3_cross_vendor.csv", index=False)
        print(c.to_string(index=False))
    if "errors" in args.parts:
        e = error_analysis(con)
        (REPORTS_DIR / "week3_errors.json").write_text(json.dumps(e, indent=2, default=str), encoding="utf-8")
        print(json.dumps(e, indent=2, default=str))


if __name__ == "__main__":
    main()
