"""Week 2 實驗：規則式 baseline、邏輯斯迴歸、LightGBM、去掉 Backblaze 5 指標的消融。

切分（不能隨機切，否則會用未來預測過去）：
- 時間：train < VALID_START ≤ valid < TEST_START ≤ test
- 硬碟：hash 出 20% 保留組，只出現在 valid／test
- 每段資料只用該段結束前已知的資訊定標籤（未來才故障 → 不確定）

門檻：在 valid 上找「每 1000 顆每月誤報 ≤ 預算」的門檻，套到 test。
"""

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from drivefail.evaluate import label, operating_point, pr_auc, threshold_for_budget
from drivefail.features import BACKBLAZE_FIVE
from drivefail.paths import DERIVED_DIR, REPORTS_DIR

VALID_START = date(2025, 10, 1)
TEST_START = date(2026, 1, 1)
HOLDOUT_MOD = 5
TRAIN_NEG_RATE = 10
BUDGETS = [1.0, 2.0, 5.0]  # 每 1000 顆健康硬碟每月誤報上限
HORIZONS = [7, 14, 30]

ID_COLS = ["serial_number", "date", "model", "failed", "weight",
           "days_to_fail", "days_to_last", "days_to_end"]
BB5_RE = re.compile(r"^s(" + "|".join(map(str, BACKBLAZE_FIVE)) + r")(_|$)")


def load_split(con, start: date | None, end: date | None, holdout: bool, neg_rate: int = 1) -> pd.DataFrame:
    """讀一段時間的特徵，並把標籤欄位截斷在 end 之前。"""
    feats = DERIVED_DIR / "features.parquet"
    end_sql = f"DATE '{end}'" if end else "(SELECT max(date) + 1 FROM read_parquet('" + feats.as_posix() + "'))"
    where = [f"(hash(serial_number || ':split') % {HOLDOUT_MOD} = 0) = {str(holdout).lower()}"]
    if start:
        where.append(f"date >= DATE '{start}'")
    if end:
        where.append(f"date < DATE '{end}'")
    if neg_rate > 1:
        # 正樣本候選（30 天內故障）全留，其餘依天抽樣並放大權重
        where.append(f"(days_to_fail <= 30 OR hash(serial_number || date::VARCHAR) % {neg_rate} = 0)")
    df = con.execute(
        f"""WITH e AS (SELECT {end_sql} AS end_d)
        SELECT f.* REPLACE (
            -- 這段資料結束後才故障 → 在這段裡看不到
            CASE WHEN f.date + f.days_to_fail::INT < e.end_d THEN f.days_to_fail END AS days_to_fail,
            coalesce(f.date + f.days_to_fail::INT < e.end_d, false) AS failed,
            datediff('day', f.date, e.end_d) - 1 AS days_to_end,
            least(f.days_to_last, datediff('day', f.date, e.end_d) - 1) AS days_to_last,
            f.weight * CASE WHEN {neg_rate} > 1 AND NOT coalesce(f.days_to_fail <= 30, false)
                            THEN {neg_rate} ELSE 1 END AS weight
        )
        FROM read_parquet('{feats.as_posix()}') f, e
        WHERE {' AND '.join(where)}"""
    ).df()
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    df[feat_cols] = df[feat_cols].astype("float32")
    # DuckDB 給的是可空整數（pd.NA），布林遮罩會炸；統一成 float + NaN
    dist = ["days_to_fail", "days_to_last", "days_to_end"]
    df[dist] = df[dist].astype("float64")
    df["failed"] = df["failed"].astype(bool)
    return df


def feature_cols(df: pd.DataFrame, drop_bb5: bool = False) -> list[str]:
    cols = [c for c in df.columns if c not in ID_COLS]
    return [c for c in cols if not (drop_bb5 and BB5_RE.match(c))]


# ---------- 模型 ----------

def rule_any(df: pd.DataFrame) -> np.ndarray:
    """Backblaze 規則：5 個指標任一 raw > 0 就警告。"""
    cols = [f"s{a}" for a in BACKBLAZE_FIVE if f"s{a}" in df]
    return (df[cols].fillna(0).to_numpy() > 0).any(axis=1).astype(float)


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """連續版規則分數：5 個指標 log1p 加總，可以掃門檻。"""
    cols = [f"s{a}" for a in BACKBLAZE_FIVE if f"s{a}" in df]
    return np.log1p(df[cols].fillna(0).clip(lower=0).to_numpy()).sum(axis=1)


@dataclass
class Fitted:
    name: str
    predict: callable
    importance: pd.Series | None = None
    booster: lgb.Booster | None = None


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def fit_logreg(tr: pd.DataFrame, y: pd.Series, cols: list[str]) -> Fitted:
    scaler = StandardScaler()
    X = scaler.fit_transform(_signed_log(tr[cols].fillna(0).to_numpy()))
    m = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
    m.fit(X, y, sample_weight=tr["weight"])
    pred = lambda d: m.decision_function(scaler.transform(_signed_log(d[cols].fillna(0).to_numpy())))
    return Fitted("logreg", pred, pd.Series(m.coef_[0], index=cols))


def fit_lgbm(tr, y, va, yv, cols, name="lgbm", init_model=None, **overrides) -> Fitted:
    # 不用 scale_pos_weight：疊在抽樣權重上會把機率推到 1.0，大量同分導致門檻切不開。
    # 強正則化：每顆故障硬碟貢獻約 N 筆幾乎相同的正樣本，不壓住就會「背硬碟」
    # （未正則化時驗證集 AP 在第 1 輪就最高，之後一路下降）。
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=500,
                  lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  metric="average_precision", verbose=-1, seed=0) | overrides
    dtr = lgb.Dataset(tr[cols], y, weight=tr["weight"])
    dva = lgb.Dataset(va[cols], yv, weight=va["weight"], reference=dtr)
    # init_model：在既有模型上繼續長樹（遷移學習的微調）
    m = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva], init_model=init_model,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    # 用 logit 排序，避免機率飽和成同分
    return Fitted(name, lambda d: m.predict(d[cols], num_iteration=m.best_iteration, raw_score=True), imp, m)


# ---------- 實驗 ----------

def _labeled(df, horizon):
    y = label(df, horizon)
    m = y.notna()
    return df[m], y[m].astype(int)


def evaluate_scores(name, horizon, va, sv, te, st) -> list[dict]:
    rows = [{"model": name, "horizon": horizon, "budget": "pr_auc",
             "test_pr_auc": round(pr_auc(te, st, horizon), 4)}]
    for b in BUDGETS:
        thr = threshold_for_budget(va, sv, b)
        rows.append({"model": name, "horizon": horizon, "budget": b,
                     **operating_point(te, st, thr, horizon)})
    return rows


def run(horizons: list[int]) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    tr_all = load_split(con, None, VALID_START, holdout=False, neg_rate=TRAIN_NEG_RATE)
    va = load_split(con, VALID_START, TEST_START, holdout=True)
    te = load_split(con, TEST_START, None, holdout=True)
    print(f"train {len(tr_all):,} rows / valid {len(va):,} / test {len(te):,}")

    results, importances = [], {}
    # 規則式：固定一個操作點，直接在 test 上量
    for h in horizons:
        st = rule_any(te)
        results.append({"model": "rule_bb5_any", "horizon": h, "budget": "fixed",
                        **operating_point(te, st, 1.0, h)})
        results += evaluate_scores("rule_bb5_score", h, va, rule_score(va), te, rule_score(te))

    for h in horizons:
        tr, y = _labeled(tr_all, h)
        vl, yv = _labeled(va, h)
        print(f"[h={h}] train pos {int(y.sum()):,} / neg {int((y == 0).sum()):,}")
        all_cols = feature_cols(tr)
        no_bb5 = feature_cols(tr, drop_bb5=True)
        models = [
            fit_logreg(tr, y, all_cols),
            fit_lgbm(tr, y, vl, yv, all_cols, "lgbm"),
            fit_lgbm(tr, y, vl, yv, no_bb5, "lgbm_no_bb5"),
        ]
        for m in models:
            results += evaluate_scores(m.name, h, va, m.predict(va), te, m.predict(te))
            if m.importance is not None:
                importances[f"{m.name}_h{h}"] = m.importance
            print(f"  done {m.name}")

    REPORTS_DIR.mkdir(exist_ok=True)
    res = pd.DataFrame(results)
    res.to_csv(REPORTS_DIR / "week2_results.csv", index=False)
    pd.DataFrame(importances).to_csv(REPORTS_DIR / "week2_importance.csv")
    meta = {"train_rows": len(tr_all), "valid_rows": len(va), "test_rows": len(te),
            "valid": [str(VALID_START), str(TEST_START)], "test_start": str(TEST_START),
            "budgets_per_1k_drive_month": BUDGETS}
    (REPORTS_DIR / "week2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    args = ap.parse_args()
    res = run(args.horizons)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
