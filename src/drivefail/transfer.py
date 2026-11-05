"""遷移學習：新廠牌只有少量資料時怎麼辦（跨廠牌實驗的延伸，參考 TLDFP／DFPTL）。

模擬：目標廠牌的訓練硬碟只留 frac（依硬碟 hash 抽），其他廠牌資料完整。
比較：
- zero_shot：只用其他廠牌訓練
- target_only：只用少量目標資料
- pooled：其他廠牌 + 少量目標資料一起訓練
- finetune：其他廠牌預訓練 → 用少量目標資料以較小學習率繼續長樹

門檻與 early stopping 都用目標廠牌的驗證集（假設有少量標註可以校準，Week 3 已證明門檻不能搬）。
"""

import zlib

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from drivefail.evaluate import operating_point, pr_auc, threshold_for_budget
from drivefail.paths import REPORTS_DIR
from drivefail.train import (TEST_START, TRAIN_NEG_RATE, VALID_START, _labeled, feature_cols,
                             fit_lgbm, load_split)
from drivefail.week3 import vendor

H = 30
BUDGET = 2.0
FRACS = [0.05, 0.1, 0.25, 1.0]
TARGETS = ["WDC/HGST", "Toshiba"]
# 少量資料時 min_child_samples=500 會讓樹長不出來
SMALL = dict(min_child_samples=100)


def _keep_frac(serials: pd.Series, frac: float) -> np.ndarray:
    h = serials.map(lambda s: zlib.crc32(f"{s}:transfer".encode()) % 1000)
    return (h < frac * 1000).to_numpy()


def run() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    tr_all = load_split(con, None, VALID_START, holdout=False, neg_rate=TRAIN_NEG_RATE)
    va_all = load_split(con, VALID_START, TEST_START, holdout=True)
    te_all = load_split(con, TEST_START, None, holdout=True)
    for d in (tr_all, va_all, te_all):
        d["vendor"] = vendor(d["model"])
    tr_all, y_all = _labeled(tr_all, H)
    cols = feature_cols(tr_all.drop(columns="vendor"))

    rows = []
    for tgt in TARGETS:
        is_tgt = (tr_all.vendor == tgt).to_numpy()
        va = va_all[va_all.vendor == tgt].reset_index(drop=True)
        te = te_all[te_all.vendor == tgt].reset_index(drop=True)
        vl, yv = _labeled(va, H)
        src, ys = tr_all[~is_tgt], y_all[~is_tgt]

        def score(name, frac, model, n_pos):
            s_va, s_te = model.predict(va), model.predict(te)
            op = operating_point(te, s_te, threshold_for_budget(va, s_va, BUDGET), H)
            rows.append({"target": tgt, "strategy": name, "frac": frac, "train_pos_rows_target": n_pos,
                         "pr_auc": round(pr_auc(te, s_te, H), 4), "recall": op["recall"],
                         "false_alarms_per_1k_month": op["false_alarms_per_1k_month"],
                         "n_failures": op["n_failures"]})

        base = fit_lgbm(src, ys, vl, yv, cols, "zero_shot")
        score("zero_shot", 0.0, base, 0)
        # 從最佳輪數接著長（early stopping 後面還多留 50 棵樹）
        base_best = lgb.Booster(model_str=base.booster.model_to_string(num_iteration=base.booster.best_iteration))
        for frac in FRACS:
            keep = is_tgt & _keep_frac(tr_all.serial_number, frac)
            t, yt = tr_all[keep], y_all[keep]
            n_pos = int(yt.sum())
            score("target_only", frac, fit_lgbm(t, yt, vl, yv, cols, **SMALL), n_pos)
            pool = (~is_tgt) | keep
            score("pooled", frac, fit_lgbm(tr_all[pool], y_all[pool], vl, yv, cols), n_pos)
            ft = fit_lgbm(t, yt, vl, yv, cols, init_model=base_best, learning_rate=0.02, **SMALL)
            score("finetune", frac, ft, n_pos)
            print(f"[transfer] {tgt} frac={frac} pos_rows={n_pos} done")
    return pd.DataFrame(rows)


def main() -> None:
    res = run()
    REPORTS_DIR.mkdir(exist_ok=True)
    res.to_csv(REPORTS_DIR / "transfer.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
