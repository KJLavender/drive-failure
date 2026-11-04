"""部署用風險模型：訓練、校準、規則萃取、快照評分。

- 模型：LightGBM（N = 30），用 TEST_START 之前的非保留組硬碟訓練，2025Q4 保留組做 early stopping
- 校準：Platt（logit → 30 天內故障機率），依抽樣權重還原到整個機群
- 規則：深度 4 的決策樹，只保留驗證集上 lift ≥ RULE_MIN_LIFT 的葉子（FailureMiner 的「關鍵決策」簡化版）。
  單一規則精準度都不高（最好的「187 七天內上升」也只有約 14%），所以規則只當證據，不直接決定換碟
- 快照：測試期幾個日期，對保留組硬碟打分；之後 30 天是否故障當作標準答案
"""

import json
import pickle
from dataclasses import dataclass
from datetime import date

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from drivefail.evaluate import label
from drivefail.paths import DERIVED_DIR
from drivefail.train import (TEST_START, TRAIN_NEG_RATE, VALID_START, _labeled, feature_cols,
                             fit_lgbm, load_split)

H = 30
MODEL_DIR = DERIVED_DIR / "risk_model"
SNAPSHOTS = [date(2026, 3, 1), date(2026, 4, 1), date(2026, 5, 1), date(2026, 5, 31)]
RULE_MIN_LIFT = 10.0          # 命中規則的故障率 ÷ 整體故障率
RULE_MIN_FAILED_DRIVES = 5
RULE_FEATURES = 12  # 規則只用最重要的幾個特徵，人才看得懂


@dataclass
class RiskModel:
    booster: lgb.Booster
    cols: list[str]
    platt: LogisticRegression
    rules: list[dict]

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(df[self.cols], num_iteration=self.booster.best_iteration, raw_score=True)

    def prob(self, df: pd.DataFrame) -> np.ndarray:
        return self.platt.predict_proba(self.raw(df).reshape(-1, 1))[:, 1]

    def contributions(self, row: pd.DataFrame, k: int = 6) -> list[dict]:
        """單筆的 SHAP 貢獻（logit 空間），取絕對值最大的 k 個。"""
        c = self.booster.predict(row[self.cols], num_iteration=self.booster.best_iteration, pred_contrib=True)[0]
        s = pd.Series(c[:-1], index=self.cols)
        top = s.reindex(s.abs().sort_values(ascending=False).index[:k])
        return [{"feature": f, "contribution": round(float(v), 3), "value": _num(row.iloc[0][f])}
                for f, v in top.items()]

    def matched_rules(self, row: pd.Series) -> list[dict]:
        return [r for r in self.rules if all(_cond(row, c) for c in r["conditions"])]

    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(MODEL_DIR / "lgbm.txt", num_iteration=self.booster.best_iteration)
        pickle.dump({"cols": self.cols, "platt": self.platt}, open(MODEL_DIR / "meta.pkl", "wb"))
        (MODEL_DIR / "rules.json").write_text(json.dumps(self.rules, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls) -> "RiskModel":
        meta = pickle.load(open(MODEL_DIR / "meta.pkl", "rb"))
        booster = lgb.Booster(model_file=str(MODEL_DIR / "lgbm.txt"))
        rules = json.loads((MODEL_DIR / "rules.json").read_text(encoding="utf-8"))
        return cls(booster, meta["cols"], meta["platt"], rules)


def _num(v) -> float | None:
    return None if pd.isna(v) else round(float(v), 4)


def _cond(row: pd.Series, c: dict) -> bool:
    v = row.get(c["feature"])
    v = 0.0 if pd.isna(v) else float(v)  # 訓練時缺值補 0
    return v <= c["threshold"] if c["op"] == "<=" else v > c["threshold"]


def mine_rules(tr: pd.DataFrame, y: pd.Series, va: pd.DataFrame, yv: pd.Series, cols: list[str]) -> list[dict]:
    tree = DecisionTreeClassifier(max_depth=4, min_weight_fraction_leaf=0.0002, random_state=0)
    tree.fit(tr[cols].fillna(0), y, sample_weight=tr["weight"])
    t = tree.tree_
    rules = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            rules.append(conds)
            return
        f, thr = cols[t.feature[node]], float(t.threshold[node])
        walk(t.children_left[node], conds + [{"feature": f, "op": "<=", "threshold": round(thr, 4)}])
        walk(t.children_right[node], conds + [{"feature": f, "op": ">", "threshold": round(thr, 4)}])

    walk(0, [])
    out = []
    X, w = va.fillna(0), va["weight"].to_numpy()
    base_rate = float((w * yv.to_numpy()).sum() / w.sum())
    for conds in rules:
        m = np.ones(len(va), bool)
        for c in conds:
            col = X[c["feature"]].to_numpy()
            m &= col <= c["threshold"] if c["op"] == "<=" else col > c["threshold"]
        if not m.any():
            continue
        prec = float((w[m] * yv.to_numpy()[m]).sum() / w[m].sum())
        n_failed = int(va.loc[m & (yv.to_numpy() == 1), "serial_number"].nunique())
        if prec / base_rate >= RULE_MIN_LIFT and n_failed >= RULE_MIN_FAILED_DRIVES:
            out.append({"id": f"R{len(out) + 1}", "conditions": conds, "valid_precision": round(prec, 3),
                        "lift": round(prec / base_rate, 1), "valid_failed_drives": n_failed})
    return out


def train_and_save() -> RiskModel:
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    tr_all = load_split(con, None, TEST_START, holdout=False, neg_rate=TRAIN_NEG_RATE)
    va = load_split(con, VALID_START, TEST_START, holdout=True)
    tr, y = _labeled(tr_all, H)
    vl, yv = _labeled(va, H)
    cols = feature_cols(tr)
    m = fit_lgbm(tr, y, vl, yv, cols)
    booster = m.booster
    # 驗證集 early stopping 後再在同一份上校準：略樂觀，但測試集完全沒碰
    platt = LogisticRegression(C=1e6).fit(
        booster.predict(vl[cols], num_iteration=booster.best_iteration, raw_score=True).reshape(-1, 1),
        yv, sample_weight=vl["weight"])
    top = list(m.importance.index[:RULE_FEATURES])
    rules = mine_rules(tr, y, vl, yv, top)
    rm = RiskModel(booster, cols, platt, rules)
    rm.save()
    print(f"best_iteration={booster.best_iteration}  rules kept={len(rules)}")
    for r in rules:
        print(" ", r["id"], r["valid_precision"], r["lift"], r["valid_failed_drives"],
              " AND ".join(f"{c['feature']} {c['op']} {c['threshold']}" for c in r["conditions"]))
    return rm


def score_snapshots(rm: RiskModel) -> pd.DataFrame:
    """每個快照日、每顆保留組硬碟一列：機率、規則命中、之後 30 天是否故障。"""
    con = duckdb.connect()
    te = load_split(con, TEST_START, None, holdout=True)
    te["date"] = pd.to_datetime(te["date"])
    snaps = te[te.date.isin(pd.to_datetime(SNAPSHOTS))].copy()
    snaps["p30"] = rm.prob(snaps)
    snaps["rules_hit"] = [",".join(r["id"] for r in rm.matched_rules(row)) for _, row in snaps.iterrows()]
    snaps["failed_within_30d"] = snaps["days_to_fail"].between(1, H)
    keep = ["serial_number", "date", "model", "weight", "p30", "rules_hit", "failed_within_30d", "days_to_fail"]
    out = snaps[keep].sort_values("p30", ascending=False)
    out.to_parquet(MODEL_DIR / "snapshots.parquet", index=False)
    return out


def main() -> None:
    rm = train_and_save()
    s = score_snapshots(rm)
    # 校準檢查：依機率分組，預測 vs 實際（依權重還原）
    s["bin"] = pd.cut(s.p30, [0, 0.005, 0.02, 0.05, 0.2, 0.4, 1], include_lowest=True)
    g = s.groupby("bin", observed=True).apply(lambda d: pd.Series({
        "drives": len(d), "mean_p30": round(np.average(d.p30, weights=d.weight), 4),
        "actual_rate": round(np.average(d.failed_within_30d, weights=d.weight), 4)}), include_groups=False)
    print(g.to_string())


if __name__ == "__main__":
    main()
