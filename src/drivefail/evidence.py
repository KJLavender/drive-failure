"""趨勢摘要層：把一顆硬碟在某天的狀態整理成「結構化證據 + 中文趨勢句」。

不讓 LLM 讀原始數字（SMARTTalk 的教訓）：所有計算都在這裡做完，LLM 只負責整理與措辭。
每個證據都有穩定的 key（例如 smart.197.change_7d），之後忠實度檢查逐條回查。
"""

import json
from functools import lru_cache

import duckdb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from drivefail.paths import DERIVED_DIR, REPORTS_DIR
from drivefail.risk import RiskModel
from drivefail.train import TEST_START, TRAIN_NEG_RATE, _labeled, load_split
from drivefail.week3 import ERROR_COUNTERS, vendor

SMART_NAMES = {
    1: "讀取錯誤率", 3: "起轉時間", 5: "重新配置磁區數", 7: "搜尋錯誤率", 9: "通電時數",
    10: "起轉重試次數", 12: "開機次數", 183: "SATA 降速次數", 184: "端對端錯誤", 187: "無法修正的錯誤",
    188: "指令逾時", 189: "高飛寫入", 192: "斷電收回次數", 193: "磁頭載入次數", 194: "溫度",
    197: "待處理磁區", 198: "離線無法修正", 199: "UDMA CRC 錯誤", 240: "磁頭飛行時數",
    241: "總寫入量", 242: "總讀取量",
}
SUFFIX = {"": "目前值", "d7": "7 天增加量", "d30": "30 天增加量", "new7": "7 天內首次出現",
          "slope30": "30 天斜率"}
K_SIMILAR = 20
SIMILAR_FEATURES = 15


def readable(feature: str) -> str:
    """s197_d7 → SMART 197（待處理磁區）7 天增加量"""
    if feature.startswith(("s", "n")) and feature[1:].split("_")[0].isdigit():
        attr, _, suf = feature[1:].partition("_")
        kind = "normalized " if feature[0] == "n" else ""
        return f"SMART {attr}（{SMART_NAMES.get(int(attr), '?')}）{kind}{SUFFIX.get(suf, suf)}"
    return {"power_on_days": "通電天數", "capacity_tb": "容量（TB）", "temp": "溫度",
            "temp_avg30": "30 天平均溫度", "temp_max30": "30 天最高溫度"}.get(feature, feature)


def rule_text(rule: dict) -> str:
    return " 且 ".join(f"{readable(c['feature'])} {'≤' if c['op'] == '<=' else '>'} {c['threshold']:g}"
                      for c in rule["conditions"])


def _f(v) -> float | None:
    return None if v is None or pd.isna(v) else round(float(v), 4)


class EvidenceBuilder:
    def __init__(self, rm: RiskModel | None = None):
        self.rm = rm or RiskModel.load()
        self.con = duckdb.connect()
        self.con.execute("SET memory_limit='4GB'")
        self.feats = (DERIVED_DIR / "features.parquet").as_posix()
        afr = pd.read_csv(REPORTS_DIR / "afr_by_model.csv")
        self.model_afr = dict(zip(afr.model, afr.afr_pct))
        errors = json.loads((REPORTS_DIR / "week3_errors.json").read_text(encoding="utf-8"))
        self.vendor_stats = {v["vendor"]: v for v in errors["by_vendor"]}
        # 錯誤計數器全讀不到的那天，之後 30 天的故障率（依權重還原）；只用測試期之前的資料
        missing = " AND ".join(f"s{a} IS NULL" for a in (5, 187, 197))
        self.missing_fail_rate = float(self.con.execute(
            f"""SELECT sum(weight * (days_to_fail BETWEEN 1 AND 30)::INT) / sum(weight)
                FROM read_parquet('{self.feats}') WHERE {missing} AND date < DATE '{TEST_START}'""").fetchone()[0])

    # ---------- 相似案例 ----------
    @lru_cache(maxsize=1)
    def _neighbors(self):
        tr = load_split(self.con, None, TEST_START, holdout=False, neg_rate=TRAIN_NEG_RATE)
        tr, y = _labeled(tr, 30)
        imp = pd.Series(self.rm.booster.feature_importance("gain"), index=self.rm.cols)
        cols = list(imp.sort_values(ascending=False).index[:SIMILAR_FEATURES])
        X = self._embed(tr[cols])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        nn = NearestNeighbors(n_neighbors=K_SIMILAR).fit((X - mu) / sd)
        meta = tr[["serial_number", "model", "date", "days_to_fail", "weight"]].reset_index(drop=True)
        return nn, cols, mu, sd, meta, y.to_numpy()

    @staticmethod
    def _embed(df: pd.DataFrame) -> np.ndarray:
        # 缺值本身是強訊號（讀不到 SMART 的硬碟常在 30 天內故障），補 0 會把它當成健康
        n_missing = df.isna().sum(axis=1).to_numpy(np.float64).reshape(-1, 1)
        x = df.fillna(0).to_numpy(np.float64)
        return np.hstack([np.sign(x) * np.log1p(np.abs(x)), n_missing])

    def similar(self, row: pd.DataFrame) -> dict:
        nn, cols, mu, sd, meta, y = self._neighbors()
        _, idx = nn.kneighbors((self._embed(row[cols]) - mu) / sd)
        idx = idx[0]
        w = meta.weight.to_numpy()[idx]
        failed = y[idx] == 1
        ex = meta.iloc[idx[failed]].drop_duplicates("serial_number").head(3)
        return {
            "k": K_SIMILAR,
            # 依抽樣權重還原：健康樣本被降採樣過
            "failed_within_30d_share": round(float((w * failed).sum() / w.sum()), 4),
            "failed_neighbors": int(failed.sum()),
            "examples": [{"model": r.model, "days_before_failure": int(r.days_to_fail)} for r in ex.itertuples()],
        }

    # ---------- 主流程 ----------
    def build(self, serial: str, day) -> dict:
        day = pd.Timestamp(day)
        hist = self.con.execute(
            f"SELECT * FROM read_parquet('{self.feats}') WHERE serial_number = ? AND date <= ? ORDER BY date",
            [serial, day.date()]).df()
        if hist.empty or pd.Timestamp(hist.date.iloc[-1]) != day:
            raise KeyError(f"{serial} 在 {day.date()} 沒有紀錄")
        feat_cols = [c for c in hist.columns if c in self.rm.cols]
        hist[feat_cols] = hist[feat_cols].astype("float32")
        row = hist.iloc[[-1]]
        r = row.iloc[0]
        p30 = float(self.rm.prob(row)[0])
        v = str(vendor(pd.Series([r.model]))[0])

        smart = {}
        for a in ERROR_COUNTERS:
            col = f"s{a}"
            if col not in hist or pd.isna(r[col]):
                continue
            cur, d7, d30 = _f(r[col]), _f(r.get(f"{col}_d7")), _f(r.get(f"{col}_d30"))
            if not (cur or d7 or d30):
                continue  # 全為 0 的指標不列
            nz = hist.loc[hist[col] > 0, "date"]
            smart[str(a)] = {
                "name": SMART_NAMES.get(a, ""), "current": cur, "change_7d": d7, "change_30d": d30,
                "first_nonzero_date": str(pd.Timestamp(nz.iloc[0]).date()) if len(nz) else None,
                "new_in_7d": bool(r.get(f"{col}_new7") == 1),
            }

        smart_missing = all(pd.isna(r.get(f"s{a}")) for a in ERROR_COUNTERS)
        ev = {
            "drive": {"serial_number": serial, "model": r.model, "vendor": v,
                      "capacity_tb": _f(r.capacity_tb), "age_years": _f(r.power_on_days / 365.25)},
            "snapshot_date": str(day.date()),
            "risk": {"p_fail_30d": round(p30, 4)},
            "smart": smart,
            "smart_missing": {"all_error_counters_missing": smart_missing,
                              "historical_fail_30d_rate_when_missing": round(self.missing_fail_rate, 4)},
            "temperature": {"current_c": _f(r.get("temp")), "max_30d_c": _f(r.get("temp_max30"))},
            "top_factors": [{**c, "name": readable(c["feature"]) + ("（缺值）" if c["value"] is None else "")}
                            for c in self.rm.contributions(row)],
            "rules_hit": [{"id": x["id"], "text": rule_text(x), "valid_precision": x["valid_precision"],
                           "lift": x["lift"]} for x in self.rm.matched_rules(r)],
            "model_context": {"model_afr_pct": _f(self.model_afr.get(r.model)),
                              "vendor_recall": _f(self.vendor_stats.get(v, {}).get("recall")),
                              "vendor_silent_failure_share": _f(self.vendor_stats.get(v, {}).get("silent_share"))},
            "similar_cases": self.similar(row),
        }
        ev["trend_text"] = trend_sentences(ev)
        return ev


def trend_sentences(ev: dict) -> list[str]:
    """確定性產生的中文趨勢句。LLM 只能引用這些句子與證據裡的數字。"""
    out = []
    d = ev["drive"]
    age = f"已使用 {d['age_years']:.1f} 年" if d["age_years"] is not None else "使用年數未知（無 SMART 9）"
    cap = f"{d['capacity_tb']:g} TB" if d["capacity_tb"] is not None else "容量未知"
    out.append(f"{d['model']}（{d['vendor']}，{cap}），{age}。")
    out.append(f"模型估計 30 天內故障機率 {ev['risk']['p_fail_30d']:.1%}。")
    for a, s in ev["smart"].items():
        parts = [f"SMART {a}（{s['name']}）目前 {s['current']:g}"]
        if s["change_7d"]:
            parts.append(f"過去 7 天增加 {s['change_7d']:g}")
        if s["change_30d"] and s["change_30d"] != s["change_7d"]:
            parts.append(f"過去 30 天增加 {s['change_30d']:g}")
        if s["first_nonzero_date"]:
            parts.append(f"首次出現於 {s['first_nonzero_date']}" + ("（7 天內新出現）" if s["new_in_7d"] else ""))
        out.append("，".join(parts) + "。")
    sm = ev["smart_missing"]
    if sm["all_error_counters_missing"]:
        out.append(f"當天讀不到任何 SMART 錯誤計數器（缺值，不是 0）；歷史上出現這種狀況時，"
                   f"{sm['historical_fail_30d_rate_when_missing']:.1%} 的硬碟在 30 天內故障。")
    elif not ev["smart"]:
        out.append("所有錯誤計數器（5、10、183、184、187、188、189、197、198、199）皆為 0 且無變化。")
    t = ev["temperature"]
    if t["current_c"] is not None and t["max_30d_c"] is not None:
        out.append(f"溫度 {t['current_c']:g}°C，30 天最高 {t['max_30d_c']:g}°C。")
    for rh in ev["rules_hit"]:
        out.append(f"命中規則 {rh['id']}：{rh['text']}（驗證集上 30 天內故障率 {rh['valid_precision']:.1%}，"
                   f"為平均的 {rh['lift']:g} 倍）。")
    m = ev["model_context"]
    if m["model_afr_pct"] is not None:
        out.append(f"同型號年化故障率 {m['model_afr_pct']:g}%。")
    sc = ev["similar_cases"]
    out.append(f"最相似的 {sc['k']} 個歷史案例中，依機群比例還原後 {sc['failed_within_30d_share']:.1%} 在 30 天內故障。")
    return out


if __name__ == "__main__":
    import sys
    snaps = pd.read_parquet(DERIVED_DIR / "risk_model" / "snapshots.parquet")
    top = snaps.iloc[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
    ev = EvidenceBuilder().build(top.serial_number, top.date)
    print(json.dumps(ev, ensure_ascii=False, indent=2))
