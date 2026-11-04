"""產生靜態 dashboard（site/index.html）：高風險硬碟清單 → 點進去看 SMART 趨勢、cascade 路徑、agent 報告。

單一 HTML 檔，資料內嵌，不需要伺服器，可直接放 GitHub Pages。
"""

import json
import sqlite3

import duckdb
import pandas as pd

from drivefail.agent import AGENT_DIR
from drivefail.evidence import SMART_NAMES
from drivefail.paths import DERIVED_DIR, REPORTS_DIR, ROOT
from drivefail.week3 import ERROR_COUNTERS

HISTORY_DAYS = 120
SITE = ROOT / "site"


def _tiers_by_case(n: int) -> list[list[dict]]:
    """judge.db 的決策紀錄依 id 排序 = 案例順序。"""
    db = sqlite3.connect(AGENT_DIR / "judge.db")
    rows = db.execute("SELECT tiers FROM decisions ORDER BY id").fetchall()
    out = []
    for (t,) in rows[-n:]:
        tiers = []
        for x in json.loads(t):
            ans = x.get("answer") or {}
            tiers.append({"judge": x["judge"], "probs": ans.get("probabilities"),
                          "accepted": x.get("accepted"), "error": x.get("error")})
        out.append(tiers)
    return out


def _history(con, serial: str, day: str) -> dict:
    cols = ", ".join(f"s{a}" for a in ERROR_COUNTERS)
    h = con.execute(
        f"""SELECT date, {cols} FROM read_parquet('{(DERIVED_DIR / 'features.parquet').as_posix()}')
            WHERE serial_number = ? AND date BETWEEN DATE '{day}' - {HISTORY_DAYS} AND DATE '{day}' ORDER BY date""",
        [serial]).df()
    series = {}
    for a in ERROR_COUNTERS:
        v = h[f"s{a}"]
        if v.notna().any() and (v.fillna(0) != 0).any():
            series[str(a)] = [None if pd.isna(x) else float(x) for x in v]
    return {"dates": [str(pd.Timestamp(d).date()) for d in h.date], "series": series,
            "missing_days": int(h[[f"s{a}" for a in (5, 187, 197)]].isna().all(axis=1).sum())}


def build() -> None:
    dec = pd.read_csv(AGENT_DIR / "decisions.csv")
    tiers = _tiers_by_case(len(dec))
    con = duckdb.connect()
    cases = []
    for r, t in zip(dec.itertuples(), tiers):
        state = json.loads((AGENT_DIR / f"case_{r.case:03d}_state.json").read_text(encoding="utf-8"))
        rep_path = AGENT_DIR / f"case_{r.case:03d}_report.json"
        rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else None
        cases.append({
            "case": r.case, "serial": r.serial_number, "date": r.date, "zone": r.zone, "p30": r.p30,
            "model": state["drive"]["model"], "vendor": state["drive"]["vendor"],
            "status": r.status, "decision": r.decision if r.status == "decided" else "needs_human",
            "decided_by": r.decided_by if isinstance(r.decided_by, str) else "human_queue",
            "failed": bool(r.failed_within_30d), "tiers": t,
            "trend": state["trend_text"], "factors": state["top_factors"], "rules": state["rules_hit"],
            "costs": state["cost_model"]["expected_costs_at_model_risk"],
            "history": _history(con, r.serial_number, r.date),
            "report": rep and {**rep["report"], "faithfulness": {k: rep["faithfulness"][k] for k in (
                "faithfulness_precision", "unsupported_numbers", "recommendation_matches_decision")},
                "checked": [c["verified"] for c in rep["faithfulness"]["checked_findings"]], **rep["meta"]},
        })
    summary = {"policy": pd.read_csv(REPORTS_DIR / "agent_eval_policy.csv").to_dict("records")
               if (REPORTS_DIR / "agent_eval_policy.csv").exists() else []}
    data = json.dumps({"cases": cases, "names": {str(k): v for k, v in SMART_NAMES.items()}, **summary},
                      ensure_ascii=False)
    SITE.mkdir(exist_ok=True)
    html = (ROOT / "src" / "drivefail" / "dashboard_template.html").read_text(encoding="utf-8")
    (SITE / "index.html").write_text(html.replace("/*__DATA__*/null", data), encoding="utf-8")
    print(f"site/index.html：{len(cases)} 個案例")


if __name__ == "__main__":
    build()
