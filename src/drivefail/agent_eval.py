"""Agent 評估：決策成本、cascade 分層、報告忠實度，並產生人工評分表。

事後成本（realized cost）用 policies/costs.yaml，依實際 30 天內是否故障計算；
健康硬碟依抽樣權重還原到整個機群，結果以「每 1000 顆硬碟、每個快照」表示。
"""

import json

import duckdb
import numpy as np
import pandas as pd

from drivefail.agent import (AGENT_DIR, boundaries, expected_costs, faithfulness, load_costs, realized_cost,
                            report_facts)
from drivefail.paths import DERIVED_DIR, REPORTS_DIR
from drivefail.features import BACKBLAZE_FIVE


def _cost_per_1k(actions: pd.Series, snaps: pd.DataFrame, c: dict) -> dict:
    cost = np.array([realized_cost(a, f, c) for a, f in zip(actions, snaps.failed_within_30d)])
    w = snaps.weight.to_numpy()
    n_snap = snaps.date.nunique()
    fails = snaps.failed_within_30d.to_numpy()
    rep = (actions == "replace").to_numpy()
    watch = (actions == "watch").to_numpy()
    per1k = 1000 / (w.sum() / n_snap)
    return {
        "cost_per_1k_drives": round(float((cost * w).sum() / n_snap * per1k), 2),
        "replacements_per_1k": round(float((rep * w).sum() / n_snap * per1k), 2),
        "watched_per_1k": round(float((watch * w).sum() / n_snap * per1k), 2),
        "failures_replaced_in_advance": f"{(rep & fails).sum()}/{fails.sum()}",
        "replace_precision": round(float(fails[rep].mean()), 3) if rep.any() else None,
    }


def policy_comparison(c: dict) -> pd.DataFrame:
    snaps = pd.read_parquet(DERIVED_DIR / "risk_model" / "snapshots.parquet")
    snaps["date"] = pd.to_datetime(snaps["date"])
    con = duckdb.connect()
    bb5 = ", ".join(f"s{a}" for a in BACKBLAZE_FIVE)
    vals = con.execute(f"""SELECT serial_number, date, {bb5} FROM read_parquet('{(DERIVED_DIR / 'features.parquet').as_posix()}')
                           WHERE date IN ({', '.join(f"DATE '{d.date()}'" for d in snaps.date.unique())})""").df()
    vals["date"] = pd.to_datetime(vals["date"])
    snaps = snaps.merge(vals, on=["serial_number", "date"], how="left")
    any_bb5 = (snaps[[f"s{a}" for a in BACKBLAZE_FIVE]].fillna(0) > 0).any(axis=1)
    _, hi = boundaries(c)
    policies = {
        "always_ignore": pd.Series("ignore", index=snaps.index),
        "backblaze_rule (BB5 任一 > 0 → 換)": pd.Series(np.where(any_bb5, "replace", "ignore"), index=snaps.index),
        "model + cost (換／觀察／忽略)": snaps.p30.map(lambda p: min(expected_costs(p, c).items(), key=lambda kv: kv[1])[0]),
        "model, replace only (p ≥ 換碟分界)": pd.Series(np.where(snaps.p30 >= hi, "replace", "ignore"), index=snaps.index),
        "oracle (壞的才換)": pd.Series(np.where(snaps.failed_within_30d, "replace", "ignore"), index=snaps.index),
    }
    return pd.DataFrame([{"policy": k, **_cost_per_1k(v, snaps, c)} for k, v in policies.items()])


def cost_sensitivity(c: dict, failure_costs=(3, 5, 10, 20)) -> pd.DataFrame:
    """非計畫故障的成本不確定；看結論是否隨成本比例改變。"""
    rows = []
    for fc in failure_costs:
        cc = {**c, "failure_cost": float(fc)}
        pol = policy_comparison(cc).set_index("policy")["cost_per_1k_drives"]
        lo, hi = boundaries(cc)
        rows.append({"failure_cost": fc, "watch_from_p": round(lo, 3), "replace_from_p": round(hi, 3), **pol.to_dict()})
    return pd.DataFrame(rows)


def cascade_summary(c: dict, agent_dir=AGENT_DIR) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(agent_dir / "decisions.csv")
    # 排隊等人工的案例，在人回覆前先當作「觀察」（保守、便宜）
    d["final"] = np.where(d.status == "decided", d.decision, "watch")
    d["cost_cascade"] = [realized_cost(a, f, c) for a, f in zip(d.final, d.failed_within_30d)]
    d["cost_model"] = [realized_cost(a, f, c) for a, f in zip(d.model_action, d.failed_within_30d)]
    by_tier = (d.assign(tier=d.decided_by.fillna("human_queue"))
                .groupby(["zone", "tier"]).agg(cases=("case", "size"), failed=("failed_within_30d", "sum"))
                .reset_index())
    llm = d[d.decided_by == "llm"]
    g = d[d.zone == "gray"]
    stats = {
        "cases": len(d),
        "escalated_to_llm": int(d.path.str.contains("llm").sum()),
        "decided_by_llm": len(llm),
        "queued_for_human": int((d.status == "queued").sum()),
        "llm_agrees_with_model_action": int((llm.decision == llm.model_action).sum()) if len(llm) else 0,
        # 案例是從快照表逐列抽的，健康硬碟在快照表裡已經是 1/10 抽樣 → 必須依權重還原
        "mean_cost_model_only": round(np.average(d.cost_model, weights=d.weight), 4),
        "mean_cost_cascade": round(np.average(d.cost_cascade, weights=d.weight), 4),
        "gray_zone_cost_model_only": round(np.average(g.cost_model, weights=g.weight), 4),
        "gray_zone_cost_cascade": round(np.average(g.cost_cascade, weights=g.weight), 4),
        "judge_cost_usd_total": round(d.judge_cost_usd.sum(), 4),
        "judge_latency_s_mean_when_llm": round(d[d.path.str.contains("llm")].judge_latency_s.mean(), 1),
    }
    return by_tier, stats


def recheck_faithfulness() -> pd.DataFrame:
    """用目前的檢查器重新檢查已存的報告（檢查器修正後不必重新呼叫 LLM）。"""
    r = pd.read_csv(AGENT_DIR / "reports.csv")
    for i, row in r.iterrows():
        path = AGENT_DIR / f"case_{row.case:03d}_report.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        state = json.loads((AGENT_DIR / f"case_{row.case:03d}_state.json").read_text(encoding="utf-8"))
        fa = faithfulness(saved["report"], report_facts(state, row.decision), state["trend_text"], row.decision)
        saved["faithfulness"] = fa
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        r.loc[i, ["faithfulness_precision", "unsupported_numbers"]] = [fa["faithfulness_precision"], len(fa["unsupported_numbers"])]
    r.to_csv(AGENT_DIR / "reports.csv", index=False)
    return r


def report_summary() -> tuple[dict, pd.DataFrame]:
    r = recheck_faithfulness()
    stats = {
        "reports": len(r),
        "mean_faithfulness_precision": round(r.faithfulness_precision.mean(), 4),
        "reports_fully_faithful": int((r.faithfulness_precision == 1).sum()),
        "reports_with_unsupported_numbers": int((r.unsupported_numbers > 0).sum()),
        "recommendation_matches_decision": int(r.recommendation_matches_decision.sum()),
        "mean_latency_s": round(r.latency_s.mean(), 1),
        "mean_cost_usd": round(r.cost_usd.mean(), 4),
    }
    # 人工評分表：使用者填寫
    rows = []
    for case in r.case:
        rep = json.loads((AGENT_DIR / f"case_{case:03d}_report.json").read_text(encoding="utf-8"))
        rows.append({"case": case, "headline": rep["report"]["headline"],
                     "recommendation": rep["report"]["recommendation"],
                     "evidence_correct_1to5": "", "recommendation_reasonable_1to5": "", "notes": ""})
    return stats, pd.DataFrame(rows)


def main() -> None:
    c = load_costs()
    pol = policy_comparison(c)
    pol.to_csv(REPORTS_DIR / "agent_eval_policy.csv", index=False)
    sens = cost_sensitivity(c)
    by_tier, cstats = cascade_summary(c)
    v1_dir = AGENT_DIR.parent / "agent_v1"
    v1 = cascade_summary(c, v1_dir)[1] if (v1_dir / "decisions.csv").exists() else None
    rstats, sheet = report_summary()
    sheet.to_csv(REPORTS_DIR / "agent_human_review.csv", index=False, encoding="utf-8-sig")
    lo, hi = boundaries(c)
    md = [
        "# Agent 評估", "",
        f"成本模型（policies/costs.yaml）：換碟 {c['replace_cost']}、非計畫故障 {c['failure_cost']}、"
        f"觀察 {c['watch_cost']}（故障時省 {c['watch_mitigation']:.0%}）→ p < {lo:.0%} 忽略、"
        f"{lo:.0%}–{hi:.0%} 觀察、> {hi:.0%} 換碟。", "",
        "## 1. 決策策略的事後成本（所有快照硬碟，依權重還原）", pol.to_markdown(index=False), "",
        "### 成本敏感度（非計畫故障成本 = 換碟的幾倍）", sens.to_markdown(index=False), "",
        "## 2. Cascade（每區抽 20 顆）", by_tier.to_markdown(index=False), "",
        "```json", json.dumps(cstats, indent=2, ensure_ascii=False), "```", "",
        *(["### v1（沒有成本感知升級）對照", "同一批 60 個案例；v1 只看動作信心，不看選錯的代價。",
            "| | v1 | v2 |", "|---|---|---|",
            *[f"| {k} | {v1[k]} | {cstats[k]} |" for k in
              ("escalated_to_llm", "decided_by_llm", "queued_for_human", "mean_cost_cascade", "judge_cost_usd_total")],
            ""] if v1 else []),
        "## 3. 報告忠實度", "```json", json.dumps(rstats, indent=2, ensure_ascii=False), "```", "",
        "人工評分表：`reports/agent_human_review.csv`（證據正確性、建議合理性，1–5 分）。",
    ]
    (REPORTS_DIR / "agent_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
