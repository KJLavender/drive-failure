"""實務導向的評估：固定誤報預算下的故障偵測率。

所有指標都在「硬碟」層級計算，不是「天」：
- 誤報率：從未故障的硬碟中，每 1000 顆每月有幾顆被警告（依抽樣權重還原）
- 偵測率：故障硬碟中，在故障前 1..N 天內至少被警告一次的比例
- 提前天數：被偵測到的故障，最早一次警告距離故障幾天（上限 N）
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def label(df: pd.DataFrame, horizon: int) -> pd.Series:
    """1 = 故障前 1..N 天；0 = 確定健康；NaN = 不確定（丟掉）。

    故障當天（days_to_fail = 0）不算正樣本：那天才警告已經來不及。
    健康要確定：未來 N 天內資料沒結束、硬碟也沒消失。
    """
    dtf = df["days_to_fail"]
    pos = dtf.between(1, horizon)
    neg = (dtf.isna() | (dtf > horizon)) & (df["days_to_end"] >= horizon) & (df["days_to_last"] >= horizon)
    y = pd.Series(np.nan, index=df.index)
    y[neg] = 0.0
    y[pos] = 1.0
    return y


def false_alarm_rate(df: pd.DataFrame, alarm: np.ndarray) -> float:
    """每 1000 顆健康硬碟、每月被警告的顆數。"""
    healthy = ~df["failed"].to_numpy()
    h = df.loc[healthy, ["serial_number", "date", "weight"]].copy()
    h["alarm"] = alarm[healthy]
    h["month"] = pd.to_datetime(h["date"]).dt.to_period("M")
    dm = h.groupby(["serial_number", "month"]).agg(alarm=("alarm", "max"), weight=("weight", "first"))
    return float((dm["alarm"] * dm["weight"]).sum() / dm["weight"].sum() * 1000)


def detection(df: pd.DataFrame, alarm: np.ndarray, horizon: int) -> tuple[float, float, int]:
    """回傳 (偵測率, 平均提前天數, 故障硬碟數)。只看故障日落在這段資料內的硬碟。"""
    failed_in_window = df.loc[df["days_to_fail"] == 0, "serial_number"].unique()
    in_win = df["days_to_fail"].between(1, horizon).to_numpy() & alarm.astype(bool)
    hit = df.loc[in_win].groupby("serial_number")["days_to_fail"].max()
    hit = hit[hit.index.isin(failed_in_window)]
    n = len(failed_in_window)
    if n == 0:
        return float("nan"), float("nan"), 0
    return len(hit) / n, float(hit.mean()) if len(hit) else float("nan"), n


def threshold_for_budget(df: pd.DataFrame, score: np.ndarray, budget: float) -> float:
    """在驗證集上找最低門檻，使誤報率 ≤ budget（每 1000 顆每月）。

    一顆硬碟某月被警告 ⇔ 當月最高分 ≥ 門檻，所以先算每個「硬碟-月」的最高分，
    再由高到低累加權重，找到剛好不超過預算的分數。
    """
    healthy = ~df["failed"].to_numpy()
    h = df.loc[healthy, ["serial_number", "date", "weight"]].copy()
    h["score"] = score[healthy]
    h["month"] = pd.to_datetime(h["date"]).dt.to_period("M")
    dm = h.groupby(["serial_number", "month"]).agg(score=("score", "max"), weight=("weight", "first"))
    s = dm["score"].to_numpy()
    w = dm["weight"].to_numpy()
    order = np.argsort(-s)
    s, w = s[order], w[order]
    fa = np.cumsum(w) / w.sum() * 1000          # 門檻 = s[i] 時的誤報率（含同分）
    # 同分的硬碟-月一起被警告，只能在分數變化處切
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    ok = np.flatnonzero(last_of_tie & (fa <= budget))
    if len(ok) == 0:
        return float(np.nextafter(s[0], np.inf))  # 連最高分都超標 → 不發警告
    return float(s[ok[-1]])


def pr_auc(df: pd.DataFrame, score: np.ndarray, horizon: int) -> float:
    y = label(df, horizon)
    m = y.notna().to_numpy()
    return float(average_precision_score(y[m], score[m], sample_weight=df["weight"].to_numpy()[m]))


def operating_point(df: pd.DataFrame, score: np.ndarray, thr: float, horizon: int) -> dict:
    alarm = score >= thr
    rec, lead, n = detection(df, alarm, horizon)
    return {
        "threshold": thr,
        "false_alarms_per_1k_month": round(false_alarm_rate(df, alarm), 3),
        "recall": round(rec, 4),
        "mean_lead_days": round(lead, 1),
        "n_failures": n,
    }
