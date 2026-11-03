"""存活分析：硬碟「還能活多久」。

關鍵：硬碟進入觀測時已經用了幾年（左截斷）。若把年齡當作從 0 開始觀測，
會高估存活率——早夭的硬碟根本沒機會進入資料集。用 lifelines 的 entry 參數修正。
- 年齡 = 通電時數（SMART 9）換算成年
- 事件 = 故障；消失但沒標故障 = 在最後一筆右設限
- 排除故障後又復活的序號、序號碰撞、SSD
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from drivefail.paths import DERIVED_DIR, REPORTS_DIR
from drivefail.week3 import vendor

# dataviz 參考調色盤：分類色依固定順序指派（已跑 validate_palette.js）
VENDOR_COLOR = {"Seagate": "#2a78d6", "Toshiba": "#eb6834", "WDC/HGST": "#1baf7a"}
INK, INK_MUTED, GRID, SURFACE = "#1a1a19", "#6b6a63", "#e6e5df", "#fcfcfb"
MIN_DRIVE_YEARS = 2000  # 曝險不足的年齡區間不畫


def load() -> pd.DataFrame:
    d = pd.read_parquet(DERIVED_DIR / "drives.parquet")
    d = d[(d.media == "HDD") & (d.n_models == 1) & (d.days_after_fail == 0) & (d.poh_first > 0)].copy()
    d["vendor"] = vendor(d["model"])
    d = d[d.vendor != "other"]
    d["age_entry"] = d.poh_first / 24 / 365.25
    span = (pd.to_datetime(d.last_date) - pd.to_datetime(d.first_date)).dt.days
    d["age_exit"] = d.age_entry + (span + 1) / 365.25
    d["event"] = d.failed.astype(int)
    d["capacity_tb"] = d.capacity_bytes / 1e12
    return d


def km_by_vendor(d: pd.DataFrame, at=(3, 5, 7)) -> tuple[pd.DataFrame, dict]:
    rows, fits = [], {}
    for v, g in d.groupby("vendor"):
        km = KaplanMeierFitter().fit(g.age_exit, g.event, entry=g.age_entry, label=v)
        fits[v] = km
        row = {"vendor": v, "drives": len(g), "failures": int(g.event.sum())}
        for t in at:
            row[f"S({t}y)"] = round(float(km.survival_function_at_times(t).iloc[0]), 4)
        rows.append(row)
    return pd.DataFrame(rows), fits


def afr_by_age(d: pd.DataFrame, max_age: int = 10) -> pd.DataFrame:
    """各年齡區間的年化故障率：該區間故障數 ÷ 落在該區間的 drive-years。"""
    rows = []
    for v, g in d.groupby("vendor"):
        entry, exit_, ev = g.age_entry.to_numpy(), g.age_exit.to_numpy(), g.event.to_numpy()
        for a in range(max_age):
            exposure = np.clip(np.minimum(exit_, a + 1) - np.maximum(entry, a), 0, None).sum()
            fails = int(((exit_ > a) & (exit_ <= a + 1) & (ev == 1)).sum())
            rows.append({"vendor": v, "age": a, "drive_years": round(exposure), "failures": fails,
                         "afr_pct": round(fails / exposure * 100, 2) if exposure else np.nan})
    return pd.DataFrame(rows)


def cox(d: pd.DataFrame) -> pd.DataFrame:
    x = pd.get_dummies(d[["vendor"]], drop_first=False, dtype=float).drop(columns="vendor_Seagate")
    x["capacity_tb"] = d.capacity_tb
    x["age_exit"], x["event"], x["age_entry"] = d.age_exit, d.event, d.age_entry
    cph = CoxPHFitter(penalizer=0.01).fit(x, "age_exit", "event", entry_col="age_entry")
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    return s.rename(columns={"exp(coef)": "hazard_ratio"}).round(4)


def _style(ax, title, ylabel, xlabel="硬碟年齡（年）"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def plot(fits: dict, age: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    out = REPORTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    for v, km in fits.items():
        sf = km.survival_function_
        ax.step(sf.index, sf.iloc[:, 0], where="post", color=VENDOR_COLOR[v], linewidth=2, label=v)
        # 右端三條線收斂會疊字；第 3 年三條線分得最開
        ax.annotate(v, (3.0, float(km.survival_function_at_times(3.0).iloc[0])),
                    xytext=(0, -8 if v == "Seagate" else 6), textcoords="offset points", color=INK,
                    fontsize=9, ha="center", va="top" if v == "Seagate" else "bottom")  # Seagate 最低，標在下方
    ax.set_xlim(0, 10.5)
    _style(ax, "Kaplan–Meier 存活曲線（左截斷修正）", "存活機率")
    ax.legend(frameon=False, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "km_by_vendor.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    a = age[age.drive_years >= MIN_DRIVE_YEARS]
    for v, g in a.groupby("vendor"):
        x = g.age + 0.5
        ax.plot(x, g.afr_pct, color=VENDOR_COLOR[v], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=2, label=v)
        ax.annotate(v, (x.iloc[-1], g.afr_pct.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")
    ax.set_ylim(bottom=0)
    _style(ax, f"各年齡的年化故障率（曝險至少 {MIN_DRIVE_YEARS:,} drive-years）", "AFR（%）")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "afr_by_age.png", dpi=150)


def main() -> None:
    d = load()
    km_tab, fits = km_by_vendor(d)
    age = afr_by_age(d)
    hr = cox(d)
    plot(fits, age)
    # 對照：忽略左截斷（假設從 0 歲開始觀測）會高估多少
    naive = {v: round(float(KaplanMeierFitter().fit(g.age_exit, g.event).survival_function_at_times(5).iloc[0]), 4)
             for v, g in d.groupby("vendor")}
    km_tab["S(5y) 忽略左截斷"] = km_tab.vendor.map(naive)
    age.to_csv(REPORTS_DIR / "afr_by_age.csv", index=False)
    md = ["# 存活分析", "", "## Kaplan–Meier（左截斷修正）", km_tab.to_markdown(index=False), "",
          "## 各年齡年化故障率", age.pivot(index="age", columns="vendor", values="afr_pct").to_markdown(), "",
          "## Cox 比例風險（基準：Seagate）", hr.to_markdown(), ""]
    (REPORTS_DIR / "survival.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
