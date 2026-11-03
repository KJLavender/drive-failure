"""用 Ceph diskprediction_local（Red Hat 模型）替測試資料打分。

必須在舊版 sklearn 環境執行（模型是 0.19 時代的 pickle）：
    uv run --no-project --python 3.8 --with scikit-learn==0.23.2 --with numpy==1.19.5 \
        --with scipy==1.5.4 --with pandas==1.3.5 --with pyarrow==8.0.0 \
        python scripts/ceph_score.py

特徵轉換照抄 ceph/src/pybind/mgr/diskprediction_local/predictor.py（RHDiskFailurePredictor）：
最近 6 天的 mean、std(ddof=1)、cv，再接 user_capacity，經 RobustScaler 後丟進 RandomForest。
輸出每顆硬碟每天的類別：0 Good（>6 週）、1 Warning（2–6 週）、2 Bad（<2 週）、-1 Unknown。
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
CEPH = ROOT / "data" / "ceph"
WINDOW = 6


def manufacturer(model: str):
    # 與 Ceph 的 MANUFACTURER_MODELNAME_PREFIXES 相同的判斷；只有 seagate／hgst 有模型
    if model.startswith("HGST"):
        return "hgst"
    if model.startswith("S") or model.startswith("ZA"):
        return "seagate"
    return None


def featurize(g: pd.DataFrame, attrs: list) -> np.ndarray:
    x = g[[a for a in attrs if a.startswith("smart_")]].to_numpy(np.float64)
    r = pd.DataFrame(x).rolling(WINDOW)
    means, stds = r.mean().to_numpy(), r.std(ddof=1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cvs = stds / means
    cvs[np.isnan(cvs)] = 0
    # 與 Ceph 原碼的唯一差異：平均為 0 但標準差非 0（有負值）時 cv = inf，原碼會直接報錯
    cvs[np.isinf(cvs)] = 0
    return np.hstack([means, stds, cvs, g[["user_capacity"]].to_numpy(np.float64)])


def main() -> None:
    config = json.loads((CEPH / "config.json").read_text())
    models = {m: (pickle.load(open(CEPH / f"{m}_scaler.pkl", "rb")),
                  pickle.load(open(CEPH / f"{m}_predictor.pkl", "rb"))) for m in config}
    df = pd.read_parquet(ROOT / "data" / "derived" / "ceph_input.parquet")
    df = df.sort_values(["serial_number", "date"])
    df["mfr"] = df["model"].map(manufacturer)
    out = []
    for mfr, part in df.groupby("mfr"):
        attrs = config[mfr]
        scaler, clf = models[mfr]
        for serial, g in part.groupby("serial_number", sort=False):
            g = g.fillna(0)  # Ceph 假設欄位都有值；缺值補 0
            feats = featurize(g, attrs)
            pred = np.full(len(g), -1)
            ok = ~np.isnan(feats).any(axis=1)  # 前 5 天湊不滿 6 天
            if ok.any():
                pred[ok] = clf.predict(scaler.transform(feats[ok]))
            out.append(pd.DataFrame({"serial_number": serial, "date": g["date"].to_numpy(), "ceph_class": pred}))
        print(mfr, "done")
    pd.concat(out).to_parquet(ROOT / "data" / "derived" / "ceph_pred.parquet", index=False)


if __name__ == "__main__":
    main()
