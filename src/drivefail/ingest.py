"""Backblaze 每季 zip → data/parquet/<季>/<日期>.parquet。

一次只解壓一季（約 11 GB CSV），轉完立刻刪掉 CSV。
SMART 欄位一律轉成 DOUBLE，避免某天全空時被推斷成 VARCHAR。
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import duckdb

from drivefail.paths import PARQUET_DIR, RAW_DIR, TMP_DIR

META_COLS = {
    "date": "DATE",
    "serial_number": "VARCHAR",
    "model": "VARCHAR",
    "capacity_bytes": "BIGINT",
    "failure": "TINYINT",
    "datacenter": "VARCHAR",
    "cluster_id": "VARCHAR",
    "vault_id": "VARCHAR",
    "pod_id": "VARCHAR",
    "pod_slot_num": "VARCHAR",
    "is_legacy_format": "VARCHAR",
}


def _select_list(columns: list[str]) -> str:
    parts = []
    for c in columns:
        if c in META_COLS:
            parts.append(f'TRY_CAST("{c}" AS {META_COLS[c]}) AS "{c}"')
        elif c.startswith("smart_"):
            parts.append(f'TRY_CAST("{c}" AS DOUBLE) AS "{c}"')
    return ", ".join(parts)


def convert_quarter(quarter: str, keep_csv: bool = False) -> Path:
    zip_path = RAW_DIR / f"data_{quarter}.zip"
    out = PARQUET_DIR / quarter
    if (out / "_DONE").exists():
        print(f"[skip] {quarter} 已轉好")
        return out

    extract_dir = TMP_DIR / quarter
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv") and "__MACOSX" not in m]
        todo = [m for m in members if not (extract_dir / m).exists()]
        zf.extractall(extract_dir, todo)

    # 逐日轉檔：每天欄位集合可能不同，一次 union 整季會吃光記憶體
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute(f"SET temp_directory='{(TMP_DIR / 'duckdb_spill').as_posix()}'")
    csvs = sorted(extract_dir.rglob("*.csv"))
    for i, csv in enumerate(csvs, 1):
        day_out = out / f"{csv.stem}.parquet"
        if day_out.exists():
            continue
        # 全部當字串讀，再自己轉型：全空的 SMART 欄會被推斷成 VARCHAR
        src = f"read_csv('{csv.as_posix()}', all_varchar=true, header=true)"
        columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
        tmp_out = day_out.with_suffix(".tmp")
        con.execute(
            f"COPY (SELECT {_select_list(columns)} FROM {src}) "
            f"TO '{tmp_out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
        )
        tmp_out.replace(day_out)
        if i % 10 == 0 or i == len(csvs):
            print(f"[parquet] {quarter} {i}/{len(csvs)}")
    con.close()
    (out / "_DONE").touch()

    if not keep_csv:
        shutil.rmtree(extract_dir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("quarters", nargs="*", help="例如 Q1_2025；不給則轉 data/raw 下所有 zip")
    ap.add_argument("--keep-csv", action="store_true")
    args = ap.parse_args()
    quarters = args.quarters or sorted(
        p.stem.removeprefix("data_") for p in RAW_DIR.glob("data_*.zip")
    )
    for q in quarters:
        convert_quarter(q, keep_csv=args.keep_csv)


if __name__ == "__main__":
    main()
