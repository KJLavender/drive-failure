from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TMP_DIR = DATA_DIR / "tmp"
PARQUET_DIR = DATA_DIR / "parquet"
DERIVED_DIR = DATA_DIR / "derived"
REPORTS_DIR = ROOT / "reports"
