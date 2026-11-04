#!/usr/bin/env bash
# 下載 Backblaze Drive Stats 每季資料到 data/raw/（每季約 1–2 GB zip）
# 用法：bash scripts/download.sh Q1_2025 Q2_2025 Q3_2025 Q4_2025 Q1_2026 Q2_2026
set -euo pipefail
mkdir -p data/raw
for q in "$@"; do
  curl -sS -C - -o "data/raw/data_${q}.zip" "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_${q}.zip"
  echo "done $q"
done
