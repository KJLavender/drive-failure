# drive-failure

[English](README.md) | **繁體中文**

硬碟會記錄自己的健康指標，叫做 SMART，例如有多少磁區被換掉、有多少讀取錯誤修不好、溫度多少。雲端儲存公司 Backblaze 每天公開整個機房所有硬碟的這些數字，也公開哪些硬碟壞了。

這個專案用這份資料，提前警告哪些硬碟可能在 30 天內壞掉。每發出一個警告，它會決定該現在換掉、先觀察，還是不用理。它也會寫一份簡短的報告說明原因，報告裡的每個數字都會回頭和資料核對。

線上 dashboard：https://kjlavender.github.io/drive-failure/

![dashboard](docs/dashboard.png)

## 找到了什麼

資料涵蓋 6 季（2025 年 1 月到 2026 年 6 月），共 372,456 顆硬碟、6,861 次故障。模型用 2025 年的資料訓練，在 2026 上半年測試，而且測試用的硬碟是訓練時沒看過的。

警告系統如果一直亂叫就沒人會理。所以下面每個結果都在固定的上限下量：每 1000 顆健康硬碟每月誤報不超過 2 次左右。誤報是指發了警告，但硬碟其實沒壞。

在這個上限下，模型能抓到大約一半的故障（50%，合理範圍在 46–55% 之間），平均在故障前 19 天發出警告。

| 做法 | 抓到的故障 | 每 1000 顆每月誤報 |
|---|---|---|
| Backblaze 自己的規則（5 個關鍵指標任一大於 0 就警告） | 77% | 78 |
| 簡單模型（邏輯斯迴歸） | 39% | 1.6 |
| 這個專案的模型（LightGBM，一種決策樹模型） | 50% | 1.4 |

Backblaze 的規則抓得比較多，但它會對 8% 的硬碟發警告，而這些硬碟大多還能撐好幾個月。

其他發現：

- 指標上升得多快，比它現在是多少更重要。最有用的訊號是 187（硬碟修不好的錯誤）和 197（等待被換掉的磁區）在過去 30 天增加了多少。最強的單一警訊是硬碟連 SMART 都讀不到：這種硬碟大約一半會在一個月內壞掉。
- 大約每 5 次故障就有 1 次沒有預兆。21% 的故障在壞掉前 30 天，沒有任何錯誤指標有變化。這種故障用 SMART 抓不到。
- 提早警告幾乎沒有代價。往前看 30 天和往前看 7 天，抓到的故障比例差不多（50% 和 47%），但警告提早了將近 3 倍。
- 模型要定期重新訓練。三季沒重訓的模型，判斷哪顆比較危險的能力還在，但警告門檻跑掉了：誤報比上限多了 26–38%。每季用所有歷史資料重訓就能修正。
- 用一個廠牌訓練的模型，不能直接用在另一個廠牌。只用 WDC/HGST 硬碟訓練的模型，拿去用在 Seagate 上，每 1000 顆每月誤報 424 次。把所有廠牌放在一起訓練，比每個廠牌各自訓練好。
- 有一部分標籤可能是循環的。Backblaze 有時會在硬碟還沒壞之前就換掉它，依據正是它規則裡那 5 個指標，而這些硬碟也會被標成故障。拿掉這 5 個指標後，模型只抓到 25%，而不是 50%。所以它們確實重要，但這份資料沒辦法完全分清楚「它們能預測故障」和「故障標籤是因它們而來」。
- Ceph 內建的預測器在這裡表現較差。Ceph 是開源的儲存軟體，有自己的故障預測器。在相同誤報率下，它抓到 34% 的 14 天內故障，這個模型抓到 76%。不過 Ceph 的模型是多年前用舊硬碟訓練的，所以這主要顯示的是模型從不重訓會怎樣。

完整報告：[模型比較](reports/week2.md)、[重訓、跨廠牌、漏掉的故障](reports/week3.md)、[Ceph、硬碟壽命、資料少的廠牌](reports/week3_extra.md)、[agent](reports/agent_eval.md)。

## 它怎麼決定要做什麼

決策這一步沿用我 [ops-judge](https://github.com/KJLavender/ops-judge) 專案裡的 judge-core。簡單的案例用便宜的步驟解決，只有看不清楚的才往後送到比較貴的步驟：

```
整理事實 → 風險模型 + 成本規則 → LLM → 人工審核佇列
```

1. 整理事實。程式先把事實算好：哪些指標在上升、從哪天開始、哪些因素讓風險變高最多、最相似的 20 顆歷史硬碟後來怎麼樣。LLM 不直接讀原始數字。它拿到的是這些事實，以及由事實組成的短句，例如「SMART 197 目前 1936，過去 7 天增加 1936，首次出現於 2026-04-27」。
2. 風險和成本。模型給出這顆硬碟 30 天內壞掉的機率，再用一條簡單的成本規則把機率換成動作。預設是一次意外故障的成本，等於提早換一顆硬碟的 5 倍。照這個數字：
   - 風險低於 2%：不用理
   - 風險在 2% 到 38%：先觀察
   - 風險高於 38%：現在換

   成本可以在 `policies/costs.yaml` 修改。
3. 只在重要的時候問。如果風險剛好落在分界附近，模型就不確定。但如果兩個動作的成本幾乎一樣，選錯也沒什麼差別，模型就直接決定。沒有這條規則時，60 個測試案例裡有 15 個送到人工。它們幾乎都是「不用理」和「先觀察」之間的選擇，選錯的代價大約只有換一顆硬碟的 1%。加上這條規則後，60 個裡只剩 1 個送到人工。
4. 寫報告並檢查。LLM 根據事實寫報告。報告裡每一點都要註明出自哪個事實，再由檢查程式核對數值，並標出任何在事實裡找不到的數字。30 份報告中，所有引用的數值都核對正確，也沒有找不到出處的數字。每份報告約 10 秒、約 $0.02。不過檢查程式只核對數字，推理合不合理還是要人來看，評分表在 `reports/agent_human_review.csv`。

LLM 透過本機的 Claude Code CLI（`claude -p`）執行，走你的 Claude 訂閱，不需要 API key。預設模型是 Sonnet，可以用環境變數 `DRIVEFAIL_LLM_MODEL` 切換。

在預設成本下，照這些決定做，總成本比什麼都不做少 22%。照 Backblaze 的規則做，成本是什麼都不做的 15 倍。不論一次故障的成本是換碟的 3 倍還是 20 倍，這個結論都成立。

## 執行

需要 [uv](https://docs.astral.sh/uv/) 和約 25 GB 的磁碟空間。agent 那一步還需要登入的 [Claude Code](https://claude.com/claude-code) CLI。

```bash
uv sync
bash scripts/download.sh Q1_2025 Q2_2025 Q3_2025 Q4_2025 Q1_2026 Q2_2026   # 約 8 GB
uv run python -m drivefail.ingest      # 解壓並轉成 Parquet，一次處理一季
uv run python -m drivefail.drives      # 每顆硬碟一列，並檢查標籤
uv run python -m drivefail.stats       # 各型號的年化故障率
uv run python -m drivefail.features    # 7 天和 30 天的變化量、趨勢

uv run python -m drivefail.train       # 比較規則、邏輯斯迴歸和 LightGBM
uv run python -m drivefail.week3       # 重訓、其他廠牌、漏掉的故障
uv run python -m drivefail.survival    # 硬碟依年齡的壽命
uv run python -m drivefail.transfer    # 資料很少的新廠牌

uv run python -m drivefail.risk        # 最終模型和每日風險快照
uv run python -m drivefail.agent --dry-run   # 不呼叫 LLM，先試跑決策流程
uv run python -m drivefail.agent       # 60 個決策和 30 份報告，約 $1
uv run python -m drivefail.agent_eval
uv run python -m drivefail.dashboard   # 產生 docs/index.html
uv run pytest
```

Ceph 的比較需要舊版的 scikit-learn。先跑 `uv run python -m drivefail.ceph_baseline extract`，再跑 `scripts/ceph_score.py` 開頭寫的指令，最後跑 `uv run python -m drivefail.ceph_baseline compare`。

## 限制

- 只有傳統硬碟。資料裡也有 3,516 顆 SSD，但 18 個月內一顆都沒壞，沒東西可以學。
- agent 只測了 60 個決策和 30 份報告，報告還沒有經過人工評分。
- 成本的數字是假設的，請依自己的情況修改。
- 每季測試只有 190 到 310 次故障，所以季與季之間差幾個百分點可能只是雜訊。

## 目錄

```
src/drivefail/
  ingest.py db.py drives.py stats.py   載入資料（DuckDB 查 Parquet）
  features.py                           每顆硬碟的變化量和趨勢
  evaluate.py                           誤報、抓到的故障、提早天數
  train.py week3.py transfer.py         實驗
  survival.py ceph_baseline.py          硬碟壽命、Ceph 比較
  risk.py                               最終模型、機率校準、規則
  evidence.py                           給 LLM 的事實和句子
  agent.py agent_eval.py                決策、報告、報告檢查
  dashboard.py dashboard_template.html  dashboard 頁面
policies/                               決策設定和成本
reports/                                結果和圖表
scripts/                                下載資料、Ceph 評分
tests/
```
