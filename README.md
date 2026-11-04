# drive-failure

**English** | [繁體中文](README.zh-TW.md)

Predict which hard drives are about to fail from Backblaze's public SMART data, then let an agent decide what to do about each one (replace, watch, ignore) and write a report whose every number can be traced back to the data.

Most public Backblaze projects stop at "model X gets AUC Y". This one focuses on the parts that decide whether it would work in operations: a time-ordered split with held-out drives, recall at a fixed false-alarm budget, what happens when the model ages or meets a new vendor, and what to do after the prediction.

![dashboard](docs/dashboard.png)

## Results in one table

Six quarters of Backblaze Drive Stats (2025 Q1 – 2026 Q2): 372,456 drives, 180 M drive-days, 6,861 failures. Test set: 2026 H1, drives never seen in training (503 failures).

| Question | Answer |
|---|---|
| How many failures can we catch at an acceptable false-alarm rate? | **50% (95% CI 46–55%)** within 30 days, at 1.4 false alarms per 1,000 healthy drives per month. Backblaze's own rule (any of SMART 5/187/188/197/198 non-zero) catches 77% but raises 78 false alarms per 1,000 drives per month, 40× the budget. |
| How early? | Average 18.8 days before failure. Going from a 7-day to a 30-day horizon barely changes recall (47% → 50%) but triples the lead time. |
| Which SMART attributes matter? | 30-day *increases* of 187 and 197 rank above their current values. A drive that cannot report SMART at all is the strongest single signal (48% fail within 30 days). |
| Is the model just relearning Backblaze's replacement rule? | Partly unanswerable. Backblaze's failure labels include proactive replacements triggered by the same five attributes. Removing them halves recall (50% → 25%), so they carry real signal, but the experiment cannot separate "predictive" from "label-generating". |
| Does a model go stale? | The ranking holds up; the **threshold** does not. A model frozen for three quarters overshoots its false-alarm budget by 26–38% every quarter. Retraining on all history each quarter keeps it on budget. |
| Does it transfer to another vendor? | No. Ranking quality drops 2–9×, and borrowed thresholds are unusable (a WDC/HGST model on Seagate drives raises 424 false alarms per 1,000 drives per month). Training on all vendors together beats per-vendor models. |
| What can't SMART see? | 21% of failures show no change in any error counter in the 30 days before failing (the industry rule of thumb is ~23%). |
| Compared with Ceph's built-in predictor? | At the same false-alarm rate, 76% vs 34% recall (14-day horizon). Ceph's models were trained years ago on older drives, so this mostly shows the cost of never retraining. |
| Does acting on it save money? | Under the default cost model, model + cost-based actions cut realized cost by 22% versus doing nothing; Backblaze's rule costs 15× more than doing nothing. The conclusion holds for failure costs from 3× to 20× a replacement. |

Details: [Week 2](reports/week2.md) · [Week 3](reports/week3.md) · [Ceph / survival / transfer](reports/week3_extra.md) · [Agent](reports/agent_eval.md)

## How the agent decides

The decision layer reuses [judge-core](https://github.com/KJLavender/ops-judge) from ops-judge: cheap tiers first, and only unclear cases go further.

```
evidence layer → risk model + cost model → LLM (claude -p) → human queue (SQLite)
```

1. **Evidence layer** (`evidence.py`). The LLM never reads raw SMART numbers. Code computes the facts first: which counters are rising and since when, SHAP contributions, matched rules, the model's failure rate, and how the 20 most similar historical cases ended. It then writes them out as fixed Chinese trend sentences ("SMART 197 is 1936, up 1936 in 7 days, first seen 2026-04-27"). Every fact has a stable key such as `smart.197.change_7d`.
2. **Risk model + cost model** (`agent.py`, `policies/costs.yaml`). A calibrated 30-day failure probability *p* gives the expected cost of each action. With the default costs (an unplanned failure costs 5× a replacement), the cheapest action is ignore below 2%, watch from 2% to 38%, and replace above 38%. Confidence comes from how likely the action would change if the risk estimate were off by ±0.5 in log-odds.
3. **Cost-aware escalation.** If the best and second-best actions differ by less than 0.02 in expected cost, picking the "wrong" one barely matters, so the model decides. Without this rule, 15 of 60 sampled cases went to a human, almost all of them "ignore vs. watch" at *p* ≈ 2% where the stakes were about 1% of a replacement. With it, 1 of 60 went to a human. Realized cost rose by 0.012 per case, inside the 0.02 tolerance, and LLM spend fell 15×.
4. **Report + faithfulness check.** The LLM writes the report from the evidence. Each finding must cite an evidence key, and a checker verifies the value and flags any number in the text that does not appear in the evidence. Across 30 reports: 100% of findings verified, 0 unsupported numbers, 9.5 s and $0.023 per report (Sonnet via the local Claude Code CLI). The checker only covers cited numbers, so qualitative claims still need a human: `reports/agent_human_review.csv` is the scoring sheet.

## Run it

Requires [uv](https://docs.astral.sh/uv/), about 25 GB of free disk space (8 GB of zips, 6 GB of Parquet, and one quarter of extracted CSV at a time), and for the agent, the [Claude Code](https://claude.com/claude-code) CLI signed in (it uses your subscription, no API key).

```bash
uv sync
bash scripts/download.sh Q1_2025 Q2_2025 Q3_2025 Q4_2025 Q1_2026 Q2_2026   # ~8 GB
uv run python -m drivefail.ingest          # zip → per-day Parquet (11 GB CSV per quarter → ~800 MB)
uv run python -m drivefail.drives          # one row per drive + label quality checks
uv run python -m drivefail.stats           # AFR by model and quarter
uv run python -m drivefail.features        # windowed features, drive-level sampling

uv run python -m drivefail.train           # Week 2: rule vs logistic regression vs LightGBM vs ablation
uv run python -m drivefail.week3           # drift, cross-vendor, error analysis, CIs
uv run python -m drivefail.survival        # Kaplan–Meier with left truncation, Cox
uv run python -m drivefail.transfer        # small-data vendor transfer

uv run python -m drivefail.risk            # deployment model, calibration, rule mining, snapshots
uv run python -m drivefail.agent --dry-run # cascade without calling the LLM
uv run python -m drivefail.agent           # 60 cases + 30 reports (~$1)
uv run python -m drivefail.agent_eval
uv run python -m drivefail.dashboard       # → site/index.html
uv run pytest
```

The Ceph comparison needs an old scikit-learn: `uv run python -m drivefail.ceph_baseline extract`, then the `uv run --python 3.8 ...` command at the top of `scripts/ceph_score.py`, then `... ceph_baseline compare`. Set `DRIVEFAIL_LLM_MODEL` to change the model (default `claude-sonnet-5-5`).

## Limits

- **HDD only.** The 3,516 SSDs in the data had zero failures in 18 months (1.8 M drive-days), so there is nothing to train on. Published SSD work (Alibaba, Tencent) relies on wear indicators that this data does not have.
- **Label circularity is measured, not solved.** See the ablation above.
- **Small agent evaluation.** 60 decisions and 30 reports; the human scoring sheet has not been filled in yet. The cost model's numbers are assumptions; `policies/costs.yaml` is meant to be edited.
- **Per-quarter comparisons are noisy.** Only 190–310 failures per test quarter (±6–7 points on recall). Most differences between retraining strategies fall inside that range.

## Layout

```
src/drivefail/
  ingest.py db.py drives.py stats.py   data pipeline (DuckDB over Parquet)
  features.py                           windowed features, drive-level sampling
  evaluate.py                           false-alarm budget, recall, lead time, bootstrap CIs
  train.py week3.py transfer.py         experiments
  survival.py ceph_baseline.py          survival analysis, Ceph comparison
  risk.py                               deployment model, calibration, rule mining
  evidence.py                           evidence layer (facts + trend sentences)
  agent.py agent_eval.py                judge-core cascade, reports, faithfulness, evaluation
  dashboard.py dashboard_template.html  static dashboard
policies/                               judge-core policy and cost model
reports/                                results (Markdown, CSV, figures)
scripts/                                download, Ceph scoring (old sklearn env)
tests/
```
