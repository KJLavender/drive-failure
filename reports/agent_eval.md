# Agent 評估

成本模型（policies/costs.yaml）：換碟 1.0、非計畫故障 5.0、觀察 0.05（故障時省 50%）→ p < 2% 忽略、2%–38% 觀察、> 38% 換碟。

## 1. 決策策略的事後成本（所有快照硬碟，依權重還原）
| policy                          |   cost_per_1k_drives |   replacements_per_1k |   watched_per_1k | failures_replaced_in_advance   |   replace_precision |
|:--------------------------------|---------------------:|----------------------:|-----------------:|:-------------------------------|--------------------:|
| always_ignore                   |                 5.38 |                  0    |              0   | 0/291                          |             nan     |
| backblaze_rule (BB5 任一 > 0 → 換) |                80.45 |                 78.86 |              0   | 205/291                        |               0.079 |
| model + cost (換／觀察／忽略)          |                 4.18 |                  0.23 |              7.7 | 40/291                         |               0.656 |
| model, replace only (p ≥ 換碟分界)  |                 4.87 |                  0.23 |              0   | 40/291                         |               0.656 |
| oracle (壞的才換)                   |                 1.08 |                  1.08 |              0   | 291/291                        |               1     |

### 成本敏感度（非計畫故障成本 = 換碟的幾倍）
|   failure_cost |   watch_from_p |   replace_from_p |   always_ignore |   backblaze_rule (BB5 任一 > 0 → 換) |   model + cost (換／觀察／忽略) |   model, replace only (p ≥ 換碟分界) |   oracle (壞的才換) |
|---------------:|---------------:|-----------------:|----------------:|----------------------------------:|-------------------------:|---------------------------------:|----------------:|
|              3 |          0.033 |            0.633 |            3.23 |                             79.81 |                     2.68 |                             3.15 |            1.08 |
|              5 |          0.02  |            0.38  |            5.38 |                             80.45 |                     4.18 |                             4.87 |            1.08 |
|             10 |          0.01  |            0.19  |           10.76 |                             82.04 |                     7.46 |                             8.73 |            1.08 |
|             20 |          0.005 |            0.095 |           21.53 |                             85.22 |                    13.68 |                            15.38 |            1.08 |

## 2. Cascade（每區抽 20 顆）
| zone    | tier        |   cases |   failed |
|:--------|:------------|--------:|---------:|
| gray    | human_queue |       1 |        0 |
| gray    | jev         |      18 |        3 |
| gray    | llm         |       1 |        1 |
| replace | jev         |      18 |        9 |
| replace | llm         |       2 |        2 |
| watch   | jev         |      20 |        6 |

```json
{
  "cases": 60,
  "escalated_to_llm": 4,
  "decided_by_llm": 3,
  "queued_for_human": 1,
  "llm_agrees_with_model_action": 3,
  "mean_cost_model_only": 0.2431,
  "mean_cost_cascade": 0.2431,
  "gray_zone_cost_model_only": 0.1554,
  "gray_zone_cost_cascade": 0.1554,
  "judge_cost_usd_total": 0.0175,
  "judge_latency_s_mean_when_llm": 4.2
}
```

### v1（沒有成本感知升級）對照
同一批 60 個案例；v1 只看動作信心，不看選錯的代價。
| | v1 | v2 |
|---|---|---|
| escalated_to_llm | 20 | 4 |
| decided_by_llm | 5 | 3 |
| queued_for_human | 15 | 1 |
| mean_cost_cascade | 0.2312 | 0.2431 |
| judge_cost_usd_total | 0.2606 | 0.0175 |

## 3. 報告忠實度
```json
{
  "reports": 30,
  "mean_faithfulness_precision": 1.0,
  "reports_fully_faithful": 30,
  "reports_with_unsupported_numbers": 0,
  "recommendation_matches_decision": 30,
  "mean_latency_s": 9.5,
  "mean_cost_usd": 0.0227
}
```

人工評分表：`reports/agent_human_review.csv`（證據正確性、建議合理性，1–5 分）。
