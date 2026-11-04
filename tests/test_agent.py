from judge_core import JudgeRequest, Question

from drivefail.agent import (RiskModelJudge, boundaries, expected_costs, faithfulness, flatten,
                             realized_cost)

COSTS = {"replace_cost": 1.0, "failure_cost": 5.0, "watch_cost": 0.05, "watch_mitigation": 0.5,
         "risk_logit_sigma": 0.5, "min_stakes": 0.0}


def test_boundaries_are_break_even_points():
    lo, hi = boundaries(COSTS)
    assert abs(lo - 0.02) < 1e-9 and abs(hi - 0.38) < 1e-9
    for p, best in [(0.01, "ignore"), (0.1, "watch"), (0.6, "replace")]:
        ec = expected_costs(p, COSTS)
        assert min(ec, key=ec.get) == best
    assert realized_cost("ignore", True, COSTS) == 5.0 and realized_cost("replace", True, COSTS) == 1.0


def _decide(p):
    q = Question(key="drive_action", kind="choice", instructions="x",
                 criteria={"replace": None, "watch": None, "ignore": None})
    return RiskModelJudge(COSTS).judge(JudgeRequest(state={"risk": {"p_fail_30d": p}}, questions=[q])).answers["drive_action"]


def test_risk_judge_is_unsure_near_boundary_and_sure_far_from_it():
    far, near = _decide(0.9), _decide(0.38)
    assert far.value == "replace" and far.confidence > 0.95
    assert near.confidence < 0.6                       # 正好在 watch/replace 分界
    assert _decide(0.001).value == "ignore" and _decide(0.1).value == "watch"


def test_faithfulness_catches_wrong_values_and_invented_numbers():
    facts = flatten({"risk": {"p_fail_30d": 0.4829}, "smart": {"197": {"change_7d": 8.0}}})
    facts["decision"] = "replace"
    report = {
        "headline": "故障機率 48.3%，197 七天增加 8",
        "recommendation": "replace",
        "findings": [
            {"claim": "30 天內故障機率 48.3%", "evidence_key": "risk.p_fail_30d", "value": "48.3%"},
            {"claim": "待處理磁區七天增加 8", "evidence_key": "smart.197.change_7d", "value": 8},
            {"claim": "溫度 55 度", "evidence_key": "temperature.current_c", "value": 55},   # 不存在的 key
            {"claim": "197 增加 12", "evidence_key": "smart.197.change_7d", "value": 12},    # 值不對
        ],
        "risk_explanation": "同型號年化故障率 4.2%",                                       # 捏造的數字
        "next_steps": [],
    }
    fa = faithfulness(report, facts, trend=[], decision="replace")
    assert fa["verified"] == 2 and fa["faithfulness_precision"] == 0.5
    assert "4.2" in fa["unsupported_numbers"] and "12" in fa["unsupported_numbers"]
    assert fa["recommendation_matches_decision"]


def test_low_stakes_cases_are_decided_without_escalation():
    q = Question(key="drive_action", kind="choice", instructions="x",
                 criteria={"replace": None, "watch": None, "ignore": None})
    judge = RiskModelJudge({**COSTS, "min_stakes": 0.02})
    near_lo = judge.judge(JudgeRequest(state={"risk": {"p_fail_30d": 0.018}}, questions=[q])).answers["drive_action"]
    assert near_lo.value == "ignore" and near_lo.confidence == 1.0     # 忽略 vs 觀察只差約 0.005
    near_hi = judge.judge(JudgeRequest(state={"risk": {"p_fail_30d": 0.36}}, questions=[q])).answers["drive_action"]
    assert near_hi.confidence < 0.8                                      # 觀察 vs 換碟差 0.05，仍要升級


def test_negative_numbers_are_matched_with_sign():
    facts = {"top_factors.0.contribution": -1.492, "decision": "replace"}
    report = {"headline": "x", "recommendation": "replace", "next_steps": [],
              "findings": [{"claim": "貢獻為負（-1.492）", "evidence_key": "top_factors.0.contribution", "value": -1.492},
                           {"claim": "y", "evidence_key": "decision", "value": "replace"}],
              "risk_explanation": "7 天增加量貢獻 -1.492"}
    assert faithfulness(report, facts, trend=[], decision="replace")["unsupported_numbers"] == []
