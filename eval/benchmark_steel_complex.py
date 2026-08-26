"""Steel advanced-plan and repair benchmark."""

from eval.steel_agent_challenge_v1 import SUITE as _CHALLENGE
from eval.steel_benchmark_v1 import SUITE as _CORE

SUITE = {
    "name": "steel_complex_agent",
    "version": "2.0.0",
    "profile": "steel_industry",
    "description": "钢铁窗口、相关性、累计占比和条件分析 Agent 挑战",
    "cases": [case for case in _CORE["cases"] if "llm_fallback" in case.get("tags", [])]
    + list(_CHALLENGE["cases"]),
}
