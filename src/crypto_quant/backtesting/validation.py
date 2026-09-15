from __future__ import annotations

from typing import Any, Dict


RISK_GATES = {
    "minimum_out_of_sample_sharpe": 0.30,
    "maximum_out_of_sample_drawdown": -0.60,
    "maximum_p95_participation": 0.01,
    "minimum_trade_count": 20,
}


def risk_gate(metrics: Dict[str, Any]) -> Dict[str, Any]:
    checks = {
        "out_of_sample_sharpe": metrics["sharpe_ratio"]
        >= RISK_GATES["minimum_out_of_sample_sharpe"],
        "drawdown_control": metrics["max_drawdown"]
        >= RISK_GATES["maximum_out_of_sample_drawdown"],
        "liquidity": metrics["p95_trade_participation"]
        <= RISK_GATES["maximum_p95_participation"],
        "enough_trade_evidence": metrics["trade_count"]
        >= RISK_GATES["minimum_trade_count"],
        "positive_total_return": metrics["total_return"] > 0.0,
    }
    return {
        "status": "candidate" if all(checks.values()) else "rejected",
        "checks": checks,
        "gates": RISK_GATES,
    }


def cross_symbol_risk_status(
    results: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, str]:
    """Approve a hypothesis only when every evaluated symbol clears the gates."""
    strategy_names = sorted(
        {name for by_strategy in results.values() for name in by_strategy}
    )
    return {
        name: (
            "candidate"
            if all(
                by_strategy.get(name, {}).get("risk", {}).get("status") == "candidate"
                for by_strategy in results.values()
            )
            else "rejected"
        )
        for name in strategy_names
    }
