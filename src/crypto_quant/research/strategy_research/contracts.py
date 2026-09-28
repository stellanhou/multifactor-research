"""Explicit operator choices and strict model-output boundaries."""
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from crypto_quant.research.factor_mining.contracts import identifier, number, require, text

PARAMETERS = ("lookback", "spread_threshold", "min_winner_return")
SEMANTIC_DIMENSIONS = ("inputs", "signal", "execution", "position", "costs", "parameters")
RULES = {
    "family": "btc_eth_spot_relative_strength",
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "interval": "1h",
    "inputs": "BTCUSDT and ETHUSDT spot 1h OHLC and quote_volume; complete common hourly grid; "
              "bar t is available at t+1h-1ms; no missing-bar filling",
    "signal": "close / close.shift(lookback) - 1; compare BTC minus ETH",
    "entry": "abs(spread) >= spread_threshold and winner_return > min_winner_return",
    "position": "target 100% long the stronger asset or cash; no short or leverage; hourly target recomputation; "
                "actual buys constrained by available cash including fees; no separate stop loss",
    "execution": "closed-hour target executes at next hour open; sell before buy; each segment starts from fresh cash; "
                 "one pre-start signal bar; final close valuation without forced liquidation",
    "exit": "rotate or return to cash when the next target changes; no separate stop loss",
    "benchmark": "50/50 BTC/ETH hourly target with identical fees, slippage and no-trade band",
}


def fields(value: Any, names: tuple[str, ...]) -> dict:
    require(isinstance(value, dict) and set(value) == set(names), f"expected exact fields: {names}")
    return value


def strings(value: Any, name: str) -> list[str]:
    require(isinstance(value, list) and bool(value), f"{name} must be a nonempty list")
    return [text(item, name) for item in value]


@dataclass(frozen=True)
class StrategyResearchContract:
    run_id: str
    purpose: str
    task: dict
    development_start: str
    validation_start: str
    validation_end: str
    prior_data_use: str
    imputed_hours: list
    parameter_space: dict
    costs: dict
    gates: dict
    context_bytes: int

    @classmethod
    def from_dict(cls, value: dict):
        fields(value, tuple(cls.__dataclass_fields__))
        result = cls(**value)
        identifier(result.run_id)
        require(result.purpose in {"engineering", "research"}, "invalid purpose")
        fields(result.task, ("research_type", "strategy_family", "question", "deliverable"))
        for name, value in result.task.items():
            text(value, f"task.{name}")
        text(result.prior_data_use, "prior_data_use")
        require(isinstance(result.imputed_hours, list), "imputed_hours must be an explicit list")
        repair_times = []
        for value in result.imputed_hours:
            t = pd.Timestamp(text(value, "imputed hour"))
            require(t.tzinfo is not None and t == t.floor("h"), "imputed hours must be timezone-aware hours")
            repair_times.append(t.tz_convert("UTC"))
        require(len(set(repair_times)) == len(repair_times), "duplicate imputed hours")
        times = []
        for name in ("development_start", "validation_start", "validation_end"):
            t = pd.Timestamp(getattr(result, name))
            require(t.tzinfo is not None and t == t.floor("h"), "boundaries must be timezone-aware hours")
            times.append(t.tz_convert("UTC"))
        require(times[0] < times[1] < times[2] <= pd.Timestamp.now(tz="UTC"), "invalid historical split")
        require(all((b - a) >= pd.Timedelta(hours=3) for a, b in zip(times, times[1:])), "segments need three hours")
        if result.task["strategy_family"] == "spot_perp_basis":
            from .basis_strategy import check_space
            check_space(result.parameter_space)
            require(not result.imputed_hours, "basis route requires complete real spot/perp/mark bars; no imputation")
        elif result.task["strategy_family"] == "factor_rule_spot":
            fields(result.parameter_space, ("warmup_hours",))
            require(type(result.parameter_space["warmup_hours"]) is int and result.parameter_space["warmup_hours"] > 0,
                    "warmup_hours must be a positive integer")
        else:
            fields(result.parameter_space, PARAMETERS)
            for name, values in result.parameter_space.items():
                require(isinstance(values, list) and bool(values), "parameter choices must be nonempty lists")
                for item in values:
                    number(item, name)
                    if name == "lookback":
                        require(type(item) is int and item > 0, "lookback must be positive integer hours")
                    elif name == "spread_threshold":
                        require(item >= 0, "spread_threshold must be nonnegative")
                require(len(set(values)) == len(values), "duplicate parameter choices")
        fields(result.costs, ("initial_capital", "fee_bps", "slippage_bps", "min_trade_fraction", "stress_multiplier"))
        for name, value in result.costs.items():
            number(value, name)
        require(result.costs["initial_capital"] > 0, "capital must be positive")
        require(0 <= result.costs["min_trade_fraction"] < 1, "invalid no-trade band")
        if result.task["strategy_family"] == "spot_perp_basis":
            require(result.costs["min_trade_fraction"] == 0, "basis uses full pair open/close; min_trade_fraction must be 0")
        require(result.costs["stress_multiplier"] >= 1, "stress multiplier must be at least one")
        require(all(0 <= result.costs[n] * result.costs["stress_multiplier"] < 10000
                    for n in ("fee_bps", "slippage_bps")), "invalid base/stress costs")
        fields(result.gates, ("min_net_return", "min_excess_return", "max_drawdown", "min_traded_bars", "min_stress_return"))
        for name, value in result.gates.items():
            number(value, name)
        require(0 < result.gates["max_drawdown"] <= 1, "drawdown limit is a positive magnitude")
        require(type(result.gates["min_traded_bars"]) is int and result.gates["min_traded_bars"] > 0, "traded bars must be positive")
        return result

    def as_dict(self):
        return asdict(self)

    def parameters(self, value: dict) -> dict:
        if self.task["strategy_family"] == "spot_perp_basis":
            from .basis_strategy import check_parameters
            return check_parameters(value, self.parameter_space)
        if self.task["strategy_family"] == "factor_rule_spot":
            from .rule_strategy import compile_definition
            compile_definition(value, self.parameter_space["warmup_hours"])
            return value
        fields(value, PARAMETERS)
        for name in PARAMETERS:
            number(value[name], name)
            require(value[name] in self.parameter_space[name], f"{name} outside frozen parameter space")
        require(type(value["lookback"]) is int, "lookback must be integer hours")
        return value


def check_idea(value: dict) -> None:
    for name in ("id", "source_type"):
        text(value[name], name)
    for name in ("original_claim", "economic_mechanism", "market_and_horizon", "data_and_coverage",
                 "falsification_conditions", "unverified_assumptions", "next_research_plan"):
        require(name in value, f"idea card missing {name}")
