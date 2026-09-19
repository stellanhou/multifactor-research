"""Explicit operator choices and strict model-output boundaries."""
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from crypto_quant.research.factor_mining.contracts import identifier, number, require, text

PARAMETERS = ("lookback", "spread_threshold", "min_winner_return")
RULES = {
    "family": "btc_eth_spot_relative_strength",
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "interval": "1h",
    "signal": "close / close.shift(lookback) - 1; compare BTC minus ETH",
    "entry": "abs(spread) >= spread_threshold and winner_return > min_winner_return",
    "position": "100% long the stronger asset, otherwise cash; ties remain cash",
    "execution": "recompute each closed hour, execute at next hour open; sells before buys",
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
    development_start: str
    validation_start: str
    validation_end: str
    prior_data_use: str
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
        text(result.prior_data_use, "prior_data_use")
        times = []
        for name in ("development_start", "validation_start", "validation_end"):
            t = pd.Timestamp(getattr(result, name))
            require(t.tzinfo is not None and t == t.floor("h"), "boundaries must be timezone-aware hours")
            times.append(t.tz_convert("UTC"))
        require(times[0] < times[1] < times[2] <= pd.Timestamp.now(tz="UTC"), "invalid historical split")
        require(all((b - a) >= pd.Timedelta(hours=3) for a, b in zip(times, times[1:])), "segments need three hours")
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
        require(result.costs["stress_multiplier"] >= 1, "stress multiplier must be at least one")
        require(all(0 <= result.costs[n] * result.costs["stress_multiplier"] < 10000
                    for n in ("fee_bps", "slippage_bps")), "invalid base/stress costs")
        fields(result.gates, ("min_net_return", "min_excess_return", "max_drawdown", "min_traded_bars", "min_stress_return"))
        for name, value in result.gates.items():
            number(value, name)
        require(0 < result.gates["max_drawdown"] <= 1, "drawdown limit is a positive magnitude")
        require(type(result.gates["min_traded_bars"]) is int and result.gates["min_traded_bars"] > 0, "traded bars must be positive")
        require(type(result.context_bytes) is int and result.context_bytes > 0, "context_bytes must be positive")
        return result

    def as_dict(self):
        return asdict(self)

    def parameters(self, value: dict) -> dict:
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
