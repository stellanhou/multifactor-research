"""Declarative signals and portfolio rules executed by the existing spot backtester."""
import ast
import copy

import numpy as np
import pandas as pd

from crypto_quant.features.factor_expressions import compile_expression, evaluate_expression, OPERATORS, TEMPLATES
from crypto_quant.features.factor_inputs import FactorInputPanel, FIELD_BY_NAME
from crypto_quant.research.factor_mining.contracts import number, require, text
from .contracts import fields


FAMILY = "factor_rule_spot"
INPUTS = ("spot_open", "spot_high", "spot_low", "spot_close", "spot_volume", "spot_quote_volume")
COMPONENTS = ("signals", "score", "entry", "exit", "allocation", "rebalance_hours")
DEFINITION_FORMAT = {
    "signals": [{"name": "unique identifier", "expression": "existing causal operator expression using supported inputs"}],
    "score": "numeric expression using named signals or supported inputs; larger ranks first",
    "entry": "numeric expression; value > 0 enters",
    "exit": "numeric expression; value > 0 exits, takes precedence over entry",
    "allocation": {"max_positions": "integer 1..2", "gross_exposure": "number > 0 and <= 1",
                   "max_asset_weight": "number > 0 and <= 1"},
    "rebalance_hours": "positive integer hours",
}


def compile_definition(definition: dict, warmup_hours: int) -> dict:
    fields(definition, COMPONENTS)
    require(isinstance(definition["signals"], list), "signals must be a list")
    aliases = {}
    for item in definition["signals"]:
        fields(item, ("name", "expression"))
        name = text(item["name"], "signal name")
        require(name.isidentifier() and name not in {*FIELD_BY_NAME, *OPERATORS, *TEMPLATES, *aliases},
                "signal names must be unique and must not shadow inputs or operators")
        compiled = compile_expression(text(item["expression"], "signal expression"))
        require(set(compiled.fields) <= set(INPUTS), "signal requires unsupported strategy inputs")
        require(compiled.lookback_hours <= warmup_hours, "signal exceeds contract warmup_hours")
        aliases[name] = compiled.tree

    class ExpandSignals(ast.NodeTransformer):
        def visit_Name(self, node):
            return copy.deepcopy(aliases[node.id]) if node.id in aliases else node

    result = {}
    for name in ("score", "entry", "exit"):
        expression = text(definition[name], name)
        tree = ExpandSignals().visit(ast.parse(expression, mode="eval"))
        compiled = compile_expression(ast.unparse(tree))
        require(set(compiled.fields) <= set(INPUTS), "rule requires unsupported strategy inputs")
        require(compiled.lookback_hours <= warmup_hours, "rule exceeds contract warmup_hours")
        result[name] = compiled
    allocation = fields(definition["allocation"], ("max_positions", "gross_exposure", "max_asset_weight"))
    require(type(allocation["max_positions"]) is int and 1 <= allocation["max_positions"] <= 2,
            "max_positions must be 1 or 2")
    for name in ("gross_exposure", "max_asset_weight"):
        require(0 < number(allocation[name], name) <= 1, f"{name} must be in (0,1]")
    require(type(definition["rebalance_hours"]) is int and definition["rebalance_hours"] > 0,
            "rebalance_hours must be a positive integer")
    return result


def generate_rule_targets(frames: dict, definition: dict, warmup_hours: int, start: pd.Timestamp):
    compiled = compile_definition(definition, warmup_hours)
    values = pd.concat({s: f[[name.removeprefix("spot_") for name in INPUTS]].rename(
        columns=lambda name: "spot_" + name) for s, f in frames.items()}, names=["symbol", "timestamp"])
    values = values.reorder_levels(["timestamp", "symbol"]).sort_index()
    panel = FactorInputPanel(values, pd.Series(True, index=values.index), {})
    rules = {name: evaluate_expression(item.expanded_expression, panel).values.unstack("symbol")
             for name, item in compiled.items()}
    index = rules["score"].index
    targets = pd.DataFrame(0.0, index=index, columns=rules["score"].columns)
    first = start - pd.Timedelta(hours=1)
    active = set()
    weights = pd.Series(0.0, index=targets.columns)
    allocation = definition["allocation"]
    for timestamp in index[index >= first]:
        elapsed = int((timestamp - first) / pd.Timedelta(hours=1))
        if elapsed % definition["rebalance_hours"] == 0:
            row = {name: table.loc[timestamp] for name, table in rules.items()}
            require(all(np.isfinite(series).all() for series in row.values()),
                    f"strategy rule is undefined at rebalance {timestamp}")
            eligible = [s for s in targets.columns if row["exit"][s] <= 0 and
                        (s in active or row["entry"][s] > 0)]
            selected = sorted(eligible, key=lambda s: (-row["score"][s], s))[:allocation["max_positions"]]
            weights = pd.Series(0.0, index=targets.columns)
            if selected:
                weights.loc[selected] = min(allocation["gross_exposure"] / len(selected), allocation["max_asset_weight"])
            active = set(selected)
        targets.loc[timestamp] = weights
    trace = pd.concat(rules, axis=1)
    return {symbol: targets[symbol] for symbol in targets}, trace
