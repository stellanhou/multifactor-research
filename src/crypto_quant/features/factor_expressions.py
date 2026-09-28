"""Restricted, causal factor expressions without formula-unit inference."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from crypto_quant.features.factor_inputs import DEFERRED_FIELDS, FIELD_BY_NAME, FactorInputPanel, validate_universe


@dataclass(frozen=True)
class Operator:
    syntax: str
    meaning: str
    arity: int
    scope: str


OPERATORS = {
    "add": Operator("add(x,y)", "逐点相加", 2, "pointwise"),
    "sub": Operator("sub(x,y)", "逐点相减", 2, "pointwise"),
    "mul": Operator("mul(x,y)", "逐点相乘", 2, "pointwise"),
    "div": Operator("div(x,y)", "相除，零分母为空", 2, "pointwise"),
    "abs": Operator("abs(x)", "绝对值", 1, "pointwise"),
    "neg": Operator("neg(x)", "取负", 1, "pointwise"),
    "sign": Operator("sign(x)", "符号：-1、0、1", 1, "pointwise"),
    "min": Operator("min(x,y)", "逐点较小值", 2, "pointwise"),
    "max": Operator("max(x,y)", "逐点较大值", 2, "pointwise"),
    "log": Operator("log(x)", "自然对数，非正数为空", 1, "pointwise"),
    "sqrt": Operator("sqrt(x)", "平方根，负数为空", 1, "pointwise"),
    "power": Operator("power(x,p)", "幂；指数为数字常量，非有限实数结果为空", 2, "pointwise"),
    "ts_delay": Operator("ts_delay(x,n)", "准确n小时前的值", 2, "timeseries"),
    "ts_delta": Operator("ts_delta(x,n)", "当前减n小时前", 2, "timeseries"),
    "ts_return": Operator("ts_return(x,n)", "当前/n小时前-1，零分母为空", 2, "timeseries"),
    "ts_rsi": Operator("ts_rsi(x,n)", "Wilder RSI：n个连续变化均值初始化，之后1/n递推；涨跌均为0时50，仅无跌时100，仅无涨时0；缺失后重新初始化；依赖本面板初始化后的全部过去值", 2, "timeseries"),
    "ts_mean": Operator("ts_mean(x,n)", "完整n小时窗口均值，含当前已结束小时", 2, "timeseries"),
    "ts_sum": Operator("ts_sum(x,n)", "完整n小时窗口之和；小时重复费率之和不代表结算支付", 2, "timeseries"),
    "ts_min": Operator("ts_min(x,n)", "完整n小时窗口最小值", 2, "timeseries"),
    "ts_max": Operator("ts_max(x,n)", "完整n小时窗口最大值", 2, "timeseries"),
    "ts_std": Operator("ts_std(x,n)", "完整n小时窗口总体标准差，ddof=0", 2, "timeseries"),
    "ts_corr": Operator("ts_corr(x,y,n)", "完整成对n小时窗口Pearson相关，零方差为空", 3, "timeseries"),
    "ts_cov": Operator("ts_cov(x,y,n)", "完整成对n小时窗口总体协方差，ddof=0", 3, "timeseries"),
    "ts_rank": Operator("ts_rank(x,n)", "完整n小时窗口当前值排名：(平均并列名次-1)/(n-1)，n<2为空", 2, "timeseries"),
    "cross_rank": Operator("cross_rank(x)", "当时币池内有效值排名：(平均并列名次-1)/(有效数-1)，有效数<2为空", 1, "cross_section"),
    "cross_zscore": Operator("cross_zscore(x)", "当时币池内(值-均值)/总体标准差，零标准差为空", 1, "cross_section"),
    "cross_pct": Operator("cross_pct(x)", "100*cross_rank(x)的别名", 1, "cross_section"),
}

TEMPLATES = {
    "basis_trade_spot": "sub(div(perp_close, spot_close), 1)",
    "basis_mark_index": "sub(div(mark_close, index_close), 1)",
    "trade_mark_spread": "sub(div(perp_close, mark_close), 1)",
    "basis_change_1bar": "ts_delta(basis_trade_spot, 1)",
    "open_interest_change_1bar": "ts_return(open_interest_base, 1)",
    "open_interest_change_24h": "ts_return(open_interest_base, 24)",
    "toptrader_account_net": "div(sub(toptrader_account_long_short_ratio, 1), add(toptrader_account_long_short_ratio, 1))",
    "toptrader_position_net": "div(sub(toptrader_position_long_short_ratio, 1), add(toptrader_position_long_short_ratio, 1))",
    "global_account_net": "div(sub(global_account_long_short_ratio, 1), add(global_account_long_short_ratio, 1))",
    "spot_taker_buy_share": "div(spot_taker_buy_quote_volume, spot_quote_volume)",
    "perp_taker_buy_share": "div(perp_taker_buy_quote_volume, perp_quote_volume)",
    "spot_quote_volume_24h": "ts_sum(spot_quote_volume, 24)",
    "perp_quote_volume_24h": "ts_sum(perp_quote_volume, 24)",
    "perp_to_spot_quote_volume": "div(perp_quote_volume_24h, spot_quote_volume_24h)",
    "spot_log_return_1bar": "log(div(spot_close, ts_delay(spot_close, 1)))",
    "perp_log_return_1bar": "log(div(perp_close, ts_delay(perp_close, 1)))",
    "spot_perp_return_spread_1bar": "sub(perp_log_return_1bar, spot_log_return_1bar)",
    "perp_realized_vol_24h": "mul(ts_std(perp_log_return_1bar, 24), sqrt(8766))",
}


def operator_catalog() -> list[dict[str, Any]]:
    return [{"name": name, **vars(op)} for name, op in OPERATORS.items()]


@dataclass(frozen=True)
class CompiledExpression:
    expression: str
    expanded_expression: str
    fields: tuple[str, ...]
    lookback_hours: int
    tree: ast.AST

    def description(self) -> dict[str, Any]:
        return {
            "expression": self.expression, "expanded_expression": self.expanded_expression,
            "fields": list(self.fields),
            "lookback_hours": self.lookback_hours,
        }

    def calculation_steps(self) -> list[dict[str, Any]]:
        """Post-order facts from the same expanded tree used for execution."""
        steps: list[dict[str, Any]] = []

        def visit(node: ast.AST) -> int:
            expression = ast.unparse(node)
            if isinstance(node, ast.Name):
                field = FIELD_BY_NAME[node.id]
                detail = {"field": node.id, "market": field.source,
                          "meaning": field.meaning, "unit": field.unit,
                          "availability": field.alignment}
            elif isinstance(node, ast.Call):
                arguments = [visit(arg) for arg in node.args]
                operator = OPERATORS[node.func.id]
                detail = {"operator": node.func.id, "arguments": arguments,
                          "meaning": operator.meaning, "scope": operator.scope}
                if operator.scope == "timeseries":
                    detail["window_hours"] = node.args[-1].value
                    detail["current_period"] = (
                        "only the value exactly n hours ago" if node.func.id == "ts_delay"
                        else "includes the current completed hour of each input expression"
                    )
                    if node.func.id == "ts_rsi":
                        detail["history"] = "recursive since panel initialization; window_hours is minimum seed history, not a finite dependency bound"
                if operator.scope == "cross_section":
                    detail["membership"] = "eligible assets with valid input at this historical hour"
            else:
                detail = {"constant": _number(node)}
            step_id = len(steps) + 1
            steps.append({"step": step_id, "expression": expression, **detail})
            return step_id

        visit(self.tree)
        return steps


def _number(node: ast.AST) -> float:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _number(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if not isinstance(node, ast.Constant) or type(node.value) not in (int, float):
        raise ValueError("parameter must be a numeric literal")
    value = float(node.value)
    if not math.isfinite(value):
        raise ValueError("numeric literal must be finite")
    return value


def compile_expression(expression: str) -> CompiledExpression:
    """Parse only literal/field/function expressions; no Python eval is used."""
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"invalid expression syntax: {exc.msg}") from exc
    fields: set[str] = set()

    def visit(node: ast.AST) -> tuple[ast.AST, int]:
        if isinstance(node, (ast.Constant, ast.UnaryOp)):
            _number(node)
            return node, 0
        if isinstance(node, ast.Name):
            if node.id in TEMPLATES:
                return visit(ast.parse(TEMPLATES[node.id], mode="eval").body)
            if node.id in DEFERRED_FIELDS:
                raise ValueError(f"field is deferred: {node.id}: {DEFERRED_FIELDS[node.id]}")
            if node.id not in FIELD_BY_NAME:
                raise ValueError(f"field/template is not allowed: {node.id}")
            fields.add(node.id)
            return node, 0
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError("only allowed fields, numeric literals and operator calls are accepted")
        name = node.func.id
        if name not in OPERATORS:
            raise ValueError(f"operator is not allowed: {name}")
        op = OPERATORS[name]
        if node.keywords or len(node.args) != op.arity:
            raise ValueError(f"expected {op.syntax}")
        if name == "power":
            _number(node.args[1])
        if op.scope == "timeseries":
            window = node.args[-1]
            if not isinstance(window, ast.Constant) or type(window.value) is not int or window.value <= 0:
                raise ValueError("window must be an explicit positive integer number of hours")
        args = [visit(arg) for arg in node.args]
        lookback = max(arg[1] for arg in args)
        if op.scope == "timeseries":
            n = node.args[-1].value
            lookback += n if name in {"ts_delay", "ts_delta", "ts_return", "ts_rsi"} else n - 1
        node = ast.Call(func=node.func, args=[arg[0] for arg in args], keywords=[])
        return node, lookback

    expanded, lookback = visit(tree)
    return CompiledExpression(expression, ast.unparse(expanded), tuple(sorted(fields)), lookback, expanded)


def template_catalog() -> list[dict[str, Any]]:
    return [{"name": name, **compile_expression(expression).description()} for name, expression in TEMPLATES.items()]


@dataclass
class ExpressionResult:
    values: pd.Series
    definition: dict[str, Any]
    cross_section_counts: pd.DataFrame


def wilder_rsi(values: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Causal SMA seed followed by Wilder recursion; gaps reset the state."""
    result = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    for column in values:
        previous = np.nan
        count, gain, loss = 0, 0.0, 0.0
        output = np.full(len(values), np.nan)
        for i, value in enumerate(values[column].to_numpy(dtype=float)):
            if not np.isfinite(value):
                previous, count, gain, loss = np.nan, 0, 0.0, 0.0
                continue
            if np.isfinite(previous):
                change = value - previous
                up, down = max(change, 0.0), max(-change, 0.0)
                if count < periods:
                    gain += up
                    loss += down
                    count += 1
                    if count == periods:
                        gain /= periods
                        loss /= periods
                else:
                    gain = (gain * (periods - 1) + up) / periods
                    loss = (loss * (periods - 1) + down) / periods
                if count == periods:
                    output[i] = 50.0 if gain + loss == 0 else 100.0 * gain / (gain + loss)
            previous = value
        result[column] = output
    return result


def evaluate_expression(expression: str, panel: FactorInputPanel) -> ExpressionResult:
    compiled = compile_expression(expression)
    universe = validate_universe(panel.universe)
    if not panel.values.index.equals(universe.index):
        raise ValueError("values and universe must have identical canonical hourly indices")
    missing = set(compiled.fields) - set(panel.values.columns)
    if missing:
        raise ValueError(f"required input columns are missing: {sorted(missing)}")
    members = universe.unstack("symbol")
    counts: dict[str, pd.Series] = {}

    def constant(number: float) -> pd.DataFrame:
        return pd.DataFrame(number, index=members.index, columns=members.columns)

    def evaluate(node: ast.AST) -> pd.DataFrame:
        if isinstance(node, (ast.Constant, ast.UnaryOp)):
            return constant(_number(node))
        if isinstance(node, ast.Name):
            values = panel.values[node.id].unstack("symbol").astype(float)
            return values.where(np.isfinite(values))
        name = node.func.id
        op = OPERATORS[name]
        data_args = node.args[:-1] if op.scope == "timeseries" or name == "power" else node.args
        args = [evaluate(arg) for arg in data_args]
        x = args[0]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if name == "add": output = x + args[1]
            elif name == "sub": output = x - args[1]
            elif name == "mul": output = x * args[1]
            elif name == "div": output = x / args[1].where(args[1] != 0)
            elif name == "abs": output = x.abs()
            elif name == "neg": output = -x
            elif name == "sign": output = np.sign(x)
            elif name == "min": output = np.minimum(x, args[1])
            elif name == "max": output = np.maximum(x, args[1])
            elif name == "log": output = np.log(x.where(x > 0))
            elif name == "sqrt": output = np.sqrt(x.where(x >= 0))
            elif name == "power": output = np.power(x, _number(node.args[1]))
            elif op.scope == "cross_section":
                x = x.where(members)
                n = x.count(axis=1)
                counts[ast.unparse(node)] = n
                if name == "cross_zscore":
                    std = x.std(axis=1, ddof=0)
                    output = x.sub(x.mean(axis=1), axis=0).div(std.where(std != 0), axis=0)
                else:
                    output = x.rank(axis=1, method="average").sub(1).div((n - 1).where(n > 1), axis=0)
                    if name == "cross_pct": output = output * 100
            else:
                n = node.args[-1].value
                rolling = x.rolling(n, min_periods=n)
                if name == "ts_delay": output = x.shift(n)
                elif name == "ts_delta": output = x - x.shift(n)
                elif name == "ts_return": output = x / x.shift(n).where(x.shift(n) != 0) - 1
                elif name == "ts_rsi": output = wilder_rsi(x, n)
                elif name == "ts_mean": output = rolling.mean()
                elif name == "ts_sum": output = rolling.sum()
                elif name == "ts_min": output = rolling.min()
                elif name == "ts_max": output = rolling.max()
                elif name == "ts_std": output = rolling.std(ddof=0)
                elif name == "ts_rank": output = (rolling.rank(method="average") - 1) / (n - 1) if n > 1 else constant(np.nan)
                elif name == "ts_corr":
                    output = rolling.corr(args[1], pairwise=False, ddof=0)
                    valid = (rolling.std(ddof=0) > 0) & (args[1].rolling(n, min_periods=n).std(ddof=0) > 0)
                    output = output.where(valid)
                elif name == "ts_cov": output = rolling.cov(args[1], pairwise=False, ddof=0)
                else: raise RuntimeError(f"operator has no implementation: {name}")
        return output.where(np.isfinite(output))

    calculated = evaluate(compiled.tree).where(members)
    result = pd.Series(calculated.to_numpy().reshape(-1), index=universe.index, name="factor_value")
    return ExpressionResult(result, compiled.description(), pd.DataFrame(counts, index=members.index))
