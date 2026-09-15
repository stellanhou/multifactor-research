from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from crypto_quant.backtesting.config import BacktestConfig


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark_equity: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    config: BacktestConfig
    initial_timestamp: Optional[pd.Timestamp] = None
    initial_equity: Optional[float] = None
    initial_benchmark_equity: Optional[float] = None


def bars_per_year(index: pd.DatetimeIndex, interval_seconds: int) -> float:
    observed_seconds = (index[-1] - index[0]).total_seconds()
    if len(index) < 2 or observed_seconds <= 0:
        raise ValueError("index must contain at least two timestamps")
    inferred_bar_seconds = observed_seconds / (len(index) - 1)
    if not np.isclose(inferred_bar_seconds, interval_seconds, rtol=0.05):
        # Gaps make this a warning context rather than a fatal issue; annualization
        # still follows the requested exchange interval.
        pass
    return 365.0 * 24.0 * 3600.0 / interval_seconds


def run_backtest(
    data: pd.DataFrame,
    target_weights: pd.Series,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    required = {"open", "close", "quote_volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing market fields: {sorted(missing)}")
    if not data.index.equals(target_weights.index):
        raise ValueError("data and target weights must share identical indexes")

    if config.allow_short:
        raise ValueError(
            "generic spot backtester supports long/flat targets only; "
            "use an explicit futures engine for short exposure"
        )
    targets = target_weights.astype(float).clip(lower=0.0, upper=1.0).copy()

    cash = float(config.initial_capital)
    units = 0.0
    average_entry = np.nan
    benchmark_units = config.initial_capital / float(data["open"].iloc[1])
    fee_rate = config.fee_bps / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []

    # The first available close is a decision point, never an execution point.
    # Thus iteration starts at the second bar's open.
    for position in range(1, len(data)):
        bar_open = float(data["open"].iloc[position])
        bar_close = float(data["close"].iloc[position])
        decision_time = data.index[position - 1]
        desired_weight = float(targets.iloc[position - 1])

        marked_equity = cash + units * bar_open
        current_weight = units * bar_open / marked_equity if marked_equity > 0 else 0.0
        trade_fraction = abs(desired_weight - current_weight)
        trade_notional = 0.0
        fee = 0.0
        slippage_cost = 0.0

        if trade_fraction > config.min_trade_fraction and marked_equity > 0:
            buying = desired_weight > current_weight
            fill_price = bar_open * (1 + slippage_rate if buying else 1 - slippage_rate)
            target_units = desired_weight * marked_equity / fill_price
            delta_units = target_units - units
            trade_notional = abs(delta_units) * fill_price
            fee = trade_notional * fee_rate
            slippage_cost = abs(delta_units) * abs(fill_price - bar_open)

            if delta_units >= 0:
                if units == 0:
                    new_average = fill_price
                else:
                    new_average = (
                        average_entry * units + fill_price * delta_units
                    ) / (units + delta_units)
                cash -= delta_units * fill_price + fee
                units += delta_units
                average_entry = new_average
            else:
                sold_units = min(units, -delta_units)
                gross_pnl = (fill_price - average_entry) * sold_units
                net_pnl = gross_pnl - fee
                # Signed quantity keeps both sides in one ledger:
                # buying removes cash; selling adds proceeds.
                cash -= delta_units * fill_price + fee
                units += delta_units
                entry_cost = average_entry * sold_units
                trade_rows.append(
                    {
                        "exit_time": data.index[position],
                        "direction": "long",
                        "quantity": sold_units,
                        "entry_price": average_entry,
                        "exit_price": fill_price,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "return": net_pnl / entry_cost if entry_cost else 0.0,
                        "fees": fee,
                    }
                )
                if np.isclose(units, 0.0, atol=1e-12):
                    units = 0.0
                    average_entry = np.nan

        closing_equity = cash + units * bar_close
        executed_weight = units * bar_close / closing_equity if closing_equity else 0.0
        rows.append(
            {
                "timestamp": data.index[position],
                "equity": closing_equity,
                "benchmark_equity": benchmark_units * bar_close,
                "target_weight": desired_weight,
                "executed_weight": executed_weight,
                "trade_notional": trade_notional,
                "cost": fee + slippage_cost,
                "turnover": trade_notional / marked_equity if marked_equity else 0.0,
                "bar_quote_volume": float(data["quote_volume"].iloc[position]),
            }
        )

    frame = pd.DataFrame(rows).set_index("timestamp")
    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
                "exit_time",
                "direction",
                "quantity",
                "entry_price",
                "exit_price",
                "gross_pnl",
                "net_pnl",
                "return",
                "fees",
            ]
        )
    return BacktestResult(
        equity=frame["equity"],
        benchmark_equity=frame["benchmark_equity"],
        weights=frame[
            [
                "target_weight",
                "executed_weight",
                "turnover",
                "trade_notional",
                "cost",
                "bar_quote_volume",
            ]
        ],
        trades=trades,
        config=config,
        initial_timestamp=data.index[0],
        initial_equity=float(config.initial_capital),
        initial_benchmark_equity=float(config.initial_capital),
    )
