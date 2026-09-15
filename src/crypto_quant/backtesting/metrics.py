from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from crypto_quant.backtesting.backtest import BacktestResult


MONTHLY_RETURN_DISTRIBUTION_FIELDS = (
    "monthly_return_count",
    "monthly_return_mean",
    "monthly_return_std",
    "monthly_return_best",
    "monthly_return_worst",
    "monthly_return_p05",
    "monthly_return_p50",
    "monthly_return_p95",
    "monthly_return_positive_share",
)
MAX_CONSECUTIVE_LOSS_FIELDS = (
    "max_consecutive_losing_trades",
    "max_consecutive_loss_net_pnl",
)
STANDARD_RISK_RETURN_FIELDS = (
    *MONTHLY_RETURN_DISTRIBUTION_FIELDS,
    *MAX_CONSECUTIVE_LOSS_FIELDS,
)

def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def monthly_return_distribution(equity: pd.Series) -> Dict[str, Any]:
    """Summarize compounded calendar-month returns without filling missing months."""
    if not isinstance(equity.index, pd.DatetimeIndex):
        raise ValueError("monthly returns require a datetime equity index")
    month_ends = equity.sort_index().resample("ME").last()
    monthly_returns = month_ends.pct_change().dropna()

    def value(function) -> Any:
        return float(function(monthly_returns)) if len(monthly_returns) else np.nan

    positive_share = (
        float((monthly_returns > 0).mean()) if len(monthly_returns) else np.nan
    )
    return {
        "monthly_return_count": int(len(monthly_returns)),
        "monthly_return_mean": value(lambda values: values.mean()),
        "monthly_return_std": value(lambda values: values.std(ddof=1)),
        "monthly_return_best": value(lambda values: values.max()),
        "monthly_return_worst": value(lambda values: values.min()),
        "monthly_return_p05": value(lambda values: values.quantile(0.05)),
        "monthly_return_p50": value(lambda values: values.quantile(0.50)),
        "monthly_return_p95": value(lambda values: values.quantile(0.95)),
        "monthly_return_positive_share": positive_share,
    }


def max_consecutive_losses(net_pnl: pd.Series) -> Dict[str, Any]:
    """Return the longest closed-trade losing streak and its cumulative P&L."""
    longest = 0
    current = 0
    worst_streak_pnl = 0.0
    current_streak_pnl = 0.0
    for value in pd.to_numeric(pd.Series(net_pnl), errors="coerce").fillna(0.0):
        value = float(value)
        if value >= 0:
            current = 0
            current_streak_pnl = 0.0
            continue
        current += 1
        current_streak_pnl += value
        longest = max(longest, current)
        worst_streak_pnl = min(worst_streak_pnl, current_streak_pnl)
    return {
        "max_consecutive_losing_trades": int(longest),
        "max_consecutive_loss_net_pnl": float(worst_streak_pnl),
    }


def _years(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max((end - start).total_seconds() / (365.0 * 24 * 3600), 1e-12)


def _evaluation_paths(
    result: BacktestResult,
    equity: pd.Series,
    benchmark: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Prepend the causal pre-period account value used to earn the first return."""
    first_timestamp = equity.index[0]
    prior = result.equity.loc[result.equity.index < first_timestamp]
    if len(prior):
        anchor_timestamp = prior.index[-1]
        anchor_equity = float(prior.iloc[-1])
        anchor_benchmark = float(result.benchmark_equity.loc[anchor_timestamp])
    elif (
        result.initial_timestamp is not None
        and result.initial_equity is not None
        and result.initial_benchmark_equity is not None
        and result.initial_timestamp < first_timestamp
    ):
        anchor_timestamp = result.initial_timestamp
        anchor_equity = float(result.initial_equity)
        anchor_benchmark = float(result.initial_benchmark_equity)
    else:
        return equity, benchmark

    equity_anchor = pd.Series(
        [anchor_equity], index=pd.DatetimeIndex([anchor_timestamp]), dtype=float
    )
    benchmark_anchor = pd.Series(
        [anchor_benchmark], index=pd.DatetimeIndex([anchor_timestamp]), dtype=float
    )
    return (
        pd.concat([equity_anchor, equity.astype(float)]),
        pd.concat([benchmark_anchor, benchmark.astype(float)]),
    )


def performance_metrics(
    result: BacktestResult,
    bars_per_year_value: float,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    start = result.equity.index[0] if start is None else start
    end = result.equity.index[-1] if end is None else end
    equity = result.equity.loc[start:end]
    benchmark = result.benchmark_equity.loc[start:end]
    weights = result.weights.loc[start:end]
    if len(equity) < 2:
        raise ValueError("evaluation period needs at least two observations")

    equity_path, benchmark_path = _evaluation_paths(result, equity, benchmark)
    years = _years(equity_path.index[0], equity_path.index[-1])
    strategy_returns = equity_path.pct_change().dropna()
    benchmark_returns = benchmark_path.pct_change().dropna()
    total_return = float(equity_path.iloc[-1] / equity_path.iloc[0] - 1.0)
    equity_ratio = float(equity_path.iloc[-1] / equity_path.iloc[0])
    # A non-positive terminal account has no meaningful finite geometric growth rate.
    cagr = (
        float(equity_ratio ** (1.0 / years) - 1.0)
        if equity_ratio > 0
        else -1.0
    )
    volatility = float(strategy_returns.std(ddof=1) * math.sqrt(bars_per_year_value))
    mean_return = float(strategy_returns.mean() * bars_per_year_value)
    sharpe = mean_return / volatility if volatility > 0 else 0.0

    mar = 0.0
    downside = strategy_returns.clip(upper=0.0) - mar
    downside_deviation = float(
        np.sqrt(np.mean(np.square(downside))) * math.sqrt(bars_per_year_value)
    )
    sortino = mean_return / downside_deviation if downside_deviation > 0 else 0.0

    drawdown = drawdown_series(equity_path)
    max_drawdown = float(drawdown.min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    active_weight = weights["executed_weight"].abs().shift(1).fillna(0.0)
    active_returns = strategy_returns[active_weight.reindex(strategy_returns.index) > 1e-12]
    period_win_rate = (
        float((active_returns > 0).mean()) if len(active_returns) else np.nan
    )

    trades = result.trades.copy()
    if not trades.empty:
        trades = trades.set_index("exit_time").sort_index().loc[start:end].reset_index()
    wins = trades[trades["net_pnl"] > 0]["return"] if not trades.empty else pd.Series(dtype=float)
    losses = trades[trades["net_pnl"] <= 0]["return"] if not trades.empty else pd.Series(dtype=float)
    trade_win_rate = float(len(wins) / len(trades)) if len(trades) else np.nan
    payoff_ratio = (
        float(wins.mean() / abs(losses.mean()))
        if len(wins) and len(losses) and losses.mean() != 0
        else np.nan
    )
    gross_profit = float(trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()) if not trades.empty else 0.0
    gross_loss = float(abs(trades.loc[trades["net_pnl"] <= 0, "net_pnl"].sum())) if not trades.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    turnover_annualized = float(weights["turnover"].sum() / years)
    cost_total = float(weights["cost"].sum())
    traded = weights[weights["trade_notional"] > 0]
    participation = traded["trade_notional"] / traded["bar_quote_volume"].replace(0, np.nan)
    p95_participation = float(participation.quantile(0.95)) if len(participation) else 0.0
    execution_feasible = bool(p95_participation <= 0.01)

    benchmark_total = float(
        benchmark_path.iloc[-1] / benchmark_path.iloc[0] - 1.0
    )
    benchmark_ratio = float(benchmark_path.iloc[-1] / benchmark_path.iloc[0])
    benchmark_cagr = (
        float(benchmark_ratio ** (1.0 / years) - 1.0)
        if benchmark_ratio > 0
        else -1.0
    )
    benchmark_vol = float(benchmark_returns.std(ddof=1) * math.sqrt(bars_per_year_value))
    benchmark_sharpe = (
        float(benchmark_returns.mean() * bars_per_year_value / benchmark_vol)
        if benchmark_vol > 0
        else 0.0
    )
    benchmark_drawdown = float(drawdown_series(benchmark_path).min())

    return {
        "start": equity.index[0].isoformat(),
        "end": equity.index[-1].isoformat(),
        "return_anchor": equity_path.index[0].isoformat(),
        "years": years,
        "total_return": total_return,
        "annualized_return": cagr,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        **monthly_return_distribution(equity_path),
        "period_win_rate": period_win_rate,
        "exposure": float(weights["executed_weight"].abs().mean()),
        "turnover_annualized": turnover_annualized,
        "cost_to_initial_capital": cost_total / result.config.initial_capital,
        "trade_count": int(len(trades)),
        "trade_win_rate": trade_win_rate,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        **(
            max_consecutive_losses(trades["net_pnl"])
            if not trades.empty and "net_pnl" in trades.columns
            else max_consecutive_losses(pd.Series(dtype=float))
        ),
        "median_trade_participation": float(participation.median()) if len(participation) else 0.0,
        "p95_trade_participation": p95_participation,
        "max_trade_participation": float(participation.max()) if len(participation) else 0.0,
        "execution_feasible_at_research_size": execution_feasible,
        "benchmark_total_return": benchmark_total,
        "benchmark_annualized_return": benchmark_cagr,
        "benchmark_sharpe_ratio": benchmark_sharpe,
        "benchmark_max_drawdown": benchmark_drawdown,
    }


def excess_vs_benchmark(metrics: Dict[str, Any], field: str) -> float:
    return float(metrics[field] - metrics[f"benchmark_{field}"])
