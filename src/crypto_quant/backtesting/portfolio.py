from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from crypto_quant.backtesting.backtest import BacktestConfig, BacktestResult, bars_per_year, run_backtest
from crypto_quant.data_access.data import interval_to_ms, load_klines, validate_klines
from crypto_quant.backtesting.metrics import STANDARD_RISK_RETURN_FIELDS, performance_metrics
from crypto_quant.research.provenance import build_source_manifest
from crypto_quant.research.reporting import _sanitize_json, file_hash
from crypto_quant.strategies.strategies import strategy_registry
from crypto_quant.backtesting.validation import RISK_GATES, risk_gate


PORTFOLIO_STRATEGY = "donchian_96_48"
PORTFOLIO_NO_TRADE_BAND = 0.05
SLEEVE_WEIGHTS = {
    "BTCUSDT": 0.50,
    "ETHUSDT": 0.50,
}


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "exit_time",
            "symbol",
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


def run_portfolio_backtest(
    market_data: Dict[str, pd.DataFrame],
    target_weights: Dict[str, pd.Series],
    sleeve_weights: Dict[str, float],
    config: Optional[BacktestConfig] = None,
) -> Tuple[BacktestResult, pd.DataFrame]:
    """Event backtest for long/flat sleeves sharing one cash account.

    Each sleeve's cap is a portfolio-level target. Thus two 50% sleeves invest at
    most 100% combined; a single active signal leaves the other half in cash.
    """
    config = config or BacktestConfig()
    if config.allow_short:
        raise ValueError("current portfolio engine supports long/flat spot sleeves")
    symbols = sorted(market_data)
    if not symbols:
        raise ValueError("market_data cannot be empty")
    if set(target_weights) != set(market_data):
        raise ValueError("target weights and market-data symbols must match")
    if set(sleeve_weights) != set(market_data):
        raise ValueError("sleeve weights and market-data symbols must match")
    if any(weight <= 0 or weight > 1 for weight in sleeve_weights.values()):
        raise ValueError("each sleeve weight must be in (0, 1]")
    if sum(sleeve_weights.values()) > 1 + 1e-12:
        raise ValueError("sleeve weights cannot exceed 100% gross exposure")

    common_index = market_data[symbols[0]].index
    for symbol in symbols[1:]:
        common_index = common_index.intersection(market_data[symbol].index)
    if len(common_index) < 3:
        raise ValueError("assets need at least three common timestamps")

    aligned_data: Dict[str, pd.DataFrame] = {}
    aligned_targets: Dict[str, pd.Series] = {}
    for symbol in symbols:
        frame = market_data[symbol].loc[common_index]
        target = target_weights[symbol].reindex(common_index).fillna(0.0).astype(float)
        missing_required = {"open", "close", "quote_volume"} - set(frame.columns)
        if missing_required:
            raise ValueError(f"{symbol} missing fields: {sorted(missing_required)}")
        if not np.isfinite(target).all() or target.min() < 0:
            raise ValueError(f"{symbol} has invalid target weights")
        aligned_data[symbol] = frame
        aligned_targets[symbol] = target.clip(
            lower=0.0,
            upper=sleeve_weights[symbol] if not config.allow_short else 1.0,
        )

    fee_rate = config.fee_bps / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    cash = float(config.initial_capital)
    units = {symbol: 0.0 for symbol in symbols}
    average_entry = {symbol: np.nan for symbol in symbols}
    benchmark_units = {
        symbol: config.initial_capital * sleeve_weights[symbol]
        / float(aligned_data[symbol]["open"].iloc[1])
        for symbol in symbols
    }
    rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []

    # The first common close is a decision point; execution starts next open.
    for position in range(1, len(common_index)):
        timestamp = common_index[position]
        opens = {
            symbol: float(aligned_data[symbol]["open"].iloc[position])
            for symbol in symbols
        }
        closes = {
            symbol: float(aligned_data[symbol]["close"].iloc[position])
            for symbol in symbols
        }
        marked_equity = cash + sum(units[symbol] * opens[symbol] for symbol in symbols)
        trade_notional_total = 0.0
        cost_total = 0.0
        desired_total = 0.0
        executed_close_notional = 0.0
        trade_notional_by_symbol = {symbol: 0.0 for symbol in symbols}

        for symbol in symbols:
            desired_weight = float(aligned_targets[symbol].iloc[position - 1])
            desired_total += desired_weight
            current_weight = (
                units[symbol] * opens[symbol] / marked_equity if marked_equity else 0.0
            )
            trade_notional = 0.0
            fee = 0.0
            slippage_cost = 0.0

            if (
                marked_equity > 0
                and abs(desired_weight - current_weight) > config.min_trade_fraction
            ):
                buying = desired_weight > current_weight
                fill_price = opens[symbol] * (
                    1 + slippage_rate if buying else 1 - slippage_rate
                )
                target_units = desired_weight * marked_equity / fill_price
                delta_units = target_units - units[symbol]
                trade_notional = abs(delta_units) * fill_price
                fee = trade_notional * fee_rate
                slippage_cost = abs(delta_units) * abs(fill_price - opens[symbol])
                trade_notional_by_symbol[symbol] = trade_notional

                if delta_units >= 0:
                    if units[symbol] == 0:
                        new_average = fill_price
                    else:
                        new_average = (
                            average_entry[symbol] * units[symbol]
                            + fill_price * delta_units
                        ) / (units[symbol] + delta_units)
                    cash -= delta_units * fill_price + fee
                    units[symbol] += delta_units
                    average_entry[symbol] = new_average
                else:
                    sold_units = min(units[symbol], -delta_units)
                    gross_pnl = (fill_price - average_entry[symbol]) * sold_units
                    net_pnl = gross_pnl - fee
                    cash -= delta_units * fill_price + fee
                    units[symbol] += delta_units
                    entry_cost = average_entry[symbol] * sold_units
                    trade_rows.append(
                        {
                            "exit_time": timestamp,
                            "symbol": symbol,
                            "direction": "long",
                            "quantity": sold_units,
                            "entry_price": average_entry[symbol],
                            "exit_price": fill_price,
                            "gross_pnl": gross_pnl,
                            "net_pnl": net_pnl,
                            "return": net_pnl / entry_cost if entry_cost else 0.0,
                            "fees": fee,
                        }
                    )
                    if np.isclose(units[symbol], 0.0, atol=1e-12):
                        units[symbol] = 0.0
                        average_entry[symbol] = np.nan

            trade_notional_total += trade_notional
            cost_total += fee + slippage_cost
            executed_close_notional += units[symbol] * closes[symbol]

        closing_equity = cash + executed_close_notional
        benchmark_equity = sum(
            benchmark_units[symbol] * closes[symbol] for symbol in symbols
        )
        row: Dict[str, Any] = {
            "timestamp": timestamp,
            "equity": closing_equity,
            "benchmark_equity": benchmark_equity,
            "target_weight": desired_total,
            "executed_weight": executed_close_notional / closing_equity
            if closing_equity
            else 0.0,
            "trade_notional": trade_notional_total,
            "cost": cost_total,
            "turnover": trade_notional_total / marked_equity if marked_equity else 0.0,
            "bar_quote_volume": sum(
                float(aligned_data[symbol]["quote_volume"].iloc[position])
                for symbol in symbols
            ),
        }
        for symbol in symbols:
            row[f"trade_notional_{symbol}"] = trade_notional_by_symbol[symbol]
        for symbol in symbols:
            row[f"target_weight_{symbol}"] = float(
                aligned_targets[symbol].iloc[position - 1]
            )
            row[f"executed_weight_{symbol}"] = (
                units[symbol] * closes[symbol] / closing_equity if closing_equity else 0.0
            )
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("timestamp")
    trades = pd.DataFrame(trade_rows) if trade_rows else _empty_trades()
    result = BacktestResult(
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
        initial_timestamp=common_index[0],
        initial_equity=float(config.initial_capital),
        initial_benchmark_equity=float(config.initial_capital),
    )
    exposure_columns = [
        f"target_weight_{symbol}" for symbol in symbols
    ] + [f"executed_weight_{symbol}" for symbol in symbols]
    trade_notional_columns = [f"trade_notional_{symbol}" for symbol in symbols]
    diagnostics = frame[exposure_columns]
    diagnostics = pd.concat(
        [diagnostics, frame[trade_notional_columns]],
        axis=1,
    )
    return result, diagnostics


def _metrics_row(label: str, metrics: Dict[str, Any], status: str) -> Dict[str, Any]:
    return {
        "strategy": label,
        "annualized_return": metrics["annualized_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "sortino_ratio": metrics["sortino_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "annualized_volatility": metrics["annualized_volatility"],
        **{field: metrics[field] for field in STANDARD_RISK_RETURN_FIELDS},
        "turnover_annualized": metrics["turnover_annualized"],
        "cost_to_initial_capital": metrics["cost_to_initial_capital"],
        "trade_count": metrics["trade_count"],
        "p95_trade_participation": metrics["p95_trade_participation"],
        "execution_feasible": metrics["execution_feasible_at_research_size"],
        "status": status,
    }


def _period_metrics_row(
    label: str,
    period: str,
    metrics: Dict[str, Any],
    status: str = "not_applicable",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "strategy": label,
        "period": period,
        "status": status,
    }
    fields = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        *STANDARD_RISK_RETURN_FIELDS,
        "exposure",
        "turnover_annualized",
        "cost_to_initial_capital",
        "trade_count",
        "trade_win_rate",
        "payoff_ratio",
        "p95_trade_participation",
    ]
    row.update({field: metrics.get(field) for field in fields})
    return row


def _markdown_report(
    test_start: str,
    quality: Dict[str, Dict[str, Any]],
    overlap: Dict[str, Any],
    summary: pd.DataFrame,
) -> str:
    lines = [
        "# Donchian BTC/ETH Equal-Sleeve Portfolio",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Fixed out-of-sample start: **{test_start}**",
        "",
        "## Contract",
        "",
        "- Uses the already-declared Donchian 96/48 rule without parameter changes.",
        "- BTC and ETH are separate 50% sleeves sharing one cash account.",
        "- One active signal invests 50%; both active signals invest 100%.",
        "- Orders execute at the next common bar open.",
        "- Sleeve weights have a 5% no-trade band to avoid micro-rebalancing.",
        "- Costs are 10 bps commission plus 5 bps slippage per side.",
        "- The 50/50 buy-and-hold basket is the embedded benchmark.",
        "- This study tests diversification; it does not optimize allocations.",
        "",
        "## Data",
        "",
    ]
    for symbol, item in quality.items():
        lines.append(
            f"- **{symbol}**: {item['validation']['rows']} complete bars; "
            f"{item['validation']['missing_bars_by_span']} missing intervals."
        )
    lines.extend(
        [
            "",
            "## Signal Overlap",
            "",
            f"- Both sleeves active: **{overlap['both_active_share']:.1%}** of bars.",
            f"- Exactly one sleeve active: **{overlap['one_active_share']:.1%}**.",
            f"- No sleeve active: **{overlap['none_active_share']:.1%}**.",
            f"- Binary signal correlation: **{overlap['signal_correlation']:.3f}**.",
            "",
            "## Results",
            "",
            summary.to_markdown(index=False, floatfmt=".3f"),
            "",
            "Full, train, and out-of-sample rows are in `period_summary.csv`. ",
            "The displayed summary is full-period; status is based on the fixed ",
            "out-of-sample gate.",
            "",
            "`avg_standalone` is the arithmetic mean of full-capital BTC and ETH ",
            "runs and is a diagnostic comparator, not a directly tradable account.",
            "",
            "## Decision Discipline",
            "",
            "A better portfolio Sharpe than both standalone runs supports continued ",
            "research but does not establish live readiness. Fresh out-of-sample ",
            "confirmation, funding-aware hedging, capacity replay, margin stress, and ",
            "human approval remain required.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_donchian_portfolio_study(
    db_path: Path,
    output_root: Path,
    symbols: List[str],
    interval: str,
    start: str,
    test_start: str,
) -> Path:
    normalized_symbols = [symbol.upper() for symbol in symbols]
    if set(normalized_symbols) != set(SLEEVE_WEIGHTS):
        raise ValueError("current predeclared portfolio requires exactly BTCUSDT and ETHUSDT")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_donchian_btc_eth_portfolio"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    specs = strategy_registry()
    spec = specs[PORTFOLIO_STRATEGY]
    # A 50% sleeve drifts through a tiny threshold more often than a 100% single-asset position.
    config = BacktestConfig(min_trade_fraction=PORTFOLIO_NO_TRADE_BAND)
    split_timestamp = pd.Timestamp(test_start, tz="UTC")
    market_data: Dict[str, pd.DataFrame] = {}
    raw_targets: Dict[str, pd.Series] = {}
    quality: Dict[str, Dict[str, Any]] = {}

    for raw_symbol in normalized_symbols:
        symbol = raw_symbol.upper()
        data = load_klines(db_path, symbol, interval, start=start)
        validation = validate_klines(data, interval)
        quality[symbol] = {
            "interval": interval,
            "requested_start": start,
            "first_available": data.index[0].isoformat(),
            "last_available": data.index[-1].isoformat(),
            "validation": validation,
        }
        annualization = bars_per_year(data.index, interval_to_ms(interval) // 1000)
        params = dict(spec.parameters)
        params["bars_per_year"] = int(round(annualization))
        market_data[symbol] = data
        raw_targets[symbol] = spec.generate(data, params)

    result, exposures = run_portfolio_backtest(
        market_data=market_data,
        target_weights=raw_targets,
        sleeve_weights={symbol: SLEEVE_WEIGHTS[symbol.upper()] for symbol in market_data},
        config=config,
    )
    annualization = bars_per_year(result.equity.index, interval_to_ms(interval) // 1000)
    portfolio_full = performance_metrics(result, annualization)
    portfolio_train = performance_metrics(
        result,
        annualization,
        end=split_timestamp - pd.Timedelta(seconds=1),
    )
    portfolio_oos = performance_metrics(result, annualization, start=split_timestamp)
    portfolio_gate = risk_gate(portfolio_oos)

    standalone_metrics: Dict[str, Dict[str, Any]] = {}
    for symbol in sorted(market_data):
        standalone_result = run_backtest(
            market_data[symbol], raw_targets[symbol], config=config
        )
        standalone_metrics[symbol] = {
            "full": performance_metrics(standalone_result, annualization),
            "train": performance_metrics(
                standalone_result,
                annualization,
                end=split_timestamp - pd.Timedelta(seconds=1),
            ),
            "out_of_sample": performance_metrics(
                standalone_result, annualization, start=split_timestamp
            ),
        }

    common_index = result.weights.index
    binary_signals = pd.DataFrame(
        {
            symbol: raw_targets[symbol].reindex(common_index).fillna(0.0).gt(0.001).astype(int)
            for symbol in sorted(market_data)
        },
        index=common_index,
    )
    active_count = binary_signals.sum(axis=1)
    signal_correlation = float(binary_signals.corr().iloc[0, 1])
    overlap = {
        "both_active_share": float(active_count.eq(len(binary_signals.columns)).mean()),
        "one_active_share": float(active_count.eq(1).mean()),
        "none_active_share": float(active_count.eq(0).mean()),
        "signal_correlation": signal_correlation,
    }

    rows = [
        _metrics_row(
            "portfolio_50_50",
            portfolio_full,
            portfolio_gate["status"],
        ),
        _metrics_row(
            "BTCUSDT_standalone",
            standalone_metrics["BTCUSDT"]["full"],
            risk_gate(standalone_metrics["BTCUSDT"]["out_of_sample"])["status"],
        ),
        _metrics_row(
            "ETHUSDT_standalone",
            standalone_metrics["ETHUSDT"]["full"],
            risk_gate(standalone_metrics["ETHUSDT"]["out_of_sample"])["status"],
        ),
    ]
    average_standalone = {
        field: float(
            np.mean([standalone_metrics[symbol]["full"][field] for symbol in sorted(market_data)])
        )
        for field in [
            "annualized_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "annualized_volatility",
            "turnover_annualized",
            "cost_to_initial_capital",
            "trade_count",
            "p95_trade_participation",
            *STANDARD_RISK_RETURN_FIELDS,
        ]
    }
    average_standalone["execution_feasible_at_research_size"] = all(
        standalone_metrics[symbol]["full"]["execution_feasible_at_research_size"]
        for symbol in sorted(market_data)
    )
    rows.append(_metrics_row("avg_standalone", average_standalone, "diagnostic"))
    summary = pd.DataFrame(rows)

    standalone_statuses = {
        symbol: risk_gate(standalone_metrics[symbol]["out_of_sample"])["status"]
        for symbol in sorted(market_data)
    }
    average_for_period: Dict[str, Dict[str, Any]] = {}
    for period in ("full", "train", "out_of_sample"):
        average_for_period[period] = {
            field: float(
                np.mean(
                    [standalone_metrics[symbol][period][field] for symbol in sorted(market_data)]
                )
            )
            for field in [
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown",
                "calmar_ratio",
                "exposure",
                "turnover_annualized",
                "cost_to_initial_capital",
                "trade_count",
                "trade_win_rate",
                "payoff_ratio",
                "p95_trade_participation",
            ]
        }

    period_rows = []
    portfolio_periods = {
        "full": portfolio_full,
        "train": portfolio_train,
        "out_of_sample": portfolio_oos,
    }
    for period, metrics in portfolio_periods.items():
        status = portfolio_gate["status"] if period == "out_of_sample" else "not_applicable"
        period_rows.append(_period_metrics_row("portfolio_50_50", period, metrics, status))
    for symbol in sorted(market_data):
        for period in ("full", "train", "out_of_sample"):
            status = (
                standalone_statuses[symbol]
                if period == "out_of_sample"
                else "not_applicable"
            )
            period_rows.append(
                _period_metrics_row(
                    f"{symbol}_standalone",
                    period,
                    standalone_metrics[symbol][period],
                    status,
                )
            )
    for period in ("full", "train", "out_of_sample"):
        period_rows.append(
            _period_metrics_row("avg_standalone", period, average_for_period[period])
        )
    period_summary = pd.DataFrame(period_rows)

    pd.DataFrame(
        {
            "equity": result.equity,
            "benchmark_equity": result.benchmark_equity,
            "target_weight": result.weights["target_weight"],
            "executed_weight": result.weights["executed_weight"],
            "turnover": result.weights["turnover"],
            **exposures,
        }
    ).to_csv(run_dir / "portfolio_equity.csv")
    result.trades.to_csv(run_dir / "portfolio_trades.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    period_summary.to_csv(run_dir / "period_summary.csv", index=False)

    source_manifest = build_source_manifest()
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": "donchian_btc_eth_equal_sleeve_portfolio",
        "strategy": PORTFOLIO_STRATEGY,
        "symbols": normalized_symbols,
        "interval": interval,
        "requested_start": start,
        "fixed_test_start": test_start,
        "config": config.__dict__,
        "no_trade_band": PORTFOLIO_NO_TRADE_BAND,
        "sleeve_weights": SLEEVE_WEIGHTS,
        "risk_gates": RISK_GATES,
        "data_quality": quality,
        "signal_overlap": overlap,
        "portfolio_risk_gate": portfolio_gate,
        "summary": summary.to_dict(orient="records"),
        "period_summary": period_summary.to_dict(orient="records"),
        "database_sha256": file_hash(db_path),
        "source_provenance": source_manifest,
    }
    with (run_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(_sanitize_json(payload), handle, indent=2, allow_nan=False)

    markdown = _markdown_report(test_start, quality, overlap, summary)
    report_path = run_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")

    ledger_record = {
        "run_id": run_id,
        "created_at": payload["created_at"],
        "study": payload["study"],
        "database_sha256": payload["database_sha256"],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "symbols": normalized_symbols,
        "interval": interval,
        "fixed_test_start": test_start,
        "portfolio_oos_status": portfolio_gate["status"],
        "report_path": str(report_path),
    }
    with (output_root / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_sanitize_json(ledger_record)) + "\n")
    return run_dir
