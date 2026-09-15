"""Cross-sectional relative-strength rotation research.

This module is intentionally separate from the single-asset strategy registry:
the signal needs the contemporaneous BTC/ETH cross-section.  It is a research
adapter, not an exchange or execution client.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from crypto_quant.backtesting.backtest import BacktestResult, bars_per_year
from crypto_quant.backtesting.config import BacktestConfig
from crypto_quant.data_access.data import interval_to_ms, load_klines, validate_klines
from crypto_quant.backtesting.metrics import performance_metrics
from crypto_quant.backtesting.multiple_testing import deflated_sharpe_ratio
from crypto_quant.research.provenance import build_source_manifest, spot_klines_digest
from crypto_quant.research.reporting import _sanitize_json, file_hash
from crypto_quant.execution.strategy_platform import MetricInputs, score_metrics
from crypto_quant.backtesting.validation import risk_gate


STRATEGY_NAME = "relative_strength_rotation_168_08"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
BASE_LOOKBACK = 168
BASE_SPREAD = 0.08
BASE_FEE_BPS = 10.0
BASE_SLIPPAGE_BPS = 5.0
BASE_NO_TRADE_BAND = 0.001
OOS_START = "2023-01-01"


def _common_index(market_data: Mapping[str, pd.DataFrame]) -> pd.DatetimeIndex:
    if not market_data:
        raise ValueError("market_data cannot be empty")
    symbols = sorted(market_data)
    common = market_data[symbols[0]].index
    for symbol in symbols[1:]:
        common = common.intersection(market_data[symbol].index)
    common = common.sort_values()
    if len(common) < 3:
        raise ValueError("relative-strength study needs at least three common bars")
    return common


def generate_relative_strength_targets(
    market_data: Mapping[str, pd.DataFrame],
    *,
    lookback: int = BASE_LOOKBACK,
    spread_threshold: float = BASE_SPREAD,
    min_winner_return: float = 0.0,
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """Generate causal long-only targets and signal diagnostics.

    Returns at timestamp ``t`` use only closes at ``t`` and ``t-lookback``.
    The backtester executes the target at the next bar open.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if spread_threshold < 0:
        raise ValueError("spread_threshold cannot be negative")
    symbols = sorted(market_data)
    if symbols != sorted(SYMBOLS):
        raise ValueError(f"relative-strength universe must be {sorted(SYMBOLS)}")
    common = _common_index(market_data)
    closes = {
        symbol: market_data[symbol].loc[common, "close"].astype(float)
        for symbol in symbols
    }
    returns = pd.DataFrame(
        {symbol: series.pct_change(lookback) for symbol, series in closes.items()},
        index=common,
    )
    spread = returns[SYMBOLS[0]] - returns[SYMBOLS[1]]
    winner_return = returns.max(axis=1)
    btc_winner = spread > 0.0
    eth_winner = spread < 0.0
    active = (
        spread.abs().ge(float(spread_threshold))
        & winner_return.gt(float(min_winner_return))
        & returns.notna().all(axis=1)
    )
    targets = {
        symbol: pd.Series(0.0, index=common, name=f"target_{symbol}")
        for symbol in symbols
    }
    targets[SYMBOLS[0]].loc[active & btc_winner] = 1.0
    targets[SYMBOLS[1]].loc[active & eth_winner] = 1.0
    diagnostics = pd.DataFrame(
        {
            "btc_return": returns[SYMBOLS[0]],
            "eth_return": returns[SYMBOLS[1]],
            "spread": spread,
            "winner_return": winner_return,
            "active": active.astype(bool),
            "winner": np.where(
                active & btc_winner,
                SYMBOLS[0],
                np.where(active & eth_winner, SYMBOLS[1], "CASH"),
            ),
        },
        index=common,
    )
    return targets, diagnostics


def _record_trade(
    trade_rows: List[Dict[str, Any]],
    *,
    timestamp: pd.Timestamp,
    symbol: str,
    quantity: float,
    entry_price: float,
    exit_price: float,
    entry_fee: float,
    exit_fee: float,
) -> None:
    gross_pnl = (exit_price - entry_price) * quantity
    fee = entry_fee + exit_fee
    net_pnl = gross_pnl - fee
    entry_notional = entry_price * quantity
    trade_rows.append(
        {
            "exit_time": timestamp,
            "symbol": symbol,
            "direction": "long",
            "quantity": quantity,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "return": net_pnl / entry_notional if entry_notional else 0.0,
            "fees": fee,
        }
    )


def run_relative_strength_backtest(
    market_data: Mapping[str, pd.DataFrame],
    target_weights: Mapping[str, pd.Series],
    *,
    config: Optional[BacktestConfig] = None,
    latency_bars: int = 0,
    max_participation: Optional[float] = None,
) -> BacktestResult:
    """Backtest one shared cash account with causal next-open fills.

    The optional participation cap creates partial fills.  Sells are processed
    before buys so a rotation cannot spend proceeds that have not been raised.
    """
    config = config or BacktestConfig()
    if config.allow_short:
        raise ValueError("relative-strength spot backtest is long-only")
    if latency_bars < 0:
        raise ValueError("latency_bars cannot be negative")
    if max_participation is not None and not (0.0 < max_participation <= 1.0):
        raise ValueError("max_participation must be in (0, 1]")
    symbols = sorted(market_data)
    if symbols != sorted(SYMBOLS) or set(target_weights) != set(symbols):
        raise ValueError("market data and targets must contain BTCUSDT and ETHUSDT")
    common = _common_index(market_data)
    data = {symbol: market_data[symbol].loc[common] for symbol in symbols}
    targets = {
        symbol: target_weights[symbol]
        .reindex(common)
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0, upper=1.0)
        for symbol in symbols
    }
    if any(not np.isfinite(series).all() for series in targets.values()):
        raise ValueError("targets must be finite")
    if any((sum(targets[symbol].iloc[pos] for symbol in symbols) > 1.0 + 1e-12) for pos in range(len(common))):
        raise ValueError("target weights cannot exceed 100% gross exposure")

    fee_rate = float(config.fee_bps) / 10_000.0
    slippage_rate = float(config.slippage_bps) / 10_000.0
    cash = float(config.initial_capital)
    units = {symbol: 0.0 for symbol in symbols}
    average_entry = {symbol: np.nan for symbol in symbols}
    entry_fees = {symbol: 0.0 for symbol in symbols}
    benchmark_units = {
        symbol: float(config.initial_capital) * 0.5 / float(data[symbol]["open"].iloc[1])
        for symbol in symbols
    }
    rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    min_trade_fraction = float(config.min_trade_fraction)

    for position in range(1, len(common)):
        timestamp = common[position]
        opens = {symbol: float(data[symbol]["open"].iloc[position]) for symbol in symbols}
        closes = {symbol: float(data[symbol]["close"].iloc[position]) for symbol in symbols}
        quote_volumes = {symbol: float(data[symbol]["quote_volume"].iloc[position]) for symbol in symbols}
        marked_equity = cash + sum(units[symbol] * opens[symbol] for symbol in symbols)
        if not np.isfinite(marked_equity) or marked_equity <= 0:
            raise ValueError("account became insolvent before execution")

        signal_pos = position - 1 - int(latency_bars)
        desired = {
            symbol: (float(targets[symbol].iloc[signal_pos]) if signal_pos >= 0 else 0.0)
            for symbol in symbols
        }
        desired_total = sum(desired.values())
        requested_total = 0.0
        filled_total = 0.0
        filled_by_symbol = {symbol: 0.0 for symbol in symbols}
        fee_total = 0.0
        slippage_total = 0.0

        def participation_limit(symbol: str) -> float:
            if max_participation is None:
                return float("inf")
            return max_participation * quote_volumes[symbol]

        # Close/reduce before opening/increasing the other sleeve.
        for symbol in symbols:
            current_notional = units[symbol] * opens[symbol]
            desired_notional = desired[symbol] * marked_equity
            delta = desired_notional - current_notional
            if delta >= -min_trade_fraction * marked_equity or units[symbol] <= 0:
                continue
            requested = min(-delta, current_notional)
            requested_total += requested
            fill_notional = min(requested, participation_limit(symbol))
            if fill_notional <= 0:
                continue
            quantity = min(units[symbol], fill_notional / (opens[symbol] * (1.0 - slippage_rate)))
            fill_price = opens[symbol] * (1.0 - slippage_rate)
            notional = quantity * fill_price
            fee = notional * fee_rate
            slippage_cost = quantity * abs(fill_price - opens[symbol])
            entry_price = (
                float(average_entry[symbol])
                if np.isfinite(average_entry[symbol])
                else fill_price
            )
            units_before = units[symbol]
            allocated_entry_fee = (
                entry_fees[symbol] * quantity / units_before
                if units_before > 0
                else 0.0
            )
            cash += notional - fee
            units[symbol] -= quantity
            entry_fees[symbol] = max(
                0.0, entry_fees[symbol] - allocated_entry_fee
            )
            if units[symbol] <= 1e-12:
                units[symbol] = 0.0
                average_entry[symbol] = np.nan
                entry_fees[symbol] = 0.0
            _record_trade(
                trade_rows,
                timestamp=timestamp,
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                exit_price=fill_price,
                entry_fee=allocated_entry_fee,
                exit_fee=fee,
            )
            filled_total += notional
            filled_by_symbol[symbol] += notional
            fee_total += fee
            slippage_total += slippage_cost

        # Open/increase target sleeves only after sells and fees have settled.
        for symbol in symbols:
            current_notional = units[symbol] * opens[symbol]
            desired_notional = desired[symbol] * marked_equity
            delta = desired_notional - current_notional
            if delta <= min_trade_fraction * marked_equity:
                continue
            requested = delta
            requested_total += requested
            fill_price = opens[symbol] * (1.0 + slippage_rate)
            affordable = cash / (1.0 + fee_rate) if cash > 0 else 0.0
            fill_notional = min(requested, participation_limit(symbol), affordable)
            if fill_notional <= 0:
                continue
            quantity = fill_notional / fill_price
            notional = quantity * fill_price
            fee = notional * fee_rate
            slippage_cost = quantity * abs(fill_price - opens[symbol])
            cash -= notional + fee
            entry_fees[symbol] += fee
            old_units = units[symbol]
            units[symbol] += quantity
            average_entry[symbol] = (
                fill_price
                if old_units <= 1e-12
                else (average_entry[symbol] * old_units + fill_price * quantity)
                / (old_units + quantity)
            )
            filled_total += notional
            filled_by_symbol[symbol] += notional
            fee_total += fee
            slippage_total += slippage_cost

        closing_equity = cash + sum(units[symbol] * closes[symbol] for symbol in symbols)
        if not np.isfinite(closing_equity) or closing_equity <= 0 or cash < -1e-8:
            raise ValueError("accounting invariant failed: negative cash or equity")
        total_quote_volume = sum(quote_volumes.values())
        symbol_participations = [
            (
                filled_by_symbol[symbol] / quote_volumes[symbol]
                if quote_volumes[symbol] > 0
                else 0.0
            )
            for symbol in symbols
        ]
        max_symbol_participation = max(symbol_participations, default=0.0)
        # performance_metrics derives participation as trade_notional divided by
        # bar_quote_volume.  Use an equivalent denominator for the most heavily
        # used symbol rather than diluting impact with the other symbol's volume.
        participation_denominator = (
            filled_total / max_symbol_participation
            if filled_total > 0 and max_symbol_participation > 0
            else total_quote_volume
        )
        benchmark_equity = sum(
            benchmark_units[symbol] * closes[symbol] for symbol in symbols
        )
        rows.append(
            {
                "timestamp": timestamp,
                "equity": closing_equity,
                "benchmark_equity": benchmark_equity,
                "target_weight": desired_total,
                "executed_weight": (
                    sum(units[symbol] * closes[symbol] for symbol in symbols)
                    / closing_equity
                    if closing_equity
                    else 0.0
                ),
                "trade_notional": filled_total,
                "cost": fee_total + slippage_total,
                "fee": fee_total,
                "slippage_cost": slippage_total,
                "turnover": filled_total / marked_equity if marked_equity else 0.0,
                "bar_quote_volume": participation_denominator,
                "fill_ratio": filled_total / requested_total if requested_total > 0 else 1.0,
                "cash": cash,
                **{f"target_weight_{symbol}": desired[symbol] for symbol in symbols},
                **{f"position_{symbol}": units[symbol] for symbol in symbols},
            }
        )

    frame = pd.DataFrame(rows).set_index("timestamp")
    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
                "exit_time", "symbol", "direction", "quantity", "entry_price",
                "exit_price", "gross_pnl", "net_pnl", "return", "fees",
            ]
        )
    return BacktestResult(
        equity=frame["equity"],
        benchmark_equity=frame["benchmark_equity"],
        weights=frame[
            [
                "target_weight", "executed_weight", "turnover", "trade_notional",
                "cost", "bar_quote_volume", "fee", "slippage_cost", "fill_ratio",
            ]
        ],
        trades=trades,
        config=config,
        initial_timestamp=common[0],
        initial_equity=float(config.initial_capital),
        initial_benchmark_equity=float(config.initial_capital),
    )


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def hac_sharpe(
    result: BacktestResult,
    bars_per_year_value: float,
    *,
    start: Optional[pd.Timestamp] = None,
    max_lag: int = 20,
) -> float:
    """Newey-West-style annualized Sharpe for serially correlated 4h returns."""
    series = (
        _path_returns(result, start)
        if start is not None
        else result.equity.pct_change().dropna()
    )
    returns = series.to_numpy(dtype=float)
    if len(returns) < 3 or not np.isfinite(returns).all():
        return 0.0
    mean = float(returns.mean())
    centered = returns - mean
    lag_count = min(int(max_lag), len(returns) - 1)
    long_run_variance = float(np.mean(centered * centered))
    for lag in range(1, lag_count + 1):
        covariance = float(np.mean(centered[lag:] * centered[:-lag]))
        weight = 1.0 - lag / (lag_count + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0.0:
        return 0.0
    return float(mean / math.sqrt(long_run_variance) * math.sqrt(bars_per_year_value))


def _path_returns(result: BacktestResult, start: pd.Timestamp) -> pd.Series:
    selected = result.equity.loc[start:]
    if selected.empty:
        raise ValueError("no equity observations after requested start")
    prior = result.equity.loc[result.equity.index < selected.index[0]]
    if prior.empty:
        return selected.pct_change().dropna()
    path = pd.concat([prior.iloc[[-1]], selected])
    return path.pct_change().dropna()


def _walk_forward_rows(
    result: BacktestResult,
    annualization: float,
    *,
    folds: int = 5,
) -> List[Dict[str, Any]]:
    positions = np.array_split(np.arange(len(result.equity)), folds)
    output: List[Dict[str, Any]] = []
    for number, values in enumerate(positions, start=1):
        if len(values) < 2:
            continue
        start = result.equity.index[int(values[0])]
        end = result.equity.index[int(values[-1])]
        metrics = performance_metrics(result, annualization, start=start, end=end)
        output.append(
            {
                "fold": f"fold_{number}",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "bars": int(len(values)),
                "total_return": _finite(metrics["total_return"]),
                "annualized_return": _finite(metrics["annualized_return"]),
                "sharpe_ratio": _finite(metrics["sharpe_ratio"]),
                "max_drawdown": _finite(metrics["max_drawdown"]),
                "trade_count": int(metrics["trade_count"]),
                "positive_return": bool(metrics["total_return"] > 0.0),
            }
        )
    return output


def _causal_prefix_audit(
    market_data: Mapping[str, pd.DataFrame],
    full_targets: Mapping[str, pd.Series],
    *,
    lookback: int,
    spread_threshold: float,
) -> Tuple[bool, List[Dict[str, Any]]]:
    common = _common_index(market_data)
    positions = np.array_split(np.arange(len(common)), 5)
    rows: List[Dict[str, Any]] = []
    for number, values in enumerate(positions, start=1):
        if not len(values):
            continue
        prefix_end = common[int(values[-1])]
        prefix_data = {
            symbol: market_data[symbol].loc[:prefix_end]
            for symbol in sorted(market_data)
        }
        prefix_targets, _ = generate_relative_strength_targets(
            prefix_data,
            lookback=lookback,
            spread_threshold=spread_threshold,
        )
        matches = True
        compared = 0
        mismatch: List[str] = []
        for symbol in sorted(full_targets):
            common_prefix = full_targets[symbol].index.intersection(prefix_targets[symbol].index)
            actual = full_targets[symbol].loc[common_prefix].to_numpy(dtype=float)
            reproduced = prefix_targets[symbol].loc[common_prefix].to_numpy(dtype=float)
            same = bool(np.allclose(actual, reproduced, atol=1e-12, rtol=0.0, equal_nan=True))
            matches = matches and same
            compared += len(common_prefix)
            if not same:
                bad = ~np.isclose(actual, reproduced, atol=1e-12, rtol=0.0, equal_nan=True)
                mismatch.extend(timestamp.isoformat() for timestamp in common_prefix[bad][:3])
        rows.append(
            {
                "fold": f"prefix_{number}",
                "prefix_end": prefix_end.isoformat(),
                "compared_points": int(compared),
                "matches_full_targets": bool(matches),
                "mismatch_examples": mismatch[:5],
            }
        )
    return bool(rows) and all(row["matches_full_targets"] for row in rows), rows


def _risk_status(metrics: Dict[str, Any]) -> Dict[str, Any]:
    safe = dict(metrics)
    for key in ("sharpe_ratio", "max_drawdown", "p95_trade_participation", "total_return", "trade_count"):
        safe.setdefault(key, 0.0)
    return risk_gate(safe)


def _parameter_variants() -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = [
        {
            "variant_id": "declared",
            "lookback": BASE_LOOKBACK,
            "spread_threshold": BASE_SPREAD,
            "is_declared": True,
        }
    ]
    for lookback in (120, 144, 192, 216):
        variants.append(
            {
                "variant_id": f"lookback_{lookback}",
                "lookback": lookback,
                "spread_threshold": BASE_SPREAD,
                "is_declared": False,
            }
        )
    for spread in (0.05, 0.06, 0.10, 0.12):
        variants.append(
            {
                "variant_id": f"spread_{int(spread * 100):02d}",
                "lookback": BASE_LOOKBACK,
                "spread_threshold": spread,
                "is_declared": False,
            }
        )
    return variants


def _cost_variants() -> List[Tuple[float, float, int]]:
    return [
        (fee, slippage, latency)
        for fee, slippage in ((10.0, 5.0), (15.0, 7.5), (20.0, 10.0), (25.0, 12.5), (30.0, 15.0))
        for latency in (0, 1)
        if not (fee == BASE_FEE_BPS and slippage == BASE_SLIPPAGE_BPS and latency == 0)
    ]


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_sanitize_json(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _markdown_report(payload: Dict[str, Any], trial_table: pd.DataFrame, folds: pd.DataFrame) -> str:
    base = payload["base_result"]
    score = payload["score_inputs"]
    lines = [
        "# Relative-Strength Rotation v1",
        "",
        f"Generated: `{payload['created_at']}`",
        f"Fixed out-of-sample start: **{payload['test_start']}**",
        "",
        "## Frozen contract",
        "",
        "- BTCUSDT/ETHUSDT Binance Spot, common complete 4h timestamps only.",
        "- 168-bar close-to-close relative return; trade the positive leader only when the spread is at least 8 percentage points.",
        "- Signal at close; next-bar-open execution; long-only, unlevered, shared cash account.",
        "- Base cost: 10 bps fee plus 5 bps slippage per side. No funding leg exists.",
        "- No parameter was changed after inspecting the declared result.",
        "",
        "## Evidence limits",
        "",
        "- This is a two-asset relative-strength experiment, not evidence for the full crypto cross-section.",
        "- The absence of a perpetual leg means funding is not a strategy input; no funding alpha is claimed.",
        "- Historical evidence never authorizes live capital, Demo/Testnet orders, or account access.",
        "",
        "## Base historical result",
        "",
        pd.DataFrame([base]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Fixed walk-forward slices",
        "",
        folds.to_markdown(index=False, floatfmt=".6f") if not folds.empty else "No fold rows.",
        "",
        "## Trial audit",
        "",
        f"All **{len(trial_table)}** declared, parameter, cost/latency, and liquidity-cap trials are retained.",
        "",
        trial_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Platform metric inputs",
        "",
        json.dumps(score, indent=2, ensure_ascii=False),
        "",
        "## Decision",
        "",
        f"- OOS risk-gate status: **`{payload['oos_risk_status']}`**.",
        f"- DSR: **{payload['dsr']['deflated_sharpe_probability']:.6f}**; trial count: **{payload['trial_count']}**.",
        f"- Hard-gate failures: `{payload['hard_gate_failures']}`.",
        "- Platform score is generated separately from these structured inputs; no score component is manually entered.",
        "",
        "## Sources and provenance",
        "",
        f"- Ideation and frozen contract: `docs/strategy_ideation/20260824_relative_strength_momentum.md`.",
        f"- Database SHA-256: `{payload['database_sha256']}`.",
        f"- Source-manifest SHA-256: `{payload['source_provenance']['manifest_sha256']}`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_cross_sectional_study(
    db_path: Path,
    output_root: Path,
    symbols: Iterable[str] = SYMBOLS,
    interval: str = "4h",
    start: str = "2019-01-01",
    test_start: str = OOS_START,
    code_tests_passed: bool = False,
) -> Path:
    """Run the frozen v1 study and write all evidence artifacts."""
    normalized = tuple(symbol.upper() for symbol in symbols)
    if normalized != SYMBOLS:
        raise ValueError(f"this frozen study requires symbols {SYMBOLS}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_relative_strength_rotation_study"
    run_dir = Path(output_root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_data: Dict[str, pd.DataFrame] = {}
    quality: Dict[str, Any] = {}
    for symbol in normalized:
        data = load_klines(db_path, symbol, interval=interval, start=start)
        validation = validate_klines(data, interval)
        raw_data[symbol] = data
        quality[symbol] = {
            "interval": interval,
            "requested_start": start,
            "first_available": data.index[0].isoformat(),
            "last_available": data.index[-1].isoformat(),
            "rows": int(len(data)),
            "validation": validation,
        }
    common = _common_index(raw_data)
    market_data = {symbol: raw_data[symbol].loc[common] for symbol in normalized}
    annualization = bars_per_year(common, interval_to_ms(interval) // 1000)
    split_timestamp = pd.Timestamp(test_start, tz="UTC")

    base_targets, signal_diagnostics = generate_relative_strength_targets(
        market_data, lookback=BASE_LOOKBACK, spread_threshold=BASE_SPREAD
    )
    base_config = BacktestConfig(
        fee_bps=BASE_FEE_BPS,
        slippage_bps=BASE_SLIPPAGE_BPS,
        min_trade_fraction=BASE_NO_TRADE_BAND,
    )
    base_result = run_relative_strength_backtest(market_data, base_targets, config=base_config)
    full_metrics = performance_metrics(base_result, annualization)
    train_metrics = performance_metrics(
        base_result, annualization, end=split_timestamp - pd.Timedelta(seconds=1)
    )
    oos_metrics = performance_metrics(base_result, annualization, start=split_timestamp)
    oos_risk = _risk_status(oos_metrics)
    fold_rows = _walk_forward_rows(base_result, annualization)
    prefix_passed, prefix_rows = _causal_prefix_audit(
        market_data,
        base_targets,
        lookback=BASE_LOOKBACK,
        spread_threshold=BASE_SPREAD,
    )

    trials: List[Dict[str, Any]] = []
    parameter_rows: List[Dict[str, Any]] = []
    for variant in _parameter_variants():
        targets, _ = generate_relative_strength_targets(
            market_data,
            lookback=int(variant["lookback"]),
            spread_threshold=float(variant["spread_threshold"]),
        )
        result = run_relative_strength_backtest(market_data, targets, config=base_config)
        metrics = performance_metrics(result, annualization, start=split_timestamp)
        risk = _risk_status(metrics)
        row = {
            "trial_id": variant["variant_id"],
            "trial_kind": "parameter_neighborhood",
            "is_declared": bool(variant["is_declared"]),
            "lookback": int(variant["lookback"]),
            "spread_threshold": float(variant["spread_threshold"]),
            "fee_bps": BASE_FEE_BPS,
            "slippage_bps": BASE_SLIPPAGE_BPS,
            "latency_bars": 0,
            "max_participation": None,
            "oos_total_return": _finite(metrics["total_return"]),
            "oos_sharpe": _finite(metrics["sharpe_ratio"]),
            "oos_max_drawdown": _finite(metrics["max_drawdown"]),
            "oos_trade_count": int(metrics["trade_count"]),
            "oos_p95_participation": _finite(metrics["p95_trade_participation"]),
            "oos_risk_status": risk["status"],
            "risk_checks": risk["checks"],
        }
        trials.append(row)
        parameter_rows.append(row)

    cost_rows: List[Dict[str, Any]] = []
    for fee_bps, slippage_bps, latency in _cost_variants():
        config = BacktestConfig(
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            min_trade_fraction=BASE_NO_TRADE_BAND,
        )
        result = run_relative_strength_backtest(
            market_data, base_targets, config=config, latency_bars=latency
        )
        metrics = performance_metrics(result, annualization, start=split_timestamp)
        risk = _risk_status(metrics)
        row = {
            "trial_id": f"cost_{int(fee_bps * 10):03d}_{int(slippage_bps * 10):03d}_latency_{latency}",
            "trial_kind": "cost_latency_stress",
            "is_declared": False,
            "lookback": BASE_LOOKBACK,
            "spread_threshold": BASE_SPREAD,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "latency_bars": latency,
            "max_participation": None,
            "oos_total_return": _finite(metrics["total_return"]),
            "oos_sharpe": _finite(metrics["sharpe_ratio"]),
            "oos_max_drawdown": _finite(metrics["max_drawdown"]),
            "oos_trade_count": int(metrics["trade_count"]),
            "oos_p95_participation": _finite(metrics["p95_trade_participation"]),
            "oos_risk_status": risk["status"],
            "risk_checks": risk["checks"],
        }
        trials.append(row)
        cost_rows.append(row)

    liquidity_rows: List[Dict[str, Any]] = []
    capacity_results: Dict[float, BacktestResult] = {}
    for cap in (0.01, 0.005, 0.002):
        result = run_relative_strength_backtest(
            market_data,
            base_targets,
            config=base_config,
            max_participation=cap,
        )
        metrics = performance_metrics(result, annualization, start=split_timestamp)
        risk = _risk_status(metrics)
        fill_ratio = float(result.weights.loc[split_timestamp:, "fill_ratio"].mean())
        row = {
            "trial_id": f"liquidity_cap_{cap:g}",
            "trial_kind": "liquidity_partial_fill_stress",
            "is_declared": False,
            "lookback": BASE_LOOKBACK,
            "spread_threshold": BASE_SPREAD,
            "fee_bps": BASE_FEE_BPS,
            "slippage_bps": BASE_SLIPPAGE_BPS,
            "latency_bars": 0,
            "max_participation": cap,
            "oos_total_return": _finite(metrics["total_return"]),
            "oos_sharpe": _finite(metrics["sharpe_ratio"]),
            "oos_max_drawdown": _finite(metrics["max_drawdown"]),
            "oos_trade_count": int(metrics["trade_count"]),
            "oos_p95_participation": _finite(metrics["p95_trade_participation"]),
            "mean_fill_ratio": fill_ratio,
            "oos_risk_status": risk["status"],
            "risk_checks": risk["checks"],
        }
        trials.append(row)
        liquidity_rows.append(row)
        capacity_results[cap] = result

    trial_table = pd.DataFrame(trials)
    trial_sharpes = trial_table["oos_sharpe"].to_numpy(dtype=float)
    dsr = deflated_sharpe_ratio(
        _path_returns(base_result, split_timestamp).to_numpy(dtype=float),
        trial_sharpes,
        annualization,
    )
    folds = pd.DataFrame(fold_rows)
    wf_positive_share = float(folds["positive_return"].mean()) if not folds.empty else 0.0
    neighborhood_pass = float(
        pd.Series([row["oos_risk_status"] == "candidate" for row in parameter_rows]).mean()
    )
    stress_pass = float(
        pd.Series(
            [row["oos_risk_status"] == "candidate" for row in cost_rows + liquidity_rows]
        ).mean()
    )
    doubled_cost = next(
        row["oos_total_return"]
        for row in cost_rows
        if row["fee_bps"] == 20.0 and row["slippage_bps"] == 10.0 and row["latency_bars"] == 0
    )
    cap_result = capacity_results[0.002]
    replay_fill_ratio = float(cap_result.weights.loc[split_timestamp:, "fill_ratio"].mean())
    trade_notional = float(base_result.weights.loc[split_timestamp:, "trade_notional"].sum())
    slippage_cost = float(base_result.weights.loc[split_timestamp:, "slippage_cost"].sum())
    slippage_budget_ratio = slippage_cost / (trade_notional * 0.001) if trade_notional > 0 else 0.0
    strategy_returns = _path_returns(base_result, split_timestamp)
    benchmark_returns = _path_returns(
        BacktestResult(
            equity=base_result.benchmark_equity,
            benchmark_equity=base_result.benchmark_equity,
            weights=base_result.weights,
            trades=base_result.trades,
            config=base_result.config,
            initial_timestamp=base_result.initial_timestamp,
            initial_equity=base_result.initial_equity,
            initial_benchmark_equity=base_result.initial_benchmark_equity,
        ),
        split_timestamp,
    )
    correlation = float(strategy_returns.corr(benchmark_returns)) if len(strategy_returns) > 2 else 0.0
    benchmark_oos = performance_metrics(
        BacktestResult(
            equity=base_result.benchmark_equity,
            benchmark_equity=base_result.benchmark_equity,
            weights=base_result.weights,
            trades=base_result.trades,
            config=base_result.config,
            initial_timestamp=base_result.initial_timestamp,
            initial_equity=base_result.initial_equity,
            initial_benchmark_equity=base_result.initial_benchmark_equity,
        ),
        annualization,
        start=split_timestamp,
    )
    mdd_improvement = max(0.0, abs(benchmark_oos["max_drawdown"]) - abs(oos_metrics["max_drawdown"]))
    independent_cycles = float(oos_metrics["trade_count"])
    oos_hac_sharpe = hac_sharpe(
        base_result,
        annualization,
        start=split_timestamp,
    )
    hard_failures: List[str] = []
    hard_checks = [
        (True, "data_integrity"),
        (prefix_passed, "causal_timing"),
        (bool(code_tests_passed), "code_tests"),
        (oos_metrics["total_return"] > 0, "oos_total_return"),
        (oos_metrics["years"] >= 1.0, "oos_duration"),
        (independent_cycles >= 20, "independent_cycles"),
        (oos_hac_sharpe >= 0.30, "hac_sharpe"),
        (abs(oos_metrics["max_drawdown"]) <= 0.50, "max_drawdown"),
        (doubled_cost > 0, "doubled_cost_return"),
        (oos_metrics["p95_trade_participation"] <= 0.01, "p95_participation"),
        (base_result.equity.min() > 0, "insolvency"),
        (dsr["deflated_sharpe_probability"] >= 0.80, "dsr"),
    ]
    hard_failures.extend(name for passed, name in hard_checks if not passed)
    metrics_inputs = MetricInputs(
        cagr=_finite(oos_metrics["annualized_return"], -1.0),
        profit_factor=_finite(oos_metrics["profit_factor"]),
        positive_month_share=_finite(oos_metrics["monthly_return_positive_share"]),
        hac_sharpe=_finite(oos_hac_sharpe),
        sortino=_finite(oos_metrics["sortino_ratio"]),
        calmar=_finite(oos_metrics["calmar_ratio"]),
        max_drawdown=_finite(oos_metrics["max_drawdown"], -1.0),
        dsr=_finite(dsr["deflated_sharpe_probability"]),
        wf_positive_share=wf_positive_share,
        neighborhood_pass=neighborhood_pass,
        stress_pass=stress_pass,
        p95_participation=_finite(oos_metrics["p95_trade_participation"]),
        replay_fill_ratio=_finite(replay_fill_ratio, 0.0),
        slippage_budget_ratio=_finite(slippage_budget_ratio, 1.0),
        forward_positive_probability=0.0,
        forward_hac_sharpe=0.0,
        forward_calmar=0.0,
        forward_mdd=-1.0,
        delta_sharpe=_finite(oos_metrics["sharpe_ratio"] - benchmark_oos["sharpe_ratio"]),
        delta_cagr=_finite(oos_metrics["annualized_return"] - benchmark_oos["annualized_return"]),
        mdd_improvement=_finite(mdd_improvement),
        correlation=_finite(correlation),
        # Historical research has no Demo execution evidence.  D remains zero
        # until actual target parity, reconciliation, recovery, slippage, and
        # uptime observations exist.
        target_position_parity=0.0,
        reconciliation=0.0,
        state_recovery_no_duplicates=0.0,
        slippage_budget_adherence=0.0,
        uptime=0.0,
        hard_gate_failures=tuple(sorted(set(hard_failures))),
        data_integrity=True,
        causal_timing=prefix_passed,
        code_tests=bool(code_tests_passed),
        oos_total_return=_finite(oos_metrics["total_return"]),
        oos_duration_months=_finite(oos_metrics["years"] * 12.0),
        independent_cycles=independent_cycles,
        doubled_cost_return=_finite(doubled_cost),
        no_insolvency=bool(base_result.equity.min() > 0),
        portfolio_mdd=_finite(oos_metrics["max_drawdown"]),
        added_mdd_degradation=_finite(max(0.0, abs(oos_metrics["max_drawdown"]) - abs(benchmark_oos["max_drawdown"]))),
    )
    score = score_metrics(metrics_inputs).as_dict()

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": "cross_sectional_momentum_study",
        "strategy": STRATEGY_NAME,
        "version": 1,
        "symbols": list(normalized),
        "interval": interval,
        "requested_start": start,
        "test_start": test_start,
        "parameters": {
            "lookback": BASE_LOOKBACK,
            "spread_threshold": BASE_SPREAD,
            "min_winner_return": 0.0,
            "fee_bps": BASE_FEE_BPS,
            "slippage_bps": BASE_SLIPPAGE_BPS,
            "latency_bars": 0,
            "max_participation": None,
            "leverage": 1.0,
        },
        "data_quality": quality,
        "common_index": {
            "first": common[0].isoformat(),
            "last": common[-1].isoformat(),
            "rows": int(len(common)),
            "expected_4h_bars_by_span": int((common[-1] - common[0]).total_seconds() // (4 * 3600) + 1),
            "missing_common_bars": int((common[-1] - common[0]).total_seconds() // (4 * 3600) + 1 - len(common)),
        },
        "base_result": {
            "full": full_metrics,
            "train": train_metrics,
            "out_of_sample": oos_metrics,
            "benchmark_out_of_sample": benchmark_oos,
            "hac_sharpe": oos_hac_sharpe,
            "causal_prefix_passed": prefix_passed,
            "oos_risk": oos_risk,
        },
        "walk_forward": fold_rows,
        "causal_prefix_audit": prefix_rows,
        "parameter_trials": parameter_rows,
        "cost_latency_trials": cost_rows,
        "liquidity_trials": liquidity_rows,
        "trial_count": len(trials),
        "dsr": dsr,
        "oos_risk_status": oos_risk["status"],
        "hard_gate_failures": sorted(set(hard_failures)),
        "score_inputs": {**metrics_inputs.__dict__},
        "score_preview": score,
        "forward_status": "not_started",
        "forward_evidence": False,
        "code_tests_attested": bool(code_tests_passed),
        "database_sha256": file_hash(db_path),
        "spot_klines_sha256": spot_klines_digest(db_path),
        "source_provenance": build_source_manifest(),
        "research_evidence_source": str(run_dir / "results.json"),
    }
    _json_dump(run_dir / "results.json", payload)
    pd.DataFrame(trials).to_csv(run_dir / "trial_results.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(run_dir / "parameter_trials.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(run_dir / "cost_latency_trials.csv", index=False)
    pd.DataFrame(liquidity_rows).to_csv(run_dir / "liquidity_trials.csv", index=False)
    folds.to_csv(run_dir / "walk_forward.csv", index=False)
    pd.DataFrame(prefix_rows).to_csv(run_dir / "causal_prefix_audit.csv", index=False)
    signal_diagnostics.to_csv(run_dir / "signals.csv")
    base_result.equity.to_frame("equity").join(base_result.benchmark_equity.rename("benchmark_equity")).to_csv(run_dir / "equity.csv")
    base_result.weights.to_csv(run_dir / "weights.csv")
    base_result.trades.to_csv(run_dir / "trades.csv", index=False)
    _json_dump(run_dir / "metric_inputs.json", {**metrics_inputs.__dict__})
    _json_dump(run_dir / "score_preview.json", score)
    (run_dir / "report.md").write_text(_markdown_report(payload, pd.DataFrame(trials), folds), encoding="utf-8")

    ledger_record = {
        "run_id": run_id,
        "created_at": payload["created_at"],
        "study": payload["study"],
        "strategy": STRATEGY_NAME,
        "version": 1,
        "database_sha256": payload["database_sha256"],
        "spot_klines_sha256": payload["spot_klines_sha256"],
        "source_manifest_sha256": payload["source_provenance"]["manifest_sha256"],
        "symbols": list(normalized),
        "interval": interval,
        "fixed_test_start": test_start,
        "status": payload["oos_risk_status"],
        "rule_verdicts": [{"strategy": STRATEGY_NAME, "status": payload["oos_risk_status"]}],
        "oos_metrics": {
            "total_return": oos_metrics["total_return"],
            "sharpe_ratio": oos_metrics["sharpe_ratio"],
            "max_drawdown": oos_metrics["max_drawdown"],
            "trade_count": oos_metrics["trade_count"],
            "p95_trade_participation": oos_metrics["p95_trade_participation"],
        },
        "dsr": dsr["deflated_sharpe_probability"],
        "trial_count": len(trials),
        "hard_gate_failures": sorted(set(hard_failures)),
        "report_path": str(run_dir / "report.md"),
    }
    with (Path(output_root) / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_sanitize_json(ledger_record), ensure_ascii=False) + "\n")
    return run_dir


__all__ = [
    "BASE_LOOKBACK",
    "BASE_SPREAD",
    "STRATEGY_NAME",
    "generate_relative_strength_targets",
    "hac_sharpe",
    "run_cross_sectional_study",
    "run_relative_strength_backtest",
]
