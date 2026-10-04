"""Fixed-rank USD-M perpetual portfolios and a small auditable account model.

Market indexes are UTC hourly bar-open timestamps. A target indexed by a bar
timestamp uses that bar's close and is submitted for execution at the next
bar's open. The account assumes full fills, continuous contract quantities,
linear USDT-margined PnL, a single cash wallet, and fixed-bps slippage. It does
not model venue lot sizes, partial fills, liquidation, or intrabar margin paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


HOUR = pd.Timedelta(hours=1)
HOURS_PER_YEAR = 365.0 * 24.0


@dataclass(frozen=True)
class RebalancePolicy:
    """Hourly position-aware decisions; limits use pre-trade signal-close equity."""

    long_count: int
    short_count: int
    gross_exposure: float
    max_asset_weight: float
    holding_rank: int | None = None
    weight_buffer: float = 0.0

    def __post_init__(self):
        for name in ("long_count", "short_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.holding_rank is not None and (
            type(self.holding_rank) is not int
            or self.holding_rank < max(self.long_count, self.short_count)
        ):
            raise ValueError("holding_rank must cover both entry ranks")
        for name in ("gross_exposure", "max_asset_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if not np.isfinite(self.weight_buffer) or not 0 <= self.weight_buffer < 1:
            raise ValueError("weight_buffer must be in [0,1)")


@dataclass(frozen=True)
class ContinuousTargetPolicy:
    """Quantity-based no-trade bands for an already combined target portfolio."""

    gross_exposure: float
    max_asset_weight: float
    weight_buffer: float = 0.0

    def __post_init__(self):
        for name in ("gross_exposure", "max_asset_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if not np.isfinite(self.weight_buffer) or not 0 <= self.weight_buffer < 1:
            raise ValueError("weight_buffer must be in [0,1)")


DECISION_COLUMNS = [
    "score", "long_rank", "short_rank", "actual_weight", "lower_weight", "upper_weight",
    "execution_weight", "action", "reason", "trade_category",
]


def _rank_decision(row, quantities, policy, exact_targets):
    """Retain eligible old seats, then fill entry seats; ties use symbol order."""
    available = [symbol for symbol in row.index if pd.notna(row[symbol])]
    high = sorted(available, key=lambda symbol: (-float(row[symbol]), symbol))
    low = sorted(available, key=lambda symbol: (float(row[symbol]), symbol))
    high_rank = {symbol: rank for rank, symbol in enumerate(high, 1)}
    low_rank = {symbol: rank for rank, symbol in enumerate(low, 1)}
    if policy.holding_rank is None:
        return exact_targets, high_rank, low_rank
    longs = [s for s in high[:policy.holding_rank] if quantities[s] > 0]
    shorts = [s for s in low[:policy.holding_rank] if quantities[s] < 0]
    if len(longs) > policy.long_count or len(shorts) > policy.short_count:
        raise ValueError("actual positions exceed configured seat counts")
    for symbol in high[:policy.long_count]:
        if len(longs) < policy.long_count and symbol not in longs and symbol not in shorts:
            longs.append(symbol)
    for symbol in low[:policy.short_count]:
        if len(shorts) < policy.short_count and symbol not in shorts and symbol not in longs:
            shorts.append(symbol)
    targets = dict.fromkeys(row.index, 0.0)
    for selected, count, sign in ((longs, policy.long_count, 1), (shorts, policy.short_count, -1)):
        weight = sign * min(policy.gross_exposure / (2 * count), policy.max_asset_weight)
        for symbol in selected:
            targets[symbol] = weight
    return targets, high_rank, low_rank


def _quantity_decisions(row, quantities, closes, equity, policy, exact_targets):
    targets, high_rank, low_rank = _rank_decision(row, quantities, policy, exact_targets)
    decisions = {}
    for symbol, target in targets.items():
        current = quantities[symbol]
        actual = current * closes[symbol] / equity if current else 0.0
        lower = max(0.0, abs(target) - policy.weight_buffer)
        upper = min(abs(target) + policy.weight_buffer, policy.max_asset_weight)
        if target == 0:
            lower = upper = 0.0
            quantity = 0.0
            reason = ("missing_signal" if pd.isna(row[symbol]) else "rank_exit") if current else "no_seat"
        elif current == 0 or current * target < 0:
            quantity = _target_quantity(target, equity, closes[symbol])
            reason = "entry" if current == 0 else "direction_reversal"
        elif lower <= abs(actual) <= upper:
            quantity, reason = current, "within_buffer"
        else:
            boundary = lower if abs(actual) < lower else upper
            quantity = _target_quantity(np.sign(target) * boundary, equity, closes[symbol])
            reason = "underweight" if abs(actual) < lower else (
                "asset_cap" if abs(actual) > policy.max_asset_weight else "overweight")
        decisions[symbol] = {
            "target_weight": target, "target_quantity": quantity,
            "score": row[symbol], "long_rank": high_rank.get(symbol, np.nan),
            "short_rank": low_rank.get(symbol, np.nan), "actual_weight": actual,
            "lower_weight": lower, "upper_weight": upper, "reason": reason,
        }
    gross = sum(abs(item["target_quantity"] * closes[symbol] / equity)
                for symbol, item in decisions.items() if item["target_quantity"])
    if gross > policy.gross_exposure + 1e-12:
        scale = policy.gross_exposure / gross
        for item in decisions.values():
            if item["target_quantity"]:
                item["target_quantity"] *= scale
                item["reason"] = "gross_cap"
    for symbol, item in decisions.items():
        old, new = quantities[symbol], item["target_quantity"]
        item["execution_weight"] = new * closes[symbol] / equity if new else 0.0
        item["action"] = "hold" if old == new else ("reduce" if abs(new) < abs(old) else "add")
        item["trade_category"] = ("hold" if old == new else "direction_reversal" if old * new < 0
                                  else "symbol_change" if old == 0 or new == 0 else "resize")
    return decisions


def _continuous_target_decisions(row, quantities, closes, equity, policy):
    """Apply continuous signed targets without ranking or changing held quantities in-band."""
    decisions = {}
    for symbol, raw_target in row.items():
        current = quantities[symbol]
        close = closes[symbol]
        actual = current * close / equity if current else 0.0
        target = float(np.clip(raw_target, -policy.max_asset_weight, policy.max_asset_weight))
        lower = max(0.0, abs(target) - policy.weight_buffer)
        upper = min(abs(target) + policy.weight_buffer, policy.max_asset_weight)
        if target == 0.0:
            lower = upper = 0.0
            quantity = 0.0
            reason = "target_exit" if current else "no_target"
        elif current == 0.0 or current * target < 0.0:
            quantity = _target_quantity(target, equity, close)
            reason = "entry" if current == 0.0 else "direction_reversal"
        elif lower <= abs(actual) <= upper:
            quantity, reason = current, "within_buffer"
        else:
            boundary = lower if abs(actual) < lower else upper
            quantity = _target_quantity(np.sign(target) * boundary, equity, close)
            reason = "underweight" if abs(actual) < lower else (
                "asset_cap" if abs(actual) > policy.max_asset_weight else "overweight")
        decisions[symbol] = {
            "target_weight": target,
            "target_quantity": quantity,
            "score": np.nan,
            "long_rank": np.nan,
            "short_rank": np.nan,
            "actual_weight": actual,
            "lower_weight": lower,
            "upper_weight": upper,
            "reason": reason,
        }

    gross = sum(abs(item["target_quantity"] * closes[symbol] / equity)
                for symbol, item in decisions.items() if item["target_quantity"])
    if gross > policy.gross_exposure + 1e-12:
        scale = policy.gross_exposure / gross
        for item in decisions.values():
            if item["target_quantity"]:
                item["target_quantity"] *= scale
                item["reason"] = "gross_cap"
    for symbol, item in decisions.items():
        old, new = quantities[symbol], item["target_quantity"]
        item["execution_weight"] = new * closes[symbol] / equity if new else 0.0
        item["action"] = "hold" if old == new else ("reduce" if abs(new) < abs(old) else "add")
        item["trade_category"] = ("hold" if old == new else "direction_reversal" if old * new < 0
                                  else "symbol_change" if old == 0 or new == 0 else "resize")
    return decisions


@dataclass(frozen=True)
class MultifactorAccountResult:
    """Tables from one account run; timestamps identify signal or mark events."""

    orders: pd.DataFrame
    fills: pd.DataFrame
    positions: pd.DataFrame
    ledger: pd.DataFrame
    metrics: dict[str, Any]
    funding_events: pd.DataFrame


ORDER_COLUMNS = [
    "signal_timestamp",
    "execution_timestamp",
    "symbol",
    "target_weight",
    "signal_close",
    "signal_equity",
    "current_quantity",
    "target_quantity",
    "signed_order_quantity",
]
FILL_COLUMNS = [
    "timestamp",
    "symbol",
    "side",
    "quantity",
    "signed_quantity",
    "reference_open",
    "fill_price",
    "notional",
    "fee",
    "slippage_cost",
    "realized_pnl",
    "previous_quantity",
    "new_quantity",
    "average_entry_price",
]
FUNDING_COLUMNS = [
    "timestamp",
    "symbol",
    "quantity",
    "funding_rate",
    "mark_price",
    "cashflow",
]
POSITION_COLUMNS = [
    "timestamp",
    "symbol",
    "quantity",
    "average_entry_price",
    "mark_price",
    "signed_notional",
    "gross_notional",
    "unrealized_pnl",
    "target_weight",
]
LEDGER_COLUMNS = [
    "timestamp",
    "cash",
    "realized_pnl",
    "funding_cashflow",
    "fees",
    "slippage_cost",
    "trade_notional",
    "turnover",
    "gross_notional",
    "net_notional",
    "unrealized_pnl",
    "equity",
    "return",
    "maintenance_margin",
    "margin_buffer",
    "margin_ratio",
]


def _utc_timestamp(value: Any, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    return timestamp.tz_convert("UTC")


def _utc_index(index: pd.Index, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{name} must be a DatetimeIndex")
    if index.tz is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    return index.tz_convert("UTC")


def _require_hour_boundary(timestamp: pd.Timestamp, name: str) -> None:
    if timestamp != timestamp.floor("h"):
        raise ValueError(f"{name} must lie on a UTC hour boundary")


def _validate_scores(
    scores: pd.DataFrame,
    *,
    start: pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        raise ValueError("scores must be a nonempty DataFrame")
    if scores.columns.has_duplicates or not all(
        isinstance(symbol, str) and symbol for symbol in scores.columns
    ):
        raise ValueError("score columns must be unique nonempty symbol strings")
    index = _utc_index(scores.index, "scores.index")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("scores.index must be unique and increasing")
    first_signal = start - HOUR
    if first_signal not in index:
        raise ValueError("scores must contain the start-1h warm-up signal bar")
    scoped = scores.copy()
    scoped.index = index
    scoped = scoped.loc[first_signal:]
    expected = pd.date_range(first_signal, scoped.index[-1], freq=HOUR, tz="UTC")
    if not scoped.index.equals(expected):
        raise ValueError("scores must have a complete hourly grid from start-1h")
    numeric = scoped.to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("scores may be NaN for unavailable assets but cannot be infinite")
    return scoped


def generate_rank_targets(
    scores: pd.DataFrame,
    *,
    long_count: int,
    short_count: int,
    gross_exposure: float,
    max_asset_weight: float,
    rebalance_hours: int,
    start: Any,
) -> pd.DataFrame:
    """Select deterministic long/short ranks at fixed intervals.

    The first signal is anchored at ``start - 1h``. Each side receives half of
    the declared gross budget, divided by its configured seat count. Missing
    scores and unfilled seats remain cash. Longs are chosen first by descending
    score then symbol; shorts are chosen from the remaining names by ascending
    score then symbol, so one asset can never occupy both sides.

    The returned rows contain only rebalance decisions. Their index is the
    signal bar's open timestamp; execution follows at the next hourly open.
    """
    start_ts = _utc_timestamp(start, "start")
    _require_hour_boundary(start_ts, "start")
    if type(long_count) is not int or long_count <= 0:
        raise ValueError("long_count must be a positive integer")
    if type(short_count) is not int or short_count <= 0:
        raise ValueError("short_count must be a positive integer")
    if type(rebalance_hours) is not int or rebalance_hours <= 0:
        raise ValueError("rebalance_hours must be a positive integer")
    if not np.isfinite(gross_exposure) or not 0 < gross_exposure <= 1:
        raise ValueError("gross_exposure must be in (0, 1]")
    if not np.isfinite(max_asset_weight) or not 0 < max_asset_weight <= 1:
        raise ValueError("max_asset_weight must be in (0, 1]")

    scoped = _validate_scores(scores, start=start_ts)
    symbols = list(scoped.columns)
    targets: list[pd.Series] = []
    target_times: list[pd.Timestamp] = []
    long_slot_weight = min(gross_exposure / 2 / long_count, max_asset_weight)
    short_slot_weight = min(gross_exposure / 2 / short_count, max_asset_weight)

    for offset, (timestamp, row) in enumerate(scoped.iterrows()):
        if offset % rebalance_hours:
            continue
        available = [symbol for symbol in symbols if pd.notna(row[symbol])]
        ranked_high = sorted(available, key=lambda symbol: (-float(row[symbol]), symbol))
        selected_longs = ranked_high[:long_count]
        selected_long_set = set(selected_longs)
        remaining = [symbol for symbol in available if symbol not in selected_long_set]
        selected_shorts = sorted(
            remaining, key=lambda symbol: (float(row[symbol]), symbol)
        )[:short_count]
        weights = pd.Series(0.0, index=symbols, dtype=float)
        if selected_longs:
            weights.loc[selected_longs] = long_slot_weight
        if selected_shorts:
            weights.loc[selected_shorts] = -short_slot_weight
        targets.append(weights)
        target_times.append(timestamp)

    result = pd.DataFrame(targets, index=pd.DatetimeIndex(target_times, tz="UTC"))
    result.index.name = "signal_timestamp"
    result.columns.name = "symbol"
    return result


def _validate_market_frames(
    frames: dict[str, pd.DataFrame],
    expected_index: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    if not isinstance(frames, dict) or not frames:
        raise ValueError("frames must be a nonempty symbol-to-DataFrame mapping")
    if not all(isinstance(symbol, str) and symbol for symbol in frames):
        raise ValueError("market symbols must be nonempty strings")
    normalized: dict[str, pd.DataFrame] = {}
    for symbol in sorted(frames):
        frame = frames[symbol]
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{symbol} market data must be a DataFrame")
        required = {"open", "close", "mark_close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{symbol} market data missing fields: {sorted(missing)}")
        index = _utc_index(frame.index, f"{symbol}.index")
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError(f"{symbol} market index must be unique and increasing")
        candidate = frame.copy()
        candidate.index = index
        absent = expected_index.difference(index)
        if len(absent):
            raise ValueError(f"{symbol} market data is missing hourly bars, first={absent[0]}")
        candidate = candidate.reindex(expected_index)
        prices = candidate[["open", "close", "mark_close"]].to_numpy(dtype=float)
        if "inactive" in candidate:
            if candidate["inactive"].dtype != bool or candidate["inactive"].isna().any():
                raise ValueError(f"{symbol} inactive flags must be explicit booleans")
            active = ~candidate["inactive"].to_numpy()
            if not np.isnan(prices[~active]).all():
                raise ValueError(f"{symbol} inactive hours must have no market prices")
        else:
            active = np.ones(len(candidate), dtype=bool)
        if not np.isfinite(prices[active]).all() or (prices[active] <= 0).any():
            raise ValueError(f"{symbol} prices must be finite and positive on the account grid")
        normalized[symbol] = candidate
    return normalized


def _validate_targets(
    targets: pd.DataFrame,
    *,
    symbols: list[str],
    expected_signal_index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(targets, pd.DataFrame) or targets.empty:
        raise ValueError("targets must contain the initial start-1h rebalance row")
    if targets.columns.has_duplicates or set(targets.columns) != set(symbols):
        raise ValueError("target columns must match market symbols exactly")
    if list(targets.columns) != symbols:
        targets = targets.loc[:, symbols]
    index = _utc_index(targets.index, "targets.index")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("targets.index must be unique and increasing")
    first_signal = start - HOUR
    if index[0] != first_signal:
        raise ValueError("the first target signal must be the start-1h warm-up bar")
    if not index.isin(expected_signal_index).all():
        raise ValueError("every target signal timestamp must be an available hourly bar")
    if (index + HOUR >= end).any():
        raise ValueError("target signals must execute strictly before the end-exclusive boundary")
    normalized = targets.copy()
    normalized.index = index
    values = normalized.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("target weights must be finite")
    if (np.abs(values) > 1 + 1e-12).any():
        raise ValueError("absolute target weight for a symbol cannot exceed one")
    if (np.abs(values).sum(axis=1) > 1 + 1e-12).any():
        raise ValueError("gross target exposure cannot exceed one")
    return normalized


def _validate_funding(
    funding: pd.DataFrame,
    *,
    symbols: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    required = {"timestamp", "symbol", "funding_rate", "mark_price"}
    if not isinstance(funding, pd.DataFrame) or not required.issubset(funding.columns):
        raise ValueError(f"funding must contain columns: {sorted(required)}")
    events = funding.loc[:, ["timestamp", "symbol", "funding_rate", "mark_price"]].copy()
    parsed: list[pd.Timestamp] = []
    for value in events["timestamp"]:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            raise ValueError("funding timestamps must be timezone-aware UTC")
        parsed.append(stamp.tz_convert("UTC"))
    events["timestamp"] = pd.DatetimeIndex(parsed, tz="UTC")
    if not events["symbol"].map(lambda symbol: isinstance(symbol, str) and symbol).all():
        raise ValueError("funding symbols must be nonempty strings")
    if not set(events["symbol"]).issubset(symbols):
        raise ValueError("funding contains a symbol without market data")
    if ((events["timestamp"] < start) | (events["timestamp"] >= end)).any():
        raise ValueError("funding timestamps must lie in the [start, end) account window")
    rates = events["funding_rate"].to_numpy(dtype=float)
    marks = events["mark_price"].to_numpy(dtype=float)
    if not np.isfinite(rates).all() or not np.isfinite(marks).all() or (marks <= 0).any():
        raise ValueError("funding rates must be finite and settlement marks finite and positive")
    if events.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("funding events cannot repeat a symbol at the same timestamp")
    return events.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)


def _close_equity(
    cash: float,
    quantities: dict[str, float],
    entries: dict[str, float],
    marks: dict[str, float],
) -> tuple[float, float]:
    unrealized = sum(
        quantities[symbol] * (marks[symbol] - entries[symbol])
        for symbol in quantities
        if quantities[symbol] != 0.0
    )
    return cash + unrealized, unrealized


def _check_margin(
    *,
    timestamp: pd.Timestamp,
    cash: float,
    quantities: dict[str, float],
    entries: dict[str, float],
    marks: dict[str, float],
    margin_fraction: float,
) -> tuple[float, float, float, float]:
    equity, unrealized = _close_equity(cash, quantities, entries, marks)
    gross = sum(abs(quantities[symbol] * marks[symbol]) for symbol in quantities
                if quantities[symbol])
    maintenance = margin_fraction * gross
    if not np.isfinite(equity) or equity <= 0:
        raise ValueError(f"non-positive account equity at {timestamp}")
    if gross > 0 and equity <= maintenance:
        raise ValueError(
            f"sampled margin guard breached at {timestamp}: equity={equity}, required={maintenance}"
        )
    return equity, unrealized, gross, maintenance


def _realize_trade(
    *,
    old_quantity: float,
    delta_quantity: float,
    average_entry: float,
    fill_price: float,
) -> tuple[float, float, float]:
    """Return realized PnL, new quantity, and new weighted entry price."""
    new_quantity = old_quantity + delta_quantity
    realized = 0.0
    if old_quantity == 0.0 or old_quantity * delta_quantity > 0:
        old_abs = abs(old_quantity)
        new_abs = abs(delta_quantity)
        entry = (
            fill_price
            if old_abs == 0.0
            else (average_entry * old_abs + fill_price * new_abs) / (old_abs + new_abs)
        )
        return realized, new_quantity, entry

    closed = min(abs(old_quantity), abs(delta_quantity))
    realized = closed * np.sign(old_quantity) * (fill_price - average_entry)
    if new_quantity == 0.0:
        return realized, 0.0, np.nan
    if old_quantity * new_quantity < 0:
        return realized, new_quantity, fill_price
    return realized, new_quantity, average_entry


def _target_quantity(target_weight: float, signal_equity: float, signal_close: float) -> float:
    """Freeze linear-contract quantity from information known at signal close."""
    if target_weight == 0.0:
        return 0.0
    return float(target_weight) * float(signal_equity) / float(signal_close)


def _fill_terms(
    delta_quantity: float,
    reference_open: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[float, float, float, float]:
    """Return the shared research/forward full-fill price and costs."""
    buying = delta_quantity > 0
    fill_price = reference_open * (1 + slippage_rate if buying else 1 - slippage_rate)
    notional = abs(delta_quantity) * fill_price
    fee = notional * fee_rate
    slippage_cost = abs(delta_quantity) * abs(fill_price - reference_open)
    return fill_price, notional, fee, slippage_cost


def _apply_fill(
    cash: float,
    old_quantity: float,
    average_entry: float,
    delta_quantity: float,
    fill_price: float,
    fee: float,
) -> tuple[float, float, float, float]:
    """Apply one complete signed fill using the shared linear-PnL accounting."""
    realized, new_quantity, new_entry = _realize_trade(
        old_quantity=old_quantity,
        delta_quantity=delta_quantity,
        average_entry=average_entry,
        fill_price=fill_price,
    )
    return cash + (realized - fee), realized, new_quantity, new_entry


def _funding_cashflow(quantity: float, funding_rate: float, mark_price: float) -> float:
    """Return the signed USD-M funding cashflow for a linear position."""
    return -float(quantity) * float(funding_rate) * float(mark_price)


def _performance_metrics(
    *,
    ledger: pd.DataFrame,
    initial_capital: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fills: pd.DataFrame,
    funding_events: pd.DataFrame,
) -> dict[str, float]:
    equity = np.concatenate(([initial_capital], ledger["equity"].to_numpy(dtype=float)))
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    net_return = float(equity[-1] / initial_capital - 1.0)
    elapsed_hours = (end - start) / HOUR
    annualized_return = float(
        (equity[-1] / initial_capital) ** (HOURS_PER_YEAR / elapsed_hours) - 1.0
    )
    returns = pd.Series(equity).pct_change().dropna().to_numpy(dtype=float)
    annualized_volatility = (
        float(np.std(returns, ddof=1) * np.sqrt(HOURS_PER_YEAR))
        if len(returns) > 1
        else 0.0
    )
    mean_annual_return = float(np.mean(returns) * HOURS_PER_YEAR) if len(returns) else 0.0
    sharpe_ratio = mean_annual_return / annualized_volatility if annualized_volatility > 0 else 0.0
    return {
        "net_return": net_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(-np.min(drawdown)),
        "total_fees": float(fills["fee"].sum()) if not fills.empty else 0.0,
        "total_slippage_cost": float(fills["slippage_cost"].sum()) if not fills.empty else 0.0,
        "total_funding": float(funding_events["cashflow"].sum()) if not funding_events.empty else 0.0,
        "total_turnover": float(ledger["turnover"].sum()),
        "final_equity": float(equity[-1]),
    }


def run_perpetual_account(
    frames: dict[str, pd.DataFrame],
    targets: pd.DataFrame | None,
    funding: pd.DataFrame,
    *,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
    start: Any,
    end: Any,
    margin_fraction: float,
    scores: pd.DataFrame | None = None,
    rebalance_policy: RebalancePolicy | None = None,
    continuous_target_policy: ContinuousTargetPolicy | None = None,
) -> MultifactorAccountResult:
    """Run a full-fill USD-M cross-margin ledger over the half-open UTC window.

    ``frames`` hold hourly bars indexed by open time. Targets are close-time
    decisions encoded on the previous bar's open timestamp and execute at the
    next open. Order quantities are frozen from that signal close and its
    closing equity. Funding at an execution open is paid by the old position;
    later events in the hour are paid by the post-trade position. Long cashflow
    is ``-quantity * rate * settlement_mark`` (positive rates charge longs).

    Equity is cash plus mark-to-market unrealized PnL. Fees reduce cash; slippage
    is already reflected in fill prices and is reported separately, never
    deducted twice. Margin is checked only at supplied funding marks and each
    hourly mark close. A breach stops the run; no forced liquidation is modeled.

    With ``rebalance_policy``, provide hourly ``scores`` and ``targets=None``.
    Each account decides quantities from its own previous closing equity and
    actual holdings, before funding at the execution open. A hold preserves
    quantity exactly. Bands and exposure limits apply at signal-close prices;
    execution gaps, costs and subsequent marks can cause exposure to drift.

    With ``continuous_target_policy``, provide a signed target-weight DataFrame.
    It uses the same account-local equity and quantity bands without converting
    targets into ranked seats. Zero targets exit immediately.
    """
    start_ts = _utc_timestamp(start, "start")
    end_ts = _utc_timestamp(end, "end")
    _require_hour_boundary(start_ts, "start")
    _require_hour_boundary(end_ts, "end")
    if end_ts <= start_ts:
        raise ValueError("end must be later than start")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive")
    if not np.isfinite(fee_bps) or fee_bps < 0:
        raise ValueError("fee_bps must be finite and nonnegative")
    if not np.isfinite(slippage_bps) or not 0 <= slippage_bps < 10_000:
        raise ValueError("slippage_bps must be in [0, 10000)")
    if not np.isfinite(margin_fraction) or not 0 <= margin_fraction < 1:
        raise ValueError("margin_fraction must be in [0, 1)")

    first_signal = start_ts - HOUR
    expected_index = pd.date_range(first_signal, end_ts - HOUR, freq=HOUR, tz="UTC")
    execution_index = pd.date_range(start_ts, end_ts - HOUR, freq=HOUR, tz="UTC")
    market = _validate_market_frames(frames, expected_index)
    symbols = sorted(market)
    if continuous_target_policy is not None:
        if (not isinstance(continuous_target_policy, ContinuousTargetPolicy)
                or rebalance_policy is not None or scores is not None or targets is None):
            raise ValueError(
                "continuous targets require ContinuousTargetPolicy, targets, and no scores or RebalancePolicy"
            )
    if rebalance_policy is None:
        if scores is not None:
            raise ValueError("scores require a rebalance_policy")
    else:
        if not isinstance(rebalance_policy, RebalancePolicy) or targets is not None:
            raise ValueError("position-aware decisions require RebalancePolicy and targets=None")
        scores = _validate_scores(scores, start=start_ts)
        signal_grid = execution_index - HOUR
        if not scores.index.equals(signal_grid) or set(scores.columns) != set(symbols):
            raise ValueError("scores must match the complete account signal grid and symbols")
        targets = generate_rank_targets(
            scores, long_count=rebalance_policy.long_count, short_count=rebalance_policy.short_count,
            gross_exposure=rebalance_policy.gross_exposure,
            max_asset_weight=rebalance_policy.max_asset_weight, rebalance_hours=1, start=start_ts,
        )
    target_weights = _validate_targets(
        targets,
        symbols=symbols,
        expected_signal_index=expected_index,
        start=start_ts,
        end=end_ts,
    )
    funding_events = _validate_funding(
        funding, symbols=set(symbols), start=start_ts, end=end_ts
    )
    for symbol in symbols:
        market[symbol] = market[symbol].loc[expected_index]
    inactive_by_timestamp: dict[pd.Timestamp, set[str]] = {}
    for symbol in symbols:
        if "inactive" in market[symbol]:
            for timestamp in market[symbol].index[market[symbol]["inactive"]]:
                inactive_by_timestamp.setdefault(timestamp, set()).add(symbol)

    quantities = {symbol: 0.0 for symbol in symbols}
    entries = {symbol: np.nan for symbol in symbols}
    active_targets = {symbol: 0.0 for symbol in symbols}
    cash = float(initial_capital)
    fee_rate = float(fee_bps) / 10_000.0
    slip_rate = float(slippage_bps) / 10_000.0
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    payments: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    target_lookup = {timestamp: row for timestamp, row in target_weights.iterrows()}

    def settle_events(group: pd.DataFrame | None) -> float:
        nonlocal cash
        nonlocal latest_marks
        if group is None:
            return 0.0
        total = 0.0
        event_marks = latest_marks.copy()
        for event in group.itertuples(index=False):
            quantity = quantities[event.symbol]
            cashflow = _funding_cashflow(quantity, event.funding_rate, event.mark_price)
            cash += cashflow
            total += cashflow
            payments.append(
                {
                    "timestamp": event.timestamp,
                    "symbol": event.symbol,
                    "quantity": quantity,
                    "funding_rate": float(event.funding_rate),
                    "mark_price": float(event.mark_price),
                    "cashflow": cashflow,
                }
            )
            event_marks[event.symbol] = float(event.mark_price)
        _check_margin(
            timestamp=group["timestamp"].iloc[0],
            cash=cash,
            quantities=quantities,
            entries=entries,
            marks=event_marks,
            margin_fraction=float(margin_fraction),
        )
        latest_marks.update(event_marks)
        return total

    latest_marks = {
        symbol: float(market[symbol].loc[first_signal, "mark_close"])
        for symbol in symbols
    }

    for timestamp in execution_index:
        bar_end = timestamp + HOUR
        signal_timestamp = timestamp - HOUR
        inactive = inactive_by_timestamp.get(timestamp, set())
        for symbol in inactive:
            if quantities[symbol] != 0.0:
                raise ValueError(f"{symbol} has an unclosed position at inactive hour {timestamp}")
        bar_funding = funding_events.loc[
            (funding_events["timestamp"] >= timestamp)
            & (funding_events["timestamp"] < bar_end)
        ]
        open_funding = bar_funding.loc[bar_funding["timestamp"] == timestamp]
        funding_total = settle_events(open_funding if not open_funding.empty else None)
        realized_total = 0.0
        fee_total = 0.0
        slippage_total = 0.0
        traded_notional = 0.0
        signal_equity = (
            float(ledger_rows[-1]["equity"]) if ledger_rows else float(initial_capital)
        )

        if signal_timestamp in target_lookup:
            target_row = target_lookup[signal_timestamp]
            decisions = None
            if continuous_target_policy is not None:
                closes = {symbol: float(market[symbol].loc[signal_timestamp, "close"]) for symbol in symbols}
                decisions = _continuous_target_decisions(target_row, quantities, closes,
                                                         signal_equity, continuous_target_policy)
            elif rebalance_policy is not None:
                closes = {symbol: float(market[symbol].loc[signal_timestamp, "close"]) for symbol in symbols}
                decisions = _quantity_decisions(scores.loc[signal_timestamp], quantities, closes,
                                                 signal_equity, rebalance_policy, target_row.to_dict())
            for symbol in symbols:
                signal_close = float(market[symbol].loc[signal_timestamp, "close"])
                target_weight = float(target_row[symbol] if decisions is None else decisions[symbol]["target_weight"])
                target_quantity = (_target_quantity(target_weight, signal_equity, signal_close)
                                   if decisions is None else decisions[symbol]["target_quantity"])
                if symbol in inactive and target_quantity != 0.0:
                    raise ValueError(f"{symbol} has a nonzero target at inactive hour {timestamp}")
                current_quantity = quantities[symbol]
                delta_quantity = target_quantity - current_quantity
                orders.append(
                    {
                        "signal_timestamp": signal_timestamp,
                        "execution_timestamp": timestamp,
                        "symbol": symbol,
                        "target_weight": target_weight,
                        "signal_close": signal_close,
                        "signal_equity": signal_equity,
                        "current_quantity": current_quantity,
                        "target_quantity": target_quantity,
                        "signed_order_quantity": delta_quantity,
                        **({field: decisions[symbol][field] for field in DECISION_COLUMNS}
                           if decisions is not None else {}),
                    }
                )
                active_targets[symbol] = target_weight
                if delta_quantity == 0.0:
                    continue

                reference_open = float(market[symbol].loc[timestamp, "open"])
                buying = delta_quantity > 0
                fill_price, notional, fee, slippage_cost = _fill_terms(
                    delta_quantity, reference_open, fee_rate, slip_rate
                )
                old_quantity = current_quantity
                old_entry = entries[symbol]
                cash, realized, new_quantity, new_entry = _apply_fill(
                    cash,
                    old_quantity,
                    float(old_entry) if old_quantity else np.nan,
                    delta_quantity,
                    fill_price,
                    fee,
                )
                quantities[symbol] = new_quantity
                entries[symbol] = new_entry
                realized_total += realized
                fee_total += fee
                slippage_total += slippage_cost
                traded_notional += notional
                fills.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "side": "BUY" if buying else "SELL",
                        "quantity": abs(delta_quantity),
                        "signed_quantity": delta_quantity,
                        "reference_open": reference_open,
                        "fill_price": fill_price,
                        "notional": notional,
                        "fee": fee,
                        "slippage_cost": slippage_cost,
                        "realized_pnl": realized,
                        "previous_quantity": old_quantity,
                        "new_quantity": new_quantity,
                        "average_entry_price": new_entry,
                    }
                )

        later_funding = bar_funding.loc[bar_funding["timestamp"] > timestamp]
        for _event_timestamp, event_group in later_funding.groupby("timestamp", sort=True):
            funding_total += settle_events(event_group)
        closing_marks = {
            symbol: float(market[symbol].loc[timestamp, "mark_close"])
            for symbol in symbols
        }
        latest_marks.update(closing_marks)
        equity, unrealized, gross, maintenance = _check_margin(
            timestamp=bar_end,
            cash=cash,
            quantities=quantities,
            entries=entries,
            marks=closing_marks,
            margin_fraction=float(margin_fraction),
        )
        long_notional = sum(
            max(quantities[symbol] * closing_marks[symbol], 0.0) for symbol in symbols
            if quantities[symbol]
        )
        short_notional = sum(
            max(-quantities[symbol] * closing_marks[symbol], 0.0) for symbol in symbols
            if quantities[symbol]
        )
        ledger_rows.append(
            {
                "timestamp": bar_end,
                "cash": cash,
                "realized_pnl": realized_total,
                "funding_cashflow": funding_total,
                "fees": fee_total,
                "slippage_cost": slippage_total,
                "trade_notional": traded_notional,
                "turnover": traded_notional / signal_equity,
                "gross_notional": gross,
                "net_notional": long_notional - short_notional,
                "unrealized_pnl": unrealized,
                "equity": equity,
                "return": np.nan,
                "maintenance_margin": maintenance,
                "margin_buffer": equity - maintenance,
                "margin_ratio": equity / gross if gross else np.inf,
            }
        )
        for symbol in symbols:
            quantity = quantities[symbol]
            mark = closing_marks[symbol]
            position_rows.append(
                {
                    "timestamp": bar_end,
                    "symbol": symbol,
                    "quantity": quantity,
                    "average_entry_price": entries[symbol],
                    "mark_price": mark,
                    "signed_notional": quantity * mark if quantity else 0.0,
                    "gross_notional": abs(quantity * mark) if quantity else 0.0,
                    "unrealized_pnl": quantity * (mark - entries[symbol]) if quantity else 0.0,
                    "target_weight": active_targets[symbol],
                }
            )

    has_quantity_decisions = rebalance_policy is not None or continuous_target_policy is not None
    orders_frame = pd.DataFrame(orders, columns=ORDER_COLUMNS + (DECISION_COLUMNS if has_quantity_decisions else []))
    fills_frame = pd.DataFrame(fills, columns=FILL_COLUMNS)
    payments_frame = pd.DataFrame(payments, columns=FUNDING_COLUMNS)
    positions_frame = pd.DataFrame(position_rows, columns=POSITION_COLUMNS)
    ledger_frame = pd.DataFrame(ledger_rows, columns=LEDGER_COLUMNS).set_index("timestamp")
    ledger_frame["return"] = ledger_frame["equity"].pct_change()
    if not ledger_frame.empty:
        ledger_frame.iloc[0, ledger_frame.columns.get_loc("return")] = (
            float(ledger_frame["equity"].iloc[0]) / float(initial_capital) - 1.0
        )
    metrics = _performance_metrics(
        ledger=ledger_frame,
        initial_capital=float(initial_capital),
        start=start_ts,
        end=end_ts,
        fills=fills_frame,
        funding_events=payments_frame,
    )
    return MultifactorAccountResult(
        orders=orders_frame,
        fills=fills_frame,
        positions=positions_frame,
        ledger=ledger_frame,
        metrics=metrics,
        funding_events=payments_frame,
    )
