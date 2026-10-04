"""Stepwise factor selection using causal net-Sharpe account simulations.

Each candidate subset and Ridge alpha is scored by running the same fixed-rank
R0 account over the latest matured inner-validation window. Candidate accounts
are advanced together in NumPy arrays so the combinatorial search does not
build an audit ledger for every proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import multifactor_account as account
from . import multifactor_selection as selection
from .multifactor_contracts import MultifactorContract


HOUR = pd.Timedelta(hours=1)
HOURS_PER_YEAR = 365.0 * 24.0
ROUTE = "stepwise_net_sharpe"
OBJECTIVE_NAME = "validation_net_sharpe"


@dataclass(frozen=True)
class _MarketArrays:
    symbols: list[str]
    index: pd.DatetimeIndex
    position: dict[pd.Timestamp, int]
    opens: np.ndarray
    closes: np.ndarray
    marks: np.ndarray
    inactive: np.ndarray


@dataclass(frozen=True)
class _RidgeTrainingStats:
    train_gram: np.ndarray | None
    train_rhs: np.ndarray | None
    target_rms: float | None
    sample_count: int
    invalid_status: str | None


def _ridge_training_stats(x_train: np.ndarray, y_train: np.ndarray) -> _RidgeTrainingStats:
    if not len(y_train):
        return _RidgeTrainingStats(None, None, None, 0, "empty_inner_training_sample")
    target_rms = float(np.sqrt(np.mean(np.square(y_train))))
    if not np.isfinite(target_rms) or target_rms <= 0.0:
        return _RidgeTrainingStats(None, None, target_rms, len(y_train),
                                   "invalid_training_target_rms")
    train_gram = np.einsum("ni,nj->ij", x_train, x_train) / len(y_train)
    train_rhs = (np.einsum("ni,n->i", x_train, y_train)
                 / (len(y_train) * target_rms))
    return _RidgeTrainingStats(train_gram, train_rhs, target_rms, len(y_train), None)


def _market_arrays(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    funding: pd.DataFrame,
) -> tuple[_MarketArrays, pd.DataFrame]:
    selection._require(isinstance(frames, dict) and set(frames) == set(symbols),
                       "frames must contain the factor universe exactly")
    selection._require(bool(frames), "frames cannot be empty")
    index_source = frames[sorted(frames)[0]]
    selection._require(isinstance(index_source, pd.DataFrame),
                       "market frames must be DataFrames")
    expected_index = account._utc_index(index_source.index, "market index")
    selection._require(expected_index.is_unique and expected_index.is_monotonic_increasing,
                       "market index must be unique and increasing")
    selection._require(len(expected_index) > 0, "market index cannot be empty")
    normalized = account._validate_market_frames(frames, expected_index)
    # A common hourly frame index makes all candidate state arrays positionally
    # comparable and matches the account engine's complete market contract.
    for symbol in symbols:
        selection._require(normalized[symbol].index.equals(expected_index),
                           f"{symbol} market index differs from the common grid")

    start = expected_index[0] + HOUR
    end = expected_index[-1] + HOUR
    funding_events = account._validate_funding(
        funding, symbols=set(symbols), start=start, end=end,
    )
    opens = np.column_stack([normalized[symbol]["open"].to_numpy(dtype=float)
                             for symbol in symbols])
    closes = np.column_stack([normalized[symbol]["close"].to_numpy(dtype=float)
                              for symbol in symbols])
    marks = np.column_stack([normalized[symbol]["mark_close"].to_numpy(dtype=float)
                             for symbol in symbols])
    inactive = np.column_stack([
        (normalized[symbol]["inactive"].to_numpy(dtype=bool)
         if "inactive" in normalized[symbol] else np.zeros(len(expected_index), dtype=bool))
        for symbol in symbols
    ])
    return _MarketArrays(
        symbols=symbols,
        index=expected_index,
        position={pd.Timestamp(value): i for i, value in enumerate(expected_index)},
        opens=opens,
        closes=closes,
        marks=marks,
        inactive=inactive,
    ), funding_events


def _equity_and_margin(
    cash: np.ndarray,
    quantities: np.ndarray,
    entries: np.ndarray,
    marks: np.ndarray,
    margin_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = quantities != 0.0
    unrealized = np.where(active, quantities * (marks[None, :] - entries), 0.0).sum(axis=1)
    gross = np.where(active, np.abs(quantities * marks[None, :]), 0.0).sum(axis=1)
    equity_values = cash + unrealized
    breached = (~np.isfinite(equity_values) | (equity_values <= 0.0)
                | ((gross > 0.0) & (equity_values <= margin_fraction * gross)))
    return equity_values, breached


def _apply_funding_batch(
    cash: np.ndarray,
    quantities: np.ndarray,
    entries: np.ndarray,
    latest_marks: np.ndarray,
    event_symbols: np.ndarray,
    event_rates: np.ndarray,
    event_marks: np.ndarray,
    margin_fraction: float,
    alive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Settle one timestamp's old or current positions and check margin."""
    event_payments = -quantities[:, event_symbols] * (event_rates * event_marks)[None, :]
    cashflow = np.where(alive[:, None], event_payments, 0.0).sum(axis=1)
    cash += cashflow
    event_latest = latest_marks.copy()
    event_latest[event_symbols] = event_marks
    equity_values, breached = _equity_and_margin(
        cash, quantities, entries, event_latest, margin_fraction,
    )
    newly_breached = alive & breached
    alive[newly_breached] = False
    latest_marks[event_symbols] = event_marks
    return cashflow, equity_values, newly_breached, event_latest


def _funding_by_execution_bar(
    funding: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbols: list[str],
    bar_count: int,
) -> list[list[tuple[pd.Timestamp, np.ndarray, np.ndarray, np.ndarray]]]:
    by_bar: list[list[tuple[pd.Timestamp, np.ndarray, np.ndarray, np.ndarray]]] = [
        [] for _ in range(bar_count)
    ]
    scoped = funding.loc[(funding["timestamp"] >= start) & (funding["timestamp"] < end)]
    if scoped.empty:
        return by_bar
    symbol_positions = {symbol: i for i, symbol in enumerate(symbols)}
    elapsed = (scoped["timestamp"] - start) / HOUR
    bar_positions = elapsed.to_numpy(dtype=np.int64)
    grouped = scoped.assign(_bar=bar_positions).groupby(["_bar", "timestamp"], sort=True)
    for (bar, timestamp), events in grouped:
        by_bar[int(bar)].append((
            pd.Timestamp(timestamp),
            events["symbol"].map(symbol_positions).to_numpy(dtype=np.int64),
            events["funding_rate"].to_numpy(dtype=float),
            events["mark_price"].to_numpy(dtype=float),
        ))
    return by_bar


def _r0_targets(
    score_cube: np.ndarray,
    complete: np.ndarray,
    source_times: pd.DatetimeIndex,
    execution_times: pd.DatetimeIndex,
    *,
    minimum_symbols: int,
    long_count: int,
    short_count: int,
    gross_exposure: float,
    max_asset_weight: float,
    rebalance_hours: int,
    anchor: pd.Timestamp,
) -> dict[int, np.ndarray]:
    candidate_count, _, symbol_count = score_cube.shape
    slot_long = min(gross_exposure / (2 * long_count), max_asset_weight)
    slot_short = min(gross_exposure / (2 * short_count), max_asset_weight)
    events: dict[int, np.ndarray] = {}
    for source_position, timestamp in enumerate(source_times):
        offset = int((timestamp - anchor) / HOUR)
        if offset % rebalance_hours:
            continue
        scores = score_cube[:, source_position, :]
        is_complete = complete[source_position]
        available = np.isfinite(scores) & is_complete[None, :]
        enough = available.sum(axis=1) >= minimum_symbols
        weights = np.zeros((candidate_count, symbol_count), dtype=float)
        if enough.any():
            high_order = np.argsort(
                np.where(available, -scores, np.inf), axis=1, kind="stable",
            )
            selected_longs = high_order[:, :long_count]
            long_valid = np.take_along_axis(available, selected_longs, axis=1)
            long_valid &= enough[:, None]
            candidate_rows = np.broadcast_to(np.arange(candidate_count)[:, None], selected_longs.shape)
            weights[candidate_rows[long_valid], selected_longs[long_valid]] = slot_long
            selected_long_mask = weights > 0.0
            low_order = np.argsort(
                np.where(available & ~selected_long_mask, scores, np.inf),
                axis=1, kind="stable",
            )
            selected_shorts = low_order[:, :short_count]
            short_valid = np.take_along_axis(available & ~selected_long_mask,
                                             selected_shorts, axis=1)
            short_valid &= enough[:, None]
            candidate_rows_short = np.broadcast_to(np.arange(candidate_count)[:, None],
                                                   selected_shorts.shape)
            weights[candidate_rows_short[short_valid], selected_shorts[short_valid]] = -slot_short
        execution_position = int(execution_times.get_indexer([timestamp + HOUR])[0])
        if execution_position >= 0:
            events[execution_position] = weights
    return events


def _simulate_r0_batch(
    coefficients: np.ndarray,
    validation_values: np.ndarray,
    validation_complete: np.ndarray,
    source_times: pd.DatetimeIndex,
    fit_time: pd.Timestamp,
    market: _MarketArrays,
    funding: pd.DataFrame,
    contract: MultifactorContract,
    policy: dict[str, Any],
    *,
    include_equity_path: bool = False,
) -> list[dict[str, Any]]:
    """Run candidate accounts together with account-engine event ordering."""
    candidate_count, factor_count = coefficients.shape
    symbol_count = len(market.symbols)
    if candidate_count == 0:
        return []
    # Only scheduled R0 sources issue targets. Avoid materializing every
    # candidate's scores for the other 23 hours of a daily holding interval.
    anchor = contract.bounds[0] - HOUR
    scheduled = ((source_times.asi8 - anchor.value)
                 % (contract.portfolio["rebalance_hours"] * HOUR.value)) == 0
    scheduled_values = validation_values[scheduled]
    clean_values = np.where(np.isfinite(scheduled_values), scheduled_values, 0.0)
    score_cube = np.einsum("tsf,cf->cts", clean_values, coefficients, optimize=True)
    score_cube = np.where(validation_complete[None, scheduled, :], score_cube, np.nan)

    account_start = source_times[0] + HOUR
    account_end = fit_time
    selection._require(account_start < account_end,
                       "net-Sharpe validation account window is empty")
    execution_times = pd.date_range(account_start, account_end - HOUR, freq="h", tz="UTC")
    execution_positions = market.index.get_indexer(execution_times)
    selection._require((execution_positions >= 0).all(),
                       "validation execution bars are missing from market frames")
    source_positions = market.index.get_indexer(source_times)
    selection._require((source_positions >= 0).all(),
                       "validation source bars are missing from market frames")
    schedule = _r0_targets(
        score_cube,
        validation_complete[scheduled],
        source_times[scheduled],
        execution_times,
        minimum_symbols=policy["min_cross_section_symbols"],
        long_count=contract.portfolio["long_count"],
        short_count=contract.portfolio["short_count"],
        gross_exposure=contract.portfolio["gross_exposure"],
        max_asset_weight=contract.portfolio["max_asset_weight"],
        rebalance_hours=contract.portfolio["rebalance_hours"],
        anchor=anchor,
    )

    opens = market.opens[execution_positions]
    closing_marks = market.marks[execution_positions]
    signal_closes = market.closes[source_positions]
    inactive = market.inactive[execution_positions]
    funding_by_bar = _funding_by_execution_bar(
        funding,
        start=account_start,
        end=account_end,
        symbols=market.symbols,
        bar_count=len(execution_times),
    )

    initial_capital = float(contract.costs["initial_capital"])
    fee_rate = float(contract.costs["fee_bps"]) / 10_000.0
    slippage_rate = float(contract.costs["slippage_bps"]) / 10_000.0
    margin_fraction = float(contract.portfolio["margin_fraction"])
    cash = np.full(candidate_count, initial_capital, dtype=float)
    quantities = np.zeros((candidate_count, symbol_count), dtype=float)
    entries = np.full((candidate_count, symbol_count), np.nan, dtype=float)
    latest_marks = market.marks[source_positions[0]].copy()
    alive = np.ones(candidate_count, dtype=bool)
    invalid_reason: list[str | None] = [None] * candidate_count
    invalid_timestamp: list[pd.Timestamp | None] = [None] * candidate_count
    equity_path = np.full((candidate_count, len(execution_times)), np.nan, dtype=float)
    hourly_fees = (np.zeros_like(equity_path) if include_equity_path else None)
    hourly_slippage = (np.zeros_like(equity_path) if include_equity_path else None)
    hourly_funding = (np.zeros_like(equity_path) if include_equity_path else None)
    funding_event_cashflows = ([[] for _ in range(candidate_count)]
                               if include_equity_path else None)
    previous_equity = np.full(candidate_count, initial_capital, dtype=float)
    total_fees = np.zeros(candidate_count, dtype=float)
    total_slippage = np.zeros(candidate_count, dtype=float)
    total_funding = np.zeros(candidate_count, dtype=float)
    traded_bars = np.zeros(candidate_count, dtype=np.int64)
    trade_count = np.zeros(candidate_count, dtype=np.int64)

    for bar_position, execution_time in enumerate(execution_times):
        active_candidates = np.flatnonzero(alive)
        if len(active_candidates) == 0:
            break
        inactive_now = inactive[bar_position]
        stranded = (quantities[:, inactive_now] != 0.0).any(axis=1) if inactive_now.any() else np.zeros(candidate_count, dtype=bool)
        stranded &= alive
        for candidate in np.flatnonzero(stranded):
            alive[candidate] = False
            invalid_reason[candidate] = "position_open_at_inactive_market_bar"
            invalid_timestamp[candidate] = pd.Timestamp(execution_time)

        # Funding at the execution open is charged before the scheduled trade.
        event_groups = funding_by_bar[bar_position]
        for event_time, event_symbols, rates, event_marks in event_groups:
            if event_time != execution_time:
                break
            alive_before_event = alive.copy()
            payments, event_equity, newly_breached, _ = _apply_funding_batch(
                cash, quantities, entries, latest_marks, event_symbols, rates,
                event_marks, margin_fraction, alive,
            )
            total_funding += payments
            if include_equity_path:
                hourly_funding[:, bar_position] += payments
                for symbol_index, rate, mark in zip(event_symbols, rates, event_marks):
                    event_payment = np.where(
                        alive_before_event,
                        -quantities[:, symbol_index] * rate * mark,
                        0.0,
                    )
                    for candidate in np.flatnonzero(alive_before_event):
                        funding_event_cashflows[candidate].append({
                            "timestamp": event_time,
                            "symbol": market.symbols[symbol_index],
                            "cashflow": float(event_payment[candidate]),
                        })
            for candidate in np.flatnonzero(newly_breached):
                invalid_reason[candidate] = (
                    "non_positive_account_equity_at_funding"
                    if not np.isfinite(event_equity[candidate]) or event_equity[candidate] <= 0.0
                    else "sampled_margin_guard_breach_at_funding"
                )
                invalid_timestamp[candidate] = event_time

        # Target quantities use the previous hourly closing equity and signal close.
        target_weights = schedule.get(bar_position)
        if target_weights is not None:
            signal_close = signal_closes[bar_position]
            selected_targets = target_weights != 0.0
            invalid_price = selected_targets & (~np.isfinite(signal_close)[None, :]
                                                | (signal_close[None, :] <= 0.0))
            bad_price_candidates = invalid_price.any(axis=1) & alive
            selection._require(not bad_price_candidates.any(),
                               "a nonzero validation target has no finite signal close")

            target_quantities = np.zeros_like(quantities)
            target_quantities[selected_targets] = (
                target_weights[selected_targets]
                * np.broadcast_to(previous_equity[:, None], target_weights.shape)[selected_targets]
                / np.broadcast_to(signal_close[None, :], target_weights.shape)[selected_targets]
            )
            bad_inactive_target = (selected_targets & inactive_now[None, :]).any(axis=1) & alive
            for candidate in np.flatnonzero(bad_inactive_target):
                alive[candidate] = False
                invalid_reason[candidate] = "nonzero_target_at_inactive_market_bar"
                invalid_timestamp[candidate] = pd.Timestamp(execution_time)

            delta = target_quantities - quantities
            trade_mask = (delta != 0.0) & alive[:, None]
            signed_slip = np.where(delta > 0.0, slippage_rate,
                                   np.where(delta < 0.0, -slippage_rate, 0.0))
            fill_prices = opens[bar_position][None, :] * (1.0 + signed_slip)
            fee = np.where(trade_mask, np.abs(delta) * fill_prices * fee_rate, 0.0)
            slippage_cost = np.where(
                trade_mask, np.abs(delta) * np.abs(fill_prices - opens[bar_position][None, :]), 0.0,
            )
            old = quantities.copy()
            old_entries = entries.copy()
            same_direction_or_entry = (old == 0.0) | (old * delta > 0.0)
            add_quantity = np.abs(old) + np.abs(delta)
            old_entry_notional = np.where(old == 0.0, 0.0,
                                          old_entries * np.abs(old))
            new_average = np.divide(
                old_entry_notional + fill_prices * np.abs(delta),
                add_quantity,
                out=fill_prices.copy(),
                where=add_quantity > 0.0,
            )
            closed = np.minimum(np.abs(old), np.abs(delta))
            realized = np.where(
                trade_mask & ~same_direction_or_entry,
                closed * np.sign(old) * (fill_prices - old_entries),
                0.0,
            )
            cash += (realized - fee).sum(axis=1)
            total_fees += fee.sum(axis=1)
            total_slippage += slippage_cost.sum(axis=1)
            if include_equity_path:
                hourly_fees[:, bar_position] += fee.sum(axis=1)
                hourly_slippage[:, bar_position] += slippage_cost.sum(axis=1)
            traded_bars += trade_mask.any(axis=1)
            trade_count += trade_mask.sum(axis=1)
            new_quantities = old + delta
            new_entries = np.where(
                same_direction_or_entry,
                new_average,
                np.where(new_quantities == 0.0, np.nan,
                         np.where(old * new_quantities < 0.0, fill_prices, old_entries)),
            )
            quantities = np.where(alive[:, None], new_quantities, quantities)
            entries = np.where(alive[:, None], new_entries, entries)

        # Later-in-hour funding is paid by post-trade positions, one event time at a time.
        for event_time, event_symbols, rates, event_marks in event_groups:
            if event_time == execution_time:
                continue
            alive_before_event = alive.copy()
            payments, event_equity, newly_breached, _ = _apply_funding_batch(
                cash, quantities, entries, latest_marks, event_symbols, rates,
                event_marks, margin_fraction, alive,
            )
            total_funding += payments
            if include_equity_path:
                hourly_funding[:, bar_position] += payments
                for symbol_index, rate, mark in zip(event_symbols, rates, event_marks):
                    event_payment = np.where(
                        alive_before_event,
                        -quantities[:, symbol_index] * rate * mark,
                        0.0,
                    )
                    for candidate in np.flatnonzero(alive_before_event):
                        funding_event_cashflows[candidate].append({
                            "timestamp": event_time,
                            "symbol": market.symbols[symbol_index],
                            "cashflow": float(event_payment[candidate]),
                        })
            for candidate in np.flatnonzero(newly_breached):
                invalid_reason[candidate] = (
                    "non_positive_account_equity_at_funding"
                    if not np.isfinite(event_equity[candidate]) or event_equity[candidate] <= 0.0
                    else "sampled_margin_guard_breach_at_funding"
                )
                invalid_timestamp[candidate] = event_time

        latest_marks[:] = closing_marks[bar_position]
        closing_equity, breached = _equity_and_margin(
            cash, quantities, entries, latest_marks, margin_fraction,
        )
        newly_breached = alive & breached
        for candidate in np.flatnonzero(newly_breached):
            invalid_reason[candidate] = (
                "non_positive_account_equity_at_hourly_close"
                if not np.isfinite(closing_equity[candidate]) or closing_equity[candidate] <= 0.0
                else "sampled_margin_guard_breach_at_hourly_close"
            )
            invalid_timestamp[candidate] = execution_time + HOUR
        alive[newly_breached] = False
        equity_path[alive, bar_position] = closing_equity[alive]
        previous_equity[alive] = closing_equity[alive]

    results: list[dict[str, Any]] = []
    for candidate in range(candidate_count):
        if not alive[candidate]:
            results.append({
                "validation_net_sharpe": None,
                "validation_net_return": None,
                "final_equity": None,
                "annualized_volatility": None,
                "total_fees": float(total_fees[candidate]),
                "total_slippage_cost": float(total_slippage[candidate]),
                "total_funding": float(total_funding[candidate]),
                "traded_bars": int(traded_bars[candidate]),
                "trade_count": int(trade_count[candidate]),
                "valid_objective": False,
                "status": invalid_reason[candidate],
                "failure_timestamp": invalid_timestamp[candidate],
            })
            continue
        equity_values = equity_path[candidate]
        selection._require(np.isfinite(equity_values).all(),
                           "an active candidate account has an incomplete equity path")
        returns = np.empty(len(equity_values), dtype=float)
        returns[0] = equity_values[0] / initial_capital - 1.0
        returns[1:] = equity_values[1:] / equity_values[:-1] - 1.0
        mean = float(np.mean(returns))
        volatility = (float(np.std(returns, ddof=1) * np.sqrt(HOURS_PER_YEAR))
                      if len(returns) > 1 else 0.0)
        sharpe = float(mean * HOURS_PER_YEAR / volatility) if volatility > 0.0 else 0.0
        final_equity = float(equity_values[-1])
        net_return = final_equity / initial_capital - 1.0
        valid_objective = bool(np.isfinite(sharpe))
        result = {
            "validation_net_sharpe": sharpe,
            "validation_net_return": net_return,
            "final_equity": final_equity,
            "annualized_volatility": volatility,
            "total_fees": float(total_fees[candidate]),
            "total_slippage_cost": float(total_slippage[candidate]),
            "total_funding": float(total_funding[candidate]),
            "traded_bars": int(traded_bars[candidate]),
            "trade_count": int(trade_count[candidate]),
            "valid_objective": valid_objective,
            "status": "evaluated" if valid_objective else "invalid_net_sharpe",
            "failure_timestamp": None,
        }
        if include_equity_path:
            result["hourly_equity"] = equity_values.copy()
            result["hourly_returns"] = returns.copy()
            result["hourly_fees"] = hourly_fees[candidate].copy()
            result["hourly_slippage_cost"] = hourly_slippage[candidate].copy()
            result["hourly_funding_cashflow"] = hourly_funding[candidate].copy()
            result["funding_event_cashflows"] = list(funding_event_cashflows[candidate])
        results.append(result)
    return results


def _batch_subset_results(
    subsets: list[tuple[int, ...]],
    *,
    train_stats: _RidgeTrainingStats,
    factor_names: list[str],
    fit_time: pd.Timestamp,
    market: _MarketArrays,
    funding: pd.DataFrame,
    contract: MultifactorContract,
    policy: dict[str, Any],
    validation_source_times: pd.DatetimeIndex,
    validation_values: np.ndarray,
    validation_complete: np.ndarray,
) -> dict[tuple[int, ...], tuple[float | None, float, list[dict[str, Any]]]]:
    if train_stats.invalid_status is not None:
        return {
            subset: (None, np.nan, [{"alpha": float(alpha),
                                     "validation_objective": None,
                                     "validation_net_sharpe": None,
                                     "validation_net_return": None,
                                     "valid_objective": False,
                                     "status": train_stats.invalid_status}
                                    for alpha in sorted(policy["alpha_grid"])])
            for subset in subsets
        }
    selection._require(train_stats.train_gram is not None and train_stats.train_rhs is not None,
                       "valid Ridge training stats must contain Gram and RHS arrays")

    betas: list[np.ndarray] = []
    owners: list[tuple[int, float]] = []
    for subset_position, subset in enumerate(subsets):
        columns = list(subset)
        gram = train_stats.train_gram[np.ix_(columns, columns)]
        rhs = train_stats.train_rhs[columns]
        for alpha_value in sorted(policy["alpha_grid"]):
            alpha = float(alpha_value)
            beta_subset = np.linalg.solve(gram + alpha * np.eye(len(columns)), rhs)
            beta = np.zeros(len(factor_names), dtype=float)
            beta[columns] = beta_subset
            betas.append(beta)
            owners.append((subset_position, alpha))
    trial_metrics = _simulate_r0_batch(
        np.asarray(betas, dtype=float), validation_values, validation_complete,
        validation_source_times, fit_time, market, funding, contract, policy,
    )
    trials_by_subset: dict[tuple[int, ...], list[dict[str, Any]]] = {
        subset: [] for subset in subsets
    }
    for (subset_position, alpha), metrics in zip(owners, trial_metrics):
        subset = subsets[subset_position]
        objective = (float(metrics["validation_net_sharpe"])
                     if metrics["valid_objective"] else None)
        trial = {
            "alpha": alpha,
            "validation_objective": objective,
            "validation_net_sharpe": metrics["validation_net_sharpe"],
            "validation_net_return": metrics["validation_net_return"],
            "final_equity": metrics["final_equity"],
            "annualized_volatility": metrics["annualized_volatility"],
            "total_fees": metrics["total_fees"],
            "total_slippage_cost": metrics["total_slippage_cost"],
            "total_funding": metrics["total_funding"],
            "traded_bars": metrics["traded_bars"],
            "trade_count": metrics["trade_count"],
            "valid_objective": metrics["valid_objective"],
            "status": metrics["status"],
            "failure_timestamp": metrics["failure_timestamp"],
        }
        trials_by_subset[subset].append(trial)
    results = {}
    for subset, trials in trials_by_subset.items():
        valid = [trial for trial in trials if trial["validation_objective"] is not None]
        if not valid:
            results[subset] = (None, np.nan, trials)
            continue
        selected = max(valid, key=lambda trial: (trial["validation_objective"], -trial["alpha"]))
        results[subset] = (selected["alpha"], selected["validation_objective"], trials)
    return results


def _fit_audit_base(
    timestamp: pd.Timestamp,
    matured_times: list[pd.Timestamp],
    train_times: list[pd.Timestamp],
    validation_times: list[pd.Timestamp],
    horizon: int,
) -> dict[str, Any]:
    return {
        "route": ROUTE,
        "timestamp": timestamp,
        "objective_name": OBJECTIVE_NAME,
        "horizon_hours": horizon,
        "train_source_start": matured_times[0] if matured_times else None,
        "train_source_end": matured_times[-1] if matured_times else None,
        "inner_train_source_start": train_times[0] if train_times else None,
        "inner_train_source_end": train_times[-1] if train_times else None,
        "validation_source_start": validation_times[0] if validation_times else None,
        "validation_source_end": validation_times[-1] if validation_times else None,
        "latest_train_label_maturity": (train_times[-1] + (horizon + 1) * HOUR
                                         if train_times else None),
        "train_periods": len(matured_times),
        "inner_train_periods": len(train_times),
        "validation_periods": len(validation_times),
    }


def _split_counts(
    data_times: pd.DatetimeIndex,
    timestamp: pd.Timestamp,
    horizon: int,
    policy: dict[str, Any],
) -> tuple[int, int, int]:
    mature_cutoff = timestamp - (horizon + 1) * HOUR
    window_start = timestamp - policy["fit_window_hours"] * HOUR
    validation_start = mature_cutoff - (policy["inner_validation_hours"] - 1) * HOUR
    train_cutoff = validation_start - (horizon + 1) * HOUR
    train_left = data_times.searchsorted(window_start, side="left")
    matured_right = data_times.searchsorted(mature_cutoff, side="right")
    train_right = data_times.searchsorted(train_cutoff, side="left")
    validation_left = data_times.searchsorted(validation_start, side="left")
    return (matured_right - train_left,
            train_right - train_left,
            matured_right - validation_left)


def _common_ready(
    data_times: pd.DatetimeIndex,
    timestamp: pd.Timestamp,
    position: int,
    horizon: int,
    policy: dict[str, Any],
    base_ready: bool,
) -> bool:
    if not base_ready or position < policy["correlation_window_hours"]:
        return False
    latest = timestamp - (horizon + 1) * HOUR
    earliest = latest - (policy["icir_window_hours"] - 1) * HOUR
    count = (data_times.searchsorted(latest, side="right")
             - data_times.searchsorted(earliest, side="left"))
    return count >= policy["icir_min_periods"]


def _last_r0_event(
    validation_end: pd.Timestamp,
    fit_time: pd.Timestamp,
    anchor: pd.Timestamp,
    rebalance_hours: int,
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    phase_residual = int((validation_end - anchor) / HOUR) % rebalance_hours
    source = validation_end - phase_residual * HOUR
    execution = source + HOUR
    return source, execution, int((fit_time - execution) / HOUR)


def generate_net_sharpe_scores(
    values: pd.DataFrame,
    opens: pd.Series,
    *,
    horizon_hours: int,
    policy: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    funding: pd.DataFrame,
    contract: MultifactorContract,
) -> selection.SelectionResult:
    """Select factor subsets by account-level net Sharpe on matured validation.

    Each inner-validation account starts flat at the first validation source's
    next open and ends at the model fit timestamp. Rebalance timestamps retain
    the R0 phase anchored to ``contract.start - 1h``. Only matured validation
    factor rows can issue targets; the last held target remains open through
    the fit-time mark. Every candidate's hourly equity path includes fees,
    slippage in fill prices, correctly timed funding, and sampled margin guards.
    """
    selection._validate_policy(policy)
    selection._require(isinstance(contract, MultifactorContract),
                       "contract must be a frozen multifactor research contract")
    selection._require(horizon_hours == contract.horizon_hours,
                       "horizon_hours must match the account contract")
    selection._require(contract.stage == "development",
                       "net-Sharpe scoring must retain the frozen development-account phase")
    selection._require(
        contract.portfolio["long_count"] == 2
        and contract.portfolio["short_count"] == 2
        and contract.portfolio["gross_exposure"] == 0.8
        and contract.portfolio["max_asset_weight"] == 0.2
        and contract.portfolio["rebalance_hours"] == horizon_hours,
        "net-Sharpe scoring requires the frozen R0 two-long/two-short portfolio",
    )
    timestamps, symbols = selection._validate_inputs(values, opens, horizon_hours)
    factor_names = list(values.columns)
    market, validated_funding = _market_arrays(frames, symbols, funding)
    factor_cube = values.to_numpy(dtype=float).reshape(len(timestamps), len(symbols), len(factor_names))
    complete = np.isfinite(factor_cube).all(axis=2)
    dataset = selection._make_labels(
        values, opens, timestamps, symbols, horizon_hours,
        policy["min_cross_section_symbols"],
    )
    data_times = pd.DatetimeIndex(sorted(dataset), name="timestamp")

    score = pd.Series(np.nan, index=values.index, name="score", dtype=float)
    fits: list[dict[str, Any]] = []
    shared_ready = np.zeros(len(timestamps), dtype=bool)
    state: tuple[list[int], np.ndarray] | None = None
    shared_model_active = False
    first_timestamp = timestamps[0]
    factor_time_positions = {pd.Timestamp(value): i for i, value in enumerate(timestamps)}

    for position, timestamp_value in enumerate(timestamps):
        timestamp = pd.Timestamp(timestamp_value)
        due = int((timestamp - first_timestamp) / HOUR) % policy["refit_every_hours"] == 0
        if due:
            _, train_count, validation_count = _split_counts(
                data_times, timestamp, horizon_hours, policy,
            )
            base_ready = (train_count >= policy["min_train_periods"]
                          and validation_count >= policy["min_inner_validation_periods"])
            common_ready = _common_ready(
                data_times, timestamp, position, horizon_hours, policy, base_ready,
            )
            shared_model_active = common_ready
            if not common_ready:
                state = None
                continue
            matured_times = selection._matured_window(data_times, timestamp, horizon_hours, policy)
            train_times, validation_times = selection._window_split(
                matured_times, timestamp, horizon_hours, policy,
            )
            validation_end = timestamp - (horizon_hours + 1) * HOUR
            validation_start = validation_end - (policy["inner_validation_hours"] - 1) * HOUR
            source_times = pd.date_range(validation_start, validation_end, freq="h", tz="UTC",
                                         name="timestamp")
            source_positions = np.asarray([factor_time_positions[pd.Timestamp(t)]
                                           for t in source_times], dtype=np.int64)
            validation_values = factor_cube[source_positions]
            validation_complete = complete[source_positions]
            account_start = validation_start + HOUR
            if account_start < market.index[0] + HOUR:
                state = None
                fits.append({**_fit_audit_base(timestamp, matured_times, train_times,
                                               validation_times, horizon_hours),
                             "status": "validation_begins_before_market_window"})
                continue
            if timestamp > market.index[-1] + HOUR:
                state = None
                fits.append({**_fit_audit_base(timestamp, matured_times, train_times,
                                               validation_times, horizon_hours),
                             "status": "fit_timestamp_exceeds_market_window"})
                continue

            cached_results: dict[tuple[int, ...], tuple[float | None, float,
                                                        list[dict[str, Any]]]] = {}
            x_train, y_train = selection._stack_sample(dataset, train_times)
            train_stats = _ridge_training_stats(x_train, y_train)

            def evaluate_batch(subsets: list[tuple[int, ...]]):
                evaluated = _batch_subset_results(
                    subsets,
                    train_stats=train_stats,
                    factor_names=factor_names,
                    fit_time=timestamp,
                    market=market,
                    funding=validated_funding,
                    contract=contract,
                    policy=policy,
                    validation_source_times=source_times,
                    validation_values=validation_values,
                    validation_complete=validation_complete,
                )
                cached_results.update(evaluated)
                return evaluated

            empty_scalar = lambda _subset: (None, np.nan, [])
            stepwise = selection._stepwise_select_with_evaluator(
                factor_names, policy, empty_scalar,
                objective_name=OBJECTIVE_NAME,
                batch_subset_evaluator=evaluate_batch,
            )
            selected = stepwise["selected"]
            selected_alpha = stepwise["selected_alpha"]
            coefficient = None
            selected_result = cached_results.get(tuple(selected)) if selected else None
            selected_trial = None
            if selected_result is not None and selected_alpha is not None:
                selected_trial = next(
                (trial for trial in selected_result[2]
                     if trial["alpha"] == selected_alpha and trial["valid_objective"]),
                None,
            )
            final_return_admitted = bool(
                selected_trial is not None
                and selected_trial["validation_net_return"] is not None
                and selected_trial["validation_net_return"] > 0.0
            )
            if (selected and selected_alpha is not None and selected_trial is not None
                    and final_return_admitted):
                x_fit, y_fit = selection._stack_sample(dataset, matured_times, selected)
                target_scale = float(np.sqrt(np.mean(np.square(y_fit))))
                selection._require(np.isfinite(target_scale) and target_scale > 0.0,
                                   "matured Ridge targets have zero or invalid RMS")
                coefficient = selection._ridge_coefficients(
                    x_fit, y_fit / target_scale, float(selected_alpha),
                )
                state = (selected, coefficient)
            else:
                state = None

            evaluation_start = validation_start + HOUR
            last_mark_bar = timestamp - HOUR
            used_funding = validated_funding.loc[
                (validated_funding["timestamp"] >= evaluation_start)
                & (validated_funding["timestamp"] < timestamp)
            ]
            audit_base = _fit_audit_base(timestamp, matured_times, train_times,
                                         validation_times, horizon_hours)
            last_r0_source, last_r0_execution, terminal_carry = _last_r0_event(
                validation_end, timestamp, contract.bounds[0] - HOUR,
                contract.portfolio["rebalance_hours"],
            )
            fits.append({
                **audit_base,
                "capacity": policy["pool_capacity"],
                "proposal_search_budget_total": stepwise["proposal_search_budget_total"],
                "proposal_evaluations": stepwise["proposal_evaluations"],
                "remaining_proposal_budget": stepwise["remaining_proposal_budget"],
                "proposal_budget_unit": stepwise["proposal_budget_unit"],
                "alpha_trial_evaluations": stepwise["alpha_trial_evaluations"],
                "unique_subset_evaluations": stepwise["unique_subset_evaluations"],
                "inner_training_sample_count": train_stats.sample_count,
                "inner_training_target_rms": train_stats.target_rms,
                "training_statistics_method": "numpy_einsum_full_column_ridge",
                "factor_order": factor_names,
                "selected_factors": [factor_names[index] for index in selected],
                "selection_order": stepwise["selection_order"],
                "selected_alpha": selected_alpha,
                "objective_name": OBJECTIVE_NAME,
                "validation_objective": (float(stepwise["objective"])
                                         if np.isfinite(stepwise["objective"]) else None),
                "validation_net_sharpe": (float(stepwise["objective"])
                                          if np.isfinite(stepwise["objective"]) else None),
                "validation_net_return": (selected_trial["validation_net_return"]
                                          if selected_trial else None),
                "final_net_return_admitted": final_return_admitted,
                "validation_final_equity": (selected_trial["final_equity"]
                                            if selected_trial else None),
                "validation_annualized_volatility": (selected_trial["annualized_volatility"]
                                                      if selected_trial else None),
                "validation_total_fees": (selected_trial["total_fees"] if selected_trial else None),
                "validation_total_slippage_cost": (selected_trial["total_slippage_cost"]
                                                   if selected_trial else None),
                "validation_total_funding": (selected_trial["total_funding"]
                                             if selected_trial else None),
                "validation_traded_bars": (selected_trial["traded_bars"] if selected_trial else None),
                "validation_trade_count": (selected_trial["trade_count"] if selected_trial else None),
                "validation_account_start": evaluation_start,
                "validation_account_end_exclusive": timestamp,
                "validation_source_window_start": validation_start,
                "validation_source_window_end": validation_end,
                "validation_source_window_bars": len(source_times),
                "common_model_ready": shared_model_active,
                "validation_last_bar_open": last_mark_bar,
                "validation_last_r0_target_source": last_r0_source,
                "validation_last_r0_target_execution": last_r0_execution,
                "validation_terminal_carry_hours": terminal_carry,
                "validation_last_mark_timestamp": timestamp,
                "validation_last_funding_timestamp": (used_funding["timestamp"].max()
                                                       if not used_funding.empty else None),
                "r0_rebalance_anchor": contract.bounds[0] - HOUR,
                "r0_rebalance_hours": contract.portfolio["rebalance_hours"],
                "validation_initial_capital": contract.costs["initial_capital"],
                "validation_fee_bps": contract.costs["fee_bps"],
                "validation_slippage_bps": contract.costs["slippage_bps"],
                "validation_portfolio": dict(contract.portfolio),
                "selection_rounds": stepwise["rounds"],
                "stop_reason": stepwise["stop_reason"],
                "fit_source_through": audit_base["train_source_end"],
                "fit_label_matured_through": audit_base["latest_train_label_maturity"],
                "coefficients": (dict(zip([factor_names[index] for index in selected],
                                           coefficient.tolist()))
                                 if coefficient is not None else None),
                "status": ("active" if state is not None else
                           "rejected_final_net_return_nonpositive" if selected_trial else
                           "cash_no_positive_sharpe_gain"),
            })

        row_matrix = factor_cube[position]
        complete_row = complete[position]
        enough_symbols = int(complete_row.sum()) >= policy["min_cross_section_symbols"]
        shared_ready[position] = bool(shared_model_active and enough_symbols)
        if state is not None and shared_model_active and enough_symbols:
            selected, coefficient = state
            scores = np.einsum("ni,i->n", row_matrix[complete_row][:, selected], coefficient)
            flat_start = position * len(symbols)
            score.iloc[flat_start + np.flatnonzero(complete_row)] = scores

    return selection.SelectionResult(
        scores={ROUTE: score},
        fits=fits,
        shared_model_ready=pd.Series(shared_ready, index=timestamps,
                                     name="shared_model_ready", dtype=bool),
    )
