"""Independent numerical and account-engine verification for combination90.

The search evaluator is deliberately treated as the system under test. This
module checks its frozen calendar inputs and coefficients, then replays a
deterministic sample of its candidate accounts through the production ledger.
It also exposes the E0 calibration driver; it never starts a forward or Paper
account.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import argparse
from pathlib import Path
import platform
import resource
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from . import multifactor_account as account
from . import multifactor_combination_search as combination
from . import multifactor_net_sharpe as net
from . import multifactor_selection as selection


HOUR = pd.Timedelta(hours=1)
SOURCE_HOURS = 90 * 24
HORIZON_HOURS = 24
MIN_SYMBOLS = 3
E0_BATCH_CANDIDATES = 1024
FIT_MONTHS = (202303, 202403, 202503)
AUDIT_TOLERANCES = {"rtol": 1e-10, "atol": 1e-7}
E0_ROUTES = (
    ("grid_equal", "grid", "equal"),
    ("grid_ridge", "grid", "ridge"),
    ("ga_ridge_seed0", "ga", "ridge"),
)


@dataclass(frozen=True)
class WindowAudit:
    """Provenance checks for a 90/90/90 calendar window."""

    training_source_hours: int
    validation_source_hours: int
    refit_source_hours: int
    training_last_label_maturity: pd.Timestamp
    validation_last_label_maturity: pd.Timestamp
    validation_account_start: pd.Timestamp
    validation_account_end_exclusive: pd.Timestamp
    validation_account_hours: int
    training_label_gap_hours: int
    validation_label_gap_to_fit_hours: int

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


def _utc_index(value: Any, name: str) -> pd.DatetimeIndex:
    if not isinstance(value, pd.DatetimeIndex) or value.tz is None:
        raise ValueError(f"{name} must be a timezone-aware DatetimeIndex")
    result = value.tz_convert("UTC")
    if result.empty:
        raise ValueError(f"{name} cannot be empty")
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise ValueError(f"{name} must be unique and increasing")
    if not result.equals(pd.date_range(result[0], result[-1], freq=HOUR, tz="UTC")):
        raise ValueError(f"{name} must be a complete hourly calendar")
    return result


def _check_calendar(grid: pd.DatetimeIndex, *, name: str, hours: int) -> pd.DatetimeIndex:
    normalized = _utc_index(grid, name)
    if len(normalized) != hours:
        raise ValueError(f"{name} has {len(normalized)} hours; expected {hours}")
    return normalized


def audit_windows(windows: Any, *, fit_time: pd.Timestamp,
                  horizon_hours: int = HORIZON_HOURS,
                  training_hours: int = SOURCE_HOURS,
                  validation_hours: int = SOURCE_HOURS,
                  refit_hours: int = SOURCE_HOURS) -> WindowAudit:
    """Check exact calendar widths and label maturity against the fit boundary.

    The training purge is strict: the last training label must mature before
    the first validation source, matching the Plan and the audited stepwise
    split rule.
    """
    fit_time = pd.Timestamp(fit_time)
    if fit_time.tzinfo is None:
        raise ValueError("fit_time must be timezone-aware")
    fit_time = fit_time.tz_convert("UTC")
    if fit_time != fit_time.floor("h"):
        raise ValueError("fit_time must be on a UTC hour boundary")
    if type(horizon_hours) is not int or horizon_hours <= 0:
        raise ValueError("horizon_hours must be a positive integer")

    training = _check_calendar(windows.training, name="training", hours=training_hours)
    validation = _check_calendar(windows.validation, name="validation", hours=validation_hours)
    refit = _check_calendar(windows.refit, name="refit", hours=refit_hours)
    label_delay = (horizon_hours + 1) * HOUR
    expected_validation_end = fit_time - label_delay
    if validation[-1] != expected_validation_end:
        raise ValueError("validation source window does not end at the last mature source hour")
    if refit[-1] != expected_validation_end:
        raise ValueError("refit source window does not end at the last mature source hour")
    if training[-1] + label_delay != validation[0] - HOUR:
        raise ValueError("training label purge must leave exactly one full hour before validation starts")
    if training[-1] >= validation[0]:
        raise ValueError("training and validation source calendars overlap")
    if refit[-1] + label_delay > fit_time:
        raise ValueError("refit labels extend beyond the fit time")

    account_start = validation[0] + HOUR
    account_hours = int((fit_time - account_start) / HOUR)
    if account_hours <= 0:
        raise ValueError("validation account window is empty")
    return WindowAudit(
        training_source_hours=len(training),
        validation_source_hours=len(validation),
        refit_source_hours=len(refit),
        training_last_label_maturity=training[-1] + label_delay,
        validation_last_label_maturity=validation[-1] + label_delay,
        validation_account_start=account_start,
        validation_account_end_exclusive=fit_time,
        validation_account_hours=account_hours,
        training_label_gap_hours=int((validation[0] - training[-1] - label_delay) / HOUR),
        validation_label_gap_to_fit_hours=int((fit_time - validation[-1] - label_delay) / HOUR),
    )


def first_fit_in_months(fit_times: Iterable[pd.Timestamp],
                        months: Iterable[int] = FIT_MONTHS) -> dict[str, pd.Timestamp]:
    """Choose the earliest eligible weekly fit in each requested calendar month."""
    values = [pd.Timestamp(value) for value in fit_times]
    if any(value.tzinfo is None for value in values):
        raise ValueError("fit times must be timezone-aware")
    ordered = sorted(value.tz_convert("UTC") for value in values)
    if len(ordered) != len(set(ordered)):
        raise ValueError("eligible fit times must be unique")
    selected = {}
    for month in months:
        if type(month) is not int or month % 100 not in range(1, 13):
            raise ValueError(f"invalid YYYYMM month: {month}")
        match = next((value for value in ordered if value.year * 100 + value.month == month), None)
        if match is None:
            raise ValueError(f"no eligible weekly fit in {month}")
        selected[str(month)] = match
    return selected


def _training_arrays_from_prices(
    factors: pd.DataFrame,
    opens: pd.Series,
    training_times: pd.DatetimeIndex,
    *,
    horizon_hours: int,
    min_symbols: int,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Rebuild centered, causal regression samples from raw indexed inputs."""
    selection._require(isinstance(factors.index, pd.MultiIndex)
                       and list(factors.index.names) == ["timestamp", "symbol"],
                       "factors must use a (timestamp, symbol) index")
    selection._require(isinstance(opens, pd.Series) and opens.index.equals(factors.index),
                       "opens must be aligned to factors")
    timestamps = factors.index.get_level_values("timestamp")
    symbols = list(pd.Index(factors.index.get_level_values("symbol")).unique().sort_values())
    delay = (horizon_hours + 1) * HOUR
    price_times = training_times.union(training_times + HOUR)
    price_times = price_times.union(training_times + delay).sort_values()
    factor_rows = factors.loc[timestamps.isin(training_times)]
    price_rows = opens.loc[opens.index.get_level_values("timestamp").isin(price_times)]
    factor_panel = factor_rows.unstack("symbol").reindex(columns=pd.MultiIndex.from_product(
        [factors.columns, symbols]))
    price_panel = price_rows.unstack("symbol").reindex(columns=symbols)
    factor_panel.columns = factor_panel.columns.set_names([None, "symbol"])
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    included: list[pd.Timestamp] = []
    for source_time in training_times:
        source_time = pd.Timestamp(source_time)
        entry_time, exit_time = source_time + HOUR, source_time + delay
        if source_time not in factor_panel.index or entry_time not in price_panel.index or exit_time not in price_panel.index:
            continue
        factor_row = factor_panel.loc[source_time].to_numpy(dtype=float).reshape(len(factors.columns), len(symbols)).T
        entry = price_panel.loc[entry_time].to_numpy(dtype=float)
        exit_price = price_panel.loc[exit_time].to_numpy(dtype=float)
        valid = (np.isfinite(factor_row).all(axis=1) & np.isfinite(entry)
                 & np.isfinite(exit_price) & (entry > 0.0) & (exit_price > 0.0))
        if int(valid.sum()) < min_symbols:
            continue
        x = factor_row[valid]
        y = exit_price[valid] / entry[valid] - 1.0
        x_rows.append(x - x.mean(axis=0, keepdims=True))
        y_rows.append(y - y.mean())
        included.append(source_time)
    if not x_rows:
        return np.empty((0, len(factors.columns))), np.empty(0), included
    return np.concatenate(x_rows), np.concatenate(y_rows), included


def audit_training_sample(prepared: Any, windows: Any, *, x_train: np.ndarray,
                          y_train: np.ndarray, training_source_times: Iterable[pd.Timestamp] | None = None,
                          calendar_times: pd.DatetimeIndex | None = None,
                          horizon_hours: int = HORIZON_HOURS,
                          min_symbols: int = MIN_SYMBOLS,
                          rtol: float = 1e-12, atol: float = 1e-12) -> dict[str, Any]:
    """Rebuild training labels and design rows, excluding every later source."""
    source_grid = windows.training if calendar_times is None else calendar_times
    training = _check_calendar(source_grid, name="training", hours=len(source_grid))
    expected_x, expected_y, included = _training_arrays_from_prices(
        prepared.factors, prepared.opens, training,
        horizon_hours=horizon_hours, min_symbols=min_symbols,
    )
    observed_x = np.asarray(x_train, dtype=float)
    observed_y = np.asarray(y_train, dtype=float)
    if observed_x.shape != expected_x.shape or observed_y.shape != expected_y.shape:
        raise AssertionError(
            f"training sample shape mismatch: x {observed_x.shape}/{expected_x.shape}, "
            f"y {observed_y.shape}/{expected_y.shape}"
        )
    if not np.allclose(observed_x, expected_x, rtol=rtol, atol=atol):
        raise AssertionError("training predictors differ from the causal common-mask reconstruction")
    if not np.allclose(observed_y, expected_y, rtol=rtol, atol=atol):
        raise AssertionError("training targets differ from the causal next-open return reconstruction")
    supplied = None if training_source_times is None else [pd.Timestamp(value) for value in training_source_times]
    if supplied is not None and supplied != included:
        raise AssertionError("training source timestamps differ from rebuilt valid label sources")
    return {
        "status": "passed",
        "calendar_source_hours": len(training),
        "eligible_label_source_hours": len(included),
        "sample_rows": int(len(expected_y)),
        "factor_count": int(expected_x.shape[1]),
        "label_rule": "open[t+1] to open[t+horizon+1], cross-section demeaned",
        "latest_training_label_maturity": (max(included) + (horizon_hours + 1) * HOUR
                                            if included else None),
        "used_sources_at_or_after_validation": 0,
    }


def audit_validation_mask(evaluator: Any, windows: Any, *, factor_count: int,
                          expected_hours: int = SOURCE_HOURS) -> dict[str, Any]:
    """Require all calendar source hours and one shared all-factor asset mask."""
    source_times = _check_calendar(evaluator.times, name="validation source times", hours=expected_hours)
    expected_times = _check_calendar(windows.validation, name="validation window", hours=expected_hours)
    if not source_times.equals(expected_times):
        raise AssertionError("validation scorer omitted or shifted calendar source hours")
    values = np.asarray(evaluator.values, dtype=float)
    if values.ndim != 3 or values.shape[0] != expected_hours or values.shape[2] != factor_count:
        raise AssertionError("validation cube must retain every source hour and every candidate factor")
    complete = np.asarray(evaluator.complete, dtype=bool)
    if complete.shape != values.shape[:2]:
        raise AssertionError("common validation mask shape differs from values")
    expected_complete = np.isfinite(values).all(axis=2)
    if not np.array_equal(complete, expected_complete):
        raise AssertionError("candidate accounts do not share the full factor-completeness mask")
    return {
        "status": "passed",
        "calendar_source_hours": int(len(source_times)),
        "factor_count": int(factor_count),
        "symbol_count": int(values.shape[1]),
        "complete_asset_hours": int(complete.sum()),
        "common_mask_sha256": hashlib.sha256(np.packbits(complete).tobytes()).hexdigest(),
    }


def audit_training_evaluator(evaluator: Any, windows: Any, training_cube: np.ndarray,
                            *, factor_count: int,
                            expected_hours: int = SOURCE_HOURS) -> dict[str, Any]:
    """Prove clustering and single-factor prescreen consume only training hours."""
    source_times = _check_calendar(evaluator.times, name="training evaluator source times",
                                   hours=expected_hours)
    expected_times = _check_calendar(windows.training, name="training window",
                                     hours=expected_hours)
    if not source_times.equals(expected_times):
        raise AssertionError("training prescreen omitted or shifted calendar source hours")
    values = np.asarray(evaluator.values, dtype=float)
    cube = np.asarray(training_cube, dtype=float)
    if values.shape != cube.shape or values.ndim != 3 or values.shape[2] != factor_count:
        raise AssertionError("cluster and prescreen cubes must contain the complete training factor pool")
    if not np.allclose(values, cube, rtol=0.0, atol=0.0, equal_nan=True):
        raise AssertionError("clustering and prescreen use different training factor values")
    complete = np.asarray(evaluator.complete, dtype=bool)
    expected_complete = np.isfinite(values).all(axis=2)
    if complete.shape != expected_complete.shape or not np.array_equal(complete, expected_complete):
        raise AssertionError("training prescreen does not use the shared full-pool complete-factor mask")
    expected_end = expected_times[-1] + (HORIZON_HOURS + 1) * HOUR
    if pd.Timestamp(evaluator.fit_time).tz_convert("UTC") > expected_end + HOUR:
        raise AssertionError("training prescreen account extends past its training label boundary")
    return {
        "status": "passed",
        "calendar_source_hours": int(len(source_times)),
        "factor_count": int(factor_count),
        "symbol_count": int(values.shape[1]),
        "complete_asset_hours": int(complete.sum()),
        "common_mask_sha256": hashlib.sha256(np.packbits(complete).tobytes()).hexdigest(),
        "prescreen_account_end_exclusive": pd.Timestamp(evaluator.fit_time),
        "latest_training_label_maturity": expected_end,
    }


def _coefficients_from_sample(method: str, subset: tuple[int, ...], alpha: float | None,
                              x_train: np.ndarray, y_train: np.ndarray,
                              factor_count: int) -> np.ndarray:
    result = np.zeros(factor_count, dtype=float)
    if method == "equal":
        result[list(subset)] = 1.0 / len(subset)
        return result
    if method != "ridge" or alpha is None:
        raise ValueError("E0 coefficient audit supports equal and Ridge only")
    stats = net._ridge_training_stats(x_train, y_train)
    if stats.invalid_status is not None:
        raise AssertionError(f"training regression is invalid: {stats.invalid_status}")
    columns = list(subset)
    result[columns] = np.linalg.solve(
        stats.train_gram[np.ix_(columns, columns)] + float(alpha) * np.eye(len(columns)),
        stats.train_rhs[columns],
    )
    return result


def _expected_coefficients(evaluator: Any, subset: tuple[int, ...], alpha: float | None) -> np.ndarray:
    return _coefficients_from_sample(
        evaluator.method, subset, alpha, evaluator.x_train, evaluator.y_train,
        evaluator.factor_count,
    )


def audit_refit_coefficients(observed_coefficients: np.ndarray, method: str,
                             subset: Iterable[int], alpha: float | None,
                             *, x_refit: np.ndarray, y_refit: np.ndarray,
                             factor_names: list[str], refit_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the post-selection coefficients use only the latest matured 90d."""
    subset = tuple(int(value) for value in subset)
    coefficients = np.asarray(observed_coefficients, dtype=float)
    expected = _coefficients_from_sample(method, subset, alpha, x_refit, y_refit,
                                         len(factor_names))
    if coefficients.shape != expected.shape or not np.allclose(
        coefficients, expected, rtol=1e-12, atol=1e-12,
    ):
        raise AssertionError("post-selection coefficients differ from the refit-window solution")
    outside = np.ones(len(factor_names), dtype=bool)
    outside[list(subset)] = False
    if np.any(np.abs(coefficients[outside]) > 1e-14):
        raise AssertionError("post-selection refit has nonzero weights outside the chosen subset")
    return {
        "status": "passed",
        "method": method,
        "subset": list(subset),
        "factor_names": [factor_names[index] for index in subset],
        "alpha_frozen_from_validation": alpha,
        "sample_count": int(len(y_refit)),
        "coefficients": coefficients.tolist(),
        "coefficient_fit_source": "windows.refit only",
        "runner_refit_metadata": dict(refit_metadata),
    }


def _assert_same_training_fit(evaluator: Any, x_train: np.ndarray, y_train: np.ndarray,
                              label: str) -> None:
    observed_x = np.asarray(evaluator.x_train, dtype=float)
    observed_y = np.asarray(evaluator.y_train, dtype=float)
    if observed_x.shape != np.asarray(x_train).shape or not np.allclose(
        observed_x, x_train, rtol=0.0, atol=0.0, equal_nan=True,
    ):
        raise AssertionError(f"{label} coefficients use different training predictors")
    if observed_y.shape != np.asarray(y_train).shape or not np.allclose(
        observed_y, y_train, rtol=0.0, atol=0.0, equal_nan=True,
    ):
        raise AssertionError(f"{label} coefficients use different training targets")


def _compare_numeric(name: str, observed: Any, expected: Any,
                     checks: dict[str, Any], *, rtol: float, atol: float) -> None:
    observed_array = np.asarray(observed, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    if observed_array.shape != expected_array.shape:
        raise AssertionError(f"{name} shape mismatch: {observed_array.shape} != {expected_array.shape}")
    difference = np.abs(observed_array - expected_array)
    maximum = float(np.nanmax(difference)) if difference.size and np.isfinite(difference).any() else 0.0
    checks[name] = {"points": int(difference.size), "max_abs_difference": maximum}
    if not np.allclose(observed_array, expected_array, rtol=rtol, atol=atol, equal_nan=True):
        raise AssertionError(f"{name} differs from full account engine; max abs={maximum}")


def _compare_cached_trial_value(name: str, cached: Any, fast: Any, reference: Any,
                                checks: dict[str, Any], *, rtol: float, atol: float) -> None:
    if cached is None:
        if fast is not None or reference is not None:
            raise AssertionError(f"cached {name} is missing while replay produced a value")
        checks[f"cached_{name}"] = {"cached": None, "fast_replay": None, "account_engine": None}
        return
    values = {"cached": float(cached), "fast_replay": float(fast),
              "account_engine": float(reference)}
    if not np.isclose(values["cached"], values["fast_replay"], rtol=rtol, atol=atol):
        raise AssertionError(f"cached {name} differs from its independent batch replay")
    if not np.isclose(values["cached"], values["account_engine"], rtol=rtol, atol=atol):
        raise AssertionError(f"cached {name} differs from the full account engine")
    checks[f"cached_{name}"] = {
        **values,
        "cached_vs_fast_abs_difference": abs(values["cached"] - values["fast_replay"]),
        "cached_vs_account_abs_difference": abs(values["cached"] - values["account_engine"]),
    }


def _reference_account(evaluator: Any, coefficients: np.ndarray,
                       market_frames: Mapping[str, pd.DataFrame]):
    source_times = evaluator.times
    fit_time = evaluator.fit_time
    contract = evaluator.contract
    symbols = evaluator.market.symbols
    score_values = np.einsum(
        "tsf,f->ts", np.where(np.isfinite(evaluator.values), evaluator.values, 0.0),
        coefficients, optimize=True,
    )
    score_values[~evaluator.complete] = np.nan
    score_frame = pd.DataFrame(score_values, index=source_times, columns=symbols)
    account_start = source_times[0] + HOUR
    every_hour = account.generate_rank_targets(
        score_frame,
        long_count=contract.portfolio["long_count"],
        short_count=contract.portfolio["short_count"],
        gross_exposure=contract.portfolio["gross_exposure"],
        max_asset_weight=contract.portfolio["max_asset_weight"],
        rebalance_hours=1,
        start=account_start,
    )
    anchor = contract.bounds[0] - HOUR
    scheduled = [timestamp for timestamp in source_times
                 if (timestamp.value - anchor.value) % (
                     contract.portfolio["rebalance_hours"] * HOUR.value
                 ) == 0]
    if not scheduled:
        raise AssertionError("validation window has no globally anchored R0 source")
    targets = every_hour.loc[scheduled].copy()
    if source_times[0] not in targets.index:
        targets.loc[source_times[0]] = 0.0
        targets = targets.sort_index(kind="stable")

    expected_index = pd.date_range(source_times[0], fit_time - HOUR, freq=HOUR, tz="UTC")
    sliced_frames = {symbol: frame.reindex(expected_index) for symbol, frame in market_frames.items()}
    scoped_funding = evaluator.funding.loc[
        (evaluator.funding["timestamp"] >= account_start)
        & (evaluator.funding["timestamp"] < fit_time)
    ]
    result = account.run_perpetual_account(
        sliced_frames, targets, scoped_funding,
        initial_capital=contract.costs["initial_capital"],
        fee_bps=contract.costs["fee_bps"],
        slippage_bps=contract.costs["slippage_bps"],
        start=account_start,
        end=fit_time,
        margin_fraction=contract.portfolio["margin_fraction"],
    )
    return result, targets, score_frame, expected_index, scheduled


def _account_case_counts(reference: Any, targets: pd.DataFrame, frames: Mapping[str, pd.DataFrame],
                         expected_index: pd.DatetimeIndex,
                         scheduled: list[pd.Timestamp]) -> dict[str, Any]:
    orders = reference.orders
    old = orders["current_quantity"].to_numpy(dtype=float)
    new = orders["target_quantity"].to_numpy(dtype=float)
    inactive_bars = 0
    inactive_open_positions = 0
    if not reference.positions.empty:
        quantities = reference.positions.pivot(index="timestamp", columns="symbol", values="quantity")
    else:
        quantities = pd.DataFrame(index=expected_index, columns=sorted(frames), dtype=float).fillna(0.0)
    for symbol, frame in frames.items():
        if "inactive" not in frame:
            continue
        scoped = frame.reindex(expected_index)
        inactive = scoped["inactive"].fillna(False).to_numpy(dtype=bool)
        inactive_bars += int(inactive.sum())
        held = quantities.reindex(index=expected_index + HOUR, columns=[symbol])[symbol].fillna(0.0).to_numpy(dtype=float)
        inactive_open_positions += int((inactive & (held != 0.0)).sum())
    funding_times = reference.funding_events["timestamp"]
    return {
        "entries": int(((old == 0.0) & (new != 0.0)).sum()),
        "exits": int(((old != 0.0) & (new == 0.0)).sum()),
        "direction_reversals": int((old * new < 0.0).sum()),
        "target_rows_with_unfilled_cash_seats": int(
            (targets.loc[scheduled].ne(0.0).sum(axis=1) < 4).sum()
        ),
        "inactive_asset_bars": inactive_bars,
        "inactive_open_positions": inactive_open_positions,
        "funding_at_execution_open": int((funding_times == funding_times.dt.floor("h")).sum()),
        "funding_later_within_hour": int((funding_times > funding_times.dt.floor("h")).sum()),
        "account_hours_with_zero_gross_exposure": int(
            (reference.ledger["gross_notional"] == 0.0).sum()
        ),
    }


def verify_candidate_account(evaluator: Any, subset: Iterable[int], alpha: float | None,
                             market_frames: Mapping[str, pd.DataFrame],
                             factor_names: list[str],
                             *, rtol: float = AUDIT_TOLERANCES["rtol"],
                             atol: float = AUDIT_TOLERANCES["atol"]) -> dict[str, Any]:
    """Compare one cached candidate trial with an independent full ledger replay."""
    subset = tuple(int(value) for value in subset)
    if not subset or tuple(sorted(set(subset))) != subset:
        raise ValueError("subset must be nonempty, sorted, and unique")
    coefficients = np.asarray(evaluator.coefficients(subset, alpha), dtype=float)
    expected_coefficients = _expected_coefficients(evaluator, subset, alpha)
    if coefficients.shape != expected_coefficients.shape or not np.allclose(
        coefficients, expected_coefficients, rtol=1e-12, atol=1e-12,
    ):
        raise AssertionError("candidate coefficients are not fitted from the frozen training window")
    outside = np.ones(evaluator.factor_count, dtype=bool)
    outside[list(subset)] = False
    if np.any(np.abs(coefficients[outside]) > 1e-14):
        raise AssertionError("candidate has nonzero weights outside its factor subset")
    cache_row = evaluator.cache[subset]
    cached_trials = [trial for trial in cache_row[2] if trial.get("alpha") == alpha]
    if len(cached_trials) != 1:
        raise AssertionError("the audited alpha does not identify exactly one cached batch trial")
    cached_trial = cached_trials[0]

    fast = net._simulate_r0_batch(
        coefficients.reshape(1, -1), evaluator.values, evaluator.complete,
        evaluator.times, evaluator.fit_time, evaluator.market, evaluator.funding,
        evaluator.contract, {"min_cross_section_symbols": MIN_SYMBOLS},
        include_equity_path=True,
    )[0]
    reference, targets, _scores, expected_index, scheduled = _reference_account(
        evaluator, coefficients, market_frames,
    )
    checks: dict[str, Any] = {}
    ledger = reference.ledger
    for name, observed, expected in (
        ("hourly_equity", fast.get("hourly_equity"), ledger["equity"].to_numpy()),
        ("hourly_returns", fast.get("hourly_returns"), ledger["return"].to_numpy()),
        ("hourly_fees", fast.get("hourly_fees"), ledger["fees"].to_numpy()),
        ("hourly_slippage_cost", fast.get("hourly_slippage_cost"), ledger["slippage_cost"].to_numpy()),
        ("hourly_funding_cashflow", fast.get("hourly_funding_cashflow"), ledger["funding_cashflow"].to_numpy()),
    ):
        _compare_numeric(name, observed, expected, checks, rtol=rtol, atol=atol)

    traced_events = fast.get("funding_event_cashflows", [])
    ref_events = reference.funding_events
    traced_keys = [(pd.Timestamp(row["timestamp"]), row["symbol"]) for row in traced_events]
    reference_keys = list(zip(ref_events["timestamp"], ref_events["symbol"]))
    if traced_keys != reference_keys:
        raise AssertionError("funding event timestamps or symbols differ from the account engine")
    _compare_numeric(
        "funding_event_cashflows",
        np.asarray([row["cashflow"] for row in traced_events]),
        ref_events["cashflow"].to_numpy(), checks, rtol=rtol, atol=atol,
    )
    for name, observed, expected in (
        ("net_return", fast["validation_net_return"], reference.metrics["net_return"]),
        ("net_sharpe", fast["validation_net_sharpe"], reference.metrics["sharpe_ratio"]),
        ("total_fees", fast["total_fees"], reference.metrics["total_fees"]),
        ("total_slippage_cost", fast["total_slippage_cost"], reference.metrics["total_slippage_cost"]),
        ("total_funding", fast["total_funding"], reference.metrics["total_funding"]),
        ("final_equity", fast["final_equity"], reference.metrics["final_equity"]),
    ):
        error = abs(float(observed) - float(expected))
        checks[name] = {"module": float(observed), "account_engine": float(expected),
                        "abs_difference": error}
        if not np.isclose(observed, expected, rtol=rtol, atol=atol):
            raise AssertionError(f"{name} differs from account engine by {error}")

    cached_reference_values = (
        ("net_sharpe", "validation_net_sharpe", fast["validation_net_sharpe"], reference.metrics["sharpe_ratio"]),
        ("net_return", "validation_net_return", fast["validation_net_return"], reference.metrics["net_return"]),
        ("final_equity", "final_equity", fast["final_equity"], reference.metrics["final_equity"]),
        ("annualized_volatility", "annualized_volatility", fast["annualized_volatility"],
         reference.metrics["annualized_volatility"]),
        ("total_fees", "total_fees", fast["total_fees"], reference.metrics["total_fees"]),
        ("total_slippage_cost", "total_slippage_cost", fast["total_slippage_cost"],
         reference.metrics["total_slippage_cost"]),
        ("total_funding", "total_funding", fast["total_funding"], reference.metrics["total_funding"]),
        ("trade_count", "trade_count", fast["trade_count"], len(reference.fills)),
        ("traded_bars", "traded_bars", fast["traded_bars"],
         int(reference.fills["timestamp"].nunique()) if not reference.fills.empty else 0),
    )
    for check_name, trial_name, fast_value, account_value in cached_reference_values:
        _compare_cached_trial_value(
            check_name, cached_trial.get(trial_name), fast_value, account_value,
            checks, rtol=rtol, atol=atol,
        )
    if bool(cached_trial["valid_objective"]) is not bool(fast["valid_objective"]):
        raise AssertionError("cached batch objective-validity gate differs from the replay")
    checks["cached_valid_objective"] = {
        "cached": bool(cached_trial["valid_objective"]),
        "fast_replay": bool(fast["valid_objective"]),
    }

    fit_time = pd.Timestamp(evaluator.fit_time)
    account_start = evaluator.times[0] + HOUR
    funding_max = reference.funding_events["timestamp"].max() if len(reference.funding_events) else None
    if funding_max is not None and not funding_max < fit_time:
        raise AssertionError("funding at or beyond fit_time entered the validation ledger")
    if expected_index[-1] != fit_time - HOUR:
        raise AssertionError("account market slice extends beyond the fit-time boundary")
    case_counts = _account_case_counts(reference, targets, market_frames, expected_index, scheduled)
    if case_counts["inactive_open_positions"] != 0:
        raise AssertionError("account retained a position through an inactive market bar")
    return {
        "status": "passed",
        "subset": list(subset),
        "factor_names": [factor_names[index] for index in subset],
        "alpha": alpha,
        "cached_validation_net_sharpe": cached_trial.get("validation_net_sharpe"),
        "coefficients": coefficients.tolist(),
        "fast_simulator_status": fast["status"],
        "net_return": float(reference.metrics["net_return"]),
        "net_sharpe": float(reference.metrics["sharpe_ratio"]),
        "trade_count": int(len(reference.fills)),
        "account_hours": int(len(ledger)),
        "account_start": account_start,
        "account_end_exclusive": fit_time,
        "last_account_bar_open": expected_index[-1],
        "last_mark_timestamp": fit_time,
        "last_funding_timestamp": funding_max,
        "account_case_counts": case_counts,
        "checks": checks,
    }


def deterministic_audit_subsets(evaluator: Any, winner: tuple[int, ...] | None,
                                *, sample_size: int = 8, seed_material: str = "") -> list[tuple[int, ...]]:
    """Return the winner plus a stable pseudorandom sample from cached subsets."""
    if type(sample_size) is not int or sample_size < 0:
        raise ValueError("sample_size must be a non-negative integer")
    cached = sorted(evaluator.cache)
    if winner is None or tuple(winner) not in evaluator.cache:
        raise ValueError("the selected winner must exist in the evaluator cache")
    alternatives = [subset for subset in cached if subset != tuple(winner)]
    count = min(sample_size, len(alternatives))
    seed_bytes = hashlib.sha256(seed_material.encode("utf-8")).digest()
    seed = int.from_bytes(seed_bytes[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(alternatives), size=count, replace=False).tolist()) if count else []
    return [tuple(winner), *(alternatives[index] for index in chosen)]


def process_peak_rss_bytes() -> int:
    """Return the current process high-water resident set size in bytes."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_source_hashes() -> dict[str, str]:
    """Freeze the Plan, evaluator, production account, and E0 verifier sources."""
    root = Path(__file__).resolve().parents[4]
    paths = {
        "plan": root / "docs/plans/多因子组合算法90天选优实验Plan_20261003.md",
        "combination_research": root / "src/crypto_quant/research/strategy_research/multifactor_combination_research.py",
        "combination_search": Path(combination.__file__).resolve(),
        "factor_selection": Path(selection.__file__).resolve(),
        "net_sharpe_evaluator": Path(net.__file__).resolve(),
        "production_account": Path(account.__file__).resolve(),
        "selection_research_loader": root / "src/crypto_quant/research/strategy_research/multifactor_selection_research.py",
        "multifactor_contracts": root / "src/crypto_quant/research/strategy_research/multifactor_contracts.py",
        "combination_verify": Path(__file__).resolve(),
        "combination_search_tests": root / "tests/test_multifactor_combination_search.py",
        "open_precision_tests": root / "tests/test_multifactor_combination_open_precision.py",
        "account_tests": root / "tests/test_multifactor_account.py",
        "net_sharpe_tests": root / "tests/test_multifactor_net_sharpe.py",
        "existing_net_sharpe_verifier": root / "experiments/strategy_research/stepwise_net_sharpe30_20261003/verify_net_sharpe.py",
        "focused_verifier_tests": root / "tests/test_multifactor_combination_verify.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E0 audit source is missing: {missing[0]}")
    return {name: sha256_file(path) for name, path in paths.items()}


def prepared_input_hashes(prepared: Any) -> dict[str, str]:
    actual = {name: sha256_file(path) for name, path in sorted(prepared.hash_paths.items())}
    if actual != prepared.source_hashes:
        changed = [name for name in prepared.source_hashes
                   if actual.get(name) != prepared.source_hashes[name]]
        raise ValueError(f"frozen E0 source inputs changed: {changed[:5]}")
    return actual


def _json_default(value: Any):
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_safe(value: Any):
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), default=_json_default, indent=2,
                   ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_candidate_scores(path: str | Path, evaluator: Any, factor_names: list[str]) -> float:
    """Write every unique candidate and every alpha trial as stable gzip JSONL."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with gzip.open(destination, "wt", encoding="utf-8", newline="\n") as stream:
        for subset in sorted(evaluator.cache):
            alpha, objective, trials = evaluator.cache[subset]
            row = {
                "subset": list(subset),
                "factor_names": [factor_names[index] for index in subset],
                "selected_alpha": alpha,
                "validation_net_sharpe": objective if np.isfinite(objective) else None,
                "trials": trials,
            }
            stream.write(json.dumps(_json_safe(row), default=_json_default, ensure_ascii=False,
                                    allow_nan=False, sort_keys=True) + "\n")
    return time.perf_counter() - started


def _evaluator_audit_timing(evaluator: Any) -> dict[str, Any]:
    audit = evaluator.audit()
    total = float(audit.get("evaluation_seconds", 0.0))
    fit = float(audit.get("coefficient_fit_seconds", 0.0))
    scoring = float(audit.get("account_scoring_seconds", 0.0))
    residual = total - fit - scoring
    return {
        **audit,
        "cache_bookkeeping_seconds_estimate": max(0.0, residual),
        "timing_partition_note": "evaluation_seconds minus coefficient_fit_seconds and account_scoring_seconds; includes evaluator overhead",
    }


def _edge_case_coverage(windows: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [audit for window in windows for route in window["routes"]
              for audit in route["account_audits"]]
    totals = {}
    metrics = {
        "delisting": ("inactive_asset_bars",),
        "cash": ("target_rows_with_unfilled_cash_seats", "account_hours_with_zero_gross_exposure"),
        "switch": ("entries", "exits", "direction_reversals"),
        "funding": ("funding_at_execution_open", "funding_later_within_hour"),
    }
    targeted_tests = {
        "delisting": [
            "tests/test_multifactor_account.py::test_announced_inactive_asset_stays_flat_without_synthetic_prices",
            "tests/test_multifactor_account.py::test_inactive_asset_rejects_held_position_or_new_target",
        ],
        "cash": [
            "tests/test_multifactor_net_sharpe.py::test_cash_account_has_zero_objective_and_cannot_be_admitted",
            "tests/test_multifactor_account.py::test_rank_targets_ties_shortages_cash_and_fixed_rebalance_cycle",
        ],
        "switch": [
            "tests/test_multifactor_account.py::test_account_reversal_realizes_old_side_then_opens_and_closes_short",
            "tests/test_multifactor_combination_verify.py::test_candidate_account_audit_matches_hourly_engine_and_cost_components",
        ],
        "funding": [
            "tests/test_multifactor_combination_verify.py::test_candidate_account_audit_matches_hourly_engine_and_cost_components",
        ],
    }
    for case, fields in metrics.items():
        counts = {field: sum(int(audit["account_case_counts"][field]) for audit in audits)
                  for field in fields}
        totals[case] = {
            "status": "observed_in_full_account_samples" if any(counts.values())
            else "use_targeted_test_references",
            "sample_count": len(audits),
            "observed_counts": counts,
            "targeted_test_references_if_needed": targeted_tests[case],
        }
    return totals


def run_e0_window(context: Any, *, prepared: Any, fit_time: pd.Timestamp, output_dir: str | Path,
                   factor_names: list[str], batch_candidates: int = E0_BATCH_CANDIDATES,
                   market_frames: Mapping[str, pd.DataFrame],
                   input_hashes: Mapping[str, str] | None = None,
                   audit_sample_size: int = 3) -> dict[str, Any]:
    """Run the three complete-window E0 calibration routes for one fit time.

    `context` is the runner's frozen WindowContext. The prescreen evaluator
    consumes the earlier 2160 training hours; each route evaluator scores the
    following 2160 source hours with coefficients fitted only on training.
    """
    fit_time = pd.Timestamp(fit_time).tz_convert("UTC")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    windows = context.windows
    if context.readiness_status != "ready":
        raise ValueError(f"E0 window is not fully eligible: {context.readiness_status}")
    training_evaluator = context.prescreen_evaluator(batch_candidates)
    training_cube = np.asarray(context.training_values, dtype=float)
    window_record = audit_windows(windows, fit_time=fit_time)
    training_record = audit_training_sample(
        prepared, windows, x_train=context.x_train, y_train=context.y_train,
        training_source_times=getattr(context, "training_source_times", None),
    )
    refit_record = audit_training_sample(
        prepared, windows, x_train=context.x_refit, y_train=context.y_refit,
        training_source_times=getattr(context, "refit_source_times", None),
        calendar_times=windows.refit,
    )
    training_mask_record = audit_training_evaluator(
        training_evaluator, windows, training_cube,
        factor_count=len(factor_names), expected_hours=SOURCE_HOURS,
    )
    if training_evaluator.method != "equal":
        raise AssertionError("cluster pre-screen must use standardized equal-weight single factors")
    prescreen_fit_end = pd.Timestamp(training_evaluator.fit_time).tz_convert("UTC")
    if prescreen_fit_end != windows.validation[0]:
        raise AssertionError("training prescreen account does not stop at validation start")

    cluster_started = time.perf_counter()
    training_cube = np.asarray(training_cube, dtype=float)
    if training_cube.ndim != 3 or training_cube.shape[0] != SOURCE_HOURS:
        raise ValueError("clustering must use exactly 2160 training calendar source hours")
    if training_cube.shape[2] != len(factor_names):
        raise ValueError("training cluster cube must include the complete factor pool")
    groups, correlation = combination.hierarchical_clusters(
        training_cube, cluster_count=8, min_symbols=MIN_SYMBOLS,
    )
    cluster_seconds = time.perf_counter() - cluster_started

    prescreen_started = time.perf_counter()
    grid_candidates, prescreen_records = combination.prescreen(groups, training_evaluator)
    prescreen_seconds = time.perf_counter() - prescreen_started
    grid = combination.enumerate_grid(grid_candidates, size=5)
    if len(grid) != len(set(grid)):
        raise AssertionError("cluster grid contains duplicate subsets")
    if any(len(subset) != 5 for subset in grid):
        raise AssertionError("cluster grid contains a subset with the wrong size")
    if any(len({next(i for i, group in enumerate(groups) if index in group)
               for index in subset}) != 5 for subset in grid):
        raise AssertionError("cluster grid contains more than one factor from a cluster")

    calibration_records = []
    save_seconds = 0.0
    audit_seconds = 0.0
    progress_path = root / "e0_window.progress.json"
    progress = {
        "schema_version": 1,
        "stage": "E0",
        "status": "running",
        "fit_time": fit_time,
        "window_audit": window_record.as_dict(),
        "training_sample_audit": training_record,
        "post_selection_refit_sample_audit": refit_record,
        "cluster_count": len(groups),
        "unique_grid_subsets": len(grid),
        "input_hashes": dict(input_hashes or {}),
        "routes": calibration_records,
    }
    _write_json(progress_path, progress)
    for route_name, algorithm, method in E0_ROUTES:
        evaluator = context.evaluator_factory(method, batch_candidates)
        _assert_same_training_fit(evaluator, context.x_train, context.y_train,
                                  f"{route_name} validation scorer")
        mask_record = audit_validation_mask(
            evaluator, windows, factor_count=len(factor_names), expected_hours=SOURCE_HOURS,
        )
        search_started = time.perf_counter()
        result = combination.run_search(
            evaluator, algorithm=algorithm, groups=groups, grid=grid,
            budget=len(grid), size=5, seed=0, factor_names=factor_names,
        )
        search_wall_seconds = time.perf_counter() - search_started
        winner = tuple(result["selected"]) if result.get("selected") is not None else None
        if winner is None:
            raise RuntimeError(f"{route_name} produced no eligible candidate at {fit_time}")
        if evaluator.best() != winner:
            raise AssertionError("reported search winner differs from cached net-Sharpe maximum")
        if algorithm == "grid" and len(evaluator.cache) != len(grid):
            raise AssertionError("complete grid did not score every unique candidate")
        if algorithm == "ga" and len(evaluator.cache) > len(grid):
            raise AssertionError("GA exceeded the matched unique-subset budget")

        alpha = evaluator.cache[winner][0]
        subsets = deterministic_audit_subsets(
            evaluator, winner, sample_size=audit_sample_size,
            seed_material=f"combination90-e0-v1|{fit_time.isoformat()}|{route_name}",
        )
        candidate_audits = []
        candidate_audit_started = time.perf_counter()
        for subset in subsets:
            selected_alpha, _objective, trials = evaluator.cache[subset]
            trial_alpha = selected_alpha
            if trial_alpha is None and method != "equal":
                valid = [row for row in trials if row.get("valid_objective")]
                if valid:
                    trial_alpha = float(valid[0]["alpha"])
            candidate_audits.append(verify_candidate_account(
                evaluator, subset, trial_alpha, market_frames=market_frames,
                factor_names=factor_names,
            ))
        route_audit_seconds = time.perf_counter() - candidate_audit_started
        audit_seconds += route_audit_seconds

        synthesis = "equal_weight" if method == "equal" else method
        refit_coefficients, refit_metadata = context.refit_coefficients(
            synthesis, winner, alpha,
        )
        refit_audit = audit_refit_coefficients(
            refit_coefficients, method, winner, alpha,
            x_refit=context.x_refit, y_refit=context.y_refit,
            factor_names=factor_names, refit_metadata=refit_metadata,
        )

        candidate_path = root / f"{route_name}.candidate_scores.jsonl.gz"
        route_save_seconds = save_candidate_scores(candidate_path, evaluator, factor_names)
        save_seconds += route_save_seconds
        timing = _evaluator_audit_timing(evaluator)
        calibration_records.append({
            "route": route_name,
            "algorithm": algorithm,
            "weighting": method,
            "seed": 0 if algorithm == "ga" else None,
            "budget_unique_subsets": len(grid),
            "unique_subsets_scored": len(evaluator.cache),
            "selected_subset": list(winner),
            "selected_factor_names": [factor_names[index] for index in winner],
            "selected_alpha": alpha,
            "selected_net_sharpe": float(evaluator.cache[winner][1]),
            "cluster_members": groups,
            "cluster_candidates": grid_candidates,
            "grid_unique_subsets": len(grid),
            "prescreen_records": prescreen_records,
            "training_prescreen_mask_audit": training_mask_record,
            "validation_mask_audit": mask_record,
            "search_result": result,
            "timing": {
                **timing,
                "batch_candidates": batch_candidates,
                "search_wall_seconds": search_wall_seconds,
                "full_account_audit_seconds": route_audit_seconds,
                "candidate_score_save_seconds": route_save_seconds,
            },
            "candidate_score_file": candidate_path.name,
            "account_audits": candidate_audits,
            "account_audit_sample_size": audit_sample_size,
            "account_audit_seed_material": f"combination90-e0-v1|{fit_time.isoformat()}|{route_name}",
            "post_selection_refit_audit": refit_audit,
        })
        _write_json(progress_path, progress)

    record = {
        "schema_version": 1,
        "stage": "E0",
        "status": "passed",
        "fit_time": fit_time,
        "horizon_hours": HORIZON_HOURS,
        "window_audit": window_record.as_dict(),
        "training_sample_audit": training_record,
        "post_selection_refit_sample_audit": refit_record,
        "cluster_method": "average_linkage_1_minus_abs_mean_hourly_cross_sectional_pearson",
        "cluster_correlation_matrix": correlation.tolist(),
        "cluster_count": len(groups),
        "unique_grid_subsets": len(grid),
        "training_prescreen_evaluator_timing": _evaluator_audit_timing(training_evaluator),
        "timing": {
            "cluster_seconds": cluster_seconds,
            "training_prescreen_seconds": prescreen_seconds,
            "full_account_audit_seconds": audit_seconds,
            "candidate_score_save_seconds": save_seconds,
            "total_window_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": process_peak_rss_bytes(),
        },
        "input_hashes": dict(input_hashes or {}),
        "routes": calibration_records,
    }
    manifest_path = root / "e0_window.json"
    save_start = time.perf_counter()
    _write_json(manifest_path, record)
    manifest_save_seconds = time.perf_counter() - save_start
    record["timing"]["manifest_save_seconds"] = manifest_save_seconds
    (root / "e0_window_save_timing.json").write_text(
        json.dumps({"manifest_save_seconds": manifest_save_seconds,
                    "manifest_path": manifest_path.name}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record


def run_e0_calibration(prepared: Any, *, eligible_fit_times: Iterable[pd.Timestamp],
                       context_factory: Callable[[pd.Timestamp], Any],
                       output_dir: str | Path,
                       input_hashes: Mapping[str, str] | None = None,
                       audit_sample_size: int = 3,
                       batch_candidates: int = E0_BATCH_CANDIDATES) -> dict[str, Any]:
    """Run E0 for the first eligible weekly fit in March 2023, 2024, and 2025."""
    source_code_hashes_at_start = audit_source_hashes()
    input_hash_started = time.perf_counter()
    input_hashes_at_start = prepared_input_hashes(prepared)
    input_hash_verification_start_seconds = time.perf_counter() - input_hash_started
    fit_candidates = list(eligible_fit_times.values()) if isinstance(eligible_fit_times, Mapping) else list(eligible_fit_times)
    selected = first_fit_in_months(fit_candidates)
    anchor = pd.Timestamp(prepared.timestamps[0]).tz_convert("UTC")
    weekly_period_ns = int(168 * HOUR.value)
    for fit_time in selected.values():
        if (fit_time.value - anchor.value) % weekly_period_ns:
            raise ValueError(f"selected E0 fit time is off the frozen weekly phase: {fit_time}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    context_build_seconds = {}
    for month, fit_time in selected.items():
        build_started = time.perf_counter()
        context = context_factory(fit_time)
        context_build_seconds[month] = time.perf_counter() - build_started
        records.append(run_e0_window(
            context,
            prepared=prepared,
            fit_time=fit_time,
            output_dir=root / str(month),
            factor_names=list(prepared.factor_names),
            batch_candidates=batch_candidates,
            market_frames=prepared.frames,
            input_hashes=input_hashes or prepared.source_hashes,
            audit_sample_size=audit_sample_size,
        ))
    input_hash_ended = time.perf_counter()
    input_hashes_at_end = prepared_input_hashes(prepared)
    input_hash_verification_end_seconds = time.perf_counter() - input_hash_ended
    source_code_hashes_at_end = audit_source_hashes()
    code_unchanged = source_code_hashes_at_start == source_code_hashes_at_end
    inputs_unchanged = input_hashes_at_start == input_hashes_at_end
    expected_routes = [route[0] for route in E0_ROUTES]
    e0_complete = (
        len(records) == len(FIT_MONTHS)
        and all(record.get("status") == "passed" for record in records)
        and all([route["route"] for route in record["routes"]] == expected_routes
                for record in records)
        and all(record["window_audit"]["training_source_hours"] == SOURCE_HOURS
                and record["window_audit"]["validation_source_hours"] == SOURCE_HOURS
                and record["window_audit"]["refit_source_hours"] == SOURCE_HOURS
                and record["window_audit"]["training_label_gap_hours"] == 1
                and record["window_audit"]["validation_label_gap_to_fit_hours"] == 0
                for record in records)
        and all(all(len(route["account_audits"]) >= 2
                    and all(audit["status"] == "passed" for audit in route["account_audits"])
                    and route["post_selection_refit_audit"]["status"] == "passed"
                    for route in record["routes"])
                for record in records)
    )
    status = "passed" if e0_complete and code_unchanged and inputs_unchanged else "incomplete_or_unverified"
    result = {
        "schema_version": 1,
        "stage": "E0",
        "selection_rule": "first eligible weekly fit in each fixed March month; no return-based date selection",
        "fit_times": selected,
        "weekly_phase_anchor": anchor,
        "candidate_account_batch_size": batch_candidates,
        "context_build_seconds_by_month": context_build_seconds,
        "windows": records,
        "edge_case_coverage": _edge_case_coverage(records),
        "completed_route_count": sum(len(row["routes"]) for row in records),
        "peak_process_rss_bytes": process_peak_rss_bytes(),
        "input_hashes": dict(input_hashes or getattr(prepared, "source_hashes", {})),
        "prepared_overlap_audit": _json_safe(getattr(prepared, "overlap_audit", {})),
        "source_code_hashes_at_start": source_code_hashes_at_start,
        "source_code_hashes_at_end": source_code_hashes_at_end,
        "source_code_unchanged_during_calibration": code_unchanged,
        "prepared_input_hashes_at_start": input_hashes_at_start,
        "prepared_input_hashes_at_end": input_hashes_at_end,
        "prepared_inputs_unchanged_during_calibration": inputs_unchanged,
        "source_hash_verification_seconds": {
            "start": input_hash_verification_start_seconds,
            "end": input_hash_verification_end_seconds,
        },
        "all_e0_completion_gates_passed": e0_complete,
        "status": status,
    }
    (root / "e0_calibration.json").write_text(
        json.dumps(_json_safe(result), default=_json_default, indent=2,
                   ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def run_e0_from_source(source_root: str | Path, output_dir: str | Path, *,
                       batch_candidates: int = E0_BATCH_CANDIDATES,
                       audit_sample_size: int = 3) -> dict[str, Any]:
    """Load the frozen selection30 source, choose first March fits, and run E0."""
    from . import multifactor_combination_research as runner

    source_started = time.perf_counter()
    prepared = runner.prepare_inputs(Path(source_root))
    prepare_seconds = time.perf_counter() - source_started
    fit_time_started = time.perf_counter()
    eligible = runner.calibration_fit_times(prepared)
    fit_time_discovery_seconds = time.perf_counter() - fit_time_started
    result = run_e0_calibration(
        prepared,
        eligible_fit_times=eligible,
        context_factory=lambda fit_time: runner.build_window_context(
            prepared, fit_time, batch_candidates=batch_candidates,
        ),
        output_dir=output_dir,
        input_hashes=prepared.source_hashes,
        audit_sample_size=audit_sample_size,
        batch_candidates=batch_candidates,
    )
    result["source_root"] = str(Path(source_root).resolve())
    result["output_directory"] = str(Path(output_dir).resolve())
    result["source_preparation_seconds"] = prepare_seconds
    result["eligible_fit_time_discovery_seconds"] = fit_time_discovery_seconds
    result["candidate_account_batch_size"] = batch_candidates
    _write_json(Path(output_dir) / "e0_calibration.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="Run numerical E0 calibration for combination90")
    parser.add_argument(
        "--source-root", type=Path,
        default=project_root / "experiments/strategy_research/selection30_20261003/source_inputs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-candidates", type=int, default=E0_BATCH_CANDIDATES)
    parser.add_argument("--audit-sample-size", type=int, default=3)
    args = parser.parse_args(argv)
    result = run_e0_from_source(
        args.source_root, args.output_dir,
        batch_candidates=args.batch_candidates,
        audit_sample_size=args.audit_sample_size,
    )
    print(json.dumps({
        "status": result["status"],
        "fit_times": _json_safe(result["fit_times"]),
        "completed_route_count": result["completed_route_count"],
        "peak_process_rss_bytes": result["peak_process_rss_bytes"],
        "output_directory": result["output_directory"],
    }, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
