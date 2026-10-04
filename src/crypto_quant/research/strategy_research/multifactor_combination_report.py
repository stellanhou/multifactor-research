"""Independent reconciliation and reporting for combination90 runs.

The report consumes saved full-engine account artifacts. It does not rerun
signals or accounts, and it ranks routes from development accounts only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HOURS_PER_YEAR = 365.0 * 24.0
TOLERANCE_RTOL = 2e-10
TOLERANCE_ATOL = 1e-7


def _utc_series(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, format="mixed", utc=True, errors="raise")
    if parsed.isna().any():
        raise ValueError(f"{label} contains missing timestamps")
    return parsed


def _read_table(path: Path, *, indexed: bool = False) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=0 if indexed else None, float_precision="round_trip")
    if indexed:
        table.index = pd.to_datetime(table.index, utc=True, errors="raise")
        table.index.name = "timestamp"
        if table.index.has_duplicates or not table.index.is_monotonic_increasing:
            raise ValueError(f"{path.name} index must be unique and increasing")
    return table


def _assert_close(actual: Any, expected: Any, label: str) -> float:
    actual_values = np.asarray(actual, dtype=float)
    expected_values = np.asarray(expected, dtype=float)
    if actual_values.shape != expected_values.shape:
        raise ValueError(f"{label}: shape differs ({actual_values.shape} vs {expected_values.shape})")
    try:
        np.testing.assert_allclose(actual_values, expected_values, rtol=TOLERANCE_RTOL,
                                   atol=TOLERANCE_ATOL, equal_nan=True, err_msg=label)
    except AssertionError as exc:
        raise ValueError(str(exc)) from exc
    finite = np.isfinite(actual_values) & np.isfinite(expected_values)
    return float(np.max(np.abs(actual_values[finite] - expected_values[finite]))) if finite.any() else 0.0


def _group_to_index(frame: pd.DataFrame, timestamp: pd.Series, field: str,
                    index: pd.DatetimeIndex) -> pd.Series:
    if frame.empty:
        return pd.Series(0.0, index=index)
    grouped = frame.groupby(timestamp)[field].sum()
    return grouped.reindex(index, fill_value=0.0)


def _validate_required(frame: pd.DataFrame, fields: set[str], label: str) -> None:
    missing = fields - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _read_account(directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    ledger = _read_table(directory / "ledger.csv", indexed=True)
    positions = _read_table(directory / "positions.csv")
    fills = _read_table(directory / "fills.csv")
    funding = _read_table(directory / "funding_events.csv")
    orders = _read_table(directory / "orders.csv")
    metrics = json.loads((directory / "metrics.json").read_text())

    _validate_required(ledger, {"cash", "realized_pnl", "funding_cashflow", "fees",
                                "slippage_cost", "trade_notional", "turnover", "gross_notional",
                                "net_notional", "unrealized_pnl", "equity", "return"}, "ledger")
    _validate_required(positions, {"timestamp", "symbol", "quantity", "average_entry_price",
                                   "mark_price", "signed_notional", "gross_notional",
                                   "unrealized_pnl"}, "positions")
    _validate_required(fills, {"timestamp", "symbol", "side", "quantity", "signed_quantity",
                               "reference_open", "fill_price", "notional", "fee",
                               "slippage_cost", "realized_pnl", "previous_quantity",
                               "new_quantity", "average_entry_price"}, "fills")
    _validate_required(funding, {"timestamp", "symbol", "quantity", "funding_rate",
                                 "mark_price", "cashflow"}, "funding events")
    _validate_required(orders, {"signal_timestamp", "execution_timestamp", "symbol",
                                "target_weight", "signal_close", "signal_equity",
                                "current_quantity", "target_quantity", "signed_order_quantity"}, "orders")
    if not isinstance(metrics, dict):
        raise ValueError("metrics.json must contain an object")
    for frame, columns, label in (
        (positions, ("timestamp",), "positions"),
        (fills, ("timestamp",), "fills"),
        (funding, ("timestamp",), "funding events"),
        (orders, ("signal_timestamp", "execution_timestamp"), "orders"),
    ):
        for column in columns:
            frame[column] = _utc_series(frame[column], f"{label}.{column}")
    positions["symbol"] = positions["symbol"].astype(str)
    fills["symbol"] = fills["symbol"].astype(str)
    funding["symbol"] = funding["symbol"].astype(str)
    orders["symbol"] = orders["symbol"].astype(str)
    if positions.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("positions repeat a symbol at a timestamp")
    if funding.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("funding events repeat a symbol at a timestamp")
    if fills.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("fills repeat a symbol at an execution timestamp")
    if orders.duplicated(["execution_timestamp", "symbol"]).any():
        raise ValueError("orders repeat a symbol at an execution timestamp")
    if positions.empty or ledger.empty:
        raise ValueError("account ledger and positions must be nonempty")
    return {"ledger": ledger, "positions": positions, "fills": fills,
            "funding": funding, "orders": orders, "metrics": metrics}


def verify_account(directory: Path, metadata: dict[str, Any],
                   tables: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recompute account conservation, trade lifecycle, event timing and metrics."""
    _validate_route_metadata(metadata)
    _verify_model_schedule(metadata)
    return verify_account_ledger(directory, metadata, tables)


def verify_account_ledger(directory: Path, metadata: dict[str, Any],
                          tables: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reconcile a fixed target account without asserting a model-search schema.

    The combination90 wrapper above additionally verifies its frozen model
    schedule. Fixed-signal experiments supply their own signal provenance.
    """
    directory = Path(directory)
    if tables is None:
        tables = _read_account(directory, metadata)
    ledger, positions = tables["ledger"], tables["positions"]
    fills, funding, orders, metrics = (tables[name] for name in
                                       ("fills", "funding", "orders", "metrics"))
    index = ledger.index
    if len(index) < 1 or not index.equals(pd.date_range(index[0], index[-1], freq="h", tz="UTC")):
        raise ValueError("ledger must cover a complete hourly grid")
    if not np.isfinite(ledger["equity"].to_numpy(dtype=float)).all() or (ledger["equity"] <= 0).any():
        raise ValueError("ledger equity must be finite and positive")
    initial = float(metadata["initial_capital"])
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError("account metadata initial_capital must be finite and positive")
    expected_start = pd.Timestamp(metadata["start"])
    expected_end = pd.Timestamp(metadata["end"])
    if expected_start.tzinfo is None or expected_end.tzinfo is None:
        raise ValueError("account metadata start/end must include UTC offsets")
    expected_start, expected_end = expected_start.tz_convert("UTC"), expected_end.tz_convert("UTC")
    if index[0] != expected_start + pd.Timedelta(hours=1) or index[-1] != expected_end:
        raise ValueError("ledger timestamps do not match the saved account [start, end) bounds")
    if len(index) != int((expected_end - expected_start) / pd.Timedelta(hours=1)):
        raise ValueError("ledger hour count does not match saved account bounds")

    positions["timestamp"] = pd.to_datetime(positions["timestamp"], utc=True)
    if set(positions["timestamp"]) != set(index):
        raise ValueError("position timestamps do not cover the ledger grid")
    expected_keys = pd.MultiIndex.from_product([index, sorted(positions["symbol"].unique())],
                                               names=["timestamp", "symbol"])
    position_keys = pd.MultiIndex.from_frame(positions[["timestamp", "symbol"]])
    if not position_keys.sort_values().equals(expected_keys):
        raise ValueError("positions must have one row per ledger timestamp and symbol")
    gross = positions.groupby("timestamp")["gross_notional"].sum().reindex(index)
    net = positions.groupby("timestamp")["signed_notional"].sum().reindex(index)
    unrealized = positions.groupby("timestamp")["unrealized_pnl"].sum().reindex(index)
    deviations = {}
    for column, values in (("gross_notional", gross), ("net_notional", net),
                           ("unrealized_pnl", unrealized)):
        deviations[column] = _assert_close(ledger[column], values, f"ledger {column} vs positions")
    deviations["equity"] = _assert_close(ledger["equity"],
                                         ledger["cash"] + ledger["unrealized_pnl"],
                                         "equity = cash + unrealized PnL")

    fill_fee_rate = metadata.get("fee_bps")
    fill_slippage_rate = metadata.get("slippage_bps")
    if fill_fee_rate is None or fill_slippage_rate is None:
        raise ValueError("account metadata must include fee_bps and slippage_bps for fill reconciliation")
    fee_rate = float(fill_fee_rate) / 10_000.0
    slippage_rate = float(fill_slippage_rate) / 10_000.0
    if not np.isfinite([fee_rate, slippage_rate]).all() or fee_rate < 0 or not 0 <= slippage_rate < 1:
        raise ValueError("saved fee/slippage rates are invalid")
    if not fills.empty:
        signed = fills["signed_quantity"].to_numpy(dtype=float)
        if (fills["side"].eq("BUY").to_numpy() != (signed > 0)).any():
            raise ValueError("fill sides do not match signed quantities")
        deviations["fill_quantity"] = _assert_close(fills["quantity"], np.abs(signed), "fill absolute quantity")
        deviations["fill_notional"] = _assert_close(fills["notional"],
            np.abs(signed) * fills["fill_price"], "fill notional")
        expected_fill = fills["reference_open"].to_numpy(dtype=float) * (1 + np.sign(signed) * slippage_rate)
        deviations["fill_price"] = _assert_close(fills["fill_price"], expected_fill, "fill price vs slippage")
        deviations["fill_fee"] = _assert_close(fills["fee"], fills["notional"] * fee_rate, "fill fee")
        deviations["fill_slippage"] = _assert_close(fills["slippage_cost"],
            np.abs(signed) * np.abs(fills["fill_price"] - fills["reference_open"]), "fill slippage cost")

    # Rebuild quantity and average-entry lifecycle using vectorized grouped shifts.
    fills = fills.sort_values(["symbol", "timestamp"], kind="mergesort").copy()
    expected_old = fills.groupby("symbol", sort=False)["new_quantity"].shift().fillna(0.0)
    deviations["fill_previous_quantity"] = _assert_close(
        fills["previous_quantity"], expected_old, "fill previous quantity lifecycle")
    expected_new = expected_old + fills["signed_quantity"]
    deviations["fill_new_quantity"] = _assert_close(
        fills["new_quantity"], expected_new, "fill resulting quantity lifecycle")
    previous_entry = fills.groupby("symbol", sort=False)["average_entry_price"].shift()
    delta = fills["signed_quantity"].to_numpy(dtype=float)
    old = expected_old.to_numpy(dtype=float)
    new = expected_new.to_numpy(dtype=float)
    same_side_add = (old == 0.0) | (old * delta > 0.0)
    flat_after = new == 0.0
    direction_reversal = (old * new) < 0.0
    expected_entry = previous_entry.to_numpy(dtype=float)
    expected_entry[same_side_add & (old == 0.0)] = fills.loc[same_side_add & (old == 0.0), "fill_price"].to_numpy(dtype=float)
    weighted_add = same_side_add & (old != 0.0)
    expected_entry[weighted_add] = (
        previous_entry.to_numpy(dtype=float)[weighted_add] * np.abs(old[weighted_add])
        + fills.loc[weighted_add, "fill_price"].to_numpy(dtype=float) * np.abs(delta[weighted_add])
    ) / (np.abs(old[weighted_add]) + np.abs(delta[weighted_add]))
    expected_entry[flat_after] = np.nan
    expected_entry[direction_reversal] = fills.loc[direction_reversal, "fill_price"].to_numpy(dtype=float)
    deviations["fill_average_entry_price"] = _assert_close(
        fills["average_entry_price"], expected_entry, "fill average-entry lifecycle")

    events = pd.DataFrame({
        "timestamp": fills["timestamp"] + pd.Timedelta(hours=1),
        "symbol": fills["symbol"].to_numpy(),
        "expected_quantity": expected_new.to_numpy(dtype=float),
        "expected_entry_price": expected_entry,
    })
    position_states = positions[["timestamp", "symbol", "quantity", "average_entry_price"]].copy()
    if events.empty:
        position_states["expected_quantity"] = 0.0
        position_states["expected_entry_price"] = np.nan
    else:
        position_states = pd.merge_asof(
            position_states.sort_values(["timestamp", "symbol"], kind="mergesort"),
            events.sort_values(["timestamp", "symbol"], kind="mergesort"),
            on="timestamp", by="symbol", direction="backward",
        )
        position_states["expected_quantity"] = position_states["expected_quantity"].fillna(0.0)
    deviations["position_quantity_lifecycle"] = _assert_close(
        position_states["quantity"], position_states["expected_quantity"],
        "positions vs replayed fill quantities")
    deviations["position_entry_lifecycle"] = _assert_close(
        position_states["average_entry_price"], position_states["expected_entry_price"],
        "positions vs replayed average-entry prices")
    held = positions["quantity"].to_numpy(dtype=float) != 0.0
    expected_unrealized = positions.loc[held, "quantity"].to_numpy(dtype=float) * (
        positions.loc[held, "mark_price"].to_numpy(dtype=float)
        - positions.loc[held, "average_entry_price"].to_numpy(dtype=float))
    deviations["position_unrealized_pnl"] = _assert_close(
        positions.loc[held, "unrealized_pnl"], expected_unrealized,
        "position unrealized PnL from quantity, mark and average entry")

    # Every order is a full decision; only nonzero signed orders create fills.
    deviations["order_quantity"] = _assert_close(orders["signed_order_quantity"],
        orders["target_quantity"] - orders["current_quantity"], "order signed quantity")
    deviations["order_delay"] = _assert_close(
        (orders["execution_timestamp"] - orders["signal_timestamp"]).dt.total_seconds() / 3600,
        np.ones(len(orders)), "order execution follows signal by one hour")
    order_values = orders[["target_weight", "signal_equity", "current_quantity",
                           "target_quantity", "signed_order_quantity"]].to_numpy(dtype=float)
    if not np.isfinite(order_values).all():
        raise ValueError("order quantities, weights and equity must be finite")
    signal_close = orders["signal_close"].to_numpy(dtype=float)
    missing_signal_close = np.isnan(signal_close)
    if np.isinf(signal_close).any():
        raise ValueError("order signal close cannot be infinite")
    if (orders["signal_equity"] <= 0).any():
        raise ValueError("order signal equity must be positive")
    if missing_signal_close.any() and np.any(order_values[missing_signal_close][:, [0, 2, 3, 4]] != 0.0):
        raise ValueError("orders with missing signal prices must have zero target weight and quantities")
    exposed_orders = (order_values[:, 2] != 0.0) | (order_values[:, 3] != 0.0)
    invalid_exposed_price = ~np.isfinite(signal_close) | (signal_close <= 0)
    if (exposed_orders & invalid_exposed_price).any():
        raise ValueError("orders with current or target exposure require a finite positive signal close")
    expected_target_weight = np.zeros(len(orders), dtype=float)
    priced_orders = ~missing_signal_close
    expected_target_weight[priced_orders] = (
        orders.loc[priced_orders, "target_quantity"].to_numpy(dtype=float)
        * signal_close[priced_orders]
        / orders.loc[priced_orders, "signal_equity"].to_numpy(dtype=float)
    )
    deviations["order_target_weight"] = _assert_close(
        orders["target_weight"],
        expected_target_weight,
        "order target weight from target quantity and signal data")
    portfolio = metadata.get("portfolio")
    if not isinstance(portfolio, dict):
        raise ValueError("account metadata must include the frozen portfolio limits")
    gross_limit = float(portfolio["gross_exposure"])
    asset_limit = float(portfolio["max_asset_weight"])
    rebalance_hours = portfolio["rebalance_hours"]
    if not np.isfinite([gross_limit, asset_limit]).all() or not 0 < gross_limit <= 1 \
            or not 0 < asset_limit <= 1 or type(rebalance_hours) is not int or rebalance_hours <= 0:
        raise ValueError("saved gross, asset and rebalance limits are invalid")
    if (orders["target_weight"].abs() > asset_limit + 1e-12).any():
        raise ValueError("order target exceeds the frozen per-asset weight limit")
    target_gross = orders.groupby("execution_timestamp")["target_weight"].apply(lambda values: values.abs().sum())
    if (target_gross > gross_limit + 1e-12).any():
        raise ValueError("order targets exceed the frozen gross exposure limit")
    r0_anchor = _parsed_utc(metadata["r0_anchor"], "r0_anchor")
    execution_offset_hours = (
        orders["execution_timestamp"] - r0_anchor - pd.Timedelta(hours=1)
    ).dt.total_seconds() / 3600
    if not np.isclose(execution_offset_hours / rebalance_hours,
                      np.round(execution_offset_hours / rebalance_hours), rtol=0, atol=1e-10).all():
        raise ValueError("R0 order timestamps do not preserve the original rebalance phase")

    marks = positions["mark_price"].to_numpy(dtype=float)
    inactive = np.isnan(marks)
    if np.isinf(marks).any() or (marks[~inactive] <= 0).any():
        raise ValueError("position marks must be positive or explicitly missing for inactive assets")
    if inactive.any():
        for column in ("quantity", "signed_notional", "gross_notional", "unrealized_pnl"):
            if not np.isclose(positions.loc[inactive, column].to_numpy(dtype=float), 0.0,
                              rtol=TOLERANCE_RTOL, atol=TOLERANCE_ATOL).all():
                raise ValueError(f"inactive assets must remain flat with zero {column}")
        if positions.loc[inactive, "average_entry_price"].notna().any():
            raise ValueError("inactive flat assets cannot retain an average entry price")
        inactive_keys = pd.MultiIndex.from_frame(positions.loc[inactive, ["timestamp", "symbol"]])
        fill_after_keys = pd.MultiIndex.from_arrays(
            [fills["timestamp"] + pd.Timedelta(hours=1), fills["symbol"]],
            names=["timestamp", "symbol"])
        if len(inactive_keys.intersection(fill_after_keys)):
            raise ValueError("fills must stop before inactive/delisted mark hours")
    lifecycle_events = metadata.get("lifecycle_events")
    if not isinstance(lifecycle_events, list):
        raise ValueError("account metadata must include the frozen lifecycle event list")
    for event in lifecycle_events:
        if not isinstance(event, dict) or not isinstance(event.get("symbol"), str):
            raise ValueError("lifecycle events must name a symbol")
        settlement = _parsed_utc(event["settlement_at"], "lifecycle settlement_at")
        if settlement >= expected_end:
            continue
        symbol = event["symbol"]
        symbol_positions = positions.loc[positions["symbol"] == symbol]
        if symbol_positions.empty:
            raise ValueError(f"lifecycle symbol {symbol} is absent from the account universe")
        post_settlement = symbol_positions.loc[symbol_positions["timestamp"] >= max(index[0], settlement)]
        if not np.isclose(post_settlement["quantity"].to_numpy(dtype=float), 0.0,
                          rtol=TOLERANCE_RTOL, atol=TOLERANCE_ATOL).all():
            raise ValueError(f"{symbol} position is not flat at/after its announced settlement")
        if not fills.empty:
            post_settlement_fills = fills.loc[(fills["symbol"] == symbol)
                                              & (fills["timestamp"] >= max(expected_start, settlement))]
            if not post_settlement_fills.empty:
                raise ValueError(f"{symbol} has a fill at/after its announced settlement")
        inactive_start = max(index[0], settlement + pd.Timedelta(hours=1))
        after_inactive = symbol_positions.loc[symbol_positions["timestamp"] >= inactive_start]
        if after_inactive["mark_price"].notna().any():
            raise ValueError(f"{symbol} has fabricated marks after announced settlement")
    position_quantity = positions.set_index(["timestamp", "symbol"])["quantity"]
    order_keys = pd.MultiIndex.from_arrays(
        [orders["execution_timestamp"], orders["symbol"]], names=["timestamp", "symbol"])
    fill_quantity = fills.set_index(["timestamp", "symbol"])["signed_quantity"]
    order_fill_quantity = fill_quantity.reindex(order_keys)
    changed = orders["signed_order_quantity"].to_numpy(dtype=float) != 0.0
    fill_present = order_fill_quantity.notna().to_numpy()
    if np.any(changed != fill_present):
        raise ValueError("nonzero orders and fills do not have matching execution keys")
    if not fills.empty and len(fill_quantity.index.difference(order_keys)):
        raise ValueError("fills contain execution keys without a saved order")
    deviations["order_fill_quantity"] = _assert_close(
        orders.loc[changed, "signed_order_quantity"], order_fill_quantity.loc[changed],
        "order signed quantities vs fills")
    initial_time = pd.Timestamp(metadata["start"]).tz_convert("UTC")
    pretrade_keys = pd.MultiIndex.from_arrays(
        [orders["execution_timestamp"], orders["symbol"]], names=["timestamp", "symbol"])
    current_position = position_quantity.reindex(pretrade_keys)
    missing_pretrade = current_position.isna()
    allowed_initial = orders["execution_timestamp"].eq(initial_time).to_numpy()
    if np.any(missing_pretrade.to_numpy() & ~allowed_initial):
        raise ValueError("orders lack a saved pre-trade position at a noninitial execution")
    current_position = current_position.fillna(0.0)
    deviations["order_current_position"] = _assert_close(
        orders["current_quantity"], current_position, "order current quantity vs pre-trade position")
    posttrade_keys = pd.MultiIndex.from_arrays(
        [orders["execution_timestamp"] + pd.Timedelta(hours=1), orders["symbol"]],
        names=["timestamp", "symbol"])
    posttrade_position = position_quantity.reindex(posttrade_keys)
    if posttrade_position.isna().any():
        raise ValueError("orders lack a saved post-trade position")
    deviations["order_target_position"] = _assert_close(
        orders["target_quantity"], posttrade_position, "order target quantity vs post-trade position")

    # Funding at an exact bar open settles from the prior closing position;
    # within a bar it settles from that bar's post-trade position.
    if not funding.empty:
        boundary = funding["timestamp"].dt.floor("h")
        reference_time = boundary.where(funding["timestamp"].eq(boundary), boundary + pd.Timedelta(hours=1))
        funding_keys = pd.MultiIndex.from_arrays([reference_time, funding["symbol"]],
                                                 names=["timestamp", "symbol"])
        expected_funding_quantity = position_quantity.reindex(funding_keys)
        before_first_mark = (reference_time < index[0]).to_numpy()
        missing_funding_position = expected_funding_quantity.isna().to_numpy()
        if np.any(missing_funding_position & ~before_first_mark):
            raise ValueError("funding events lack a matching saved position")
        expected_funding_quantity = expected_funding_quantity.fillna(0.0)
        deviations["funding_quantity_timing"] = _assert_close(
            funding["quantity"], expected_funding_quantity, "funding position timing")
        deviations["funding_cashflow"] = _assert_close(funding["cashflow"],
            -funding["quantity"] * funding["funding_rate"] * funding["mark_price"],
            "funding cashflow")

    fill_bar = fills["timestamp"] + pd.Timedelta(hours=1) if not fills.empty else pd.Series(dtype="datetime64[ns, UTC]")
    funding_bar = funding["timestamp"].dt.floor("h") + pd.Timedelta(hours=1) if not funding.empty else pd.Series(dtype="datetime64[ns, UTC]")
    for column, frame, timestamp, field in (
        ("fees", fills, fill_bar, "fee"),
        ("slippage_cost", fills, fill_bar, "slippage_cost"),
        ("trade_notional", fills, fill_bar, "notional"),
        ("realized_pnl", fills, fill_bar, "realized_pnl"),
        ("funding_cashflow", funding, funding_bar, "cashflow"),
    ):
        expected = _group_to_index(frame, timestamp, field, index)
        deviations[f"ledger_{column}"] = _assert_close(ledger[column], expected, f"ledger {column} vs events")
    deviations["cash"] = _assert_close(ledger["cash"],
        initial + (ledger["realized_pnl"] + ledger["funding_cashflow"] - ledger["fees"]).cumsum(),
        "cash conservation")
    signal_equity = orders.groupby("execution_timestamp")["signal_equity"].agg(["first", "nunique"])
    if (signal_equity["nunique"] != 1).any() or not np.isfinite(signal_equity["first"].to_numpy(dtype=float)).all() \
            or (signal_equity["first"] <= 0).any():
        raise ValueError("orders at each execution time must share a finite positive signal equity")
    notional_by_execution = fills.groupby("timestamp")["notional"].sum().reindex(
        signal_equity.index, fill_value=0.0)
    execution_turnover = notional_by_execution / signal_equity["first"]
    expected_turnover = pd.Series(0.0, index=index)
    ledger_execution_times = execution_turnover.index + pd.Timedelta(hours=1)
    expected_turnover.loc[ledger_execution_times] = execution_turnover.to_numpy(dtype=float)
    deviations["turnover"] = _assert_close(ledger["turnover"], expected_turnover, "ledger turnover")

    equity_with_initial = np.r_[initial, ledger["equity"].to_numpy(dtype=float)]
    hourly_returns = equity_with_initial[1:] / equity_with_initial[:-1] - 1.0
    expected_return = ledger["equity"].to_numpy(dtype=float) / np.r_[initial, ledger["equity"].to_numpy(dtype=float)[:-1]] - 1.0
    deviations["return"] = _assert_close(ledger["return"], expected_return, "hourly equity returns")
    annual_vol = float(np.std(hourly_returns, ddof=1) * np.sqrt(HOURS_PER_YEAR)) if len(hourly_returns) > 1 else 0.0
    sharpe = float(np.mean(hourly_returns) * HOURS_PER_YEAR / annual_vol) if annual_vol > 0 else 0.0
    max_dd = float(-(equity_with_initial / np.maximum.accumulate(equity_with_initial) - 1.0).min())
    expected_metrics = {
        "net_return": float(ledger["equity"].iloc[-1] / initial - 1),
        "annualized_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_fees": float(fills["fee"].sum()),
        "total_slippage_cost": float(fills["slippage_cost"].sum()),
        "total_funding": float(funding["cashflow"].sum()),
        "total_turnover": float(ledger["turnover"].sum()),
        "final_equity": float(ledger["equity"].iloc[-1]),
    }
    for name, expected in expected_metrics.items():
        if name not in metrics:
            raise ValueError(f"metrics.json is missing {name}")
        deviations[f"metric_{name}"] = _assert_close([metrics[name]], [expected], f"metric {name}")
    deviations["metric_annualized_return"] = _assert_close(
        [metrics["annualized_return"]],
        [((ledger["equity"].iloc[-1] / initial) ** (HOURS_PER_YEAR / len(ledger)) - 1)
         if ledger["equity"].iloc[-1] > 0 else np.nan], "metric annualized return")
    if "gross_pnl_including_funding" in metrics:
        deviations["metric_gross_pnl_including_funding"] = _assert_close(
            [metrics["gross_pnl_including_funding"]],
            [ledger["equity"].iloc[-1] - initial + fills["fee"].sum() + fills["slippage_cost"].sum()],
            "gross PnL including funding")
    if not np.isfinite([annual_vol, sharpe, max_dd]).all():
        raise ValueError("recomputed metrics are not finite")
    cash_ratio = ledger["cash"] / ledger["equity"]
    gross_exposure = ledger["gross_notional"] / ledger["equity"]
    net_exposure = ledger["net_notional"] / ledger["equity"]
    return {"path": str(directory), "status": "passed", "deviations": deviations,
            "hours": int(len(ledger)), "fills": int(len(fills)), "orders": int(len(orders)),
            "funding_events": int(len(funding)),
            "average_gross_exposure": float(gross_exposure.mean()),
            "average_net_exposure": float(net_exposure.mean()),
            "average_cash_ratio": float(cash_ratio.mean()),
            "position_time_fraction": float((ledger["gross_notional"] > 0).mean()),
            "no_trade_hour_fraction": float((ledger["trade_notional"] == 0).mean())}


def _metadata_for_record(run_root: Path, record: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    relative = record["path"]
    if not isinstance(relative, str) or not relative:
        raise ValueError("each account record must include a nonempty path")
    directory = Path(relative)
    if directory.is_absolute():
        raise ValueError("account record paths must be relative to the run root")
    directory = (run_root / directory).resolve()
    if not directory.is_relative_to(run_root.resolve()):
        raise ValueError("account record path escapes the run root")
    account_json = json.loads((directory / "account.json").read_text())
    if not isinstance(account_json, dict):
        raise ValueError(f"{directory}/account.json must contain an object")
    return directory, {**record, **account_json}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _account_artifact_hashes(directory: Path) -> dict[str, str]:
    names = ("account.json", "metrics.json", "ledger.csv", "positions.csv", "fills.csv",
             "funding_events.csv", "orders.csv")
    return {name: _file_sha256(directory / name) for name in names}


def _funding_input_hash(funding: pd.DataFrame) -> str:
    fields = ["timestamp", "symbol", "funding_rate", "mark_price"]
    rows = funding[fields].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(rows, index=False).to_numpy().tobytes()).hexdigest()


def _parsed_utc(value: Any, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return timestamp.tz_convert("UTC")


def _verify_model_schedule(metadata: dict[str, Any]) -> None:
    schedule = metadata.get("model_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("account metadata must include a nonempty model_schedule")
    start, end = _parsed_utc(metadata["start"], "account start"), _parsed_utc(metadata["end"], "account end")
    selection_window_bars = metadata["selection_window_days"] * 24
    fit_times = []
    score_windows = []
    for update in schedule:
        if not isinstance(update, dict):
            raise ValueError("model schedule entries must be objects")
        fit = _parsed_utc(update["fit_timestamp"], "model fit_timestamp")
        selected = update.get("selected_factors")
        coefficients = update.get("coefficients")
        if not isinstance(selected, list) or len(selected) != len(set(selected)):
            raise ValueError("model schedule selected_factors must be unique identities")
        if not isinstance(coefficients, dict) or not np.isfinite(list(coefficients.values())).all():
            raise ValueError("model schedule coefficients must be a finite factor-to-weight mapping")
        alpha = update.get("alpha")
        if alpha is not None and not np.isfinite(float(alpha)):
            raise ValueError("model schedule alpha must be finite")
        for key in ("train_start", "train_end", "selection_start", "selection_end",
                    "validation_account_start", "validation_account_end",
                    "account_score_start", "account_score_end"):
            if key not in update:
                raise ValueError(f"model schedule is missing {key}")
        train_start = _parsed_utc(update["train_start"], "train_start")
        train_end = _parsed_utc(update["train_end"], "train_end")
        selection_start = _parsed_utc(update["selection_start"], "selection_start")
        selection_end = _parsed_utc(update["selection_end"], "selection_end")
        validation_account_start = _parsed_utc(update["validation_account_start"], "validation_account_start")
        validation_account_end = _parsed_utc(update["validation_account_end"], "validation_account_end")
        score_start = _parsed_utc(update["account_score_start"], "account_score_start")
        score_end = _parsed_utc(update["account_score_end"], "account_score_end")
        if not train_start < train_end <= selection_start < selection_end <= fit:
            raise ValueError("training/selection data boundaries violate fit-time causality")
        source_bars = (selection_end - selection_start) / pd.Timedelta(hours=1) + 1
        if source_bars != selection_window_bars:
            raise ValueError("saved selection source bounds do not match selection_window_days")
        if validation_account_start != selection_start + pd.Timedelta(hours=1) \
                or validation_account_end != fit or validation_account_end <= validation_account_start:
            raise ValueError("internal validation account bounds do not match the saved source window and fit time")
        if not start <= score_start < score_end <= end or fit + pd.Timedelta(hours=1) > score_start:
            raise ValueError("account scoring bounds are outside the account or fit follows scoring")
        fit_times.append(fit)
        score_windows.append((score_start, score_end))
    if fit_times != sorted(set(fit_times)):
        raise ValueError("model fit timestamps must be unique and increasing")
    if score_windows[0][0] != start or score_windows[-1][1] != end:
        raise ValueError("model schedule scoring windows must cover the account bounds")
    for previous, current in zip(score_windows, score_windows[1:]):
        if previous[1] != current[0]:
            raise ValueError("model schedule scoring windows contain a gap or overlap")


def _validate_route_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("experiment") not in {"E1", "E2", "E3", "E4", "E5"}:
        raise ValueError("account metadata experiment must be E1 through E5")
    if type(metadata.get("primary_selection_eligible")) is not bool:
        raise ValueError("account metadata primary_selection_eligible must be an explicit boolean")
    if type(metadata.get("selection_window_days")) is not int or metadata["selection_window_days"] not in {21, 90}:
        raise ValueError("account metadata selection_window_days must be 21 or 90")
    for name in ("method", "synthesis", "route_family"):
        if not isinstance(metadata.get(name), str) or not metadata[name]:
            raise ValueError(f"account metadata {name} must be a nonempty string")
    if not isinstance(metadata["route_id"], str) or not metadata["route_id"]:
        raise ValueError("account metadata route_id must be a nonempty string")


def _route_key(metadata: dict[str, Any]) -> str:
    route = metadata["route_id"]
    if not isinstance(route, str) or not route:
        raise ValueError("account metadata route_id must be a nonempty string")
    return route


def _stage(metadata: dict[str, Any]) -> str:
    raw = metadata["stage"]
    if raw in {"development", "C"}:
        return raw
    raise ValueError(f"unsupported account stage: {raw!r}")


def _cost_name(metadata: dict[str, Any]) -> str:
    multiplier = float(metadata.get("cost_multiplier", np.nan))
    if multiplier == 1.0:
        return "base"
    if multiplier == 2.0:
        return "stress"
    raise ValueError(f"cost_multiplier must be 1 or 2, got {multiplier}")


def _account_row(account: dict[str, Any]) -> dict[str, Any]:
    metadata, tables, verification = account["metadata"], account["tables"], account["verification"]
    metrics, ledger, fills = tables["metrics"], tables["ledger"], tables["fills"]
    schedule_lengths = {}
    if verification["status"] == "passed":
        schedule = metadata["model_schedule"]
        selection_source_bars = {
            int((_parsed_utc(update["selection_end"], "selection_end")
                 - _parsed_utc(update["selection_start"], "selection_start")) / pd.Timedelta(hours=1)) + 1
            for update in schedule
        }
        validation_account_hours = {
            int((_parsed_utc(update["validation_account_end"], "validation_account_end")
                 - _parsed_utc(update["validation_account_start"], "validation_account_start")) / pd.Timedelta(hours=1))
            for update in schedule
        }
        if len(selection_source_bars) != 1 or len(validation_account_hours) != 1:
            raise ValueError("model schedule selection and validation bounds must have stable hourly lengths")
        schedule_lengths = {"selection_source_bars": next(iter(selection_source_bars)),
                            "internal_validation_account_hours": next(iter(validation_account_hours))}
    row = {
        "route": account["route"], "route_family": metadata["route_family"],
        "method": metadata["method"], "synthesis": metadata["synthesis"],
        "selection_window_days": metadata["selection_window_days"],
        "seed": metadata["seed"], "experiment": metadata["experiment"],
        "primary_selection_eligible": metadata["primary_selection_eligible"],
        "stage": account["stage"],
        "cost": account["cost"], "cost_multiplier": metadata["cost_multiplier"],
        "verified": verification["status"] == "passed", "verification_status": verification["status"],
        "account_failure_reason": ";".join(verification.get("errors", [])),
        "route_complete": False, "completed_account_paths": np.nan,
        "start": metadata["start"], "end": metadata["end"],
        "hours": len(ledger), "fills": len(fills),
        "net_return": metrics["net_return"], "sharpe_ratio": metrics["sharpe_ratio"],
        "annualized_volatility": metrics["annualized_volatility"],
        "max_drawdown": metrics["max_drawdown"],
        "total_fees": metrics["total_fees"], "total_slippage_cost": metrics["total_slippage_cost"],
        "total_funding": metrics["total_funding"], "total_turnover": metrics["total_turnover"],
        "final_equity": metrics["final_equity"],
        "average_gross_exposure": verification.get("average_gross_exposure", np.nan),
        "average_net_exposure": verification.get("average_net_exposure", np.nan),
        "average_cash_ratio": verification.get("average_cash_ratio", np.nan),
        "position_time_fraction": verification.get("position_time_fraction", np.nan),
        "no_trade_hour_fraction": verification.get("no_trade_hour_fraction", np.nan),
    }
    row["traded_bars"] = int((ledger["trade_notional"] > 0).sum())
    row.update(schedule_lengths)
    return row


def _failed_account_row(route: str, stage: str, cost: str,
                        metadata: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": route, "route_family": metadata["route_family"],
        "method": metadata["method"], "synthesis": metadata["synthesis"],
        "selection_window_days": metadata["selection_window_days"],
        "seed": metadata["seed"], "experiment": metadata["experiment"],
        "primary_selection_eligible": metadata["primary_selection_eligible"],
        "stage": stage, "cost": cost, "cost_multiplier": metadata.get("cost_multiplier", np.nan),
        "verified": False, "verification_status": "failed",
        "account_failure_reason": ";".join(verification.get("errors", [])),
        "route_complete": False, "completed_account_paths": np.nan,
        "start": metadata.get("start", ""), "end": metadata.get("end", ""),
        "hours": 0, "fills": 0, "traded_bars": 0,
        "net_return": np.nan, "sharpe_ratio": np.nan,
        "annualized_volatility": np.nan, "max_drawdown": np.nan,
        "total_fees": np.nan, "total_slippage_cost": np.nan,
        "total_funding": np.nan, "total_turnover": np.nan, "final_equity": np.nan,
        "average_gross_exposure": np.nan, "average_net_exposure": np.nan,
        "average_cash_ratio": np.nan, "position_time_fraction": np.nan,
        "no_trade_hour_fraction": np.nan,
    }


def _monthly_yearly(accounts: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    for account in accounts:
        ledger = account["tables"]["ledger"]
        fills = account["tables"]["fills"]
        metadata = account["metadata"]
        gross = ledger["gross_notional"] / ledger["equity"]
        net = ledger["net_notional"] / ledger["equity"]
        cash = ledger["cash"] / ledger["equity"]
        for frequency, target in (("M", monthly_rows), ("Y", yearly_rows)):
            periods = ledger.index.tz_convert(None).to_period(frequency)
            fill_periods = (fills["timestamp"] + pd.Timedelta(hours=1)).dt.tz_convert(None).dt.to_period(frequency)
            fill_counts = fill_periods.value_counts()
            for period, loc in pd.Series(np.arange(len(ledger)), index=periods).groupby(level=0):
                subset = ledger.iloc[loc.to_numpy(dtype=int)]
                period_start, period_end = subset.index[0], subset.index[-1]
                target.append({
                    "route": account["route"], "stage": account["stage"], "cost": account["cost"],
                    "verified": account["verification"]["status"] == "passed",
                    "period": str(period), "first_ledger_timestamp": period_start.isoformat(),
                    "last_ledger_timestamp": period_end.isoformat(), "bars": len(subset),
                    "net_return": float(np.prod(1.0 + subset["return"].to_numpy(dtype=float)) - 1.0),
                    "fees": float(subset["fees"].sum()), "slippage_cost": float(subset["slippage_cost"].sum()),
                    "funding_cashflow": float(subset["funding_cashflow"].sum()),
                    "turnover": float(subset["turnover"].sum()), "fills": int(fill_counts.get(period, 0)),
                    "average_gross_exposure": float(gross.loc[subset.index].mean()),
                    "average_net_exposure": float(net.loc[subset.index].mean()),
                    "average_cash_ratio": float(cash.loc[subset.index].mean()),
                    "position_time_fraction": float((subset["gross_notional"] > 0).mean()),
                })
    return pd.DataFrame(monthly_rows), pd.DataFrame(yearly_rows)


def _contributions(accounts: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for account in accounts:
        fills, funding = account["tables"]["fills"], account["tables"]["funding"]
        positions = account["tables"]["positions"]
        last_time = account["tables"]["ledger"].index[-1]
        end_positions = positions.loc[positions["timestamp"] == last_time].set_index("symbol")
        symbols = sorted(set(positions["symbol"]) | set(fills["symbol"]) | set(funding["symbol"]))
        realized = fills.groupby("symbol")["realized_pnl"].sum().reindex(symbols, fill_value=0.0)
        fees = fills.groupby("symbol")["fee"].sum().reindex(symbols, fill_value=0.0)
        slippage = fills.groupby("symbol")["slippage_cost"].sum().reindex(symbols, fill_value=0.0)
        funding_cashflow = funding.groupby("symbol")["cashflow"].sum().reindex(symbols, fill_value=0.0)
        ending_unrealized = end_positions["unrealized_pnl"].reindex(symbols, fill_value=0.0)
        account_contributions = []
        for symbol in symbols:
            symbol_realized = float(realized.loc[symbol])
            symbol_ending_unrealized = float(ending_unrealized.loc[symbol])
            symbol_funding = float(funding_cashflow.loc[symbol])
            symbol_fees = float(fees.loc[symbol])
            symbol_slippage = float(slippage.loc[symbol])
            net = symbol_realized + symbol_ending_unrealized + symbol_funding - symbol_fees
            rows.append({
                "route": account["route"], "stage": account["stage"], "cost": account["cost"],
                "verified": account["verification"]["status"] == "passed",
                "symbol": symbol, "realized_pnl": symbol_realized,
                "ending_unrealized_pnl": symbol_ending_unrealized, "funding_cashflow": symbol_funding,
                "fees": symbol_fees, "slippage_cost_embedded_in_fill_prices": symbol_slippage,
                "net_contribution": net,
                "pre_fee_and_slippage_contribution": net + symbol_fees + symbol_slippage,
            })
            account_contributions.append(net)
        attributed = sum(account_contributions)
        initial = float(account["metadata"]["initial_capital"])
        actual = float(account["tables"]["ledger"]["equity"].iloc[-1]) - initial
        if account["verification"]["status"] == "passed":
            _assert_close([attributed], [actual], f"symbol contributions reconcile for {account['route']}/{account['stage']}/{account['cost']}")
    return pd.DataFrame(rows)


def _selection_tables(accounts: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frequency: dict[tuple[str, str, str], int] = {}
    coefficient_values: dict[tuple[str, str, str], list[float]] = {}
    alpha_values: dict[tuple[str, str], list[float]] = {}
    parameter_values: dict[tuple[str, str, str], list[float]] = {}
    route_schedules: dict[tuple[str, str], list[dict[str, Any]]] = {}
    concentration_rows: list[dict[str, Any]] = []
    for account in accounts:
        if account["cost"] != "base" or account["verification"]["status"] != "passed":
            continue
        schedule = account["metadata"].get("model_schedule", [])
        if not isinstance(schedule, list):
            raise ValueError(f"{account['route']} model_schedule must be a list")
        route_key = (account["route"], account["stage"])
        route_schedules[route_key] = schedule
        for update in schedule:
            selected = update.get("selected_factors", [])
            if not isinstance(selected, list) or len(selected) != len(set(selected)):
                raise ValueError(f"{account['route']} model schedule factors must be a unique list")
            for factor in selected:
                key = (account["route"], account["stage"], str(factor))
                frequency[key] = frequency.get(key, 0) + 1
            coefficients = update.get("coefficients", {})
            if not isinstance(coefficients, dict):
                raise ValueError(f"{account['route']} coefficients must be a factor-to-weight object")
            beta = np.asarray(list(coefficients.values()), dtype=float)
            if len(beta) and not np.isfinite(beta).all():
                raise ValueError(f"{account['route']} coefficients must be finite")
            l1 = float(np.abs(beta).sum())
            if len(beta) == 0:
                concentration_status = "undefined_no_coefficient_vector"
                maximum_share = effective_count = np.nan
            elif l1 == 0.0:
                concentration_status = "undefined_zero_vector"
                maximum_share = effective_count = np.nan
            else:
                concentration_status = "defined"
                maximum_share = float(np.abs(beta).max() / l1)
                effective_count = float(l1 * l1 / np.square(beta).sum())
            concentration_rows.append({
                "route": account["route"], "stage": account["stage"],
                "fit_timestamp": update["fit_timestamp"],
                "selected_factor_count": len(selected), "coefficient_count": len(beta),
                "nonzero_coefficient_count": int(np.count_nonzero(beta)),
                "coefficient_l1": l1, "max_absolute_coefficient_share": maximum_share,
                "effective_factor_count": effective_count,
                "concentration_status": concentration_status,
                "zero_vector_update": concentration_status == "undefined_zero_vector",
                "no_coefficient_vector": concentration_status == "undefined_no_coefficient_vector",
                "alpha": update.get("alpha"), "regularization": update.get("regularization"),
                "validation_net_sharpe": update.get("validation_net_sharpe"),
                "validation_net_return": update.get("validation_net_return"),
                "selected_factors": json.dumps(selected, ensure_ascii=False),
                "proposed_factors": json.dumps(update.get("proposed_factors", []), ensure_ascii=False),
                "coefficient_scope": "full factor-space vector; unselected factors are zero-filled; cash updates have no fitted vector",
            })
            for factor, value in coefficients.items():
                coefficient_values.setdefault((account["route"], account["stage"], str(factor)), []).append(float(value))
            if update.get("alpha") is not None:
                alpha_values.setdefault((account["route"], account["stage"]), []).append(float(update["alpha"]))
            regularization = update.get("regularization")
            if regularization is not None and isinstance(regularization, (int, float)):
                parameter_values.setdefault((account["route"], account["stage"], "regularization"), []).append(float(regularization))
    frequency_rows = []
    route_fit_count = {key: len(value) for key, value in route_schedules.items()}
    for (route, stage, factor), count in sorted(frequency.items()):
        fits = route_fit_count[(route, stage)]
        frequency_rows.append({"route": route, "stage": stage, "factor": factor,
                               "selected_updates": count, "fit_updates": fits,
                               "selection_frequency": count / fits if fits else np.nan})
    weight_rows = []
    for (route, stage, factor), values in sorted(coefficient_values.items()):
        array = np.asarray(values, dtype=float)
        weight_rows.append({"route": route, "stage": stage, "parameter": "coefficient",
                            "factor": factor, "count": len(array), "mean": float(array.mean()),
                            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                            "median": float(np.median(array)), "p05": float(np.quantile(array, .05)),
                            "p95": float(np.quantile(array, .95)),
                            "mean_absolute": float(np.abs(array).mean()),
                            "positive_share": float((array > 0).mean()),
                            "negative_share": float((array < 0).mean()),
                            "coefficient_scope": "full factor-space vector; unselected factors are zero-filled; cash updates have no fitted vector"})
    for (route, stage), values in sorted(alpha_values.items()):
        array = np.asarray(values, dtype=float)
        weight_rows.append({"route": route, "stage": stage, "parameter": "alpha", "factor": "",
                            "count": len(array), "mean": float(array.mean()),
                            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                            "median": float(np.median(array)), "p05": float(np.quantile(array, .05)),
                            "p95": float(np.quantile(array, .95)), "mean_absolute": float(np.abs(array).mean()),
                            "positive_share": float((array > 0).mean()), "negative_share": float((array < 0).mean()),
                            "coefficient_scope": "not applicable"})
    param_rows = []
    for (route, stage, name), values in sorted(parameter_values.items()):
        array = np.asarray(values, dtype=float)
        param_rows.append({"route": route, "stage": stage, "parameter": name, "count": len(array),
                           "unique_values": json.dumps(sorted(set(map(float, array)))),
                           "mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                           "median": float(np.median(array)), "p05": float(np.quantile(array, .05)),
                           "p95": float(np.quantile(array, .95))})
    turnover_rows = []
    for (route, stage), schedule in sorted(route_schedules.items()):
        ordered = sorted(schedule, key=lambda item: item["fit_timestamp"])
        entrants = exits = union_total = transitions = 0
        for old, new in zip(ordered, ordered[1:]):
            old_set, new_set = set(old.get("selected_factors", [])), set(new.get("selected_factors", []))
            entrants += len(new_set - old_set)
            exits += len(old_set - new_set)
            union_total += len(old_set | new_set)
            transitions += 1
        turnover_rows.append({"route": route, "stage": stage, "fit_updates": len(ordered),
                              "transitions": transitions, "total_factor_entries": entrants,
                              "total_factor_exits": exits,
                              "mean_jaccard_distance": (1 - sum(
                                  len(set(a.get("selected_factors", [])) & set(b.get("selected_factors", []))) /
                                  len(set(a.get("selected_factors", [])) | set(b.get("selected_factors", [])))
                                  if set(a.get("selected_factors", [])) | set(b.get("selected_factors", [])) else 0.0
                                  for a, b in zip(ordered, ordered[1:])) / transitions) if transitions else np.nan,
                              "factor_turnover_per_transition": (entrants + exits) / transitions if transitions else np.nan})
    concentration = pd.DataFrame(concentration_rows)
    concentration_summary_rows = []
    concentration_summary_columns = ["route", "stage", "fit_updates", "zero_vector_updates",
                                     "zero_vector_update_share", "no_coefficient_vector_updates",
                                     "no_coefficient_vector_share", "mean_selected_factor_count",
                                     "mean_nonzero_coefficient_count"]
    if not concentration.empty:
        for (route, stage), group in concentration.groupby(["route", "stage"]):
            updates = len(group)
            zeros = int(group["zero_vector_update"].sum())
            no_vectors = int(group["no_coefficient_vector"].sum())
            concentration_summary_rows.append({
                "route": route, "stage": stage, "fit_updates": updates,
                "zero_vector_updates": zeros,
                "zero_vector_update_share": zeros / updates if updates else np.nan,
                "no_coefficient_vector_updates": no_vectors,
                "no_coefficient_vector_share": no_vectors / updates if updates else np.nan,
                "mean_selected_factor_count": float(group["selected_factor_count"].mean()),
                "mean_nonzero_coefficient_count": float(group["nonzero_coefficient_count"].mean()),
            })
    return (pd.DataFrame(frequency_rows), pd.DataFrame(weight_rows), pd.DataFrame(param_rows),
            pd.DataFrame(turnover_rows), concentration,
            pd.DataFrame(concentration_summary_rows, columns=concentration_summary_columns))


def _window_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["experiment", "route_family", "method", "synthesis", "seed", "stage",
               "route_21d", "route_90d", "net_return_21d", "net_return_90d",
               "selection_source_bars_21d", "selection_source_bars_90d",
               "internal_validation_account_hours_21d", "internal_validation_account_hours_90d",
               "delta_net_return_90d_minus_21d", "sharpe_21d", "sharpe_90d",
               "delta_sharpe_90d_minus_21d", "gross_exposure_21d", "gross_exposure_90d",
               "delta_gross_exposure"]
    if rows.empty or "selection_window_days" not in rows:
        return pd.DataFrame(columns=columns)
    base = rows.loc[rows["cost"] == "base"].copy()
    if "verified" in base:
        base = base.loc[base["verified"]]
    if "route_complete" in base:
        base = base.loc[base["route_complete"]]
    dimensions = [name for name in ("experiment", "route_family", "method", "synthesis",
                                    "seed", "stage") if name in base]
    comparisons = []
    for keys, group in base.groupby(dimensions, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        by_window = {int(float(row.selection_window_days)): row for row in group.itertuples()
                     if pd.notna(row.selection_window_days)}
        if 21 not in by_window or 90 not in by_window:
            continue
        short, long = by_window[21], by_window[90]
        comparisons.append({**dict(zip(dimensions, keys)), "route_21d": short.route,
                            "route_90d": long.route,
                            "net_return_21d": short.net_return, "net_return_90d": long.net_return,
                            "selection_source_bars_21d": short.selection_source_bars,
                            "selection_source_bars_90d": long.selection_source_bars,
                            "internal_validation_account_hours_21d": short.internal_validation_account_hours,
                            "internal_validation_account_hours_90d": long.internal_validation_account_hours,
                            "delta_net_return_90d_minus_21d": long.net_return - short.net_return,
                            "sharpe_21d": short.sharpe_ratio, "sharpe_90d": long.sharpe_ratio,
                            "delta_sharpe_90d_minus_21d": long.sharpe_ratio - short.sharpe_ratio,
                            "gross_exposure_21d": short.average_gross_exposure,
                            "gross_exposure_90d": long.average_gross_exposure,
                            "delta_gross_exposure": long.average_gross_exposure - short.average_gross_exposure})
    return pd.DataFrame(comparisons, columns=columns)


def _synthesis_kind(value: Any) -> str | None:
    if value == "elastic_net":
        return None
    if value in {"equal_weight", "ridge"}:
        return value
    raise ValueError(f"unsupported candidate synthesis: {value!r}")


def _paired_candidate_scores(root: Path, summaries: list[dict[str, Any]],
                             cache_path: Path, code_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["window_id", "route_family", "experiment", "method", "selection_window_days", "seed",
               "paired_subsets", "equal_net_sharpe_mean", "ridge_best_net_sharpe_mean",
               "ridge_minus_equal_mean", "ridge_minus_equal_median", "ridge_win_share",
               "best_equal_net_sharpe", "best_ridge_net_sharpe", "best_paired_delta", "best_ridge_alpha"]
    files = sorted(root.glob("routes/*/windows/*/candidate_trials.json.gz"))
    if not files:
        return pd.DataFrame(columns=columns), {"status": "unavailable", "files": 0, "paired_windows": 0,
                                "errors": []}
    route_windows: dict[str, list[Path]] = {}
    for path in files:
        route_windows.setdefault(path.parent.name, []).append(path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    new_cache = {}
    result_rows: list[dict[str, Any]] = []
    errors = []
    for window_id, window_files in sorted(route_windows.items()):
        input_hashes = {str(path.relative_to(root)): _file_sha256(path) for path in window_files}
        cached = cache.get(window_id, {})
        if (cached.get("input_hashes") == input_hashes and cached.get("code_sha256") == code_hash):
            group_rows = cached.get("rows", [])
            group_errors = cached.get("errors", [])
        else:
            scores: dict[tuple[str, str, str, Any, Any, str], dict[str, Any]] = {}
            group_errors = []
            for path in window_files:
                try:
                    with gzip.open(path, "rt", encoding="utf-8") as source:
                        payload = json.load(source)
                    candidate_rows = payload["candidate_rows"]
                    if not isinstance(candidate_rows, list):
                        raise ValueError("candidate_rows must be a list")
                    required_fields = {"route_id", "route_family", "experiment", "method",
                                       "synthesis", "selection_window_days", "seed", "identity",
                                       "validation_net_sharpe", "alpha", "valid_objective"}
                    for candidate in candidate_rows:
                        if not isinstance(candidate, dict):
                            raise ValueError("candidate rows must be objects")
                        missing = required_fields - candidate.keys()
                        if missing:
                            raise ValueError(f"candidate row is missing fields: {sorted(missing)}")
                        for field in ("route_id", "route_family", "experiment", "method"):
                            if not isinstance(candidate[field], str) or not candidate[field]:
                                raise ValueError(f"candidate {field} must be a nonempty string")
                        if candidate["experiment"] not in {"E1", "E2", "E3", "E4", "E5"}:
                            raise ValueError("candidate experiment must be E1 through E5")
                        if type(candidate["selection_window_days"]) is not int \
                                or candidate["selection_window_days"] not in {21, 90}:
                            raise ValueError("candidate selection_window_days must be 21 or 90")
                        if type(candidate["seed"]) is not int:
                            raise ValueError("candidate seed must be an integer")
                        if candidate["route_id"] != path.parents[2].name:
                            raise ValueError("candidate route_id differs from its saved route directory")
                        if candidate["synthesis"] not in {"equal_weight", "ridge", "elastic_net"}:
                            raise ValueError(f"unsupported candidate synthesis: {candidate['synthesis']!r}")
                        if type(candidate["valid_objective"]) is not bool:
                            raise ValueError("candidate valid_objective must be a boolean")
                        if not isinstance(candidate["identity"], list):
                            raise ValueError("candidate identity must be a factor-name list")
                        if not candidate["identity"] or any(
                                not isinstance(name, str) or not name for name in candidate["identity"]):
                            raise ValueError("candidate identity entries must be nonempty factor names")
                        sharpe = candidate["validation_net_sharpe"]
                        if sharpe is not None and (type(sharpe) not in {int, float}
                                                   or not np.isfinite(float(sharpe))):
                            raise ValueError("candidate validation_net_sharpe must be finite or null")
                        alpha = candidate["alpha"]
                        if alpha is not None and (type(alpha) not in {int, float}
                                                  or not np.isfinite(float(alpha))):
                            raise ValueError("candidate alpha must be finite or null")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    group_errors.append({"path": str(path.relative_to(root)), "error": str(exc)})
                    continue
                for candidate in candidate_rows:
                    if not candidate["valid_objective"]:
                        continue
                    kind = _synthesis_kind(candidate["synthesis"])
                    if kind is None:
                        continue
                    identity = candidate["identity"]
                    subset_key = json.dumps(sorted(map(str, identity)), separators=(",", ":"))
                    if candidate["validation_net_sharpe"] is None:
                        continue
                    sharpe = float(candidate["validation_net_sharpe"])
                    if not np.isfinite(sharpe):
                        continue
                    method = candidate["method"]
                    family = candidate["route_family"]
                    experiment = candidate["experiment"]
                    selection_days = candidate["selection_window_days"]
                    seed = candidate["seed"]
                    key = (family, experiment, method, selection_days, seed, subset_key)
                    score = scores.setdefault(key, {"equal": [], "ridge": []})
                    alpha = candidate["alpha"]
                    alpha_value = float(alpha) if alpha is not None else None
                    score["equal" if kind == "equal_weight" else "ridge"].append((sharpe, alpha_value))
            by_method: dict[tuple[str, str, str, Any, Any], list[dict[str, Any]]] = {}
            for (family, experiment, method, selection_days, seed, subset), values in scores.items():
                if not values["equal"] or not values["ridge"]:
                    continue
                equal_score = max(values["equal"], key=lambda item: item[0])[0]
                ridge_score, ridge_alpha = sorted(
                    values["ridge"], key=lambda item: (-item[0], item[1] if item[1] is not None else math.inf))[0]
                by_method.setdefault((family, experiment, method, selection_days, seed), []).append({
                    "subset_id": subset, "equal_net_sharpe": equal_score,
                    "ridge_best_net_sharpe": ridge_score, "ridge_alpha": ridge_alpha,
                    "ridge_minus_equal_net_sharpe": ridge_score - equal_score,
                })
            group_rows = []
            for (family, experiment, method, selection_days, seed), pairs in sorted(
                    by_method.items(), key=lambda item: tuple(str(value) for value in item[0])):
                deltas = np.asarray([item["ridge_minus_equal_net_sharpe"] for item in pairs], dtype=float)
                group_rows.append({
                    "window_id": window_id, "route_family": family, "experiment": experiment,
                    "method": method, "selection_window_days": selection_days, "seed": seed,
                    "paired_subsets": len(pairs),
                    "equal_net_sharpe_mean": float(np.mean([item["equal_net_sharpe"] for item in pairs])),
                    "ridge_best_net_sharpe_mean": float(np.mean([item["ridge_best_net_sharpe"] for item in pairs])),
                    "ridge_minus_equal_mean": float(deltas.mean()),
                    "ridge_minus_equal_median": float(np.median(deltas)),
                    "ridge_win_share": float((deltas > 0).mean()),
                    "best_equal_net_sharpe": max(item["equal_net_sharpe"] for item in pairs),
                    "best_ridge_net_sharpe": max(item["ridge_best_net_sharpe"] for item in pairs),
                    "best_paired_delta": float(deltas.max()),
                    "best_ridge_alpha": next(item["ridge_alpha"] for item in pairs
                                               if item["ridge_best_net_sharpe"] == max(
                                                   pair["ridge_best_net_sharpe"] for pair in pairs)),
                })
            new_cache[window_id] = {"input_hashes": input_hashes, "code_sha256": code_hash,
                                    "rows": group_rows, "errors": group_errors}
        result_rows.extend(group_rows)
        errors.extend(group_errors)
    # Retain unchanged cache groups too; only groups whose file set changed are recomputed.
    for window_id, value in cache.items():
        if window_id not in new_cache and window_id in route_windows:
            new_cache[window_id] = value
    cache_path.write_text(json.dumps(new_cache, indent=2, allow_nan=False) + "\n")
    summary = {"status": "passed" if not errors and result_rows else "failed" if errors else "no_pairs",
               "files": len(files), "paired_windows": len({row["window_id"] for row in result_rows}),
               "paired_groups": len(result_rows), "errors": errors}
    return pd.DataFrame(result_rows, columns=columns), summary


def _qualification(row: dict[str, Any], stress_row: dict[str, Any] | None) -> list[str]:
    failures = []
    if not row.get("route_complete", False):
        failures.append("route_missing_dev_or_C_base_stress_account")
    if not row["verified"]:
        failures.append("base_account_unverified")
    if stress_row is None:
        failures.append("stress_account_missing")
    elif not stress_row["verified"]:
        failures.append("stress_account_unverified")
    if not np.isfinite(row["net_return"]) or row["net_return"] <= 0:
        failures.append("base_net_return_not_positive")
    if not np.isfinite(row["sharpe_ratio"]) or row["sharpe_ratio"] <= 0:
        failures.append("base_net_sharpe_not_positive_finite")
    if not np.isfinite(row["annualized_volatility"]) or row["annualized_volatility"] <= 0:
        failures.append("base_volatility_not_positive_finite")
    if row["traded_bars"] <= 0:
        failures.append("no_traded_bars")
    if not np.isfinite(row["max_drawdown"]) or row["max_drawdown"] > 0.15:
        failures.append("max_drawdown_over_15pct")
    if stress_row is not None and (not np.isfinite(stress_row["net_return"]) or stress_row["net_return"] < 0):
        failures.append("stress_net_return_negative")
    return failures


def _mark_failed(account: dict[str, Any], reason: str) -> None:
    account["verification"]["status"] = "failed"
    account["verification"].setdefault("errors", []).append(reason)


def _experiment_progress(accounts: list[dict[str, Any]]) -> dict[str, Any]:
    expected_routes = {"E1": 2, "E2": 3, "E3": 6, "E4": 2, "E5": 4}
    output = {}
    for experiment, route_count in expected_routes.items():
        routes = {item["route"] for item in accounts if item["metadata"].get("experiment") == experiment}
        route_complete = 0
        for route in routes:
            members = [item for item in accounts if item["route"] == route]
            keys = {(item["stage"], item["cost"]) for item in members}
            if keys == {("development", "base"), ("development", "stress"), ("C", "base"), ("C", "stress")} \
                    and all(item["verification"]["status"] == "passed" for item in members):
                route_complete += 1
        expected_accounts = route_count * 4
        received = sum(1 for item in accounts if item["metadata"].get("experiment") == experiment)
        verified = sum(1 for item in accounts if item["metadata"].get("experiment") == experiment
                       and item["verification"]["status"] == "passed")
        status = "complete" if route_complete == route_count and received == expected_accounts and verified == expected_accounts \
            else "failed" if any(item["metadata"].get("experiment") == experiment
                                 and item["verification"]["status"] == "failed" for item in accounts) \
            else "incomplete"
        output[experiment] = {"expected_routes": route_count, "received_routes": len(routes),
                              "complete_routes": route_complete, "expected_accounts": expected_accounts,
                              "received_accounts": received, "verified_accounts": verified, "status": status}
    return output


def _selected_scope(machine_state: dict[str, Any] | None,
                    accounts: list[dict[str, Any]]) -> tuple[int, int]:
    statuses = (machine_state or {}).get("statuses", {})
    selected_statuses = {"running", "partial", "passed", "failed"}
    experiments = {item["metadata"].get("experiment") for item in accounts}
    e4_selected = (statuses.get("E4", {}).get("status") in selected_statuses or "E4" in experiments)
    e5_selected = (statuses.get("E5", {}).get("status") in selected_statuses or "E5" in experiments)
    if e5_selected:
        e4_selected = True
    frozen = (machine_state or {}).get("expected_stage_accounts", {})
    core_accounts = int(frozen.get("core_E1_E3", 44))
    expanded_accounts = int(frozen.get("expanded_E1_E5", 68))
    expected_accounts = core_accounts + (8 if e4_selected else 0) + (16 if e5_selected else 0)
    expected_routes = 11 + (2 if e4_selected else 0) + (4 if e5_selected else 0)
    if expected_accounts > expanded_accounts:
        raise ValueError("selected account scope exceeds the frozen expanded account budget")
    return expected_accounts, expected_routes


def _selection_rule(root: Path) -> tuple[set[str] | None, dict[str, Any]]:
    path = root / "selection_rule.json"
    if not path.exists():
        return None, {"status": "missing", "path": str(path)}
    try:
        rule = json.loads(path.read_text())
        contract = json.loads((root / "run_contract.json").read_text())
        if rule != contract.get("selection_rule"):
            raise ValueError("selection_rule.json differs from the frozen run_contract selection_rule")
        if rule.get("schema_version") != 1 or rule.get("decision_stage") != "development":
            raise ValueError("unsupported primary selection rule schema or decision stage")
        eligible_experiments = rule.get("eligible_experiments")
        excluded_experiments = rule.get("excluded_experiments")
        route_ids = rule.get("eligible_route_ids")
        if eligible_experiments != ["E2", "E3"] or excluded_experiments != ["E1", "E4", "E5"]:
            raise ValueError("primary selection eligibility differs from frozen E2/E3 rule")
        if not isinstance(route_ids, list) or len(route_ids) != 9 or len(set(route_ids)) != 9:
            raise ValueError("primary selection rule must contain the nine frozen E2/E3 route identities")
        admission = rule["admission"]
        if admission.get("stage") != "development" \
                or admission.get("base", {}).get("net_return") != {"operator": ">", "value": 0.0} \
                or admission.get("base", {}).get("net_sharpe") != {"operator": ">", "value": 0.0, "finite": True} \
                or admission.get("base", {}).get("traded_bars") != {"operator": ">", "value": 0} \
                or admission.get("base", {}).get("max_drawdown") != {"operator": "<=", "value": 0.15} \
                or admission.get("stress") != {"net_return": {"operator": ">=", "value": 0.0}}:
            raise ValueError("primary selection admission gates differ from the frozen contract")
        ranking = rule["ranking"]
        if ranking.get("primary_metric") != "development_base_net_sharpe" \
                or ranking.get("primary_direction") != "descending" \
                or ranking.get("tie_break_field") != "route_id" \
                or ranking.get("tie_break_direction") != "ascending_lexicographic" \
                or ranking.get("fixed_route_identity_order") != sorted(route_ids):
            raise ValueError("primary ranking or exact-tie rule differs from the frozen contract")
        return set(route_ids), {"status": "passed", "sha256": _file_sha256(path),
                                "eligible_route_ids": sorted(route_ids)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, {"status": "failed", "path": str(path), "error": str(exc)}


def _equity_plot(path: Path, accounts: list[dict[str, Any]], stage: str) -> None:
    selected = [account for account in accounts if account["stage"] == stage and account["cost"] == "base"
                and account["verification"]["status"] == "passed"]
    if not selected:
        path.unlink(missing_ok=True)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(13, 7.5), constrained_layout=True)
    for account in sorted(selected, key=lambda item: item["route"]):
        equity = account["daily_equity"]
        start = _parsed_utc(account["metadata"]["start"], "account start")
        dates = pd.DatetimeIndex([start]).append(equity.index)
        values = np.r_[float(account["metadata"]["initial_capital"]), equity.to_numpy(dtype=float)]
        axis.plot(dates, values, linewidth=1.2, label=account["route"])
    axis.set_title(f"{stage} · base-cost account equity")
    axis.set_xlabel("UTC date")
    axis.set_ylabel("Equity (USDT)")
    axis.grid(True, alpha=0.25)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.savefig(path, format=path.suffix.lstrip("."), dpi=150)
    plt.close(fig)


def _write_morning_report(root: Path, rows: pd.DataFrame,
                          verification: dict[str, Any], window_rows: pd.DataFrame,
                          paired_rows: pd.DataFrame, paired_status: dict[str, Any],
                          concentration_summary: pd.DataFrame, selection_rule_status: dict[str, Any],
                          experiment_progress: dict[str, Any], machine_state: dict[str, Any] | None,
                          final_selection_ready: bool, winner: dict[str, Any] | None) -> None:
    passed, failed = verification["passed_accounts"], verification["failed_accounts"]
    lines = ["# 多因子组合 90 天实验报告", "",
             "本报告根据保存的连续全引擎账户生成。最终路线只能用开发段准入后按净夏普排序；C 段是已经使用过的历史评价数据，不参与选冠。没有启动前向或 Paper 账户。", "",
             f"当前纳入 {verification['account_count']} 个账户，独立核验通过 {passed} 个，未通过 {failed} 个。",
             f"冻结主选规则 `selection_rule.json`：{selection_rule_status['status']}。", ""]
    if failed:
        lines += ["未通过的账户明确标为未核验，并排除在准入和净值图之外。", ""]
    lines += ["## 队列进度", "",
              "| 阶段 | 预期账户数 | 已收到 | 已核验通过 | 当前状态 |",
              "|---|---:|---:|---:|---|"]
    for label, detail in experiment_progress.items():
        lines.append(f"| {label} | {detail['expected_accounts']} | {detail['received_accounts']} | {detail['verified_accounts']} | {detail['status']} |")
    lines.append("")
    if machine_state:
        lines += ["机器状态清单 `machine_state.json` 的 E0–E6 状态：", ""]
        for stage_id in ("E0", "E1", "E2", "E3", "E4", "E5", "E6"):
            item = machine_state.get("statuses", {}).get(stage_id, {})
            if item:
                completed = len(item.get("completed_fit_times", []))
                lines.append(f"- {stage_id}：{item.get('status', 'unknown')}；已完成更新点 {completed}；已记录阶段账户 {item.get('stage_accounts', 0)}。")
        lines.append("")
    if not window_rows.empty:
        lines += ["## 21 天与 90 天选优窗口配对", "",
                  "正差值表示 90 天路线的数值较高；敞口变化以毛敞口比例计。源信号条数及内部验证账本小时数均由账户模型计划中保存的边界计算。表内全段账户指标按保存的完整评分日期计算。", "",
                  "| 路线族 | 方法 | 合成 | 阶段 | 净收益 21 天 | 净收益 90 天 | 收益差 | 净夏普 21 天 | 净夏普 90 天 | 夏普差 | 源信号条数 21/90 天 | 内部验证账本小时 21/90 天 | 毛敞口差 |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in window_rows.itertuples(index=False):
            lines.append(f"| {row.route_family} | {row.method} | {row.synthesis} | {row.stage} | {row.net_return_21d:.2%} | {row.net_return_90d:.2%} | {row.delta_net_return_90d_minus_21d:+.2%} | {row.sharpe_21d:.3f} | {row.sharpe_90d:.3f} | {row.delta_sharpe_90d_minus_21d:+.3f} | {row.selection_source_bars_21d}/{row.selection_source_bars_90d} | {row.internal_validation_account_hours_21d}/{row.internal_validation_account_hours_90d} | {row.delta_gross_exposure:+.2%} |")
        lines.append("")
    if not paired_rows.empty:
        lines += ["## 同一子集的等权与 Ridge 评分", "",
                  "数据来自已保存的内部选优候选评分，不增加账户回测。每个子集的 Ridge 正则档按同一内部净夏普取最高档，再与等权分数配对。它只说明相同因子名单的分数组合差异，不等于整条流程的因果收益。", "",
                  "| 实验 | 路线族 | 方法 | 选优窗口 | 种子 | 更新窗口数 | 配对子集数 | Ridge 减等权均值 | 窗口中位数差 | Ridge 胜出占比 |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        group_columns = ["experiment", "route_family", "method", "selection_window_days", "seed"]
        for keys, group in paired_rows.groupby(group_columns, dropna=False):
            experiment, family, method, selection_days, seed = keys
            pairs = group["paired_subsets"].to_numpy(dtype=float)
            mean_delta = float(np.average(group["ridge_minus_equal_mean"], weights=pairs))
            win_share = float(np.average(group["ridge_win_share"], weights=pairs))
            lines.append(f"| {experiment} | {family} | {method} | {selection_days} | {seed} | {len(group)} | {int(pairs.sum())} | {mean_delta:+.3f} | {group['ridge_minus_equal_median'].median():+.3f} | {win_share:.1%} |")
        lines.append("明细按更新窗口列在 `paired_score_summary.csv`。")
        lines.append("")
    else:
        lines += ["## 同一子集的等权与 Ridge 评分", "",
                  f"当前没有可配对的已保存候选评分（状态：{paired_status['status']}）；本报告不把不同因子名单的流程差异称作纯权重收益。", ""]
    if not concentration_summary.empty:
        lines += ["## 权重集中度与零向量", "",
                  "每次更新按保存的全因子空间系数向量计算最大绝对权重占比与有效因子数；未选因子的系数明确记为零，cash 更新没有拟合向量并单独统计。Elastic Net 或其他路线出现全零拟合向量时，单独统计为零向量更新。", "",
                  "| 路线 | 阶段 | 更新次数 | 零向量更新 | 占比 | 无系数向量更新 | 平均所选因子数 | 平均非零系数数 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in concentration_summary.itertuples(index=False):
            lines.append(f"| {row.route} | {row.stage} | {row.fit_updates} | {row.zero_vector_updates} | {row.zero_vector_update_share:.1%} | {row.no_coefficient_vector_updates} | {row.mean_selected_factor_count:.2f} | {row.mean_nonzero_coefficient_count:.2f} |")
        lines.append("全零系数仍按预先冻结的评分并列规则处理，不因此改动交易准入条件。")
        lines.append("")
    dev = rows.loc[(rows["stage"] == "development") & (rows["cost"] == "base")].copy() if not rows.empty else pd.DataFrame()
    if not dev.empty:
        lines += ["## 开发段准入与排序", "",
                  "固定准入条件：基础成本净收益严格大于零、净夏普与波动率有限且为正、至少一个有成交小时、最大回撤不超过 15%、两倍成本净收益不小于零。只有 E2/E3 主选路线进入净夏普排序，同分按路线身份排序；E1、E4、E5 用作描述性对照。", "",
                  "| 路线 | 实验 | 选择范围 | 方法 | 合成 | 净收益 | 净夏普 | 最大回撤 | 两倍成本净收益 | 毛敞口 | 持仓时间 | 成交笔数 | 状态 |",
                  "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for row in dev.sort_values(["primary_selection_eligible", "qualified", "sharpe_ratio", "route"],
                                   ascending=[False, False, False, True]).itertuples(index=False):
            gate = "通过" if row.qualified else "未过：" + row.gate_failures
            scope = "主选" if row.primary_selection_eligible else "描述对照"
            lines.append(f"| {row.route} | {row.experiment} | {scope} | {row.method} | {row.synthesis} | {row.net_return:.2%} | {row.sharpe_ratio:.3f} | {row.max_drawdown:.2%} | {row.stress_net_return:.2%} | {row.average_gross_exposure:.2%} | {row.position_time_fraction:.2%} | {row.fills} | {gate} |")
        qualified = dev.loc[dev["qualified"]]
        primary_qualified = qualified.loc[qualified["primary_selection_eligible"]]
        if primary_qualified.empty and qualified.empty:
            lines += ["", "目前没有路线通过冻结的开发段准入条件。"]
        elif primary_qualified.empty:
            lines += ["", "当前通过准入的只有 E1/E4/E5 描述性对照组；尚无 E2/E3 主选范围内的达标路线。"]
        elif not final_selection_ready:
            current = primary_qualified.sort_values(["sharpe_ratio", "route"], ascending=[False, True]).iloc[0]
            lines += ["", f"当前已完成路线中净夏普领先者为 **{current['route']}**（{current['sharpe_ratio']:.3f}）；核心 E1–E3 尚未全部完成，因此这是阶段领先者，不作最终选择。"]
        else:
            if winner is None:
                lines += ["", "E2/E3 主选路线中没有候选通过冻结的开发段准入条件。"]
            else:
                lines += ["", f"按冻结准入条件和净夏普排序，开发段选出的 E2/E3 主路线为 **{winner['route']}**（净夏普 {winner['development_net_sharpe']:.3f}）。"]
            if winner is None:
                c = rows.iloc[0:0]
            else:
                c = rows.loc[(rows["route"] == winner["route"]) & (rows["stage"] == "C") & (rows["cost"] == "base")]
            if not c.empty:
                cm = c.iloc[0]
                lines.append(f"该路线 C 段评价为净收益 {cm['net_return']:.2%}、净夏普 {cm['sharpe_ratio']:.3f}；此结果没有改变选路。")
        lines.append("")
    lines += ["## 对照与核验文件", "",
              "`comparison.csv` 列出每条路线在开发段和 C 段的基础成本与两倍成本账户。月度、年度收益按小时净值连乘；币种贡献含已实现盈亏、期末未实现盈亏、资金费和手续费。滑点已体现在成交价中，单独列示但不重复扣减。", "",
              "独立核验检查账户边界、订单与成交数量、成交价和费率、持仓及开仓成本的生命周期、资金费持仓时点和现金流、现金与净值恒等式、小时收益率、风险指标和敞口。账户逐项结果见 `account_verification.json`。审计缓存按账户文件 SHA-256 和核验器版本校验，缓存命中数也记录在该文件中。", "",
              "- `comparison.csv`、`development_selection.csv`、`window_comparison.csv`、`paired_score_summary.csv`",
              "- `monthly.csv`、`yearly.csv`、`symbol_contribution.csv`、`turnover_categories.csv`",
              "- `selection_frequency.csv`、`selection_turnover.csv`、`weight_distribution.csv`、`fit_weight_concentration.csv`、`fit_weight_summary.csv`、`parameter_distribution.csv`",
              "- `plots/equity_development.svg`、`plots/equity_development.png`、`plots/equity_C.svg`、`plots/equity_C.png`",
              "- `account_verification.json`、`experiment_progress.json`", ""]
    (root / "morning_report.md").write_text("\n".join(lines))


def build_report(run_root: Path, account_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconcile supplied accounts and refresh combination90 report artifacts.

    Each manifest record must contain a path to the account directory. The
    directory has account.json, metrics.json, and saved CSVs for the ledger,
    orders, fills, positions, and funding events.
    """
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    eligible_route_ids, selection_rule_status = _selection_rule(root)
    cache_path = root / "account_verification_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    verifier_hash = _file_sha256(Path(__file__))
    summaries: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    monthly_parts, yearly_parts, contribution_parts = [], [], []
    turnover_rows: list[dict[str, Any]] = []
    new_cache = dict(cache)
    seen_keys: set[tuple[str, str, str]] = set()

    # Load, audit, summarize, and release one account at a time. Position and
    # funding tables are large; keeping all 44–68 accounts resident can exceed RAM.
    for record in account_records:
        directory, metadata = _metadata_for_record(root, record)
        route, stage, cost = _route_key(metadata), _stage(metadata), _cost_name(metadata)
        identity = (route, stage, cost)
        if identity in seen_keys:
            raise ValueError(f"duplicate account route/stage/cost identity: {identity}")
        seen_keys.add(identity)
        relative_directory = str(directory.relative_to(root)) if directory.is_relative_to(root) else str(directory)
        if metadata.get("account_status") == "failed_at_account_guard":
            failure_file = directory / "account_failure.json"
            failure = json.loads(failure_file.read_text()) if failure_file.exists() else None
            reason = json.dumps(failure, ensure_ascii=False, sort_keys=True) if failure is not None else "account_failure.json is missing"
            verification = {"path": str(directory), "status": "failed", "route": route,
                            "stage": stage, "cost": cost, "cache_hit": False,
                            "errors": ["account guard stopped the full account: " + reason]}
            try:
                _validate_route_metadata(metadata)
            except ValueError as exc:
                verification["errors"].append(str(exc))
            if eligible_route_ids is not None and bool(metadata["primary_selection_eligible"]) != (route in eligible_route_ids):
                verification["errors"].append("account primary-selection flag differs from frozen selection_rule.json")
            row_records.append(_failed_account_row(route, stage, cost, metadata, verification))
            checks.append(verification)
            summaries.append({"metadata": metadata, "route": route, "stage": stage, "cost": cost,
                              "verification": verification, "daily_equity": pd.Series(dtype=float),
                              "funding_input_sha256": None,
                              "model_schedule_json": json.dumps(metadata.get("model_schedule"),
                                                                ensure_ascii=False, sort_keys=True,
                                                                separators=(",", ":"))})
            continue
        try:
            hashes = _account_artifact_hashes(directory)
        except OSError as exc:
            check = {"path": str(directory), "status": "failed", "route": route,
                     "stage": stage, "cost": cost, "cache_hit": False,
                     "artifact_hashes": {}, "errors": [str(exc)]}
            checks.append(check)
            continue
        cache_entry = cache.get(relative_directory, {})
        cache_hit = (cache_entry.get("artifact_hashes") == hashes
                     and cache_entry.get("verifier_sha256") == verifier_hash
                     and cache_entry.get("verification", {}).get("status") == "passed")
        verification = dict(cache_entry["verification"]) if cache_hit else {}
        tables = {}
        try:
            tables = _read_account(directory, metadata)
            if not cache_hit:
                verification = verify_account(directory, metadata, tables=tables)
            if eligible_route_ids is not None \
                    and bool(metadata["primary_selection_eligible"]) != (route in eligible_route_ids):
                verification["status"] = "failed"
                verification.setdefault("errors", []).append(
                    "account primary-selection flag differs from frozen selection_rule.json")
            verification.update({"route": route, "stage": stage, "cost": cost,
                                 "cache_hit": bool(cache_hit), "artifact_hashes": hashes})
        except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
            verification = {"path": str(directory), "status": "failed", "route": route,
                            "stage": stage, "cost": cost, "cache_hit": False,
                            "artifact_hashes": hashes, "errors": [str(exc)]}
            if not tables:
                try:
                    tables = _read_account(directory, metadata)
                except (ValueError, KeyError, OSError, TypeError):
                    checks.append(verification)
                    continue
        account = {"directory": directory, "metadata": metadata, "route": route, "stage": stage,
                   "cost": cost, "tables": tables, "verification": verification}
        row_records.append(_account_row(account))
        monthly, yearly = _monthly_yearly([account])
        monthly_parts.append(monthly)
        yearly_parts.append(yearly)
        contribution_parts.append(_contributions([account]))
        if cost == "base":
            fills, ledger = tables["fills"], tables["ledger"]
            categories = fills.copy()
            if not categories.empty:
                categories["trade_category"] = np.select(
                    [categories["previous_quantity"].eq(0),
                     categories["previous_quantity"] * categories["new_quantity"] < 0,
                     categories["new_quantity"].eq(0)],
                    ["entry", "direction_reversal", "exit"], default="resize")
                for category, group in categories.groupby("trade_category"):
                    turnover_rows.append({"route": route, "stage": stage,
                                          "verified": verification["status"] == "passed",
                                          "trade_category": category, "fills": len(group),
                                          "notional": float(group["notional"].sum()),
                                          "fees": float(group["fee"].sum()),
                                          "slippage_cost": float(group["slippage_cost"].sum())})
            turnover_rows.append({"route": route, "stage": stage,
                                  "verified": verification["status"] == "passed",
                                  "trade_category": "total", "fills": len(fills),
                                  "notional": float(fills["notional"].sum()),
                                  "fees": float(fills["fee"].sum()),
                                  "slippage_cost": float(fills["slippage_cost"].sum()),
                                  "total_turnover": float(ledger["turnover"].sum())})
        daily_equity = tables["ledger"]["equity"].resample("1D").last().dropna()
        summaries.append({"metadata": metadata, "route": route, "stage": stage, "cost": cost,
                          "verification": verification, "daily_equity": daily_equity,
                          "funding_input_sha256": _funding_input_hash(tables["funding"]),
                          "model_schedule_json": json.dumps(metadata.get("model_schedule"),
                                                            ensure_ascii=False, sort_keys=True,
                                                            separators=(",", ":"))})
        checks.append(verification)
        if verification["status"] == "passed":
            new_cache[relative_directory] = {"artifact_hashes": hashes,
                                             "verifier_sha256": verifier_hash,
                                             "verification": {key: value for key, value in verification.items()
                                                              if key != "cache_hit"}}
        del account, tables, monthly, yearly

    # Validate stage alignment and the frozen double-cost pair before ranking.
    by_key = {(item["route"], item["stage"], item["cost"]): item for item in summaries}
    for stage in ("development", "C"):
        stage_members = [item for item in summaries if item["stage"] == stage]
        signatures = {(_parsed_utc(item["metadata"]["start"], "account start"),
                       _parsed_utc(item["metadata"]["end"], "account end"),
                       float(item["metadata"].get("initial_capital", np.nan))) for item in stage_members}
        if len(signatures) > 1:
            for item in stage_members:
                _mark_failed(item, f"{stage} account bounds and initial capital are not aligned across routes")
        expected_end = pd.Timestamp("2025-08-01T00:00:00Z") if stage == "development" else pd.Timestamp("2026-08-01T00:00:00Z")
        expected_start = pd.Timestamp("2025-08-01T00:00:00Z") if stage == "C" else None
        for item in stage_members:
            start = _parsed_utc(item["metadata"]["start"], "account start")
            end = _parsed_utc(item["metadata"]["end"], "account end")
            if end != expected_end or (expected_start is not None and start != expected_start):
                _mark_failed(item, f"{stage} account bounds do not match the frozen historical interval")
            if float(item["metadata"].get("initial_capital", np.nan)) != 10_000.0:
                _mark_failed(item, "initial capital differs from the frozen 10,000 USDT")
    for route, stage in sorted({(item["route"], item["stage"]) for item in summaries}):
        base, stress = by_key.get((route, stage, "base")), by_key.get((route, stage, "stress"))
        if not base or not stress:
            continue
        bm, sm = base["metadata"], stress["metadata"]
        pair_errors = []
        for name in ("initial_capital", "start", "end"):
            if bm.get(name) != sm.get(name):
                pair_errors.append(f"base/stress {name} differs")
        for name in ("fee_bps", "slippage_bps"):
            try:
                if float(sm[name]) != 2.0 * float(bm[name]):
                    pair_errors.append(f"stress {name} is not exactly 2x base")
            except (KeyError, TypeError, ValueError):
                pair_errors.append(f"base/stress {name} is missing")
        if base["model_schedule_json"] != stress["model_schedule_json"]:
            pair_errors.append("base and stress model schedules differ")
        if base["funding_input_sha256"] is not None and stress["funding_input_sha256"] is not None \
                and base["funding_input_sha256"] != stress["funding_input_sha256"]:
            pair_errors.append("base and stress funding inputs differ")
        if pair_errors:
            for item in (base, stress):
                for error in pair_errors:
                    _mark_failed(item, error)

    # Enforce unique four-account route groups; partial routes remain visible but
    # cannot receive an admission result or a final rank.
    route_meta = {}
    for item in summaries:
        route_meta.setdefault(item["route"], []).append(item)
    route_completeness = {}
    required_pairs = {("development", "base"), ("development", "stress"), ("C", "base"), ("C", "stress")}
    for route, members in route_meta.items():
        pairs = {(item["stage"], item["cost"]) for item in members}
        complete = pairs == required_pairs and all(item["verification"]["status"] == "passed" for item in members)
        route_completeness[route] = {"complete": complete, "account_paths": len(members),
                                     "expected_account_paths": 4,
                                     "verified_account_paths": sum(item["verification"]["status"] == "passed" for item in members)}

    rows = pd.DataFrame(row_records)
    if rows.empty:
        rows = pd.DataFrame(columns=["route", "route_family", "method", "synthesis", "stage", "cost",
                                     "verified", "net_return", "sharpe_ratio", "max_drawdown"])
    for index, row in rows.iterrows():
        route = row["route"]
        completeness = route_completeness.get(route, {"complete": False, "account_paths": 0})
        rows.loc[index, "route_complete"] = bool(completeness["complete"])
        rows.loc[index, "completed_account_paths"] = completeness["account_paths"]
        match = next((item for item in summaries if (item["route"], item["stage"], item["cost"])
                      == (row["route"], row["stage"], row["cost"])), None)
        if match:
            rows.loc[index, "verified"] = match["verification"]["status"] == "passed"
            rows.loc[index, "verification_status"] = match["verification"]["status"]
    if eligible_route_ids is not None:
        rows["primary_selection_eligible"] = rows["route"].isin(eligible_route_ids)
    elif selection_rule_status["status"] == "failed" and not rows.empty:
        rows["primary_selection_eligible"] = False
    lookup = {(row.route, row.stage, row.cost): row._asdict() for row in rows.itertuples(index=False)}
    if not rows.empty:
        rows["qualified"] = False
        rows["gate_failures"] = ""
        rows["stress_net_return"] = np.nan
        for i, row in rows.iterrows():
            if row["stage"] != "development" or row["cost"] != "base":
                continue
            stress = lookup.get((row["route"], "development", "stress"))
            failures = _qualification(row.to_dict(), stress)
            rows.loc[i, "qualified"] = not failures
            rows.loc[i, "gate_failures"] = ";".join(failures)
            if stress is not None:
                rows.loc[i, "stress_net_return"] = stress["net_return"]
        dev_base = rows.loc[(rows["stage"] == "development") & (rows["cost"] == "base")].copy()
        dev_base = dev_base.sort_values(["qualified", "sharpe_ratio", "route"], ascending=[False, False, True])
        ranks = {row.route: rank for rank, row in enumerate(
            dev_base.loc[dev_base["qualified"] & dev_base["primary_selection_eligible"]].itertuples(index=False), 1)}
        rows["selection_rank"] = rows["route"].map(ranks)

    experiment_progress = _experiment_progress(summaries)
    (root / "experiment_progress.json").write_text(json.dumps(experiment_progress, indent=2) + "\n")
    machine_state_path = root / "machine_state.json"
    machine_state = json.loads(machine_state_path.read_text()) if machine_state_path.exists() else None
    core_complete = all(experiment_progress[name]["status"] == "complete" for name in ("E1", "E2", "E3"))
    expected_accounts, expected_routes = _selected_scope(machine_state, summaries)
    status_e0 = (machine_state or {}).get("statuses", {}).get("E0", {}).get("status")
    e0_complete = status_e0 in {"complete", "completed", "passed", "success"}
    final_selection_ready = core_complete and e0_complete and selection_rule_status["status"] == "passed"
    dev = rows.loc[(rows["stage"] == "development") & (rows["cost"] == "base")].copy() if not rows.empty else pd.DataFrame()
    candidates = dev.loc[dev["qualified"] & dev["primary_selection_eligible"]].sort_values(
        ["sharpe_ratio", "route"], ascending=[False, True]) if not dev.empty else pd.DataFrame()
    leader = {"route": str(candidates.iloc[0]["route"]),
              "development_net_sharpe": float(candidates.iloc[0]["sharpe_ratio"])} if not candidates.empty else None
    winner = leader if final_selection_ready else None
    rows.to_csv(root / "comparison.csv", index=False)
    dev.to_csv(root / "development_selection.csv", index=False)
    monthly = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()
    yearly = pd.concat(yearly_parts, ignore_index=True) if yearly_parts else pd.DataFrame()
    contributions = pd.concat(contribution_parts, ignore_index=True) if contribution_parts else pd.DataFrame()
    monthly.to_csv(root / "monthly.csv", index=False)
    yearly.to_csv(root / "yearly.csv", index=False)
    contributions.to_csv(root / "symbol_contribution.csv", index=False)
    pd.DataFrame(turnover_rows).to_csv(root / "turnover_categories.csv", index=False)
    (selection_frequency, weight_distribution, parameter_distribution, selection_turnover,
     fit_concentration, fit_concentration_summary) = _selection_tables(summaries)
    selection_frequency.to_csv(root / "selection_frequency.csv", index=False)
    selection_turnover.to_csv(root / "selection_turnover.csv", index=False)
    weight_distribution.to_csv(root / "weight_distribution.csv", index=False)
    parameter_distribution.to_csv(root / "parameter_distribution.csv", index=False)
    fit_concentration.to_csv(root / "fit_weight_concentration.csv", index=False)
    fit_concentration_summary.to_csv(root / "fit_weight_summary.csv", index=False)
    window_rows = _window_comparison(rows)
    window_rows.to_csv(root / "window_comparison.csv", index=False)
    paired_rows, paired_status = _paired_candidate_scores(
        root, summaries, root / "paired_score_cache.json", verifier_hash)
    paired_rows.to_csv(root / "paired_score_summary.csv", index=False)
    plots = root / "plots"
    _equity_plot(plots / "equity_development.svg", summaries, "development")
    _equity_plot(plots / "equity_development.png", summaries, "development")
    _equity_plot(plots / "equity_C.svg", summaries, "C")
    _equity_plot(plots / "equity_C.png", summaries, "C")

    # The cache stores successful single-account audits only. Every build checks
    # artifact hashes; paired cost and route-completeness checks always rerun.
    cache_path.write_text(json.dumps(new_cache, indent=2, allow_nan=False) + "\n")
    verification_summary = {"status": "not_run" if not checks else "passed" if all(item["status"] == "passed" for item in checks) else "failed",
                            "account_count": len(checks),
                            "passed_accounts": sum(item["status"] == "passed" for item in checks),
                            "failed_accounts": sum(item["status"] != "passed" for item in checks),
                            "cache_hits": sum(bool(item.get("cache_hit")) for item in checks),
                            "accounts": checks}
    (root / "account_verification.json").write_text(json.dumps(verification_summary, indent=2, allow_nan=False) + "\n")
    _write_morning_report(root, rows, verification_summary, window_rows, paired_rows,
                          paired_status, fit_concentration_summary, selection_rule_status,
                          experiment_progress, machine_state,
                          final_selection_ready, winner)
    result_status = "not_run" if not account_records else "partial"
    if verification_summary["failed_accounts"] or selection_rule_status["status"] == "failed":
        result_status = "failed"
    elif final_selection_ready and len(summaries) == expected_accounts \
            and sum(int(item["complete"]) for item in route_completeness.values()) == expected_routes:
        result_status = "complete"
    if paired_status["status"] == "failed":
        result_status = "failed"
    result = {"status": result_status,
              "account_count": len(summaries), "verification": verification_summary,
              "expected_account_count": expected_accounts,
              "experiment_progress": experiment_progress,
              "selection_rule_status": selection_rule_status,
              "final_selection_ready": final_selection_ready,
              "development_leader": leader, "development_winner": winner,
              "paired_score_status": paired_status,
              "qualified_development_routes": int(dev["qualified"].sum()) if not dev.empty else 0,
              "outputs": ["morning_report.md", "comparison.csv", "development_selection.csv",
                          "window_comparison.csv", "paired_score_summary.csv", "monthly.csv", "yearly.csv",
                          "symbol_contribution.csv", "turnover_categories.csv",
                          "selection_frequency.csv", "selection_turnover.csv",
                          "weight_distribution.csv", "fit_weight_concentration.csv", "fit_weight_summary.csv",
                          "parameter_distribution.csv",
                          "account_verification.json", "experiment_progress.json",
                          "plots/equity_development.svg", "plots/equity_development.png",
                          "plots/equity_C.svg", "plots/equity_C.png"]}
    (root / "report_build.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def refresh_morning_report(run_root: Path) -> dict[str, Any]:
    """Refresh the prose after runner-owned machine states change, without rereading accounts."""
    root = Path(run_root).resolve()
    result_path = root / "report_build.json"
    result = json.loads(result_path.read_text())
    rows = pd.read_csv(root / "comparison.csv")
    if "qualified" in rows:
        rows["qualified"] = rows["qualified"].map(lambda value: str(value).lower() == "true"
                                                   if isinstance(value, str) else bool(value))
    if "primary_selection_eligible" in rows:
        rows["primary_selection_eligible"] = rows["primary_selection_eligible"].map(
            lambda value: str(value).lower() == "true" if isinstance(value, str) else bool(value))
    if "route_complete" in rows:
        rows["route_complete"] = rows["route_complete"].map(
            lambda value: str(value).lower() == "true" if isinstance(value, str) else bool(value))
    verification = json.loads((root / "account_verification.json").read_text())
    progress = json.loads((root / "experiment_progress.json").read_text())
    machine_path = root / "machine_state.json"
    machine_state = json.loads(machine_path.read_text()) if machine_path.exists() else None
    windows = pd.read_csv(root / "window_comparison.csv")
    paired = pd.read_csv(root / "paired_score_summary.csv")
    concentration_summary = pd.read_csv(root / "fit_weight_summary.csv")
    paired_status = result.get("paired_score_status", {"status": "unavailable"})
    selection_rule_status = result.get("selection_rule_status", {"status": "missing"})
    core_complete = all(progress[name]["status"] == "complete" for name in ("E1", "E2", "E3"))
    e0_status = (machine_state or {}).get("statuses", {}).get("E0", {}).get("status")
    final_ready = (core_complete and e0_status in {"complete", "completed", "passed", "success"}
                   and selection_rule_status.get("status") == "passed")
    dev = rows.loc[(rows["stage"] == "development") & (rows["cost"] == "base")].copy() if not rows.empty else pd.DataFrame()
    candidates = dev.loc[dev["qualified"] & dev["primary_selection_eligible"]].sort_values(
        ["sharpe_ratio", "route"], ascending=[False, True]) if not dev.empty else pd.DataFrame()
    leader = {"route": str(candidates.iloc[0]["route"]),
              "development_net_sharpe": float(candidates.iloc[0]["sharpe_ratio"])} if not candidates.empty else None
    winner = leader if final_ready else None
    expected_accounts, expected_routes = _selected_scope(machine_state, [])
    if not machine_state:
        expected_accounts, expected_routes = int(result.get("expected_account_count", 44)), 11
    complete_routes = (dev.loc[dev["route_complete"].astype(bool), "route"].nunique()
                       if not dev.empty and "route_complete" in dev else 0)
    if verification.get("failed_accounts", 0) or paired_status.get("status") == "failed" \
            or selection_rule_status.get("status") == "failed":
        status = "failed"
    elif result.get("account_count", 0) == 0:
        status = "not_run"
    elif final_ready and result.get("account_count", 0) == expected_accounts and complete_routes == expected_routes:
        status = "complete"
    else:
        status = "partial"
    result.update({"status": status, "expected_account_count": expected_accounts,
                   "final_selection_ready": final_ready, "development_leader": leader,
                   "development_winner": winner})
    _write_morning_report(root, rows, verification, windows, paired, paired_status,
                          concentration_summary, selection_rule_status, progress,
                          machine_state, final_ready, winner)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
