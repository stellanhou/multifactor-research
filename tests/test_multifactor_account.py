import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_account import (
    generate_rank_targets,
    run_perpetual_account,
)


def _market(index, *, opens, closes, marks=None):
    if marks is None:
        marks = closes
    return pd.DataFrame(
        {"open": opens, "close": closes, "mark_close": marks}, index=index
    )


def _account_inputs():
    start = pd.Timestamp("2024-01-01T01:00:00Z")
    index = pd.date_range(start - pd.Timedelta(hours=1), periods=4, freq="h")
    frames = {
        "AUSDT": _market(
            index,
            opens=[100, 100, 100, 100],
            closes=[100, 100, 112, 114],
            marks=[100, 100, 110, 120],
        ),
        "BUSDT": _market(
            index,
            opens=[100, 100, 100, 100],
            closes=[100, 100, 95, 90],
            marks=[100, 100, 90, 80],
        ),
    }
    targets = pd.DataFrame(
        [{"AUSDT": 0.25, "BUSDT": -0.25}], index=index[:1]
    )
    funding = pd.DataFrame(
        columns=["timestamp", "symbol", "funding_rate", "mark_price"]
    )
    return start, index, frames, targets, funding


def _run(start, index, frames, targets, funding, *, end=None):
    return run_perpetual_account(
        frames,
        targets,
        funding,
        initial_capital=1_000.0,
        fee_bps=10.0,
        slippage_bps=10.0,
        start=start,
        end=end or start + pd.Timedelta(hours=3),
        margin_fraction=0.05,
    )


def test_announced_inactive_asset_stays_flat_without_synthetic_prices():
    start, index, frames, targets, funding = _account_inputs()
    frames["AUSDT"]["inactive"] = [False, False, False, True]
    frames["AUSDT"].loc[index[-1], ["open", "close", "mark_close"]] = np.nan
    targets = pd.DataFrame(
        {"AUSDT": [0.25, 0.0, 0.0], "BUSDT": [-0.25, -0.25, -0.25]},
        index=index[:3],
    )
    result = _run(start, index, frames, targets, funding)
    final = result.positions.loc[(result.positions.symbol == "AUSDT")].iloc[-1]
    assert final.quantity == final.gross_notional == final.signed_notional == 0.0
    assert pd.isna(final.mark_price)
    assert np.isfinite(result.ledger.equity).all()
    assert len(result.fills.loc[result.fills.symbol == "AUSDT"]) == 2


@pytest.mark.parametrize("held", [False, True])
def test_inactive_asset_rejects_held_position_or_new_target(held):
    start, index, frames, targets, funding = _account_inputs()
    frames["AUSDT"]["inactive"] = [False, False, False, True]
    frames["AUSDT"].loc[index[-1], ["open", "close", "mark_close"]] = np.nan
    if not held:
        targets = pd.DataFrame({"AUSDT": [0.0, 0.25], "BUSDT": [0.0, 0.0]},
                               index=index[[0, 2]])
    with pytest.raises(ValueError, match="inactive hour"):
        _run(start, index, frames, targets, funding)


def test_inactive_flag_does_not_permit_missing_active_prices():
    start, index, frames, targets, funding = _account_inputs()
    frames["AUSDT"]["inactive"] = False
    frames["AUSDT"].loc[index[-1], "open"] = np.nan
    with pytest.raises(ValueError, match="finite and positive"):
        _run(start, index, frames, targets, funding)


@pytest.mark.parametrize("mode", ["rank_buffer", "continuous_buffer"])
def test_buffered_accounts_clear_position_before_inactive_prices(mode):
    from crypto_quant.research.strategy_research.multifactor_account import (
        ContinuousTargetPolicy, RebalancePolicy,
    )
    start, index, frames, _, funding = _account_inputs()
    frames["AUSDT"]["inactive"] = [False, False, False, True]
    frames["AUSDT"].loc[index[-1], ["open", "close", "mark_close"]] = np.nan
    if mode == "rank_buffer":
        targets = None
        kwargs = dict(
            scores=pd.DataFrame({"AUSDT": [2.0, np.nan, np.nan],
                                 "BUSDT": [1.0, 1.0, 1.0]}, index=index[:3]),
            rebalance_policy=RebalancePolicy(long_count=1, short_count=1,
                gross_exposure=0.5, max_asset_weight=0.25, holding_rank=2, weight_buffer=0.02),
        )
    else:
        targets = pd.DataFrame({"AUSDT": [0.25, 0.0, 0.0], "BUSDT": [-0.25]*3}, index=index[:3])
        kwargs = dict(continuous_target_policy=ContinuousTargetPolicy(
            gross_exposure=0.5, max_asset_weight=0.25, weight_buffer=0.02))
    account = run_perpetual_account(frames, targets, funding, initial_capital=1000.0,
        fee_bps=10.0, slippage_bps=5.0, start=start, end=start+pd.Timedelta(hours=3),
        margin_fraction=0.05, **kwargs)
    last = account.orders.loc[account.orders.symbol == "AUSDT"].iloc[-1]
    assert last.target_quantity == last.current_quantity == 0.0
    assert last.actual_weight == last.execution_weight == 0.0
    assert np.isfinite(account.ledger.equity).all()


@pytest.mark.parametrize("mode", ["scheduled", "rank_buffer", "continuous_buffer"])
def test_asset_inactive_for_entire_validation_account_requires_no_prices(mode):
    from crypto_quant.research.strategy_research.multifactor_account import (
        ContinuousTargetPolicy, RebalancePolicy,
    )
    start, index, frames, _, funding = _account_inputs()
    frames["AUSDT"][["open", "close", "mark_close"]] = np.nan
    frames["AUSDT"]["inactive"] = True
    targets = pd.DataFrame({"AUSDT": [0.0]*3, "BUSDT": [0.25]*3}, index=index[:3])
    kwargs = {}
    if mode == "rank_buffer":
        targets = None
        kwargs = dict(scores=pd.DataFrame({"AUSDT": [np.nan]*3, "BUSDT": [1.0]*3}, index=index[:3]),
            rebalance_policy=RebalancePolicy(long_count=1, short_count=1,
                gross_exposure=0.5, max_asset_weight=0.25, holding_rank=2, weight_buffer=0.02))
    elif mode == "continuous_buffer":
        kwargs = dict(continuous_target_policy=ContinuousTargetPolicy(
            gross_exposure=0.5, max_asset_weight=0.25, weight_buffer=0.02))
    account = run_perpetual_account(frames, targets, funding, initial_capital=1000.0,
        fee_bps=10.0, slippage_bps=5.0, start=start, end=start+pd.Timedelta(hours=3),
        margin_fraction=0.05, **kwargs)
    inactive = account.positions.loc[account.positions.symbol == "AUSDT"]
    assert inactive.quantity.eq(0.0).all() and inactive.mark_price.isna().all()
    assert np.isfinite(account.ledger.equity).all()


def test_rank_targets_ties_shortages_cash_and_fixed_rebalance_cycle():
    start = pd.Timestamp("2024-01-01T01:00:00Z")
    index = pd.date_range(start - pd.Timedelta(hours=1), periods=6, freq="h")
    scores = pd.DataFrame(
        {
            "A": [1.0, 1.0, 0.0, 1.0, 5.0, 5.0],
            "B": [1.0, 1.0, 2.0, 0.0, np.nan, np.nan],
            "C": [1.0, 1.0, 1.0, 2.0, np.nan, np.nan],
            "D": [1.0, 1.0, -1.0, 3.0, np.nan, np.nan],
        },
        index=index,
    )

    targets = generate_rank_targets(
        scores,
        long_count=1,
        short_count=1,
        gross_exposure=0.8,
        max_asset_weight=0.5,
        rebalance_hours=2,
        start=start,
    )

    assert list(targets.index) == [index[0], index[2], index[4]]
    assert targets.loc[index[0]].to_dict() == {
        "A": 0.4,
        "B": -0.4,
        "C": 0.0,
        "D": 0.0,
    }
    assert targets.loc[index[2]].to_dict() == {
        "A": 0.0,
        "B": 0.4,
        "C": 0.0,
        "D": -0.4,
    }
    # A single eligible long fills one fixed seat; the unfilled short seat stays cash.
    assert targets.loc[index[4], "A"] == 0.4
    assert targets.loc[index[4], ["B", "C", "D"]].eq(0.0).all()


def test_account_hand_calculation_and_unrebalanced_positions_are_auditable():
    start, _index, frames, targets, funding = _account_inputs()
    result = _run(start, _index, frames, targets, funding)

    orders = result.orders.set_index("symbol")
    assert orders.loc["AUSDT", "target_quantity"] == 2.5
    assert orders.loc["BUSDT", "target_quantity"] == -2.5
    assert orders.loc["AUSDT", "signal_close"] == 100.0
    assert orders.loc["AUSDT", "signal_equity"] == 1_000.0

    # Entry fees are 0.50 total. Slippage is embedded in the fill prices and
    # therefore in unrealized PnL, not deducted a second time from cash.
    assert np.isclose(result.fills["fee"].sum(), 0.5)
    assert np.isclose(result.fills["slippage_cost"].sum(), 0.5)
    assert np.isclose(result.ledger.iloc[0]["cash"], 999.5)
    assert np.isclose(result.ledger.iloc[0]["equity"], 999.0)
    assert np.isclose(result.ledger.iloc[1]["unrealized_pnl"], 49.5)
    assert np.isclose(result.ledger.iloc[1]["equity"], 1_049.0)
    assert np.isclose(result.metrics["net_return"], 0.099)
    assert result.metrics["max_drawdown"] >= 0.0

    # No target row after the initial rebalance means the signed positions carry
    # through every mark and remain open at the end of the requested window.
    for symbol, quantity in (("AUSDT", 2.5), ("BUSDT", -2.5)):
        path = result.positions.loc[result.positions["symbol"] == symbol, "quantity"]
        np.testing.assert_allclose(path.to_numpy(), quantity)
    assert len(result.orders) == 2


def test_order_quantity_uses_signal_close_and_equity_not_future_open():
    start, index, frames, targets, funding = _account_inputs()
    baseline = _run(start, index, frames, targets, funding)
    changed = {symbol: frame.copy() for symbol, frame in frames.items()}
    changed["AUSDT"].loc[index[1], "open"] = 250.0
    changed["BUSDT"].loc[index[1], "open"] = 40.0
    repriced = _run(start, index, changed, targets, funding)

    np.testing.assert_allclose(
        baseline.orders["target_quantity"], repriced.orders["target_quantity"]
    )
    assert not np.allclose(baseline.fills["fill_price"], repriced.fills["fill_price"])


def test_funding_signs_use_signed_quantity_and_exact_settlement_time():
    start, index, frames, targets, _funding = _account_inputs()
    targets = pd.DataFrame(
        [{"AUSDT": 0.1, "BUSDT": -0.1}], index=index[:1]
    )
    # First event occurs exactly at the entry open and belongs to the old flat
    # position. The later events charge longs at positive rates and shorts at
    # negative rates, with opposite-side credits shown separately.
    funding = pd.DataFrame(
        [
            (index[1], "AUSDT", 0.01, 100.0),
            (index[2], "AUSDT", 0.01, 100.0),
            (index[2], "BUSDT", 0.01, 100.0),
            (index[3], "AUSDT", -0.01, 100.0),
            (index[3], "BUSDT", -0.01, 100.0),
        ],
        columns=["timestamp", "symbol", "funding_rate", "mark_price"],
    )
    result = run_perpetual_account(
        frames,
        targets,
        funding,
        initial_capital=1_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        start=start,
        end=start + pd.Timedelta(hours=3),
        margin_fraction=0.05,
    )

    payments = result.funding_events.set_index(["timestamp", "symbol"])
    assert payments.loc[(index[1], "AUSDT"), "quantity"] == 0.0
    assert payments.loc[(index[1], "AUSDT"), "cashflow"] == 0.0
    assert np.isclose(payments.loc[(index[2], "AUSDT"), "cashflow"], -1.0)
    assert np.isclose(payments.loc[(index[2], "BUSDT"), "cashflow"], 1.0)
    assert np.isclose(payments.loc[(index[3], "AUSDT"), "cashflow"], 1.0)
    assert np.isclose(payments.loc[(index[3], "BUSDT"), "cashflow"], -1.0)
    assert np.isclose(result.ledger["funding_cashflow"].sum(), 0.0)


def test_account_reversal_realizes_old_side_then_opens_and_closes_short():
    start = pd.Timestamp("2024-01-01T01:00:00Z")
    index = pd.date_range(start - pd.Timedelta(hours=1), periods=4, freq="h")
    frames = {
        "AUSDT": _market(
            index,
            opens=[100, 100, 120, 115],
            closes=[100, 110, 120, 115],
            marks=[100, 110, 120, 115],
        )
    }
    targets = pd.DataFrame(
        {"AUSDT": [0.25, -0.25, 0.0]}, index=index[:3]
    )
    funding = pd.DataFrame(
        columns=["timestamp", "symbol", "funding_rate", "mark_price"]
    )

    result = run_perpetual_account(
        frames,
        targets,
        funding,
        initial_capital=1_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        start=start,
        end=start + pd.Timedelta(hours=3),
        margin_fraction=0.05,
    )

    short_quantity = -0.25 * 1_025.0 / 110.0
    assert np.isclose(result.ledger.iloc[0]["equity"], 1_025.0)
    assert np.isclose(result.orders.loc[1, "signal_equity"], 1_025.0)
    assert np.isclose(result.orders.loc[1, "signal_close"], 110.0)
    assert np.isclose(result.orders.loc[1, "target_quantity"], short_quantity)
    flip = result.fills.loc[1]
    assert np.isclose(flip["realized_pnl"], 50.0)
    assert np.isclose(flip["new_quantity"], short_quantity)
    assert np.isclose(flip["average_entry_price"], 120.0)
    close = result.fills.loc[2]
    assert np.isclose(close["realized_pnl"], abs(short_quantity) * 5.0)
    assert close["new_quantity"] == 0.0
    assert np.isclose(result.ledger.iloc[-1]["cash"], 1_050.0 + abs(short_quantity) * 5.0)
    assert result.positions.iloc[-1]["quantity"] == 0.0


def test_each_intrahour_funding_time_checks_margin_before_later_credit():
    start = pd.Timestamp("2024-01-01T01:00:00Z")
    index = pd.date_range(start - pd.Timedelta(hours=1), periods=2, freq="h")
    frames = {
        "AUSDT": _market(
            index, opens=[100, 100], closes=[100, 100], marks=[100, 100]
        )
    }
    targets = pd.DataFrame({"AUSDT": [1.0]}, index=index[:1])
    funding = pd.DataFrame(
        [
            (start + pd.Timedelta(minutes=15), "AUSDT", 0.2, 100.0),
            (start + pd.Timedelta(minutes=30), "AUSDT", -0.5, 100.0),
        ],
        columns=["timestamp", "symbol", "funding_rate", "mark_price"],
    )

    with pytest.raises(ValueError, match="sampled margin guard breached at 2024-01-01 01:15:00"):
        run_perpetual_account(
            frames,
            targets,
            funding,
            initial_capital=1_000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            start=start,
            end=start + pd.Timedelta(hours=1),
            margin_fraction=0.9,
        )


def test_target_on_end_minus_one_hour_is_rejected_as_unexecutable():
    start, index, frames, targets, funding = _account_inputs()
    end = start + pd.Timedelta(hours=2)
    targets = pd.concat(
        [targets, pd.DataFrame([{"AUSDT": 0.0, "BUSDT": 0.0}], index=index[2:3])]
    )

    with pytest.raises(ValueError, match="strictly before the end-exclusive boundary"):
        run_perpetual_account(
            frames,
            targets,
            funding,
            initial_capital=1_000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            start=start,
            end=end,
            margin_fraction=0.05,
        )
