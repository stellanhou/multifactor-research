import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_account import (
    ContinuousTargetPolicy,
    run_perpetual_account,
)
from crypto_quant.research.strategy_research.multifactor_horizon import combine_horizon_targets


def _policy(method="fixed", *, mean_gate=False, lookback=720, minimum=168):
    return {
        "method": method,
        "gross_exposure": 0.8,
        "max_asset_weight": 0.2,
        "lookback_hours": lookback,
        "min_history": minimum,
        "mean_gate": mean_gate,
        "covariance_shrinkage": 0.1,
    }


def _panels(rows=6):
    index = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="h")
    symbols = ["A", "B", "C"]
    targets = {
        horizon: pd.DataFrame(0.0, index=index, columns=symbols)
        for horizon in (1, 4, 24)
    }
    returns = pd.DataFrame(0.0, index=index, columns=[1, 4, 24])
    returns.iloc[0] = np.nan
    return index, targets, returns


def test_fixed_horizon_targets_net_before_caps_and_one_account_fill():
    index, targets, returns = _panels()
    targets[1]["A"] = 0.6
    targets[4]["A"] = -0.6
    targets[24]["A"] = 0.0
    for panel in targets.values():
        panel["B"] = 0.3
        panel["C"] = -0.1

    combined = combine_horizon_targets(targets, returns, policy=_policy())
    np.testing.assert_allclose(combined.budgets.iloc[0].to_numpy(), [1 / 3] * 3)
    assert combined.targets.loc[index[0], "A"] == 0.0
    assert combined.targets.loc[index[0], "B"] == 0.2
    assert combined.targets.loc[index[0], "C"] == -0.1
    assert combined.targets.abs().sum(axis=1).le(0.8 + 1e-12).all()
    assert combined.targets.abs().max(axis=1).le(0.2 + 1e-12).all()

    frames = {
        symbol: pd.DataFrame(
            {"open": 100.0, "close": 100.0, "mark_close": 100.0}, index=index
        )
        for symbol in ("A", "B", "C")
    }
    funding = pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"])
    result = run_perpetual_account(
        frames,
        combined.targets.iloc[:-1],
        funding,
        initial_capital=1_000.0,
        fee_bps=10.0,
        slippage_bps=5.0,
        start=index[1],
        end=index[-1] + pd.Timedelta(hours=1),
        margin_fraction=0.1,
        continuous_target_policy=ContinuousTargetPolicy(0.8, 0.2),
    )
    assert "A" not in set(result.fills.symbol)
    assert set(result.fills.symbol) == {"B", "C"}


def test_risk_budget_uses_prior_returns_and_positive_mean_gate():
    index, targets, returns = _panels()
    returns.loc[index[0], [1, 4, 24]] = [-0.02, 0.01, 0.02]
    returns.loc[index[1], [1, 4, 24]] = [-0.01, 0.03, 0.01]
    returns.loc[index[2], [1, 4, 24]] = [0.05, -0.02, -0.03]
    returns.loc[index[3], [1, 4, 24]] = [0.04, -0.01, 0.04]
    first = combine_horizon_targets(
        targets, returns, policy=_policy("risk_budget", mean_gate=True, lookback=4, minimum=2)
    )
    assert first.budgets.loc[index[0]].sum() == 0.0
    assert first.budgets.loc[index[1]].sum() == 0.0
    assert first.budgets.loc[index[2], 1] == 0.0
    assert first.budgets.loc[index[2], 4] > 0.0
    assert first.budgets.loc[index[2], 24] > 0.0
    assert first.budgets.loc[index[2]].sum() == 1.0
    assert first.fits.loc[first.fits.timestamp == index[2], "solver_error"].iloc[0] <= 1e-10
    assert pd.isna(first.fits.loc[first.fits.timestamp == index[0], "latest_return_timestamp"].iloc[0])
    assert first.fits.loc[first.fits.timestamp == index[2], "latest_return_timestamp"].iloc[0] == index[1]
    assert first.audit.loc[first.audit.timestamp == index[2], "latest_return_timestamp"].eq(index[1]).all()

    changed_returns = returns.copy()
    changed_returns.loc[index[2]:, [1, 4, 24]] = [0.4, -0.4, 0.2]
    changed_targets = {h: panel.copy() for h, panel in targets.items()}
    for panel in changed_targets.values():
        panel.loc[index[3]:, "A"] = 0.2
    changed = combine_horizon_targets(
        changed_targets, changed_returns,
        policy=_policy("risk_budget", mean_gate=True, lookback=4, minimum=2),
    )
    pd.testing.assert_frame_equal(first.budgets.loc[:index[2]], changed.budgets.loc[:index[2]])
    pd.testing.assert_frame_equal(first.targets.loc[:index[2]], changed.targets.loc[:index[2]])


def test_risk_budget_fails_fast_when_active_sleeve_has_zero_variance():
    index, targets, returns = _panels(rows=4)
    returns.loc[index[0], [1, 4, 24]] = [0.01, 0.01, -0.02]
    returns.loc[index[1], [1, 4, 24]] = [0.01, 0.02, -0.01]
    returns.loc[index[2], [1, 4, 24]] = [0.03, 0.01, 0.02]
    with pytest.raises(ValueError, match="positive definite"):
        combine_horizon_targets(
            targets, returns,
            policy=_policy("risk_budget", mean_gate=True, lookback=4, minimum=2),
        )


def _account_case(rows=4):
    start = pd.Timestamp("2024-01-01T01:00:00Z")
    index = pd.date_range(start - pd.Timedelta(hours=1), periods=rows + 1, freq="h")
    symbols = ["A", "B"]
    frames = {
        symbol: pd.DataFrame(
            {"open": 100.0, "close": 100.0, "mark_close": 100.0}, index=index
        )
        for symbol in symbols
    }
    targets = pd.DataFrame(0.0, index=index[:-1], columns=symbols)
    funding = pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"])
    return start, index, frames, targets, funding


def _run_account(start, index, frames, targets, funding, *, fee, buffer):
    return run_perpetual_account(
        frames,
        targets,
        funding,
        initial_capital=1_000.0,
        fee_bps=fee,
        slippage_bps=0.0,
        start=start,
        end=index[-1] + pd.Timedelta(hours=1),
        margin_fraction=0.1,
        continuous_target_policy=ContinuousTargetPolicy(0.8, 0.2, buffer),
    )


def test_continuous_quantity_band_holds_and_zero_target_exits_immediately():
    start, index, frames, targets, funding = _account_case()
    targets.loc[index[0], "A"] = 0.2
    targets.loc[index[1], "A"] = 0.21
    targets.loc[index[2], "A"] = 0.0
    result = _run_account(start, index, frames, targets, funding, fee=0.0, buffer=0.02)
    orders = result.orders.set_index(["signal_timestamp", "symbol"])
    held = orders.loc[(index[1], "A")]
    assert held.reason == "within_buffer"
    assert held.target_quantity == held.current_quantity
    exited = orders.loc[(index[2], "A")]
    assert exited.reason == "target_exit"
    assert exited.lower_weight == 0.0
    assert exited.upper_weight == 0.0
    assert exited.target_quantity == 0.0
    assert (result.positions.loc[result.positions.symbol == "A", "quantity"].iloc[-1]) == 0.0


def test_base_and_stress_accounts_use_their_own_equity_for_targets():
    start, index, frames, targets, funding = _account_case()
    targets.loc[index[:3], "A"] = 0.2
    base = _run_account(start, index, frames, targets, funding, fee=10.0, buffer=0.0)
    stress = _run_account(start, index, frames, targets, funding, fee=100.0, buffer=0.0)
    base_order = base.orders.loc[
        (base.orders.signal_timestamp == index[1]) & (base.orders.symbol == "A")
    ].iloc[0]
    stress_order = stress.orders.loc[
        (stress.orders.signal_timestamp == index[1]) & (stress.orders.symbol == "A")
    ].iloc[0]
    assert base_order.signal_equity != stress_order.signal_equity
    assert base_order.target_quantity != stress_order.target_quantity
    assert base.metrics["total_fees"] < stress.metrics["total_fees"]
