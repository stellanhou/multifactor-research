import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_account import (
    RebalancePolicy, generate_rank_targets, run_perpetual_account,
)
from crypto_quant.research.strategy_research import multifactor_rebalance as experiment


def _inputs(rows=5):
    start = pd.Timestamp("2024-01-01T01:00:00Z")
    index = pd.date_range(start - pd.Timedelta(hours=1), periods=rows + 1, freq="h")
    symbols = list("ABCDEFGHIJ")
    scores = pd.DataFrame([dict(zip(symbols, range(10, 0, -1)))] * rows, index=index[:-1], dtype=float)
    frames = {symbol: pd.DataFrame({"open": 100.0, "close": 100.0, "mark_close": 100.0}, index=index)
              for symbol in symbols}
    funding = pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"])
    return start, index, scores, frames, funding


def _run(inputs, policy, *, fee=0.0, slip=0.0):
    start, index, scores, frames, funding = inputs
    return run_perpetual_account(frames, None, funding, initial_capital=1000.0,
        fee_bps=fee, slippage_bps=slip, start=start, end=index[-1] + pd.Timedelta(hours=1),
        margin_fraction=0.1, scores=scores, rebalance_policy=policy)


def _policy(**kwargs):
    return RebalancePolicy(long_count=2, short_count=2, gross_exposure=0.8,
                           max_asset_weight=0.2, **kwargs)


def test_hourly_exact_policy_reproduces_legacy_account():
    inputs = _inputs()
    start, index, scores, frames, funding = inputs
    frames["A"].loc[index[1]:, ["close", "mark_close"]] = 102.0
    expected = run_perpetual_account(frames, generate_rank_targets(scores, start=start,
        long_count=2, short_count=2, gross_exposure=0.8, max_asset_weight=0.2, rebalance_hours=1),
        funding, initial_capital=1000.0, fee_bps=10.0, slippage_bps=5.0, start=start,
        end=index[-1] + pd.Timedelta(hours=1), margin_fraction=0.1)
    actual = _run(inputs, _policy(), fee=10.0, slip=5.0)
    pd.testing.assert_frame_equal(actual.ledger, expected.ledger, check_exact=False, atol=1e-10, rtol=1e-12)
    pd.testing.assert_frame_equal(actual.orders[expected.orders.columns], expected.orders,
                                  check_exact=False, atol=1e-12, rtol=1e-12)


def test_rank_buffer_retains_ranks_three_and_four_then_exits_before_filling():
    inputs = _inputs()
    _, index, scores, _, _ = inputs
    scores.loc[index[1], "C"] = 11
    scores.loc[index[2], ["C", "D"]] = [11, 12]
    scores.loc[index[3]:, ["C", "D", "E"]] = [11, 12, 13]
    result = _run(inputs, _policy(holding_rank=4, weight_buffer=0.02))
    orders = result.orders.set_index(["signal_timestamp", "symbol"])
    for timestamp, rank in ((index[1], 3), (index[2], 4)):
        row = orders.loc[(timestamp, "B")]
        assert row.long_rank == rank
        assert row.action == "hold"
        assert row.target_quantity == row.current_quantity == 2.0
        assert orders.loc[(timestamp, "C"), "target_quantity"] == 0.0
    assert orders.loc[(index[3], "B"), "reason"] == "rank_exit"
    assert orders.loc[(index[3], "B"), "target_quantity"] == 0
    assert orders.loc[(index[3], "E"), "reason"] == "entry"
    assert orders.loc[(index[3], "A"), "target_quantity"] == 2.0


def test_no_trade_preserves_quantity_with_prices_and_funding_changing():
    inputs = _inputs()
    start, index, _, frames, _ = inputs
    for symbol in ("A", "B", "I", "J"):
        frames[symbol].loc[index[1]:, ["open", "close", "mark_close"]] = 99.0
    inputs = (*inputs[:-1], pd.DataFrame([(start + pd.Timedelta(hours=1), "A", -0.001, 99.0)],
        columns=["timestamp", "symbol", "funding_rate", "mark_price"]))
    result = _run(inputs, _policy(weight_buffer=0.02))
    after_entry = result.orders.loc[result.orders.signal_timestamp > index[0]]
    assert after_entry.signed_order_quantity.eq(0).all()
    assert after_entry.action.eq("hold").all()
    assert len(result.fills) == 4
    assert result.positions.groupby("symbol").quantity.nunique().eq(1).all()
    assert result.ledger.iloc[1:].fees.eq(0).all()
    assert result.ledger.iloc[1:].slippage_cost.eq(0).all()
    assert result.metrics["total_funding"] == pytest.approx(0.198)
    assert result.ledger.iloc[-1].equity == pytest.approx(1000.198)


@pytest.mark.parametrize("side", ["A", "J"])
def test_position_band_resizes_to_nearest_boundary_and_caps_twenty_percent(side):
    inputs = _inputs(rows=3)
    _, index, _, frames, _ = inputs
    frames[side].loc[index[1], ["close", "mark_close"]] = 80.0
    frames[side].loc[index[2]:, ["close", "mark_close"]] = 150.0
    result = _run(inputs, _policy(weight_buffer=0.02))
    orders = result.orders.set_index(["signal_timestamp", "symbol"])
    underweight = orders.loc[(index[1], side)]
    assert abs(underweight.execution_weight) == pytest.approx(0.18)
    assert underweight.reason == "underweight"
    overweight = orders.loc[(index[2], side)]
    assert abs(overweight.execution_weight) == pytest.approx(0.2)
    assert overweight.reason == "asset_cap"
    assert result.orders.upper_weight.le(0.2).all()
    gross = result.orders.groupby("signal_timestamp").execution_weight.apply(lambda row: row.abs().sum())
    assert gross.le(0.8 + 1e-12).all()


def test_missing_scores_exit_and_ties_shortages_conflicts_are_deterministic():
    inputs = _inputs(rows=3)
    _, index, scores, _, _ = inputs
    scores.loc[index[1], "A"] = np.nan
    scores.loc[index[2]] = np.nan
    scores.loc[index[2], ["A", "B", "I"]] = 1.0
    result = _run(inputs, _policy(holding_rank=4, weight_buffer=0.02))
    orders = result.orders.set_index(["signal_timestamp", "symbol"])
    assert orders.loc[(index[1], "A"), "target_quantity"] == 0
    assert orders.loc[(index[1], "A"), "reason"] == "missing_signal"
    assert orders.loc[(index[2], "J"), "target_quantity"] == 0
    last = orders.loc[index[2]]
    assert last.loc["B", "target_quantity"] > 0
    assert last.loc["I", "target_quantity"] < 0
    assert last.loc["A", "target_quantity"] > 0
    assert last.target_quantity.gt(0).sum() == 2
    assert last.target_quantity.lt(0).sum() == 1


def test_rank_buffer_allows_direction_reversal_after_old_side_exits():
    inputs = _inputs(rows=3)
    _, index, scores, _, _ = inputs
    scores.loc[index[1]:, ["A", "J"]] = [-99, 99]
    result = _run(inputs, _policy(holding_rank=4, weight_buffer=0.02), fee=10, slip=5)
    reversals = result.orders.loc[(result.orders.signal_timestamp == index[1])
                                 & result.orders.symbol.isin(["A", "J"])]
    assert reversals.reason.eq("direction_reversal").all()
    assert reversals.trade_category.eq("direction_reversal").all()
    assert (reversals.current_quantity * reversals.target_quantity < 0).all()
    fills = result.fills.loc[(result.fills.timestamp == index[2]) & result.fills.symbol.isin(["A", "J"])]
    np.testing.assert_allclose(fills.quantity, reversals.signed_order_quantity.abs())


def test_future_changes_and_execution_open_funding_do_not_change_frozen_decisions():
    inputs = _inputs(rows=4)
    start, index, scores, frames, funding = inputs
    original = _run(inputs, _policy(holding_rank=4, weight_buffer=0.02), fee=10, slip=5)
    modified = {s: f.copy() for s, f in frames.items()}
    modified["A"].loc[index[3]:, ["open", "close", "mark_close"]] = 101
    future_scores = scores.copy()
    future_scores.loc[index[3], "A"] = -20
    changed = _run((start, index, future_scores, modified, funding), _policy(holding_rank=4, weight_buffer=0.02), fee=10, slip=5)
    pd.testing.assert_frame_equal(original.orders.loc[original.orders.signal_timestamp < index[3]],
                                  changed.orders.loc[changed.orders.signal_timestamp < index[3]])
    event = pd.DataFrame([(start + pd.Timedelta(hours=1), "A", 0.01, 100.0)],
                        columns=["timestamp", "symbol", "funding_rate", "mark_price"])
    funded = _run((start, index, scores, frames, event), _policy(holding_rank=4, weight_buffer=0.02), fee=10, slip=5)
    pd.testing.assert_frame_equal(original.orders.loc[original.orders.signal_timestamp <= index[1]],
                                  funded.orders.loc[funded.orders.signal_timestamp <= index[1]])
    assert funded.funding_events.iloc[0].quantity == 2.0


def test_cost_accounts_decide_independently_and_trade_audit_reconciles(tmp_path):
    inputs = _inputs()
    start, index, signals, frames, funding = inputs
    frames["A"].loc[index[1]:, ["close", "mark_close"]] = 90
    contract = SimpleNamespace(start=start, bounds=(start, index[-1] + pd.Timedelta(hours=1)),
        costs={"initial_capital": 1000.0, "fee_bps": 10.0, "slippage_bps": 5.0},
        portfolio={"long_count": 2, "short_count": 2, "gross_exposure": 0.8,
                   "max_asset_weight": 0.2, "rebalance_hours": 24, "margin_fraction": 0.1})
    request = {"holding_rank": 4, "weight_buffer": 0.02}
    base, result = experiment._run_arm(tmp_path / "base", signals, frames, funding, contract, experiment.ARMS[4], request, 1)
    stress, _ = experiment._run_arm(tmp_path / "stress", signals, frames, funding, contract, experiment.ARMS[4], request, 2)
    assert not base.orders.target_quantity.equals(stress.orders.target_quantity)
    audit = pd.read_csv(tmp_path / "base" / "decisions.csv")
    assert audit.fee.sum() == pytest.approx(base.metrics["total_fees"])
    assert audit.slippage_cost.sum() == pytest.approx(base.metrics["total_slippage_cost"])
    assert (audit.loc[audit.action == "hold", "new_quantity"] == audit.loc[audit.action == "hold", "current_quantity"]).all()
    assert result["metrics"]["gross_pnl_including_funding"] == pytest.approx(
        base.metrics["final_equity"] - 1000 + base.metrics["total_fees"] + base.metrics["total_slippage_cost"])
    assert (tmp_path / "base" / "monthly.csv").is_file()


def test_invalid_policy_and_incomplete_signal_grid_fail_at_boundary():
    with pytest.raises(ValueError, match="holding_rank"):
        _policy(holding_rank=1)
    inputs = _inputs()
    broken = (*inputs[:2], inputs[2].iloc[:-1], *inputs[3:])
    with pytest.raises(ValueError, match="signal grid"):
        _run(broken, _policy())


def test_runner_freezes_then_executes_sixty_accounts_and_reproduces_original(tmp_path, monkeypatch):
    from crypto_quant.research.strategy_research.multifactor_ablation import ARMS as SOURCE_ARMS
    from crypto_quant.research.strategy_research.multifactor_pool_research import PoolResearchContract
    from crypto_quant.research.strategy_research.multifactor_contracts import ResearchContract
    from crypto_quant.research.strategy_research.multifactor_workflow import _write_account_result
    from test_multifactor_pool_research import _icir_contract

    source = tmp_path / "source"
    source.mkdir()
    start, index, scores, frames, funding = _inputs(rows=3)
    split = start + pd.Timedelta(hours=3)
    pool_value = _icir_contract()
    pool_value.update(horizons=[1, 4, 24], development_start=start.isoformat(),
                      validation_start=split.isoformat(), validation_end=(split + pd.Timedelta(hours=3)).isoformat())
    pool_value["portfolio"].update(long_count=2, short_count=2, gross_exposure=0.8, max_asset_weight=0.2)
    pool = PoolResearchContract.from_dict(pool_value)
    (source / "pool_contract.json").write_text(json.dumps(pool_value))
    (source / "plan.json").write_text(json.dumps({"arms": SOURCE_ARMS, "icir": pool.combination, "horizons": pool.horizons}))
    (source / "dataset_manifest.json").write_text(json.dumps({"historical_causality_certified": False}))
    (source / "cards_manifest.json").write_text("{}")
    for stage in experiment.STAGES:
        offset = pd.Timedelta(hours=0 if stage == "development" else 3)
        shifted_scores = scores.copy()
        shifted_scores.index += offset
        shifted_frames = {s: f.copy() for s, f in frames.items()}
        for frame in shifted_frames.values():
            frame.index += offset
        market = source / stage / "inputs" / "market"
        market.mkdir(parents=True)
        for symbol, frame in shifted_frames.items():
            frame.to_csv(market / f"{symbol}.csv")
        funding.to_csv(market.parent / "funding.csv", index=False)
        for horizon in pool.horizons:
            directory = source / stage / f"h{horizon}"
            directory.mkdir()
            contract = ResearchContract.from_dict(pool.stage_dict(horizon, ["a.json", "b.json"],
                                                                  "universe.csv", "manifest.json", stage))
            (directory / "contract.json").write_text(json.dumps(contract.as_dict()))
            for name, multiplier in (("B", 1), ("B_stress", 2)):
                account_dir = directory / "experiments" / name
                account_dir.mkdir(parents=True)
                legacy = run_perpetual_account(shifted_frames, generate_rank_targets(shifted_scores,
                    start=contract.start, long_count=2, short_count=2, gross_exposure=0.8,
                    max_asset_weight=0.2, rebalance_hours=horizon), funding,
                    **experiment._account_kwargs(contract, contract.costs["fee_bps"] * multiplier,
                                                  contract.costs["slippage_bps"] * multiplier))
                _write_account_result(account_dir, legacy)
                shifted_scores.to_csv(account_dir / "signals.csv")
    request = {"schema_version": 1, "run_id": "rebalance-test", "source_run": "source",
               "holding_rank": 4, "weight_buffer": 0.02}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(request))
    calls = []
    original = experiment._run_arm

    def checked_run(directory, *args):
        frozen = json.loads((tmp_path / "runs" / "rebalance-test" / "plan.json").read_text())
        assert frozen["frozen_before_outcome_access"] is True
        assert frozen["account_runs"] == 60
        calls.append(directory)
        return original(directory, *args)

    monkeypatch.setattr(experiment, "_capture_sources", lambda root: None)
    monkeypatch.setattr(experiment, "_run_arm", checked_run)
    result = experiment.run_rebalance(contract_path, tmp_path / "runs")
    assert len(calls) == 60
    assert [(trial["horizon_hours"], trial["stage"], trial["name"]) for trial in result["trials"]] == [
        (horizon, stage, arm["name"]) for horizon in (24, 4, 1) for stage in experiment.STAGES for arm in experiment.ARMS]
    assert result["status"] == "rebalance_complete"
    assert not result["paper_started"]
    assert len(pd.read_csv(Path(result["root"]) / "comparison.csv")) == 30
    assert (Path(result["root"]) / "report.md").is_file()
    with pytest.raises(ValueError, match="refusing overwrite"):
        experiment.run_rebalance(contract_path, tmp_path / "runs")
