import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_account import run_perpetual_account
from crypto_quant.research.strategy_research.multifactor_contracts import ResearchContract
from crypto_quant.research.strategy_research.multifactor_net_sharpe import (
    ROUTE,
    _last_r0_event,
    _market_arrays,
    _r0_targets,
    _ridge_training_stats,
    _simulate_r0_batch,
    generate_net_sharpe_scores,
)
from crypto_quant.research.strategy_research.multifactor_selection import generate_selection_scores


SYMBOLS = [f"S{i:02d}" for i in range(8)]


def _contract(times, horizon=1):
    end = times[-1] + pd.Timedelta(hours=1)
    return ResearchContract(
        schema_version=2,
        run_id="net-sharpe-test",
        purpose="research",
        stage="development",
        start=times[0].isoformat(),
        end=end.isoformat(),
        warmup_hours=24,
        horizon_hours=horizon,
        cards=["card-a", "card-b"],
        universe="universe.csv",
        prior_data_use="synthetic test data",
        data_processing="synthetic test data",
        costs={
            "initial_capital": 10_000.0,
            "fee_bps": 10.0,
            "slippage_bps": 5.0,
            "stress_multiplier": 2.0,
        },
        portfolio={
            "long_count": 2,
            "short_count": 2,
            "gross_exposure": 0.8,
            "max_asset_weight": 0.2,
            "rebalance_hours": horizon,
            "margin_fraction": 0.1,
        },
        dataset_manifest="manifest.json",
    )


def _market(periods=112, *, seed=204):
    times = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC",
                          name="timestamp")
    frame_index = pd.date_range(times[0] - pd.Timedelta(hours=1), periods=periods + 1,
                                freq="h", tz="UTC", name="timestamp")
    rng = np.random.default_rng(seed)
    prices = {}
    frames = {}
    for column, symbol in enumerate(SYMBOLS):
        increments = rng.normal(0.0, 0.0015, size=periods + 1)
        opening = 100.0 * np.exp(np.cumsum(increments)) * (1 + column * 0.03)
        close = opening * np.exp(rng.normal(0.0, 0.001, size=periods + 1))
        mark = close * np.exp(rng.normal(0.0, 0.0002, size=periods + 1))
        frames[symbol] = pd.DataFrame(
            {"open": opening, "close": close, "mark_close": mark}, index=frame_index,
        )
        prices[symbol] = opening[1:]
    return times, frames, prices, rng


def _factor_panel(times, prices, *, seed=911, factor_count=2):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(len(times), len(SYMBOLS), factor_count))
    raw -= raw.mean(axis=1, keepdims=True)
    raw /= raw.std(axis=1, keepdims=True)
    index = pd.MultiIndex.from_product([times, SYMBOLS], names=["timestamp", "symbol"])
    names = [f"factor_{i}" for i in range(factor_count)]
    values = pd.DataFrame(raw.reshape(-1, factor_count), index=index, columns=names)
    opens = pd.Series(np.column_stack([prices[symbol] for symbol in SYMBOLS]).reshape(-1),
                      index=index, name="perp_open")
    return values, opens, raw


def _policy(**updates):
    policy = {
        "fit_window_hours": 80,
        "min_train_periods": 10,
        "refit_every_hours": 24,
        "min_cross_section_symbols": 4,
        "icir_window_hours": 8,
        "icir_min_periods": 3,
        "correlation_window_hours": 6,
        "cluster_abs_correlation_threshold": 0.8,
        "inner_validation_hours": 8,
        "min_inner_validation_periods": 4,
        "alpha_grid": [1e-4, 0.01],
        "l1_ratio": 0.5,
        "pool_capacity": 2,
        "pool_search_budget": 8,
        "min_objective_improvement": 1e-6,
    }
    policy.update(updates)
    return policy


def _funding_for_validation(times, frames, *, start_pos, end_pos):
    events = []
    points = [
        (times[start_pos + 1], 0.0001),
        (times[start_pos + 6] + pd.Timedelta(minutes=20), -0.0002),
        (times[start_pos + 12], 0.00015),
    ]
    for timestamp, rate in points:
        bar = timestamp.floor("h")
        for symbol in SYMBOLS:
            mark = frames[symbol].loc[bar, "mark_close"]
            events.append({"timestamp": timestamp, "symbol": symbol,
                           "funding_rate": rate, "mark_price": mark})
    return pd.DataFrame(events)


def test_batched_account_matches_full_engine_hourly_equity_costs_and_funding():
    times, frames, prices, _rng = _market(periods=112)
    values, _opens, factor_cube = _factor_panel(times, prices)
    contract = _contract(times)
    funding = _funding_for_validation(times, frames, start_pos=20, end_pos=80)
    fit_time = times[80]
    source_times = pd.date_range(times[20], fit_time - pd.Timedelta(hours=2),
                                 freq="h", tz="UTC", name="timestamp")
    source_positions = times.get_indexer(source_times)
    validation_values = factor_cube[source_positions]
    complete = np.isfinite(validation_values).all(axis=2)
    coefficients = np.array([[0.9, -0.25]])
    market, validated_funding = _market_arrays(frames, SYMBOLS, funding)
    policy = _policy(min_cross_section_symbols=4)
    fast = _simulate_r0_batch(
        coefficients, validation_values, complete, source_times, fit_time,
        market, validated_funding, contract, policy, include_equity_path=True,
    )[0]

    start = source_times[0] + pd.Timedelta(hours=1)
    end = fit_time
    execution_times = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h", tz="UTC")
    score_cube = np.einsum("tsf,cf->cts", validation_values, coefficients)
    events = _r0_targets(
        score_cube, complete, source_times, execution_times,
        minimum_symbols=4,
        long_count=2,
        short_count=2,
        gross_exposure=0.8,
        max_asset_weight=0.2,
        rebalance_hours=1,
        anchor=contract.bounds[0] - pd.Timedelta(hours=1),
    )
    target_rows = [events[position][0] for position in range(len(source_times))]
    targets = pd.DataFrame(
        target_rows,
        index=source_times,
        columns=SYMBOLS,
    )
    engine = run_perpetual_account(
        frames, targets, funding,
        initial_capital=contract.costs["initial_capital"],
        fee_bps=contract.costs["fee_bps"],
        slippage_bps=contract.costs["slippage_bps"],
        start=start,
        end=end,
        margin_fraction=contract.portfolio["margin_fraction"],
    )

    np.testing.assert_allclose(fast["hourly_equity"], engine.ledger["equity"].to_numpy(),
                               rtol=2e-11, atol=2e-8)
    np.testing.assert_allclose(fast["hourly_returns"], engine.ledger["return"].to_numpy(),
                               rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(fast["hourly_fees"], engine.ledger["fees"].to_numpy(),
                               rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(fast["hourly_slippage_cost"],
                               engine.ledger["slippage_cost"].to_numpy(),
                               rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(fast["hourly_funding_cashflow"],
                               engine.ledger["funding_cashflow"].to_numpy(),
                               rtol=2e-11, atol=2e-10)
    assert fast["validation_net_return"] == pytest.approx(engine.metrics["net_return"], abs=2e-12)
    assert fast["validation_net_sharpe"] == pytest.approx(engine.metrics["sharpe_ratio"], abs=1e-10)
    assert fast["valid_objective"]
    assert fast["validation_net_return"] < 0.0
    assert fast["total_fees"] == pytest.approx(engine.metrics["total_fees"], abs=2e-10)
    assert fast["total_slippage_cost"] == pytest.approx(
        engine.metrics["total_slippage_cost"], abs=2e-10,
    )
    assert fast["total_funding"] == pytest.approx(engine.metrics["total_funding"], abs=2e-10)

    fast_events = pd.DataFrame(fast["funding_event_cashflows"])
    expected_events = engine.funding_events[["timestamp", "symbol", "cashflow"]]
    pd.testing.assert_frame_equal(
        fast_events.reset_index(drop=True), expected_events.reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=2e-11, atol=2e-10,
    )


def test_net_sharpe_scores_preserve_shared_mask_and_audit_cash_gate():
    times, frames, prices, _rng = _market(periods=64, seed=41)
    values, opens, _factor_cube = _factor_panel(times, prices, seed=17, factor_count=2)
    funding = pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"])
    contract = _contract(times)
    policy = _policy(
        fit_window_hours=40,
        min_train_periods=8,
        refit_every_hours=24,
        correlation_window_hours=8,
        inner_validation_hours=8,
        min_inner_validation_periods=4,
    )

    result = generate_net_sharpe_scores(
        values, opens, horizon_hours=1, policy=policy,
        frames=frames, funding=funding, contract=contract,
    )

    assert tuple(result.scores) == (ROUTE,)
    assert result.scores[ROUTE].index.equals(values.index)
    assert result.scores[ROUTE].name == "score"
    assert result.shared_model_ready.index.equals(times)
    r2 = generate_selection_scores(values, opens, horizon_hours=1, policy=policy,
                                   methods=["stepwise"])
    pd.testing.assert_series_equal(result.shared_model_ready, r2.shared_model_ready)
    fits = [fit for fit in result.fits if fit["route"] == ROUTE]
    assert fits
    fit = fits[0]
    assert fit["objective_name"] == "validation_net_sharpe"
    assert fit["validation_source_window_bars"] == policy["inner_validation_hours"]
    assert fit["validation_account_end_exclusive"] == fit["timestamp"]
    assert fit["validation_last_bar_open"] == fit["timestamp"] - pd.Timedelta(hours=1)
    assert fit["validation_last_funding_timestamp"] is None
    assert fit["validation_terminal_carry_hours"] >= 1
    for round_audit in fit["selection_rounds"]:
        for proposal in round_audit["proposals"]:
            for trial in proposal["alpha_trials"]:
                assert "validation_net_return" in trial
                assert "validation_net_sharpe" in trial
                if trial["valid_objective"]:
                    assert trial["validation_objective"] == trial["validation_net_sharpe"]
                else:
                    assert trial["validation_objective"] is None
    if fit["validation_net_return"] is not None:
        assert fit["final_net_return_admitted"] is (fit["validation_net_return"] > 0)
    if fit["status"] == "rejected_final_net_return_nonpositive":
        assert not result.scores[ROUTE].notna().any()


def test_cash_account_has_zero_objective_and_cannot_be_admitted():
    times, frames, prices, _rng = _market(periods=80, seed=81)
    _values, _opens, factor_cube = _factor_panel(times, prices, seed=5)
    contract = _contract(times)
    fit_time = times[60]
    source_times = pd.date_range(times[20], fit_time - pd.Timedelta(hours=2),
                                 freq="h", tz="UTC", name="timestamp")
    source_positions = times.get_indexer(source_times)
    validation_values = factor_cube[source_positions]
    unavailable = np.zeros(validation_values.shape[:2], dtype=bool)
    funding = pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"])
    market, validated_funding = _market_arrays(frames, SYMBOLS, funding)

    result = _simulate_r0_batch(
        np.array([[0.5, -0.2]]), validation_values, unavailable, source_times,
        fit_time, market, validated_funding, contract, _policy(), include_equity_path=True,
    )[0]

    assert result["validation_net_return"] == 0.0
    assert result["validation_net_sharpe"] == 0.0
    assert result["traded_bars"] == 0
    assert result["valid_objective"] is True  # a cash Sharpe of zero cannot beat the empty baseline
    assert result["hourly_equity"] == pytest.approx(np.full(len(source_times), 10_000.0))


def test_sampled_margin_breach_invalidates_the_candidate_with_timestamp():
    times, frames, prices, _rng = _market(periods=80, seed=33)
    _values, _opens, factor_cube = _factor_panel(times, prices, seed=7)
    contract = _contract(times)
    fit_time = times[60]
    source_times = pd.date_range(times[20], fit_time - pd.Timedelta(hours=2),
                                 freq="h", tz="UTC", name="timestamp")
    source_positions = times.get_indexer(source_times)
    validation_values = factor_cube[source_positions].copy()
    validation_values[0, 0, 0] = -1e6
    complete = np.isfinite(validation_values).all(axis=2)
    funding = pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"])
    execution_time = source_times[0] + pd.Timedelta(hours=1)
    frames[SYMBOLS[0]].loc[execution_time, "mark_close"] = (
        frames[SYMBOLS[0]].loc[execution_time, "open"] * 1_000.0
    )
    market, validated_funding = _market_arrays(frames, SYMBOLS, funding)
    execution_position = market.position[source_times[0] + pd.Timedelta(hours=1)]
    symbol_position = market.symbols.index(SYMBOLS[0])
    assert market.marks[execution_position, symbol_position] == frames[SYMBOLS[0]].loc[
        execution_time, "mark_close"
    ]

    result = _simulate_r0_batch(
        np.array([[1.0, 0.0]]), validation_values, complete, source_times,
        fit_time, market, validated_funding, contract, _policy(),
    )[0]

    assert result["status"] == "non_positive_account_equity_at_hourly_close"
    assert result["failure_timestamp"] == source_times[0] + pd.Timedelta(hours=2)
    assert result["validation_net_sharpe"] is None
    assert not result["valid_objective"]

    score_row = validation_values[0, :, 0]
    long_names = sorted(SYMBOLS, key=lambda symbol: (-score_row[SYMBOLS.index(symbol)], symbol))[:2]
    remaining = [symbol for symbol in SYMBOLS if symbol not in long_names]
    short_names = sorted(remaining, key=lambda symbol: (score_row[SYMBOLS.index(symbol)], symbol))[:2]
    target = pd.DataFrame(0.0, index=[source_times[0]], columns=SYMBOLS)
    target.loc[source_times[0], long_names] = 0.2
    target.loc[source_times[0], short_names] = -0.2
    with pytest.raises(ValueError, match="non-positive account equity|sampled margin guard breached") as engine_error:
        run_perpetual_account(
            frames, target, funding,
            initial_capital=contract.costs["initial_capital"],
            fee_bps=contract.costs["fee_bps"],
            slippage_bps=contract.costs["slippage_bps"],
            start=source_times[0] + pd.Timedelta(hours=1),
            end=fit_time,
            margin_fraction=contract.portfolio["margin_fraction"],
        )
    assert str(result["failure_timestamp"]) in str(engine_error.value)


@pytest.mark.parametrize(
    ("horizon", "terminal_carry"),
    [(1, 1), (4, 7), (24, 47)],
)
def test_r0_schedule_phase_and_terminal_carry_stay_anchored(horizon, terminal_carry):
    times = pd.date_range("2025-01-01", periods=600, freq="h", tz="UTC")
    anchor = times[0] - pd.Timedelta(hours=1)
    fit_time = anchor + pd.Timedelta(hours=168)
    validation_end = fit_time - pd.Timedelta(hours=horizon + 1)
    validation_start = validation_end - pd.Timedelta(hours=503)
    source, execution, carry = _last_r0_event(validation_end, fit_time, anchor, horizon)

    assert source <= validation_end
    assert (source - anchor) % pd.Timedelta(hours=horizon) == pd.Timedelta(0)
    assert execution == source + pd.Timedelta(hours=1)
    assert carry == terminal_carry
    assert (validation_start - anchor) % pd.Timedelta(hours=horizon) == pd.Timedelta(0)


def test_einsum_training_statistics_match_previous_matrix_products_and_invalid_states():
    rng = np.random.default_rng(18)
    x_train = rng.normal(size=(180, 7))
    y_train = rng.normal(size=180)
    target_rms = float(np.sqrt(np.mean(np.square(y_train))))
    expected_gram = (x_train.T @ x_train) / len(y_train)
    expected_rhs = (x_train.T @ (y_train / target_rms)) / len(y_train)

    actual = _ridge_training_stats(x_train, y_train)

    np.testing.assert_allclose(actual.train_gram, expected_gram, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.train_rhs, expected_rhs, rtol=1e-14, atol=1e-14)
    assert actual.target_rms == pytest.approx(target_rms)
    assert actual.sample_count == len(y_train)
    assert actual.invalid_status is None
    assert _ridge_training_stats(np.empty((0, 7)), np.empty(0)).invalid_status == (
        "empty_inner_training_sample"
    )
    assert _ridge_training_stats(x_train, np.zeros_like(y_train)).invalid_status == (
        "invalid_training_target_rms"
    )
