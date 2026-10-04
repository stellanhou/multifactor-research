from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research import multifactor_net_sharpe as net
from crypto_quant.research.strategy_research import multifactor_selection as selection
from crypto_quant.research.strategy_research.multifactor_combination_verify import (
    HOUR,
    audit_refit_coefficients,
    audit_training_sample,
    audit_validation_mask,
    audit_windows,
    first_fit_in_months,
    verify_candidate_account,
)
from crypto_quant.research.strategy_research.multifactor_contracts import ResearchContract


def test_audit_windows_requires_full_calendars_and_the_exact_training_purge():
    horizon = 1
    fit_time = pd.Timestamp("2025-04-02T00:00:00Z")
    validation = pd.date_range(end=fit_time - (horizon + 1) * HOUR,
                               periods=6, freq="h")
    training_end = validation[0] - (horizon + 2) * HOUR
    training = pd.date_range(end=training_end, periods=6, freq="h")
    refit = pd.date_range(end=fit_time - (horizon + 1) * HOUR,
                          periods=6, freq="h")
    windows = SimpleNamespace(training=training, validation=validation, refit=refit)

    result = audit_windows(
        windows, fit_time=fit_time, horizon_hours=horizon,
        training_hours=6, validation_hours=6, refit_hours=6,
    )

    assert result.training_source_hours == 6
    assert result.validation_source_hours == 6
    assert result.refit_source_hours == 6
    assert result.training_label_gap_hours == 1
    assert result.validation_label_gap_to_fit_hours == 0
    assert result.validation_account_hours == 6

    overlapped = SimpleNamespace(
        training=pd.date_range(end=validation[0] - horizon * HOUR, periods=6, freq="h"),
        validation=validation,
        refit=refit,
    )
    with pytest.raises(ValueError, match="purge"):
        audit_windows(
            overlapped, fit_time=fit_time, horizon_hours=horizon,
            training_hours=6, validation_hours=6, refit_hours=6,
        )


def test_first_fit_in_each_month_is_selected_from_eligible_weekly_dates():
    dates = [
        pd.Timestamp("2023-03-06T00:00:00Z"),
        pd.Timestamp("2023-03-13T00:00:00Z"),
        pd.Timestamp("2024-03-04T00:00:00Z"),
        pd.Timestamp("2025-03-03T00:00:00Z"),
    ]
    selected = first_fit_in_months(iter(reversed(dates)))
    assert selected == {
        "202303": dates[0],
        "202403": dates[2],
        "202503": dates[3],
    }


def _training_inputs():
    times = pd.date_range("2024-01-01T00:00:00Z", periods=20, freq="h", name="timestamp")
    symbols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(88)
    values = rng.normal(size=(len(times), len(symbols), 2))
    values -= values.mean(axis=1, keepdims=True)
    values[4, 0, 1] = np.nan
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(times), len(symbols))), axis=0))
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    factors = pd.DataFrame(values.reshape(-1, 2), index=index, columns=["a", "b"])
    opens = pd.Series(prices.reshape(-1), index=index, name="perp_open")
    training_times = times[2:8]
    dataset = selection._make_labels(factors, opens, times, symbols, 1, 3)
    included = list(training_times.intersection(pd.DatetimeIndex(sorted(dataset))))
    x_train, y_train = selection._stack_sample(dataset, included)
    return SimpleNamespace(factors=factors, opens=opens), SimpleNamespace(training=training_times), x_train, y_train, included


def test_training_sample_audit_rebuilds_causal_next_open_features_and_targets():
    prepared, windows, x_train, y_train, included = _training_inputs()
    result = audit_training_sample(
        prepared, windows, x_train=x_train, y_train=y_train,
        training_source_times=included, horizon_hours=1, min_symbols=3,
    )
    assert result["status"] == "passed"
    assert result["calendar_source_hours"] == 6
    assert result["eligible_label_source_hours"] == len(included)
    assert result["used_sources_at_or_after_validation"] == 0

    bad_targets = y_train.copy()
    bad_targets[0] += 0.01
    with pytest.raises(AssertionError, match="targets differ"):
        audit_training_sample(
            prepared, windows, x_train=x_train, y_train=bad_targets,
            training_source_times=included, horizon_hours=1, min_symbols=3,
        )


def test_post_selection_refit_keeps_validation_alpha_but_recomputes_recent_coefficients():
    rng = np.random.default_rng(230)
    x_refit = rng.normal(size=(120, 4))
    y_refit = rng.normal(0.0, 0.002, size=120)
    subset = (0, 2, 3)
    alpha = 0.1
    target_rms = float(np.sqrt(np.mean(np.square(y_refit))))
    selected = selection._ridge_coefficients(
        x_refit[:, list(subset)], y_refit / target_rms, alpha,
    )
    coefficients = np.zeros(4)
    coefficients[list(subset)] = selected

    result = audit_refit_coefficients(
        coefficients, "ridge", subset, alpha,
        x_refit=x_refit, y_refit=y_refit,
        factor_names=["a", "b", "c", "d"],
        refit_metadata={"fit_sample_count": len(y_refit)},
    )
    assert result["status"] == "passed"
    assert result["alpha_frozen_from_validation"] == alpha
    assert result["sample_count"] == len(y_refit)
    assert result["coefficient_fit_source"] == "windows.refit only"


def test_validation_mask_requires_every_factor_and_a_shared_asset_mask():
    times = pd.date_range("2024-02-01T00:00:00Z", periods=6, freq="h")
    values = np.ones((6, 4, 3))
    complete = np.isfinite(values).all(axis=2)
    evaluator = SimpleNamespace(times=times, values=values, complete=complete)
    windows = SimpleNamespace(validation=times)
    result = audit_validation_mask(evaluator, windows, factor_count=3, expected_hours=6)
    assert result["calendar_source_hours"] == 6

    evaluator.complete[0, 0] = False
    with pytest.raises(AssertionError, match="full factor-completeness mask"):
        audit_validation_mask(evaluator, windows, factor_count=3, expected_hours=6)


def _account_fixture():
    symbols = [f"S{i:02d}" for i in range(8)]
    source_times = pd.date_range("2024-05-01T00:00:00Z", periods=36, freq="h", name="timestamp")
    market_times = pd.date_range(source_times[0], periods=37, freq="h", name="timestamp")
    rng = np.random.default_rng(105)
    frames = {}
    for index, symbol in enumerate(symbols):
        drift = rng.normal(0, 0.004, size=len(market_times)) + (index - 3.5) * 0.0003
        opens = 100.0 * np.exp(np.cumsum(drift))
        closes = opens * np.exp(rng.normal(0, 0.002, size=len(market_times)))
        marks = closes * np.exp(rng.normal(0, 0.0002, size=len(market_times)))
        frames[symbol] = pd.DataFrame(
            {"open": opens, "close": closes, "mark_close": marks}, index=market_times,
        )
    values = rng.normal(size=(len(source_times), len(symbols), 2))
    values -= values.mean(axis=1, keepdims=True)
    complete = np.isfinite(values).all(axis=2)
    fit_time = source_times[-1] + 2 * HOUR
    funding_rows = []
    for timestamp, rate, symbol in (
        (source_times[5] + HOUR, 0.0002, symbols[0]),
        (source_times[13] + pd.Timedelta(minutes=20), -0.0003, symbols[-1]),
    ):
        bar = timestamp.floor("h")
        funding_rows.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "funding_rate": rate,
            "mark_price": frames[symbol].loc[bar, "mark_close"],
        })
    funding = pd.DataFrame(funding_rows)
    market, validated_funding = net._market_arrays(frames, symbols, funding)
    contract = ResearchContract(
        schema_version=2,
        run_id="combination90-verifier-test",
        purpose="research",
        stage="development",
        start=(source_times[0] + HOUR).isoformat(),
        end=fit_time.isoformat(),
        warmup_hours=24,
        horizon_hours=1,
        cards=["card-a", "card-b"],
        universe="universe.csv",
        prior_data_use="synthetic test data",
        data_processing="synthetic test data",
        costs={"initial_capital": 10_000.0, "fee_bps": 10.0,
               "slippage_bps": 5.0, "stress_multiplier": 2.0},
        portfolio={"long_count": 2, "short_count": 2, "gross_exposure": 0.8,
                   "max_asset_weight": 0.2, "rebalance_hours": 12,
                   "margin_fraction": 0.1},
        dataset_manifest="manifest.json",
    )

    class EqualEvaluator:
        def __init__(self):
            self.method = "equal"
            self.factor_count = 2
            self.factor_names = ["a", "b"]
            self.x_train = rng.normal(size=(40, self.factor_count))
            self.y_train = rng.normal(size=40)
            self.values = values
            self.complete = complete
            self.times = source_times
            self.fit_time = fit_time
            self.market = market
            self.funding = validated_funding
            self.contract = contract
            self.market_frames = frames

        def coefficients(self, subset, alpha):
            beta = np.zeros(self.factor_count)
            beta[list(subset)] = 1.0 / len(subset)
            return beta

    evaluator = EqualEvaluator()
    coefficient = evaluator.coefficients((0, 1), None)
    batch_trial = net._simulate_r0_batch(
        coefficient.reshape(1, -1), values, complete, source_times, fit_time,
        market, validated_funding, contract, {"min_cross_section_symbols": 3},
    )[0]
    is_valid = bool(batch_trial["valid_objective"] and batch_trial["trade_count"] > 0
                    and batch_trial["annualized_volatility"] > 0)
    trial = {
        **batch_trial,
        "alpha": None,
        "valid_objective": is_valid,
        "validation_objective": batch_trial["validation_net_sharpe"] if is_valid else None,
    }
    evaluator.cache = {
        (0, 1): (None, batch_trial["validation_net_sharpe"] if is_valid else np.nan, [trial]),
    }
    return evaluator, frames


def test_candidate_account_audit_matches_hourly_engine_and_cost_components():
    evaluator, frames = _account_fixture()
    result = verify_candidate_account(
        evaluator, (0, 1), None, market_frames=frames, factor_names=["a", "b"],
    )
    assert result["status"] == "passed"
    assert result["checks"]["hourly_equity"]["max_abs_difference"] <= 1e-7
    assert result["checks"]["total_fees"]["abs_difference"] <= 1e-7
    assert result["checks"]["total_slippage_cost"]["abs_difference"] <= 1e-7
    assert result["checks"]["funding_event_cashflows"]["points"] == 2
    assert result["account_case_counts"]["funding_at_execution_open"] == 1
    assert result["account_case_counts"]["funding_later_within_hour"] == 1
    assert result["trade_count"] > 0
