import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_combination import rolling_icir_combine
from crypto_quant.research.strategy_research.multifactor_selection import (
    ROUTES,
    _elastic_net_coefficients,
    generate_selection_scores,
)


SYMBOLS = [f"S{i:02d}" for i in range(8)]


def _policy(**updates):
    policy = {
        "fit_window_hours": 60,
        "min_train_periods": 10,
        "refit_every_hours": 6,
        "min_cross_section_symbols": 4,
        "icir_window_hours": 8,
        "icir_min_periods": 3,
        "correlation_window_hours": 6,
        "cluster_abs_correlation_threshold": 0.9,
        "inner_validation_hours": 8,
        "min_inner_validation_periods": 4,
        "alpha_grid": [1e-4, 0.01],
        "l1_ratio": 0.5,
        "pool_capacity": 2,
        "pool_search_budget": 5,
        "min_objective_improvement": 1e-6,
    }
    policy.update(updates)
    return policy


def _fixture(periods=96, *, factor_count=4, seed=37):
    times = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC", name="timestamp")
    index = pd.MultiIndex.from_product([times, SYMBOLS], names=["timestamp", "symbol"])
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(periods, len(SYMBOLS), max(4, factor_count)))
    raw[:, :, 1] = raw[:, :, 0]
    raw[:, :, 2] = -raw[:, :, 0]
    raw -= raw.mean(axis=1, keepdims=True)
    raw /= raw.std(axis=1, keepdims=True)
    names = ["signal", "duplicate", "inverse", "noise"][:factor_count]
    values = pd.DataFrame(raw[:, :, :factor_count].reshape(-1, factor_count),
                          index=index, columns=names)
    log_price = np.zeros((periods, len(SYMBOLS)))
    log_price[0] = rng.normal(scale=0.02, size=len(SYMBOLS))
    for t in range(1, periods):
        source = raw[t - 2, :, 0] if t >= 2 else 0.0
        log_price[t] = log_price[t - 1] + 0.006 * source + rng.normal(scale=0.004, size=len(SYMBOLS))
    opens = pd.Series((100 * np.exp(log_price)).reshape(-1), index=index, name="perp_open")
    return values, opens


def _complementary_fixture(periods=144, factor_count=53, seed=93):
    symbols = [f"C{i:02d}" for i in range(10)]
    times = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC", name="timestamp")
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(periods, len(symbols), factor_count))
    factors[:, :, 0] = rng.normal(size=(periods, len(symbols)))
    factors[:, :, 1] = rng.normal(size=(periods, len(symbols)))
    factors -= factors.mean(axis=1, keepdims=True)
    factors /= factors.std(axis=1, keepdims=True)
    columns = ["signal_a", "signal_b", *[f"noise_{i:02d}" for i in range(factor_count - 2)]]
    values = pd.DataFrame(factors.reshape(-1, factor_count), index=index, columns=columns)
    log_price = np.zeros((periods, len(symbols)))
    log_price[0] = rng.normal(scale=0.02, size=len(symbols))
    for t in range(1, periods):
        if t >= 2:
            source = factors[t - 2, :, 0] + factors[t - 2, :, 1]
        else:
            source = 0.0
        log_price[t] = log_price[t - 1] + 0.012 * source + rng.normal(scale=0.002, size=len(symbols))
    opens = pd.Series((100 * np.exp(log_price)).reshape(-1), index=index, name="perp_open")
    return values, opens


def _first_fit(result, route):
    return next(item for item in result.fits if item["route"] == route)


def test_all_routes_share_full_index_and_causal_maturity_and_purged_split():
    values, opens = _fixture()
    result = generate_selection_scores(values, opens, horizon_hours=1, policy=_policy())
    assert tuple(result.scores) == ROUTES
    assert all(score.index.equals(values.index) and score.name == "score"
               for score in result.scores.values())
    fit = _first_fit(result, "ridge")
    assert fit["latest_label_maturity"] <= fit["timestamp"]
    assert fit["latest_train_label_maturity"] <= fit["timestamp"]
    assert fit["inner_train_source_end"] + pd.Timedelta(hours=2) < fit["validation_source_start"]
    assert fit["train_source_end"] + pd.Timedelta(hours=2) == fit["timestamp"]
    assert set(fit["coefficients"]) == set(values.columns)
    assert np.isfinite(list(fit["coefficients"].values())).all()
    legacy = rolling_icir_combine(
        values, pd.DataFrame(index=values.index.get_level_values("timestamp").unique()), opens,
        horizon_hours=1,
        policy={"method": "symmetric_orthogonalized_rolling_icir", "window_hours": 8,
                "min_periods": 3, "negative_icir": "flip_factor"},
    )
    common_ready = result.scores["equal"].notna()
    pd.testing.assert_series_equal(result.scores["rolling_icir"].loc[common_ready],
                                   legacy.score.loc[common_ready])
    first_ridge_update = next(item["timestamp"] for item in result.fits if item["route"] == "ridge")
    between_updates = first_ridge_update + pd.Timedelta(hours=1)
    assert not any(item["route"] == "ridge" and item["timestamp"] == between_updates
                   for item in result.fits)
    icir_update = next(item for item in result.fits
                       if item["route"] == "rolling_icir" and item["timestamp"] == between_updates)
    np.testing.assert_allclose(
        list(icir_update["factor_weights"].values()), legacy.weights.loc[between_updates].to_numpy(),
    )
    pd.testing.assert_series_equal(
        result.scores["rolling_icir"].xs(between_updates, level="timestamp"),
        legacy.score.xs(between_updates, level="timestamp"),
    )


@pytest.mark.parametrize("horizon", [1, 4, 24])
def test_future_factor_and_open_changes_do_not_change_scores_or_fits_before_cutoff(horizon):
    values, opens = _fixture(periods=112)
    policy = _policy(refit_every_hours=8)
    baseline = generate_selection_scores(values, opens, horizon_hours=horizon, policy=policy)
    cutoff = values.index.get_level_values("timestamp").unique()[70]
    future_rows = values.index.get_level_values("timestamp") > cutoff
    altered_values = values.copy()
    altered_values.loc[future_rows] = altered_values.loc[future_rows] * -1.0
    altered_opens = opens.copy()
    altered_opens.loc[future_rows] *= np.linspace(0.2, 3.0, int(future_rows.sum()))
    changed = generate_selection_scores(altered_values, altered_opens,
                                        horizon_hours=horizon, policy=policy)
    before_cutoff = values.index.get_level_values("timestamp") <= cutoff
    for route in ROUTES:
        pd.testing.assert_series_equal(changed.scores[route].loc[before_cutoff],
                                       baseline.scores[route].loc[before_cutoff])
    for route in {"cluster", "rolling_icir", "ridge", "elastic_net", "pool"}:
        original_fits = [fit for fit in baseline.fits if fit["route"] == route and fit["timestamp"] <= cutoff]
        changed_fits = [fit for fit in changed.fits if fit["route"] == route and fit["timestamp"] <= cutoff]
        assert changed_fits == original_fits


def test_cluster_groups_duplicate_and_inverse_factors_with_logged_orientation():
    values, opens = _fixture(factor_count=3)
    result = generate_selection_scores(values, opens, horizon_hours=1, policy=_policy())
    fit = _first_fit(result, "cluster")
    assert fit["grouping"] == "absolute_correlation_connected_components"
    assert len(fit["groups"]) == 1
    group = fit["groups"][0]
    assert group["members"] == ["signal", "duplicate", "inverse"]
    assert group["orientations"] == {"signal": 1, "duplicate": 1, "inverse": -1}
    timestamp = fit["timestamp"]
    expected = values.xs(timestamp, level="timestamp")["signal"]
    pd.testing.assert_series_equal(result.scores["cluster"].xs(timestamp, level="timestamp"),
                                   expected.rename("score"), check_names=True)


def test_pool_capacity_total_search_budget_and_cash_when_no_positive_gain():
    values, opens = _fixture(factor_count=4)
    result = generate_selection_scores(values, opens, horizon_hours=1,
                                       policy=_policy(pool_capacity=1, pool_search_budget=2))
    fit = _first_fit(result, "pool")
    assert len(fit["selected_factors"]) <= 1
    assert fit["candidate_evaluations"] <= 2
    assert fit["candidate_search_budget_total"] == 2
    assert fit["selected_factors"]
    assert set(fit["candidate_order"][:2]) == {"signal", "duplicate"}

    cash = generate_selection_scores(values, opens, horizon_hours=1,
                                     policy=_policy(min_objective_improvement=2.0))
    cash_fit = _first_fit(cash, "pool")
    assert cash_fit["selected_factors"] == []
    assert cash_fit["status"] == "cash_no_positive_marginal_gain"
    assert cash.scores["pool"].isna().all()


def test_pool_can_add_two_complementary_predictors_with_more_factors_than_budget():
    values, opens = _complementary_fixture()
    policy = _policy(fit_window_hours=100, min_train_periods=20,
                     icir_window_hours=24, icir_min_periods=8,
                     correlation_window_hours=6, inner_validation_hours=20,
                     min_inner_validation_periods=16, alpha_grid=[1e-5, 0.001],
                     pool_capacity=2, pool_search_budget=32)
    result = generate_selection_scores(values, opens, horizon_hours=1, policy=policy)
    fit = _first_fit(result, "pool")
    assert fit["selected_factors"] == ["signal_a", "signal_b"]
    assert fit["candidate_evaluations"] <= 32
    assert len(fit["selected_factors"]) <= 2
    assert result.scores["pool"].notna().any()


def test_elastic_net_matches_soft_threshold_and_duplicate_predictor_kkt_conditions():
    x = np.array([-2.0, -1.0, 1.0, 2.0])[:, None]
    y = np.array([-1.5, -0.7, 0.9, 1.6])
    alpha, l1_ratio = 0.1, 0.5
    actual = _elastic_net_coefficients(x, y, alpha, l1_ratio)[0]
    variance = float(np.mean(x[:, 0] ** 2))
    covariance = float(np.mean(x[:, 0] * y))
    expected = np.sign(covariance) * max(abs(covariance) - alpha * l1_ratio, 0.0)
    expected /= variance + alpha * (1.0 - l1_ratio)
    assert actual == pytest.approx(expected, abs=1e-12)

    duplicated = np.column_stack([x[:, 0], x[:, 0], -x[:, 0]])
    coefficients = _elastic_net_coefficients(duplicated, y, alpha, l1_ratio)
    gram = np.einsum("ni,nj->ij", duplicated, duplicated) / len(y)
    rhs = np.einsum("ni,n->i", duplicated, y) / len(y)
    gradient = np.einsum("ij,j->i", gram, coefficients) - rhs + alpha * (1.0 - l1_ratio) * coefficients
    kkt = np.empty_like(coefficients)
    positive = coefficients > 1e-10
    negative = coefficients < -1e-10
    zero = ~(positive | negative)
    kkt[positive] = np.abs(gradient[positive] + alpha * l1_ratio)
    kkt[negative] = np.abs(gradient[negative] - alpha * l1_ratio)
    kkt[zero] = np.maximum(np.abs(gradient[zero]) - alpha * l1_ratio, 0.0)
    # The solver's dual-gap tolerance does not bound each coordinate gradient;
    # this checks that duplicate signed predictors still meet tight original KKT.
    assert float(kkt.max()) < 1e-7


def test_elastic_net_near_collinear_predictors_satisfy_original_space_kkt():
    rng = np.random.default_rng(211)
    base = rng.normal(size=240)
    other = rng.normal(size=240)
    x = np.column_stack([
        base,
        base + 1e-5 * rng.normal(size=len(base)),
        -base + 2e-5 * rng.normal(size=len(base)),
        other,
        rng.normal(size=len(base)),
    ])
    y = 0.7 * base - 0.3 * other + rng.normal(scale=0.4, size=len(base))
    alpha, l1_ratio = 0.005, 0.7
    coefficients = _elastic_net_coefficients(x, y, alpha, l1_ratio)
    gram = np.einsum("ni,nj->ij", x, x) / len(y)
    rhs = np.einsum("ni,n->i", x, y) / len(y)
    gradient = np.einsum("ij,j->i", gram, coefficients) - rhs
    gradient += alpha * (1.0 - l1_ratio) * coefficients
    kkt = np.empty_like(coefficients)
    positive = coefficients > 1e-10
    negative = coefficients < -1e-10
    zero = ~(positive | negative)
    kkt[positive] = np.abs(gradient[positive] + alpha * l1_ratio)
    kkt[negative] = np.abs(gradient[negative] - alpha * l1_ratio)
    kkt[zero] = np.maximum(np.abs(gradient[zero]) - alpha * l1_ratio, 0.0)
    assert float(kkt.max()) < 1e-7


def test_elastic_net_reproducible_seed_10_low_rank_stress_satisfies_kkt():
    rng = np.random.default_rng(10)
    base = rng.normal(size=(180, 9))
    raw = np.einsum("ni,ij->nj", base, rng.normal(size=(9, 30)))
    raw += rng.normal(scale=1e-4, size=(180, 30))
    x = raw - raw.mean(axis=0)
    x /= x.std(axis=0)
    y = np.einsum("ni,i->n", x, rng.normal(size=30)) + rng.normal(size=180)
    y = (y - y.mean()) / y.std()
    alpha, l1_ratio = 0.001, 0.5
    coefficients = _elastic_net_coefficients(x, y, alpha, l1_ratio)
    gram = np.einsum("ni,nj->ij", x, x) / len(y)
    rhs = np.einsum("ni,n->i", x, y) / len(y)
    gradient = np.einsum("ij,j->i", gram, coefficients) - rhs
    gradient += alpha * (1.0 - l1_ratio) * coefficients
    kkt = np.empty_like(coefficients)
    positive = coefficients > 1e-10
    negative = coefficients < -1e-10
    zero = ~(positive | negative)
    kkt[positive] = np.abs(gradient[positive] + alpha * l1_ratio)
    kkt[negative] = np.abs(gradient[negative] - alpha * l1_ratio)
    kkt[zero] = np.maximum(np.abs(gradient[zero]) - alpha * l1_ratio, 0.0)
    assert float(kkt.max()) < 1e-6


def test_common_warmup_and_minimum_prediction_cross_section_gate():
    values, opens = _fixture()
    result = generate_selection_scores(values, opens, horizon_hours=1, policy=_policy())
    equal_ready = result.scores["equal"].notna()
    cluster_ready = result.scores["cluster"].notna()
    pd.testing.assert_series_equal(equal_ready, cluster_ready)
    first_ready = values.index.get_level_values("timestamp")[equal_ready.to_numpy()][0]
    assert result.scores["equal"].loc[values.index.get_level_values("timestamp") < first_ready].isna().all()

    timestamp = values.index.get_level_values("timestamp").unique()[-2]
    broken = values.copy()
    row_mask = broken.index.get_level_values("timestamp") == timestamp
    symbols_at_time = broken.index.get_level_values("symbol")[row_mask]
    broken.loc[(timestamp, symbols_at_time[:5]), :] = np.nan
    changed = generate_selection_scores(broken, opens, horizon_hours=1, policy=_policy())
    for route in ROUTES:
        assert changed.scores[route].xs(timestamp, level="timestamp").isna().all()


def test_invalid_policy_and_non_utc_or_incomplete_panel_fail_fast():
    values, opens = _fixture()
    with pytest.raises(ValueError, match="policy fields"):
        generate_selection_scores(values, opens, horizon_hours=1,
                                  policy={**_policy(), "extra": True})
    naive_index = values.index.set_levels(values.index.levels[0].tz_localize(None), level="timestamp")
    naive_values = values.set_axis(naive_index)
    naive_opens = opens.set_axis(naive_index)
    with pytest.raises(ValueError, match="timezone-aware"):
        generate_selection_scores(naive_values, naive_opens, horizon_hours=1, policy=_policy())
    incomplete = values.iloc[:-1]
    with pytest.raises(ValueError, match="every symbol"):
        generate_selection_scores(incomplete, opens.iloc[:-1], horizon_hours=1, policy=_policy())
