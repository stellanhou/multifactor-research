import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_combination import (
    rolling_icir_combine, symmetric_orthogonalize, validate_combination,
)


POLICY = {"method": "symmetric_orthogonalized_rolling_icir", "window_hours": 8,
          "min_periods": 3, "negative_icir": "flip_factor"}
SYMBOLS = sorted(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"])


def _index(periods=36):
    times = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC", name="timestamp")
    return pd.MultiIndex.from_product([times, SYMBOLS], names=["timestamp", "symbol"])


def _fixture(periods=36):
    index = _index(periods)
    rng = np.random.default_rng(47)
    x = rng.normal(size=(periods, len(SYMBOLS), 3))
    x[:, :, 1] += x[:, :, 0] * 0.4
    values = pd.DataFrame(x.reshape(-1, 3), index=index, columns=["a", "b", "c"])
    eligible = pd.Series(True, index=index)
    log_price = np.cumsum(rng.normal(scale=0.005, size=(periods, len(SYMBOLS))), axis=0)
    # Give the first factor a positive, variable relation to next-open returns.
    for t in range(2, periods):
        log_price[t] = log_price[t - 1] + 0.008 * x[t - 2, :, 0] + rng.normal(scale=0.008, size=len(SYMBOLS))
    opens = pd.Series((100 * np.exp(log_price)).reshape(-1), index=index, name="perp_open")
    transformed, diagnostics = symmetric_orthogonalize(values, eligible)
    return values, eligible, transformed, diagnostics, opens


def test_symmetric_whitening_matches_covariance_inverse_square_root_and_order():
    values, eligible, transformed, _, _ = _fixture(4)
    for t in values.index.get_level_values("timestamp").unique():
        x = values.xs(t).to_numpy(copy=True)
        x -= x.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(x.T @ x / len(x))
        expected = x @ eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T
        actual = transformed.xs(t).to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(actual.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(actual.T @ actual / len(x), np.eye(3), atol=1e-12)
    permuted, _ = symmetric_orthogonalize(values[["c", "a", "b"]], eligible)
    np.testing.assert_allclose(permuted[values.columns], transformed, atol=1e-12)


def test_symmetric_rank_deficiency_retains_projector_without_ridge_or_column_order_bias():
    index = _index(1)
    rng = np.random.default_rng(18)
    # More factors than symbols, including two exactly duplicate columns.
    x = rng.normal(size=(len(SYMBOLS), 9))
    x[:, 1] = x[:, 0]
    values = pd.DataFrame(x, index=index, columns=[f"f{i}" for i in range(9)])
    eligible = pd.Series(True, index=index)
    actual, diagnostics = symmetric_orthogonalize(values, eligible)
    covariance = actual.to_numpy().T @ actual.to_numpy() / len(x)
    assert diagnostics.iloc[0]["effective_rank"] == len(SYMBOLS) - 1
    assert diagnostics.iloc[0]["status"] == "rank_deficient"
    np.testing.assert_allclose(covariance @ covariance, covariance, atol=1e-12)
    np.testing.assert_allclose(actual["f0"], actual["f1"], atol=1e-12)
    reversed_values, _ = symmetric_orthogonalize(values[values.columns[::-1]], eligible)
    np.testing.assert_allclose(reversed_values[values.columns], actual, atol=1e-12)
    assert np.isfinite(actual).all().all()


def test_missing_factors_are_excluded_and_cannot_be_imputed_inside_common_sample():
    values, eligible, _, _, _ = _fixture(2)
    values.iloc[0, 0] = np.nan
    common = eligible & values.notna().all(axis=1)
    actual, diagnostics = symmetric_orthogonalize(values, common)
    assert actual.iloc[0].isna().all()
    assert diagnostics.iloc[0]["common_symbols"] == 5
    with pytest.raises(ValueError, match="missing factors"):
        symmetric_orthogonalize(values, eligible)
    common.iloc[:6] = False
    common.iloc[:2] = True
    actual, diagnostics = symmetric_orthogonalize(values.fillna(0), common)
    assert actual.iloc[:6].isna().all().all()
    assert diagnostics.iloc[0]["status"] == "insufficient_cross_section"


@pytest.mark.parametrize("horizon", [1, 4, 24])
def test_rolling_icir_uses_only_mature_labels_and_matches_sample_std_formula(horizon):
    _, _, transformed, diagnostics, opens = _fixture(72)
    result = rolling_icir_combine(transformed, diagnostics, opens, horizon_hours=horizon, policy=POLICY)
    t = result.weights.index[45]
    mature_sample = result.rank_ic.iloc[45 - horizon - POLICY["window_hours"]:45 - horizon]
    expected_icir = mature_sample.mean() / mature_sample.std(ddof=1)
    np.testing.assert_allclose(result.icir.loc[t], expected_icir, rtol=1e-12)
    strength = expected_icir.abs()
    expected_weights = strength / strength.sum() if strength.sum() else strength * 0
    np.testing.assert_allclose(result.weights.loc[t], expected_weights, rtol=1e-12)
    np.testing.assert_allclose(result.directions.loc[t], np.sign(expected_icir))
    expected_score = transformed.xs(t).to_numpy() @ (expected_weights * np.sign(expected_icir)).to_numpy()
    np.testing.assert_allclose(result.score.xs(t), expected_score, rtol=1e-12)
    assert (result.weights >= 0).all().all()
    sums = result.weights.sum(axis=1)
    np.testing.assert_allclose(sums[sums > 0], 1)
    assert (result.weights.iloc[:horizon + POLICY["min_periods"]] == 0).all().all()
    assert result.score.loc[result.score.index.get_level_values("timestamp") < result.weights.index[horizon + POLICY["min_periods"]]].isna().all()


@pytest.mark.parametrize("horizon", [1, 4, 24])
def test_future_price_and_factor_changes_do_not_change_current_weights_or_scores(horizon):
    values, eligible, transformed, diagnostics, opens = _fixture(72)
    baseline = rolling_icir_combine(transformed, diagnostics, opens, horizon_hours=horizon, policy=POLICY)
    cutoff = baseline.weights.index[43]
    future = values.index.get_level_values("timestamp") > cutoff
    changed_opens = opens.copy()
    changed_opens.loc[future] *= np.linspace(0.3, 3, int(future.sum()))
    changed_values = values.copy()
    changed_values.loc[future] *= -7
    changed_transformed, changed_diagnostics = symmetric_orthogonalize(changed_values, eligible)
    changed = rolling_icir_combine(changed_transformed, changed_diagnostics, changed_opens,
                                  horizon_hours=horizon, policy=POLICY)
    pd.testing.assert_frame_equal(changed.weights.loc[:cutoff], baseline.weights.loc[:cutoff])
    pd.testing.assert_series_equal(changed.score.loc[~future], baseline.score.loc[~future])
    # The next opening belongs to the order-execution boundary and cannot
    # affect weights generated from the current completed signal bar.
    assert not changed.rank_ic.equals(baseline.rank_ic)


@pytest.mark.parametrize("horizon", [1, 4, 24])
def test_pending_labels_and_rolling_history_continue_across_stage_boundary(horizon):
    _, _, transformed, diagnostics, opens = _fixture(90)
    times = diagnostics.index
    split = times[53]
    continuous = rolling_icir_combine(transformed, diagnostics, opens, horizon_hours=horizon, policy=POLICY)
    previous_mask = transformed.index.get_level_values("timestamp") < split
    first = rolling_icir_combine(transformed.loc[previous_mask], diagnostics.loc[:times[52]],
                                 opens.loc[previous_mask], horizon_hours=horizon, policy=POLICY)
    cutoff = split - pd.Timedelta(hours=POLICY["window_hours"] + horizon + 1)
    tail = transformed.index.get_level_values("timestamp") >= cutoff
    history = first.orthogonalized.loc[tail[previous_mask]], opens.loc[previous_mask & tail]
    current_mask = transformed.index.get_level_values("timestamp") >= split - pd.Timedelta(hours=3)
    current = transformed.loc[current_mask].copy()
    # Repeated formula warmup may be incomplete; use previously computed values.
    current.loc[current.index.get_level_values("timestamp") < split] = np.nan
    second = rolling_icir_combine(current, diagnostics.loc[times[50]:], opens.loc[current_mask],
                                  horizon_hours=horizon, policy=POLICY, history=history)
    pd.testing.assert_frame_equal(second.weights.loc[split:], continuous.weights.loc[split:])
    compare_mask = second.score.index.get_level_values("timestamp") >= split
    pd.testing.assert_series_equal(second.score.loc[compare_mask], continuous.score.loc[second.score.loc[compare_mask].index])


def test_negative_policy_and_zero_variance_or_insufficient_history_produce_declared_weights():
    index = _index(16)
    times = index.get_level_values("timestamp").unique()
    ranks = np.arange(len(SYMBOLS), dtype=float)
    sample = []
    for t in range(len(times)):
        x = ranks.copy()
        if t % 2:
            x[[0, 1]] = x[[1, 0]]
        sample.extend(np.column_stack([x, -x]))
    values = pd.DataFrame(sample, index=index, columns=["positive", "negative"])
    prices = np.array([100 * (1 + 0.001 * ranks) ** t for t in range(len(times))])
    opens = pd.Series(prices.reshape(-1), index=index, name="perp_open")
    diagnostics = pd.DataFrame(index=times)
    result = rolling_icir_combine(values, diagnostics, opens, horizon_hours=1, policy=POLICY)
    np.testing.assert_allclose(result.weights.iloc[8], [0.5, 0.5])
    np.testing.assert_allclose(result.directions.iloc[8], [1, -1])
    np.testing.assert_allclose(result.score.xs(times[8]), values.xs(times[8])["positive"])
    negative_only = rolling_icir_combine(values[["negative"]], diagnostics, opens,
                                        horizon_hours=1, policy=POLICY)
    assert negative_only.weights.iloc[8, 0] == 1
    assert negative_only.directions.iloc[8, 0] == -1
    np.testing.assert_allclose(negative_only.score.xs(times[8]), -values.xs(times[8])["negative"])
    constant_values = pd.DataFrame(np.tile(ranks, len(times)), index=index, columns=["constant_ic"])
    constant = rolling_icir_combine(constant_values, diagnostics, opens, horizon_hours=1, policy=POLICY)
    assert constant.icir.isna().all().all()
    assert constant.score.isna().all()
    changed_prices = opens.copy()
    changed_prices.iloc[0] *= 2
    with pytest.raises(ValueError, match="history prices differ"):
        rolling_icir_combine(values, diagnostics, changed_prices, horizon_hours=1,
                             policy=POLICY, history=(values, opens))


def test_invalid_combination_policies_fail():
    for update in ({"window_hours": True}, {"min_periods": 1}, {"min_periods": 9},
                   {"negative_icir": "equal_weight_fallback"}, {"method": "equal_weight"}):
        with pytest.raises(ValueError):
            validate_combination({**POLICY, **update})
