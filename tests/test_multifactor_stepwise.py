import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_selection import (
    ROUTES,
    STEPWISE_ROUTE,
    _ridge_validation_stats,
    _select_alpha,
    _select_alpha_ridge_stats,
    _stepwise_select_with_evaluator,
    generate_selection_scores,
)
from crypto_quant.research.strategy_research import multifactor_selection


def _policy(**updates):
    policy = {
        "fit_window_hours": 48,
        "min_train_periods": 8,
        "refit_every_hours": 12,
        "min_cross_section_symbols": 4,
        "icir_window_hours": 8,
        "icir_min_periods": 3,
        "correlation_window_hours": 6,
        "cluster_abs_correlation_threshold": 0.9,
        "inner_validation_hours": 8,
        "min_inner_validation_periods": 4,
        "alpha_grid": [1e-4, 0.01],
        "l1_ratio": 0.5,
        "pool_capacity": 3,
        "pool_search_budget": 64,
        "min_objective_improvement": 1e-6,
    }
    policy.update(updates)
    return policy


def _small_dataset(seed=17, periods=18, symbols=7, factors=4):
    rng = np.random.default_rng(seed)
    times = list(pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC"))
    dataset = {}
    beta = rng.normal(size=factors)
    for timestamp in times:
        x = rng.normal(size=(symbols, factors))
        x -= x.mean(axis=0, keepdims=True)
        y = x @ beta + rng.normal(scale=0.3, size=symbols)
        y -= y.mean()
        dataset[pd.Timestamp(timestamp)] = x, y
    return dataset, times


def test_cached_ridge_subset_objective_matches_direct_validation_r2():
    dataset, times = _small_dataset()
    train_times, validation_times = times[:12], times[12:]
    policy = _policy()
    stats = _ridge_validation_stats(dataset, train_times, validation_times, factor_count=4)
    for columns in ([0], [1, 3], [0, 1, 2, 3]):
        direct = _select_alpha("ridge", dataset, train_times, validation_times,
                               list(columns), policy)
        cached = _select_alpha_ridge_stats(stats, list(columns), policy)
        assert cached[0] == direct[0]
        assert cached[1] == pytest.approx(direct[1], abs=2e-13)
        assert [trial["alpha"] for trial in cached[2]] == [trial["alpha"] for trial in direct[2]]
        for cached_trial, direct_trial in zip(cached[2], direct[2]):
            assert cached_trial["validation_r2"] == pytest.approx(
                direct_trial["validation_r2"], abs=2e-13,
            )


def test_stepwise_selects_best_proposals_prunes_and_reconsiders_features():
    dataset, _ = _small_dataset(factors=3)
    policy = _policy(pool_capacity=3)
    objectives = {
        (0,): 0.50,
        (1,): 0.40,
        (2,): 0.30,
        (0, 1): 0.55,
        (0, 2): 0.60,
        (1, 2): 0.80,
        (0, 1, 2): 0.70,
    }

    def evaluate(columns):
        objective = objectives[columns]
        return 0.01, objective, [{"alpha": 0.01, "validation_score": objective}]

    result = _stepwise_select_with_evaluator(
        ["a", "b", "c"], policy, evaluate, objective_name="test_score",
    )

    assert result["selection_order"] == ["a", "c", "b"]
    assert result["selected"] == [1, 2]
    assert result["objective"] == pytest.approx(0.80)
    assert result["stop_reason"] == "no_positive_forward_gain"
    assert result["proposal_evaluations"] == sum(
        len(round_record["proposals"]) for round_record in result["rounds"]
    )
    assert result["proposal_evaluations"] <= result["proposal_search_budget_total"]
    assert result["unique_subset_evaluations"] < result["proposal_evaluations"]

    forward_rounds = [item for item in result["rounds"] if item["phase"] == "forward"]
    assert [len(item["proposals"]) for item in forward_rounds] == [3, 2, 1, 1]
    assert forward_rounds[0]["proposals"][0]["action"] == "add"
    prune = [item for item in result["rounds"]
             if item["phase"] == "backward_prune" and item["action"] == "remove_best_pruning_gain"]
    assert len(prune) == 1
    assert prune[0]["proposals"][0]["candidate"] == "a"
    assert prune[0]["proposals"][0]["delta"] == pytest.approx(0.10)
    assert prune[0]["selected_after"] == ["b", "c"]
    assert result["rounds"][-2]["phase"] == "backward_prune"
    assert result["rounds"][-2]["action"] == "keep_no_positive_pruning_gain"
    assert result["rounds"][-1]["action"] == "stop_no_positive_forward_gain"
    assert [proposal["candidate"] for proposal in result["rounds"][-1]["proposals"]] == ["a"]


def test_stepwise_batch_evaluator_runs_once_per_complete_round_for_uncached_subsets():
    objectives = {
        (0,): 0.50,
        (1,): 0.40,
        (2,): 0.30,
        (0, 1): 0.55,
        (0, 2): 0.60,
        (1, 2): 0.80,
        (0, 1, 2): 0.70,
    }
    calls = []

    def evaluate_batch(subsets):
        calls.append(tuple(subsets))
        return {
            columns: (0.01, objectives[columns],
                      [{"alpha": 0.01, "validation_score": objectives[columns]}])
            for columns in subsets
        }

    result = _stepwise_select_with_evaluator(
        ["a", "b", "c"], _policy(pool_capacity=3), None,
        objective_name="test_score", batch_subset_evaluator=evaluate_batch,
    )

    assert result["selected"] == [1, 2]
    assert calls == [
        ((0,), (1,), (2,)),
        ((0, 1), (0, 2)),
        ((0, 1, 2),),
        ((1, 2),),
    ]
    assert result["alpha_trial_evaluations"] == 7


@pytest.mark.parametrize(
    ("budget", "expected_stop", "expected_selected", "expected_proposals"),
    [
        (2, "budget_insufficient_forward_round", [], 0),
        (3, "budget_insufficient_backward_round", [0], 3),
    ],
)
def test_stepwise_never_uses_partial_proposal_round(budget, expected_stop,
                                                    expected_selected, expected_proposals):
    policy = _policy(pool_capacity=3, pool_search_budget=budget)
    objectives = {(0,): 0.5, (1,): 0.4, (2,): 0.3}

    def evaluate(columns):
        return 0.01, objectives[columns], [{"alpha": 0.01, "validation_score": objectives[columns]}]

    result = _stepwise_select_with_evaluator(
        ["a", "b", "c"], policy, evaluate, objective_name="test_score",
    )
    assert result["stop_reason"] == expected_stop
    assert result["selected"] == expected_selected
    assert result["proposal_evaluations"] == expected_proposals
    incomplete = result["rounds"][-1]
    assert incomplete["status"] == "budget_insufficient_for_complete_round"
    assert incomplete["proposals"] == []
    assert incomplete["required_proposals"] > incomplete["available_budget"]

    batch_calls = []

    def evaluate_batch(columns):
        batch_calls.append(tuple(columns))
        return {subset: (0.01, objectives[subset],
                         [{"alpha": 0.01, "validation_score": objectives[subset]}])
                for subset in columns}

    batched = _stepwise_select_with_evaluator(
        ["a", "b", "c"], policy, None, objective_name="test_score",
        batch_subset_evaluator=evaluate_batch,
    )
    assert batched["stop_reason"] == expected_stop
    assert batched["selected"] == expected_selected
    assert batched["proposal_evaluations"] == expected_proposals
    assert batch_calls == ([((0,), (1,), (2,))] if expected_proposals else [])


def _panel(periods=60):
    symbols = [f"C{i:02d}" for i in range(8)]
    times = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC", name="timestamp")
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    rng = np.random.default_rng(91)
    factors = rng.normal(size=(periods, len(symbols), 3))
    factors -= factors.mean(axis=1, keepdims=True)
    factors /= factors.std(axis=1, keepdims=True)
    values = pd.DataFrame(factors.reshape(-1, 3), index=index,
                          columns=["signal_a", "signal_b", "noise"])
    log_price = np.zeros((periods, len(symbols)))
    for position in range(1, periods):
        source = factors[position - 2, :, 0] if position >= 2 else 0.0
        log_price[position] = (log_price[position - 1] + 0.008 * source
                               + rng.normal(scale=0.003, size=len(symbols)))
    opens = pd.Series((100 * np.exp(log_price)).reshape(-1), index=index, name="perp_open")
    return values, opens


def test_stepwise_method_filter_skips_legacy_routes_and_keeps_shared_readiness(monkeypatch):
    values, opens = _panel()
    policy = _policy(pool_capacity=2, pool_search_budget=24)
    legacy = generate_selection_scores(values, opens, horizon_hours=1, policy=policy)
    expected_ready = legacy.scores["equal"].notna().groupby(level="timestamp").any()

    def unexpected_call(*args, **kwargs):
        raise AssertionError("an unrequested legacy route was evaluated")

    monkeypatch.setattr(
        "crypto_quant.research.strategy_research.multifactor_selection._cluster_members",
        unexpected_call,
    )
    monkeypatch.setattr(
        "crypto_quant.research.strategy_research.multifactor_selection._rolling_icir_weights",
        unexpected_call,
    )
    monkeypatch.setattr(
        "crypto_quant.research.strategy_research.multifactor_selection._elastic_net_coefficients",
        unexpected_call,
    )
    matured_calls = 0
    split_calls = 0
    original_matured_window = multifactor_selection._matured_window
    original_window_split = multifactor_selection._window_split

    def count_matured_window(*args, **kwargs):
        nonlocal matured_calls
        matured_calls += 1
        return original_matured_window(*args, **kwargs)

    def count_window_split(*args, **kwargs):
        nonlocal split_calls
        split_calls += 1
        return original_window_split(*args, **kwargs)

    monkeypatch.setattr(multifactor_selection, "_matured_window", count_matured_window)
    monkeypatch.setattr(multifactor_selection, "_window_split", count_window_split)
    selected = generate_selection_scores(
        values, opens, horizon_hours=1, policy=policy, methods=(STEPWISE_ROUTE,),
    )

    assert tuple(selected.scores) == (STEPWISE_ROUTE,)
    assert set(ROUTES).isdisjoint(selected.scores)
    assert any(score.notna().any() for score in selected.scores.values())
    expected_refits = (len(expected_ready) + policy["refit_every_hours"] - 1) // policy["refit_every_hours"]
    assert matured_calls == split_calls == expected_refits
    pd.testing.assert_series_equal(selected.shared_model_ready, expected_ready.rename("shared_model_ready"))
    stepwise_fits = [fit for fit in selected.fits if fit["route"] == STEPWISE_ROUTE]
    assert stepwise_fits
    assert not any(fit["route"] in ROUTES for fit in selected.fits)
    for fit in stepwise_fits:
        assert fit["proposal_evaluations"] == sum(
            len(round_record["proposals"]) for round_record in fit["selection_rounds"]
        )
        assert fit["proposal_evaluations"] <= fit["proposal_search_budget_total"]
        assert fit["fit_label_matured_through"] <= fit["timestamp"]
        for round_record in fit["selection_rounds"]:
            if round_record["status"] == "complete":
                assert all("alpha_trials" in proposal and "action" in proposal
                           and "delta" in proposal
                           for proposal in round_record["proposals"])


def test_stepwise_only_refit_optimization_preserves_fixed_scores_fit_and_readiness():
    values, opens = _panel()
    policy = _policy(pool_capacity=2, pool_search_budget=24)
    result = generate_selection_scores(
        values, opens, horizon_hours=1, policy=policy, methods=(STEPWISE_ROUTE,),
    )

    timestamp = result.shared_model_ready.index[36]
    score = result.scores[STEPWISE_ROUTE].xs(timestamp, level="timestamp")
    np.testing.assert_allclose(score.to_numpy(), [
        0.2872712991564994,
        0.885374485417195,
        -0.5836141707100989,
        0.783375112570695,
        -0.5558892319231183,
        -1.357903229573859,
        1.4677322605806633,
        -0.9263465255179764,
    ], rtol=0.0, atol=2e-14)
    assert [i for i, ready in enumerate(result.shared_model_ready.tolist()) if ready] == list(range(24, 60))

    fit = next(item for item in result.fits if item["timestamp"] == timestamp)
    assert fit["selected_factors"] == ["signal_a"]
    assert fit["selected_alpha"] == 0.0001
    assert fit["validation_r2"] == pytest.approx(0.8878657431594168, abs=2e-14)
    assert fit["stop_reason"] == "no_positive_forward_gain"
    assert fit["proposal_evaluations"] == 6
    assert fit["fit_source_through"] == pd.Timestamp("2024-01-02 10:00:00+00:00")
    assert fit["fit_label_matured_through"] == timestamp
    assert fit["coefficients"]["signal_a"] == pytest.approx(0.9344761058434864, abs=2e-14)


def test_stepwise_truncation_is_causal_through_the_cutoff():
    values, opens = _panel(periods=72)
    policy = _policy(pool_capacity=2, pool_search_budget=24)
    original = generate_selection_scores(
        values, opens, horizon_hours=1, policy=policy, methods=(STEPWISE_ROUTE,),
    )
    cutoff = values.index.get_level_values("timestamp").unique()[48]
    future_rows = values.index.get_level_values("timestamp") > cutoff
    changed_values = values.copy()
    changed_values.loc[future_rows] *= -1.0
    changed_opens = opens.copy()
    changed_opens.loc[future_rows] *= np.linspace(0.2, 3.0, int(future_rows.sum()))
    changed = generate_selection_scores(
        changed_values, changed_opens, horizon_hours=1, policy=policy,
        methods=(STEPWISE_ROUTE,),
    )

    before_cutoff = values.index.get_level_values("timestamp") <= cutoff
    pd.testing.assert_series_equal(
        changed.scores[STEPWISE_ROUTE].loc[before_cutoff],
        original.scores[STEPWISE_ROUTE].loc[before_cutoff],
    )
    assert [fit for fit in changed.fits if fit["timestamp"] <= cutoff] == [
        fit for fit in original.fits if fit["timestamp"] <= cutoff
    ]
    pd.testing.assert_series_equal(
        changed.shared_model_ready.loc[:cutoff], original.shared_model_ready.loc[:cutoff],
    )


def test_stepwise_waits_for_common_readiness_and_matches_pool_fit_schedule():
    values, opens = _panel()
    policy = _policy(pool_capacity=2, pool_search_budget=24,
                     refit_every_hours=6, correlation_window_hours=30)
    timestamps = values.index.get_level_values("timestamp").unique()
    early_timestamp = timestamps[24]
    matured = multifactor_selection._matured_window(
        pd.DatetimeIndex(timestamps), early_timestamp, 1, policy,
    )
    train, validation = multifactor_selection._window_split(matured, early_timestamp, 1, policy)
    assert len(train) >= policy["min_train_periods"]
    assert len(validation) >= policy["min_inner_validation_periods"]
    assert 24 < policy["correlation_window_hours"]

    result = generate_selection_scores(
        values, opens, horizon_hours=1, policy=policy,
        methods=(STEPWISE_ROUTE, "pool"),
    )
    stepwise_times = [fit["timestamp"] for fit in result.fits if fit["route"] == STEPWISE_ROUTE]
    pool_times = [fit["timestamp"] for fit in result.fits if fit["route"] == "pool"]
    assert stepwise_times == pool_times
    assert early_timestamp not in stepwise_times
    assert result.scores[STEPWISE_ROUTE].xs(early_timestamp, level="timestamp").isna().all()
