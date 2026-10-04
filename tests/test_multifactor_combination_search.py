from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform

from crypto_quant.research.strategy_research import multifactor_net_sharpe as net
from crypto_quant.research.strategy_research import multifactor_selection as selection
from crypto_quant.research.strategy_research.multifactor_combination_search import (
    ALPHAS,
    SubsetEvaluator,
    admitted,
    calendar_windows,
    enumerate_grid,
    genetic_search,
    hierarchical_clusters,
    random_search,
    run_search,
    valid_metric,
)


class _FakeEvaluator:
    """Small deterministic scorer for search-space invariants."""

    def __init__(self, factor_count: int, *, route: str = "equal"):
        self.factor_count = factor_count
        self.method = route
        self.cache: dict[tuple[int, ...], tuple[float | None, float, list[dict]]] = {}
        self.cache_hits = 0
        self.requests: list[tuple[int, ...]] = []

    def evaluate(self, subsets: list[tuple[int, ...]]) -> dict:
        self.requests.extend(subsets)
        self.cache_hits += sum(subset in self.cache for subset in subsets)
        for subset in dict.fromkeys(subsets):
            if subset in self.cache:
                continue
            score = sum((factor + 1) * ((factor % 3) + 1) for factor in subset) / 100.0
            self.cache[subset] = (
                None,
                score,
                [{"alpha": None, "validation_net_sharpe": score,
                  "validation_net_return": score, "trade_count": 1,
                  "annualized_volatility": 1.0, "valid_objective": True}],
            )
        return {subset: self.cache[subset] for subset in subsets}

    def best(self):
        valid = [subset for subset, value in self.cache.items() if np.isfinite(value[1])]
        return min(valid, key=lambda subset: (-self.cache[subset][1], subset)) if valid else None

    def audit(self) -> dict:
        return {"unique_subset_evaluations": len(self.cache), "cache_hits": self.cache_hits}


def _metric(*, sharpe=0.4, net_return=0.1, trade_count=1, volatility=0.2,
            valid=True) -> dict:
    return {
        "valid_objective": valid,
        "trade_count": trade_count,
        "annualized_volatility": volatility,
        "validation_net_sharpe": sharpe,
        "validation_net_return": net_return,
        "status": "evaluated" if valid else "invalid_net_sharpe",
    }


@pytest.mark.parametrize("horizon", [1, 4, 24])
def test_calendar_windows_use_independent_90_day_calendars_and_mature_labels(horizon):
    fit_time = pd.Timestamp("2026-01-12 08:00:00", tz="UTC")

    windows = calendar_windows(fit_time, horizon=horizon)
    audit = windows.audit(horizon)
    label_delay = pd.Timedelta(hours=horizon + 1)

    assert len(windows.training) == 2160
    assert len(windows.validation) == 2160
    assert len(windows.refit) == 2160
    assert windows.training.freq == pd.tseries.frequencies.to_offset("h")
    assert windows.validation[0] - windows.training[-1] == label_delay + pd.Timedelta(hours=1)
    assert windows.training[-1] + label_delay == windows.validation[0] - pd.Timedelta(hours=1)
    assert windows.validation[-1] + label_delay == fit_time
    assert windows.refit[-1] + label_delay == fit_time
    assert audit["training_calendar_hours"] == 2160
    assert audit["validation_calendar_hours"] == 2160
    assert audit["refit_calendar_hours"] == 2160
    assert audit["validation_account_start"] == windows.validation[0] + pd.Timedelta(hours=1)
    assert audit["validation_account_end_exclusive"] == fit_time
    assert audit["training_last_label_maturity"] <= audit["validation_account_start"]
    assert audit["refit_last_label_maturity"] <= fit_time


def test_average_linkage_clusters_are_stable_and_use_absolute_mean_correlation():
    rng = np.random.default_rng(804)
    cube = np.empty((18, 9, 4), dtype=float)
    for hour in range(len(cube)):
        base = rng.normal(size=9)
        independent = rng.normal(size=9)
        cube[hour] = np.column_stack([base, base, independent, -base])

    groups, mean_correlation = hierarchical_clusters(cube, cluster_count=2)
    repeated, repeated_correlation = hierarchical_clusters(cube, cluster_count=2)

    assert groups == [[0, 1, 3], [2]]
    assert repeated == groups
    np.testing.assert_allclose(repeated_correlation, mean_correlation)
    assert mean_correlation[0, 1] == pytest.approx(1.0)
    assert mean_correlation[0, 3] == pytest.approx(-1.0)
    distances = 1.0 - np.abs(mean_correlation)
    np.fill_diagonal(distances, 0.0)
    expected_labels = cut_tree(linkage(squareform(distances), method="average"),
                               n_clusters=[2]).ravel()
    expected_groups = [np.flatnonzero(expected_labels == label).tolist()
                       for label in sorted(set(expected_labels))]
    expected_groups.sort(key=lambda group: group[0])
    assert groups == expected_groups


def test_average_linkage_has_expected_partition_where_single_linkage_differs():
    cube = np.random.default_rng(0).normal(size=(1, 12, 7))

    groups, mean_correlation = hierarchical_clusters(cube, cluster_count=3)
    distances = 1.0 - np.abs(mean_correlation)
    np.fill_diagonal(distances, 0.0)
    single_labels = cut_tree(linkage(squareform(distances), method="single"),
                             n_clusters=[3]).ravel()
    single_groups = [np.flatnonzero(single_labels == label).tolist()
                     for label in sorted(set(single_labels))]
    single_groups.sort(key=lambda group: group[0])

    assert groups == [[0, 5], [1, 4], [2, 3, 6]]
    assert single_groups == [[0, 5], [1, 2, 3, 6], [4]]
    assert groups != single_groups


def test_grid_enumerates_1792_full_combinations_and_exact_ragged_count():
    full_candidates = [[2 * cluster, 2 * cluster + 1] for cluster in range(8)]
    full_grid = enumerate_grid(full_candidates)

    assert len(full_grid) == math.comb(8, 5) * 2**5 == 1792
    assert len(set(full_grid)) == 1792
    assert all(len(subset) == 5 and len(set(subset)) == 5 for subset in full_grid)

    ragged = [[0], [1, 2], [], [3, 4], [5], [6, 7], [8], [9, 10]]
    ragged_grid = enumerate_grid(ragged)
    expected = sum(
        math.prod(len(ragged[index]) for index in chosen)
        for chosen in itertools.combinations(
            [index for index, group in enumerate(ragged) if group], 5
        )
    )
    assert len(ragged_grid) == expected
    assert len(set(ragged_grid)) == expected
    assert all(len(subset) == 5 for subset in ragged_grid)


def test_random_search_uses_unique_five_factor_subsets_from_full_pool():
    evaluator = _FakeEvaluator(53)

    result = random_search(evaluator, size=5, budget=37, seed=0)

    evaluated = set(evaluator.cache)
    assert result["unique_subset_evaluations"] == 37
    assert len(evaluated) == 37
    assert all(len(subset) == 5 and len(set(subset)) == 5 for subset in evaluated)
    assert all(0 <= factor < 53 for subset in evaluated for factor in subset)
    assert result["selected"] == evaluator.best()


def test_ga_has_distinct_cluster_initialization_unique_factors_and_same_cluster_offspring():
    groups = [list(range(cluster * 4, cluster * 4 + 4)) for cluster in range(5)]
    budget = 220

    first_evaluator = _FakeEvaluator(20)
    first = genetic_search(first_evaluator, groups, size=5, budget=budget, seed=0,
                           generations=5)
    second_evaluator = _FakeEvaluator(20)
    second = genetic_search(second_evaluator, groups, size=5, budget=budget, seed=0,
                            generations=5)

    assert first["initial_population"] == second["initial_population"]
    assert first["selected"] == second["selected"]
    assert first["generations"] == second["generations"]
    assert first["population_size"] == 100
    assert first["unique_subset_evaluations"] <= budget
    assert first["unique_subset_evaluations"] == len(first_evaluator.cache)
    assert all(len(individual) == 5 and len(set(individual)) == 5
               for individual in first["initial_population"])
    for individual in first["initial_population"]:
        assert len({next(cluster for cluster, members in enumerate(groups)
                         if factor in members) for factor in individual}) == 5
    complete_generations = [row for row in first["generations"]
                            if "same_cluster_individuals" in row]
    assert any(row["same_cluster_individuals"] > 0 for row in complete_generations)
    assert all(len(subset) == 5 and len(set(subset)) == 5
               for subset in first_evaluator.cache)


@pytest.mark.parametrize("budget", [16, 100])
def test_ga_population_uses_budget_and_stops_when_initial_population_spends_it_all(budget):
    groups = [list(range(cluster * 4, cluster * 4 + 4)) for cluster in range(5)]
    first_evaluator = _FakeEvaluator(20)
    first = genetic_search(first_evaluator, groups, size=5, budget=budget, seed=21,
                           generations=50)
    second = genetic_search(_FakeEvaluator(20), groups, size=5, budget=budget, seed=21,
                            generations=50)

    assert first["population_size"] == budget
    assert first["unique_subset_evaluations"] == budget
    assert first["stop_reason"] == "unique_subset_budget_exhausted"
    assert len(first["generations"]) == 1
    assert first["initial_population"] == second["initial_population"]
    assert first["selected"] == second["selected"]
    for individual in first["initial_population"]:
        assert len(individual) == 5 and len(set(individual)) == 5
        assert len({factor // 4 for factor in individual}) == 5


def test_ga_respects_generation_limit_when_budget_remains():
    groups = [[cluster * 3, cluster * 3 + 1, cluster * 3 + 2]
              for cluster in range(5)]
    result = genetic_search(_FakeEvaluator(15), groups, size=5, budget=500,
                            seed=3, generations=2)

    assert len(result["generations"]) <= 2
    assert result["unique_subset_evaluations"] <= 500
    assert result["stop_reason"] == "generation_limit"


def test_subset_evaluator_uses_fixed_alpha_order_and_factor_identity_for_exact_ties(monkeypatch):
    values = np.ones((3, 4, 2), dtype=float)
    x_train = np.array([[-1.0, 0.5], [0.0, -1.0], [1.0, 0.5], [0.0, 0.0]])
    y_train = np.array([-0.5, 1.0, -0.5, 0.0])
    evaluator = SubsetEvaluator(
        method="ridge", x_train=x_train, y_train=y_train, values=values,
        source_times=pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
        fit_time=pd.Timestamp("2025-01-01 04:00:00", tz="UTC"),
        market=None, funding=pd.DataFrame(), contract=None,
    )
    trial_metric = _metric(sharpe=0.25)

    def same_score(coefficients, *args, **kwargs):
        assert coefficients.shape == (2 * len(ALPHAS), 2)
        return [dict(trial_metric) for _ in coefficients]

    monkeypatch.setattr(net, "_simulate_r0_batch", same_score)
    evaluator.evaluate([(1,), (0,)])

    assert evaluator.cache[(1,)][0] == ALPHAS[0]
    assert evaluator.cache[(0,)][0] == ALPHAS[0]
    assert evaluator.best() == (0,)
    assert all(row["validation_objective"] == pytest.approx(0.25)
               for subset in ((0,), (1,)) for row in evaluator.cache[subset][2])


@pytest.mark.parametrize(
    "metric",
    [
        _metric(trade_count=0),
        _metric(volatility=0.0),
        _metric(volatility=None),
        _metric(sharpe=None),
        _metric(valid=False),
    ],
)
def test_invalid_cash_or_zero_volatility_metrics_are_not_search_objectives(metric):
    assert not valid_metric(metric)


def test_full_equal_baseline_evaluates_the_whole_pool_once_and_uses_common_admission_gate():
    evaluator = _FakeEvaluator(6)

    result = run_search(evaluator, algorithm="full", groups=[], grid=[], budget=1,
                        size=5, seed=0, factor_names=[f"f{i}" for i in range(6)])

    assert evaluator.requests == [(0, 1, 2, 3, 4, 5)]
    assert result["selected"] == (0, 1, 2, 3, 4, 5)
    assert result["unique_subset_evaluations"] == 1
    assert admitted(evaluator, result["selected"])[0]

    evaluator.cache[(0, 1, 2, 3, 4, 5)] = (
        None, 0.4, [{"alpha": None, **_metric(net_return=-0.01)}]
    )
    assert not admitted(evaluator, result["selected"])[0]


def test_stepwise_counts_unique_nonempty_subsets_reuses_cache_and_never_starts_partial_round():
    names = ["a", "b", "c"]
    objectives = {
        (0,): -2.0,
        (1,): -1.0,
        (2,): -3.0,
        (0, 1): -0.5,
        (0, 2): -1.5,
        (1, 2): -0.4,
        (0, 1, 2): -0.3,
    }
    calls: list[tuple[tuple[int, ...], ...]] = []

    def evaluate_batch(subsets):
        calls.append(tuple(subsets))
        return {
            subset: (None, objectives[subset],
                     [{"alpha": None, "validation_net_sharpe": objectives[subset]}])
            for subset in subsets
        }

    result = selection._stepwise_select_with_evaluator(
        names,
        {"pool_capacity": 3, "pool_search_budget": 4,
         "min_objective_improvement": 1e-6},
        None,
        objective_name="validation_net_sharpe",
        batch_subset_evaluator=evaluate_batch,
        budget_unit="unique_nonempty_subset",
        initial_objective=-np.inf,
    )

    assert result["selected"] == [1]
    assert result["objective"] == -1.0
    assert result["unique_subset_evaluations"] == 3
    assert result["proposal_evaluations"] == 4
    assert calls == [((0,), (1,), (2,))]
    assert result["stop_reason"] == "budget_insufficient_forward_round"
    assert result["rounds"][-1]["status"] == "budget_insufficient_for_complete_round"
    assert result["rounds"][-1]["required_proposals"] == 2
    assert result["rounds"][-1]["available_budget"] == 1


def test_stepwise_unique_subset_budget_reuses_singleton_scores_during_pruning():
    objectives = {
        (0,): -2.0,
        (1,): -1.0,
        (2,): -3.0,
        (0, 1): -0.5,
        (0, 2): -1.5,
        (1, 2): -0.4,
        (0, 1, 2): -0.3,
    }
    calls: list[tuple[tuple[int, ...], ...]] = []

    def evaluate_batch(subsets):
        calls.append(tuple(subsets))
        return {
            subset: (None, objectives[subset],
                     [{"alpha": None, "validation_net_sharpe": objectives[subset]}])
            for subset in subsets
        }

    result = selection._stepwise_select_with_evaluator(
        ["a", "b", "c"],
        {"pool_capacity": 3, "pool_search_budget": 7,
         "min_objective_improvement": 1e-6},
        None,
        objective_name="validation_net_sharpe",
        batch_subset_evaluator=evaluate_batch,
        budget_unit="unique_nonempty_subset",
        initial_objective=-np.inf,
    )

    assert result["selected"] == [0, 1, 2]
    assert result["objective"] == -0.3
    assert result["unique_subset_evaluations"] == 7
    assert result["proposal_evaluations"] > result["unique_subset_evaluations"]
    assert result["stop_reason"] == "capacity_reached"
    assert any(proposal["cache_hit"]
               for round_record in result["rounds"]
               for proposal in round_record["proposals"])
    assert calls[0] == ((0,), (1,), (2,))
