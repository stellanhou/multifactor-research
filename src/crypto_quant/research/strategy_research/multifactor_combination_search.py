"""Frozen 90-day net-Sharpe combination searches (24h, daily R0).

The account implementation is shared with multifactor_net_sharpe. This module
defines calendar windows, candidate spaces and search, not alternative fills.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform

from . import multifactor_net_sharpe as net
from . import multifactor_selection as selection

HOUR = pd.Timedelta(hours=1)
ALPHAS = [0.00001, 0.001, 0.1, 10.0]


@dataclass(frozen=True)
class Windows:
    training: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    refit: pd.DatetimeIndex
    fit_time: pd.Timestamp

    def audit(self, horizon: int) -> dict[str, Any]:
        result = {"fit_time": self.fit_time, "horizon_hours": horizon}
        for name in ("training", "validation", "refit"):
            grid = getattr(self, name)
            result.update({f"{name}_source_start": grid[0],
                           f"{name}_source_end": grid[-1],
                           f"{name}_calendar_hours": len(grid),
                           f"{name}_last_label_maturity": grid[-1] + (horizon + 1) * HOUR})
        result.update(validation_account_start=self.validation[0] + HOUR,
                      validation_account_end_exclusive=self.fit_time,
                      prescreen_account_start=self.training[0] + HOUR,
                      prescreen_account_end_exclusive=self.validation[0])
        return result


def calendar_windows(fit_time: pd.Timestamp, *, horizon: int = 24,
                     training_hours: int = 2160, validation_hours: int = 2160,
                     refit_hours: int = 2160) -> Windows:
    """Full independent source calendars; training labels end before scoring."""
    selection._require(fit_time.tzinfo is not None, "fit time must be timezone aware")
    selection._require(all(type(n) is int and n > 0 for n in
                           (horizon, training_hours, validation_hours, refit_hours)),
                       "window lengths must be positive integers")
    validation_end = fit_time - (horizon + 1) * HOUR
    validation = pd.date_range(end=validation_end, periods=validation_hours, freq="h")
    # Retain the existing strict inner purge: a training label must mature
    # before, rather than at, the first validation source timestamp.
    training_end = validation[0] - (horizon + 2) * HOUR
    training = pd.date_range(end=training_end, periods=training_hours, freq="h")
    refit = pd.date_range(end=validation_end, periods=refit_hours, freq="h")
    return Windows(training, validation, refit, fit_time)


def hierarchical_clusters(cube: np.ndarray, *, cluster_count: int = 8,
                          min_symbols: int = 3) -> tuple[list[list[int]], np.ndarray]:
    """Average linkage on 1 - abs(mean hourly cross-sectional Pearson r)."""
    factor_count = cube.shape[2]
    selection._require(1 <= cluster_count <= factor_count, "invalid fixed cluster count")
    total = np.zeros((factor_count, factor_count))
    observations = np.zeros_like(total, dtype=np.int64)
    for section in cube:
        section = section[np.isfinite(section).all(axis=1)]
        if len(section) < min_symbols:
            continue
        centered = section - section.mean(axis=0)
        norm = np.sqrt(np.square(centered).sum(axis=0))
        denominator = norm[:, None] * norm[None, :]
        corr = np.divide(centered.T @ centered, denominator,
                         out=np.full_like(total, np.nan), where=denominator > 0)
        valid = np.isfinite(corr)
        total += np.where(valid, corr, 0.0)
        observations += valid
    selection._require((observations > 0).all(),
                       "training correlations contain a factor pair without observations")
    mean = np.clip(total / observations, -1.0, 1.0)
    distances = 1.0 - np.abs(mean)
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0.0)
    labels = cut_tree(linkage(squareform(distances), method="average"),
                      n_clusters=[cluster_count]).ravel()
    groups = [np.flatnonzero(labels == label).tolist() for label in sorted(set(labels))]
    groups.sort(key=lambda group: group[0])
    return groups, mean


def enumerate_grid(candidates: list[list[int]], size: int = 5) -> list[tuple[int, ...]]:
    usable = [members for members in candidates if members]
    return sorted(tuple(sorted(subset)) for groups in combinations(usable, size)
                  for subset in product(*groups))


def valid_metric(metric: dict[str, Any]) -> bool:
    return bool(metric["valid_objective"] and metric["trade_count"] > 0
                and metric["annualized_volatility"] is not None
                and metric["annualized_volatility"] > 0
                and metric["validation_net_sharpe"] is not None
                and np.isfinite(metric["validation_net_sharpe"]))


class SubsetEvaluator:
    """Cache every unique subset and retain every regularization trial."""
    def __init__(self, *, method: str, x_train: np.ndarray, y_train: np.ndarray,
                 values: np.ndarray, source_times: pd.DatetimeIndex,
                 fit_time: pd.Timestamp, market: Any, funding: pd.DataFrame,
                 contract: Any, batch_candidates: int = 1024):
        selection._require(method in {"equal", "ridge", "elastic_net"}, "unknown weighting")
        selection._require(type(batch_candidates) is int
                           and batch_candidates >= (1 if method == "equal" else len(ALPHAS)),
                           "candidate batch must hold all alpha trials for one subset")
        self.method = method
        self.x_train, self.y_train = x_train, y_train
        self.stats = net._ridge_training_stats(x_train, y_train)
        self.values = values
        self.complete = np.isfinite(values).all(axis=2)
        self.times, self.fit_time = source_times, fit_time
        self.market, self.funding, self.contract = market, funding, contract
        self.factor_count = values.shape[2]
        self.batch_candidates = batch_candidates
        self.cache: dict[tuple[int, ...], tuple[float | None, float, list[dict]]] = {}
        self.elapsed_seconds = 0.0
        self.fit_seconds = 0.0
        self.account_seconds = 0.0
        self.cache_hits = 0

    def coefficients(self, subset: tuple[int, ...], alpha: float | None) -> np.ndarray:
        beta = np.zeros(self.factor_count)
        columns = list(subset)
        if self.method == "equal":
            beta[columns] = 1.0 / len(columns)
        else:
            selection._require(self.stats.invalid_status is None,
                               f"invalid training sample: {self.stats.invalid_status}")
            if self.method == "ridge":
                gram = self.stats.train_gram[np.ix_(columns, columns)]
                beta[columns] = np.linalg.solve(gram + alpha * np.eye(len(columns)),
                                                self.stats.train_rhs[columns])
            else:
                beta[columns] = selection._elastic_net_coefficients(
                    self.x_train[:, columns], self.y_train / self.stats.target_rms, alpha, 0.5)
        return beta

    def evaluate(self, subsets: list[tuple[int, ...]]) -> dict:
        started = time.monotonic()
        for subset in subsets:
            selection._require(bool(subset) and tuple(sorted(set(subset))) == subset
                               and subset[0] >= 0 and subset[-1] < self.factor_count,
                               "subset must contain sorted unique valid factors")
        self.cache_hits += sum(subset in self.cache for subset in subsets)
        unseen = list(dict.fromkeys(subset for subset in subsets if subset not in self.cache))
        alphas = [None] if self.method == "equal" else ALPHAS
        per_batch = self.batch_candidates // len(alphas)
        for offset in range(0, len(unseen), per_batch):
            batch = unseen[offset:offset + per_batch]
            if self.method != "equal" and self.stats.invalid_status is not None:
                for subset in batch:
                    self.cache[subset] = (None, np.nan, [
                        {"alpha": alpha, "validation_objective": None, "valid_objective": False,
                         "status": self.stats.invalid_status} for alpha in alphas])
                continue
            owners = [(subset, alpha) for subset in batch for alpha in alphas]
            before = time.monotonic()
            coefficients = np.asarray([self.coefficients(subset, alpha) for subset, alpha in owners])
            self.fit_seconds += time.monotonic() - before
            before = time.monotonic()
            metrics = net._simulate_r0_batch(
                coefficients, self.values, self.complete, self.times, self.fit_time,
                self.market, self.funding, self.contract, {"min_cross_section_symbols": 3})
            self.account_seconds += time.monotonic() - before
            trials: dict[tuple[int, ...], list[dict]] = {subset: [] for subset in batch}
            for (subset, alpha), metric in zip(owners, metrics):
                valid = valid_metric(metric)
                trials[subset].append({**metric, "alpha": alpha, "valid_objective": valid,
                                       "validation_objective": metric["validation_net_sharpe"] if valid else None,
                                       "status": metric["status"] if valid else
                                       "cash_or_zero_volatility" if metric["trade_count"] == 0
                                       or metric["annualized_volatility"] == 0 else metric["status"]})
            for subset, rows in trials.items():
                valid = [row for row in rows if row["valid_objective"]]
                # Ordered alpha trials resolve exact ties without a second performance metric.
                best = max(valid, key=lambda row: row["validation_objective"]) if valid else None
                self.cache[subset] = ((best["alpha"], best["validation_objective"], rows)
                                      if best else (None, np.nan, rows))
        self.elapsed_seconds += time.monotonic() - started
        return {subset: self.cache[subset] for subset in subsets}

    def best(self) -> tuple[int, ...] | None:
        valid = [subset for subset, result in self.cache.items() if np.isfinite(result[1])]
        return min(valid, key=lambda subset: (-self.cache[subset][1], subset)) if valid else None

    def audit(self) -> dict:
        return {"unique_subset_evaluations": len(self.cache), "cache_hits": self.cache_hits,
                "alpha_trial_evaluations": sum(len(row[2]) for row in self.cache.values()),
                "actual_account_evaluations": sum("trade_count" in trial for row in self.cache.values()
                                                   for trial in row[2]),
                "evaluation_seconds": self.elapsed_seconds, "coefficient_fit_seconds": self.fit_seconds,
                "account_scoring_seconds": self.account_seconds}


def prescreen(groups: list[list[int]], evaluator: SubsetEvaluator) -> tuple[list[list[int]], list[dict]]:
    evaluator.evaluate([(index,) for group in groups for index in group])
    candidates, records = [], []
    for group_number, group in enumerate(groups):
        valid = [index for index in group if np.isfinite(evaluator.cache[(index,)][1])]
        valid.sort(key=lambda index: (-evaluator.cache[(index,)][1], index))
        candidates.append(valid[:2])
        for index in group:
            row = evaluator.cache[(index,)]
            records.append({"cluster": group_number, "factor_index": index,
                            "net_sharpe": row[1], "kept": index in valid[:2],
                            "trials": row[2]})
    return candidates, records


def random_search(evaluator: SubsetEvaluator, *, size: int, budget: int, seed: int) -> dict:
    selection._require(0 <= budget <= math.comb(evaluator.factor_count, size),
                       "random search budget exceeds its finite space")
    rng = np.random.default_rng(seed)
    subsets: set[tuple[int, ...]] = set()
    while len(subsets) < budget:
        subsets.add(tuple(sorted(rng.choice(evaluator.factor_count, size, replace=False).tolist())))
    evaluator.evaluate(sorted(subsets))
    return {"selected": evaluator.best(), "stop_reason": "unique_subset_budget_exhausted",
            "seed": seed, "budget": budget, **evaluator.audit()}


def genetic_search(evaluator: SubsetEvaluator, groups: list[list[int]], *,
                   size: int, budget: int, seed: int, population_limit: int = 100,
                   generations: int = 50) -> dict:
    selection._require(budget > 0 and len(groups) >= size, "GA requires a positive feasible budget")
    rng = np.random.default_rng(seed)
    population_size = min(population_limit, budget)
    cluster_by_factor = {factor: group for group, members in enumerate(groups) for factor in members}
    initial_space = sum(math.prod(len(groups[g]) for g in chosen)
                        for chosen in combinations(range(len(groups)), size))
    selection._require(population_size <= initial_space, "GA population exceeds initialization space")
    population: list[tuple[int, ...]] = []
    initial_seen: set[tuple[int, ...]] = set()
    while len(population) < population_size:
        chosen_groups = rng.choice(len(groups), size, replace=False)
        child = tuple(sorted(int(rng.choice(groups[g])) for g in chosen_groups))
        if child not in initial_seen:
            population.append(child)
            initial_seen.add(child)
    evaluator.evaluate(population)
    initial_population = population.copy()
    history = []

    def rank(individual):
        objective = evaluator.cache[individual][1]
        return (-objective if np.isfinite(objective) else np.inf, individual)

    def tournament():
        participants = [population[int(i)] for i in rng.integers(0, len(population), size=3)]
        return min(participants, key=rank)

    stop_reason = "generation_limit"
    for generation in range(generations):
        best = evaluator.best()
        history.append({"generation": generation, "unique_evaluations": len(evaluator.cache),
                        "best_subset": best, "best_net_sharpe": evaluator.cache[best][1] if best else None,
                        "distinct_population": len(set(population)),
                        "same_cluster_individuals": sum(len({cluster_by_factor[i] for i in child}) < size
                                                        for child in population)})
        if len(evaluator.cache) >= budget:
            stop_reason = "unique_subset_budget_exhausted"
            break
        if generation == generations - 1:
            break
        elite_count = min(3, population_size)
        children = sorted(population, key=rank)[:elite_count]
        for _ in range(population_size - elite_count):
            union = sorted(set(tournament()) | set(tournament()))
            child = sorted(int(i) for i in rng.choice(union, size, replace=False))
            if rng.random() < 0.2:
                removed_position = int(rng.integers(size))
                retained = child[:removed_position] + child[removed_position + 1:]
                occupied = {cluster_by_factor[i] for i in retained}
                if rng.random() < 0.8:
                    choices = [i for g, members in enumerate(groups) if g not in occupied
                               for i in members if i not in child]
                else:
                    choices = [i for i in range(evaluator.factor_count) if i not in child]
                selection._require(bool(choices), "GA mutation has no replacement factor")
                child = sorted([*retained, int(rng.choice(choices))])
            children.append(tuple(child))
        new = list(dict.fromkeys(child for child in children if child not in evaluator.cache))
        remaining = budget - len(evaluator.cache)
        evaluator.evaluate(new[:remaining])
        if len(new) > remaining:
            stop_reason = "unique_subset_budget_exhausted_mid_generation"
            history.append({"generation": generation + 1, "partial_generation": True,
                            "unique_evaluations": len(evaluator.cache),
                            "unscored_unique_children": len(new) - remaining})
            break
        population = children
    return {"selected": evaluator.best(), "stop_reason": stop_reason, "seed": seed,
            "budget": budget, "population_size": population_size, "initial_population": initial_population,
            "generations": history, **evaluator.audit()}


def run_search(evaluator: SubsetEvaluator, *, algorithm: str, groups: list[list[int]],
               grid: list[tuple[int, ...]], budget: int, size: int, seed: int,
               factor_names: list[str], proposal_budget: bool = False) -> dict:
    if algorithm == "full":
        selected = tuple(range(evaluator.factor_count))
        evaluator.evaluate([selected])
        return {"selected": selected, "stop_reason": "full_pool", **evaluator.audit()}
    if budget == 0:
        return {"selected": None, "stop_reason": "no_feasible_grid_budget", **evaluator.audit()}
    if algorithm == "grid":
        selection._require(len(grid) == budget, "grid must exhaust its exact finite space")
        evaluator.evaluate(grid)
        return {"selected": evaluator.best(), "stop_reason": "complete_grid", **evaluator.audit()}
    if algorithm == "random":
        return random_search(evaluator, size=size, budget=budget, seed=seed)
    if algorithm == "ga":
        return genetic_search(evaluator, groups, size=size, budget=budget, seed=seed)
    selection._require(algorithm == "stepwise", "unknown combination search")
    result = selection._stepwise_select_with_evaluator(
        factor_names, {"pool_capacity": size, "pool_search_budget": budget,
                       "min_objective_improvement": 1e-6}, None,
        objective_name="validation_net_sharpe", batch_subset_evaluator=evaluator.evaluate,
        budget_unit="feature_subset_proposal" if proposal_budget else "unique_nonempty_subset",
        initial_objective=-np.inf)
    result.update(evaluator.audit())
    return result


def admitted(evaluator: SubsetEvaluator, selected) -> tuple[bool, dict | None]:
    if not selected:
        return False, None
    alpha, objective, trials = evaluator.cache[tuple(selected)]
    best = next((trial for trial in trials if trial["alpha"] == alpha
                 and trial["valid_objective"]), None)
    return bool(best and np.isfinite(objective) and objective > 0
                and best["validation_net_return"] > 0), best
